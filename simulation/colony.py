"""
simulation/colony.py
--------------------
Colony and Faction: the two-tier agent hierarchy for the simulation.

Architecture
------------
Colony  — local operator tied to one solar system.
          Owns buildings, workers, a local stockpile, and a faction
          sub-stockpile (the "tax" pool). Receives a Directive from
          its parent Faction each tick and executes rule-based logic
          to decide what to build, repair, surge, or export.

Faction — strategic layer. Owns one or more Colonies, maintains a
          treasury (separate from any colony stockpile), issues
          Directives, reads Colony flags, and coordinates inter-colony
          transfers. Faction.tick() is the outer simulation loop.

Flag system (two tiers)
-----------------------
Critical  — existential threats that block harmful directives:
            FOOD_SHORTAGE, POWER_DEFICIT, POPULATION_COLLAPSE
Strategic — multi-tick concerns the faction agent routes around:
            DEFENSE_NEEDED, WORKER_SHORTAGE, RESOURCE_LOW,
            EXPORT_STRAINED, CONSTRUCTION_BLOCKED

Directive system
----------------
DirectiveType: HARVEST, DEFEND, EXPAND, EXPORT, IDLE
Each Directive carries priority weights, a tax rate, and optional
target parameters. The colony rule engine reads these each tick.

Rule-based colony decision priority (lowest index = highest priority)
----------------------------------------------------------------------
0. Critical flag response  — survival always first, regardless of directive
1. Repair damaged buildings — keep the fleet healthy
2. Directive execution      — faction orders shape all remaining decisions
3. Idle / balanced upkeep   — default when no directive pressure exists

Exports
-------
CriticalFlag    - IntEnum
StrategicFlag   - IntEnum
DirectiveType   - IntEnum
Directive       - dataclass
Colony          - local agent
Faction         - strategic agent

(Worker, WorkerLevel, and Building are now defined in buildings.py.)
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from buildings import (
    Building,
    BuildingType,
    BuildingState,
    BuildingLevelStats,
    BUILDING_STATS,
    MAX_BUILDING_LEVEL,
    INITIAL_HEALTH,
    REPAIR_THRESHOLD,
    DAMAGED_THRESHOLD,
    ResourceType,
    Worker,
    WorkerLevel,
    colony_production_rates,
    colony_production_costs,
)

R = ResourceType


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

ORGANICS_PER_POP:  float = 0.05   # organics consumed per population unit per tick
POP_PER_WORKER:    int   = 10     # population units per level-1 worker
BASE_ORGANICS_RATE: int  = 5      # base organics made by a starter colony each tick

# Flag thresholds
FOOD_SHORTAGE_TICKS:       int   = 1     # consecutive starving ticks before FOOD_SHORTAGE
FOOD_DEFICIT_THRESHOLD:    float = 10.0    # net ORGANICS below this → starving tick
POWER_DEFICIT_THRESHOLD:   float = 0.0   # net POWER below this → POWER_DEFICIT
POPULATION_COLLAPSE_FRAC:  float = 0.25  # population fallen to this fraction of start → COLLAPSE
DEFENSE_LOW_THRESHOLD:     float = 10.0  # net DEFENSE score below this → DEFENSE_NEEDED
BOLSTER_DEFENSE_TICKS:     int   = 25    # how many ticks of DEFENSE_NEEDED before bolstering response escalates
WORKER_SHORTAGE_RATIO:     float = 0.1   # unassigned workers / total workforce below this
RESOURCE_LOW_TICKS:        int   = 5     # ticks of negative net rate before RESOURCE_LOW
EXPORT_STRAINED_THRESHOLD: float = 0.10  # local stockpile fraction remaining after tax

# Rule-based thresholds
SURGE_HEALTH_MIN:     float = 0.80   # don't surge buildings below this health
BUILD_STOCKPILE_MIN:  float = 1.5    # multiplier: must have 1.5× build cost in stockpile


# ---------------------------------------------------------------------------
# FLAGS
# ---------------------------------------------------------------------------

class WorkerLevel(IntEnum):
    L1 = 1
    L2 = 2
    L3 = 3
    L4 = 4
    L5 = 5


@dataclass
class Worker:
    """A single worker unit with a skill level."""
    level:                WorkerLevel
    assigned_building_id: Optional[int] = None   # None = unassigned pool

    @property
    def is_assigned(self) -> bool:
        return self.assigned_building_id is not None


# ---------------------------------------------------------------------------
# BUILDING  (runtime instance)
# ---------------------------------------------------------------------------

@dataclass
class Building:
    """
    Runtime instance of a building owned by a Colony.

    Static stats are always fetched live from BUILDING_STATS[type][level]
    and never duplicated here.

    Fields
    ------
    id              : unique int within the colony
    building_type   : BuildingType enum
    level           : current level (1-5)
    state           : BuildingState enum
    health          : float in [0.0, 1.0]
    ticks_remaining : countdown for CONSTRUCTING state
    planet_index    : which planet in the solar system this sits on
    """
    id:              int
    building_type:   BuildingType
    level:           int           = 1
    state:           BuildingState = BuildingState.CONSTRUCTING
    health:          float         = INITIAL_HEALTH
    max_health:      float         = INITIAL_HEALTH
    ticks_remaining: int           = 0
    planet_index:    Optional[int] = None

    @property
    def stats(self) -> BuildingLevelStats:
        return BUILDING_STATS[self.building_type][self.level]

    @property
    def is_active(self) -> bool:
        return self.state in (BuildingState.ACTIVE, BuildingState.SURGING)

    @property
    def surge_multiplier(self) -> float:
        return 1.5 if self.state == BuildingState.SURGING else 1.0
    
    @property
    def properly_staffed(self) -> bool:
        """True if all required workers are assigned to this building."""
        return all(
            sum(w.assigned_building_id == self.id and int(w.level) == lvl for w in self._workers) >= count
            for lvl, count in self.stats.workforce.items()
        )

    # ------------------------------------------------------------------
    # Per-tick methods
    # ------------------------------------------------------------------

    def apply_damage(self) -> None:
        """Reduce health by this tick's damage rate (doubled while SURGING)."""
        if self.state not in (BuildingState.ACTIVE, BuildingState.SURGING):
            return
        rate = self.stats.damage_rate * (2.0 if self.state == BuildingState.SURGING else 1.0)
        self.health = max(0.0, self.health - rate / 100.0)
        if self.health <= 0.0:
            self.state = BuildingState.DESTROYED
        elif self.health < REPAIR_THRESHOLD:
            self.state = BuildingState.REPAIRING
        elif self.health < DAMAGED_THRESHOLD:
            self.state = BuildingState.DAMAGED

    def apply_repair(self) -> None:
        """Advance repair health; transition to ACTIVE when full."""
        if self.state not in [BuildingState.ACTIVE, BuildingState.REPAIRING, BuildingState.DAMAGED]:
            return
        self.health = min(1.0, self.health + self.stats.repair_rate / 100.0)
        if self.health >= 1.0:
            self.state = BuildingState.ACTIVE
        print(f"Repairing building {self.id} ({self.building_type.name} lv{self.level}): "
              f"health {self.health:.2f} (rate {self.stats.repair_rate:.2f}/tick)")

    def advance_construction(self) -> bool:
        """Count down construction. Returns True when ACTIVE."""
        if self.state != BuildingState.CONSTRUCTING:
            return False
        self.ticks_remaining = max(0, self.ticks_remaining - 1)
        if self.ticks_remaining == 0:
            self.state  = BuildingState.ACTIVE
            self.health = INITIAL_HEALTH
            return True
        return False

    def production_this_tick(self) -> Dict[int, float]:
        if not self.is_active:
            return {}
        return {r: amt * self.surge_multiplier for r, amt in self.stats.production_rate.items()}

    def upkeep_this_tick(self) -> Dict[int, float]:
        if not self.is_active:
            return {}
        return dict(self.stats.production_cost)

    def repair_upkeep_this_tick(self) -> Dict[int, float]:
        if self.state != BuildingState.REPAIRING:
            return {}
        return dict(self.stats.repair_cost)

    def summary(self) -> str:
        bar_filled = int(self.health * 20)
        bar   = "█" * bar_filled + "░" * (20 - bar_filled)
        extra = f"  ⏳ {self.ticks_remaining}t left" if self.state == BuildingState.CONSTRUCTING else ""
        return (
            f"  [{self.id:>3}] {self.building_type.name:<12} lv{self.level}"
            f"  {self.state.name:<12} [{bar}] {self.health*100:>5.1f}%{extra}"
        )


# ---------------------------------------------------------------------------
# FLAGS
# ---------------------------------------------------------------------------

class CriticalFlag(IntEnum):
    """
    Existential threats. These always surface to the Faction and block any
    directive action that would deepen the problem.
    """
    FOOD_SHORTAGE       = 0   # colony has been starving for N consecutive ticks
    POWER_DEFICIT       = 1   # net POWER production is negative
    POPULATION_COLLAPSE = 2   # population has fallen below collapse fraction


class StrategicFlag(IntEnum):
    """
    Multi-tick concerns. Fully overridable by faction directive.
    The faction agent uses these to plan resource routing over several ticks.
    """
    DEFENSE_NEEDED      = 0   # net DEFENSE score below threshold
    WORKER_SHORTAGE     = 1   # not enough workers to staff buildings
    RESOURCE_LOW        = 2   # one or more key resources trending negative
    EXPORT_STRAINED     = 3   # tax rate is cutting into operational reserves
    CONSTRUCTION_BLOCKED = 4  # colony wants to build but can't afford anything


# ---------------------------------------------------------------------------
# DIRECTIVE
# ---------------------------------------------------------------------------

class DirectiveType(IntEnum):
    HARVEST = 0   # gather large stockpile; only build if survival at risk
    BUILD   = 1   # expand building count for target resource
    UPGRADE = 2   # upgrade existing buildings for target resource
    EXPORT  = 3   # send target resource to colony, trade for WEALTH
    EXPAND  = 4   # send ships/resources to establish new colony


@dataclass
class Directive:
    """
    Issued by a Faction to one of its Colonies each tick.

    Fields
    ------
    directive_type       : primary intent (HARVEST, BUILD, UPGRADE, EXPORT, EXPAND)
    tax_rate             : fraction [0.0, 1.0+] of each resource produced that
                          flows into the colony's faction sub-stockpile.
                          Values > 1.0 will draw from local reserves.
    urgency              : scalar [0.0, 1.0] — how aggressively to pursue the
                          directive vs. balanced upkeep. 1.0 = full commitment.
    target_resource     : ResourceType to focus the directive on
    export_destination  : colony name to export target resource to (for WEALTH trade)
    export_demand        : amount of target resource to export per tick
    override_flags       : set of StrategicFlag values the faction explicitly suppresses
                          (critical flags are never suppressible)
    """
    directive_type:      DirectiveType          = DirectiveType.HARVEST
    tax_rate:            float                  = 0.10
    urgency:             float                  = 0.5
    target_resource:     Optional[int]          = None
    export_destination:  Optional[str]          = None
    export_demand:       float                  = 0.0
    override_flags:      Set[StrategicFlag]     = field(default_factory=set)


# ---------------------------------------------------------------------------
# COLONY
# ---------------------------------------------------------------------------

@dataclass
class Colony:
    """
    Local simulation agent tied to one solar system.

    Parameters
    ----------
    colony_id       : unique int within the parent Faction
    name            : display name
    system_id       : which SolarSystem this colony occupies
    population      : starting headcount
    stockpile       : local resource pool  {resource_key: amount}
    faction_stockpile : faction tax pool — drawn by Faction, not Colony
    starting_pop    : recorded at construction for collapse-threshold maths
    """
    colony_id:         int
    name:              str
    system_id:         int
    population:        float                = 1000.0
    stockpile:         Dict[int, float]     = field(default_factory=dict)
    faction_stockpile: Dict[int, float]     = field(default_factory=dict)
    starting_pop:      float                = field(init=False)

    # Runtime state
    directive:         Directive            = field(default_factory=Directive)
    critical_flags:    Set[CriticalFlag]    = field(default_factory=set)
    strategic_flags:   Set[StrategicFlag]   = field(default_factory=set)

    _buildings:        List[Building]       = field(default_factory=list,  repr=False)
    _workers:          List[Worker]         = field(default_factory=list,  repr=False)
    _next_id:          int                  = field(default=0,             repr=False)
    _tick:             int                  = field(default=0,             repr=False)
    _rng:              Optional[np.random.Generator] = field(default=None, repr=False)

    # Counters for flag hysteresis
    _starving_ticks:   int                  = field(default=0,             repr=False)
    _resource_low_ticks: Dict[int, int]     = field(default_factory=dict,  repr=False)
    _upskill_progress:   Dict[int, float]   = field(default_factory=dict,  repr=False)

    # Per-tick ledgers
    last_produced:     Dict[int, float]     = field(default_factory=dict,  repr=False)
    last_consumed:     Dict[int, float]     = field(default_factory=dict,  repr=False)
    last_events:       List[str]            = field(default_factory=list,  repr=False)

    # debugging settings
    verbose = True

    def __post_init__(self) -> None:
        self.starting_pop = self.population

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _new_id(self) -> int:
        bid = self._next_id
        self._next_id += 1
        return bid

    def _add_to(self, ledger: Dict[int, float], resource: int, amount: float) -> None:
        ledger[resource] = ledger.get(resource, 0.0) + amount

    def _deduct(self, costs: Dict[int, float], pool: Optional[Dict[int, float]] = None) -> bool:
        """
        Atomically deduct costs from pool (defaults to local stockpile).
        Returns True on success, False with no deduction if insufficient.
        """
        pool = pool if pool is not None else self.stockpile
        for resource, amount in costs.items():
            if pool.get(resource, 0.0) < amount:
                return False
        for resource, amount in costs.items():
            pool[resource] = pool.get(resource, 0.0) - amount
        return True

    def _can_afford(self, costs: Dict[int, float], multiplier: float = 1.0) -> bool:
        """True if local stockpile holds multiplier × costs for all keys."""
        return all(
            self.stockpile.get(r, 0.0) >= amt * multiplier
            for r, amt in costs.items()
        )

    # ------------------------------------------------------------------
    # POPULATION & WORKERS
    # ------------------------------------------------------------------
    @property
    def free_population(self) -> float:
        return self.population - len(self._workers) * POP_PER_WORKER

    @property
    def organics_upkeep_per_tick(self) -> float:
        return self.population * ORGANICS_PER_POP

    @property
    def power_stockpile(self) -> float:
        power = 0.0
        for b in self._buildings:
            if b.is_active:
                prod = b.stats.production_rate.get(4, 0.0)
                cons = b.stats.production_cost.get(4, 0.0)
                power += prod - cons
        return power

    def recruit_workers(self, count: int = 1) -> int:
        """Convert population into level-1 workers. Returns count created."""
        max_possible = int(self.free_population // POP_PER_WORKER)
        count = min(count, max_possible)
        for _ in range(count):
            self._workers.append(Worker(level=WorkerLevel.L1))
        return count

    def recruit_workers_of_level(self, level: int, count: int = 1) -> int:
        """
        Recruit workers at a specific level (1–5).

        Level-1 workers are created directly from the population pool (same as
        recruit_workers).  For levels 2–5, this method checks that:
        - At least one active Lab exists whose upskill_rates chain covers the
            requested level (i.e. a Lab whose upskill_rates[level-2] > 0).
        - Enough lower-level workers exist in the unassigned pool to be
            promoted (they are consumed and replaced with the higher-level worker).

        If neither condition is met the method falls back to recruiting L1 workers
        and letting the normal upskilling pipeline handle promotion over time.

        Returns
        -------
        int — number of workers actually created / promoted at the requested level.
        """
        if level == 1:
            return self.recruit_workers(count)

        if level < 1 or level > 5:
            raise ValueError(f"Worker level must be 1–5, got {level}")

        # Check that a Lab can train up to the requested level
        if not self._lab_can_train(level):
            # Fall back: recruit L1s and let upskilling pipeline handle it
            return self.recruit_workers(count)

        promoted = 0
        for _ in range(count):
            # Consume one unassigned worker at level-1 from the pool
            source_level = level - 1
            candidates = [w for w in self._workers
                        if int(w.level) == source_level and not w.is_assigned]
            if candidates:
                w = candidates[0]
                w.level = WorkerLevel(level)
                promoted += 1
            else:
                # No ready source — recruit an L1 as a placeholder
                if self.recruit_workers(1) == 1:
                    promoted += 1   # will be trained via upskilling over time
        return promoted

    @property
    def required_workers(self) -> int:
        """Total workers needed to staff all active buildings."""
        return sum(
            sum(b.stats.workforce.values())
            for b in self.active_buildings
        )

    def required_workers_by_level(self) -> Dict[int, int]:
        """
        Returns a dict of {worker_level: count_needed} across all buildings that
        need workers (ACTIVE, SURGING, INACTIVE), accounting for the current pool
        of assigned workers at each level.
        """
        needed: Dict[int, int] = {}
        # Include all buildings that need workers to function
        worker_needing_states = (BuildingState.ACTIVE, BuildingState.SURGING, BuildingState.INACTIVE)
        for b in self._buildings:
            if b.state in worker_needing_states:
                for lvl, cnt in b.stats.workforce.items():
                    needed[lvl] = needed.get(lvl, 0) + cnt
        # Subtract what we already have assigned
        for w in self._workers:
            if w.is_assigned:
                lvl = int(w.level)
                needed[lvl] = max(0, needed.get(lvl, 0) - 1)
        return {k: v for k, v in needed.items() if v > 0}

    def release_workers(self, count: int = 1) -> int:
        """Remove unassigned workers and return population. Returns count released."""
        unassigned = [w for w in self._workers if not w.is_assigned]
        count = min(count, len(unassigned))
        for w in unassigned[:count]:
            self._workers.remove(w)
        return count

    def unassigned_workers(self) -> List[Worker]:
        return [w for w in self._workers if not w.is_assigned]
    
    def unassigned_workers_by_level(self) -> Dict[int, int]:
        levels: Dict[int, int] = {}
        for w in self.unassigned_workers():
            lvl = int(w.level)
            levels[lvl] = levels.get(lvl, 0) + 1
        return levels

    def workers_at_level(self, level: int) -> List[Worker]:
        return [w for w in self._workers if int(w.level) == level]
    
    def workers_by_level(self) -> Dict[int, int]:
        levels: Dict[int, int] = {}
        for w in self._workers:
            lvl = int(w.level)
            levels[lvl] = levels.get(lvl, 0) + 1
        return levels

    def assign_workers_to_building(self, building: Building) -> bool:
        """
        Assign workers to a building based on its workforce requirements.
        Returns True if all required workers were assigned, False if partial/incomplete.
        If not enough workers exist, assigns what's available and returns False.
        
        Note: For upgrades, existing workers are first unassigned so they can be
        re-assigned if they meet the new requirements (or return to the pool).
        """
        # Unassign current workers from this building
        for w in building._workers:
            w.assigned_building_id = None
        building._workers.clear()

        required = building.stats.workforce
        if not required:
            building.state = BuildingState.ACTIVE
            return True

        assigned = []
        success = True

        for level, count in required.items():
            available = [w for w in self.unassigned_workers() if int(w.level) == level]
            assign_count = min(count, len(available))
            for w in available[:assign_count]:
                w.assigned_building_id = building.id
                assigned.append(w)
            if assign_count < count:
                for w in assigned:
                    w.assigned_building_id = None
                success = False

        if assigned and success:
            building._workers.extend(assigned)
            building.state = BuildingState.ACTIVE
            self.last_events.append(
                f"Assigned {len(assigned)} worker(s) to {building.building_type.name} "
                f"lv{building.level} (id={building.id})."
            )
        if not success:
            building.state = BuildingState.INACTIVE
            self.last_events.append(
                f"{building.building_type.name} lv{building.level} (id={building.id}) "
                f"INACTIVE — worker shortage."
            )
        return success
    
    def unassign_workers_from_building(self, building: Building) -> int:
        """Unassign all workers from the building and return count unassigned."""
        count = len(building._workers)
        for w in building._workers:
            w.assigned_building_id = None
        building._workers.clear()
        building.state = BuildingState.INACTIVE
        if count:
            self.last_events.append(
                f"Unassigned {count} worker(s) from {building.building_type.name} "
                f"lv{building.level} (id={building.id})."
            )
        return count
    
    def can_staff_L1_building(self, building_type: BuildingType) -> bool:
        """True if enough unassigned workers exist to staff the building's workforce."""
        required = BUILDING_STATS[building_type][1].workforce
        available_by_level = self.workers_by_level()
        for level, count in required.items():
            if available_by_level.get(level, 0) < count:
                return False
        return True
    
    def can_staff_building(self, building: Building) -> bool:
        """True if enough unassigned workers exist to staff the building's workforce."""
        required = building.stats.workforce
        available_by_level = self.workers_by_level()
        for level, count in required.items():
            if available_by_level.get(level, 0) < count:
                return False
        return True
    
    # ------------------------------------------------------------------
    # LAB / UPSKILLING HELPERS
    # ------------------------------------------------------------------
    
    def _active_lab_upskill_rates(self) -> List[float]:
        """
        Sum upskill_rates across all active Labs.
        Returns a list of 4 floats: [L1→L2, L2→L3, L3→L4, L4→L5] per tick.
        """
        from buildings import LabLevelStats
        rates = [0.0, 0.0, 0.0, 0.0]
        for b in self._buildings:
            stats = b.stats
            if isinstance(stats, LabLevelStats):
                for i, r in enumerate(stats.upskill_rates):
                    rates[i] += r
        return rates

    def _lab_can_train(self, target_level: int) -> bool:
        """
        True if at least one active Lab can train workers to target_level
        (i.e. the upskill_rates index for that promotion tier is > 0).
        Tier index = target_level - 2  (e.g. L1→L2 is index 0).
        """
        if target_level < 2 or target_level > 5:
            return False
        tier_index = target_level - 2
        rates = self._active_lab_upskill_rates()
        return rates[tier_index] > 0.0

    def _process_upskilling(self) -> None:
        """
        Accumulate training progress for unassigned workers using the summed
        upskill_rates of all active Labs.  When progress reaches ≥ 1.0 for a
        tier, one unassigned worker at that level is promoted to the next level
        and progress resets (carry-over is preserved for smooth multi-lab scaling).

        Called once per tick, after buildings have produced resources.
        Progress is stored in self._upskill_progress keyed by source level int.
        """
        rates = self._active_lab_upskill_rates()
        if not any(r > 0 for r in rates):
            return  # no active labs — nothing to do

        for tier_index, rate in enumerate(rates):
            if rate <= 0:
                continue
            source_level = tier_index + 1          # L1, L2, L3, L4
            target_level = source_level + 1        # L2, L3, L4, L5

            # Accumulate progress
            self._upskill_progress[source_level] = (
                self._upskill_progress.get(source_level, 0.0) + rate
            )

            # Promote as many workers as progress allows
            while self._upskill_progress[source_level] >= 1.0:
                candidates = [w for w in self._workers
                            if int(w.level) == source_level and not w.is_assigned]
                if not candidates:
                    # No eligible workers — bleed off progress to avoid runaway accumulation
                    self._upskill_progress[source_level] = min(
                        self._upskill_progress[source_level], 1.0
                    )
                    break
                w = candidates[0]
                w.level = WorkerLevel(target_level)
                self._upskill_progress[source_level] -= 1.0
                self.last_events.append(
                    f"👷 Worker promoted L{source_level}→L{target_level} "
                    f"(upskilling; {self._upskill_progress[source_level]:.2f} progress remaining)."
                )

    # ------------------------------------------------------------------
    # BUILDING MANAGEMENT
    # ------------------------------------------------------------------

    def get_building(self, building_id: int) -> Optional[Building]:
        for b in self._buildings:
            if b.id == building_id:
                return b
        return None

    @property
    def active_buildings(self) -> List[Building]:
        return [b for b in self._buildings if b.is_active]

    @property
    def building_counts(self) -> Dict[Tuple[BuildingType, int], int]:
        counts: Dict[Tuple[BuildingType, int], int] = {}
        for b in self.active_buildings:
            key = (b.building_type, b.level)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def construct_building(
        self,
        building_type: BuildingType,
        planet_index:  Optional[int] = None,
        level:         int           = 1,
    ) -> Optional[Building]:
        """Deduct build cost and queue construction. Returns Building or None."""
        stats = BUILDING_STATS[building_type][level]
        if not self._deduct(stats.build_cost):
            return None
        b = Building(
            id              = self._new_id(),
            building_type   = building_type,
            level           = level,
            state           = BuildingState.CONSTRUCTING,
            health          = 0.0,
            ticks_remaining = stats.build_ticks,
            planet_index    = planet_index,
        )
        self._buildings.append(b)
        return b

    def can_upgrade_building(self, building_id: int) -> bool:
        b = self.get_building(building_id)
        if b is None or b.state != BuildingState.ACTIVE:
            return False
        if b.level >= MAX_BUILDING_LEVEL:
            return False
        next_level = b.level + 1
        stats = BUILDING_STATS[b.building_type][next_level]
        if not self._can_afford(stats.build_cost):
            return False
        required = stats.workforce
        available_by_level = self.workers_by_level()
        for level, count in required.items():
            if available_by_level.get(level, 0) < count:
                return False
        return True

    def upgrade_building(self, building_id: int) -> bool:
        b = self.get_building(building_id)
        if b is None or b.state != BuildingState.ACTIVE:
            return False
        if b.level >= MAX_BUILDING_LEVEL:
            return False
        next_level = b.level + 1
        stats = BUILDING_STATS[b.building_type][next_level]
        if not self._deduct(stats.build_cost):
            return False
        b.level           = next_level
        b.state           = BuildingState.CONSTRUCTING
        b.ticks_remaining = stats.build_ticks
        return True

    def start_repair(self, building_id: int) -> bool:
        b = self.get_building(building_id)
        if b is None or b.state != BuildingState.DAMAGED:
            return False
        b.state = BuildingState.REPAIRING
        return True

    def set_surge(self, building_id: int, active: bool) -> bool:
        b = self.get_building(building_id)
        if b is None:
            return False
        if active and b.state == BuildingState.ACTIVE and b.health >= SURGE_HEALTH_MIN:
            b.state = BuildingState.SURGING
            return True
        if not active and b.state == BuildingState.SURGING:
            b.state = BuildingState.ACTIVE
            return True
        return False

    # ------------------------------------------------------------------
    # RESOURCE COLLECTION & UPKEEP
    # ------------------------------------------------------------------

    def collect_resources(self) -> None:
        """
        Collect production from all producing buildings.
        Tax is siphoned into faction_stockpile before the remainder
        lands in the local stockpile.
        Buildings that cannot pay upkeep go INACTIVE.
        """
        tax_rate = max(0.0, self.directive.tax_rate)
        for b in self._buildings:
            if not b.is_active: # if not ACTIVE or SURGING, skip production and upkeep
                continue
            upkeep = b.upkeep_this_tick()
            if upkeep and not self._deduct(upkeep): # if upkeep exists and can't be paid, building goes INACTIVE
                b.state = BuildingState.INACTIVE
                self.last_events.append(
                    f"Building {b.id} ({b.building_type.name} lv{b.level}) INACTIVE — upkeep deficit."
                )
                continue
            for res, amt in upkeep.items():
                self._add_to(self.last_consumed, res, amt)

            production = b.production_this_tick()
            for res, amt in production.items():
                tax_amt   = amt * tax_rate
                local_amt = amt - tax_amt
                # Tax > 1.0 can draw from local stockpile
                if tax_rate > 1.0:
                    excess    = amt * (tax_rate - 1.0)
                    local_amt = 0.0
                    tax_amt   = amt + excess
                self.stockpile[res]         = self.stockpile.get(res, 0.0) + local_amt
                self.faction_stockpile[res] = self.faction_stockpile.get(res, 0.0) + tax_amt
                self._add_to(self.last_produced, res, amt)

        # Tax > 1.0 may also eat into existing local stockpile reserves
        if tax_rate > 1.0:
            for res in list(self.stockpile.keys()):
                reserve_draw = self.stockpile.get(res, 0.0) * (tax_rate - 1.0)
                drawn = min(reserve_draw, self.stockpile.get(res, 0.0))
                self.stockpile[res]         = self.stockpile.get(res, 0.0) - drawn
                self.faction_stockpile[res] = self.faction_stockpile.get(res, 0.0) + drawn

    def pay_repair_upkeep(self) -> None:
        for b in self._buildings:
            costs = b.repair_upkeep_this_tick()
            if not costs:
                continue
            if not self._deduct(costs):
                b.state = BuildingState.DAMAGED
                self.last_events.append(f"Building {b.id} repair halted — insufficient resources.")
            else:
                for res, amt in costs.items():
                    self._add_to(self.last_consumed, res, amt)

    def feed_population(self) -> bool:
        """
        Consume ORGANICS to feed population.
        Returns True if fully fed. Starvation shrinks pop by 1% per tick.
        Critical flag FOOD_SHORTAGE fires after FOOD_SHORTAGE_TICKS consecutive ticks.
        """
        needed    = self.organics_upkeep_per_tick
        available = self.stockpile.get(R.ORGANICS, 0.0)
        if available >= needed:
            self.stockpile[R.ORGANICS] -= needed
            self._add_to(self.last_consumed, R.ORGANICS, needed)
            self._starving_ticks = 0
            self.critical_flags.discard(CriticalFlag.FOOD_SHORTAGE)
            return True
        else:
            self.stockpile[R.ORGANICS] = 0.0
            self._add_to(self.last_consumed, R.ORGANICS, available)
            loss = self.population * 0.01
            self.population = max(0.0, self.population - loss)
            self._starving_ticks += 1
            self.last_events.append(
                f"⚠ Starvation tick {self._starving_ticks}: pop -{loss:.1f} → {self.population:.1f}"
            )
            if self._starving_ticks >= FOOD_SHORTAGE_TICKS:
                self.critical_flags.add(CriticalFlag.FOOD_SHORTAGE)
            return False

    # ------------------------------------------------------------------
    # FLAG EVALUATION
    # ------------------------------------------------------------------

    def _net_rates(self) -> Dict[int, float]:
        """Approximate net resource rate = production - consumption across active buildings."""
        produced = colony_production_rates(self.building_counts)
        consumed = colony_production_costs(self.building_counts)
        all_keys = set(produced) | set(consumed)
        if R.ORGANICS not in consumed:
            consumed[R.ORGANICS] = self.organics_upkeep_per_tick
        else:
            consumed[R.ORGANICS] += self.organics_upkeep_per_tick
        
        return {k: produced.get(k, 0.0) - consumed.get(k, 0.0) for k in all_keys}

    def evaluate_flags(self) -> None:
        """
        Recompute all flags based on current colony state.
        Critical flags are set/cleared unconditionally.
        Strategic flags respect directive.override_flags.
        """
        net = self._net_rates()

        # --- Critical ---
        # POWER_DEFICIT
        power_net = net.get(4, 0.0)   # synthetic key 4 = POWER
        if power_net < POWER_DEFICIT_THRESHOLD:
            self.critical_flags.add(CriticalFlag.POWER_DEFICIT)
        else:
            self.critical_flags.discard(CriticalFlag.POWER_DEFICIT)

        # POPULATION_COLLAPSE
        if self.starting_pop > 0 and self.population / self.starting_pop < POPULATION_COLLAPSE_FRAC:
            self.critical_flags.add(CriticalFlag.POPULATION_COLLAPSE)
        else:
            self.critical_flags.discard(CriticalFlag.POPULATION_COLLAPSE)

        # FOOD_SHORTAGE is managed incrementally in feed_population()

        # --- Strategic ---
        def _set_strategic(flag: StrategicFlag, condition: bool) -> None:
            if flag in self.directive.override_flags:
                self.strategic_flags.discard(flag)
                return
            if condition:
                self.strategic_flags.add(flag)
            else:
                self.strategic_flags.discard(flag)

        # DEFENSE_NEEDED
        defense_net = net.get(6, 0.0)   # synthetic key 6 = DEFENSE
        _set_strategic(StrategicFlag.DEFENSE_NEEDED, defense_net < DEFENSE_LOW_THRESHOLD)

        # WORKER_SHORTAGE
        total_workforce = sum(s.workforce.get(1, 0) for b in self.active_buildings for s in [b.stats])
        unassigned      = len(self.unassigned_workers())
        shortage = total_workforce > 0 and (unassigned / max(total_workforce, 1)) < WORKER_SHORTAGE_RATIO
        _set_strategic(StrategicFlag.WORKER_SHORTAGE, shortage)

        # RESOURCE_LOW — track ticks of negative net rate per resource
        for rtype in ResourceType:
            key = int(rtype)
            if net.get(key, 0.0) < 0:
                self._resource_low_ticks[key] = self._resource_low_ticks.get(key, 0) + 1
            else:
                self._resource_low_ticks[key] = 0
        any_low = any(v >= RESOURCE_LOW_TICKS for v in self._resource_low_ticks.values())
        _set_strategic(StrategicFlag.RESOURCE_LOW, any_low)

        # EXPORT_STRAINED — faction tax is eating into operational headroom
        strained = False
        if self.directive.tax_rate > 0:
            for res, produced_amt in self.last_produced.items():
                local_after_tax = produced_amt * (1.0 - self.directive.tax_rate)
                total_in_pool   = self.stockpile.get(res, 0.0)
                if total_in_pool > 0 and local_after_tax / total_in_pool < EXPORT_STRAINED_THRESHOLD:
                    strained = True
                    break
        _set_strategic(StrategicFlag.EXPORT_STRAINED, strained)

        # CONSTRUCTION_BLOCKED — colony wants to build but cannot afford minimum cost
        constructing = any(b.state == BuildingState.CONSTRUCTING for b in self._buildings)
        if constructing:
            _set_strategic(StrategicFlag.CONSTRUCTION_BLOCKED, False)
        else:
            cheapest = min(
                (BUILDING_STATS[bt][1].build_cost for bt in BuildingType),
                key=lambda c: sum(c.values()),
                default={}
            )
            blocked = not self._can_afford(cheapest, BUILD_STOCKPILE_MIN)
            _set_strategic(StrategicFlag.CONSTRUCTION_BLOCKED, blocked)

    # ------------------------------------------------------------------
    # DECISION VALIDATION
    # ------------------------------------------------------------------

    def _construction_pipeline(self) -> Tuple[Dict[int, int], Dict[int, float]]:
        """
        Examine all buildings currently under construction.

        Returns
        -------
        pending_workers : Dict[int, int]
            {worker_level: count} of workers needed by all pending buildings
        pending_rate_impact : Dict[int, float]
            net production rate impact (production - consumption) for all
            pending buildings once they become active.
        """
        pending_workers: Dict[int, int] = {}
        pending_production: Dict[int, float] = {}
        pending_consumption: Dict[int, float] = {}

        for b in self._buildings:
            if b.state == BuildingState.CONSTRUCTING:
                s = b.stats
                for wl, cnt in s.workforce.items():
                    pending_workers[wl] = pending_workers.get(wl, 0) + cnt
                for res, rate in s.production_rate.items():
                    pending_production[res] = pending_production.get(res, 0) + rate
                for res, cost in s.production_cost.items():
                    pending_consumption[res] = pending_consumption.get(res, 0) + cost

        all_keys = set(pending_production) | set(pending_consumption)
        rate_impact = {
            k: pending_production.get(k, 0) - pending_consumption.get(k, 0)
            for k in all_keys
        }
        return pending_workers, rate_impact

    def _validate_build(self, building_type: BuildingType) -> Tuple[bool, List[str]]:
        """
        Validate whether building a new L1 building is safe.

        Runs all 4 checks: resource, workforce, rates, interference.
        Returns (is_valid, list_of_reasons_if_invalid).
        """
        stats = BUILDING_STATS[building_type][1]
        reasons = []
        pending_workers, pending_rate_impact = self._construction_pipeline()

        # 1. Resource check
        if not self._can_afford(stats.build_cost, BUILD_STOCKPILE_MIN):
            reasons.append(f"Cannot afford {building_type.name} L1")

        # 2. Workforce check (with interference)
        total_needed: Dict[int, int] = {}
        for wl, cnt in stats.workforce.items():
            total_needed[wl] = cnt + pending_workers.get(wl, 0)

        unassigned = self.unassigned_workers_by_level()
        free_pop = self.free_population
        for wl, needed in total_needed.items():
            avail = unassigned.get(wl, 0)
            if avail >= needed:
                continue
            if wl == 1:
                deficit = needed - avail
                if free_pop >= deficit * POP_PER_WORKER:
                    self.last_events.append(
                        f"Will need to recruit {deficit} L1 worker(s) "
                        f"({deficit * POP_PER_WORKER} pop) for {building_type.name}"
                    )
                    continue
            reasons.append(
                f"Not enough available L{wl} workers "
                f"({avail} avail, {needed} needed)"
            )

        # 3. Rates check (with interference)
        current_rates = self._net_rates()
        projected = dict(current_rates)
        for res, rate in pending_rate_impact.items():
            projected[res] = projected.get(res, 0) + rate
        for res, rate in stats.production_rate.items():
            projected[res] = projected.get(res, 0) + rate
        for res, cost in stats.production_cost.items():
            projected[res] = projected.get(res, 0) - cost

        target_resources = set(stats.production_rate.keys())
        for res, rate in projected.items():
            if rate < 0 and res not in target_resources:
                reasons.append(
                    f"{ResourceType(res).name} rate would be negative "
                    f"({rate:.1f}/tick)"
                )

        return len(reasons) == 0, reasons

    def _validate_upgrade(self, building: Building) -> Tuple[bool, List[str]]:
        """
        Validate whether upgrading an existing building is safe.

        Runs all 4 checks accounting for the current building's removal
        and the new level's addition.
        Returns (is_valid, list_of_reasons_if_invalid).
        """
        if building.level >= MAX_BUILDING_LEVEL:
            return False, ["Already at max level"]

        next_level = building.level + 1
        next_stats = BUILDING_STATS[building.building_type][next_level]
        current_stats = building.stats
        reasons = []
        pending_workers, pending_rate_impact = self._construction_pipeline()

        # 1. Resource check
        if not self._can_afford(next_stats.build_cost, BUILD_STOCKPILE_MIN):
            reasons.append(
                f"Cannot afford {building.building_type.name} "
                f"lv{building.level} \u2192 lv{next_level}"
            )

        # 2. Workforce check (with interference)
        total_needed: Dict[int, int] = {}
        for wl, cnt in next_stats.workforce.items():
            total_needed[wl] = cnt + pending_workers.get(wl, 0)

        unassigned = self.unassigned_workers_by_level()
        free_pop = self.free_population
        for wl, needed in total_needed.items():
            avail = unassigned.get(wl, 0)
            if avail >= needed:
                continue
            if wl == 1:
                deficit = needed - avail
                if free_pop >= deficit * POP_PER_WORKER:
                    self.last_events.append(
                        f"Will need to recruit {deficit} L1 worker(s) "
                        f"({deficit * POP_PER_WORKER} pop) for "
                        f"{building.building_type.name} upgrade"
                    )
                    continue
            reasons.append(
                f"Not enough available L{wl} workers "
                f"({avail} avail, {needed} needed)"
            )

        # 3. Rates check (with interference)
        current_rates = self._net_rates()
        projected = dict(current_rates)
        # Remove current building's impact (it stops producing during construction)
        for res, rate in current_stats.production_rate.items():
            projected[res] = projected.get(res, 0) - rate
        for res, cost in current_stats.production_cost.items():
            projected[res] = projected.get(res, 0) + cost
        # Add pending construction impacts
        for res, rate in pending_rate_impact.items():
            projected[res] = projected.get(res, 0) + rate
        # Add new level's impacts
        for res, rate in next_stats.production_rate.items():
            projected[res] = projected.get(res, 0) + rate
        for res, cost in next_stats.production_cost.items():
            projected[res] = projected.get(res, 0) - cost

        target_resources = set(next_stats.production_rate.keys())
        for res, rate in projected.items():
            if rate < 0 and res not in target_resources:
                reasons.append(
                    f"{ResourceType(res).name} rate would be negative "
                    f"({rate:.1f}/tick)"
                )

        return len(reasons) == 0, reasons

    # ------------------------------------------------------------------
    # RULE-BASED DECISION ENGINE
    # ------------------------------------------------------------------

    def execute_directive(self) -> None:
        """
        Rule-based colony agent. Called once per tick after resource collection.

        Priority order
        --------------
        0. Critical flag response  — always executed, cannot be overridden
        1. Repair damaged buildings
        2. Directive execution
        3. Balanced idle upkeep
        """
        d   = self.directive
        net = self._net_rates()

        # ── 0. BASIC SURVIVAL LOOP ─────────────────────────────────────────
        if not self._basic_survival_loop(net):
            return  # survival needs not met, skip directive execution

        # ── 1. REPAIR DAMAGED BUILDINGS ────────────────────────────────────
        for b in self._buildings:
            if b.state == BuildingState.DAMAGED and b.health < REPAIR_THRESHOLD:
                self.start_repair(b.id)

        # ── 2. DIRECTIVE EXECUTION ─────────────────────────────────────────
        if d.directive_type == DirectiveType.HARVEST:
            self._rule_harvest(d, net)

        elif d.directive_type == DirectiveType.BUILD:
            self._rule_build(d, net)

        elif d.directive_type == DirectiveType.UPGRADE:
            self._rule_upgrade(d, net)

        elif d.directive_type == DirectiveType.EXPAND:
            self._rule_expand(d, net)

        elif d.directive_type == DirectiveType.EXPORT:
            self._rule_export(d, net)

    # ── Basic Survival Loop ────────────────────────────────────────────────

    def _basic_survival_loop(self, net: Dict[int, float]) -> bool:
        """
        Handles survival-critical needs before directive execution.
        Returns True if survival is secured, False otherwise (skip directives).
        """
        # ── FOOD SHORTAGE ────────────────────────────────────────────────
        if CriticalFlag.FOOD_SHORTAGE in self.critical_flags or net.get(R.ORGANICS, 0.0) < FOOD_DEFICIT_THRESHOLD:
            farm_type = BuildingType.FARM
            # Check if a farm is already in the pipeline
            if any(b.building_type == farm_type and b.state == BuildingState.CONSTRUCTING for b in self._buildings):
                self.last_events.append("🚨 Farm already under construction (FOOD_SHORTAGE).")
            elif self._can_afford(BUILDING_STATS[farm_type][1].build_cost, BUILD_STOCKPILE_MIN):
                if self.can_staff_L1_building(farm_type) or self.free_population >= BUILDING_STATS[farm_type][1].workforce.get(1, 0):
                    self.construct_building(farm_type)
                    self.last_events.append("🚨 Built Farm (FOOD_SHORTAGE).")
            else:
                for b in self._buildings:
                    if b.is_active and b.building_type not in (BuildingType.FARM, BuildingType.MINE):
                        prod_cost = BUILDING_STATS[b.building_type][b.level].production_cost
                        if int(R.ORGANICS) in prod_cost:
                            self.unassign_workers_from_building(b)
                            self.last_events.append(f"🚨 Disabled {b.building_type.name} (FOOD_SHORTAGE).")

            return False

        # ── POWER DEFICIT ────────────────────────────────────────────────
        if CriticalFlag.POWER_DEFICIT in self.critical_flags:
            pp_type = BuildingType.POWER_PLANT
            if any(b.building_type == pp_type and b.state == BuildingState.CONSTRUCTING for b in self._buildings):
                self.last_events.append("🚨 Power Plant already under construction (POWER_DEFICIT).")
            elif self._can_afford(BUILDING_STATS[pp_type][1].build_cost, BUILD_STOCKPILE_MIN):
                if self.can_staff_L1_building(pp_type) or self.free_population >= BUILDING_STATS[pp_type][1].workforce.get(1, 0):
                    self.construct_building(pp_type)
                    self.last_events.append("🚨 Built Power Plant (POWER_DEFICIT).")
            else:
                for b in self._buildings:
                    if b.is_active and b.building_type not in (BuildingType.POWER_PLANT, BuildingType.MINE):
                        prod_cost = BUILDING_STATS[b.building_type][b.level].production_cost
                        if 4 in prod_cost:  # POWER
                            self.assign_workers_to_building(b)
                            self.last_events.append(f"🚨 Disabled {b.building_type.name} (POWER_DEFICIT).")
            return False

        # ── DEFENSE NEEDS ────────────────────────────────────────────────
        defense_needed = int(self.population * 10)  # ~10 defense per person
        current_defense = self.stockpile.get(R.DEFENSE, 0.0)  # synthetic key 6 = DEFENSE

        if current_defense < defense_needed:
            # check inactive buildings and see if any can be reactivated to meet defense needs
            total_rate = 0.0
            for b in self._buildings:
                if b.state == BuildingState.INACTIVE and b.building_type == BuildingType.FORT:
                    prod_rate = BUILDING_STATS[b.building_type][b.level].production_rate[6]  # DEFENSE
                    if self.can_staff_building(b) or self.free_population >= BUILDING_STATS[b.building_type][b.level].workforce.get(1, 0):
                        self.assign_workers_to_building(b)
                        self.last_events.append(f"🛡 Reactivated Fort (defense={current_defense + 1}/{defense_needed}).")
                        total_rate += prod_rate  # assume full production for immediate effect
                        if total_rate > (defense_needed - current_defense) / BOLSTER_DEFENSE_TICKS:
                            # enough buildings are staffed to supply defense in a given number of ticks
                            break
            fort_type = BuildingType.FORT
            if not any(b.building_type == fort_type and b.state == BuildingState.CONSTRUCTING for b in self._buildings):
                if self._can_afford(BUILDING_STATS[fort_type][1].build_cost, BUILD_STOCKPILE_MIN):
                    if self.free_population >= BUILDING_STATS[fort_type][1].workforce.get(1, 0):
                        self.construct_building(fort_type)
                        self.last_events.append(f"🛡 Built Fort (defense={current_defense + 1}/{defense_needed}).")
        elif current_defense > defense_needed:
            # unstaff all buildings
            excess_forts = [
                b for b in self._buildings
                if b.building_type == BuildingType.FORT and b.is_active
            ]
            for b in excess_forts:
                self.unassign_workers_from_building(b)
                self.last_events.append(f"🛡 Disabled excess Fort (defense={current_defense - 1}/{defense_needed}).")

        # ── WORKER SHORTAGE ──────────────────────────────────────────────
        buildings_needing_workers = []
        for b in self._buildings:
            if not b.is_active and (b.building_type is not BuildingType.FORT):
                if self.can_staff_building(b):
                    self.assign_workers_to_building(b)
                else:
                    buildings_needing_workers.append(b.id)


        if buildings_needing_workers:
            # go through each building and recruit workers until all are staffed or we run out of free population
            for b_id in buildings_needing_workers:
                b = self.get_building(b_id)
                if b is None:
                    continue
                required = b.stats.workforce
                unassigned = self.unassigned_workers_by_level()
                for level, count in required.items():
                    needed = count - unassigned.get(level, 0)
                    if needed > 0:
                        recruited = self.recruit_workers_of_level(level, needed)
                        if recruited > 0:
                            self.last_events.append(f"👷 Recruited {recruited} worker(s) at level {level} for {b.building_type.name} (id={b.id}).")

        # ── LAB UPGRADE ───────────────────────────────────────────────────
        labs = [b for b in self._buildings if b.building_type == BuildingType.LAB and b.is_active]
        if labs:
            labs.sort(key=lambda b: b.level, reverse=True)
            top_lab = labs[0]
            if top_lab.level < MAX_BUILDING_LEVEL:
                valid, reasons = self._validate_upgrade(top_lab)
                if valid:
                    self.upgrade_building(top_lab.id)
                    self.last_events.append(f"🔬 Upgraded LAB to lv{top_lab.level + 1}.")
                    return True
                else:
                    for r in reasons:
                        self.last_events.append(f"⛔ LAB upgrade rejected: {r}")

        return True

    # ── Directive sub-rules ────────────────────────────────────────────────

    def _rule_harvest(self, d: Directive, net: Dict[int, float]) -> None:
        """
        Gather large stockpile of target resource.
        Buildings perform normal tasks. Only build new buildings if survival at risk.
        Similar to idle state - let buildings work and maintain themselves.
        """
        for b in self._buildings:
            if b.is_active and b.health >= SURGE_HEALTH_MIN:
                self.set_surge(b.id, True)

    def _rule_build(self, d: Directive, net: Dict[int, float]) -> None:
        """
        Expand building count for target resource.
        Build new buildings to widen availability and maintain consistent flow.
        Only builds if all 4 validation checks pass.
        """
        if d.target_resource is None:
            return

        target_type = _resource_to_building(d.target_resource)
        if not target_type:
            return

        valid, reasons = self._validate_build(target_type)
        if not valid:
            for r in reasons:
                self.last_events.append(f"⛔ BUILD {target_type.name} rejected: {r}")
            return

        b = self.construct_building(target_type)
        if b:
            self.last_events.append(f"🏗 Queued {target_type.name} (BUILD).")

    def _rule_upgrade(self, d: Directive, net: Dict[int, float]) -> None:
        """
        Upgrade existing buildings for target resource.
        Only upgrades if all 4 validation checks pass.
        """
        if d.target_resource is None:
            return

        target_type = _resource_to_building(d.target_resource)
        if not target_type:
            return

        target_buildings = [
            b for b in self._buildings
            if b.building_type == target_type and b.is_active and b.level < MAX_BUILDING_LEVEL
        ]

        if not target_buildings:
            return

        target_buildings.sort(key=lambda b: b.level)

        for b in target_buildings:
            valid, reasons = self._validate_upgrade(b)
            if not valid:
                for r in reasons:
                    self.last_events.append(
                        f"⛔ UPGRADE {target_type.name} lv{b.level}\u2192lv{b.level+1} "
                        f"rejected: {r}"
                    )
                continue

            self.upgrade_building(b.id)
            self.last_events.append(
                f"⬆ Upgraded {target_type.name} to lv{b.level + 1} (UPGRADE)."
            )
            return

    def _rule_expand(self, d: Directive, net: Dict[int, float]) -> None:
        """
        Send ships and resources to establish a new colony on a new system.
        Build ships and accumulate resources for the new colony.
        """
        shipyard_type = BuildingType.SHIPYARD
        shipyard_count = sum(1 for b in self._buildings if b.building_type == shipyard_type and b.is_active)

        if shipyard_count == 0:
            valid, reasons = self._validate_build(shipyard_type)
            if valid:
                self.construct_building(shipyard_type)
                self.last_events.append("🏗 Queued Shipyard (EXPAND).")
            else:
                for r in reasons:
                    self.last_events.append(f"⛔ EXPAND Shipyard rejected: {r}")
            return

        ship_cost = BUILDING_STATS[shipyard_type][1].production_cost
        target_rate = net.get(d.target_resource, 0.0) if d.target_resource else 0.0

        if target_rate <= 0 and d.export_destination:
            self.last_events.append(f"⚠ Cannot export {d.target_resource} - no surplus.")
            return

    def _rule_export(self, d: Directive, net: Dict[int, float]) -> None:
        """
        Send target resource to another colony and trade for WEALTH.
        Do not export more than the colony can produce (net rate must stay positive).
        """
        if not d.export_destination or d.target_resource is None:
            return

        current_export = d.export_demand
        target_rate = net.get(d.target_resource, 0.0)

        if target_rate <= current_export:
            self.last_events.append(f"⚠ Export demand exceeds production for {d.target_resource}.")
            return

        self.last_events.append(f"📤 Exporting {current_export} of resource {d.target_resource} to {d.export_destination} for WEALTH.")

    # ------------------------------------------------------------------
    # TICK
    # ------------------------------------------------------------------

    def tick(self, verbose=True, enable_decision=True) -> None:
        """
        Advance the colony by one tick.

        Order of operations
        -------------------
        1. Reset per-tick ledgers
        2. Auto-recruit workers to meet building demand
        3. Advance constructions
        4. Collect resources + pay building upkeep (tax applied here)
        5. Pay repair upkeep + advance repairs
        6. Apply wear-and-tear damage
        7. Feed population
        8. Evaluate flags
        9. Execute directive (includes survival loop → then directive rule)
        """
        self._tick += 1
        self.last_produced = {}
        self.last_consumed = {}
        self.last_events   = []

        # Auto-recruit workers to meet building demand
        shortage = self.required_workers_by_level()
        if shortage:
            for lvl in sorted(shortage):   # fill lowest levels first
                gap = shortage[lvl]
                if gap <= 0:
                    continue
                if lvl == 1:
                    recruited = self.recruit_workers(gap)
                else:
                    if self._lab_can_train(lvl):
                        recruited = self.recruit_workers_of_level(lvl, gap)
                    else:
                        recruited = self.recruit_workers(gap)
                        if recruited:
                            lab_cost = BUILDING_STATS[BuildingType.LAB][lvl].build_cost
                            if self._can_afford(lab_cost, BUILD_STOCKPILE_MIN):
                                self.construct_building(BuildingType.LAB, level=lvl)

        # update construction progress and log completions
        for b in self._buildings:
            if b.advance_construction():
                self.last_events.append(
                    f"{b.building_type.name} lv{b.level} (id={b.id}) construction complete."
                )
                if not self.assign_workers_to_building(b):
                    b.state = BuildingState.INACTIVE

        # pull resources from producing buildings, applying tax and upkeep
        self.collect_resources()

        # increase population
        if self._rng is None:
            self._rng = np.random.default_rng(seed=self.colony_id)  # deterministic per-colony growth

        # check abundant organics before allowing growth
        if self.stockpile.get(R.ORGANICS, 0.0) < self.organics_upkeep_per_tick * 3:
            self.last_events.append("No population growth due to FOOD_SHORTAGE.")
        else:
            if self.directive.directive_type == DirectiveType.EXPAND:
                growth = 1.5*self._rng.lognormal(0, 0.5, size=1)
                decay = self._rng.lognormal(0, 0.75, size=1)
            else:
                growth = 1.2*self._rng.lognormal(0, 0.75, size=1)
                decay = self._rng.lognormal(0, 0.75, size=1)
            # find growth and decay
            pop_growth = np.ceil(self.population * growth[0]/250)
            pop_decay  = np.ceil(self.population * decay[0]/250)
            self.population += pop_growth - pop_decay

        # damage active buildings
        for b in self._buildings:
            if b.is_active:
                b.apply_damage()
                if b.state == BuildingState.DESTROYED:
                    self.last_events.append(
                        f"{b.building_type.name} lv{b.level} (id={b.id}) DESTROYED."
                    )

        # pay for building upkeep and apply repairs
        self.pay_repair_upkeep()
        for b in self._buildings:
            if b.state == BuildingState.ACTIVE and b.health < REPAIR_THRESHOLD*b.max_health:
                b.state = BuildingState.REPAIRING
                b.apply_repair()
            elif b.state == BuildingState.REPAIRING:
                b.apply_repair()
            elif b.state == BuildingState.DAMAGED:
                b.state = BuildingState.REPAIRING
                b.apply_repair()

        # colony action section
        self.feed_population()
        self.evaluate_flags()
        if enable_decision:
            self.execute_directive()


        # print the last events
        
        if verbose:
            print(f"======   Tick {self._tick}   ======")
            for ev in self.last_events:
                print(f"Colony {self.name} (id={self.colony_id}):  {ev}")

    # ------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------

    def flag_summary(self) -> str:
        crit = ", ".join(f.name for f in self.critical_flags) or "none"
        strat = ", ".join(f.name for f in self.strategic_flags) or "none"
        return f"  Critical: [{crit}]  Strategic: [{strat}]"

    def stockpile_summary(self) -> str:
        lines = ["  Local stockpile:"]
        for rtype in ResourceType:
            loc = self.stockpile.get(int(rtype), 0.0)
            fac = self.faction_stockpile.get(int(rtype), 0.0)
            lines.append(f"    {rtype.name:<12} local={loc:>9.2f}  faction={fac:>9.2f}")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [
            f"=== Colony: {self.name} (id={self.colony_id})  tick={self._tick} ===",
            f"  Directive : {self.directive.directive_type.name}"
            f"  tax={self.directive.tax_rate:.0%}  urgency={self.directive.urgency:.2f}",
            f"  Population: {self.population:>8.1f}"
            f"  (free: {self.free_population:.1f}, workers: {len(self._workers)})",
            f"  Organics upkeep/tick: {self.organics_upkeep_per_tick:.2f}",
            self.flag_summary(),
            self.stockpile_summary(),
            f"  Buildings ({len(self._buildings)} total):",
        ]
        for b in self._buildings:
            lines.append(b.summary())
        if self.last_events:
            lines.append("  Events this tick:")
            for ev in self.last_events:
                lines.append(f"    {ev}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# FACTION
# ---------------------------------------------------------------------------

@dataclass
class Faction:
    """
    Strategic agent owning one or more Colonies.

    Parameters
    ----------
    faction_id  : unique simulation-level int
    name        : display name
    treasury    : faction-level resource reserve (separate from colony stockpiles)
    """
    faction_id: int
    name:       str
    treasury:   Dict[int, float]  = field(default_factory=dict)

    _colonies:  List[Colony]      = field(default_factory=list, repr=False)
    _tick:      int               = field(default=0,            repr=False)

    # ------------------------------------------------------------------
    # COLONY MANAGEMENT
    # ------------------------------------------------------------------

    def add_colony(self, colony: Colony) -> None:
        self._colonies.append(colony)

    def get_colony(self, colony_id: int) -> Optional[Colony]:
        for c in self._colonies:
            if c.colony_id == colony_id:
                return c
        return None

    @property
    def colonies(self) -> List[Colony]:
        return list(self._colonies)

    # ------------------------------------------------------------------
    # DIRECTIVE ISSUING
    # ------------------------------------------------------------------

    def issue_directive(
        self,
        colony_id:         int,
        directive_type:    DirectiveType,
        tax_rate:          float                  = 0.10,
        urgency:           float                  = 0.5,
        target_resource:   Optional[int]          = None,
        export_destination: Optional[str]         = None,
        export_demand:     float                  = 0.0,
        override_flags:    Optional[Set[StrategicFlag]] = None,
    ) -> bool:
        """
        Send a directive to a colony. Returns False if colony not found.
        Critical flags on the receiving colony are checked: if the directive
        would worsen a critical condition (e.g. EXPORT to a starving colony)
        the tax_rate is clamped to 0 and an event is logged.
        """
        c = self.get_colony(colony_id)
        if c is None:
            return False

        effective_tax = tax_rate
        if CriticalFlag.FOOD_SHORTAGE in c.critical_flags or \
           CriticalFlag.POPULATION_COLLAPSE in c.critical_flags:
            if directive_type == DirectiveType.EXPORT and tax_rate > 0:
                effective_tax = 0.0

        c.directive = Directive(
            directive_type      = directive_type,
            tax_rate            = effective_tax,
            urgency             = urgency,
            target_resource     = target_resource,
            export_destination  = export_destination,
            export_demand       = export_demand,
            override_flags      = override_flags or set(),
        )
        return True

    # ------------------------------------------------------------------
    # INTER-COLONY TRANSFERS
    # ------------------------------------------------------------------

    def collect_taxes(self) -> None:
        """
        Pull each colony's faction_stockpile into the faction treasury.
        Called by Faction.tick() after all colonies have ticked.
        """
        for c in self._colonies:
            for res, amt in c.faction_stockpile.items():
                self.treasury[res] = self.treasury.get(res, 0.0) + amt
            c.faction_stockpile = {}

    def transfer_to_colony(
        self,
        colony_id: int,
        resources: Dict[int, float],
    ) -> bool:
        """
        Move resources from the faction treasury to a colony's local stockpile.
        Returns True if treasury had sufficient funds.
        """
        c = self.get_colony(colony_id)
        if c is None:
            return False
        for res, amt in resources.items():
            if self.treasury.get(res, 0.0) < amt:
                return False
        for res, amt in resources.items():
            self.treasury[res] = self.treasury.get(res, 0.0) - amt
            c.stockpile[res]   = c.stockpile.get(res, 0.0) + amt
        return True

    def transfer_between_colonies(
        self,
        from_id:   int,
        to_id:     int,
        resources: Dict[int, float],
    ) -> bool:
        """
        Direct colony-to-colony resource transfer (peer request or faction order).
        Deducts from the source colony's local stockpile.
        Returns True on success.
        """
        src = self.get_colony(from_id)
        dst = self.get_colony(to_id)
        if src is None or dst is None:
            return False
        for res, amt in resources.items():
            if src.stockpile.get(res, 0.0) < amt:
                return False
        for res, amt in resources.items():
            src.stockpile[res] = src.stockpile.get(res, 0.0) - amt
            dst.stockpile[res] = dst.stockpile.get(res, 0.0) + amt
        return True

    # ------------------------------------------------------------------
    # FLAG AGGREGATION
    # ------------------------------------------------------------------

    def critical_colonies(self) -> List[Colony]:
        """Returns colonies with any active critical flags."""
        return [c for c in self._colonies if c.critical_flags]

    def colonies_with_flag(self, flag: StrategicFlag) -> List[Colony]:
        return [c for c in self._colonies if flag in c.strategic_flags]

    # ------------------------------------------------------------------
    # TICK
    # ------------------------------------------------------------------

    def tick(self, verbose=True, enable_decision=True) -> None:
        """
        Advance the faction by one tick.

        Order of operations
        -------------------
        1. Tick all colonies (each colony ticks independently)
        2. Collect taxes into treasury
        3. Faction-level strategic logic (stub — ML agent entry point)
        """
        self._tick += 1
        for colony in self._colonies:
            colony.tick(verbose=verbose, enable_decision=enable_decision)
        self.collect_taxes()
        self._faction_strategy()
        if not enable_decision:
            self.last_events.append("Directive execution skipped (enable_decision=False).")

    def _faction_strategy(self) -> None:
        """
        Stub for the faction-level agent.
        Currently implements a simple rule: redirect treasury resources to any
        colony raising a FOOD_SHORTAGE critical flag.
        Replace with ML agent call when ready.
        """
        for c in self.critical_colonies():
            if CriticalFlag.FOOD_SHORTAGE in c.critical_flags:
                aid = {int(R.ORGANICS): min(500.0, self.treasury.get(int(R.ORGANICS), 0.0))}
                if aid[int(R.ORGANICS)] > 0:
                    self.transfer_to_colony(c.colony_id, aid)

    # ------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------

    def treasury_summary(self) -> str:
        lines = ["  Treasury:"]
        for rtype in ResourceType:
            val = self.treasury.get(int(rtype), 0.0)
            lines.append(f"    {rtype.name:<12} {val:>10.2f}")
        return "\n".join(lines)

    def summary(self) -> str:
        lines = [
            f"╔══ Faction: {self.name} (id={self.faction_id})  tick={self._tick} ══╗",
            self.treasury_summary(),
            f"  Colonies: {len(self._colonies)}",
        ]
        for c in self._colonies:
            crit  = f" 🚨[{','.join(f.name for f in c.critical_flags)}]"  if c.critical_flags  else ""
            strat = f" ⚑[{','.join(f.name for f in c.strategic_flags)}]" if c.strategic_flags else ""
            lines.append(
                f"    [{c.colony_id}] {c.name:<20} "
                f"pop={c.population:>7.1f}  "
                f"directive={c.directive.directive_type.name:<8}"
                f"{crit}{strat}"
            )
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# HELPER — map resource key to its primary producer building
# ---------------------------------------------------------------------------

def _resource_to_building(resource_key: int) -> Optional[BuildingType]:
    _map = {
        int(R.MINERALS):  BuildingType.MINE,
        int(R.POWER):     BuildingType.POWER_PLANT,
        int(R.ORGANICS):  BuildingType.FARM,
        int(R.RARE_MATS): BuildingType.MINE,
        int(R.RESEARCH):  BuildingType.LAB,
    }
    return _map.get(resource_key)


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Build a faction with two colonies ---
    faction = Faction(faction_id=0, name="House Atreides")

    home = Colony(
        colony_id  = 0,
        name       = "Arrakeen",
        system_id  = 0,
        population = 800.0,
        stockpile  = {
            int(R.MINERALS): 2000.0,
            int(R.POWER):    500.0,
            int(R.ORGANICS): 8000.0,
            int(R.RARE_MATS):  20.0,
        },
    )

    outpost = Colony(
        colony_id  = 1,
        name       = "Sietch Tabr",
        system_id  = 1,
        population = 200.0,
        stockpile  = {
            int(R.MINERALS):  500.0,
            int(R.POWER):    100.0,
            int(R.ORGANICS):  800.0,
            int(R.RARE_MATS):   5.0,
        },
    )

    faction.add_colony(home)
    faction.add_colony(outpost)

    # Seed some buildings
    home.construct_building(BuildingType.FARM,        planet_index=0)
    home.construct_building(BuildingType.MINE,        planet_index=0)
    home.construct_building(BuildingType.POWER_PLANT, planet_index=0)
    outpost.construct_building(BuildingType.FARM,     planet_index=0)

    # Issue directives
    faction.issue_directive(0, DirectiveType.HARVEST, tax_rate=0.15, urgency=0.7)
    faction.issue_directive(1, DirectiveType.EXPAND,  tax_rate=0.05, urgency=0.8)

    # Run 20 ticks
    for t in range(1, 21):
        faction.tick()
        if t in (1, 5, 10, 15, 20):
            print(faction.summary())
            print(home.summary())
            print(outpost.summary())
            print("─" * 70)
