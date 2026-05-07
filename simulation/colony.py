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
WorkerLevel     - IntEnum 1-5
Worker          - dataclass
Building        - runtime building instance
CriticalFlag    - IntEnum
StrategicFlag   - IntEnum
DirectiveType   - IntEnum
Directive       - dataclass
Colony          - local agent
Faction         - strategic agent
"""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from buildings import (
    BuildingType,
    BuildingState,
    BuildingLevelStats,
    BUILDING_STATS,
    MAX_BUILDING_LEVEL,
    ResourceType,
    colony_production_rates,
    colony_production_costs,
)

R = ResourceType


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

ORGANICS_PER_POP:  float = 0.05   # organics consumed per population unit per tick
DAMAGED_THRESHOLD: float = 0.50   # health fraction below which → DAMAGED
INITIAL_HEALTH:    float = 1.0    # health on construction complete
POP_PER_WORKER:    int   = 10     # population units per level-1 worker

# Flag thresholds
FOOD_SHORTAGE_TICKS:       int   = 3     # consecutive starving ticks before FOOD_SHORTAGE
POWER_DEFICIT_THRESHOLD:   float = 0.0   # net POWER below this → POWER_DEFICIT
POPULATION_COLLAPSE_FRAC:  float = 0.25  # population fallen to this fraction of start → COLLAPSE
DEFENSE_LOW_THRESHOLD:     float = 50.0  # net DEFENSE score below this → DEFENSE_NEEDED
WORKER_SHORTAGE_RATIO:     float = 0.5   # unassigned workers / total workforce below this
RESOURCE_LOW_TICKS:        int   = 5     # ticks of negative net rate before RESOURCE_LOW
EXPORT_STRAINED_THRESHOLD: float = 0.10  # local stockpile fraction remaining after tax

# Rule-based thresholds
SURGE_HEALTH_MIN:     float = 0.80   # don't surge buildings below this health
REPAIR_PRIORITY_FRAC: float = 0.60   # start repairs when health drops below this
BUILD_STOCKPILE_MIN:  float = 1.5    # multiplier: must have 1.5× build cost in stockpile


# ---------------------------------------------------------------------------
# WORKER
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
    ticks_remaining: int           = 0
    planet_index:    Optional[int] = None

    @property
    def stats(self) -> BuildingLevelStats:
        return BUILDING_STATS[self.building_type][self.level]

    @property
    def is_active(self) -> bool:
        return self.state == BuildingState.ACTIVE

    @property
    def is_producing(self) -> bool:
        return self.state in (BuildingState.ACTIVE, BuildingState.SURGING)

    @property
    def surge_multiplier(self) -> float:
        return 1.5 if self.state == BuildingState.SURGING else 1.0

    # ------------------------------------------------------------------
    # Per-tick methods
    # ------------------------------------------------------------------

    def apply_damage(self) -> None:
        """Reduce health by this tick's damage rate (doubled while SURGING)."""
        if self.state not in (BuildingState.ACTIVE, BuildingState.SURGING):
            return
        rate = self.stats.damage_rate * (2.0 if self.state == BuildingState.SURGING else 1.0)
        self.health = max(0.0, self.health - rate / 100.0)
        if self.health == 0.0:
            self.state = BuildingState.DESTROYED
        elif self.health < DAMAGED_THRESHOLD and self.state != BuildingState.SURGING:
            self.state = BuildingState.DAMAGED

    def apply_repair(self) -> None:
        """Advance repair health; transition to ACTIVE when full."""
        if self.state != BuildingState.REPAIRING:
            return
        self.health = min(1.0, self.health + self.stats.repair_rate / 100.0)
        if self.health >= 1.0:
            self.state = BuildingState.ACTIVE

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
        if not self.is_producing:
            return {}
        return {r: amt * self.surge_multiplier for r, amt in self.stats.production_rate.items()}

    def upkeep_this_tick(self) -> Dict[int, float]:
        if not self.is_producing:
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
    HARVEST = 0   # maximise resource collection; minimise non-essential spending
    DEFEND  = 1   # prioritise Forts and defensive buildings
    EXPAND  = 2   # prioritise new building construction and population growth
    EXPORT  = 3   # accumulate faction sub-stockpile; reduce local spending
    IDLE    = 4   # balanced upkeep only; no expansion or surge


@dataclass
class Directive:
    """
    Issued by a Faction to one of its Colonies each tick.

    Fields
    ------
    directive_type  : primary intent
    tax_rate        : fraction [0.0, 1.0+] of each resource produced that
                      flows into the colony's faction sub-stockpile.
                      Values > 1.0 will draw from local reserves.
    urgency         : scalar [0.0, 1.0] — how aggressively to pursue the
                      directive vs. balanced upkeep. 1.0 = full commitment.
    target_resource : optional ResourceType to focus HARVEST directives
    target_building : optional BuildingType to focus DEFEND/EXPAND directives
    export_dest_id  : colony id that should receive faction sub-stockpile transfers
    override_flags  : set of StrategicFlag values the faction explicitly suppresses
                      (critical flags are never suppressible)
    """
    directive_type:   DirectiveType          = DirectiveType.IDLE
    tax_rate:         float                  = 0.10
    urgency:          float                  = 0.5
    target_resource:  Optional[int]          = None
    target_building:  Optional[BuildingType] = None
    export_dest_id:   Optional[int]          = None
    override_flags:   Set[StrategicFlag]     = field(default_factory=set)


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

    def workers_at_level(self, level: int) -> List[Worker]:
        return [w for w in self._workers if int(w.level) == level]

    def assign_workers_to_building(self, building: Building) -> bool:
        """
        Assign workers to a building based on its workforce requirements.
        Returns True if all required workers were assigned, False if partial/incomplete.
        If not enough workers exist, assigns what's available and returns False.
        
        Note: For upgrades, existing workers are first unassigned so they can be
        re-assigned if they meet the new requirements (or return to the pool).
        """
        # First unassign all current workers so we can re-assign fresh for upgrades
        currently_assigned = [w for w in self._workers if w.assigned_building_id == building.id]
        for w in currently_assigned:
            w.assigned_building_id = None

        required = building.stats.workforce
        assigned = []
        success = True

        for level, count in required.items():
            available = [w for w in self.unassigned_workers() if int(w.level) == level]
            assign_count = min(count, len(available))
            for w in available[:assign_count]:
                w.assigned_building_id = building.id
                assigned.append(w)
            if assign_count < count:
                success = False

        if assigned:
            self.last_events.append(
                f"Assigned {len(assigned)} worker(s) to {building.building_type.name} "
                f"lv{building.level} (id={building.id})."
            )
        if not success:
            self.last_events.append(
                f"⚠ {building.building_type.name} lv{building.level} (id={building.id}) "
                f"INACTIVE — worker shortage."
            )
        return success
    
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
        for b in self.active_buildings:
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
        return [b for b in self._buildings if b.is_producing]

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
            if not b.is_producing: # if not ACTIVE or SURGING, skip production and upkeep
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
        cheapest = min(
            (BUILDING_STATS[bt][1].build_cost for bt in BuildingType),
            key=lambda c: sum(c.values()),
            default={}
        )
        blocked = not self._can_afford(cheapest, BUILD_STOCKPILE_MIN)
        _set_strategic(StrategicFlag.CONSTRUCTION_BLOCKED, blocked)

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

        # ── 0. CRITICAL FLAG RESPONSE ──────────────────────────────────────
        if CriticalFlag.FOOD_SHORTAGE in self.critical_flags:
            # Emergency: build a Farm if we can afford one, else surge existing
            if not any(b.building_type == BuildingType.FARM and b.state == BuildingState.CONSTRUCTING
                       for b in self._buildings):
                farm_cost = BUILDING_STATS[BuildingType.FARM][1].build_cost
                if self._can_afford(farm_cost):
                    b = self.construct_building(BuildingType.FARM)
                    if b:
                        self.last_events.append("🚨 Emergency Farm queued (FOOD_SHORTAGE).")
            for b in self._buildings:
                if b.building_type == BuildingType.FARM and b.is_active:
                    self.set_surge(b.id, True)

        if CriticalFlag.POWER_DEFICIT in self.critical_flags:
            # Emergency: build a Power Plant or stop surging the heaviest consumers
            if not any(b.building_type == BuildingType.POWER_PLANT and
                       b.state == BuildingState.CONSTRUCTING for b in self._buildings):
                pp_cost = BUILDING_STATS[BuildingType.POWER_PLANT][1].build_cost
                if self._can_afford(pp_cost):
                    b = self.construct_building(BuildingType.POWER_PLANT)
                    if b:
                        self.last_events.append("🚨 Emergency Power Plant queued (POWER_DEFICIT).")
            # De-surge all non-essential buildings to cut power consumption
            for b in self._buildings:
                if b.state == BuildingState.SURGING and b.building_type not in (
                    BuildingType.POWER_PLANT, BuildingType.FARM
                ):
                    self.set_surge(b.id, False)

        # ── 1. REPAIR DAMAGED BUILDINGS ────────────────────────────────────
        for b in self._buildings:
            if b.state == BuildingState.DAMAGED and b.health < REPAIR_PRIORITY_FRAC:
                self.start_repair(b.id)

        # ── 2. DIRECTIVE EXECUTION ─────────────────────────────────────────
        if d.directive_type == DirectiveType.HARVEST:
            self._rule_harvest(d, net)

        elif d.directive_type == DirectiveType.DEFEND:
            self._rule_defend(d, net)

        elif d.directive_type == DirectiveType.EXPAND:
            self._rule_expand(d, net)

        elif d.directive_type == DirectiveType.EXPORT:
            self._rule_export(d, net)

        else:  # IDLE
            self._rule_idle(net)

    # ── Directive sub-rules ────────────────────────────────────────────────

    def _rule_harvest(self, d: Directive, net: Dict[int, float]) -> None:
        """
        Maximise resource yield. Surge producing buildings that are healthy.
        Avoid new construction unless a critical resource is trending negative.
        """
        for b in self._buildings:
            if b.is_active and b.health >= SURGE_HEALTH_MIN:
                self.set_surge(b.id, True)

        # Only build if a targeted resource has a negative net rate
        if d.target_resource is not None and net.get(d.target_resource, 0.0) < 0:
            target_type = _resource_to_building(d.target_resource)
            if target_type:
                cost = BUILDING_STATS[target_type][1].build_cost
                if self._can_afford(cost, BUILD_STOCKPILE_MIN):
                    self.construct_building(target_type)

    def _rule_defend(self, d: Directive, net: Dict[int, float]) -> None:
        """
        Prioritise Fort construction. Keep existing buildings healthy but avoid
        surging non-defensive buildings.
        """
        target = d.target_building or BuildingType.FORT
        cost   = BUILDING_STATS[target][1].build_cost
        # Build up to urgency-scaled count of defensive buildings
        current_def = sum(1 for b in self._buildings if b.building_type == target and b.is_active)
        desired_def = max(1, int(d.urgency * 3))
        if current_def < desired_def and self._can_afford(cost, BUILD_STOCKPILE_MIN):
            b = self.construct_building(target)
            if b:
                self.last_events.append(f"🏰 Queued {target.name} (DEFEND directive).")

        # Surge Forts only
        for b in self._buildings:
            if b.building_type == BuildingType.FORT and b.is_active and b.health >= SURGE_HEALTH_MIN:
                self.set_surge(b.id, True)
            elif b.building_type != BuildingType.FORT and b.state == BuildingState.SURGING:
                self.set_surge(b.id, False)

    def _rule_expand(self, d: Directive, net: Dict[int, float]) -> None:
        """
        Prioritise new construction and population growth.
        Build the most needed building type based on net resource rates,
        then recruit workers if surplus population exists.
        """
        # Find the resource with the most negative net rate — build its producer
        deficit_resource = min(
            (k for k in net if k in [int(r) for r in ResourceType]),
            key=lambda k: net[k],
            default=None
        )
        if deficit_resource is not None and net[deficit_resource] < 0:
            target_type = _resource_to_building(deficit_resource)
            if target_type:
                cost = BUILDING_STATS[target_type][1].build_cost
                if self._can_afford(cost, BUILD_STOCKPILE_MIN):
                    b = self.construct_building(target_type)
                    if b:
                        self.last_events.append(f"🏗 Queued {target_type.name} (EXPAND — deficit).")
        elif d.target_building:
            cost = BUILDING_STATS[d.target_building][1].build_cost
            if self._can_afford(cost, BUILD_STOCKPILE_MIN):
                b = self.construct_building(d.target_building)
                if b:
                    self.last_events.append(f"🏗 Queued {d.target_building.name} (EXPAND directive).")

        # Recruit workers if free population allows
        if self.free_population >= POP_PER_WORKER * 2:
            recruited = self.recruit_workers(max(1, int(d.urgency * 3)))
            if recruited:
                self.last_events.append(f"Recruited {recruited} worker(s) (EXPAND directive).")

    def _rule_export(self, d: Directive, net: Dict[int, float]) -> None:
        """
        Accumulate faction sub-stockpile. Avoid new builds unless survival is at risk.
        De-surge non-essential buildings to reduce consumption and upkeep.
        """
        for b in self._buildings:
            if b.state == BuildingState.SURGING and b.building_type not in (
                BuildingType.FARM, BuildingType.POWER_PLANT
            ):
                self.set_surge(b.id, False)

        # Only build if critically needed and not overridden by critical flags
        if CriticalFlag.FOOD_SHORTAGE not in self.critical_flags:
            return  # handled in step 0 if critical; otherwise hold

    def _rule_idle(self, net: Dict[int, float]) -> None:
        """
        Balanced upkeep. De-surge everything, recruit minimally, upgrade if
        there is a comfortable stockpile surplus.
        """
        for b in self._buildings:
            if b.state == BuildingState.SURGING:
                self.set_surge(b.id, False)

        # Upgrade the lowest-level active building if surplus allows
        candidates = sorted(
            [b for b in self._buildings if b.is_active and b.level < MAX_BUILDING_LEVEL],
            key=lambda b: b.level,
        )
        for b in candidates:
            next_stats = BUILDING_STATS[b.building_type][b.level + 1]
            if self._can_afford(next_stats.build_cost, BUILD_STOCKPILE_MIN * 2):
                if self.upgrade_building(b.id):
                    self.last_events.append(
                        f"⬆ Upgrading {b.building_type.name} id={b.id} to lv{b.level} (IDLE surplus)."
                    )
                    break   # one upgrade per tick

    # ------------------------------------------------------------------
    # TICK
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """
        Advance the colony by one tick.

        Order of operations
        -------------------
        1. Reset per-tick ledgers
        2. Advance constructions
        3. Collect resources + pay building upkeep (tax applied here)
        4. Pay repair upkeep + advance repairs
        5. Apply wear-and-tear damage
        6. Feed population
        7. Evaluate flags
        8. Execute directive (rule-based agent)
        """
        self._tick += 1
        self.last_produced = {}
        self.last_consumed = {}
        self.last_events   = []

        # Auto-recruit / promote workers to meet per-level demand
        shortage = self.required_workers_by_level()
        if shortage:
            for lvl in sorted(shortage):   # fill lowest levels first
                gap = shortage[lvl]
                if gap <= 0:
                    continue
                if lvl == 1:
                    recruited = self.recruit_workers(gap)
                    if recruited:
                        self.last_events.append(
                            f"Auto-recruited {recruited} L1 worker(s) to meet demand."
                        )
                else:
                    # Only promote/recruit for higher levels if a lab supports it
                    if self._lab_can_train(lvl):
                        promoted = self.recruit_workers_of_level(lvl, gap)
                        if promoted:
                            self.last_events.append(
                                f"Auto-recruited/promoted {promoted} L{lvl} worker(s) to meet demand."
                            )
                    else:
                        # No lab capable — recruit L1s as a best-effort fallback
                        recruited = self.recruit_workers(gap)
                        if recruited:
                            self.last_events.append(
                                f"Auto-recruited {recruited} L1 worker(s) (no Lab for L{lvl} training)."
                            )

        # Try to reactivate INACTIVE buildings now that workers may have been recruited
        for b in self._buildings:
            if b.state == BuildingState.INACTIVE:
                if self.assign_workers_to_building(b):
                    b.state = BuildingState.ACTIVE
                    self.last_events.append(
                        f"Building {b.id} ({b.building_type.name} lv{b.level}) reactivated - workers now available."
                    )

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

        # pay for building upkeep and apply repairs
        self.pay_repair_upkeep()
        for b in self._buildings:
            b.apply_repair()

        # increase population
        if self._rng is None:
            self._rng = np.random.default_rng(seed=self.colony_id)  # deterministic per-colony growth

        # check abundant organics before allowing growth
        if self.stockpile.get(R.ORGANICS, 0.0) < self.organics_upkeep_per_tick * 3:
            self.last_events.append("No population growth due to FOOD_SHORTAGE.")
        else:
            if self.directive.directive_type == DirectiveType.EXPAND:
                growth = self._rng.lognormal(0.5, 0.5, size=1)
                decay = self._rng.lognormal(0, 0.75, size=1)
            else:
                growth = self._rng.lognormal(0.05, 0.75, size=1)
                decay = self._rng.lognormal(0, 0.75, size=1)
            # find growth and decay
            pop_growth = np.ceil(self.population * growth[0]/100)
            pop_decay  = np.ceil(self.population * decay[0]/100)
            self.population += pop_growth - pop_decay

        # damage active buildings
        for b in self._buildings:
            if b.is_producing:
                b.apply_damage()
                if b.state == BuildingState.DESTROYED:
                    self.last_events.append(
                        f"{b.building_type.name} lv{b.level} (id={b.id}) DESTROYED."
                    )

        self.feed_population()
        self.evaluate_flags()
        self.execute_directive()

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
        colony_id:      int,
        directive_type: DirectiveType,
        tax_rate:       float                  = 0.10,
        urgency:        float                  = 0.5,
        target_resource: Optional[int]         = None,
        target_building: Optional[BuildingType]= None,
        export_dest_id:  Optional[int]         = None,
        override_flags:  Optional[Set[StrategicFlag]] = None,
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

        # Guard: never export from a critically stressed colony
        effective_tax = tax_rate
        if CriticalFlag.FOOD_SHORTAGE in c.critical_flags or \
           CriticalFlag.POPULATION_COLLAPSE in c.critical_flags:
            if directive_type == DirectiveType.EXPORT and tax_rate > 0:
                effective_tax = 0.0
                # Still allow the directive, just zero the tax

        c.directive = Directive(
            directive_type  = directive_type,
            tax_rate        = effective_tax,
            urgency         = urgency,
            target_resource = target_resource,
            target_building = target_building,
            export_dest_id  = export_dest_id,
            override_flags  = override_flags or set(),
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

    def tick(self) -> None:
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
            colony.tick()
        self.collect_taxes()
        self._faction_strategy()

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
        int(R.ENERGY):    BuildingType.POWER_PLANT,
        int(R.ORGANICS):  BuildingType.FARM,
        int(R.RARE_MATS): BuildingType.MINE,
        4:                BuildingType.POWER_PLANT,   # POWER
        9:                BuildingType.LAB,            # RESEARCH
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
            int(R.ENERGY):    500.0,
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
            int(R.ENERGY):    100.0,
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
