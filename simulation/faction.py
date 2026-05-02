"""
simulation/faction.py
---------------------
Faction: the top-level agent in the simulation.

Responsibilities
----------------
- Owns a stockpile of resources collected from planets.
- Maintains a population that must be fed each tick (ORGANICS upkeep).
- Converts population into Workers (level 1) on demand.
- Owns a roster of Buildings, each with runtime state (health, ticks remaining).
- Each tick:
    1. Collect resources from all active buildings.
    2. Pay per-tick building upkeep costs.
    3. Feed the population (consume ORGANICS).
    4. Age buildings (apply damage_rate to health).
    5. Advance any buildings still under construction.
- Exposes helpers to construct, upgrade, and collect from buildings.

Exports
-------
WorkerLevel         - IntEnum  1-5
Worker              - dataclass
Building            - runtime building instance
Faction             - main class
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

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

# Organics consumed per population unit per tick.
# Population above FEED_BUFFER will begin starving if organics run short.
ORGANICS_PER_POP: float = 0.05

# Fraction of a building's max health below which it enters DAMAGED state.
DAMAGED_THRESHOLD: float = 0.50

# Starting health for any newly constructed building (as a fraction 0-1).
INITIAL_HEALTH: float = 1.0

# Worker conversion: how many population units become one level-1 worker.
POP_PER_WORKER: int = 10


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
    level: WorkerLevel
    assigned_building_id: Optional[int] = None   # None = unassigned

    @property
    def is_assigned(self) -> bool:
        return self.assigned_building_id is not None


# ---------------------------------------------------------------------------
# BUILDING  (runtime instance)
# ---------------------------------------------------------------------------

@dataclass
class Building:
    """
    Runtime instance of a building owned by a Faction.

    The static stats (costs, rates, etc.) are always fetched from
    BUILDING_STATS[building_type][level] — never duplicated here.

    Fields
    ------
    id              : unique int within the faction
    building_type   : BuildingType enum
    level           : current level (1-5)
    state           : BuildingState enum
    health          : float in [0.0, 1.0]
    ticks_remaining : ticks until CONSTRUCTING → ACTIVE (or upgrade finishes)
    planet_index    : which planet in the solar system this sits on
    """
    id:             int
    building_type:  BuildingType
    level:          int               = 1
    state:          BuildingState     = BuildingState.CONSTRUCTING
    health:         float             = INITIAL_HEALTH
    ticks_remaining: int              = 0
    planet_index:   Optional[int]     = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> BuildingLevelStats:
        return BUILDING_STATS[self.building_type][self.level]

    @property
    def is_active(self) -> bool:
        return self.state == BuildingState.ACTIVE

    @property
    def is_producing(self) -> bool:
        """Producing = ACTIVE or SURGING."""
        return self.state in (BuildingState.ACTIVE, BuildingState.SURGING)

    @property
    def surge_multiplier(self) -> float:
        return 1.5 if self.state == BuildingState.SURGING else 1.0

    # ------------------------------------------------------------------
    # Tick helpers (called by Faction.tick())
    # ------------------------------------------------------------------

    def apply_damage(self) -> None:
        """Reduce health by this tick's damage rate (double while SURGING)."""
        if self.state not in (BuildingState.ACTIVE, BuildingState.SURGING):
            return
        rate = self.stats.damage_rate * (2.0 if self.state == BuildingState.SURGING else 1.0)
        self.health = max(0.0, self.health - rate / 100.0)
        if self.health == 0.0:
            self.state = BuildingState.DESTROYED
        elif self.health < DAMAGED_THRESHOLD and self.state != BuildingState.SURGING:
            self.state = BuildingState.DAMAGED

    def apply_repair(self) -> None:
        """Increase health while REPAIRING."""
        if self.state != BuildingState.REPAIRING:
            return
        self.health = min(1.0, self.health + self.stats.repair_rate / 100.0)
        if self.health >= 1.0:
            self.state = BuildingState.ACTIVE

    def advance_construction(self) -> bool:
        """
        Count down one tick of construction/upgrade.
        Returns True when the building becomes ACTIVE.
        """
        if self.state != BuildingState.CONSTRUCTING:
            return False
        self.ticks_remaining = max(0, self.ticks_remaining - 1)
        if self.ticks_remaining == 0:
            self.state  = BuildingState.ACTIVE
            self.health = INITIAL_HEALTH
            return True
        return False

    def production_this_tick(self) -> Dict[int, float]:
        """
        Resources produced this tick.
        Returns {} if not in a producing state.
        """
        if not self.is_producing:
            return {}
        return {
            resource: amount * self.surge_multiplier
            for resource, amount in self.stats.production_rate.items()
        }

    def upkeep_this_tick(self) -> Dict[int, float]:
        """Resources consumed this tick (only while producing)."""
        if not self.is_producing:
            return {}
        return dict(self.stats.production_cost)

    def repair_upkeep_this_tick(self) -> Dict[int, float]:
        """Resources consumed per tick while REPAIRING."""
        if self.state != BuildingState.REPAIRING:
            return {}
        return dict(self.stats.repair_cost)

    def summary(self) -> str:
        bar_filled = int(self.health * 20)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        extra = f"  {self.ticks_remaining}t left" if self.state == BuildingState.CONSTRUCTING else ""
        return (
            f"  [{self.id:>3}] {self.building_type.name:<12} lv{self.level}"
            f"  {self.state.name:<12} [{bar}] {self.health*100:>5.1f}%{extra}"
        )


# ---------------------------------------------------------------------------
# FACTION
# ---------------------------------------------------------------------------

@dataclass
class Faction:
    """
    Top-level simulation agent.

    Parameters
    ----------
    name            : display name
    population      : total population (not yet workers)
    stockpile       : starting resources  {ResourceType: amount}
    seed            : optional int for reproducibility (reserved for future RNG use)
    """
    name:       str
    population: float                    = 1000.0
    stockpile:  Dict[int, float]         = field(default_factory=dict)

    # Internal
    _buildings:     List[Building]       = field(default_factory=list, repr=False)
    _workers:       List[Worker]         = field(default_factory=list, repr=False)
    _next_id:       int                  = field(default=0,            repr=False)
    _tick:          int                  = field(default=0,            repr=False)

    # Per-tick tracking (populated each call to tick())
    last_produced:  Dict[int, float]     = field(default_factory=dict, repr=False)
    last_consumed:  Dict[int, float]     = field(default_factory=dict, repr=False)
    last_events:    List[str]            = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _new_id(self) -> int:
        bid = self._next_id
        self._next_id += 1
        return bid

    def _add_to(self, ledger: Dict[int, float], resource: int, amount: float) -> None:
        ledger[resource] = ledger.get(resource, 0.0) + amount

    def _deduct(self, costs: Dict[int, float]) -> bool:
        """
        Attempt to deduct `costs` from the stockpile atomically.
        Returns True on success, False (no deduction made) if insufficient funds.
        """
        for resource, amount in costs.items():
            if self.stockpile.get(resource, 0.0) < amount:
                return False
        for resource, amount in costs.items():
            self.stockpile[resource] = self.stockpile.get(resource, 0.0) - amount
        return True

    # ------------------------------------------------------------------
    # POPULATION & WORKERS
    # ------------------------------------------------------------------

    @property
    def free_population(self) -> float:
        """Population not yet converted to workers."""
        return self.population - len(self._workers) * POP_PER_WORKER

    def recruit_workers(self, count: int = 1) -> int:
        """
        Convert `count` groups of POP_PER_WORKER population into level-1 workers.
        Returns the number of workers actually created.
        """
        max_possible = int(self.free_population // POP_PER_WORKER)
        count = min(count, max_possible)
        for _ in range(count):
            self._workers.append(Worker(level=WorkerLevel.L1))
        return count

    def workers_at_level(self, level: int) -> List[Worker]:
        return [w for w in self._workers if w.level == level]

    def unassigned_workers(self) -> List[Worker]:
        return [w for w in self._workers if not w.is_assigned]

    @property
    def organics_upkeep_per_tick(self) -> float:
        """ORGANICS needed each tick to feed the entire population."""
        return self.population * ORGANICS_PER_POP

    # ------------------------------------------------------------------
    # BUILDING MANAGEMENT
    # ------------------------------------------------------------------

    def construct_building(
        self,
        building_type: BuildingType,
        planet_index:  Optional[int] = None,
        level:         int           = 1,
    ) -> Optional[Building]:
        """
        Deduct build cost and queue a new building for construction.
        Returns the Building on success, None if the faction can't afford it.
        """
        stats = BUILDING_STATS[building_type][level]
        if not self._deduct(stats.build_cost):
            return None
        b = Building(
            id            = self._new_id(),
            building_type = building_type,
            level         = level,
            state         = BuildingState.CONSTRUCTING,
            health        = 0.0,   # health rises to 1.0 when done
            ticks_remaining = stats.build_ticks,
            planet_index  = planet_index,
        )
        self._buildings.append(b)
        return b

    def upgrade_building(self, building_id: int) -> bool:
        """
        Attempt to upgrade a building by one level.
        Returns True if successful.
        """
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
        """
        Set a DAMAGED building to REPAIRING (costs are paid per tick).
        Returns True if the transition was valid.
        """
        b = self.get_building(building_id)
        if b is None or b.state != BuildingState.DAMAGED:
            return False
        b.state = BuildingState.REPAIRING
        return True

    def set_surge(self, building_id: int, active: bool) -> bool:
        """Toggle SURGING on an ACTIVE building."""
        b = self.get_building(building_id)
        if b is None:
            return False
        if active and b.state == BuildingState.ACTIVE:
            b.state = BuildingState.SURGING
            return True
        if not active and b.state == BuildingState.SURGING:
            b.state = BuildingState.ACTIVE
            return True
        return False

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
        """Returns {(BuildingType, level): count} for active buildings."""
        counts: Dict[Tuple[BuildingType, int], int] = {}
        for b in self.active_buildings:
            key = (b.building_type, b.level)
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    @property
    def resource_rates(self) -> Dict[int, float]:
        """Returns net production rates for each resource based on active buildings."""
        rates: Dict[int, float] = {}
        for b in self.active_buildings:
            prod = b.production_this_tick()
            upkeep = b.upkeep_this_tick()
            for res, amt in prod.items():
                rates[res] = rates.get(res, 0.0) + amt
            for res, amt in upkeep.items():
                rates[res] = rates.get(res, 0.0) - amt
        return rates

    # ------------------------------------------------------------------
    # RESOURCE COLLECTION
    # ------------------------------------------------------------------

    def collect_resources(self) -> Dict[int, float]:
        """
        Collect production from all producing buildings into the stockpile.
        Upkeep costs are deducted; buildings that can't be fed become INACTIVE.
        Returns net resources gained this call.
        """
        gained: Dict[int, float] = {}
        for b in self._buildings:
            if not b.is_producing:
                continue
            upkeep = b.upkeep_this_tick()
            if upkeep and not self._deduct(upkeep):
                b.state = BuildingState.INACTIVE
                self.last_events.append(
                    f"Building {b.id} ({b.building_type.name} lv{b.level}) went INACTIVE — upkeep deficit."
                )
                continue
            production = b.production_this_tick()
            for res, amt in production.items():
                self.stockpile[res] = self.stockpile.get(res, 0.0) + amt
                self._add_to(gained, res, amt)
                self._add_to(self.last_produced, res, amt)
        return gained

    def pay_repair_upkeep(self) -> None:
        """Deduct per-tick repair costs; buildings that can't be paid halt repairs."""
        for b in self._buildings:
            costs = b.repair_upkeep_this_tick()
            if costs and not self._deduct(costs):
                b.state = BuildingState.DAMAGED   # revert to damaged, can't repair
                self.last_events.append(
                    f"Building {b.id} repair halted — insufficient resources."
                )
            else:
                for res, amt in costs.items():
                    self._add_to(self.last_consumed, res, amt)

    def feed_population(self) -> bool:
        """
        Consume ORGANICS to feed the population.
        Returns True if fully fed, False if in deficit (starvation begins).
        Starvation shrinks population by 1% per tick.
        """
        needed = self.organics_upkeep_per_tick
        available = self.stockpile.get(R.ORGANICS, 0.0)
        if available >= needed:
            self.stockpile[R.ORGANICS] -= needed
            self._add_to(self.last_consumed, R.ORGANICS, needed)
            return True
        else:
            self.stockpile[R.ORGANICS] = 0.0
            self._add_to(self.last_consumed, R.ORGANICS, available)
            starvation_loss = self.population * 0.01
            self.population = max(0.0, self.population - starvation_loss)
            self.last_events.append(
                f"Starvation! Population fell by {starvation_loss:.1f} --> {self.population:.1f}"
            )
            return False
        

    # ------------------------------------------------------------------
    # TICK
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """
        Advance the simulation by one tick.

        Order of operations
        -------------------
        1. Reset per-tick ledgers.
        2. Advance constructions.
        3. Collect production & pay building upkeep.
        4. Pay repair upkeep & advance repairs.
        5. Apply damage to producing buildings.
        6. Feed the population.
        """
        self._tick += 1
        self.last_produced = {}
        self.last_consumed = {}
        self.last_events   = []

        # 2. Construction countdown
        for b in self._buildings:
            if b.advance_construction(): # only returns True when construction finishes
                self.last_events.append(
                    f"Building {b.id} ({b.building_type.name} lv{b.level}) construction complete."
                )

        # 3. Collect production + pay per-tick upkeep
        self.collect_resources()

        # 4. Repair upkeep + repair health
        self.pay_repair_upkeep()
        for b in self._buildings:
            b.apply_repair()

        # 5. Apply wear-and-tear to producing buildings
        for b in self._buildings:
            if b.is_producing:
                b.apply_damage()
                if b.state == BuildingState.DESTROYED:
                    self.last_events.append(
                        f"Building {b.id} ({b.building_type.name} lv{b.level}) DESTROYED."
                    )

        # 6. Feed population
        self.feed_population()

    # ------------------------------------------------------------------
    # REPORTING
    # ------------------------------------------------------------------

    def stockpile_summary(self) -> str:
        lines = ["  Stockpile:"]
        for rtype in ResourceType:
            val = self.stockpile.get(rtype, 0.0)
            lines.append(f"    {rtype.name:<12} {val:>10.2f}")
        return "\n".join(lines)
    
    def stockpile_snapshot(self) -> Dict[int, float]:
        """Returns a copy of the current stockpile."""
        return dict(self.stockpile)

    def summary(self) -> str:
        lines = [
            f"=== Faction: {self.name}  (tick {self._tick}) ===",
            f"  Population : {self.population:>8.1f}  "
            f"(free: {self.free_population:.1f}, workers: {len(self._workers)})",
            f"  Organics upkeep/tick: {self.organics_upkeep_per_tick:.2f}",
            self.stockpile_summary(),
            f"  Buildings  ({len(self._buildings)} total):",
        ]
        for b in self._buildings:
            lines.append(b.summary())
        if self.last_events:
            lines.append("  Events this tick:")
            for ev in self.last_events:
                lines.append(f"    {ev}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# QUICK SMOKE-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from buildings import BuildingType

    # Spin up a faction with starter resources
    f = Faction(
        name      = "Test Faction",
        population= 500.0,
        stockpile = {
            R.MINERALS:  1000.0,
            R.ENERGY:     200.0,
            R.ORGANICS:  5000.0,
            R.RARE_MATS:   10.0,
        },
    )

    # Recruit some workers from population
    recruited = f.recruit_workers(5)
    print(f"Recruited {recruited} workers.")

    # Build a Mine and a Farm on planet 0
    mine = f.construct_building(BuildingType.MINE, planet_index=0, level=1)
    farm = f.construct_building(BuildingType.FARM, planet_index=0, level=1)
    print(f"Queued: Mine id={mine.id}, Farm id={farm.id}")

    # Run 15 ticks
    for t in range(1, 16):
        f.tick()
        if t <= 2 or t % 5 == 0:
            print(f"\n--- Tick {t} ---")
            print(f.summary())
