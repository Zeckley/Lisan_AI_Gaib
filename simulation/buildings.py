"""
simulation/buildings.py
=======================
Static building definitions for the Spherical Strategy Sim.

Design principles
-----------------
- BUILDING_STATS is the single source of truth: a nested dict keyed by
  (BuildingType, level) → BuildingLevelStats dataclass.
- All numeric fields use ResourceType as keys so colony-level aggregation
  is a simple loop:  sum(count[b][lv] * stats.production_rate[r] for b, lv)
- Nothing in this file carries runtime state (health, ticks_remaining, etc.).
  That lives in Building instances in faction.py.
- Values are deliberately round numbers — easy to re-balance in one place.

Resource keys used throughout
------------------------------
  R.MINERALS  = 0
  R.ENERGY    = 1
  R.ORGANICS  = 2
  R.RARE_MATS = 3
  POWER       = synthetic resource (index 4) for internal book-keeping
               produced by Power Plants, consumed by other buildings.
               Not in ResourceType so it won't appear in stockpile vectors
               unless you choose to add it.  Represented as integer 4 here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ResourceType(IntEnum):
    MINERALS  = 0
    ENERGY    = 1   # raw planetary energy (geothermal / solar tap)
    ORGANICS  = 2
    RARE_MATS = 3

R = ResourceType   # short alias used inside this file

POWER = 4          # synthetic resource index — produced by Power Plants


class BuildingType(IntEnum):
    MINE        = 0
    POWER_PLANT = 1
    FARM        = 2
    FACTORY     = 3   # becomes a Recycler at lv4-5 (same type, different profile)
    FORT        = 4
    SHIPYARD    = 5
    RAILYARD    = 6
    LAB         = 7


class DepartmentType(IntEnum):
    RESOURCES       = 0
    ENERGY          = 1
    AGRICULTURE     = 2
    MANUFACTURING   = 3
    DEFENSE         = 4
    TRANSPORTATION  = 5
    COMMERCE        = 6
    INTELLIGENCE    = 7


# Which department owns which building
BUILDING_DEPARTMENT: Dict[BuildingType, DepartmentType] = {
    BuildingType.MINE:        DepartmentType.RESOURCES,
    BuildingType.POWER_PLANT: DepartmentType.ENERGY,
    BuildingType.FARM:        DepartmentType.AGRICULTURE,
    BuildingType.FACTORY:     DepartmentType.MANUFACTURING,
    BuildingType.FORT:        DepartmentType.DEFENSE,
    BuildingType.SHIPYARD:    DepartmentType.TRANSPORTATION,
    BuildingType.RAILYARD:    DepartmentType.COMMERCE,
    BuildingType.LAB:         DepartmentType.INTELLIGENCE,
}


class BuildingState(IntEnum):
    CONSTRUCTING = 0
    ACTIVE       = 1
    DAMAGED      = 2   # health < 50 % — production halted
    REPAIRING    = 3
    SURGING      = 4   # 1.5× production, 2× damage rate
    DESTROYED    = 5   # health == 0, cannot be rebuilt; drop scrap
    INACTIVE     = 6   # manually toggled or workforce shortage


# ---------------------------------------------------------------------------
# Per-level stats dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildingLevelStats:
    """
    All numeric dicts are  {resource_index: amount_per_tick}.
    Use ResourceType members or the POWER constant as keys.

    production_rate  — resources produced per tick while ACTIVE
    production_cost  — resources consumed per tick while ACTIVE
    build_cost       — one-time cost to construct / upgrade to this level
    repair_cost      — cost per tick while REPAIRING
    damage_rate      — health % lost per tick while ACTIVE (or 2× while SURGING)
    repair_rate      — health % gained per tick while REPAIRING
    build_ticks      — ticks to finish construction / upgrade
    workforce        — {worker_level: count_required}  (worker levels 1-5)
    notes            — human-readable string, e.g. "unlocks RARE_MATS harvest"
    """
    level:           int
    build_cost:      Dict[int, float]
    production_rate: Dict[int, float]
    production_cost: Dict[int, float]
    damage_rate:     float                   # health % / tick
    repair_cost:     Dict[int, float]
    repair_rate:     float                   # health % / tick
    build_ticks:     int
    workforce:       Dict[int, int]          # {worker_level: count}
    notes:           str = ""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _stats(
    level:           int,
    build_cost:      Dict[int, float],
    production_rate: Dict[int, float],
    production_cost: Dict[int, float],
    damage_rate:     float,
    repair_cost:     Dict[int, float],
    repair_rate:     float,
    build_ticks:     int,
    workforce:       Dict[int, int],
    notes:           str = "",
) -> BuildingLevelStats:
    return BuildingLevelStats(
        level=level,
        build_cost=build_cost,
        production_rate=production_rate,
        production_cost=production_cost,
        damage_rate=damage_rate,
        repair_cost=repair_cost,
        repair_rate=repair_rate,
        build_ticks=build_ticks,
        workforce=workforce,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# MINE  (Department of Resources)
# ---------------------------------------------------------------------------
#
# Produces MINERALS at all levels.
# Unlocks RARE_MATS at level 4.
# Consumes nothing — extraction is purely mechanical.
# Repair cost mirrors build cost (same minerals needed).

MINE_STATS: Dict[int, BuildingLevelStats] = {
    1: _stats(
        level=1,
        build_cost      = {R.MINERALS: 50},
        production_rate = {R.MINERALS: 10},
        production_cost = {},
        damage_rate     = 0.5,
        repair_cost     = {R.MINERALS: 5},
        repair_rate     = 2.0,
        build_ticks     = 10,
        workforce       = {1: 4},
        notes           = "Basic mineral extraction.",
    ),
    2: _stats(
        level=2,
        build_cost      = {R.MINERALS: 120},
        production_rate = {R.MINERALS: 22},
        production_cost = {},
        damage_rate     = 0.6,
        repair_cost     = {R.MINERALS: 10},
        repair_rate     = 2.0,
        build_ticks     = 20,
        workforce       = {1: 3, 2: 2},
        notes           = "Deeper shafts; improved yield.",
    ),
    3: _stats(
        level=3,
        build_cost      = {R.MINERALS: 250, R.ENERGY: 30},
        production_rate = {R.MINERALS: 40},
        production_cost = {},
        damage_rate     = 0.7,
        repair_cost     = {R.MINERALS: 20, R.ENERGY: 5},
        repair_rate     = 1.5,
        build_ticks     = 35,
        workforce       = {2: 4, 3: 1},
        notes           = "Mechanised extraction; upgrade cost includes energy components.",
    ),
    4: _stats(
        level=4,
        build_cost      = {R.MINERALS: 400, R.RARE_MATS: 10},
        production_rate = {R.MINERALS: 60, R.RARE_MATS: 2},
        production_cost = {},
        damage_rate     = 0.8,
        repair_cost     = {R.MINERALS: 30, R.RARE_MATS: 1},
        repair_rate     = 1.5,
        build_ticks     = 55,
        workforce       = {2: 3, 3: 2, 4: 1},
        notes           = "Unlocks RARE_MATS harvest.",
    ),
    5: _stats(
        level=5,
        build_cost      = {R.MINERALS: 700, R.RARE_MATS: 25},
        production_rate = {R.MINERALS: 90, R.RARE_MATS: 5},
        production_cost = {},
        damage_rate     = 1.0,
        repair_cost     = {R.MINERALS: 50, R.RARE_MATS: 2},
        repair_rate     = 1.0,
        build_ticks     = 80,
        workforce       = {3: 3, 4: 2, 5: 1},
        notes           = "Deep-core extraction; maximum yield.",
    ),
}


# ---------------------------------------------------------------------------
# POWER PLANT  (Department of Energy)
# ---------------------------------------------------------------------------
#
# Produces POWER (synthetic resource index 4).
# Lv1-2 consume raw planetary ENERGY.
# Lv3-4 consume ORGANICS (biofuel / fusion feedstock).
# Lv5   consumes nothing — zero-point / antimatter; but costs RARE_MATS to build.

POWER_PLANT_STATS: Dict[int, BuildingLevelStats] = {
    1: _stats(
        level=1,
        build_cost      = {R.MINERALS: 60},
        production_rate = {POWER: 20},
        production_cost = {R.ENERGY: 5},
        damage_rate     = 0.3,
        repair_cost     = {R.MINERALS: 6},
        repair_rate     = 2.5,
        build_ticks     = 12,
        workforce       = {1: 2, 2: 1},
        notes           = "Geothermal tap; consumes planetary ENERGY.",
    ),
    2: _stats(
        level=2,
        build_cost      = {R.MINERALS: 140},
        production_rate = {POWER: 45},
        production_cost = {R.ORGANICS: 6},
        damage_rate     = 0.4,
        repair_cost     = {R.MINERALS: 12},
        repair_rate     = 2.0,
        build_ticks     = 22,
        workforce       = {1: 2, 2: 2},
        notes           = "Expanded geothermal; Converts to ORGANICS.",
    ),
    3: _stats(
        level=3,
        build_cost      = {R.MINERALS: 280, R.ORGANICS: 40},
        production_rate = {POWER: 90},
        production_cost = {R.ORGANICS: 8},
        damage_rate     = 0.4,
        repair_cost     = {R.MINERALS: 20, R.ORGANICS: 5},
        repair_rate     = 2.0,
        build_ticks     = 40,
        workforce       = {2: 3, 3: 1},
        notes           = "Biofuel reactor; switches fuel source to ORGANICS.",
    ),
    4: _stats(
        level=4,
        build_cost      = {R.MINERALS: 500, R.ORGANICS: 80, R.RARE_MATS: 5},
        production_rate = {POWER: 180},
        production_cost = {R.ORGANICS: 15},
        damage_rate     = 0.5,
        repair_cost     = {R.MINERALS: 40, R.ORGANICS: 10},
        repair_rate     = 1.5,
        build_ticks     = 60,
        workforce       = {2: 2, 3: 2, 4: 1},
        notes           = "Fusion reactor; high output, still organic-fed.",
    ),
    5: _stats(
        level=5,
        build_cost      = {R.MINERALS: 900, R.RARE_MATS: 50},
        production_rate = {POWER: 400},
        production_cost = {},   # zero running cost
        damage_rate     = 0.6,
        repair_cost     = {R.MINERALS: 60, R.RARE_MATS: 5},
        repair_rate     = 1.0,
        build_ticks     = 100,
        workforce       = {3: 2, 4: 2, 5: 1},
        notes           = "Zero-point / antimatter plant; no fuel; RARE_MATS required to build.",
    ),
}


# ---------------------------------------------------------------------------
# FARM  (Department of Agriculture)
# ---------------------------------------------------------------------------
#
# Produces ORGANICS at all levels.
# Lv1-4 require no inputs.
# Lv5 uses a small amount of POWER for hydroponics/aeroponics boost.

FARM_STATS: Dict[int, BuildingLevelStats] = {
    1: _stats(
        level=1,
        build_cost      = {R.MINERALS: 40},
        production_rate = {R.ORGANICS: 12},
        production_cost = {},
        damage_rate     = 0.2,
        repair_cost     = {R.MINERALS: 4},
        repair_rate     = 3.0,
        build_ticks     = 8,
        workforce       = {1: 5},
        notes           = "Open-field farming.",
    ),
    2: _stats(
        level=2,
        build_cost      = {R.MINERALS: 90},
        production_rate = {R.ORGANICS: 26},
        production_cost = {},
        damage_rate     = 0.2,
        repair_cost     = {R.MINERALS: 8},
        repair_rate     = 3.0,
        build_ticks     = 16,
        workforce       = {1: 4, 2: 1},
        notes           = "Irrigation systems; improved yield.",
    ),
    3: _stats(
        level=3,
        build_cost      = {R.MINERALS: 180},
        production_rate = {R.ORGANICS: 50},
        production_cost = {},
        damage_rate     = 0.25,
        repair_cost     = {R.MINERALS: 15},
        repair_rate     = 2.5,
        build_ticks     = 28,
        workforce       = {2: 4, 3: 1},
        notes           = "Greenhouse complex; climate-controlled.",
    ),
    4: _stats(
        level=4,
        build_cost      = {R.MINERALS: 320, R.ORGANICS: 50},
        production_rate = {R.ORGANICS: 90},
        production_cost = {},
        damage_rate     = 0.3,
        repair_cost     = {R.MINERALS: 25, R.ORGANICS: 5},
        repair_rate     = 2.0,
        build_ticks     = 45,
        workforce       = {2: 3, 3: 2},
        notes           = "Vertical farming towers; no power yet required.",
    ),
    5: _stats(
        level=5,
        build_cost      = {R.MINERALS: 550, R.RARE_MATS: 15},
        production_rate = {R.ORGANICS: 180},
        production_cost = {POWER: 10},   # small power draw for grow-lights
        damage_rate     = 0.35,
        repair_cost     = {R.MINERALS: 40, R.RARE_MATS: 2},
        repair_rate     = 1.5,
        build_ticks     = 65,
        workforce       = {3: 3, 4: 2},
        notes           = "Aeroponics / gene-optimised crops; needs POWER, huge yield.",
    ),
}


# ---------------------------------------------------------------------------
# FACTORY / RECYCLER  (Department of Manufacturing)
# ---------------------------------------------------------------------------
#
# Lv1-3: Factory — converts MINERALS → building materials (represented as
#         a bonus multiplier on build_cost reduction, modelled here as
#         producing a synthetic resource MATS=5; OR treat MINERALS output
#         as pre-processed build stock and multiply by a config factor).
#         For simplicity we treat "building materials" as MINERALS_PROCESSED
#         and represent them as MINERALS in the output (same resource, but
#         produced faster than raw mining).
#
# Lv4-5: Recycler mode unlocked — can ALSO accept ORGANICS as feedstock,
#         producing the same processed minerals more efficiently.
#
# Production cost at lv1-3 = raw MINERALS (input feedstock).
# Production cost at lv4-5 = MINERALS or ORGANICS (factory chooses cheapest;
#         modelled by listing both with the OR note).

FACTORY_STATS: Dict[int, BuildingLevelStats] = {
    1: _stats(
        level=1,
        build_cost      = {R.MINERALS: 80},
        production_rate = {R.MINERALS: 15},   # processed / refined output
        production_cost = {R.MINERALS: 10},   # raw feedstock consumed
        damage_rate     = 0.4,
        repair_cost     = {R.MINERALS: 8},
        repair_rate     = 2.0,
        build_ticks     = 15,
        workforce       = {1: 3, 2: 2},
        notes           = "Basic smelter; net +5 minerals/tick after feedstock.",
    ),
    2: _stats(
        level=2,
        build_cost      = {R.MINERALS: 180},
        production_rate = {R.MINERALS: 35},
        production_cost = {R.MINERALS: 20},
        damage_rate     = 0.5,
        repair_cost     = {R.MINERALS: 15},
        repair_rate     = 2.0,
        build_ticks     = 28,
        workforce       = {1: 2, 2: 3},
        notes           = "Automated assembly line; net +15/tick.",
    ),
    3: _stats(
        level=3,
        build_cost      = {R.MINERALS: 350, R.ENERGY: 40},
        production_rate = {R.MINERALS: 70},
        production_cost = {R.MINERALS: 35},
        damage_rate     = 0.6,
        repair_cost     = {R.MINERALS: 28, R.ENERGY: 5},
        repair_rate     = 1.5,
        build_ticks     = 45,
        workforce       = {2: 3, 3: 2},
        notes           = "Heavy fabrication; net +35/tick.",
    ),
    4: _stats(
        level=4,
        # Recycler mode unlocked: can substitute ORGANICS for some MINERALS
        build_cost      = {R.MINERALS: 600, R.RARE_MATS: 12},
        production_rate = {R.MINERALS: 120},
        production_cost = {R.MINERALS: 40, R.ORGANICS: 20},  # dual feedstock
        damage_rate     = 0.7,
        repair_cost     = {R.MINERALS: 45, R.RARE_MATS: 1},
        repair_rate     = 1.5,
        build_ticks     = 65,
        workforce       = {2: 2, 3: 3, 4: 1},
        notes           = "Recycler mode: ORGANICS feedstock unlocked; net +60/tick.",
    ),
    5: _stats(
        level=5,
        build_cost      = {R.MINERALS: 950, R.RARE_MATS: 30},
        production_rate = {R.MINERALS: 200, R.RARE_MATS: 1},
        production_cost = {R.MINERALS: 60, R.ORGANICS: 35},
        damage_rate     = 0.8,
        repair_cost     = {R.MINERALS: 70, R.RARE_MATS: 3},
        repair_rate     = 1.0,
        build_ticks     = 90,
        workforce       = {3: 3, 4: 2, 5: 1},
        notes           = "Nano-fabrication; net +115/tick from dual feedstock.",
    ),
}


# ---------------------------------------------------------------------------
# FORT  (Department of Defense)
# ---------------------------------------------------------------------------
#
# Does not produce a harvestable resource.
# Produces DEFENSE_SCORE — represented here as a special key DEFENSE=6.
# Consumes a small amount of POWER and ORGANICS (garrison supply).

DEFENSE = 6   # synthetic key for defense rating contribution

FORT_STATS: Dict[int, BuildingLevelStats] = {
    1: _stats(
        level=1,
        build_cost      = {R.MINERALS: 100},
        production_rate = {DEFENSE: 10},
        production_cost = {POWER: 2, R.ORGANICS: 2},
        damage_rate     = 0.3,
        repair_cost     = {R.MINERALS: 10},
        repair_rate     = 2.0,
        build_ticks     = 18,
        workforce       = {1: 4},
        notes           = "Perimeter wall; basic garrison.",
    ),
    2: _stats(
        level=2,
        build_cost      = {R.MINERALS: 220},
        production_rate = {DEFENSE: 24},
        production_cost = {POWER: 4, R.ORGANICS: 4},
        damage_rate     = 0.3,
        repair_cost     = {R.MINERALS: 18},
        repair_rate     = 2.0,
        build_ticks     = 30,
        workforce       = {1: 3, 2: 2},
        notes           = "Reinforced bunkers; improved garrison.",
    ),
    3: _stats(
        level=3,
        build_cost      = {R.MINERALS: 420, R.ENERGY: 30},
        production_rate = {DEFENSE: 50},
        production_cost = {POWER: 8, R.ORGANICS: 6},
        damage_rate     = 0.35,
        repair_cost     = {R.MINERALS: 35, R.ENERGY: 5},
        repair_rate     = 1.5,
        build_ticks     = 50,
        workforce       = {2: 4, 3: 1},
        notes           = "Shield emitters; energy components required.",
    ),
    4: _stats(
        level=4,
        build_cost      = {R.MINERALS: 700, R.RARE_MATS: 8},
        production_rate = {DEFENSE: 100},
        production_cost = {POWER: 14, R.ORGANICS: 8},
        damage_rate     = 0.4,
        repair_cost     = {R.MINERALS: 55, R.RARE_MATS: 1},
        repair_rate     = 1.5,
        build_ticks     = 70,
        workforce       = {3: 3, 4: 2},
        notes           = "Planetary defense grid; sensor arrays online.",
    ),
    5: _stats(
        level=5,
        build_cost      = {R.MINERALS: 1100, R.RARE_MATS: 25},
        production_rate = {DEFENSE: 200},
        production_cost = {POWER: 25, R.ORGANICS: 12},
        damage_rate     = 0.5,
        repair_cost     = {R.MINERALS: 80, R.RARE_MATS: 3},
        repair_rate     = 1.0,
        build_ticks     = 100,
        workforce       = {3: 2, 4: 3, 5: 1},
        notes           = "Orbital defense platform; maximum deterrence.",
    ),
}


# ---------------------------------------------------------------------------
# SHIPYARD  (Department of Transportation)
# ---------------------------------------------------------------------------
#
# Produces COLONY_SHIPS (synthetic key SHIPS=7).
# Each ship costs a batch of resources; the production_cost here is per tick
# of active production toward a ship (ship_ticks is the number of ticks per ship).
# For simplicity, treat production_rate as ship_progress/tick and let the colony
# logic accumulate until ≥ 100 to spawn a ship.

SHIPS = 7

SHIPYARD_STATS: Dict[int, BuildingLevelStats] = {
    1: _stats(
        level=1,
        build_cost      = {R.MINERALS: 150, R.ORGANICS: 20},
        production_rate = {SHIPS: 2},    # ship-progress/tick; 100 = 1 ship
        production_cost = {R.MINERALS: 8, R.ORGANICS: 4, POWER: 5},
        damage_rate     = 0.3,
        repair_cost     = {R.MINERALS: 15, R.ORGANICS: 2},
        repair_rate     = 2.0,
        build_ticks     = 25,
        workforce       = {1: 3, 2: 2},
        notes           = "Drydock; builds slow colony ships (50 ticks/ship).",
    ),
    2: _stats(
        level=2,
        build_cost      = {R.MINERALS: 300, R.ORGANICS: 40},
        production_rate = {SHIPS: 4},
        production_cost = {R.MINERALS: 14, R.ORGANICS: 7, POWER: 8},
        damage_rate     = 0.35,
        repair_cost     = {R.MINERALS: 25, R.ORGANICS: 4},
        repair_rate     = 2.0,
        build_ticks     = 40,
        workforce       = {1: 2, 2: 3, 3: 1},
        notes           = "Expanded drydock; 25 ticks/ship.",
    ),
    3: _stats(
        level=3,
        build_cost      = {R.MINERALS: 500, R.ORGANICS: 60, R.ENERGY: 30},
        production_rate = {SHIPS: 7},
        production_cost = {R.MINERALS: 20, R.ORGANICS: 10, POWER: 12},
        damage_rate     = 0.4,
        repair_cost     = {R.MINERALS: 40, R.ORGANICS: 6, R.ENERGY: 4},
        repair_rate     = 1.5,
        build_ticks     = 60,
        workforce       = {2: 3, 3: 2},
        notes           = "Modular assembly; ~14 ticks/ship.",
    ),
    4: _stats(
        level=4,
        build_cost      = {R.MINERALS: 800, R.RARE_MATS: 15, R.ORGANICS: 80},
        production_rate = {SHIPS: 12},
        production_cost = {R.MINERALS: 28, R.ORGANICS: 14, POWER: 18},
        damage_rate     = 0.45,
        repair_cost     = {R.MINERALS: 60, R.RARE_MATS: 2, R.ORGANICS: 8},
        repair_rate     = 1.5,
        build_ticks     = 80,
        workforce       = {2: 2, 3: 3, 4: 2},
        notes           = "Orbital scaffold; ~8 ticks/ship.",
    ),
    5: _stats(
        level=5,
        build_cost      = {R.MINERALS: 1200, R.RARE_MATS: 40, R.ORGANICS: 100},
        production_rate = {SHIPS: 20},
        production_cost = {R.MINERALS: 40, R.ORGANICS: 20, POWER: 25},
        damage_rate     = 0.5,
        repair_cost     = {R.MINERALS: 90, R.RARE_MATS: 5, R.ORGANICS: 10},
        repair_rate     = 1.0,
        build_ticks     = 110,
        workforce       = {3: 2, 4: 3, 5: 2},
        notes           = "Full orbital construction ring; 5 ticks/ship.",
    ),
}


# ---------------------------------------------------------------------------
# RAILYARD  (Department of Commerce)
# ---------------------------------------------------------------------------
#
# Future use: moves resources across the planetary mesh.
# Currently modelled as increasing inter-colony transfer efficiency.
# Production is TRANSFER_CAPACITY (synthetic key TRANSFER=8).

TRANSFER = 8

RAILYARD_STATS: Dict[int, BuildingLevelStats] = {
    1: _stats(
        level=1,
        build_cost      = {R.MINERALS: 120},
        production_rate = {TRANSFER: 20},   # units/tick transferable
        production_cost = {POWER: 3},
        damage_rate     = 0.2,
        repair_cost     = {R.MINERALS: 12},
        repair_rate     = 2.5,
        build_ticks     = 20,
        workforce       = {1: 3, 2: 1},
        notes           = "Cargo rail loop; basic inter-zone logistics.",
    ),
    2: _stats(
        level=2,
        build_cost      = {R.MINERALS: 250},
        production_rate = {TRANSFER: 45},
        production_cost = {POWER: 6},
        damage_rate     = 0.2,
        repair_cost     = {R.MINERALS: 22},
        repair_rate     = 2.5,
        build_ticks     = 35,
        workforce       = {1: 2, 2: 2},
        notes           = "Extended rail network.",
    ),
    3: _stats(
        level=3,
        build_cost      = {R.MINERALS: 430, R.ENERGY: 25},
        production_rate = {TRANSFER: 90},
        production_cost = {POWER: 10},
        damage_rate     = 0.25,
        repair_cost     = {R.MINERALS: 35, R.ENERGY: 4},
        repair_rate     = 2.0,
        build_ticks     = 50,
        workforce       = {2: 3, 3: 1},
        notes           = "Maglev lines; energy-assisted.",
    ),
    4: _stats(
        level=4,
        build_cost      = {R.MINERALS: 700, R.RARE_MATS: 10},
        production_rate = {TRANSFER: 180},
        production_cost = {POWER: 16},
        damage_rate     = 0.3,
        repair_cost     = {R.MINERALS: 55, R.RARE_MATS: 1},
        repair_rate     = 1.5,
        build_ticks     = 70,
        workforce       = {2: 2, 3: 2, 4: 1},
        notes           = "Hyperloop segments; high throughput.",
    ),
    5: _stats(
        level=5,
        build_cost      = {R.MINERALS: 1100, R.RARE_MATS: 28},
        production_rate = {TRANSFER: 350},
        production_cost = {POWER: 24},
        damage_rate     = 0.35,
        repair_cost     = {R.MINERALS: 80, R.RARE_MATS: 3},
        repair_rate     = 1.0,
        build_ticks     = 95,
        workforce       = {3: 3, 4: 2},
        notes           = "Planetary logistics grid; near-instant transfer.",
    ),
}


# ---------------------------------------------------------------------------
# LAB  (Department of Intelligence)
# ---------------------------------------------------------------------------
#
# Produces RESEARCH points (synthetic key RESEARCH=9).
# Also enables worker upskilling: lower-level workers → higher-level workers
# at decreasing conversion rates per tier.
# Upskill rates are separate from production — stored as a list indexed by
# target_worker_level (1→2, 2→3, 3→4, 4→5).

RESEARCH = 9

@dataclass(frozen=True)
class LabLevelStats(BuildingLevelStats):
    """Extends BuildingLevelStats with worker upskill rates."""
    # upskill_rate[i] = workers converted per tick from level i+1 → i+2
    # index 0: lv1→lv2, index 1: lv2→lv3, index 2: lv3→lv4, index 3: lv4→lv5
    upskill_rates: Tuple[float, ...] = field(default=(0.0, 0.0, 0.0, 0.0))


def _lab(
    level, build_cost, production_rate, production_cost,
    damage_rate, repair_cost, repair_rate, build_ticks,
    workforce, upskill_rates, notes="",
):
    return LabLevelStats(
        level=level,
        build_cost=build_cost,
        production_rate=production_rate,
        production_cost=production_cost,
        damage_rate=damage_rate,
        repair_cost=repair_cost,
        repair_rate=repair_rate,
        build_ticks=build_ticks,
        workforce=workforce,
        upskill_rates=upskill_rates,
        notes=notes,
    )


LAB_STATS: Dict[int, LabLevelStats] = {
    1: _lab(
        level=1,
        build_cost      = {R.MINERALS: 80, R.ORGANICS: 10},
        production_rate = {RESEARCH: 5},
        production_cost = {POWER: 3},
        damage_rate     = 0.2,
        repair_cost     = {R.MINERALS: 8, R.ORGANICS: 1},
        repair_rate     = 2.5,
        build_ticks     = 15,
        workforce       = {2: 2, 3: 1},
        upskill_rates   = (0.5, 0.0, 0.0, 0.0),   # only lv1→lv2
        notes           = "Basic research post; lv1→lv2 worker training.",
    ),
    2: _lab(
        level=2,
        build_cost      = {R.MINERALS: 180, R.ORGANICS: 25},
        production_rate = {RESEARCH: 12},
        production_cost = {POWER: 6},
        damage_rate     = 0.25,
        repair_cost     = {R.MINERALS: 15, R.ORGANICS: 3},
        repair_rate     = 2.0,
        build_ticks     = 30,
        workforce       = {2: 2, 3: 2},
        upskill_rates   = (1.0, 0.3, 0.0, 0.0),   # lv1→2 faster; lv2→3 unlocked
        notes           = "Applied science wing; unlocks lv2→lv3 training.",
    ),
    3: _lab(
        level=3,
        build_cost      = {R.MINERALS: 340, R.ORGANICS: 40, R.ENERGY: 20},
        production_rate = {RESEARCH: 25},
        production_cost = {POWER: 10},
        damage_rate     = 0.3,
        repair_cost     = {R.MINERALS: 28, R.ORGANICS: 5, R.ENERGY: 3},
        repair_rate     = 1.5,
        build_ticks     = 50,
        workforce       = {3: 3, 4: 1},
        upskill_rates   = (1.5, 0.5, 0.15, 0.0),
        notes           = "Advanced research complex; lv3→lv4 training unlocked.",
    ),
    4: _lab(
        level=4,
        build_cost      = {R.MINERALS: 580, R.RARE_MATS: 8, R.ORGANICS: 60},
        production_rate = {RESEARCH: 50},
        production_cost = {POWER: 15},
        damage_rate     = 0.35,
        repair_cost     = {R.MINERALS: 45, R.RARE_MATS: 1, R.ORGANICS: 7},
        repair_rate     = 1.5,
        build_ticks     = 70,
        workforce       = {3: 2, 4: 2, 5: 1},
        upskill_rates   = (2.0, 0.8, 0.25, 0.05),
        notes           = "Institute; all tiers trainable; lv4→5 very slow.",
    ),
    5: _lab(
        level=5,
        build_cost      = {R.MINERALS: 900, R.RARE_MATS: 25, R.ORGANICS: 80},
        production_rate = {RESEARCH: 100},
        production_cost = {POWER: 22},
        damage_rate     = 0.4,
        repair_cost     = {R.MINERALS: 70, R.RARE_MATS: 3, R.ORGANICS: 10},
        repair_rate     = 1.0,
        build_ticks     = 95,
        workforce       = {4: 3, 5: 2},
        upskill_rates   = (2.5, 1.2, 0.4, 0.1),
        notes           = "Grand academy; maximum research output.",
    ),
}


# ---------------------------------------------------------------------------
# Master lookup table
# ---------------------------------------------------------------------------

BUILDING_STATS: Dict[BuildingType, Dict[int, BuildingLevelStats]] = {
    BuildingType.MINE:        MINE_STATS,
    BuildingType.POWER_PLANT: POWER_PLANT_STATS,
    BuildingType.FARM:        FARM_STATS,
    BuildingType.FACTORY:     FACTORY_STATS,
    BuildingType.FORT:        FORT_STATS,
    BuildingType.SHIPYARD:    SHIPYARD_STATS,
    BuildingType.RAILYARD:    RAILYARD_STATS,
    BuildingType.LAB:         LAB_STATS,
}

MAX_BUILDING_LEVEL = 5


# ---------------------------------------------------------------------------
# Convenience aggregation helpers
# ---------------------------------------------------------------------------

def colony_production_rates(
    building_counts: Dict[Tuple[BuildingType, int], int],
) -> Dict[int, float]:
    """
    Given a dict of  {(building_type, level): count}  return a summed
    production_rate dict across all active buildings.

    Example
    -------
    counts = {
        (BuildingType.MINE, 1): 3,
        (BuildingType.MINE, 3): 1,
        (BuildingType.FARM, 2): 2,
    }
    rates = colony_production_rates(counts)
    # rates[R.MINERALS] == 3*10 + 1*40 == 70
    # rates[R.ORGANICS] == 2*26 == 52
    """
    totals: Dict[int, float] = {}
    for (btype, level), count in building_counts.items():
        stats = BUILDING_STATS[btype][level]
        for resource, rate in stats.production_rate.items():
            totals[resource] = totals.get(resource, 0.0) + rate * count
    return totals


def colony_production_costs(
    building_counts: Dict[Tuple[BuildingType, int], int],
) -> Dict[int, float]:
    """Same as colony_production_rates but for costs (upkeep per tick)."""
    totals: Dict[int, float] = {}
    for (btype, level), count in building_counts.items():
        stats = BUILDING_STATS[btype][level]
        for resource, cost in stats.production_cost.items():
            totals[resource] = totals.get(resource, 0.0) + cost * count
    return totals


def net_rates(
    building_counts: Dict[Tuple[BuildingType, int], int],
) -> Dict[int, float]:
    """Production minus consumption — negative means deficit."""
    produced = colony_production_rates(building_counts)
    consumed = colony_production_costs(building_counts)
    all_keys = set(produced) | set(consumed)
    return {k: produced.get(k, 0.0) - consumed.get(k, 0.0) for k in all_keys}
