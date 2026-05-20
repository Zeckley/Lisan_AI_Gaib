---
tags:
  - Context
---
# Project Context — Spherical Strategy Sim

## High Level Goal
A space-based strategy simulation where factions compete for dominance across
a network of solar systems. Each solar system contains planets with resources.
Factions harvest resources, build structures, and use a small ML model to make
decisions. The long-term vision includes a planetary-scale map (spherical mesh)
but resource/faction equations are being developed first at the solar system level.

---

## Module 1 — Spherical Mesh (`world_map`)

### Key decisions
- Built on **PyVista `Icosphere`** with configurable subdivisions (`nu`)
- Icosphere is rotated so poles align to the Z axis using a rotation matrix
  derived from the antipodal vertex pair (found via `scipy.spatial.distance.cdist`)
- 2D unwrap uses standard spherical coordinates (θ, φ); seam-crossing triangles
  (where max(φ) - min(φ) > π) are culled from the flat mesh

### Terrain generation — wave collapse inspired
- Starts at `nsub=1` (coarsest), seeds land/water randomly at `land_fraction`
- Each subdivision step: split every triangle into 4 children (midpoints projected
  back onto unit sphere), children inherit parent type
- Border smoothing after each subdivision: BFS face adjacency, border cells
  probabilistically flipped toward water using `water_bias` weight
- Secondary terrain pass after full subdivision:
  - BFS shore-distance computed for all land cells
  - `beach_depth` hops from water → BEACH
  - `mountain_depth`+ hops from water → MOUNTAIN
  - Unreachable landlocked cells → MOUNTAIN

### Terrain types (`face_type` int array on mesh)
```
0 = WATER
1 = LAND
2 = BEACH
3 = MOUNTAIN
```

### Config keys
```python
config = {
    "nu":              5,      # subdivision count
    "polar_radius":    6378137.0,
    "equator_radius":  6356752.3,
    "water_bias":      0.7,    # 0.5=neutral, 1.0=all borders become water
    "land_fraction":   0.3,    # seed probability at nsub=1
    "beach_depth":     2,      # hops from shore -> beach
    "mountain_depth":  6,      # hops from shore -> mountain
}
```

### Plotting
- `plot_sphere(static=False)` — 3D icosphere colored by terrain
- `plot_map(static=False)`    — 2D equirectangular projection
- `static=True` uses PyVista `notebook=True` for inline PNG (no trame dependency)
- Colormap: `["dodgerblue", "forestgreen", "sandybrown", "white"]` → water/land/beach/mountain

### PyVista Jupyter note
Set `pv.set_jupyter_backend('static')` in its own cell before any PyVista code,
or install `trame-vtk`, `trame-vuetify`, and `nest_asyncio2` for the interactive widget.

---

## Module 2 — Solar System & Resources

### Design intent
Solar systems act as harvestable nodes in 3D space, decoupled from the mesh.
This lets resource and faction equations be developed and tuned independently
before re-integrating with the planetary map.

### Class hierarchy
```
SolarSystem
  └── Planet (1..n)
        └── resources: dict[ResourceType, float]
```

### `ResourceType` (IntEnum)
```python
MINERALS  = 0
ENERGY    = 1
ORGANICS  = 2
RARE_MATS = 3
```
Using `IntEnum` so values are named constants, iterable, numpy/JSON compatible,
and validated on construction. Easy to extend — add a line, loops and dicts adapt.

### `sample_distribution` — core flexible sampler
Supports: `"uniform"`, `"normal"`, `"lognormal"`, `"exponential"`, `"beta"`
Key params: `scale` (master volume), `clip` (hard bounds), `size` (batch draw)
Beta distribution auto-converts `mean`+`std` to alpha/beta params.

`DISTRIBUTION` is a `Literal` type alias for the five valid distribution strings —
use it to type-hint any field or parameter that accepts a distribution name.

`clip` accepts `(float, None)` for a one-sided lower clamp (e.g. `(0.0, None)`),
`(float, float)` for a two-sided clamp, or `None` for no clamping.
`None` upper bound is resolved to `np.inf` internally.

### `ResourceConfig` per resource type
```python
scale             # master output multiplier
size_exponent     # output *= planet_size ** size_exponent
quality_exponent  # output *= (quality/5) ** quality_exponent
```
Higher `quality_exponent` = steeper penalty for low-quality planets.
`lognormal` used for minerals (right-skewed hoards), `exponential` for rare mats,
`beta` for organics (bounded, flexible shape).

Dataclass defaults: `dist="lognormal"`, `mean=1.0`, `std=0.5`, `scale=100.0`,
`clip=(0.0, None)`, `size_exponent=1.2`, `quality_exponent=1.5`.

`DEFAULT_RESOURCE_CONFIGS` is a `dict[ResourceType, ResourceConfig]` with tuned
values for all four types — intended as the baseline that buildings and faction
effects will override per planet via `Planet.resource_configs`.

### Planet quality
Float in [0, 5]. Quality=0 always produces zero resources regardless of distribution.
Normalised to [0,1] internally before applying `quality_exponent`.

---

## Module 3 — Buildings (`buildings.py`)

### Design intent
`buildings.py` is the single source of truth for all static building data.
Nothing here carries runtime state — health, tick counters, etc. live in
`Building` instances in `colony.py`. All numeric fields use `ResourceType`
integer keys so colony-level aggregation is a simple loop.

### Enumerations
```python
BuildingType:   MINE=0, POWER_PLANT=1, FARM=2, FACTORY=3,
                FORT=4, SHIPYARD=5, RAILYARD=6, LAB=7

DepartmentType: RESOURCES=0, ENERGY=1, AGRICULTURE=2, MANUFACTURING=3,
                DEFENSE=4, TRANSPORTATION=5, COMMERCE=6, INTELLIGENCE=7

BuildingState:  CONSTRUCTING=0, ACTIVE=1, DAMAGED=2, REPAIRING=3,
                SURGING=4, DESTROYED=5, INACTIVE=6
```
`BUILDING_DEPARTMENT` maps each `BuildingType` to its owning `DepartmentType`.

### Synthetic resource keys (not in ResourceType)
```python
POWER    = 4   # produced by Power Plants, consumed by most buildings
DEFENSE  = 6   # produced by Forts; colony defense rating
SHIPS    = 7   # produced by Shipyards; 100 progress = 1 colony ship
TRANSFER = 8   # produced by Railyards; inter-colony throughput
RESEARCH = 9   # produced by Labs; also drives worker upskilling
```

### `BuildingLevelStats` (frozen dataclass)
All per-level stats are stored here. Key fields:
```python
build_cost      : Dict[int, float]   # one-time construction cost
production_rate : Dict[int, float]   # resources produced per tick (ACTIVE/SURGING)
production_cost : Dict[int, float]   # resources consumed per tick (upkeep)
repair_cost     : Dict[int, float]   # cost per tick while REPAIRING
damage_rate     : float              # health % lost per tick (doubled while SURGING)
repair_rate     : float              # health % gained per tick while REPAIRING
build_ticks     : int                # ticks to complete construction / upgrade
workforce       : Dict[int, int]     # {worker_level: count_required}
```

### `LabLevelStats` (extends `BuildingLevelStats`)
Adds `upskill_rates: Tuple[float, ...]` — workers converted per tick at each
training tier (index 0: L1→L2, 1: L2→L3, 2: L3→L4, 3: L4→L5).

### `BUILDING_STATS` master lookup
```python
BUILDING_STATS: Dict[BuildingType, Dict[int, BuildingLevelStats]]
```
All eight building types, five levels each. Levels follow consistent scaling:
- Build cost roughly doubles each tier
- Production roughly doubles each tier
- Damage rate increases modestly; repair rate decreases (harder to fix high-tech)
- Higher levels require higher-level workers in their `workforce` dict

### Notable building behaviours
- **Mine lv4+**: unlocks `RARE_MATS` harvest in addition to `MINERALS`
- **Power Plant**: lv1-2 consume `ENERGY`, lv3-4 consume `ORGANICS` (biofuel),
  lv5 has zero running cost (zero-point)
- **Factory lv4-5**: "Recycler mode" — accepts `ORGANICS` as feedstock alongside
  `MINERALS`; also produces a small amount of `RARE_MATS` at lv5
- **Farm lv5**: requires `POWER` for grow-lights; huge yield bump
- **Lab**: only building with `upskill_rates`; enables workforce advancement

### Aggregation helpers
```python
colony_production_rates(building_counts)  # sum production_rate across active buildings
colony_production_costs(building_counts)  # sum production_cost across active buildings
net_rates(building_counts)                # produced − consumed per resource key
```
`building_counts` is `Dict[Tuple[BuildingType, int], int]` — (type, level) → count.

---

## Module 4 — Factions & Colonies (`colony.py`)

### Class hierarchy
```
Faction
  └── Colony (1..n, keyed by colony_id)
        ├── Building (1..n)  — runtime instances
        └── Worker  (1..n)
```

### Worker system
```python
WorkerLevel: L1=1 … L5=5
```
- `Worker` dataclass: `level`, `assigned_building_id` (None = unassigned pool)
- `POP_PER_WORKER = 10` — population units consumed per worker recruited
- `recruit_workers(count)` / `release_workers(count)` manage the pool
- `Lab.upskill_rates` drives level promotions (handled by future lab tick logic)

### `Building` (runtime instance)
Carries only mutable state; static stats always fetched live from `BUILDING_STATS`:
```python
id, building_type, level, state: BuildingState, health: float [0,1],
ticks_remaining: int, planet_index: Optional[int]
```
Key methods: `apply_damage()`, `apply_repair()`, `advance_construction()`,
`production_this_tick()`, `upkeep_this_tick()`, `repair_upkeep_this_tick()`

`surge_multiplier = 1.5` when `SURGING`; damage rate also doubles.

### Flag system (two tiers)
**Critical** — existential; always surface to Faction; block harmful directives:
```python
FOOD_SHORTAGE       # N consecutive starving ticks  (N = FOOD_SHORTAGE_TICKS = 3)
POWER_DEFICIT       # net POWER < 0
POPULATION_COLLAPSE # population < 25% of starting_pop
```
**Strategic** — multi-tick concerns; suppressible via `Directive.override_flags`:
```python
DEFENSE_NEEDED      # net DEFENSE < 50
WORKER_SHORTAGE     # unassigned / required workers < 0.5
RESOURCE_LOW        # any resource net rate negative for 5+ consecutive ticks
EXPORT_STRAINED     # local stockpile headroom after tax < 10%
CONSTRUCTION_BLOCKED # can't afford cheapest building × 1.5 buffer
```

### `Directive` system
`DirectiveType: HARVEST=0, DEFEND=1, EXPAND=2, EXPORT=3, IDLE=4`

Key fields:
```python
tax_rate      : float   # fraction of production → faction_stockpile (can exceed 1.0)
urgency       : float   # [0,1] aggressiveness of directive execution
target_resource: int    # optional ResourceType focus (HARVEST)
target_building: BuildingType  # optional building focus (DEFEND/EXPAND)
override_flags : Set[StrategicFlag]  # strategic flags the faction suppresses
```
`tax_rate > 1.0` draws from local stockpile reserves in addition to new production.

### `Colony` — rule-based decision engine
`Colony.tick()` order of operations:
1. Reset per-tick ledgers (`last_produced`, `last_consumed`, `last_events`)
2. Advance constructions (`advance_construction()` on each building)
3. Collect resources + pay building upkeep (tax applied here → `faction_stockpile`)
4. Pay repair upkeep + advance repairs
5. Apply wear-and-tear damage; mark DESTROYED at health == 0
6. Feed population (`ORGANICS_PER_POP = 0.05` per population unit per tick)
7. Evaluate flags
8. Execute directive (rule-based agent)

Rule-based priority inside `execute_directive()`:
```
0. Critical flag response  — survival first; cannot be overridden
1. Repair damaged buildings (health < REPAIR_PRIORITY_FRAC = 0.60)
2. Directive execution      — HARVEST / DEFEND / EXPAND / EXPORT sub-rules
3. IDLE                     — de-surge, upgrade if surplus allows
```

`Colony` also maintains:
- `stockpile: Dict[int, float]` — local resource pool
- `faction_stockpile: Dict[int, float]` — tax accumulator, drained by `Faction.collect_taxes()`
- `_resource_low_ticks: Dict[int, int]` — hysteresis counter for `RESOURCE_LOW` flag

### `Faction` — strategic agent
```python
Faction.tick() order:
  1. Tick all colonies
  2. collect_taxes()  — drain each colony's faction_stockpile into treasury
  3. _faction_strategy()  — stub rule: send ORGANICS aid to FOOD_SHORTAGE colonies
```

Key methods:
```python
issue_directive(colony_id, directive_type, ...)
  # Guards: clamps tax_rate to 0 for EXPORT directives to critically stressed colonies
collect_taxes()
transfer_to_colony(colony_id, resources)
transfer_between_colonies(from_id, to_id, resources)
critical_colonies()                        # colonies with any active critical flag
colonies_with_flag(flag: StrategicFlag)
```

`_faction_strategy()` is the ML model entry point — currently a simple rule;
replace with `get_state_vector()` → model inference → `issue_directive()` calls.

---

## Module 5 — Snapshot & Plotting (`snapshot.py`)

### Design intent
Provides a serialisation layer (plain dicts) and a matplotlib dashboard so any
run can be inspected tick-by-tick without coupling the simulation to plotting code.
Works on both `Colony` and `Faction` instances via duck-typing.

### `take_snapshot(target) -> dict`
Captures the full observable state of a Colony or Faction at the current tick.
The returned dict is pickle- and JSON-safe.

```python
# Schema
{
  "tick"       : int,
  "label"      : str,            # colony/faction name

  # Resources
  "stockpile"  : {resource_name: float},  # local stockpile (colony) or treasury (faction)
  "net_rate"   : {resource_name: float},  # production − consumption this tick

  # Workers
  "workers"    : {level_int: {"assigned": int, "unassigned": int}},
  "population" : float,

  # Buildings — building_name → level → state bucket → count
  "buildings"  : {
      building_name: {level_int: {"producing": int, "constructing": int, "idle": int}}
  },

  # Flags & Directive
  "critical_flags"  : [str, ...],
  "strategic_flags" : [str, ...],
  "directive"       : str,        # DirectiveType.name; "MULTI" for mixed-directive Faction
  "tax_rate"        : float,      # mean across colonies for Faction snapshots
  "urgency"         : float,
}
```

For **Faction** snapshots: stockpile = treasury, workers/buildings are sums across
all colonies, flags are the union of all colony flags, directive is "MULTI" if
colonies currently hold different directives.

### `plot_history(snapshots, title="", save_path="", show=True) -> Figure`
Renders a 4-panel dark-theme matplotlib dashboard from a list of snapshot dicts.

Panel layout:
```
[1] RESOURCES   — one subplot per resource; stockpile filled-area on left axis,
                  Δ/tick dashed line on right axis
[2] WORKERS     — stacked bar per tick, each WorkerLevel a distinct blue shade;
                  hatched overlay = unassigned; population dotted on right axis
[3] BUILDINGS   — one subplot per building type; stacked area by level
                  (darker = lower level) = producing count; dashed yellow =
                  under construction; dotted = idle/damaged
[4] FLAGS &     — horizontal timeline bar per flag (color-coded by severity);
    DIRECTIVE     directive background shading + change annotations; tax rate
                  and urgency as lines on right axis
```

### Typical usage
```python
from snapshot import take_snapshot, plot_history

history = []
for _ in range(100):
    faction.tick()
    history.append(take_snapshot(home_colony))   # or take_snapshot(faction)

plot_history(history, title="Arrakeen — 100 tick run", save_path="run.png")
```

Snapshots can be stacked across runs, serialised, or filtered before plotting —
the dict format is intentionally stable and independent of the simulation classes.

---

## Python patterns used (notes for new conversations)
- `@dataclass` + `__post_init__`: auto-generates `__init__`, `__repr__`, `__eq__`;
  `__post_init__` runs after field assignment for derived attributes.
  Use `field(init=False)` for computed fields that shouldn't be constructor args.
- `IntEnum`: named integer constants that are iterable, indexable in numpy arrays,
  JSON-serialisable as ints, and validated on construction. Preferred over bare ints
  or string constants for any fixed categorical set.
- `np.random.default_rng(seed)` passed through as `rng` parameter for reproducibility
  without global state.
- Synthetic resource keys (`POWER=4`, `DEFENSE=6`, etc.) extend the resource space
  beyond `ResourceType` without touching the enum. They appear in `production_rate`
  and `production_cost` dicts but are never stored in faction stockpiles unless
  explicitly added to `ResourceType`.
- Duck-typing over `isinstance` checks: `take_snapshot()` distinguishes Colony from
  Faction via `hasattr(target, "_colonies")` — keeps the snapshot module decoupled
  from the class hierarchy.

---

## File structure
```
project_root/
│
├── PROJECT_CONTEXT.md        ← this file
│
├── world/
│   ├── world_map.py          ← world_map class, rotation_matrix_to_z
│   └── terrain.py            ← TERRAIN_* constants, terrain enums
│
├── simulation/
│   ├── resources.py          ← ResourceType, ResourceConfig, sample_distribution
│   ├── solar_system.py       ← SolarSystem, Planet
│   ├── buildings.py          ← BuildingType/State/Department, BuildingLevelStats,
│   │                            LabLevelStats, BUILDING_STATS, aggregation helpers
│   ├── colony.py             ← WorkerLevel, Worker, Building (runtime), CriticalFlag,
│   │                            StrategicFlag, DirectiveType, Directive, Colony, Faction
│   └── snapshot.py           ← take_snapshot(), plot_history()
│
└── notebooks/
    └── dev.ipynb             ← scratch/testing notebook
```

### Module import order (simulation/)
```
resources.py          ← leaf; no internal imports
solar_system.py       ← imports resources.py only
buildings.py          ← imports resources.py only (re-exports ResourceType)
colony.py             ← imports buildings.py (and transitively resources.py)
snapshot.py           ← imports colony.py and buildings.py; also matplotlib/numpy
```
Keep this direction strictly — no upward imports — to avoid circular dependencies.
