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

### Faction starting conditions
- Each faction starts in a single solar system on one randomly chosen planet
- Default number of planets per starting system: **6** (configurable parameter)
- Adjusting planet count is the primary difficulty knob for ML training:
  more planets = more resources available = easier early game

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

### `ResourceConfig` per resource type
```python
scale             # master output multiplier
size_exponent     # output *= planet_size ** size_exponent
quality_exponent  # output *= (quality/5) ** quality_exponent
```
Higher `quality_exponent` = steeper penalty for low-quality planets.
`lognormal` used for minerals (right-skewed hoards), `exponential` for rare mats,
`beta` for organics (bounded, flexible shape).

### Planet quality
Float in [0, 5]. Quality=0 always produces zero resources regardless of distribution.
Normalised to [0,1] internally before applying `quality_exponent`.

---

## Module 3 — Factions

### Class hierarchy
```
Faction
  └── Colony (1..n, keyed by system_id)
        └── Building (1..n)
              └── stats looked up from BUILDING_STATS[(BuildingType, level)]
```

### `FactionType` (IntEnum)
```python
NEUTRAL=0, AGGRESSIVE=1, ECONOMIC=2, SCIENTIFIC=3
```

### Key design notes
- `Faction.update()` order: collect → pay upkeep → AI decision → update colonies
- `can_afford()` and `spend()` are separate so costs can be validated before deducting
- `get_state_vector() -> np.ndarray` is the ML model interface — flatten stockpile,
  colony count, building counts etc. into a numeric vector
- `decide_action()` is the entry point for rule-based or ML-driven logic
- `Colony` holds a forward reference to `Faction` (`"Faction"` string) to avoid
  circular imports if modules are split into separate files

---

## Module 4 — Buildings (`buildings.py`)

### Architectural principle — static vs. dynamic split
`buildings.py` contains **only static balance data**. No runtime state lives here.
`Building` instances in `faction.py` hold `(BuildingType, level)` and look up
stats from `BUILDING_STATS` on demand. Re-balancing values never touches faction logic.

### Departments and building types
```
DepartmentType        BuildingType
──────────────────    ────────────
RESOURCES       (0)   MINE        (0)
ENERGY          (1)   POWER_PLANT (1)
AGRICULTURE     (2)   FARM        (2)
MANUFACTURING   (3)   FACTORY     (3)  ← becomes Recycler at lv4-5
DEFENSE         (4)   FORT        (4)
TRANSPORTATION  (5)   SHIPYARD    (5)
COMMERCE        (6)   RAILYARD    (6)
INTELLIGENCE    (7)   LAB         (7)
```

### `BuildingState` (IntEnum)
```python
CONSTRUCTING = 0   # being built over build_ticks ticks
ACTIVE       = 1   # producing normally
DAMAGED      = 2   # health < 50% — production halted, damage continues
REPAIRING    = 3   # consuming repair_cost/tick, gaining repair_rate health/tick
SURGING      = 4   # 1.5× production_rate, 2× damage_rate
DESTROYED    = 5   # health == 0; drops scrap = resources invested + exact rare mats used
INACTIVE     = 6   # toggled off by agent or workforce shortage; no production, no damage
```

### Building health rules
- Health ranges from 0% to 100%
- Below 50%: building enters DAMAGED state — production stops, damage accumulation stops
- At 0%: building enters DESTROYED — cannot be rebuilt; drops scrap
- INACTIVE buildings take no damage and produce nothing
- SURGING buildings take 2× damage_rate per tick

### Upgrade system
- 5 levels per building; upgrading costs resources and takes `build_ticks` ticks
- Higher levels produce more but also consume more resources, power, and require
  higher-level workers
- Some levels unlock new outputs (Mine lv4 → RARE_MATS; Farm lv5 → needs POWER;
  Factory lv4 → Recycler mode with ORGANICS feedstock; Power Plant lv5 → free fuel)

### `BuildingLevelStats` (frozen dataclass)
All numeric fields use resource integer keys. The same dict structure is used
throughout so aggregation loops need no special-casing.
```python
level:           int
build_cost:      dict[int, float]   # one-time construction / upgrade cost
production_rate: dict[int, float]   # resources produced per tick (ACTIVE)
production_cost: dict[int, float]   # resources consumed per tick (ACTIVE)
damage_rate:     float              # health % lost per tick (ACTIVE or SURGING×2)
repair_cost:     dict[int, float]   # resources consumed per tick (REPAIRING)
repair_rate:     float              # health % gained per tick (REPAIRING)
build_ticks:     int                # ticks to finish construction / upgrade
workforce:       dict[int, int]     # {worker_level: count_required}
notes:           str
```
`LabLevelStats` extends this with `upskill_rates: tuple[float, ...]` — one rate
per training tier (index 0 = lv1→lv2, index 3 = lv4→lv5). Rates decrease at
higher tiers.

### Synthetic resource keys (not in ResourceType)
These flow through the same dicts as real resources so loops need no branching,
but they are not stockpiled unless explicitly added to the faction stockpile.
```python
POWER    = 4   # produced by POWER_PLANT; consumed by FARM lv5, FORT, SHIPYARD, etc.
DEFENSE  = 6   # produced by FORT; colony sums this and compares vs attacker rating
SHIPS    = 7   # ship-progress/tick from SHIPYARD; 100 units = 1 colony ship spawned
TRANSFER = 8   # logistics capacity/tick from RAILYARD (future mesh integration)
RESEARCH = 9   # research points/tick from LAB; used to unlock building levels
```

### Building-specific design notes

**MINE**
- Produces MINERALS at all levels
- Unlocks RARE_MATS as a second output at lv4 and lv5
- No production cost — extraction is purely mechanical

**POWER_PLANT**
- Lv1-2: consumes raw planetary ENERGY (geothermal / solar tap)
- Lv3-4: consumes ORGANICS (biofuel / fusion feedstock)
- Lv5: no fuel cost; requires RARE_MATS to build/upgrade (zero-point / antimatter)

**FARM**
- Produces ORGANICS at all levels
- Lv1-4: no inputs required
- Lv5: consumes a small amount of POWER; provides a large yield increase

**FACTORY / RECYCLER**
- Lv1-3 (Factory): converts raw MINERALS feedstock into processed build stock
- Lv4-5 (Recycler mode): ORGANICS feedstock unlocked as a second input;
  colony logic chooses which feedstock(s) to draw from
- Net output = production_rate − production_cost (both are MINERALS at lv1-3)

**FORT**
- Produces DEFENSE score (synthetic key) rather than a stockpileable resource
- Consumes a small amount of POWER and ORGANICS per tick (garrison supply)
- Colony sums total DEFENSE across all forts and compares against attack ratings

**SHIPYARD**
- Produces SHIPS (ship-progress/tick); 100 accumulated units spawn one colony ship
- Colony ships carry the first workers and materials to a new planet
- Every planet in a system must contribute a ship to colonise a new system

**RAILYARD**
- Produces TRANSFER capacity (future use: moving resources across planetary mesh)
- Currently models inter-colony logistics efficiency
- Consumes only POWER

**LAB**
- Produces RESEARCH points used to unlock higher building levels faction-wide
- Also enables worker upskilling: lower-level workers → higher-level workers at
  decreasing rates per tier (stored in `upskill_rates` tuple on `LabLevelStats`)
- Worker levels range from 1 to 5; higher buildings require higher-level workers

### Master lookup and aggregation
```python
# Single source of truth
BUILDING_STATS: dict[(BuildingType, level)] → BuildingLevelStats

# Colony-level helpers (building_counts = {(BuildingType, level): count})
colony_production_rates(building_counts)  → dict[int, float]
colony_production_costs(building_counts)  → dict[int, float]
net_rates(building_counts)                → dict[int, float]  # negative = deficit
```
Aggregation is a single loop: `sum(count * stats.production_rate[r] for each building)`.

---

## Python patterns used (notes for new conversations)
- `@dataclass` + `__post_init__`: auto-generates `__init__`, `__repr__`, `__eq__`;
  `__post_init__` runs after field assignment for derived attributes.
  Use `field(init=False)` for computed fields that shouldn't be constructor args.
- `@dataclass(frozen=True)`: immutable dataclass; used for all static stats so
  balance data cannot be accidentally mutated at runtime.
- `IntEnum`: named integer constants that are iterable, indexable in numpy arrays,
  JSON-serialisable as ints, and validated on construction. Preferred over bare ints
  or string constants for any fixed categorical set.
- `np.random.default_rng(seed)` passed through as `rng` parameter for reproducibility
  without global state.

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
│   ├── solar_system.py       ← SolarSystem, Planet
│   ├── resources.py          ← ResourceType, ResourceConfig, sample_distribution
│   ├── buildings.py          ← BuildingType, DepartmentType, BuildingState,
│   │                            BuildingLevelStats, LabLevelStats, BUILDING_STATS,
│   │                            synthetic resource keys, aggregation helpers
│   └── faction.py            ← Faction, Colony, Building (runtime state only);
│                                Building instances reference buildings.py for stats
│
└── notebooks/
    └── dev.ipynb             ← scratch/testing notebook
```
