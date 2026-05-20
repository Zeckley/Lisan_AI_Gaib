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

## Module 3 — Factions (skeleton, to be filled)

### Class hierarchy
```
Faction
  └── Colony (1..n, keyed by system_id)
        └── Building (1..n)
```

### `FactionType` (IntEnum)
```python
NEUTRAL=0, AGGRESSIVE=1, ECONOMIC=2, SCIENTIFIC=3
```

### `BuildingType` (IntEnum)
```python
MINE=0, POWER_PLANT=1, FARM=2, LAB=3, SHIPYARD=4, DEFENSE=5
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

## Python patterns used (notes for new conversations)
- `@dataclass` + `__post_init__`: auto-generates `__init__`, `__repr__`, `__eq__`;
  `__post_init__` runs after field assignment for derived attributes.
  Use `field(init=False)` for computed fields that shouldn't be constructor args.
- `IntEnum`: named integer constants that are iterable, indexable in numpy arrays,
  JSON-serialisable as ints, and validated on construction. Preferred over bare ints
  or string constants for any fixed categorical set.
- `np.random.default_rng(seed)` passed through as `rng` parameter for reproducibility
  without global state.

---

## File structure (suggested)
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
│   └── faction.py            ← Faction, Colony, Building, enums
│
└── notebooks/
    └── dev.ipynb             ← scratch/testing notebook
```
