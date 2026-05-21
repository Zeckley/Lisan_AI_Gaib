# Project State — May 20, 2026

> **Lisan AI Gaib** — A space-based strategy simulation where factions compete for dominance across a network of solar systems. Factions harvest resources, build structures, and use a rule-based decision engine (with future ML model support) to make strategic choices.

---

## 1. Project Overview

### Goal
Build a simulation sandbox for faction-level strategy using discrete "ticks" as the time step. The long-term vision is a planetary-scale spherical mesh where ground-level tactics play out, but the current phase focuses on the solar-system abstraction layer — resource equations, building economies, colony management, and faction-level AI decision-making.

### Theme
Space colonization / 4X-lite. Factions (e.g. "House Atreides") own multiple colonies spread across solar systems. Each colony manages a local population, stockpile, and building portfolio. The faction issues directives (HARVEST, BUILD, UPGRADE, EXPAND, EXPORT) that shape colony behavior.

### End Goals (from docs)
- **ML-driven factions**: The `Faction._faction_strategy()` method is a stub — it currently sends basic ORGANICS aid to starving colonies. The `get_state_vector()` paradigm is set up for model inference.
- **Spherical world map**: The `world/` directory contains an early prototype (`MapMeshTesting.ipynb`) using icosphere subdivision with terrain generation. Not yet connected to the simulation.
- **Rebellion / revolt system**: `directives.py` has a full revolt-cascade implementation ready (happiness tracking, trigger thresholds, rebel faction creation, proximity effects).
- **Inter-colony trade**: `TRANSFER` resource key (8) and Railyard/Shipyard buildings exist in `BUILDING_STATS` but transfer logic is not wired into the colony tick loop.
- **Future modules** discussed in docs: Finance, Entertainment, Intelligence, Political systems.

---

## 2. Current Structure

### File Tree
```
project_root/
├── PROJECT_CONTEXT.md              ← overview context
├── README.md                       ← one-liner
│
├── world/
│   └── MapMeshTesting.ipynb        ← standalone icosphere prototype (not integrated)
│
├── simulation/
│   ├── resources.py                ← ResourceType enum, ResourceConfig, sample_distribution
│   ├── solar_system.py             ← SolarSystem, Planet dataclasses
│   ├── buildings.py                ← BuildingType/State/Department enums, BUILDING_STATS (all 8 types, 5 levels each), Worker, Building (runtime), aggregation helpers
│   ├── colony.py                   ← Colony (local agent), Faction (strategic agent), Directive/DirectiveType, Flags (Critical + Strategic)
│   ├── directives.py               ← DirectiveManager (per-colony priority system), DirectiveIssuer (faction AI interface), revolt/cascade system
│   └── snapshot.py                 ← take_snapshot(), plot_history() — matplotlib dashboard
│
├── docs/
│   ├── Colony - *.md               ← Colony subsystem docs (Tick Loop, Flag System, Decision Making, etc.)
│   └── WorldSimNotes/              ← Obsidian vault with full codebase documentation
│
└── notebooks/
    └── resources_dev.ipynb         ← scratch testing
```

### Main Components & Interactions

1. **Resources** (`resources.py`) — Leaf module. Defines `ResourceType` (MINERALS=0, WEALTH=1, ORGANICS=2, RARE_MATS=3) and the `sample_distribution()` function supporting 5 distributions. `ResourceConfig` dataclass parameterizes per-type generation.

2. **Solar System** (`solar_system.py`) — Imports from `resources`. `Planet` auto-generates resources on construction. `SolarSystem` is a collection of planets at a 3D position. Not yet wired into the simulation (colonies reference `system_id` but don't query planets for resources).

3. **Buildings** (`buildings.py`) — Single source of truth for all static building data. Defines `BuildingType` (MINE, POWER_PLANT, FARM, FACTORY, FORT, SHIPYARD, RAILYARD, LAB), `BuildingState` (CONSTRUCTING → ACTIVE → DAMAGED → REPAIRING → SURGING → DESTROYED → INACTIVE), and `BUILDING_STATS` — a nested dict keyed by `(BuildingType, level)` → `BuildingLevelStats`. Also defines runtime `Building` and `Worker` dataclasses.

4. **Colony & Faction** (`colony.py`) — The core simulation. `Colony` manages a local stockpile, buildings, workers, flags, and a directive. Its `tick()` method runs the full per-tick pipeline: auto-recruit → advance construction → collect resources → pay upkeep → apply damage → feed population → evaluate flags → execute directive. `Faction` owns multiple colonies, issues directives, collects taxes into a treasury, and runs a stub strategic agent.

5. **Directives** (`directives.py`) — An alternative/overhauled directive system with `DirectiveManager` (priority-based decision per colony) and `DirectiveIssuer` (faction-level API). Contains revolt/cascade logic (happiness tracking, rebel faction creation). This appears to be NEWER code that isn't fully integrated with `colony.py`'s legacy directive system.

6. **Snapshot** (`snapshot.py`) — Duck-typed serialization to plain dicts + a 4-panel matplotlib dashboard. Fully functional with dark theme. Used for generating the `run_*.png` images in the project root.

### Future-Proofing
- All resource keys are `int` — synthetic resources (POWER=4, DEFENSE=6, SHIPS=7) extend beyond `ResourceType` without touching the enum.
- `BuildingLevelStats` is a frozen dataclass; new building types just add a new `*_STATS` dict and a `BUILDING_STATS` entry.
- `Faction.get_state_vector()` is the designated ML model interface.
- `directives.py` has a full `DirectiveIssuer` that could replace the `Colony.directive` fields.
- Module import order is strictly linear: `resources → solar_system/buildings → colony → snapshot`.

---

## 3. Current Issues

### Code Duplication — `colony.py` vs `buildings.py`
`Worker`, `WorkerLevel`, and `Building` are defined in **both** `buildings.py` and `colony.py`. The `colony.py` versions are redundant and will shadow the `buildings.py` imports at runtime (they're imported first locally in `colony.py`). This is a significant maintenance hazard — changes to one won't propagate to the other.

### Duplicate Directive Systems
There are TWO directive systems:
- **Legacy** (in `colony.py`): `DirectiveType` enum + `Directive` dataclass + `execute_directive()` with sub-rules (`_rule_harvest`, `_rule_build`, etc.)
- **Overhaul** (in `directives.py`): `Directive` dataclass (different schema), `DirectiveManager`, `DirectiveIssuer`, revolt system

These are not integrated — the `directives.py` system defines its own `Colony` references but `colony.py`'s `Colony` class doesn't use `DirectiveManager`. The import in `directives.py` (`from colony import Faction, FactionType`) is circular-adjacent and fragile.

### Stale DirectiveType Enum
In `colony.py`, the `DirectiveType` enum has HARVEST, BUILD, UPGRADE, EXPORT, EXPAND (5 values). The old docs reference DEFEND and IDLE. The snapshot module still references `"DEFEND"` and `"IDLE"` in `_DIRECTIVE_COLORS` — these will never be used.

### World Map Not Integrated
The `world/` directory contains a working spherical mesh prototype with terrain generation, but it's a standalone notebook. The simulation (`simulation/`) has no reference to it. The `world_map.py`/`terrain.py` files mentioned in `PROJECT_CONTEXT.md` don't actually exist yet — only the notebook prototype exists.

### Solar Systems Decoupled
`Colony` stores a `system_id` but never queries `Planet.resources`. Planet resource generation is stateless and not connected to colony production. Buildings produce resources from nothing (they don't draw from planetary deposits).

### Building Repair Logic Redundancy
The `buildings.py` `Building.apply_repair()` method has a print statement. Repair logic is also handled in `colony.py`'s `tick()` (lines 1486-1494) with overlapping state transitions. The condition `b.state == BuildingState.ACTIVE and b.health < REPAIR_THRESHOLD*b.max_health` on line 1487 sets state to REPAIRING *before* the `DAMAGED` threshold check in `apply_damage()`.

### `_rule_harvest` Limited
HARVEST directive just surges all buildings — it doesn't have logic to build more production if stockpiles are low. Other directive sub-rules (EXPAND, EXPORT) are stubs that log messages but don't execute meaningful actions.

### No Persistence Layer
The snapshot system can serialize to dicts, but there's no save/load mechanism. Every simulation run starts fresh.

---

## 4. Future Plans

### Last Things Worked On
From `git log` and `TaskList.md`:
- **Directive overhaul**: Created `directives.py` with the new `DirectiveManager`/`DirectiveIssuer` system and revolt mechanics. This was the most recent major push.
- **Decision validation system**: The 4-check validation (`_validate_build`, `_validate_upgrade`) was completed to prevent colonies from making suicidal decisions.
- **Worker upskilling pipeline**: Labs can train workers through levels 1-5, with progress tracked in `_upskill_progress`. Auto-recruitment in `tick()` builds labs when higher-level workers are needed.
- **Bug observed at 400+ ticks**: Colonies enter a loop of building farms → power plants → forts, cycling endlessly. Likely related to the basic survival loop and buildings falling into disrepair simultaneously.

### Logical Places to Resume

1. **Fix the 400-tick death spiral** — This is the most impactful next step. The colony gets stuck in a survival loop that oscillates between FOOD_SHORTAGE, POWER_DEFICIT, and DEFENSE_NEEDED. Likely root cause: buildings degrade simultaneously, triggering cascading shortages that the colony can't escape. The `_basic_survival_loop()` and repair/damage timing need investigation.

2. **Integrate `directives.py` into `colony.py`** — The new directive system has the revolt mechanics and priority-based decision logic, but it's not wired into `Colony.tick()`. Decide which system to keep, remove the duplicate `Worker`/`WorkerLevel`/`Building` definitions, and make `Colony` use `DirectiveManager`.

3. **Colony expansion (EXPAND directive)** — The `_rule_expand()` method currently checks for a shipyard and logs messages but doesn't actually:
   - Accumulate ship progress (SHIPS resource)
   - Spawn a colony ship at 100 progress
   - Create a new `Colony` on a new `SolarSystem`
   - Seed the new colony with starting resources
   
   This requires a mechanism to generate/select new solar systems and assign them to the faction's expanding colony.

4. **Connect SolarSystem resources to Colony production** — Currently buildings produce resources from nothing. The intended design is:
   - `Colony` is assigned to a `SolarSystem`
   - Each `Building` sits on a `Planet` (via `planet_index`)
   - Building production rates are modified by planet resource availability/quality

5. **New Colony / Outpost Building** — The `TaskList.md` mentions creating an "outpost or capital building" that passively generates starter resources so a new colony can bootstrap itself without needing a full building portfolio.

6. **Rebellion system testing** — The revolt cascade in `directives.py` is fully coded but never exercised since `faction_happiness` tracking isn't integrated with the legacy colony tick.

7. **ML agent stub** — Replace `_faction_strategy()` with an actual model using `get_state_vector()` and `issue_directive()`.
