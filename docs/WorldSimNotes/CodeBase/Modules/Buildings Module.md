---
tags:
  - Module
---
# ResourceType
## Overview & Interactions
Describe what this does and why its a thing
## IntEnums
- Minerals
- Energy
- Organics
- Rare_Mats
- Power
- Defense
- Ships
- Transfer
- Research

# BuildingType
## Overview & Interactions
Describe what this does and why its a thing
## IntEnums
- Mine
- Farm
- Power Plant
- Factory
- Fort
- Shipyard
- Railyard
- Lab

# DepartmentType
## Overview & Interactions
Describe what this does and why its a thing
## IntEnums
- Resources
- Energy
- Agriculture
- Manufacturing
- Defnse
- Tarnsportation
- Commerce
- Intelligence

# BuildingState
## Overview & Interactions
Describe what this does and why its a thing
## IntEnums
- Constructing
- Active
- Damaged
- Repairing
- Surging
- Destroyed
- Inactive

# BuildingLevelStats
## Overview & Interactions
Describe what this does and why its a thing
## Fields
- build cost
- production rate
- production cost
- damage rate
- repair cost
- repair rate
- build ticks
- workforce
- notes

Pulls from predefined building stats. One set of stats for each level

[[Mine_Stats]]
[[Farm_Stats]]
[[Power_Plant_Stats]]
[[Factory_Stats]]
[[Fort_Stats]]
[[Shipyard_Stats]]
[[Railyard_Stats]]
[[Lab_Level_Stats]]

# WorkerLevel (IntEnum)
## Overview & Interactions
Defined in `buildings.py`. Represents worker skill levels 1–5. Used by [[Worker]], [[Building]].workforce, and [[LabLevelStats]].upskill_rates.
## Members
- L1 = 1
- L2 = 2
- L3 = 3
- L4 = 4
- L5 = 5

# Worker (dataclass)
## Overview & Interactions
Defined in `buildings.py`. A single worker unit with a `level` and `assigned_building_id`. Workers are tracked in [[Building]]._workers when assigned.

## Fields
- `level` – WorkerLevel
- `assigned_building_id` – Optional[int] (None = unassigned pool)
## Properties
- `is_assigned` – True if assigned to a building

# Building (dataclass)
## Overview & Interactions
Defined in `buildings.py`. Runtime building instance. Static stats from BUILDING_STATS. Holds assigned workers in `_workers`. See [[Building]] for full docs.

## Fields
`id`, `building_type`, `level`, `state`, `health`, `max_health`, `ticks_remaining`, `planet_index`, `_workers`

## Properties
`stats`, `is_active`, `surge_multiplier`, `properly_staffed`

## Building Constants
| Constant | Value | Description |
|---|---|---|
| `INITIAL_HEALTH` | 1.0 | Default health on construction |
| `REPAIR_THRESHOLD` | 0.70 | Health below this → REPAIRING |
| `DAMAGED_THRESHOLD` | 0.50 | Health below this → DAMAGED |

---

Helper Functions
`_stats`
colony_production_rates
colony_production_costs
net_rates


