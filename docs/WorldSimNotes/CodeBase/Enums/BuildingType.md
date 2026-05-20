---
tags:
  - Constant
---
# BuildingType

**Module:** [[Buildings Module]] (`simulation/building.py`)
**Type:** `IntEnum`

| Value | Name          | Likely Effect                |
| ----- | ------------- | ---------------------------- |
| `0`   | `MINE`        | Boosts MINERALS              |
| `1`   | `POWER_PLANT` | Boosts ENERGY                |
| `2`   | `FARM`        | Boosts ORGANICS              |
| `3`   | `LAB`         | Boosts RARE_MATS or research |
| `4`   | `SHIPYARD`    | Unit production              |
| `5`   | `DEFENSE`     | Reduces attack vulnerability |

Building counts are included in `get_state_vector()` for the ML model input.
Buildings will override `Planet.resource_configs` to modify output per planet.

→ [[Building]]
