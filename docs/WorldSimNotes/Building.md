---
tags:
  - Class
  - Building
---
# Building Class

> File: `simulation/buildings.py`

Runtime instance of a building owned by a [[Colony]]. Static stats are always fetched live from `BUILDING_STATS[type][level]`.

## Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `id` | `int` | required | Unique int within the colony |
| `building_type` | [[BuildingType]] | required | Mine, Farm, Power Plant, etc. |
| `level` | `int` | `1` | Current level (1–5) |
| `state` | [[BuildingState]] | `CONSTRUCTING` | Lifecycle state |
| `health` | `float` | `INITIAL_HEALTH` (1.0) | 0.0 → 1.0 |
| `max_health` | `float` | `INITIAL_HEALTH` (1.0) | Maximum health cap |
| `ticks_remaining` | `int` | `0` | Countdown for CONSTRUCTING |
| `planet_index` | `Optional[int]` | `None` | Which planet the building sits on |
| `_workers` | `List[[Worker]]` | `[]` | Workers currently assigned to this building |

## Properties

| Property | Return Type | Description |
|---|---|---|
| `stats` | `BuildingLevelStats` | Live lookup from `BUILDING_STATS[btype][level]` |
| `is_active` | `bool` | `True` if state is ACTIVE or SURGING |
| `surge_multiplier` | `float` | `1.5` if SURGING, else `1.0` |
| `properly_staffed` | `bool` | `True` if all required workers from `stats.workforce` are in `_workers` |

## Methods

| Method | Description |
|---|---|
| `apply_damage()` | Reduce health by `damage_rate` (2× while SURGING). Transitions to DAMAGED / REPAIRING / DESTROYED |
| `apply_repair()` | Increase health by `repair_rate`. Transitions to ACTIVE when full |
| `advance_construction()` | Count down `ticks_remaining`. Returns `True` when construction finishes |
| `production_this_tick()` | `{resource: amount}` from `stats.production_rate` (adjusted for surge) |
| `upkeep_this_tick()` | `{resource: amount}` from `stats.production_cost` |
| `repair_upkeep_this_tick()` | `{resource: amount}` from `stats.repair_cost` (only when REPAIRING) |
| `summary()` | Single-line formatted status string |

## Worker Tracking

The `_workers` list is the source of truth for which [[Worker]] instances are assigned to this building. When [[Colony - Workers & Population#assign_workers_to_building|assign_workers_to_building()]] succeeds, it replaces `_workers` with the newly assigned list. When [[Colony - Workers & Population#unassign_workers_from_building|unassign_workers_from_building()]] is called, `_workers` is cleared.

## Constants

Constants used by Building, defined in `buildings.py`:

| Constant | Value | Used In |
|---|---|---|
| `INITIAL_HEALTH` | `1.0` | Default health, construction completion |
| `REPAIR_THRESHOLD` | `0.70` | Health below 70% → REPAIRING |
| `DAMAGED_THRESHOLD` | `0.50` | Health below 50% → DAMAGED |

## Related
- [[Buildings Module]] — module where Building is defined
- [[Colony - Building Management]] — colony-level building commands
- [[Colony - Workers & Population]] — worker assignment methods