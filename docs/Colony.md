---
tags:
  - Class
---
# Colony

> File: `simulation/colony.py`

Local simulation agent tied to one solar system. Owns buildings, workers, a local stockpile, and a faction sub-stockpile. Receives a [[Directive]] from its parent [[Faction]] each tick and executes rule-based logic to decide what to build, repair, surge, or export.

---

## Constructor Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `colony_id` | `int` | required | Unique int within the parent [[Faction]] |
| `name` | `str` | required | Display name |
| `system_id` | `int` | required | Which [[SolarSystem]] this colony occupies |
| `population` | `float` | `1000.0` | Starting headcount |
| `stockpile` | `Dict[int, float]` | `{}` | Local resource pool `{resource_key: amount}` |
| `faction_stockpile` | `Dict[int, float]` | `{}` | Faction tax pool — drawn by [[Faction]], not Colony |

---

## Runtime State Fields

| Field | Type | Description |
|---|---|---|
| `starting_pop` | `float` | Recorded at construction for collapse-threshold maths |
| `directive` | [[Directive]] | Active directive from the parent faction |
| `critical_flags` | `Set[[CriticalFlag]]` | Existential threats active this tick |
| `strategic_flags` | `Set[[StrategicFlag]]` | Multi-tick concerns active this tick |
| `_buildings` | `List[[Building]]` | All buildings owned by the colony |
| `_workers` | `List[[Worker]]` | All workers in the colony |
| `_next_id` | `int` | Auto-incrementing ID counter for buildings |
| `_tick` | `int` | Tick counter |
| `_rng` | `Optional[np.random.Generator]` | Deterministic RNG for population growth |
| `_starving_ticks` | `int` | Consecutive starving tick counter for flag hysteresis |
| `_resource_low_ticks` | `Dict[int, int]` | Per-resource counter for RESOURCE_LOW flag |
| `_upskill_progress` | `Dict[int, float]` | Accumulated lab training progress per source worker level |
| `last_produced` | `Dict[int, float]` | Per-tick ledger of resources produced |
| `last_consumed` | `Dict[int, float]` | Per-tick ledger of resources consumed |
| `last_events` | `List[str]` | Per-tick event log for reporting |
| `verbose` | `bool` | Debug print toggle (default `True`) |

---

## Properties

| Property | Return Type | Description |
|---|---|---|
| `free_population` | `float` | Population not committed to workers (`pop - workers * 10`) |
| `organics_upkeep_per_tick` | `float` | Organics consumed by population per tick (`pop * 0.05`) |
| `power_stockpile` | `float` | Net power production across all active buildings |
| `required_workers` | `int` | Total workers needed to staff all active buildings |
| `active_buildings` | `List[[Building]]` | Buildings in ACTIVE or SURGING state |
| `building_counts` | `Dict[Tuple[BuildingType, int], int]` | Count of active buildings by `(type, level)` |

---

## Method Index

### Private Helpers
| Method | Description |
|---|---|
| [[Colony - Private Helpers#_new_id\|_new_id()]] | Generate and return a new unique building ID |
| [[Colony - Private Helpers#_add_to\|_add_to()]] | Add amount to a resource ledger |
| [[Colony - Private Helpers#_deduct\|_deduct()]] | Atomically deduct costs from a pool |
| [[Colony - Private Helpers#_can_afford\|_can_afford()]] | Check if stockpile holds multiplier × costs |

### Workers & Population
| Method | Description |
|---|---|
| [[Colony - Workers & Population#recruit_workers\|recruit_workers()]] | Convert population into L1 workers |
| [[Colony - Workers & Population#recruit_workers_of_level\|recruit_workers_of_level()]] | Recruit/promote workers at a specific level |
| [[Colony - Workers & Population#required_workers_by_level\|required_workers_by_level()]] | Calculate worker shortage by level |
| [[Colony - Workers & Population#release_workers\|release_workers()]] | Remove unassigned workers, return population |
| [[Colony - Workers & Population#unassigned_workers\|unassigned_workers()]] | Get list of unassigned workers |
| [[Colony - Workers & Population#unassigned_workers_by_level\|unassigned_workers_by_level()]] | Get unassigned worker counts by level |
| [[Colony - Workers & Population#workers_at_level\|workers_at_level()]] | Get workers at a specific level |
| [[Colony - Workers & Population#workers_by_level\|workers_by_level()]] | Get worker counts by level |
| [[Colony - Workers & Population#assign_workers_to_building\|assign_workers_to_building()]] | Assign workers to a building |
| [[Colony - Workers & Population#unassign_workers_from_building\|unassign_workers_from_building()]] | Unassign all workers from a building |
| [[Colony - Workers & Population#can_staff_L1_building\|can_staff_L1_building()]] | Check if L1 building can be staffed |
| [[Colony - Workers & Population#can_staff_building\|can_staff_building()]] | Check if a building can be staffed |
| [[Colony - Workers & Population#_lab_can_train\|_lab_can_train()]] | Check if a lab can train to target level |
| [[Colony - Workers & Population#_active_lab_upskill_rates\|_active_lab_upskill_rates()]] | Summed upskill rates from all active labs |
| [[Colony - Workers & Population#_process_upskilling\|_process_upskilling()]] | Process lab-driven worker promotions |

### Building Management
| Method | Description |
|---|---|
| [[Colony - Building Management#get_building\|get_building()]] | Get building by ID |
| [[Colony - Building Management#construct_building\|construct_building()]] | Deduct cost and queue construction |
| [[Colony - Building Management#can_upgrade_building\|can_upgrade_building()]] | Check if a building can be upgraded |
| [[Colony - Building Management#upgrade_building\|upgrade_building()]] | Deduct cost and begin upgrade |
| [[Colony - Building Management#start_repair\|start_repair()]] | Begin repairing a damaged building |
| [[Colony - Building Management#set_surge\|set_surge()]] | Toggle surge mode on a building |

### Resource Collection & Upkeep
| Method                                                                  | Description                                       |
| ----------------------------------------------------------------------- | ------------------------------------------------- |
| [[Colony - Resource Collection#collect_resources\|collect_resources()]] | Collect production + apply tax + pay upkeep       |
| [[Colony - Resource Collection#pay_repair_upkeep\|pay_repair_upkeep()]] | Pay repair costs for buildings in REPAIRING state |
| [[Colony - Resource Collection#feed_population\|feed_population()]]     | Consume organics to feed population               |
| [[Colony - Resource Collection#_net_rates\|_net_rates()]]               | Compute net resource production/consumption rates |

### Flag Evaluation
| Method | Description |
|---|---|
| [[Colony - Flag System#evaluate_flags\|evaluate_flags()]] | Recompute all critical and strategic flags |

### Decision Validation
| Method | Description |
|---|---|
| [[Colony - Decision Making#_construction_pipeline\|_construction_pipeline()]] | Examine buildings under construction |
| [[Colony - Decision Making#_validate_build\|_validate_build()]] | Run all 4 checks before building |
| [[Colony - Decision Making#_validate_upgrade\|_validate_upgrade()]] | Run all 4 checks before upgrading |

### Decision Engine
| Method | Description |
|---|---|
| [[Colony - Decision Making#execute_directive\|execute_directive()]] | Main decision dispatcher (tick phase 9) |
| [[Colony - Decision Making#_basic_survival_loop\|_basic_survival_loop()]] | Handle survival-critical needs |
| [[Colony - Decision Making#_rule_harvest\|_rule_harvest()]] | HARVEST directive — surge buildings |
| [[Colony - Decision Making#_rule_build\|_rule_build()]] | BUILD directive — construct new buildings |
| [[Colony - Decision Making#_rule_upgrade\|_rule_upgrade()]] | UPGRADE directive — upgrade buildings |
| [[Colony - Decision Making#_rule_expand\|_rule_expand()]] | EXPAND directive — shipyard/ships |
| [[Colony - Decision Making#_rule_export\|_rule_export()]] | EXPORT directive — send resources |

### Tick Loop
| Method | Description |
|---|---|
| [[Colony - Tick Loop#tick\|tick()]] | Advance colony by one tick |

### Reporting
| Method | Description |
|---|---|
| [[Colony - Reporting#summary\|summary()]] | Full colony state report |
| [[Colony - Reporting#stockpile_summary\|stockpile_summary()]] | Stockpile breakdown |
| [[Colony - Reporting#flag_summary\|flag_summary()]] | Active flags report |

---

## Related Classes

| Class | File | Description |
|---|---|---|
| [[Faction]] | `colony.py` | Strategic agent owning one or more Colonies |
| [[Directive]] | `colony.py` | Faction-to-colony orders |
| [[CriticalFlag]] | `colony.py` | Existential threat flags |
| [[StrategicFlag]] | `colony.py` | Multi-tick concern flags |
| [[Building]] | `buildings.py` | Runtime building instance |
| [[Worker]] | `buildings.py` | Worker unit with skill level |
| [[BuildingType]] | `buildings.py` | Building type enum (8 types) |
| [[BuildingState]] | `buildings.py` | Building lifecycle states |
| [[BuildingLevelStats]] | `buildings.py` | Per-level static building stats |
| [[ResourceType]] | `buildings.py` | Resource enum (MINERALS, ORGANICS, etc.) |

---

## Related Documentation

- [[Colony - Workers & Population]]
- [[Colony - Building Management]]
- [[Colony - Resource Collection]]
- [[Colony - Flag System]]
- [[Colony - Decision Making]]
- [[Colony - Tick Loop]]
- [[Colony - Reporting]]
- [[Colony - Private Helpers]]
- [[DirectiveManager]]
- [[DirectiveIssuer]]
