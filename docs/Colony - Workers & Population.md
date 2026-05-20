---
tags:
  - System
---
# Colony — Workers & Population

> File: `simulation/colony.py` (Worker and WorkerLevel classes are now `simulation/buildings.py`)

All methods related to worker management, recruitment, assignment, and the lab-driven upskilling pipeline.

---

## Constants

| Constant | Value | Description |
|---|---|---|
| `POP_PER_WORKER` | `10` | Population units consumed per L1 worker recruited |
| `ORGANICS_PER_POP` | `0.05` | Organics consumed per population unit per tick |

---

## Properties

### `free_population`

```python
@property
def free_population(self) -> float
```

Population not committed to workers: `population - len(workers) * POP_PER_WORKER`. This is the pool available for recruiting new workers.

### `organics_upkeep_per_tick`

```python
@property
def organics_upkeep_per_tick(self) -> float
```

Organics consumed by the entire population each tick: `population * ORGANICS_PER_POP`. Used in [[Colony - Resource Collection#_net_rates|_net_rates()]] and [[Colony - Resource Collection#feed_population|feed_population()]].

### `required_workers`

```python
@property
def required_workers(self) -> int
```

Total headcount needed to fully staff all active buildings (sum of all workforce requirements across all [[BuildingState]].ACTIVE and [[BuildingState]].SURGING buildings).

---

## Recruitment

### `recruit_workers()`

```python
def recruit_workers(self, count: int = 1) -> int
```

Convert `free_population` into level-1 [[Worker]] instances. Each worker costs `POP_PER_WORKER` population. Returns the number of workers actually created (limited by available free population).

### `recruit_workers_of_level()`

```python
def recruit_workers_of_level(self, level: int, count: int = 1) -> int
```

Recruit workers at a specific level (1–5).

- **Level 1**: Delegates to [[Colony - Workers & Population#recruit_workers|recruit_workers()]]
- **Levels 2–5**: Checks if at least one active [[BuildingType]].LAB can train to that level via [[Colony - Workers & Population#_lab_can_train|_lab_can_train()]]. If so, promotes an unassigned worker at `level-1` from the pool. Falls back to recruiting an L1 and letting the upskilling pipeline handle it.

### `release_workers()`

```python
def release_workers(self, count: int = 1) -> int
```

Remove unassigned workers from the worker list, returning their population cost back to the pool. Returns count actually released.

---

## Worker Queries

### `unassigned_workers()`

```python
def unassigned_workers(self) -> List[Worker]
```

Return list of all [[Worker]] instances with `assigned_building_id == None`.

### `unassigned_workers_by_level()`

```python
def unassigned_workers_by_level(self) -> Dict[int, int]
```

Return `{worker_level: count}` for all unassigned workers. Used extensively by [[Colony - Decision Making#_validate_build|_validate_build()]] and [[Colony - Decision Making#_validate_upgrade|_validate_upgrade()]].

### `workers_at_level()`

```python
def workers_at_level(self, level: int) -> List[Worker]
```

Return all workers at a given skill level (assigned or unassigned).

### `workers_by_level()`

```python
def workers_by_level(self) -> Dict[int, int]
```

Return `{worker_level: count}` for all workers. Used by [[Colony - Building Management#can_staff_building|can_staff_building()]] and [[Colony - Building Management#can_upgrade_building|can_upgrade_building()]].

### `required_workers_by_level()`

```python
def required_workers_by_level(self) -> Dict[int, int]
```

Calculate worker shortage per level across all buildings in [[BuildingState]].ACTIVE, [[BuildingState]].SURGING, or [[BuildingState]].INACTIVE. Subtracts currently assigned workers. Returns `{level: shortage_count}` for levels with non-zero shortage. Used in the [[Colony - Tick Loop#tick|tick()]] auto-recruit phase.

---

## Worker Assignment

### `assign_workers_to_building()`

```python
def assign_workers_to_building(self, building: Building) -> bool
```

Assign workers to a [[Building]] based on its workforce requirements from [[BuildingLevelStats]].workforce.

- First unassigns ALL current workers from the building via `building._workers` (handles upgrades cleanly)
- Tries to fill the workforce from unassigned worker pool
- On success: sets `building._workers` to the assigned list and [[BuildingState]].ACTIVE
- On failure: clears `building._workers` and sets [[BuildingState]].INACTIVE

Called on construction completion in [[Colony - Tick Loop#tick|tick()]].

### `unassign_workers_from_building()`

```python
def unassign_workers_from_building(self, building: Building) -> int
```

Unassign all workers from a building via `building._workers.clear()`. Sets building state to [[BuildingState]].INACTIVE. Returns count of workers unassigned. Used by [[Colony - Decision Making#_basic_survival_loop|_basic_survival_loop()]] during crisis response.

### `can_staff_L1_building()`

```python
def can_staff_L1_building(self, building_type: BuildingType) -> bool
```

Check if enough unassigned workers exist to staff a level-1 building of the given type. Uses `workers_by_level()` (all workers, not just unassigned). Used by [[Colony - Decision Making#_basic_survival_loop|_basic_survival_loop()]].

### `can_staff_building()`

```python
def can_staff_building(self, building: Building) -> bool
```

Check if enough unassigned workers exist to staff a specific building at its current level. Uses `workers_by_level()`.

---

## Lab / Upskill Pipeline

### `_active_lab_upskill_rates()`

```python
def _active_lab_upskill_rates(self) -> List[float]
```

Sum the `upskill_rates` across all active [[BuildingType]].LAB buildings. Returns a 4-element list: `[L1→L2, L2→L3, L3→L4, L4→L5]` rates per tick. Uses [[LabLevelStats]] from `buildings.py`.

### `_lab_can_train()`

```python
def _lab_can_train(self, target_level: int) -> bool
```

Returns `True` if at least one active lab can train workers to `target_level`. Checks if `upskill_rates[target_level - 2] > 0`.

### `_process_upskilling()`

```python
def _process_upskilling(self) -> None
```

Process lab-driven worker promotions. Called once per tick (currently from [[Colony - Tick Loop#tick|tick()]]).

- Accumulates training progress using `_active_lab_upskill_rates()`
- When progress reaches `>= 1.0` for a tier, promotes one unassigned worker at that source level
- Bleeds off excess progress if no eligible workers exist (avoids runaway accumulation)

---

## Related

- [[Colony - Building Management]]
- [[Colony - Decision Making]]
- [[Colony - Tick Loop]]
- [[Worker]]
- [[WorkerLevel]]
- [[LabLevelStats]]
