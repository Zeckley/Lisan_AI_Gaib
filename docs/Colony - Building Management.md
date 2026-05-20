---
tags:
  - System
---
# Colony — Building Management

> File: `simulation/colony.py` (Building class is now `simulation/buildings.py`)

Methods for constructing, upgrading, repairing, and surging buildings in a [[Colony]].

---

## Queries

### `get_building()`

```python
def get_building(self, building_id: int) -> Optional[Building]
```

Look up a [[Building]] by its unique `id`. Returns `None` if not found.

### `active_buildings`

```python
@property
def active_buildings(self) -> List[Building]
```

Return all buildings in [[BuildingState]].ACTIVE or [[BuildingState]].SURGING.

### `building_counts`

```python
@property
def building_counts(self) -> Dict[Tuple[BuildingType, int], int]
```

Return a dict of `{(building_type, level): count}` for all active buildings. Used by [[Colony - Resource Collection#_net_rates|_net_rates()]] and `colony_production_rates()` / `colony_production_costs()` in `buildings.py`.

---

## Construction

### `construct_building()`

```python
def construct_building(
    self,
    building_type: BuildingType,
    planet_index: Optional[int] = None,
    level: int = 1,
) -> Optional[Building]
```

Deduct build cost from local stockpile (using `_deduct()`) and queue a new [[Building]] in [[BuildingState]].CONSTRUCTING. Returns the [[Building]] instance on success, `None` if insufficient resources.

The build cost is looked up from `[[BUILDING_STATS]][building_type][level].build_cost`. Construction duration comes from `build_ticks`.

Called by:
- [[Colony - Decision Making#_rule_build|_rule_build()]]
- [[Colony - Decision Making#_basic_survival_loop|_basic_survival_loop()]]
- [[Colony - Decision Making#_rule_expand|_rule_expand()]]

### `can_upgrade_building()`

```python
def can_upgrade_building(self, building_id: int) -> bool
```

Legacy check — returns `True` if building exists, is ACTIVE, below max level, can afford the next level's build cost, and enough workers exist at the required levels. **Not used by current decision logic** which prefers [[Colony - Decision Making#_validate_upgrade|_validate_upgrade()]].

### `upgrade_building()`

```python
def upgrade_building(self, building_id: int) -> bool
```

Deduct next level's build cost, increment the building's level, and set state to [[BuildingState]].CONSTRUCTING with the appropriate build ticks. Returns `False` if building not found, not ACTIVE, or at max level.

Called by:
- [[Colony - Decision Making#_rule_upgrade|_rule_upgrade()]]
- [[Colony - Decision Making#_basic_survival_loop|_basic_survival_loop()]] (LAB upgrade)

---

## Repairs

### `start_repair()`

```python
def start_repair(self, building_id: int) -> bool
```

Begin repairing a damaged building. Transitions from [[BuildingState]].DAMAGED to [[BuildingState]].REPAIRING. Returns `False` if building not found or not in DAMAGED state.

Called by [[Colony - Decision Making#execute_directive|execute_directive()]] during the repair phase.

---

## Surge Mode

### `set_surge()`

```python
def set_surge(self, building_id: int, active: bool) -> bool
```

Toggle surge mode on a building.

- **Activating**: Requires building to be ACTIVE with health ≥ `SURGE_HEALTH_MIN` (0.80). Sets state to [[BuildingState]].SURGING (1.5× production, 2× damage rate).
- **Deactivating**: Requires building to be SURGING. Reverts to [[BuildingState]].ACTIVE.

Called by [[Colony - Decision Making#_rule_harvest|_rule_harvest()]].

---

## Building Lifecycle States

```
CONSTRUCTING → (build_ticks expire) → ACTIVE
ACTIVE → (damage) → DAMAGED → (start_repair) → REPAIRING → (health=1.0) → ACTIVE
ACTIVE → (surge) → SURGING → (de-escalate) → ACTIVE
ACTIVE / SURGING → (upkeep deficit / unstaffed) → INACTIVE
ALL STATES → (health=0) → DESTROYED
```

---

## Related

- [[Colony - Decision Making]]
- [[Colony - Resource Collection]]
- [[Colony - Tick Loop]]
- [[Building]]
- [[BuildingType]]
- [[BuildingState]]
- [[BuildingLevelStats]]
- [[BUILDING_STATS]]
