---
tags:
  - System
---
# Colony — Private Helpers

> File: `simulation/colony.py`

Low-level utility methods used throughout the [[Colony]] class.

---

## `_new_id()`

```python
def _new_id(self) -> int
```

Generate and return a new unique building ID. Increments the internal `_next_id` counter.

---

## `_add_to()`

```python
def _add_to(self, ledger: Dict[int, float], resource: int, amount: float) -> None
```

Add `amount` to a resource in a ledger dict. Used for accumulating production/consumption in [[Colony - Resource Collection#collect_resources|collect_resources()]] and [[Colony - Resource Collection#feed_population|feed_population()]].

---

## `_deduct()`

```python
def _deduct(self, costs: Dict[int, float], pool: Optional[Dict[int, float]] = None) -> bool
```

Atomically deduct costs from a pool (defaults to local `stockpile`). Returns `True` if all costs could be paid, `False` if insufficient (with no deduction made). Used by:
- [[Colony - Building Management#construct_building|construct_building()]]
- [[Colony - Building Management#upgrade_building|upgrade_building()]]
- [[Colony - Resource Collection#collect_resources|collect_resources()]]
- [[Colony - Resource Collection#pay_repair_upkeep|pay_repair_upkeep()]]

---

## `_can_afford()`

```python
def _can_afford(self, costs: Dict[int, float], multiplier: float = 1.0) -> bool
```

Check if the local stockpile holds at least `multiplier × costs` for all resource keys. Uses `BUILD_STOCKPILE_MIN = 1.5` as the standard multiplier for construction decisions. Used by:
- [[Colony - Decision Making#_validate_build|_validate_build()]]
- [[Colony - Decision Making#_validate_upgrade|_validate_upgrade()]]
- [[Colony - Decision Making#_basic_survival_loop|_basic_survival_loop()]]
- [[Colony - Building Management#can_upgrade_building|can_upgrade_building()]]
- [[Colony - Flag System#evaluate_flags|evaluate_flags()]]
