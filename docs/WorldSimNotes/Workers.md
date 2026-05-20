---
tags:
  - Class
---
# Worker Class

> File: `simulation/buildings.py`

Workers are created from a colonies population. Buildings require workers to operate and will go inactive if there is not enough workers or the right level of workers is not present.
In order to upgrade workers a lab of sufficient level is required. Workers take time to upgrade.

## Fields
- `level`: [[WorkerLevel]] (1–5)
- `assigned_building_id`: `Optional[int]` — `None` = unassigned pool

## Properties
- `is_assigned` — `True` if `assigned_building_id is not None`

## Worker Tracking
Workers assigned to a [[Building]] are stored in that building's `_workers` list. When assigning, workers are added to `building._workers`; when unassigning, the list is cleared. This makes `building._workers` the direct source of truth for which workers staff a given building.

## Future
- `is_tired`
- `is_angry`
- `is_satisfied`

---

## WorkerLevel (IntEnum)

| Member | Value |
|--------|-------|
| `L1`   | 1     |
| `L2`   | 2     |
| `L3`   | 3     |
| `L4`   | 4     |
| `L5`   | 5     |

---

## Related
- [[Building]] — `_workers` list holds assigned Worker instances
- [[Buildings Module]] — module where Worker and WorkerLevel are defined
- [[Colony - Workers & Population]] — colony-level worker management