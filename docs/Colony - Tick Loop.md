---
tags:
  - System
---
# Colony — Tick Loop

> File: `simulation/colony.py`

The main simulation advancement method for [[Colony]].

---

## `tick()`

```python
def tick(self, verbose=True, enable_decision=True) -> None
```

Advance the colony by one tick. This is the **outer loop** called by [[Faction#tick|Faction.tick()]].

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `verbose` | `True` | Print tick events to stdout |
| `enable_decision` | `True` | Whether to run `execute_directive()`. Set `False` for observation-only mode. |

---

## Order of Operations

```
   ┌─ 1. Reset per-tick ledgers (last_produced, last_consumed, last_events)
   │
   ├─ 2. Auto-recruit workers to meet building demand
   │      └─ Calls required_workers_by_level() → recruit_workers_of_level()
   │
   ├─ 3. Advance construction timers
   │      └─ Building.advance_construction()
   │      └─ On completion: assign_workers_to_building()
   │
   ├─ 4. Collect resources + pay building upkeep
   │      └─ collect_resources()
   │
   ├─ 5. Population growth / decay
   │      └─ Uses lognormal distribution, deterministic per-colony RNG
   │      └─ Growth rate boosted 25% under EXPAND directive
   │      └─ Requires 3× organics_upkeep_per_tick in stockpile for growth
   │
   ├─ 6. Apply wear-and-tear damage
   │      └─ Building.apply_damage() for each active building
   │
   ├─ 7. Pay repair upkeep + advance repairs
   │      └─ pay_repair_upkeep()
   │      └─ Apply repairs for REPAIRING / DAMAGED buildings
   │
   ├─ 8. Feed population
   │      └─ feed_population()
   │      └─ May set CriticalFlag.FOOD_SHORTAGE
   │
   ├─ 9. Evaluate flags
   │      └─ evaluate_flags()
   │
   └─ 10. Execute directive (if enable_decision)
           └─ execute_directive()
```

---

## Phase Details

### Phase 1 — Reset Ledgers
Clear `last_produced`, `last_consumed`, and `last_events` from the previous tick.

### Phase 2 — Auto-Recruit Workers
Check `required_workers_by_level()` for any worker shortages across ACTIVE, SURGING, and INACTIVE buildings. Recruits missing workers starting from the lowest skill level. If a lab cannot train the required level, recruits L1 workers and attempts to construct a lab at the needed level.

### Phase 3 — Construction Completion
Iterate buildings and call `Building.advance_construction()`. When a building completes construction, attempt `assign_workers_to_building()`. Falls to [[BuildingState]].INACTIVE if insufficient workers.

### Phase 4 — Resource Collection
Call `collect_resources()`. See [[Colony - Resource Collection]] for details.

### Phase 5 — Population Growth
Population grows/stagnates based on food availability:
- **If stockpile < 3× organics upkeep**: No growth (logged as food shortage)
- **If sufficient food**: Population changes by a lognormal random walk

Growth multiplier is 1.5× under EXPAND directive vs 1.2× otherwise.

### Phase 6 — Wear and Tear
Call `Building.apply_damage()` on each active building. Health decreases by `damage_rate/100` per tick (doubled while SURGING). Buildings at 0 health become [[BuildingState]].DESTROYED.

### Phase 7 — Repairs
Pay repair upkeep via `pay_repair_upkeep()`, then advance health for buildings in REPAIRING or DAMAGED state. Buildings below `REPAIR_THRESHOLD` are automatically set to REPAIRING.

### Phase 8 — Feeding
Call `feed_population()`. See [[Colony - Resource Collection#feed_population]] for details.

### Phase 9 — Flag Evaluation
Call `evaluate_flags()`. See [[Colony - Flag System]] for details.

### Phase 10 — Decision Execution
Call `execute_directive()`. See [[Colony - Decision Making]] for details. Skipped if `enable_decision=False`.

---

## Tick Sequence Diagram

```
Colony.tick()
  │
  ├─ Reset ledgers
  ├─ Auto-recruit workers ──────────────────► required_workers_by_level()
  ├─ Advance constructions ─────────────────► Building.advance_construction()
  │                                              └─ assign_workers_to_building()
  ├─ collect_resources() ───────────────────► Building.production_this_tick()
  │                                              Building.upkeep_this_tick()
  ├─ Population growth (lognormal walk)
  ├─ Building.apply_damage() ───────────────► BuildingState tracking
  ├─ pay_repair_upkeep()
  ├─ Building repair advancement ───────────► Building.apply_repair()
  ├─ feed_population() ─────────────────────► CriticalFlag.FOOD_SHORTAGE
  ├─ evaluate_flags() ──────────────────────► CriticalFlag + StrategicFlag
  └─ execute_directive() ───────────────────► _basic_survival_loop()
                                                  └─ _rule_build / _rule_upgrade / etc.
```

---

## Related

- [[Colony - Decision Making]]
- [[Colony - Resource Collection]]
- [[Colony - Workers & Population]]
- [[Colony - Building Management]]
- [[Colony - Flag System]]
- [[Colony - Reporting]]
- [[Faction#tick|Faction.tick()]] — calls Colony.tick() for each colony
