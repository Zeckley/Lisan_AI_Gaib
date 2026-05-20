---
tags:
  - System
---
# Colony — Reporting

> File: `simulation/colony.py`

Methods for displaying [[Colony]] state and events. Used for debugging, logging, and simulation visualization.

---

## `summary()`

```python
def summary(self) -> str
```

Full colony state report. Returns a multi-line string containing:

- Colony name, ID, and current tick
- Active [[Directive]] type, tax rate, and urgency
- Population, free population, and worker count
- Organics upkeep per tick
- Active [[CriticalFlag]] and [[StrategicFlag]] flags
- Local and faction stockpile breakdown (all [[ResourceType]] entries)
- All buildings and their status (via [[Building#summary|Building.summary()]])
- Events logged this tick

---

## `stockpile_summary()`

```python
def stockpile_summary(self) -> str
```

Stockpile breakdown by resource. For each [[ResourceType]], shows both local and faction stockpile amounts.

**Example output:**
```
  Local stockpile:
    MINERALS     local=   4704.00  faction=     6.00
    ORGANICS     local=   4646.65  faction=    19.20
    POWER        local=    644.00  faction=    16.00
    ...
```

---

## `flag_summary()`

```python
def flag_summary(self) -> str
```

Active flags report. Lists critical and strategic flag names, or "none" if none are active.

**Example output:**
```
  Critical: [FOOD_SHORTAGE]  Strategic: [DEFENSE_NEEDED, RESOURCE_LOW]
```

---

## Per-Tick Ledgers

These dicts are reset each tick and populated during the [[Colony - Tick Loop#tick|tick()]] phases:

| Ledger | Populated By | Description |
|---|---|---|
| `last_produced` | [[Colony - Resource Collection#collect_resources|collect_resources()]] | Resources produced this tick |
| `last_consumed` | [[Colony - Resource Collection#collect_resources|collect_resources()]], [[Colony - Resource Collection#feed_population|feed_population()]], [[Colony - Resource Collection#pay_repair_upkeep|pay_repair_upkeep()]] | Resources consumed this tick |
| `last_events` | All decision and lifecycle methods | Human-readable event strings for this tick |

---

## Building Summary

Each [[Building]] has a `summary()` method that outputs:

```
[  0] FARM         lv1  ACTIVE       [████████████████████] 100.0%
[  1] POWER_PLANT  lv2  CONSTRUCTING [░░░░░░░░░░░░░░░░░░░░]   0.0%  ⏳ 18t left
```

Format: `[id] Type lvN  State  [health_bar] health%  [ticks_left]`

---

## Verbose Mode

When `verbose = True`, `tick()` prints a header and all `last_events` to stdout:

```
======   Tick 11   ======
Colony Arrakeen (id=0):  🏗 Queued FARM (BUILD).
Colony Arrakeen (id=0):  👷 Assigned 5 worker(s) to FARM lv1 (id=6).
```

---

## Related

- [[Colony - Tick Loop]]
- [[Colony - Resource Collection]]
- [[Colony - Decision Making]]
- [[Building#summary|Building.summary()]]
- [[Faction#summary|Faction.summary()]]
- [[snapshot.py]] — `take_snapshot()` and `plot_history()` for data collection
