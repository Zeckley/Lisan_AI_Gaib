---
tags:
  - System
---
# Colony — Flag System

> File: `simulation/colony.py`

Two-tier flag system for communicating colony state. [[CriticalFlag]] flags represent existential threats. [[StrategicFlag]] flags represent multi-tick concerns that the [[Faction]] agent can route around.

---

## CriticalFlag (IntEnum)

Existential threats that **always** surface to the faction and **block** any directive action that would deepen the problem.

| Flag | Value | Trigger |
|---|---|---|
| `FOOD_SHORTAGE` | 0 | Colony has been starving for `FOOD_SHORTAGE_TICKS` consecutive ticks |
| `POWER_DEFICIT` | 1 | Net POWER production is negative |
| `POPULATION_COLLAPSE` | 2 | Population has fallen below `POPULATION_COLLAPSE_FRAC` (25%) of starting population |

Critical flags cannot be overridden by the [[Directive]] `override_flags`.

---

## StrategicFlag (IntEnum)

Multi-tick concerns. Fully overridable by the [[Directive]] `override_flags` set.

| Flag | Value | Trigger |
|---|---|---|
| `DEFENSE_NEEDED` | 0 | Net DEFENSE score below `DEFENSE_LOW_THRESHOLD` (10.0) |
| `WORKER_SHORTAGE` | 1 | Unassigned/total workforce ratio below `WORKER_SHORTAGE_RATIO` (0.1) |
| `RESOURCE_LOW` | 2 | One or more resources have had negative net rate for `RESOURCE_LOW_TICKS` (5) consecutive ticks |
| `EXPORT_STRAINED` | 3 | Tax rate is cutting into operational reserves below `EXPORT_STRAINED_THRESHOLD` (0.10) |
| `CONSTRUCTION_BLOCKED` | 4 | Colony cannot afford the cheapest L1 building (at `BUILD_STOCKPILE_MIN` multiplier) |

---

## `evaluate_flags()`

```python
def evaluate_flags(self) -> None
```

Recompute all flags based on current colony state. Called once per tick from [[Colony - Tick Loop#tick|tick()]].

**Critical flags** (set/cleared unconditionally):
- `POWER_DEFICIT` — based on net POWER rate
- `POPULATION_COLLAPSE` — based on population/starting_pop ratio
- `FOOD_SHORTAGE` — managed incrementally in [[Colony - Resource Collection#feed_population|feed_population()]]

**Strategic flags** (respect `directive.override_flags`):
- `DEFENSE_NEEDED` — based on net DEFENSE rate
- `WORKER_SHORTAGE` — based on unassigned/total workforce ratio
- `RESOURCE_LOW` — tracks consecutive ticks of negative net rate per resource
- `EXPORT_STRAINED` — compares local-after-tax production to total pool
- `CONSTRUCTION_BLOCKED` — skips if any building is under construction; otherwise checks cheapest building affordability

### `_set_strategic()` (nested helper)

```python
def _set_strategic(flag: StrategicFlag, condition: bool) -> None
```

Set or clear a strategic flag. If the flag is in `directive.override_flags`, it is always cleared regardless of condition.

---

## Flag Threshold Constants

| Constant | Value | Description |
|---|---|---|
| `FOOD_SHORTAGE_TICKS` | 1 | Consecutive starving ticks before FOOD_SHORTAGE fires |
| `FOOD_DEFICIT_THRESHOLD` | 10.0 | Net ORGANICS below this triggers starving tick |
| `POWER_DEFICIT_THRESHOLD` | 0.0 | Net POWER below this → POWER_DEFICIT |
| `POPULATION_COLLAPSE_FRAC` | 0.25 | Pop/starting_pop below this → POPULATION_COLLAPSE |
| `DEFENSE_LOW_THRESHOLD` | 10.0 | Net DEFENSE below this → DEFENSE_NEEDED |
| `BOLSTER_DEFENSE_TICKS` | 25 | Tick window for ramping defense |
| `WORKER_SHORTAGE_RATIO` | 0.1 | Unassigned/total ratio below this → WORKER_SHORTAGE |
| `RESOURCE_LOW_TICKS` | 5 | Consecutive negative ticks before RESOURCE_LOW |
| `EXPORT_STRAINED_THRESHOLD` | 0.10 | Local fraction remaining after tax below this → EXPORT_STRAINED |
| `BUILD_STOCKPILE_MIN` | 1.5 | Stockpile multiplier for build affordability checks |

---

## Flag Flow

```
tick()
  └─ feed_population()
  │      └─ manages FOOD_SHORTAGE critical flag
  └─ evaluate_flags()
         ├─ Critical: POWER_DEFICIT, POPULATION_COLLAPSE (FOOD_SHORTAGE handled above)
         └─ Strategic: DEFENSE_NEEDED, WORKER_SHORTAGE, RESOURCE_LOW,
                        EXPORT_STRAINED, CONSTRUCTION_BLOCKED
                              │
                              ▼
  Faction reads flags → Faction._faction_strategy()
                              │
                              ▼
  Faction adjusts directive / sends aid
```

---

## Related

- [[Colony - Decision Making#_basic_survival_loop|_basic_survival_loop()]] — reads critical flags to drive emergency response
- [[Colony - Decision Making#execute_directive|execute_directive()]] — skips directive execution if survival loop returns False
- [[Faction#_faction_strategy|Faction._faction_strategy()]] — faction-level response to colony flags
- [[Faction#issue_directive|Faction.issue_directive()]] — clamps tax rate based on critical flags
- [[Directive#override_flags|Directive.override_flags]] — faction can suppress strategic flags
