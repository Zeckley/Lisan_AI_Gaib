---
tags:
  - System
---
# Colony — Decision Making

> File: `simulation/colony.py`

The rule-based decision engine that determines what a [[Colony]] does each tick. Comprises decision validation checks, the survival loop, and directive-specific sub-rules.

---

## Decision Priority

```
0. Critical flag response  (survival always first)
1. Repair damaged buildings
2. Directive execution      (HARVEST / BUILD / UPGRADE / EXPAND / EXPORT)
3. Idle / balanced upkeep   (default state)
```

---

## Decision Validation

Three methods that implement the **4 checks** before any build or upgrade decision is made. These are used by [[Colony - Decision Making#_rule_build|_rule_build()]], [[Colony - Decision Making#_rule_upgrade|_rule_upgrade()]], [[Colony - Decision Making#_rule_expand|_rule_expand()]], and the LAB upgrade path in [[Colony - Decision Making#_basic_survival_loop|_basic_survival_loop()]].

### `_construction_pipeline()`

```python
def _construction_pipeline(self) -> Tuple[Dict[int, int], Dict[int, float]]
```

Examine all buildings currently in [[BuildingState]].CONSTRUCTING. Returns two values:

1. **`pending_workers`**: `{worker_level: count}` — workforce needed by all pending buildings
2. **`pending_rate_impact`**: `{resource_key: net_rate}` — combined production - consumption once pending buildings become active

This is the foundation for the **interference check** (check #4).

### `_validate_build()`

```python
def _validate_build(self, building_type: BuildingType) -> Tuple[bool, List[str]]
```

Validate whether building a new L1 building is safe. Runs all 4 checks:

| # | Check | What it verifies |
|---|---|---|
| 1 | **Resource** | Stockpile holds `BUILD_STOCKPILE_MIN × build_cost` |
| 2 | **Workforce** | Enough unassigned + recruitable workers exist (including pipeline) |
| 3 | **Rates** | Projected rates (current + pipeline + this building) don't go negative *except* for the building's own output |
| 4 | **Interference** | Pipeline workers and rate impacts are factored into checks 2 & 3 |

Returns `(True, [])` if all checks pass, or `(False, [reason_strings])` with descriptive failure reasons.

**Rate projection formula:**
```
projected_rate = current_rates + pending_rate_impact + building.production_rate - building.production_cost
```

### `_validate_upgrade()`

```python
def _validate_upgrade(self, building: Building) -> Tuple[bool, List[str]]
```

Same 4 checks as build, but accounts for the current building being **removed** during construction and the **new level's requirements** being added.

**Rate projection formula:**
```
projected_rate = current_rates 
               - current_stats.production_rate + current_stats.production_cost  # remove current
               + pending_rate_impact                                            # add pipeline
               + next_stats.production_rate - next_stats.production_cost         # add new level
```

Target resources (the resource the building produces) are excluded from the negative-rate rejection — building to address a deficit is expected behavior.

---

## `execute_directive()`

```python
def execute_directive(self) -> None
```

Main decision dispatcher. Called once per tick from [[Colony - Tick Loop#tick|tick()]].

**Execution flow:**
```
0. _basic_survival_loop(net) → returns False → skip directives
1. Repair any DAMAGED buildings with health < REPAIR_THRESHOLD
2. Dispatch to sub-rule based on directive_type:
   HARVEST → _rule_harvest()
   BUILD   → _rule_build()
   UPGRADE → _rule_upgrade()
   EXPAND  → _rule_expand()
   EXPORT  → _rule_export()
```

---

## `_basic_survival_loop()`

```python
def _basic_survival_loop(self, net: Dict[int, float]) -> bool
```

Handles survival-critical needs before directive execution. Returns `True` if survival is secured (continue to directives), `False` if crisis is active (skip directives this tick).

**Priority within the loop:**

| Priority | Condition | Response |
|---|---|---|
| 0 | FOOD_SHORTAGE or net ORGANICS < threshold | Build FARM (pipeline-aware — checks if one is already under construction). If can't afford, disable non-essential buildings consuming organics. |
| 1 | POWER_DEFICIT critical flag | Build POWER_PLANT (pipeline-aware). If can't afford, disable non-essential buildings consuming power. |
| 2 | DEFENSE score < needed | Reactivate inactive FORTs, build new FORTs (pipeline-aware). |
| 3 | INACTIVE buildings needing workers | Recruit workers via `recruit_workers_of_level()`. |
| 4 | LAB upgrade opportunity | Upgrade the highest-level LAB using [[Colony - Decision Making#_validate_upgrade|_validate_upgrade()]]. |

**Pipeline awareness:** For FOOD_SHORTAGE, POWER_DEFICIT, and FORT construction, the survival loop checks if a relevant building is already under construction before queuing another.

---

## Directive Sub-Rules

### `_rule_harvest()`

```python
def _rule_harvest(self, d: Directive, net: Dict[int, float]) -> None
```

**HARVEST directive** — surge all healthy active buildings for 1.5× production. Sets [[BuildingState]].SURGING on every active building with health ≥ `SURGE_HEALTH_MIN` (0.80).

### `_rule_build()`

```python
def _rule_build(self, d: Directive, net: Dict[int, float]) -> None
```

**BUILD directive** — construct new L1 buildings for the target resource.

Uses [[Colony - Decision Making#_validate_build|_validate_build()]] before constructing. If validation fails, all reasons are logged to `last_events`.

### `_rule_upgrade()`

```python
def _rule_upgrade(self, d: Directive, net: Dict[int, float]) -> None
```

**UPGRADE directive** — upgrade existing buildings for the target resource.

Sorts candidate buildings by level (ascending). For each building, runs [[Colony - Decision Making#_validate_upgrade|_validate_upgrade()]]. Upgrades the first valid candidate. Logs rejection reasons for invalid candidates.

### `_rule_expand()`

```python
def _rule_expand(self, d: Directive, net: Dict[int, float]) -> None
```

**EXPAND directive** — establish a new colony.

- If no shipyard exists, uses [[Colony - Decision Making#_validate_build|_validate_build()]] to check shipyard feasibility
- If shipyard exists, checks resource surplus for ship construction

### `_rule_export()`

```python
def _rule_export(self, d: Directive, net: Dict[int, float]) -> None
```

**EXPORT directive** — send target resource to another colony in exchange for WEALTH.

Refuses to export if the net rate of the target resource is ≤ the export demand (would put the colony into deficit).

---

## Decision Flow Diagram

```
tick()
  └─ execute_directive()
       ├─ _basic_survival_loop(net)
       │    ├─ FOOD_SHORTAGE? → build farm / disable consumers → return False
       │    ├─ POWER_DEFICIT? → build power plant / disable consumers → return False
       │    ├─ DEFENSE low?   → reactivate/build forts
       │    ├─ WORKER shortage? → recruit workers
       │    └─ LAB upgrade?  → validate_upgrade → upgrade
       │
       ├─ (if survival loop returned True)
       │    └─ Repair damaged buildings
       │
       └─ Directive dispatch
            ├─ HARVEST → _rule_harvest()
            ├─ BUILD   → _rule_build()
            │              └─ _validate_build() → construct or reject
            ├─ UPGRADE → _rule_upgrade()
            │              └─ _validate_upgrade() → upgrade or skip
            ├─ EXPAND  → _rule_expand()
            │              └─ _validate_build() (for shipyard)
            └─ EXPORT  → _rule_export()
```

---

## Related

- [[Colony - Workers & Population]]
- [[Colony - Building Management]]
- [[Colony - Resource Collection]]
- [[Colony - Flag System]]
- [[Colony - Tick Loop]]
- [[DirectiveType]]
- [[Directive]]
- [[BUILDING_STATS]]
