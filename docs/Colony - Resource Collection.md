---
tags:
  - System
---
# Colony — Resource Collection

> File: `simulation/colony.py`

Methods for collecting production, paying upkeep, feeding population, and computing net resource rates.

---

## `collect_resources()`

```python
def collect_resources(self) -> None
```

The main production collection routine. Called once per tick from [[Colony - Tick Loop#tick|tick()]].

**Process per building:**
1. Skip if not ACTIVE or SURGING
2. Pay upkeep (`production_cost`) — building goes [[BuildingState]].INACTIVE if cannot afford
3. Apply tax — fraction flows to `faction_stockpile`, remainder to local `stockpile`
4. Tax rate > 1.0 also draws from local stockpile reserves

**Tax behavior:**
- `tax_rate = 0.10` → 10% to faction, 90% to colony
- `tax_rate = 1.0` → 100% to faction, 0% to colony
- `tax_rate > 1.0` → draws additional from colony stockpile to fill faction share

**Side effects:**
- Updates `last_produced` and `last_consumed` ledgers
- Buildings unable to pay upkeep go INACTIVE

---

## `pay_repair_upkeep()`

```python
def pay_repair_upkeep(self) -> None
```

Pay repair costs for all buildings in [[BuildingState]].REPAIRING. If costs cannot be paid, the building reverts to [[BuildingState]].DAMAGED.

Called from [[Colony - Tick Loop#tick|tick()]].

---

## `feed_population()`

```python
def feed_population(self) -> bool
```

Feed the colony's population with ORGANICS from the local stockpile.

- **Returns `True`** if fully fed — deducts `organics_upkeep_per_tick` from stockpile
- **Returns `False`** if starving — consumes what's available, shrinks population by 1%, increments `_starving_ticks`, and sets `CriticalFlag.FOOD_SHORTAGE` after `FOOD_SHORTAGE_TICKS` consecutive starving ticks

Called from [[Colony - Tick Loop#tick|tick()]].

---

## `_net_rates()`

```python
def _net_rates(self) -> Dict[int, float]
```

Compute approximate net resource rates across all active buildings.

```
net_rate = production - consumption (including population organics upkeep)
```

Uses `colony_production_rates()` and `colony_production_costs()` from `buildings.py` (via [[Colony - Building Management#building_counts|building_counts]]). Automatically includes `organics_upkeep_per_tick` in ORGANICS consumption.

**Used by:**
- [[Colony - Flag System#evaluate_flags|evaluate_flags()]]
- [[Colony - Decision Making#execute_directive|execute_directive()]]
- [[Colony - Decision Making#_basic_survival_loop|_basic_survival_loop()]]
- [[Colony - Decision Making#_validate_build|_validate_build()]]
- [[Colony - Decision Making#_validate_upgrade|_validate_upgrade()]]

---

## Resource Flow

```
                  ┌─────────────────┐
                  │  Active Buildings│
                  │  (ACTIVE/SURGING)│
                  └────────┬────────┘
                           │
                    production_this_tick()
                           │
                           ▼
              ┌────────────────────────┐
              │  Pay Upkeep (deduct)   │
              │  from stockpile        │
              └────────┬───────────────┘
                       │ fail → BUILDING INACTIVE
                       │
                       ▼
              ┌────────────────────────┐
              │  Split: tax_rate       │
              │  ┌──────┐  ┌────────┐ │
              │  │Local │  │Faction │ │
              │  │stock-│  │stock-  │ │
              │  │pile  │  │pile    │ │
              │  └──────┘  └────────┘ │
              └────────────────────────┘
                           │
                           ▼ (later in tick)
              ┌────────────────────────┐
              │  feed_population()     │
              │  (deduct ORGANICS)     │
              └────────────────────────┘
```

---

## Related

- [[Colony - Workers & Population]]
- [[Colony - Building Management]]
- [[Colony - Flag System]]
- [[Colony - Decision Making]]
- [[Colony - Tick Loop]]
- [[Building]]
- [[BuildingLevelStats]]
- [[colony_production_rates]]
- [[colony_production_costs]]
