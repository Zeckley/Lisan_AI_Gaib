---
tags:
  - Class
  - Faction
---
# Faction Class

**File:** `simulation/colony.py`
**Depends on:** [[Modules/Resources]], [[Modules/Solar System]]
**Status:** 🚧 Skeleton — structure defined, logic to be filled

---

## Purpose
Factions are the competing agents. They own colonies across solar systems, build structures, accumulate stockpiles, and use either rule-based or ML-driven logic to decide actions each tick.

---

## Class Hierarchy

```
Faction
  └── Colony (1..n, keyed by system_id)
        └── Building (1..n)
```

> `Colony` holds a forward reference to `Faction` (`"Faction"` string) to avoid circular imports if modules are split into files.

---

## Faction

### Key Methods

| Method                             | Notes                                                                                |
| ---------------------------------- | ------------------------------------------------------------------------------------ |
| `tick()`                           | Main tick — see [[#update order]]                                                    |
| `can_afford(cost)`                 | Validates cost against stockpile without deducting                                   |
| `spend(cost)`                      | Deducts cost — always call `can_afford()` first                                      |
| `get_state_vector() -> np.ndarray` | ML interface — flattens stockpile, colony count, building counts into numeric vector |
| `decide_action()`                  | Entry point for rule-based or ML-driven logic                                        |

faction_strategy
treasury_summary
summary

Properties
`faction_id`
`name`
`treasury`
`_colonies`

colonies

## Fields
- faction_id
- name
- treasury
- `_colonies`
- `_tick`
## Properties
* colonies
## Methods
- add_colony
- get_colony
- issue_directive
- collect_taxes
- transfer_to_colony
- transfer_between_colonies
- critical_colonies
- colonies_with_flag
- tick
- `_faction_strategy`
- treasury_summary
- summary

### Update Order {#update order}
```
1. collect       ← harvest resources from colonies
2. pay upkeep    ← deduct maintenance costs
3. AI decision   ← decide_action()
4. update colonies
```

---

## Enums

- [[Enums/FactionType]] — `NEUTRAL, AGGRESSIVE, ECONOMIC, SCIENTIFIC`
- [[Enums/BuildingType]] — `MINE, POWER_PLANT, FARM, LAB, SHIPYARD, DEFENSE`

---
## ML Interface Notes
`get_state_vector()` should flatten:
- Stockpile values (one per `ResourceType`)
- Colony count
- Building counts (one per `BuildingType`)
- Possibly faction type, tick number, etc.

This vector is the input to whatever model drives `decide_action()`.

---

## Related
- [[Modules/Resources]] — stockpile keyed by `ResourceType`
- [[Modules/Solar System]] — colonies reference systems
- [[Enums/FactionType]]
- [[Enums/BuildingType]]
- [[CodeBase/Codebase Home|Codebase Home]]
