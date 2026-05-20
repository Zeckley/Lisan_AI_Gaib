---
tags:
  - HOME
---
# Lisan AI Gaib — Project Index

> Space-based strategy sim where factions compete across solar systems.
> Agents harvest resources, build structures, and use a small ML model to decide actions.

---

## Modules

| Module                   | File                         | Status   |
| ------------------------ | ---------------------------- | -------- |
| [[Modules/World Map]]    | `world/world_map.py`         | Active   |
| [[Modules/Terrain]]      | `world/terrain.py`           | Active   |
| [[Modules/Solar System]] | `simulation/solar_system.py` | Active   |
| [[Modules/Resources]]    | `simulation/resources.py`    | Active   |
| [[Faction]]     | `simulation/faction.py`      | Skeleton |

---

## Enums & Constants

- [[Enums/ResourceType]] — `MINERALS, ENERGY, ORGANICS, RARE_MATS`
- [[Enums/FactionType]] — `NEUTRAL, AGGRESSIVE, ECONOMIC, SCIENTIFIC`
- [[Enums/BuildingType]] — `MINE, POWER_PLANT, FARM, LAB, SHIPYARD, DEFENSE`
- [[Enums/TerrainType]] — `WATER, LAND, BEACH, MOUNTAIN`

---

## Patterns

- [[Patterns/Python Patterns]] — dataclass, IntEnum, rng conventions
- [[Patterns/Import Order]] — module dependency chain

---

## Sim Loop

1. Fill this in once I actually know what to do reference [[TaskList]]



___

## 🗂 File Structure

```
project_root/
├── world/
│   ├── world_map.py
│   └── terrain.py
├── simulation/
│   ├── solar_system.py
│   ├── resources.py
│   └── faction.py
└── notebooks/
    └── dev.ipynb
```

---

## 🔗 Quick Links
- [[Modules/Resources#sample_distribution]] — core flexible sampler
- [[Faction#update order]] — faction tick sequence
- [[Modules/World Map#terrain generation]] — wave-collapse subdivision

## Documentation

Explain how this shit works
[[Colony Module]]
[[Buildings Module]]
[[Snapshot Module]]



Faction Management
	Make a faction
	Assign a colony to faction
	Assign directive to colony
Colonies
	Make a colony
	
Assign directive to colony
Create building
Assign workers to buildings
