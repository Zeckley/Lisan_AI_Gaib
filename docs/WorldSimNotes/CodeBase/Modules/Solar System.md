---
tags:
  - Class
---
# Solar System

**File:** `simulation/solar_system.py`
**Depends on:** [[Modules/Resources]]

---

## Purpose
Defines harvestable nodes (solar systems) in 3D space, each containing planets with resources. Deliberately decoupled from the spherical mesh so resource/faction equations can be tuned independently before re-integrating with the planetary map.

---

## Class Hierarchy

```
SolarSystem
  └── Planet (1..n)
        └── resources: dict[ResourceType, float]
```

---

## `SolarSystem`
A node in 3D space. Contains one or more `Planet` objects.

- Identified by `system_id`
- Holds list of `Planet` instances
- Acts as the harvestable unit factions compete over

---

## `Planet`

| Attribute | Type | Notes |
|-----------|------|-------|
| `size` | `float` | Scales resource output via `size_exponent` |
| `quality` | `float` | Range `[0, 5]`; quality=0 always yields zero resources |
| `resources` | `dict[ResourceType, float]` | Computed from `resource_configs` |
| `resource_configs` | `dict[ResourceType, ResourceConfig]` | Overrides `DEFAULT_RESOURCE_CONFIGS` |

### Quality note
Quality is normalised to `[0, 1]` internally before applying `quality_exponent`:
```
output *= (quality / 5) ** quality_exponent
```
A quality of 0 short-circuits to 0 regardless of distribution.

---

## Resource Generation Flow
1. Pull `ResourceConfig` for each `ResourceType` (planet-level override or default)
2. Call `sample_distribution(...)` with planet `size` and `quality` applied as exponent scaling
3. Store results in `planet.resources`

---

## Related
- [[Modules/Resources]] — `ResourceConfig`, `sample_distribution`, `DEFAULT_RESOURCE_CONFIGS`
- [[Enums/ResourceType]]
- [[Faction]] — colonies reference solar systems
- [[CodeBase/Codebase Home|Codebase Home]]
