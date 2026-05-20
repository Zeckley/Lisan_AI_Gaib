---
tags:
  - Information
---
# Resources

**File:** `simulation/resources.py`
**Depends on:** nothing (leaf module — import this first)

---

## Purpose
Core resource sampling and configuration. No internal imports — safe to import from anywhere. Buildings and faction effects will override `ResourceConfig` values per planet.

---

## Key Classes

### `ResourceConfig` (dataclass)
Configuration for how a single resource type is generated on a planet.

| Field | Default | Notes |
|-------|---------|-------|
| `dist` | `"lognormal"` | Distribution name (`DISTRIBUTION` type alias) |
| `mean` | `1.0` | Distribution mean |
| `std` | `0.5` | Distribution std dev |
| `scale` | `100.0` | Master output multiplier |
| `clip` | `(0.0, None)` | Hard bounds — `None` upper = `np.inf` |
| `size_exponent` | `1.2` | Output `*= planet_size ** size_exponent` |
| `quality_exponent` | `1.5` | Output `*= (quality/5) ** quality_exponent` |

> Higher `quality_exponent` = steeper penalty for low-quality planets.

---

### `DEFAULT_RESOURCE_CONFIGS`
`dict[ResourceType, ResourceConfig]` — tuned baseline values for all four resource types. Intended as the starting point; buildings/factions override per planet via `Planet.resource_configs`.

| Resource | Distribution | Rationale |
|----------|-------------|-----------|
| `MINERALS` | `lognormal` | Right-skewed — occasional huge hoards |
| `ENERGY` | *(default)* | Standard |
| `ORGANICS` | `beta` | Bounded, flexible shape |
| `RARE_MATS` | `exponential` | Mostly scarce, rare spikes |

---

## Key Functions

### `sample_distribution(dist, mean, std, scale, clip, size, rng)` {#sample_distribution}
Core flexible sampler. Returns a float (or array of floats).

**Supported distributions:**

| Name | Notes |
|------|-------|
| `"uniform"` | |
| `"normal"` | |
| `"lognormal"` | Right-skewed |
| `"exponential"` | Mostly low, rare spikes |
| `"beta"` | Auto-converts `mean`+`std` → alpha/beta params |

**`clip` formats:**
- `(float, float)` — two-sided clamp
- `(float, None)` — one-sided lower clamp
- `None` — no clamping

---

## Types / Aliases

### `DISTRIBUTION`
`Literal["uniform", "normal", "lognormal", "exponential", "beta"]`
Use to type-hint any parameter that accepts a distribution name.

---

## Related
- [[Enums/ResourceType]]
- [[Modules/Solar System]] — uses ResourceConfig per planet
- [[Faction]] — stockpile uses ResourceType keys
- [[CodeBase/Codebase Home|Codebase Home]]
- 
