---
tags:
  - Constant
---
# Overview

**Module:** [[Modules/Resources]] (`simulation/resources.py`)
**Type:** `IntEnum`

| Value | Name |
|-------|------|
| `0` | `MINERALS` |
| `1` | `ENERGY` |
| `2` | `ORGANICS` |
| `3` | `RARE_MATS` |

**Why IntEnum?** Named constants that are iterable, numpy/JSON compatible, and validated on construction. Easy to extend — add a line and all loops/dicts adapt automatically.

Default distributions: MINERALS=lognormal, ORGANICS=beta, RARE_MATS=exponential.

→ [[Modules/Resources#DEFAULT_RESOURCE_CONFIGS]]
