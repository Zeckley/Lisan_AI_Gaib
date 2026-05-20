---
tags:
  - Module
---
# Terrain

**File:** `world/terrain.py`
**Depends on:** nothing (leaf module)

---

## Purpose
Defines terrain type constants and enums used by the world mesh. Kept separate from `world_map.py` so terrain logic can be imported without pulling in PyVista.

---

## Terrain Types (`face_type` int array on mesh)

| Value | Name | Color |
|-------|------|-------|
| `0` | `WATER` | dodgerblue |
| `1` | `LAND` | forestgreen |
| `2` | `BEACH` | sandybrown |
| `3` | `MOUNTAIN` | white |

The `face_type` array lives on the PyVista mesh object and drives both 3D and 2D colormap rendering.

---

## Notes
- Values are simple integers, compatible with numpy indexing
- See [[Enums/TerrainType]] for the enum wrapper
- Terrain assignment logic lives in [[Modules/World Map#terrain generation]]

---

## Related
- [[Modules/World Map]]
- [[Enums/TerrainType]]
- [[CodeBase/Codebase Home|Codebase Home]]
