---
tags:
  - Class
---
# World Map

**File:** `world/world_map.py`
**Depends on:** [[Terrain]], PyVista, scipy

---

## Purpose
Generates a spherical planet mesh using a PyVista `Icosphere`, oriented with poles on the Z axis. Supports both 3D and 2D flat projection views.

---

## Key Functions / Methods

### `rotation_matrix_to_z`
- Finds the antipodal vertex pair via `scipy.spatial.distance.cdist`
- Returns a rotation matrix that aligns poles to the Z axis
- Called once at mesh creation time

### `plot_sphere(static=False)`
- Renders the 3D icosphere colored by `face_type` terrain
- `static=True` → PyVista `notebook=True`, outputs inline PNG (no trame needed)
- Colormap: `dodgerblue / forestgreen / sandybrown / white`

### `plot_map(static=False)`
- 2D equirectangular (flat) projection using spherical coordinates (θ, φ)
- Seam-crossing triangles culled: any face where `max(φ) - min(φ) > π` is dropped

---

## Terrain Generation — Wave Collapse Style {#terrain generation}

1. Start at `nsub=1` (coarsest), seed faces as land/water using `land_fraction`
2. Each subdivision: split every triangle into 4 children (midpoints projected back onto unit sphere); children inherit parent terrain type
3. After each subdivision: BFS face adjacency border smoothing — border cells probabilistically flipped toward water using `water_bias`
4. **Secondary pass** after full subdivision:
   - BFS computes shore-distance for all land cells
   - Within `beach_depth` hops → `BEACH`
   - Beyond `mountain_depth` hops → `MOUNTAIN`
   - Landlocked/unreachable cells → `MOUNTAIN`

---

## Config Keys

| Key | Default | Effect |
|-----|---------|--------|
| `nu` | `5` | Subdivision count (more = finer mesh) |
| `polar_radius` | `6378137.0` | Earth-scale polar radius |
| `equator_radius` | `6356752.3` | Earth-scale equatorial radius |
| `water_bias` | `0.7` | Border flip weight (0.5=neutral, 1.0=all borders→water) |
| `land_fraction` | `0.3` | Seed probability at nsub=1 |
| `beach_depth` | `2` | Hops from shore → beach |
| `mountain_depth` | `6` | Hops from shore → mountain |

---

## Jupyter Notes
- Set `pv.set_jupyter_backend('static')` in its own cell **before** any PyVista code
- Or install `trame-vtk`, `trame-vuetify`, `nest_asyncio2` for interactive widget

---

## Related
- [[Terrain]] — terrain type constants
- [[Enums/TerrainType]]
- [[CodeBase/Codebase Home|Codebase Home]]
- 
