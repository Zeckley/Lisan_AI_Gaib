---
tags:
  - Information
---
# Import Order

Safe import chain for the `simulation/` package.

```
resources.py          ← no internal imports (leaf)
    ↑
solar_system.py       ← imports from resources.py only
    ↑
faction.py            ← imports from resources.py + solar_system.py
```

**Rule:** imports flow upward only. Never import `faction` from `solar_system` or `resources`.

`Colony` holds a **forward reference** to `Faction` as a string `"Faction"` to allow type hints without a circular import.

→ [[Patterns/Python Patterns]]
→ [[CodeBase/Codebase Home|Codebase Home]]
