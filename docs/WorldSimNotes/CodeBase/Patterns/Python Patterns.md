---
tags:
  - Information
---
# Python Patterns

Conventions used throughout this project.

---

## `@dataclass` + `__post_init__`

```python
@dataclass
class Foo:
    x: float
    y: float = field(init=False)  # computed, not a constructor arg

    def __post_init__(self):
        self.y = self.x * 2
```

- Auto-generates `__init__`, `__repr__`, `__eq__`
- `__post_init__` runs after field assignment — use for derived attributes
- `field(init=False)` for computed fields that shouldn't be constructor args

---

## `IntEnum`

```python
class ResourceType(IntEnum):
    MINERALS = 0
    ENERGY   = 1
```

- Named integer constants
- Iterable, numpy-indexable, JSON-serialisable as ints
- Validated on construction
- Easy to extend — add a line, all loops/dicts adapt
- Preferred over bare ints or string constants for any fixed categorical set

---

## `np.random.default_rng(seed)`

```python
rng = np.random.default_rng(42)
# pass rng through as parameter, never use global state
result = sample_distribution(..., rng=rng)
```

- Reproducible randomness without global state
- Pass `rng` as a parameter through the call chain
- All sampling functions accept `rng` as an argument

---

## Import Order (simulation/)

```
resources.py        ← leaf, no internal imports
    ↑
solar_system.py     ← imports resources only
    ↑
faction.py          ← imports resources + solar_system
```

Keep imports flowing upward only. `faction.py` referencing `Colony → system` uses a string forward reference `"Faction"` to avoid circular imports.

→ [[CodeBase/Codebase Home|Codebase Home]]
