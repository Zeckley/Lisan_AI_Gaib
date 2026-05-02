"""
simulation/solar_system.py
--------------------------
Planet and SolarSystem dataclasses.

Imports everything it needs from resources.py so neither file
has to be aware of the other's internals.

Exports
-------
Planet       - single planet with generated resource values
SolarSystem  - collection of planets at a position in space
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from resources import (
    ResourceType,
    ResourceConfig,
    DEFAULT_RESOURCE_CONFIGS,
    sample_distribution,
)


# ------------------------------------------------------------------
# PLANET
# ------------------------------------------------------------------

@dataclass
class Planet:
    """
    A single planet whose resource values are generated on construction.

    Parameters
    ----------
    size    : float in [0.1, 10.0]  — relative planet size
    quality : float in [0, 5]       — overall resource richness score
    rng     : shared Generator for reproducibility
    resource_configs : mapping of ResourceType -> ResourceConfig;
              defaults to DEFAULT_RESOURCE_CONFIGS
    """
    size:             float
    quality:          float
    rng:              np.random.Generator           = field(default_factory=np.random.default_rng, repr=False)
    resource_configs: dict[ResourceType, ResourceConfig] = field(
        default_factory=lambda: DEFAULT_RESOURCE_CONFIGS, repr=False
    )

    resources: dict[ResourceType, float] = field(init=False)

    def __post_init__(self) -> None:
        self.resources = self._generate_resources()

    def _generate_resources(self) -> dict[ResourceType, float]:
        out: dict[ResourceType, float] = {}
        q_norm = self.quality / 5.0   # normalise quality to [0, 1]

        for rtype, cfg in self.resource_configs.items():
            base = sample_distribution(
                dist  = cfg.dist,
                mean  = cfg.mean,
                std   = cfg.std,
                scale = cfg.scale,
                size  = 1,
                clip  = cfg.clip,
                rng   = self.rng,
            )[0]

            value = base * (self.size ** cfg.size_exponent) * (q_norm ** cfg.quality_exponent)
            out[rtype] = round(value if q_norm > 0 else 0.0, 2)

        return out

    def summary(self) -> str:
        lines = [f"  Planet  size={self.size:.2f}  quality={self.quality:.1f}/5"]
        for rtype, val in self.resources.items():
            lines.append(f"    {rtype.name:<12} {val:>10.2f}")
        return "\n".join(lines)


# ------------------------------------------------------------------
# SOLAR SYSTEM
# ------------------------------------------------------------------

@dataclass
class SolarSystem:
    """
    A collection of planets at a fixed position in simulation space.

    Parameters
    ----------
    position     : (x, y, z) in arbitrary space units
    n_planets    : number of planets to generate
    seed         : optional int for full reproducibility
    quality_dist : kwargs forwarded to sample_distribution for planet quality scores
    size_dist    : kwargs forwarded to sample_distribution for planet sizes
    """
    position:     tuple[float, float, float]
    n_planets:    int          = 5
    seed:         Optional[int] = None
    quality_dist: dict = field(default_factory=lambda: dict(
        dist="beta", mean=0.4, std=0.2, scale=5.0, clip=(0.0, 5.0)
    ))
    size_dist: dict = field(default_factory=lambda: dict(
        dist="uniform", low=0.1, high=10.0
    ))

    planets: list[Planet]          = field(init=False)
    rng:     np.random.Generator   = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng     = np.random.default_rng(self.seed)
        self.planets = self._generate_planets()

    def _generate_planets(self) -> list[Planet]:
        qualities = sample_distribution(**self.quality_dist, size=self.n_planets, rng=self.rng)
        sizes     = sample_distribution(**self.size_dist,    size=self.n_planets, rng=self.rng)

        return [
            Planet(size=float(s), quality=float(q), rng=self.rng)
            for s, q in zip(sizes, qualities)
        ]

    def total_resources(self) -> dict[ResourceType, float]:
        totals: dict[ResourceType, float] = {r: 0.0 for r in ResourceType}
        for planet in self.planets:
            for rtype, val in planet.resources.items():
                totals[rtype] += val
        return totals

    def summary(self) -> str:
        lines = [f"SolarSystem @ {self.position}  ({len(self.planets)} planets)"]
        for i, p in enumerate(self.planets):
            lines.append(f"  -- Planet {i + 1} --")
            lines.append(p.summary())
        lines.append("  -- System Totals --")
        for rtype, val in self.total_resources().items():
            lines.append(f"    {rtype.name:<12} {val:>10.2f}")
        return "\n".join(lines)
