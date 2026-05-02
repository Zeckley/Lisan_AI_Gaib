"""
simulation/resources.py
-----------------------
Resource types, configs, and the core distribution sampler.

Exports
-------
ResourceType          - IntEnum of the four resource categories
ResourceConfig        - dataclass: how one resource type is generated
sample_distribution   - flexible numpy-backed sampler
DEFAULT_RESOURCE_CONFIGS - ready-to-use config dict keyed by ResourceType
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional
from enum import IntEnum


# ------------------------------------------------------------------
# DISTRIBUTION HELPER
# ------------------------------------------------------------------

DISTRIBUTION = Literal["normal", "uniform", "lognormal", "exponential", "beta"]


def sample_distribution(
    dist:  DISTRIBUTION,
    low:   float = 0.0,
    high:  float = 1.0,
    mean:  float = 0.5,
    std:   float = 0.1,
    *,
    scale: float = 1.0,
    size:  int   = 1,
    clip:  Optional[tuple[float, float]] = None,
    rng:   Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw `size` samples from a named distribution, then scale and optionally clip.

    Parameters
    ----------
    dist    : which distribution to use
    low/high: bounds for uniform
    mean/std: centre and spread (interpretation varies by distribution — see below)
    scale   : multiply all samples by this value
    size    : number of samples to draw
    clip    : (min, max) hard clamp; None = no clamping
    rng     : numpy Generator for reproducibility (created fresh if None)

    Returns
    -------
    np.ndarray of shape (size,)

    Distribution quick-reference
    ----------------------------
    uniform     -> flat chance across [low, high].
                   Good for: resources with no preferred value.
    normal      -> bell curve around mean with spread std.
                   Good for: "typical" quantities.
    lognormal   -> right-skewed, always positive; mean/std are the
                   underlying normal params, not the output moments.
                   Good for: wealth/resource hoards — most planets
                   have little but a few have enormous reserves.
    exponential -> heavy right tail, peaks at 0; mean = 1/lambda.
                   Good for: rare windfalls.
    beta        -> flexible [0,1] shape via mean+std; internally
                   converts to alpha/beta params.
                   Good for: quality/efficiency scores.
    """
    rng = rng or np.random.default_rng()

    match dist:
        case "uniform":
            samples = rng.uniform(low, high, size=size)

        case "normal":
            samples = rng.normal(mean, std, size=size)

        case "lognormal":
            samples = rng.lognormal(mean, std, size=size)

        case "exponential":
            samples = rng.exponential(mean, size=size)

        case "beta":
            var = std ** 2
            var = min(var, mean * (1 - mean) - 1e-6)   # keep params valid
            alpha = mean * (mean * (1 - mean) / var - 1)
            beta  = (1 - mean) * (mean * (1 - mean) / var - 1)
            alpha = max(alpha, 0.01)
            beta  = max(beta,  0.01)
            samples = rng.beta(alpha, beta, size=size)

        case _:
            raise ValueError(f"Unknown distribution: {dist!r}")

    samples = samples * scale
    if clip is not None:
        lo, hi = clip
        samples = np.clip(samples, lo, hi if hi is not None else np.inf)

    return samples


# ------------------------------------------------------------------
# RESOURCE TYPES
# ------------------------------------------------------------------

class ResourceType(IntEnum):
    MINERALS  = 0   # stone, ore, metals
    ENERGY    = 1   # coal, geothermal — non-renewable
    ORGANICS  = 2   # plants, animals, lumber
    RARE_MATS = 3   # uranium, platinum, titanium, gold …


# ------------------------------------------------------------------
# RESOURCE CONFIG
# ------------------------------------------------------------------

@dataclass
class ResourceConfig:
    """
    Defines how one resource type is generated per planet.

    Fields
    ------
    dist / mean / std / scale / clip
        Passed directly to sample_distribution.
    size_exponent
        output *= planet_size ** size_exponent
        Larger planets yield proportionally more.
    quality_exponent
        output *= (quality / 5) ** quality_exponent
        Higher values = steeper penalty for low-quality planets.
        quality=0 always forces output to 0 regardless of this value.
    """
    dist:             DISTRIBUTION                   = "lognormal"
    mean:             float                          = 1.0
    std:              float                          = 0.5
    scale:            float                          = 100.0
    clip:             Optional[tuple[float, float]]  = (0.0, None)

    size_exponent:    float = 1.2
    quality_exponent: float = 1.5


# ------------------------------------------------------------------
# DEFAULT CONFIGS
# ------------------------------------------------------------------

DEFAULT_RESOURCE_CONFIGS: dict[ResourceType, ResourceConfig] = {
    ResourceType.MINERALS: ResourceConfig(
        dist="lognormal",   mean=1.0, std=0.6,  scale=500.0,
        size_exponent=1.3,  quality_exponent=1.2,
    ),
    ResourceType.ENERGY: ResourceConfig(
        dist="exponential", mean=1.0, std=0.5,  scale=250.0,
        size_exponent=1.0,  quality_exponent=1.8,
    ),
    ResourceType.ORGANICS: ResourceConfig(
        dist="beta",        mean=0.3, std=0.15, scale=350.0,
        size_exponent=1.1,  quality_exponent=2.0,
    ),
    ResourceType.RARE_MATS: ResourceConfig(
        dist="exponential", mean=0.4, std=0.3,  scale=50.0,
        size_exponent=0.8,  quality_exponent=2.5,
    ),
}
