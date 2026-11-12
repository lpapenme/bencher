"""Wind-farm layout optimisation with a Gaussian wake model.

Place N identical turbines on a square site so that the wake losses, averaged
over a wind rose, are as small as possible. A classic engineering design problem
(Mosetti et al., 1994) with the same "interchangeable objects" structure as the
Lennard-Jones clusters: relabelling the turbines leaves the objective unchanged,
so every layout has N! equivalent copies in the search space.

Search space
------------
``windfarm-N`` has 2N continuous dimensions in [0, 1], ordered (x_1, y_1, x_2, ...).
Each coordinate is scaled to a square site of side ``site_side(N) = 5 D sqrt(N)``
(D = rotor diameter), i.e. a mean spacing of five rotor diameters.

Objective (to be minimised)
---------------------------
``wake_loss + spacing_penalty`` where

* ``wake_loss = 1 - P / P_free`` in [0, 1): the fraction of the farm's energy
  production lost to wakes, with turbine power proportional to the cube of the
  local wind speed and the expectation taken over the wind rose below;
* ``spacing_penalty = sum_{i<j} max(0, 1 - d_ij / (2 D))``: turbines closer than
  two rotor diameters are penalised linearly (0 for every feasible layout).

Wake model
----------
Bastankhah & Porte-Agel (2014) Gaussian wake: a turbine at streamwise distance
``dx > 0`` and crosswind distance ``dy`` downstream of another sees the relative
velocity deficit

    C(dx) * exp(-dy^2 / (2 sigma(dx)^2)),   sigma(dx) = k* dx + eps D,
    C(dx) = 1 - sqrt(1 - C_T / (8 (sigma / D)^2)),

with C_T = 0.8, k* = 0.04 and eps = 0.2 sqrt(beta), beta = (1 + sqrt(1 - C_T)) /
(2 sqrt(1 - C_T)). Deficits from several upstream turbines are combined by
root-sum-square. The model is smooth in the turbine positions except where two
turbines pass each other in the streamwise direction.

Wind rose
---------
Twelve 30-degree sectors with a prevailing west-south-westerly wind and a
secondary north-easterly lobe (frequencies in ``WIND_ROSE``); one free-stream
speed, which cancels in the loss fraction.
"""

from __future__ import annotations

import numpy as np

SIZES = (10, 30, 50)
NAMES = tuple(f"windfarm-{n}" for n in SIZES)

ROTOR_DIAMETER = 1.0  # all lengths are in rotor diameters
C_T = 0.8
K_STAR = 0.04
MIN_SPACING = 2.0 * ROTOR_DIAMETER

_BETA = 0.5 * (1.0 + np.sqrt(1.0 - C_T)) / np.sqrt(1.0 - C_T)
_EPS = 0.2 * np.sqrt(_BETA)

# direction the wind blows FROM, degrees clockwise from north, and its frequency
WIND_DIRECTIONS = np.arange(0.0, 360.0, 30.0)
WIND_ROSE = np.array(
    [0.04, 0.07, 0.09, 0.05, 0.03, 0.04, 0.07, 0.12, 0.20, 0.15, 0.08, 0.06]
)
assert abs(WIND_ROSE.sum() - 1.0) < 1e-12


def site_side(n_turbines: int) -> float:
    """Side length of the square site, in rotor diameters."""
    return 5.0 * ROTOR_DIAMETER * np.sqrt(n_turbines)


def wake_loss(positions: np.ndarray) -> float:
    """Fraction of the free-stream energy production lost to wakes; (N, 2) positions."""
    n = len(positions)
    power = 0.0
    for direction, freq in zip(WIND_DIRECTIONS, WIND_ROSE):
        theta = np.deg2rad(direction)
        # unit vector the wind travels along (it comes from `direction`)
        wind = -np.array([np.sin(theta), np.cos(theta)])
        cross = np.array([-wind[1], wind[0]])
        s = positions @ wind  # streamwise coordinate
        c = positions @ cross  # crosswind coordinate
        dx = s[None, :] - s[:, None]  # dx[i, j] > 0: j is downstream of i
        dy = c[None, :] - c[:, None]
        down = dx > 1e-9
        sigma = K_STAR * np.where(down, dx, 1.0) + _EPS * ROTOR_DIAMETER
        radicand = np.clip(1.0 - C_T / (8.0 * (sigma / ROTOR_DIAMETER) ** 2), 0.0, 1.0)
        deficit = np.where(
            down, (1.0 - np.sqrt(radicand)) * np.exp(-0.5 * (dy / sigma) ** 2), 0.0
        )
        total = np.sqrt((deficit**2).sum(axis=0))  # deficit seen by each turbine j
        power += freq * ((1.0 - np.clip(total, 0.0, 1.0)) ** 3).sum()
    return float(1.0 - power / n)


def spacing_penalty(positions: np.ndarray) -> float:
    """Linear penalty for pairs closer than two rotor diameters."""
    diff = positions[:, None, :] - positions[None, :, :]
    i, j = np.triu_indices(len(positions), k=1)
    d = np.sqrt((diff[i, j] ** 2).sum(-1))
    return float(np.clip(1.0 - d / MIN_SPACING, 0.0, None).sum())


def evaluate(name: str, x: np.ndarray) -> float:
    """Objective of the layout encoded by ``x`` in [0, 1]^(2N); to be minimised."""
    n = int(name.split("-")[1])
    x = np.asarray(x, dtype=float).ravel()
    if x.shape != (2 * n,):
        raise ValueError(
            f"Input shape {x.shape} does not match expected shape {(2 * n,)}"
        )
    if not np.all((x >= 0.0) & (x <= 1.0)):
        raise ValueError(f"{name} expects every coordinate in [0, 1]")
    positions = x.reshape(n, 2) * site_side(n)
    return wake_loss(positions) + spacing_penalty(positions)
