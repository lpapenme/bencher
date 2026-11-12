"""Lennard-Jones cluster energy minimisation.

Find the arrangement of N identical atoms that minimises the total pair energy

    E(r_1, ..., r_N) = sum_{i<j} 4 * (r_ij^-12 - r_ij^-6)        (epsilon = sigma = 1)

A classic global-optimisation problem (Wales & Doye, 1997): the number of local
minima grows exponentially with N, the putative global minima are tabulated in
the Cambridge Cluster Database, and the optima are interior -- no coordinate of
an optimal cluster sits on a bound.

Search space
------------
``lj-N`` has 3N continuous dimensions in [0, 1], ordered (x_1, y_1, z_1, x_2, ...).
Each coordinate is scaled to a cubic box of side ``box_side(N) = 1.5 * N^(1/3)``.
The box holds the optimal cluster in any orientation with room to translate
(optimal diameters: 1.82, 2.16, 3.50 and 4.28 for N = 7, 13, 38, 55), so the
global minimum is a continuum of interior points (translations, rotations and
the N! relabellings of the atoms).

Soft core
---------
The raw potential diverges as r -> 0, so uniformly random points give energies
of 1e10 and beyond, which swamps any surrogate model and says nothing about the
landscape. Inside the core radius (0.8) the pair potential is therefore
continued linearly with the value and slope it has at the core radius (C^1).
Every pair in the tabulated global minima is further apart than 1.05, so the
minima, their energies and the landscape around them are exactly those of the
unmodified potential; only the unphysical overlapping region is tamed. Two atoms
on the same spot cost about 650 instead of infinity.
"""

from __future__ import annotations

import numpy as np

# Benchmark name -> energy of the putative global minimum, from the Cambridge
# Cluster Database (https://www-wales.ch.cam.ac.uk/CCD.html). lj-13 and lj-55
# are Mackay icosahedra; lj-38 is the fcc truncated octahedron, the textbook
# double-funnel landscape whose global minimum sits in the narrower funnel.
GLOBAL_MINIMA = {
    "lj-7": -16.505384,
    "lj-13": -44.326801,
    "lj-38": -173.928427,
    "lj-55": -279.248470,
}

R_CORE = 0.8


def _pair(r: np.ndarray) -> np.ndarray:
    s6 = r**-6
    return 4.0 * (s6 * s6 - s6)


_V_CORE = float(_pair(np.array(R_CORE)))
_DV_CORE = 4.0 * (-12.0 * R_CORE**-13 + 6.0 * R_CORE**-7)


def box_side(n_atoms: int) -> float:
    """Side length of the cubic box the unit cube is scaled to."""
    return 1.5 * n_atoms ** (1.0 / 3.0)


def energy(positions: np.ndarray) -> float:
    """Soft-core Lennard-Jones energy of an (N, 3) array of positions."""
    diff = positions[:, None, :] - positions[None, :, :]
    i, j = np.triu_indices(len(positions), k=1)
    r = np.sqrt((diff[i, j] ** 2).sum(-1))
    outside = r >= R_CORE
    e = np.empty_like(r)
    e[outside] = _pair(r[outside])
    e[~outside] = _V_CORE + _DV_CORE * (r[~outside] - R_CORE)
    return float(e.sum())


def evaluate(name: str, x: np.ndarray) -> float:
    """Energy of the cluster encoded by ``x`` in [0, 1]^(3N); to be minimised."""
    n_atoms = int(name.split("-")[1])
    x = np.asarray(x, dtype=float).ravel()
    if x.shape != (3 * n_atoms,):
        raise ValueError(
            f"Input shape {x.shape} does not match expected shape {(3 * n_atoms,)}"
        )
    if not np.all((x >= 0.0) & (x <= 1.0)):
        raise ValueError(f"{name} expects every coordinate in [0, 1]")
    return energy(x.reshape(n_atoms, 3) * box_side(n_atoms))
