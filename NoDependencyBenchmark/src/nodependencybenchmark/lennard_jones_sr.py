"""Symmetry-reduced Lennard-Jones clusters.

Same energy as ``lennard_jones`` (soft core included), but parametrised so that
the continuous symmetries and most of the relabelling symmetry of the cluster
are removed from the search space. ``lj-N`` has a continuum of global minima
(translations, rotations) repeated N! times (relabellings); ``lj-N-sr`` keeps one
representative of each translated/rotated copy and orders most of the atoms.
The pair is meant for asking how much of the difficulty of ``lj-N`` is its
symmetry and how much is the narrow, interior, multi-funnel landscape itself.

Search space: ``lj-N-sr`` has 3N - 6 continuous dimensions in [0, 1]
--------------------------------------------------------------------
With ``s = box_side(N)`` and the box centred at the origin:

* atom 1 is fixed at the origin (removes translations);
* atom 2 lies on the positive x-axis: one variable, ``x in [0, s/2]``;
* atom 3 lies in the half-plane z = 0, y >= 0: two variables,
  ``x in [-s/2, s/2]``, ``y in [0, s/2]`` (removes rotations);
* atoms 4..N are free. Their coordinates are expressed in a fixed, generic
  rotated frame ``q' = q R^T`` (``R = FREE_FRAME``), where each lies in
  ``[-s/2, s/2]^3``. Their second and third rotated coordinates are 2(N - 3)
  variables scaled to ``[-s/2, s/2]``. Their first rotated coordinates are given
  by N - 3 variables that are mapped onto *sorted* positions with the
  order-statistics transform (``t_m = u_m^(1/m)``, ``t_k = t_{k+1} u_k^(1/k)``), a
  bijection from the unit cube onto ``0 <= t_1 <= ... <= t_m <= 1``. Atoms 4..N
  are therefore listed in order of increasing first rotated coordinate, which
  removes their (N - 3)! relabellings.

Why the rotated frame: sorting along the canonical x-axis puts every symmetric
cluster's minimum on a bound of the cube. Mirror images across the plane of
atoms 1-3 share their x-coordinate, a tie in the sorted sequence, which the
order-statistics transform maps to ``u_k = 1``. For the icosahedral and
octahedral minima of LJ-13, LJ-38 and LJ-55 every choice of atoms 1-3 lies in a
mirror plane, so the tie cannot be avoided by relabelling. Sorting along a
generic direction that is not fixed by that reflection removes the ties.

Variable order: [x_2, x_3, y_3, u_1..u_m, y_4, z_4, ..., y_N, z_N], m = N - 3.
What remains: the choice of which atoms play the roles 1, 2, 3 and a mirror
image, a finite set of copies instead of a continuum times N!.
"""

from __future__ import annotations

import numpy as np

from nodependencybenchmark import lennard_jones

GLOBAL_MINIMA = {
    "lj-13-sr": lennard_jones.GLOBAL_MINIMA["lj-13"],
    "lj-38-sr": lennard_jones.GLOBAL_MINIMA["lj-38"],
}


def _rotation_from_euler(a: float, b: float, c: float) -> np.ndarray:
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    rz1 = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rz2 = np.array([[cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0]])
    return rz1 @ ry @ rz2


# Fixed generic frame for atoms 4..N (ZYZ Euler angles in radians; no special
# relation to any cluster symmetry). Part of the benchmark definition.
FREE_FRAME = _rotation_from_euler(0.3, 0.7, 1.1)


def n_atoms(name: str) -> int:
    return int(name.split("-")[1])


def dimensions(name: str) -> int:
    return 3 * n_atoms(name) - 6


def sorted_from_unit(u: np.ndarray) -> np.ndarray:
    """Order-statistics transform: unit cube -> sorted points in [0, 1]."""
    m = len(u)
    t = np.empty(m)
    upper = 1.0
    for k in range(m, 0, -1):
        upper = upper * u[k - 1] ** (1.0 / k)
        t[k - 1] = upper
    return t


def unit_from_sorted(t: np.ndarray) -> np.ndarray:
    """Inverse of ``sorted_from_unit`` for strictly positive sorted ``t``."""
    m = len(t)
    u = np.empty(m)
    upper = 1.0
    for k in range(m, 0, -1):
        u[k - 1] = (t[k - 1] / upper) ** k
        upper = t[k - 1]
    return u


def positions(name: str, x: np.ndarray) -> np.ndarray:
    """(N, 3) atom positions, box centred at the origin, for ``x`` in [0, 1]^(3N-6)."""
    n = n_atoms(name)
    m = n - 3
    s = lennard_jones.box_side(n)
    x = np.asarray(x, dtype=float).ravel()
    pos = np.zeros((n, 3))
    pos[1, 0] = 0.5 * s * x[0]
    pos[2, 0] = s * (x[1] - 0.5)
    pos[2, 1] = 0.5 * s * x[2]
    free = np.empty((m, 3))
    free[:, 0] = s * (sorted_from_unit(x[3 : 3 + m]) - 0.5)
    free[:, 1:] = s * (x[3 + m :].reshape(m, 2) - 0.5)
    pos[3:] = free @ FREE_FRAME  # back from the rotated frame (q' = q R^T)
    return pos


def encode(name: str, cluster: np.ndarray) -> np.ndarray:
    """Point of the unit cube that represents ``cluster`` (N, 3), using its first
    three atoms for the roles 1, 2, 3. Raises if the cluster does not fit."""
    n = n_atoms(name)
    s = lennard_jones.box_side(n)
    p = np.asarray(cluster, dtype=float) - cluster[0]
    e1 = p[1] / np.linalg.norm(p[1])
    v = p[2] - (p[2] @ e1) * e1
    e2 = v / np.linalg.norm(v)
    e3 = np.cross(e1, e2)
    q = p @ np.stack([e1, e2, e3]).T
    free = q[3:] @ FREE_FRAME.T
    rest = free[np.argsort(free[:, 0])]
    x = np.concatenate(
        [
            [q[1, 0] / (0.5 * s), q[2, 0] / s + 0.5, q[2, 1] / (0.5 * s)],
            unit_from_sorted(rest[:, 0] / s + 0.5),
            (rest[:, 1:] / s + 0.5).ravel(),
        ]
    )
    if not np.all((x >= 0.0) & (x <= 1.0)):
        raise ValueError("cluster does not fit the symmetry-reduced box")
    return x


def evaluate(name: str, x: np.ndarray) -> float:
    """Energy of the cluster encoded by ``x``; to be minimised."""
    x = np.asarray(x, dtype=float).ravel()
    d = dimensions(name)
    if x.shape != (d,):
        raise ValueError(f"Input shape {x.shape} does not match expected shape {(d,)}")
    if not np.all((x >= 0.0) & (x <= 1.0)):
        raise ValueError(f"{name} expects every coordinate in [0, 1]")
    return lennard_jones.energy(positions(name, x))
