"""Reusable geometry builders for common particle-spring structures."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .simulation import Particle, Simulation


def add_chain(
    sim: Simulation,
    n: int,
    start: Sequence[float],
    end: Sequence[float],
    mass: float = 1.0,
    k: float = 10_000.0,
    b: Optional[float] = None,
    L0: Optional[float] = None,
    pin_start: bool = False,
    pin_end: bool = False,
    tension_only: bool = False,
) -> list[Particle]:
    """Create a 1D chain of `n` particles connected by springs from `start` to `end`."""
    if n < 2:
        raise ValueError("A chain requires at least 2 particles.")
    p_start = np.asarray(start, dtype=np.float64).reshape(3)
    p_end = np.asarray(end, dtype=np.float64).reshape(3)

    particles: list[Particle] = []
    for i in range(n):
        frac = i / (n - 1)
        pos = p_start + frac * (p_end - p_start)
        movable = not ((i == 0 and pin_start) or (i == n - 1 and pin_end))
        particles.append(sim.add_particle(mass=mass, position=pos, movable=movable))

    for i in range(n - 1):
        sim.add_spring(
            particles[i],
            particles[i + 1],
            k=k,
            L0=L0,
            b=b,
            tension_only=tension_only,
        )
    return particles


def add_grid_2d(
    sim: Simulation,
    nx: int,
    ny: int,
    lx: float = 3.0,
    ly: float = 3.0,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    mass: float = 1.0,
    k: float = 100.0,
    b: Optional[float] = None,
    rest_length_scale: float = 1.0,
    pin_boundary: bool = True,
    pin_corners: bool = False,
    diagonals: bool = False,
    tension_only: bool = False,
) -> list[list[Particle]]:
    """Create an `nx x ny` 2D particle-spring sheet centered at `origin` in the XY plane."""
    if nx < 2 or ny < 2:
        raise ValueError("A 2D grid requires nx >= 2 and ny >= 2.")
    orig = np.asarray(origin, dtype=np.float64).reshape(3)
    dx = lx / (nx - 1)
    dy = ly / (ny - 1)

    grid: list[list[Particle]] = []
    for i in range(nx):
        row: list[Particle] = []
        for j in range(ny):
            on_edge = i == 0 or j == 0 or i == nx - 1 or j == ny - 1
            on_corner = (i in (0, nx - 1)) and (j in (0, ny - 1))
            movable = not ((pin_boundary and on_edge) or (pin_corners and on_corner))
            pos = orig + np.array([-0.5 * lx + i * dx, -0.5 * ly + j * dy, 0.0])
            row.append(sim.add_particle(mass=mass, position=pos, movable=movable))
        grid.append(row)

    l0_x = dx * rest_length_scale
    l0_y = dy * rest_length_scale
    l0_d = float(np.hypot(dx, dy)) * rest_length_scale

    for i in range(nx):
        for j in range(ny):
            if i + 1 < nx:
                sim.add_spring(
                    grid[i][j], grid[i + 1][j], k=k, L0=l0_x, b=b, tension_only=tension_only
                )
            if j + 1 < ny:
                sim.add_spring(
                    grid[i][j], grid[i][j + 1], k=k, L0=l0_y, b=b, tension_only=tension_only
                )
            if diagonals and i + 1 < nx and j + 1 < ny:
                sim.add_spring(
                    grid[i][j], grid[i + 1][j + 1], k=k, L0=l0_d, b=b, tension_only=tension_only
                )
                sim.add_spring(
                    grid[i + 1][j], grid[i][j + 1], k=k, L0=l0_d, b=b, tension_only=tension_only
                )

    return grid


def add_lattice_3d(
    sim: Simulation,
    nx: int,
    ny: int,
    nz: int,
    spacing: float = 1.0,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    mass: float = 1.0,
    k: float = 500.0,
    b: Optional[float] = None,
    connect_face_diagonals: bool = True,
    connect_body_diagonals: bool = False,
) -> list[list[list[Particle]]]:
    """Create an `nx x ny x nz` 3D elastic lattice centered at `origin`."""
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("Lattice dimensions must be >= 1.")
    orig = np.asarray(origin, dtype=np.float64).reshape(3)
    cx = 0.5 * (nx - 1) * spacing
    cy = 0.5 * (ny - 1) * spacing
    cz = 0.5 * (nz - 1) * spacing

    lattice: list[list[list[Particle]]] = []
    for i in range(nx):
        plane: list[list[Particle]] = []
        for j in range(ny):
            row: list[Particle] = []
            for m in range(nz):
                pos = orig + np.array([i * spacing - cx, j * spacing - cy, m * spacing - cz])
                row.append(sim.add_particle(mass=mass, position=pos))
            plane.append(row)
        lattice.append(plane)

    # Offsets covering each unordered pair exactly once (lexicographically positive)
    offsets: list[tuple[int, int, int]] = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    if connect_face_diagonals:
        offsets.extend(
            [
                (1, 1, 0),
                (1, -1, 0),
                (1, 0, 1),
                (1, 0, -1),
                (0, 1, 1),
                (0, 1, -1),
            ]
        )
    if connect_body_diagonals:
        offsets.extend([(1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)])

    for i in range(nx):
        for j in range(ny):
            for m in range(nz):
                p1 = lattice[i][j][m]
                for di, dj, dm in offsets:
                    ni, nj, nm = i + di, j + dj, m + dm
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nm < nz:
                        sim.add_spring(p1, lattice[ni][nj][nm], k=k, b=b)

    return lattice


def connect_all_pairs(
    sim: Simulation,
    particles: Sequence[Particle],
    k: float,
    b: Optional[float] = None,
) -> None:
    """Connect every pair of particles in `particles` with stiff springs.

    Handy for turning a small set of point masses into a pure-spring rigid body
    (such as a gyroscope rotor or Dzhanibekov T-handle).
    """
    n = len(particles)
    for i in range(n - 1):
        for j in range(i + 1, n):
            sim.add_spring(particles[i], particles[j], k=k, b=b)
