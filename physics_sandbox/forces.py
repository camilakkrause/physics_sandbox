"""Built-in and custom force laws for Physics Sandbox.

All force laws follow a consistent Newtonian convention: they compute **forces**
(in Newtons, F = m * a) acting on particles, plus optional potential energy so
`Simulation.potential_energy()` reflects total mechanical potential energy.

Built-in forces (Gravity, LinearDrag, QuadraticDrag, Coulomb, NBodyGravity,
UniformEMField, PlaneBoundary) and `HarmonicDriver` are automatically
accelerated inside the Numba JIT kernel when `numba` is installed, with a
vectorized NumPy fallback otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    from .simulation import Particle, Simulation


class Force:
    """Base class for built-in and custom force laws."""

    def accumulate_forces(self, sim: "Simulation", forces: np.ndarray) -> None:
        """Add forces (shape `(N, 3)`) into `forces` in place."""
        raise NotImplementedError

    def potential_energy(self, sim: "Simulation") -> float:
        """Return potential energy contribution (0.0 for non-conservative forces)."""
        return 0.0


class Gravity(Force):
    """Uniform gravitational field: F_i = m_i * g."""

    def __init__(self, g: Sequence[float] = (0.0, 0.0, -9.81)) -> None:
        self._g = np.asarray(g, dtype=np.float64).reshape(3).copy()

    @property
    def g(self) -> np.ndarray:
        return self._g

    @g.setter
    def g(self, value: Sequence[float]) -> None:
        self._g[:] = np.asarray(value, dtype=np.float64).reshape(3)

    def accumulate_forces(self, sim: "Simulation", forces: np.ndarray) -> None:
        forces += sim.masses[:, None] * self._g[None, :]

    def potential_energy(self, sim: "Simulation") -> float:
        # U = -sum(m_i * (g . r_i)) over movable particles
        if sim.n_particles == 0:
            return 0.0
        movable = sim.movable
        if not np.any(movable):
            return 0.0
        return float(-np.sum(sim.masses[movable] * (sim.positions[movable] @ self._g)))


class LinearDrag(Force):
    """Linear viscous drag: F_i = -gamma * v_i."""

    def __init__(self, gamma: float = 0.1) -> None:
        self.gamma = float(gamma)

    def accumulate_forces(self, sim: "Simulation", forces: np.ndarray) -> None:
        if self.gamma != 0.0:
            forces -= self.gamma * sim.velocities


class QuadraticDrag(Force):
    """Quadratic aerodynamic drag: F_i = -c * |v_i| * v_i."""

    def __init__(self, c: float = 0.01) -> None:
        self.c = float(c)

    def accumulate_forces(self, sim: "Simulation", forces: np.ndarray) -> None:
        if self.c != 0.0:
            speeds = np.linalg.norm(sim.velocities, axis=1, keepdims=True)
            forces -= self.c * speeds * sim.velocities


class Coulomb(Force):
    """Pairwise inverse-square electrostatic force.

    F_ij = k_e * q_i * q_j / (r^2 + softening^2)^(3/2) * (r_i - r_j)
    """

    def __init__(self, k_e: float = 1.0, softening: float = 1e-6) -> None:
        self.k_e = float(k_e)
        self.softening = float(softening)

    def accumulate_forces(self, sim: "Simulation", forces: np.ndarray) -> None:
        n = sim.n_particles
        if n < 2 or self.k_e == 0.0:
            return
        pos = sim.positions
        q = sim.charges
        eps2 = self.softening * self.softening
        # Vectorized pairwise difference: diff[i, j] = pos[i] - pos[j]
        diff = pos[:, None, :] - pos[None, :, :]
        dist2 = np.sum(diff * diff, axis=2) + eps2
        np.fill_diagonal(dist2, 1.0)
        inv_dist3 = dist2 ** (-1.5)
        np.fill_diagonal(inv_dist3, 0.0)
        coeff = (self.k_e * q[:, None] * q[None, :]) * inv_dist3
        forces += np.sum(coeff[:, :, None] * diff, axis=1)

    def potential_energy(self, sim: "Simulation") -> float:
        n = sim.n_particles
        if n < 2 or self.k_e == 0.0:
            return 0.0
        pos = sim.positions
        q = sim.charges
        eps2 = self.softening * self.softening
        idx_i, idx_j = np.triu_indices(n, k=1)
        diff = pos[idx_i] - pos[idx_j]
        dist = np.sqrt(np.sum(diff * diff, axis=1) + eps2)
        return float(np.sum(self.k_e * q[idx_i] * q[idx_j] / dist))


class NBodyGravity(Force):
    """Pairwise Newtonian gravitation between all particles.

    F_ij = -G * m_i * m_j / (r^2 + softening^2)^(3/2) * (r_i - r_j)
    """

    def __init__(self, G: float = 1.0, softening: float = 1e-6) -> None:
        self.G = float(G)
        self.softening = float(softening)

    def accumulate_forces(self, sim: "Simulation", forces: np.ndarray) -> None:
        n = sim.n_particles
        if n < 2 or self.G == 0.0:
            return
        pos = sim.positions
        m = sim.masses
        eps2 = self.softening * self.softening
        diff = pos[:, None, :] - pos[None, :, :]
        dist2 = np.sum(diff * diff, axis=2) + eps2
        np.fill_diagonal(dist2, 1.0)
        inv_dist3 = dist2 ** (-1.5)
        np.fill_diagonal(inv_dist3, 0.0)
        coeff = (-self.G * m[:, None] * m[None, :]) * inv_dist3
        forces += np.sum(coeff[:, :, None] * diff, axis=1)

    def potential_energy(self, sim: "Simulation") -> float:
        n = sim.n_particles
        if n < 2 or self.G == 0.0:
            return 0.0
        pos = sim.positions
        m = sim.masses
        eps2 = self.softening * self.softening
        idx_i, idx_j = np.triu_indices(n, k=1)
        diff = pos[idx_i] - pos[idx_j]
        dist = np.sqrt(np.sum(diff * diff, axis=1) + eps2)
        return float(-np.sum(self.G * m[idx_i] * m[idx_j] / dist))


class UniformEMField(Force):
    """Uniform electromagnetic Lorentz force: F_i = q_i * (E + v_i x B)."""

    def __init__(
        self,
        E: Sequence[float] = (0.0, 0.0, 0.0),
        B: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self._E = np.asarray(E, dtype=np.float64).reshape(3).copy()
        self._B = np.asarray(B, dtype=np.float64).reshape(3).copy()

    @property
    def E(self) -> np.ndarray:
        return self._E

    @E.setter
    def E(self, value: Sequence[float]) -> None:
        self._E[:] = np.asarray(value, dtype=np.float64).reshape(3)

    @property
    def B(self) -> np.ndarray:
        return self._B

    @B.setter
    def B(self, value: Sequence[float]) -> None:
        self._B[:] = np.asarray(value, dtype=np.float64).reshape(3)

    def accumulate_forces(self, sim: "Simulation", forces: np.ndarray) -> None:
        q = sim.charges[:, None]
        forces += q * (self._E[None, :] + np.cross(sim.velocities, self._B[None, :]))

    def potential_energy(self, sim: "Simulation") -> float:
        if sim.n_particles == 0 or not np.any(self._E):
            return 0.0
        movable = sim.movable
        if not np.any(movable):
            return 0.0
        return float(-np.sum(sim.charges[movable] * (sim.positions[movable] @ self._E)))


class PlaneBoundary(Force):
    """Pure Newtonian spring-damper contact plane (floor or wall).

    When a particle penetrates the half-space `dot(normal, pos) < offset`,
    a linear spring + normal damping restoring force pushes it back.
    """

    def __init__(
        self,
        normal: Sequence[float] = (0.0, 0.0, 1.0),
        offset: float = 0.0,
        stiffness: float = 1e5,
        damping: float = 10.0,
    ) -> None:
        n = np.asarray(normal, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(n)
        if norm == 0.0:
            raise ValueError("PlaneBoundary normal must be non-zero.")
        self._normal = n / norm
        self.offset = float(offset)
        self.stiffness = float(stiffness)
        self.damping = float(damping)

    @property
    def normal(self) -> np.ndarray:
        return self._normal

    def accumulate_forces(self, sim: "Simulation", forces: np.ndarray) -> None:
        signed_dist = sim.positions @ self._normal - self.offset
        penetration = -signed_dist
        mask = penetration > 0.0
        if not np.any(mask):
            return
        v_n = sim.velocities[mask] @ self._normal
        f_mag = self.stiffness * penetration[mask] - self.damping * v_n
        f_mag = np.maximum(f_mag, 0.0)
        forces[mask] += f_mag[:, None] * self._normal[None, :]

    def potential_energy(self, sim: "Simulation") -> float:
        signed_dist = sim.positions @ self._normal - self.offset
        penetration = np.maximum(-signed_dist, 0.0)
        return float(0.5 * self.stiffness * np.sum(penetration * penetration))


class CustomField(Force):
    """Wraps a user-defined Python force or acceleration function."""

    def __init__(
        self,
        fn: Callable,
        *,
        vectorized: bool = False,
        returns_acceleration: bool = False,
        potential: Optional[Callable[["Simulation"], float]] = None,
    ) -> None:
        self.fn = fn
        self.vectorized = vectorized
        self.returns_acceleration = returns_acceleration
        self._potential_fn = potential

    def accumulate_forces(self, sim: "Simulation", forces: np.ndarray) -> None:
        if self.vectorized:
            out = np.asarray(self.fn(sim), dtype=np.float64)
            if self.returns_acceleration:
                forces += sim.masses[:, None] * out
            else:
                forces += out
        else:
            masses = sim.masses
            for i, p in enumerate(sim.particles):
                if not p.movable:
                    continue
                val = np.asarray(self.fn(p), dtype=np.float64)
                if self.returns_acceleration:
                    forces[i] += masses[i] * val
                else:
                    forces[i] += val

    def potential_energy(self, sim: "Simulation") -> float:
        if self._potential_fn is None:
            return 0.0
        return float(self._potential_fn(sim))


@dataclass
class HarmonicDriver:
    """Sinusoidally drives a particle along an axis: pos = base + amp * sin(2*pi*f*t + phase).

    Because this driver is declarative, it runs directly inside the compiled
    Numba substep loop with zero Python callback overhead.
    """

    particle_index: int
    axis: np.ndarray
    amplitude: float
    frequency: float
    phase: float = 0.0
    base_position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def create(
        cls,
        particle: "Particle",
        axis: Sequence[float] = (0.0, 0.0, 1.0),
        amplitude: float = 0.1,
        frequency: float = 1.0,
        phase: float = 0.0,
    ) -> "HarmonicDriver":
        ax = np.asarray(axis, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(ax)
        if norm == 0.0:
            raise ValueError("HarmonicDriver axis must be non-zero.")
        return cls(
            particle_index=particle.index,
            axis=ax / norm,
            amplitude=float(amplitude),
            frequency=float(frequency),
            phase=float(phase),
            base_position=particle.pos.copy(),
        )

    def apply(self, pos: np.ndarray, vel: np.ndarray, t: float) -> None:
        omega = 2.0 * np.pi * self.frequency
        arg = omega * t + self.phase
        disp = self.amplitude * np.sin(arg)
        speed = self.amplitude * omega * np.cos(arg)
        pos[self.particle_index] = self.base_position + disp * self.axis
        vel[self.particle_index] = speed * self.axis
