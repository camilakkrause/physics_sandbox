"""Core Newtonian particle-spring simulation engine (Structure-of-Arrays).

Provides a beginner-friendly object interface (`Particle`, `Spring`, `Simulation`)
backed by contiguous NumPy arrays and optional Numba JIT kernels.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np

from .forces import (
    Coulomb,
    CustomField,
    Force,
    Gravity,
    HarmonicDriver,
    LinearDrag,
    NBodyGravity,
    PlaneBoundary,
    QuadraticDrag,
    UniformEMField,
)
from .kernels import (
    HAS_NUMBA,
    INTEGRATOR_EULER,
    INTEGRATOR_MAP,
    INTEGRATOR_VERLET,
    INTEGRATOR_YOSHIDA4,
    YOSHIDA_C,
    YOSHIDA_D,
    accumulate_spring_forces_numpy,
    step_numba,
)

_BUILTIN_FORCE_TYPES = (
    Gravity,
    LinearDrag,
    QuadraticDrag,
    Coulomb,
    NBodyGravity,
    UniformEMField,
    PlaneBoundary,
)


def vnorm(x: np.ndarray) -> float:
    """Euclidean norm of a 1D vector."""
    return float(np.sqrt(np.dot(x, x)))


def vnorm2(x: np.ndarray) -> float:
    """Squared Euclidean norm of a 1D vector."""
    return float(np.dot(x, x))


class Particle:
    """Lightweight handle pointing to a single particle inside a `Simulation`."""

    def __init__(self, sim: "Simulation", index: int, properties=None) -> None:
        self._sim = sim
        self.index = index
        if callable(properties):
            properties(self)
        elif isinstance(properties, dict):
            for k, v in properties.items():
                setattr(self, k, v)

    @property
    def pos(self) -> np.ndarray:
        return self._sim._pos[self.index]

    @pos.setter
    def pos(self, value: Sequence[float]) -> None:
        self._sim._pos[self.index] = value
        self._sim._acc_valid = False

    @property
    def vel(self) -> np.ndarray:
        return self._sim._vel[self.index]

    @vel.setter
    def vel(self, value: Sequence[float]) -> None:
        self._sim._vel[self.index] = value
        self._sim._acc_valid = False

    @property
    def acc(self) -> np.ndarray:
        return self._sim._acc[self.index]

    @acc.setter
    def acc(self, value: Sequence[float]) -> None:
        self._sim._acc[self.index] = value

    @property
    def mass(self) -> float:
        return float(self._sim._mass[self.index])

    @mass.setter
    def mass(self, value: float) -> None:
        m = float(value)
        if m <= 0.0:
            raise ValueError("Particle mass must be strictly positive.")
        self._sim._mass[self.index] = m
        self._sim._inv_mass[self.index] = (1.0 / m) if self._sim._movable[self.index] else 0.0
        self._sim._acc_valid = False

    @property
    def movable(self) -> bool:
        return bool(self._sim._movable[self.index])

    @movable.setter
    def movable(self, value: bool) -> None:
        flag = bool(value)
        self._sim._movable[self.index] = flag
        self._sim._inv_mass[self.index] = (
            (1.0 / self._sim._mass[self.index]) if flag else 0.0
        )
        if not flag:
            self._sim._vel[self.index] = 0.0
            self._sim._acc[self.index] = 0.0
        self._sim._acc_valid = False

    @property
    def charge(self) -> float:
        return float(self._sim._charge[self.index])

    @charge.setter
    def charge(self, value: float) -> None:
        self._sim._charge[self.index] = float(value)
        self._sim._acc_valid = False

    def kinetic_energy(self) -> float:
        v = self._sim._vel[self.index]
        return float(0.5 * self._sim._mass[self.index] * np.dot(v, v))

    # Backwards-compatible alias
    energy = kinetic_energy

    def __repr__(self) -> str:
        return (
            f"Particle(index={self.index}, mass={self.mass:.3g}, "
            f"pos={self.pos.tolist()}, movable={self.movable})"
        )


class Spring:
    """Lightweight handle pointing to a Hookean spring inside a `Simulation`."""

    def __init__(self, sim: "Simulation", index: int) -> None:
        self._sim = sim
        self.index = index

    @property
    def p1(self) -> Particle:
        return self._sim.particles[int(self._sim._s_i[self.index])]

    @property
    def p2(self) -> Particle:
        return self._sim.particles[int(self._sim._s_j[self.index])]

    @property
    def k(self) -> float:
        return float(self._sim._s_k[self.index])

    @k.setter
    def k(self, value: float) -> None:
        self._sim._s_k[self.index] = float(value)
        self._sim._acc_valid = False

    @property
    def L0(self) -> float:
        return float(self._sim._s_L0[self.index])

    @L0.setter
    def L0(self, value: float) -> None:
        self._sim._s_L0[self.index] = float(value)
        self._sim._acc_valid = False

    @property
    def b(self) -> float:
        return float(self._sim._s_b[self.index])

    @b.setter
    def b(self, value: float) -> None:
        self._sim._s_b[self.index] = float(value)
        self._sim._acc_valid = False

    @property
    def tension_only(self) -> bool:
        return bool(self._sim._s_tension_only[self.index])

    @tension_only.setter
    def tension_only(self, value: bool) -> None:
        self._sim._s_tension_only[self.index] = bool(value)
        self._sim._acc_valid = False

    def axis(self) -> np.ndarray:
        i = self._sim._s_i[self.index]
        j = self._sim._s_j[self.index]
        return self._sim._pos[j] - self._sim._pos[i]

    def length(self) -> float:
        return vnorm(self.axis())

    def extension(self) -> float:
        return self.length() - self.L0

    def strain(self) -> float:
        l0 = self.L0
        if l0 <= 1e-12:
            return self.extension()
        return self.extension() / l0

    def energy(self) -> float:
        ext = self.extension()
        if self.tension_only and ext <= 0.0:
            return 0.0
        return float(0.5 * self.k * ext * ext)

    def __repr__(self) -> str:
        return (
            f"Spring(index={self.index}, p1={self._sim._s_i[self.index]}, "
            f"p2={self._sim._s_j[self.index]}, k={self.k:.3g}, L0={self.L0:.3g})"
        )


class Simulation:
    """Pure-Newtonian particle-spring simulation engine.

    Parameters
    ----------
    dt : float
        Integration timestep in seconds.
    integrator : {"verlet", "yoshida4", "euler"}
        Single-step symplectic integrator. Defaults to `"verlet"` (Velocity Verlet).
    damping : bool
        If True, newly added springs default to `b=dissipation_coefficient`
        unless `b` is explicitly passed to `add_spring`.
    dissipation_coefficient : float
        Default axial damping coefficient when `damping=True`.
    use_numba : bool
        Use Numba JIT kernels when `numba` is installed and all active forces
        and drivers are built-in types. Falls back to vectorized NumPy otherwise.
    """

    def __init__(
        self,
        dt: float,
        integrator: str = "verlet",
        damping: bool = False,
        dissipation_coefficient: float = 0.5,
        use_numba: bool = True,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("Timestep `dt` must be strictly positive.")
        if integrator not in INTEGRATOR_MAP:
            raise ValueError(
                f"Unknown integrator {integrator!r}. Choose from {tuple(INTEGRATOR_MAP)}."
            )

        self.dt = float(dt)
        self._integrator = integrator
        self.damping = bool(damping)
        self.dissipation_coefficient = float(dissipation_coefficient)
        self.use_numba = bool(use_numba)

        self.time = 0.0
        self.step_count = 0

        # Structure-of-Arrays backing storage (grows geometrically)
        self._p_cap = 32
        self._n_particles = 0
        self._pos = np.zeros((self._p_cap, 3), dtype=np.float64)
        self._vel = np.zeros((self._p_cap, 3), dtype=np.float64)
        self._acc = np.zeros((self._p_cap, 3), dtype=np.float64)
        self._mass = np.ones(self._p_cap, dtype=np.float64)
        self._inv_mass = np.ones(self._p_cap, dtype=np.float64)
        self._movable = np.ones(self._p_cap, dtype=np.bool_)
        self._charge = np.zeros(self._p_cap, dtype=np.float64)

        self._s_cap = 64
        self._n_springs = 0
        self._s_i = np.zeros(self._s_cap, dtype=np.int64)
        self._s_j = np.zeros(self._s_cap, dtype=np.int64)
        self._s_k = np.zeros(self._s_cap, dtype=np.float64)
        self._s_L0 = np.zeros(self._s_cap, dtype=np.float64)
        self._s_b = np.zeros(self._s_cap, dtype=np.float64)
        self._s_tension_only = np.zeros(self._s_cap, dtype=np.bool_)

        self.particles: list[Particle] = []
        self.springs: list[Spring] = []
        self.forces: list[Force] = []
        self.harmonic_drivers: list[HarmonicDriver] = []
        self.custom_drivers: list[Callable[["Simulation"], None]] = []

        self._acc_valid = False
        self._edges_cache: Optional[np.ndarray] = None
        self._initial_state: Optional[tuple[np.ndarray, np.ndarray, float]] = None

    # ------------------------------------------------------------------
    # Properties & contiguous SoA array views
    # ------------------------------------------------------------------

    @property
    def integrator(self) -> str:
        return self._integrator

    @integrator.setter
    def integrator(self, value: str) -> None:
        if value not in INTEGRATOR_MAP:
            raise ValueError(
                f"Unknown integrator {value!r}. Choose from {tuple(INTEGRATOR_MAP)}."
            )
        self._integrator = value
        self._acc_valid = False

    @property
    def n_particles(self) -> int:
        return self._n_particles

    @property
    def n_springs(self) -> int:
        return self._n_springs

    @property
    def positions(self) -> np.ndarray:
        return self._pos[: self._n_particles]

    @property
    def velocities(self) -> np.ndarray:
        return self._vel[: self._n_particles]

    @property
    def accelerations(self) -> np.ndarray:
        return self._acc[: self._n_particles]

    @property
    def masses(self) -> np.ndarray:
        return self._mass[: self._n_particles]

    @property
    def inv_masses(self) -> np.ndarray:
        return self._inv_mass[: self._n_particles]

    @property
    def movable(self) -> np.ndarray:
        return self._movable[: self._n_particles]

    @property
    def charges(self) -> np.ndarray:
        return self._charge[: self._n_particles]

    @property
    def spring_i(self) -> np.ndarray:
        return self._s_i[: self._n_springs]

    @property
    def spring_j(self) -> np.ndarray:
        return self._s_j[: self._n_springs]

    @property
    def spring_k(self) -> np.ndarray:
        return self._s_k[: self._n_springs]

    @property
    def spring_L0(self) -> np.ndarray:
        return self._s_L0[: self._n_springs]

    @property
    def spring_b(self) -> np.ndarray:
        return self._s_b[: self._n_springs]

    @property
    def spring_tension_only(self) -> np.ndarray:
        return self._s_tension_only[: self._n_springs]

    @property
    def spring_edges(self) -> np.ndarray:
        """Return `(M, 2)` integer array of spring endpoint indices."""
        if self._edges_cache is None or self._edges_cache.shape[0] != self._n_springs:
            if self._n_springs == 0:
                self._edges_cache = np.empty((0, 2), dtype=np.int64)
            else:
                self._edges_cache = np.column_stack((self.spring_i, self.spring_j))
        return self._edges_cache

    @property
    def backend(self) -> str:
        """Name of the active execution backend (`'numba'` or `'numpy'`)."""
        return "numba" if self._can_use_numba() else "numpy"

    # ------------------------------------------------------------------
    # Capacity management
    # ------------------------------------------------------------------

    def _ensure_particle_capacity(self, needed: int) -> None:
        if needed <= self._p_cap:
            return
        new_cap = self._p_cap
        while new_cap < needed:
            new_cap *= 2
        self._pos = _grow_2d(self._pos, new_cap)
        self._vel = _grow_2d(self._vel, new_cap)
        self._acc = _grow_2d(self._acc, new_cap)
        self._mass = _grow_1d(self._mass, new_cap, fill=1.0)
        self._inv_mass = _grow_1d(self._inv_mass, new_cap, fill=1.0)
        self._movable = _grow_1d(self._movable, new_cap, fill=True)
        self._charge = _grow_1d(self._charge, new_cap, fill=0.0)
        self._p_cap = new_cap

    def _ensure_spring_capacity(self, needed: int) -> None:
        if needed <= self._s_cap:
            return
        new_cap = self._s_cap
        while new_cap < needed:
            new_cap *= 2
        self._s_i = _grow_1d(self._s_i, new_cap, fill=0)
        self._s_j = _grow_1d(self._s_j, new_cap, fill=0)
        self._s_k = _grow_1d(self._s_k, new_cap, fill=0.0)
        self._s_L0 = _grow_1d(self._s_L0, new_cap, fill=0.0)
        self._s_b = _grow_1d(self._s_b, new_cap, fill=0.0)
        self._s_tension_only = _grow_1d(self._s_tension_only, new_cap, fill=False)
        self._s_cap = new_cap

    # ------------------------------------------------------------------
    # Building the system
    # ------------------------------------------------------------------

    def add_particle(
        self,
        mass: float,
        position: Sequence[float],
        velocity: Optional[Sequence[float]] = None,
        movable: bool = True,
        charge: float = 0.0,
        properties=None,
    ) -> Particle:
        """Add a point mass to the simulation and return its `Particle` handle."""
        m = float(mass)
        if m <= 0.0:
            raise ValueError("Particle mass must be strictly positive.")

        idx = self._n_particles
        self._ensure_particle_capacity(idx + 1)
        self._n_particles += 1

        self._pos[idx] = np.asarray(position, dtype=np.float64).reshape(3)
        if velocity is None:
            self._vel[idx] = 0.0
        else:
            self._vel[idx] = np.asarray(velocity, dtype=np.float64).reshape(3)
        self._acc[idx] = 0.0
        self._mass[idx] = m
        self._movable[idx] = bool(movable)
        self._inv_mass[idx] = (1.0 / m) if movable else 0.0
        self._charge[idx] = float(charge)

        particle = Particle(self, idx, properties=properties)
        self.particles.append(particle)
        self._acc_valid = False
        return particle

    def add_spring(
        self,
        p1: Union[Particle, int],
        p2: Union[Particle, int],
        k: float,
        L0: Optional[float] = None,
        b: Optional[float] = None,
        tension_only: bool = False,
    ) -> Spring:
        """Connect two particles with a Hookean spring (and optional axial damping)."""
        i = p1.index if isinstance(p1, Particle) else int(p1)
        j = p2.index if isinstance(p2, Particle) else int(p2)
        if i == j:
            raise ValueError("Cannot connect a particle to itself with a spring.")

        if L0 is None:
            diff = self._pos[j] - self._pos[i]
            rest_length = float(np.sqrt(np.dot(diff, diff)))
        else:
            rest_length = float(L0)

        if b is None:
            damping_coeff = self.dissipation_coefficient if self.damping else 0.0
        else:
            damping_coeff = float(b)

        idx = self._n_springs
        self._ensure_spring_capacity(idx + 1)
        self._n_springs += 1

        self._s_i[idx] = i
        self._s_j[idx] = j
        self._s_k[idx] = float(k)
        self._s_L0[idx] = rest_length
        self._s_b[idx] = damping_coeff
        self._s_tension_only[idx] = bool(tension_only)
        self._edges_cache = None

        spring = Spring(self, idx)
        self.springs.append(spring)
        self._acc_valid = False
        return spring

    def add_force(self, force: Force) -> Force:
        """Add a built-in `Force` object (e.g., `Gravity`, `Coulomb`, `PlaneBoundary`)."""
        if not isinstance(force, Force):
            raise TypeError(
                f"Expected a Force instance, got {type(force).__name__}. "
                "For custom Python functions, use `sim.add_field(fn)`."
            )
        self.forces.append(force)
        self._acc_valid = False
        return force

    def add_field(
        self,
        fn: Union[Force, Callable],
        *,
        vectorized: bool = False,
        returns_acceleration: bool = False,
        potential: Optional[Callable[["Simulation"], float]] = None,
    ) -> Force:
        """Add a force law or custom Python field function.

        If `fn` is already a `Force` instance (such as `Gravity(...)`), it is
        registered directly. Otherwise it is wrapped in a `CustomField`.
        By default, custom functions return a **force** vector `(Fx, Fy, Fz)`
        acting on a particle; pass `returns_acceleration=True` if your function
        returns acceleration `(ax, ay, az)` instead.
        """
        if isinstance(fn, Force):
            return self.add_force(fn)
        field_obj = CustomField(
            fn,
            vectorized=vectorized,
            returns_acceleration=returns_acceleration,
            potential=potential,
        )
        self.forces.append(field_obj)
        self._acc_valid = False
        return field_obj

    def add_harmonic_driver(
        self,
        particle: Particle,
        axis: Sequence[float] = (0.0, 0.0, 1.0),
        amplitude: float = 0.1,
        frequency: float = 1.0,
        phase: float = 0.0,
    ) -> HarmonicDriver:
        """Drive a particle sinusoidally along `axis` (runs inside Numba too)."""
        particle.movable = False
        driver = HarmonicDriver.create(
            particle=particle,
            axis=axis,
            amplitude=amplitude,
            frequency=frequency,
            phase=phase,
        )
        self.harmonic_drivers.append(driver)
        self._acc_valid = False
        return driver

    def add_driver(self, fn: Callable[["Simulation"], None]) -> None:
        """Register a custom per-substep Python callback `fn(sim)` for kinematic driving."""
        self.custom_drivers.append(fn)
        self._acc_valid = False

    def set_all_spring_stiffness(self, k: float) -> None:
        """Set stiffness `k` on every spring (useful for live interactive sliders)."""
        if self._n_springs > 0:
            self.spring_k[:] = float(k)
            self._acc_valid = False

    def set_all_spring_damping(self, b: float) -> None:
        """Set damping coefficient `b` on every spring."""
        if self._n_springs > 0:
            self.spring_b[:] = float(b)
            self._acc_valid = False

    def invalidate(self) -> None:
        """Mark cached accelerations as stale after external parameter/state edits."""
        self._acc_valid = False

    # ------------------------------------------------------------------
    # State snapshot & reset (for interactive "what-if" exploration)
    # ------------------------------------------------------------------

    def save_initial_state(self) -> None:
        """Snapshot current positions, velocities, and time for `reset()`."""
        self._initial_state = (
            self.positions.copy(),
            self.velocities.copy(),
            float(self.time),
        )

    def reset(self) -> None:
        """Restore the saved initial state (positions, velocities, time)."""
        if self._initial_state is None:
            return
        pos0, vel0, t0 = self._initial_state
        n = min(pos0.shape[0], self._n_particles)
        self._pos[:n] = pos0[:n]
        self._vel[:n] = vel0[:n]
        self._acc[: self._n_particles] = 0.0
        self.time = t0
        self.step_count = 0
        self._acc_valid = False

    # ------------------------------------------------------------------
    # Force evaluation & stepping
    # ------------------------------------------------------------------

    def _can_use_numba(self) -> bool:
        if not (HAS_NUMBA and self.use_numba):
            return False
        if self.custom_drivers:
            return False
        coulomb_count = 0
        nbody_count = 0
        for f in self.forces:
            if not isinstance(f, _BUILTIN_FORCE_TYPES):
                return False
            if isinstance(f, Coulomb):
                coulomb_count += 1
            elif isinstance(f, NBodyGravity):
                nbody_count += 1
        if coulomb_count > 1 or nbody_count > 1:
            return False
        return True

    def _pack_numba_params(self):
        gravity_vec = np.zeros(3, dtype=np.float64)
        linear_drag = 0.0
        quad_drag = 0.0
        coulomb_ke = 0.0
        coulomb_eps2 = 1e-12
        nbody_G = 0.0
        nbody_eps2 = 1e-12
        em_E = np.zeros(3, dtype=np.float64)
        em_B = np.zeros(3, dtype=np.float64)
        has_em = False
        planes: list[PlaneBoundary] = []

        for f in self.forces:
            if isinstance(f, Gravity):
                gravity_vec += f.g
            elif isinstance(f, LinearDrag):
                linear_drag += f.gamma
            elif isinstance(f, QuadraticDrag):
                quad_drag += f.c
            elif isinstance(f, Coulomb):
                coulomb_ke = f.k_e
                coulomb_eps2 = f.softening * f.softening
            elif isinstance(f, NBodyGravity):
                nbody_G = f.G
                nbody_eps2 = f.softening * f.softening
            elif isinstance(f, UniformEMField):
                em_E += f.E
                em_B += f.B
                has_em = True
            elif isinstance(f, PlaneBoundary):
                planes.append(f)

        if planes:
            plane_normals = np.array([p.normal for p in planes], dtype=np.float64)
            plane_offsets = np.array([p.offset for p in planes], dtype=np.float64)
            plane_k = np.array([p.stiffness for p in planes], dtype=np.float64)
            plane_b = np.array([p.damping for p in planes], dtype=np.float64)
        else:
            plane_normals = np.empty((0, 3), dtype=np.float64)
            plane_offsets = np.empty(0, dtype=np.float64)
            plane_k = np.empty(0, dtype=np.float64)
            plane_b = np.empty(0, dtype=np.float64)

        if self.harmonic_drivers:
            drv_idx = np.array([d.particle_index for d in self.harmonic_drivers], dtype=np.int64)
            drv_base = np.array([d.base_position for d in self.harmonic_drivers], dtype=np.float64)
            drv_axis = np.array([d.axis for d in self.harmonic_drivers], dtype=np.float64)
            drv_amp = np.array([d.amplitude for d in self.harmonic_drivers], dtype=np.float64)
            drv_freq = np.array([d.frequency for d in self.harmonic_drivers], dtype=np.float64)
            drv_phase = np.array([d.phase for d in self.harmonic_drivers], dtype=np.float64)
        else:
            drv_idx = np.empty(0, dtype=np.int64)
            drv_base = np.empty((0, 3), dtype=np.float64)
            drv_axis = np.empty((0, 3), dtype=np.float64)
            drv_amp = np.empty(0, dtype=np.float64)
            drv_freq = np.empty(0, dtype=np.float64)
            drv_phase = np.empty(0, dtype=np.float64)

        return (
            gravity_vec,
            linear_drag,
            quad_drag,
            coulomb_ke,
            coulomb_eps2,
            nbody_G,
            nbody_eps2,
            em_E,
            em_B,
            has_em,
            plane_normals,
            plane_offsets,
            plane_k,
            plane_b,
            drv_idx,
            drv_base,
            drv_axis,
            drv_amp,
            drv_freq,
            drv_phase,
        )

    def _apply_drivers_python(self) -> None:
        if self.harmonic_drivers:
            pos = self.positions
            vel = self.velocities
            t = self.time
            for drv in self.harmonic_drivers:
                drv.apply(pos, vel, t)
        for fn in self.custom_drivers:
            fn(self)

    def compute_accelerations(self) -> np.ndarray:
        """Evaluate all spring and field forces and update `self.accelerations`."""
        n = self._n_particles
        acc = self.accelerations
        if n == 0:
            return acc

        acc[:] = 0.0
        # Accumulate total forces directly into `acc`, then scale by inv_mass
        if self._n_springs > 0:
            accumulate_spring_forces_numpy(
                self.positions,
                self.velocities,
                acc,
                self.spring_i,
                self.spring_j,
                self.spring_k,
                self.spring_L0,
                self.spring_b,
                self.spring_tension_only,
            )
        for force in self.forces:
            force.accumulate_forces(self, acc)

        acc *= self.inv_masses[:, None]
        self._acc_valid = True
        return acc

    def _step_numpy(self, n_steps: int) -> None:
        dt = self.dt
        pos = self.positions
        vel = self.velocities
        acc = self.accelerations
        mov = self.movable
        integrator = self._integrator

        if integrator == "verlet":
            if not self._acc_valid:
                self._apply_drivers_python()
                self.compute_accelerations()
            half_dt = 0.5 * dt
            for _ in range(n_steps):
                vel[mov] += acc[mov] * half_dt
                pos[mov] += vel[mov] * dt
                self.time += dt
                self._apply_drivers_python()
                self.compute_accelerations()
                vel[mov] += acc[mov] * half_dt

        elif integrator == "euler":
            for _ in range(n_steps):
                self._apply_drivers_python()
                self.compute_accelerations()
                vel[mov] += acc[mov] * dt
                pos[mov] += vel[mov] * dt
                self.time += dt
            self._acc_valid = False

        elif integrator == "yoshida4":
            c = [ci * dt for ci in YOSHIDA_C]
            d = [di * dt for di in YOSHIDA_D]
            for _ in range(n_steps):
                t_start = self.time
                # Drift 1
                pos[mov] += vel[mov] * c[0]
                self.time = t_start + c[0]
                self._apply_drivers_python()
                # Kick 1 + Drift 2
                self.compute_accelerations()
                vel[mov] += acc[mov] * d[0]
                pos[mov] += vel[mov] * c[1]
                self.time = t_start + c[0] + c[1]
                self._apply_drivers_python()
                # Kick 2 + Drift 3
                self.compute_accelerations()
                vel[mov] += acc[mov] * d[1]
                pos[mov] += vel[mov] * c[2]
                self.time = t_start + c[0] + c[1] + c[2]
                self._apply_drivers_python()
                # Kick 3 + Drift 4
                self.compute_accelerations()
                vel[mov] += acc[mov] * d[2]
                pos[mov] += vel[mov] * c[3]
                self.time = t_start + dt
                self._apply_drivers_python()
            self._acc_valid = False

    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation by `n_steps` timesteps."""
        if n_steps <= 0 or self._n_particles == 0:
            return
        if self._initial_state is None:
            self.save_initial_state()

        if self._can_use_numba():
            params = self._pack_numba_params()
            self.time, self._acc_valid = step_numba(
                n_steps,
                self.dt,
                self.time,
                INTEGRATOR_MAP[self._integrator],
                self._acc_valid,
                self.positions,
                self.velocities,
                self.accelerations,
                self.masses,
                self.inv_masses,
                self.movable,
                self.charges,
                self.spring_i,
                self.spring_j,
                self.spring_k,
                self.spring_L0,
                self.spring_b,
                self.spring_tension_only,
                *params,
            )
        else:
            self._step_numpy(n_steps)

        self.step_count += n_steps

    def advance(self, duration: float) -> int:
        """Advance the simulation by approximately `duration` seconds.

        Returns the number of substeps executed.
        """
        n_steps = max(1, int(round(duration / self.dt)))
        self.step(n_steps)
        return n_steps

    def update(self) -> None:
        """Advance the simulation by a single timestep (alias for `step(1)`)."""
        self.step(1)

    # ------------------------------------------------------------------
    # Vectorized spring inspection & physics diagnostics
    # ------------------------------------------------------------------

    def spring_lengths(self) -> np.ndarray:
        if self._n_springs == 0:
            return np.empty(0, dtype=np.float64)
        diff = self.positions[self.spring_j] - self.positions[self.spring_i]
        return np.sqrt(np.sum(diff * diff, axis=1))

    def spring_extensions(self) -> np.ndarray:
        if self._n_springs == 0:
            return np.empty(0, dtype=np.float64)
        return self.spring_lengths() - self.spring_L0

    def spring_strains(self) -> np.ndarray:
        if self._n_springs == 0:
            return np.empty(0, dtype=np.float64)
        ext = self.spring_extensions()
        l0 = np.where(self.spring_L0 > 1e-12, self.spring_L0, 1.0)
        return ext / l0

    def spring_tensions(self) -> np.ndarray:
        """Signed spring force magnitude (+ tension, - compression) for each spring."""
        if self._n_springs == 0:
            return np.empty(0, dtype=np.float64)
        ext = self.spring_extensions()
        ext = np.where(self.spring_tension_only & (ext < 0.0), 0.0, ext)
        return self.spring_k * ext

    def kinetic_energy(self) -> float:
        if self._n_particles == 0:
            return 0.0
        v2 = np.sum(self.velocities * self.velocities, axis=1)
        return float(0.5 * np.sum(self.masses * v2))

    def spring_energy(self) -> float:
        if self._n_springs == 0:
            return 0.0
        ext = self.spring_extensions()
        ext = np.where(self.spring_tension_only & (ext < 0.0), 0.0, ext)
        return float(0.5 * np.sum(self.spring_k * ext * ext))

    def field_potential_energy(self) -> float:
        return float(sum(f.potential_energy(self) for f in self.forces))

    def potential_energy(self) -> float:
        return self.spring_energy() + self.field_potential_energy()

    def total_energy(self) -> float:
        return self.kinetic_energy() + self.potential_energy()

    def get_energy(self) -> tuple[float, float]:
        """Return `(kinetic_energy, potential_energy)`."""
        return (self.kinetic_energy(), self.potential_energy())

    def _select_indices(self, particles: Optional[Iterable[Particle]]) -> np.ndarray:
        if particles is None:
            return np.arange(self._n_particles, dtype=np.int64)
        return np.array([p.index for p in particles], dtype=np.int64)

    def center_of_mass(self, particles: Optional[Iterable[Particle]] = None) -> np.ndarray:
        idx = self._select_indices(particles)
        if len(idx) == 0:
            return np.zeros(3, dtype=np.float64)
        m = self._mass[idx]
        total_m = float(np.sum(m))
        return (m @ self._pos[idx]) / total_m

    def center_of_mass_velocity(
        self, particles: Optional[Iterable[Particle]] = None
    ) -> np.ndarray:
        idx = self._select_indices(particles)
        if len(idx) == 0:
            return np.zeros(3, dtype=np.float64)
        m = self._mass[idx]
        total_m = float(np.sum(m))
        return (m @ self._vel[idx]) / total_m

    def linear_momentum(self, particles: Optional[Iterable[Particle]] = None) -> np.ndarray:
        idx = self._select_indices(particles)
        if len(idx) == 0:
            return np.zeros(3, dtype=np.float64)
        return self._mass[idx] @ self._vel[idx]

    def angular_momentum(
        self,
        origin: Optional[Sequence[float]] = None,
        particles: Optional[Iterable[Particle]] = None,
    ) -> np.ndarray:
        idx = self._select_indices(particles)
        if len(idx) == 0:
            return np.zeros(3, dtype=np.float64)
        rel_pos = self._pos[idx]
        if origin is not None:
            rel_pos = rel_pos - np.asarray(origin, dtype=np.float64).reshape(3)
        momenta = self._mass[idx, None] * self._vel[idx]
        return np.sum(np.cross(rel_pos, momenta), axis=0)


def _grow_1d(arr: np.ndarray, new_cap: int, fill) -> np.ndarray:
    out = np.full(new_cap, fill, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


def _grow_2d(arr: np.ndarray, new_cap: int) -> np.ndarray:
    out = np.zeros((new_cap, arr.shape[1]), dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out
