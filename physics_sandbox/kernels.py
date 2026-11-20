"""Vectorized NumPy and optional Numba-accelerated physics kernels.

All integrators are strictly single-step (current state -> next state) with no
past-step history and no algebraic constraints:
  - "verlet"   : Velocity Verlet (2nd-order symplectic, time-reversible)
  - "yoshida4" : Yoshida 4th-order symplectic (3 Verlet sub-kicks per step)
  - "euler"    : Symplectic Euler (1st-order symplectic)
"""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn

        if args and callable(args[0]):
            return args[0]
        return decorator


# Yoshida 4th-order symplectic integrator constants (Yoshida 1990)
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _CBRT2)
_W0 = -_CBRT2 / (2.0 - _CBRT2)
YOSHIDA_C = (_W1 * 0.5, (_W0 + _W1) * 0.5, (_W0 + _W1) * 0.5, _W1 * 0.5)
YOSHIDA_D = (_W1, _W0, _W1)

INTEGRATOR_EULER = 0
INTEGRATOR_VERLET = 1
INTEGRATOR_YOSHIDA4 = 2

INTEGRATOR_MAP = {
    "euler": INTEGRATOR_EULER,
    "verlet": INTEGRATOR_VERLET,
    "yoshida4": INTEGRATOR_YOSHIDA4,
}


# ---------------------------------------------------------------------------
# Pure-NumPy vectorized spring kernel (used when Numba is unavailable or when
# custom Python fields are attached)
# ---------------------------------------------------------------------------


def accumulate_spring_forces_numpy(
    pos: np.ndarray,
    vel: np.ndarray,
    forces: np.ndarray,
    s_i: np.ndarray,
    s_j: np.ndarray,
    s_k: np.ndarray,
    s_L0: np.ndarray,
    s_b: np.ndarray,
    s_tension_only: np.ndarray,
) -> None:
    """Compute Hooke + axial damping forces for all springs using vectorized NumPy."""
    if len(s_i) == 0:
        return

    axis = pos[s_j] - pos[s_i]
    dist = np.sqrt(np.sum(axis * axis, axis=1))
    safe = dist > 1e-12
    inv_dist = np.where(safe, 1.0 / np.where(safe, dist, 1.0), 0.0)

    ext = dist - s_L0
    active = safe & (~s_tension_only | (ext > 0.0))

    # Scalar factor along `axis`: F_on_i = scalar * axis
    scalar = np.where(active, s_k * ext * inv_dist, 0.0)

    if np.any(s_b != 0.0):
        v_rel = vel[s_j] - vel[s_i]
        ldot = np.sum(v_rel * axis, axis=1) * inv_dist
        scalar = scalar + np.where(active, s_b * ldot * inv_dist, 0.0)

    f_vec = scalar[:, None] * axis
    np.add.at(forces, s_i, f_vec)
    np.add.at(forces, s_j, -f_vec)


# ---------------------------------------------------------------------------
# Numba-compiled full substep loop (springs + all built-in forces + drivers)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _apply_harmonic_drivers(
    pos: np.ndarray,
    vel: np.ndarray,
    t: float,
    drv_idx: np.ndarray,
    drv_base: np.ndarray,
    drv_axis: np.ndarray,
    drv_amp: np.ndarray,
    drv_freq: np.ndarray,
    drv_phase: np.ndarray,
) -> None:
    two_pi = 2.0 * math.pi
    for m in range(drv_idx.shape[0]):
        idx = drv_idx[m]
        omega = two_pi * drv_freq[m]
        arg = omega * t + drv_phase[m]
        disp = drv_amp[m] * math.sin(arg)
        speed = drv_amp[m] * omega * math.cos(arg)
        pos[idx, 0] = drv_base[m, 0] + disp * drv_axis[m, 0]
        pos[idx, 1] = drv_base[m, 1] + disp * drv_axis[m, 1]
        pos[idx, 2] = drv_base[m, 2] + disp * drv_axis[m, 2]
        vel[idx, 0] = speed * drv_axis[m, 0]
        vel[idx, 1] = speed * drv_axis[m, 1]
        vel[idx, 2] = speed * drv_axis[m, 2]


@njit(cache=True)
def _compute_acc_numba(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    inv_mass: np.ndarray,
    charge: np.ndarray,
    acc: np.ndarray,
    # Springs
    s_i: np.ndarray,
    s_j: np.ndarray,
    s_k: np.ndarray,
    s_L0: np.ndarray,
    s_b: np.ndarray,
    s_tension_only: np.ndarray,
    # Built-in forces
    gravity_vec: np.ndarray,
    linear_drag: float,
    quad_drag: float,
    coulomb_ke: float,
    coulomb_eps2: float,
    nbody_G: float,
    nbody_eps2: float,
    em_E: np.ndarray,
    em_B: np.ndarray,
    has_em: bool,
    plane_normals: np.ndarray,
    plane_offsets: np.ndarray,
    plane_k: np.ndarray,
    plane_b: np.ndarray,
) -> None:
    n = pos.shape[0]
    gx = gravity_vec[0]
    gy = gravity_vec[1]
    gz = gravity_vec[2]
    has_grav = gx != 0.0 or gy != 0.0 or gz != 0.0

    # Initialize force accumulator inside `acc` (we convert F -> a at the end)
    for i in range(n):
        if has_grav:
            m = mass[i]
            acc[i, 0] = m * gx
            acc[i, 1] = m * gy
            acc[i, 2] = m * gz
        else:
            acc[i, 0] = 0.0
            acc[i, 1] = 0.0
            acc[i, 2] = 0.0

        if linear_drag != 0.0:
            acc[i, 0] -= linear_drag * vel[i, 0]
            acc[i, 1] -= linear_drag * vel[i, 1]
            acc[i, 2] -= linear_drag * vel[i, 2]

        if quad_drag != 0.0:
            vx = vel[i, 0]
            vy = vel[i, 1]
            vz = vel[i, 2]
            spd = math.sqrt(vx * vx + vy * vy + vz * vz)
            c_spd = quad_drag * spd
            acc[i, 0] -= c_spd * vx
            acc[i, 1] -= c_spd * vy
            acc[i, 2] -= c_spd * vz

        if has_em:
            q = charge[i]
            if q != 0.0:
                vx = vel[i, 0]
                vy = vel[i, 1]
                vz = vel[i, 2]
                bx = em_B[0]
                by = em_B[1]
                bz = em_B[2]
                acc[i, 0] += q * (em_E[0] + vy * bz - vz * by)
                acc[i, 1] += q * (em_E[1] + vz * bx - vx * bz)
                acc[i, 2] += q * (em_E[2] + vx * by - vy * bx)

    # Springs
    ns = s_i.shape[0]
    for e in range(ns):
        i = s_i[e]
        j = s_j[e]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = pos[j, 2] - pos[i, 2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist <= 1e-12:
            continue
        ext = dist - s_L0[e]
        if s_tension_only[e] and ext <= 0.0:
            continue
        inv_d = 1.0 / dist
        scalar = s_k[e] * ext * inv_d
        b = s_b[e]
        if b != 0.0:
            dvx = vel[j, 0] - vel[i, 0]
            dvy = vel[j, 1] - vel[i, 1]
            dvz = vel[j, 2] - vel[i, 2]
            ldot = (dvx * dx + dvy * dy + dvz * dz) * inv_d
            scalar += b * ldot * inv_d
        fx = scalar * dx
        fy = scalar * dy
        fz = scalar * dz
        acc[i, 0] += fx
        acc[i, 1] += fy
        acc[i, 2] += fz
        acc[j, 0] -= fx
        acc[j, 1] -= fy
        acc[j, 2] -= fz

    # Pairwise Coulomb (symmetric i < j)
    if coulomb_ke != 0.0:
        for i in range(n - 1):
            qi = charge[i]
            if qi == 0.0:
                continue
            xi = pos[i, 0]
            yi = pos[i, 1]
            zi = pos[i, 2]
            for j in range(i + 1, n):
                qj = charge[j]
                if qj == 0.0:
                    continue
                dx = xi - pos[j, 0]
                dy = yi - pos[j, 1]
                dz = zi - pos[j, 2]
                r2 = dx * dx + dy * dy + dz * dz + coulomb_eps2
                inv_r3 = 1.0 / (r2 * math.sqrt(r2))
                c = coulomb_ke * qi * qj * inv_r3
                fx = c * dx
                fy = c * dy
                fz = c * dz
                acc[i, 0] += fx
                acc[i, 1] += fy
                acc[i, 2] += fz
                acc[j, 0] -= fx
                acc[j, 1] -= fy
                acc[j, 2] -= fz

    # Pairwise Newtonian gravity (symmetric i < j)
    if nbody_G != 0.0:
        for i in range(n - 1):
            mi = mass[i]
            xi = pos[i, 0]
            yi = pos[i, 1]
            zi = pos[i, 2]
            for j in range(i + 1, n):
                dx = xi - pos[j, 0]
                dy = yi - pos[j, 1]
                dz = zi - pos[j, 2]
                r2 = dx * dx + dy * dy + dz * dz + nbody_eps2
                inv_r3 = 1.0 / (r2 * math.sqrt(r2))
                c = -nbody_G * mi * mass[j] * inv_r3
                fx = c * dx
                fy = c * dy
                fz = c * dz
                acc[i, 0] += fx
                acc[i, 1] += fy
                acc[i, 2] += fz
                acc[j, 0] -= fx
                acc[j, 1] -= fy
                acc[j, 2] -= fz

    # Plane boundaries
    np_planes = plane_offsets.shape[0]
    for p_idx in range(np_planes):
        nx = plane_normals[p_idx, 0]
        ny = plane_normals[p_idx, 1]
        nz = plane_normals[p_idx, 2]
        off = plane_offsets[p_idx]
        pk = plane_k[p_idx]
        pb = plane_b[p_idx]
        for i in range(n):
            pen = off - (pos[i, 0] * nx + pos[i, 1] * ny + pos[i, 2] * nz)
            if pen > 0.0:
                vn = vel[i, 0] * nx + vel[i, 1] * ny + vel[i, 2] * nz
                f_mag = pk * pen - pb * vn
                if f_mag > 0.0:
                    acc[i, 0] += f_mag * nx
                    acc[i, 1] += f_mag * ny
                    acc[i, 2] += f_mag * nz

    # Convert total force to acceleration (inv_mass == 0.0 for fixed particles)
    for i in range(n):
        im = inv_mass[i]
        acc[i, 0] *= im
        acc[i, 1] *= im
        acc[i, 2] *= im


@njit(cache=True)
def step_numba(
    n_steps: int,
    dt: float,
    t0: float,
    integrator_id: int,
    acc_valid: bool,
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    mass: np.ndarray,
    inv_mass: np.ndarray,
    movable: np.ndarray,
    charge: np.ndarray,
    s_i: np.ndarray,
    s_j: np.ndarray,
    s_k: np.ndarray,
    s_L0: np.ndarray,
    s_b: np.ndarray,
    s_tension_only: np.ndarray,
    gravity_vec: np.ndarray,
    linear_drag: float,
    quad_drag: float,
    coulomb_ke: float,
    coulomb_eps2: float,
    nbody_G: float,
    nbody_eps2: float,
    em_E: np.ndarray,
    em_B: np.ndarray,
    has_em: bool,
    plane_normals: np.ndarray,
    plane_offsets: np.ndarray,
    plane_k: np.ndarray,
    plane_b: np.ndarray,
    drv_idx: np.ndarray,
    drv_base: np.ndarray,
    drv_axis: np.ndarray,
    drv_amp: np.ndarray,
    drv_freq: np.ndarray,
    drv_phase: np.ndarray,
) -> tuple[float, bool]:
    """Run `n_steps` substeps entirely in compiled machine code."""
    n = pos.shape[0]
    t = t0
    has_drivers = drv_idx.shape[0] > 0

    if integrator_id == INTEGRATOR_VERLET:
        if not acc_valid:
            if has_drivers:
                _apply_harmonic_drivers(
                    pos, vel, t, drv_idx, drv_base, drv_axis, drv_amp, drv_freq, drv_phase
                )
            _compute_acc_numba(
                pos, vel, mass, inv_mass, charge, acc,
                s_i, s_j, s_k, s_L0, s_b, s_tension_only,
                gravity_vec, linear_drag, quad_drag,
                coulomb_ke, coulomb_eps2, nbody_G, nbody_eps2,
                em_E, em_B, has_em,
                plane_normals, plane_offsets, plane_k, plane_b,
            )
            acc_valid = True

        half_dt = 0.5 * dt
        for _ in range(n_steps):
            for i in range(n):
                if movable[i]:
                    vx = vel[i, 0] + acc[i, 0] * half_dt
                    vy = vel[i, 1] + acc[i, 1] * half_dt
                    vz = vel[i, 2] + acc[i, 2] * half_dt
                    vel[i, 0] = vx
                    vel[i, 1] = vy
                    vel[i, 2] = vz
                    pos[i, 0] += vx * dt
                    pos[i, 1] += vy * dt
                    pos[i, 2] += vz * dt
            t += dt
            if has_drivers:
                _apply_harmonic_drivers(
                    pos, vel, t, drv_idx, drv_base, drv_axis, drv_amp, drv_freq, drv_phase
                )
            _compute_acc_numba(
                pos, vel, mass, inv_mass, charge, acc,
                s_i, s_j, s_k, s_L0, s_b, s_tension_only,
                gravity_vec, linear_drag, quad_drag,
                coulomb_ke, coulomb_eps2, nbody_G, nbody_eps2,
                em_E, em_B, has_em,
                plane_normals, plane_offsets, plane_k, plane_b,
            )
            for i in range(n):
                if movable[i]:
                    vel[i, 0] += acc[i, 0] * half_dt
                    vel[i, 1] += acc[i, 1] * half_dt
                    vel[i, 2] += acc[i, 2] * half_dt

    elif integrator_id == INTEGRATOR_EULER:
        for _ in range(n_steps):
            if has_drivers:
                _apply_harmonic_drivers(
                    pos, vel, t, drv_idx, drv_base, drv_axis, drv_amp, drv_freq, drv_phase
                )
            _compute_acc_numba(
                pos, vel, mass, inv_mass, charge, acc,
                s_i, s_j, s_k, s_L0, s_b, s_tension_only,
                gravity_vec, linear_drag, quad_drag,
                coulomb_ke, coulomb_eps2, nbody_G, nbody_eps2,
                em_E, em_B, has_em,
                plane_normals, plane_offsets, plane_k, plane_b,
            )
            for i in range(n):
                if movable[i]:
                    vx = vel[i, 0] + acc[i, 0] * dt
                    vy = vel[i, 1] + acc[i, 1] * dt
                    vz = vel[i, 2] + acc[i, 2] * dt
                    vel[i, 0] = vx
                    vel[i, 1] = vy
                    vel[i, 2] = vz
                    pos[i, 0] += vx * dt
                    pos[i, 1] += vy * dt
                    pos[i, 2] += vz * dt
            t += dt
        acc_valid = False

    elif integrator_id == INTEGRATOR_YOSHIDA4:
        c0 = YOSHIDA_C[0] * dt
        c1 = YOSHIDA_C[1] * dt
        c2 = YOSHIDA_C[2] * dt
        c3 = YOSHIDA_C[3] * dt
        d0 = YOSHIDA_D[0] * dt
        d1 = YOSHIDA_D[1] * dt
        d2 = YOSHIDA_D[2] * dt

        for _ in range(n_steps):
            # Drift 1
            for i in range(n):
                if movable[i]:
                    pos[i, 0] += vel[i, 0] * c0
                    pos[i, 1] += vel[i, 1] * c0
                    pos[i, 2] += vel[i, 2] * c0
            if has_drivers:
                _apply_harmonic_drivers(
                    pos, vel, t + c0, drv_idx, drv_base, drv_axis, drv_amp, drv_freq, drv_phase
                )
            # Kick 1
            _compute_acc_numba(
                pos, vel, mass, inv_mass, charge, acc,
                s_i, s_j, s_k, s_L0, s_b, s_tension_only,
                gravity_vec, linear_drag, quad_drag,
                coulomb_ke, coulomb_eps2, nbody_G, nbody_eps2,
                em_E, em_B, has_em,
                plane_normals, plane_offsets, plane_k, plane_b,
            )
            for i in range(n):
                if movable[i]:
                    vel[i, 0] += acc[i, 0] * d0
                    vel[i, 1] += acc[i, 1] * d0
                    vel[i, 2] += acc[i, 2] * d0
                    pos[i, 0] += vel[i, 0] * c1
                    pos[i, 1] += vel[i, 1] * c1
                    pos[i, 2] += vel[i, 2] * c1
            if has_drivers:
                _apply_harmonic_drivers(
                    pos, vel, t + c0 + c1, drv_idx, drv_base, drv_axis, drv_amp, drv_freq, drv_phase
                )
            # Kick 2
            _compute_acc_numba(
                pos, vel, mass, inv_mass, charge, acc,
                s_i, s_j, s_k, s_L0, s_b, s_tension_only,
                gravity_vec, linear_drag, quad_drag,
                coulomb_ke, coulomb_eps2, nbody_G, nbody_eps2,
                em_E, em_B, has_em,
                plane_normals, plane_offsets, plane_k, plane_b,
            )
            for i in range(n):
                if movable[i]:
                    vel[i, 0] += acc[i, 0] * d1
                    vel[i, 1] += acc[i, 1] * d1
                    vel[i, 2] += acc[i, 2] * d1
                    pos[i, 0] += vel[i, 0] * c2
                    pos[i, 1] += vel[i, 1] * c2
                    pos[i, 2] += vel[i, 2] * c2
            if has_drivers:
                _apply_harmonic_drivers(
                    pos, vel, t + c0 + c1 + c2, drv_idx, drv_base, drv_axis, drv_amp, drv_freq, drv_phase
                )
            # Kick 3
            _compute_acc_numba(
                pos, vel, mass, inv_mass, charge, acc,
                s_i, s_j, s_k, s_L0, s_b, s_tension_only,
                gravity_vec, linear_drag, quad_drag,
                coulomb_ke, coulomb_eps2, nbody_G, nbody_eps2,
                em_E, em_B, has_em,
                plane_normals, plane_offsets, plane_k, plane_b,
            )
            for i in range(n):
                if movable[i]:
                    vel[i, 0] += acc[i, 0] * d2
                    vel[i, 1] += acc[i, 1] * d2
                    vel[i, 2] += acc[i, 2] * d2
                    pos[i, 0] += vel[i, 0] * c3
                    pos[i, 1] += vel[i, 1] * c3
                    pos[i, 2] += vel[i, 2] * c3
            t += dt
            if has_drivers:
                _apply_harmonic_drivers(
                    pos, vel, t, drv_idx, drv_base, drv_axis, drv_amp, drv_freq, drv_phase
                )
        acc_valid = False

    return t, acc_valid
