"""Gyroscopic Precession & Nutation vs. Analytical Rigid-Body Prediction.

Simulates a spinning top as 6 point masses joined by stiff springs (with one
fixed pivot at the origin) under uniform gravity, and overlays the analytical
Lagrangian rigid-body prediction in real time.
"""

from math import cos, sin
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from physics_sandbox import Gravity, Simulation, Viewer

DT = 4e-5
K = 1_000_000.0
V_ROT = 30.0
MASS = 1.0
L1 = 3.0
L2 = 1.0
G = 9.81

sim = Simulation(dt=DT, integrator="verlet")
gravity = Gravity((0.0, 0.0, -G))
sim.add_force(gravity)

# Fixed pivot + spinning rotor cross
p1 = sim.add_particle(mass=MASS, position=(0.0, 0.0, 0.0), movable=False)
p2 = sim.add_particle(mass=MASS, position=(L1, 0.0, 0.0))
p3 = sim.add_particle(mass=MASS, position=(L1, L2, 0.0), velocity=(0.0, 0.0, V_ROT))
p4 = sim.add_particle(mass=MASS, position=(L1, 0.0, L2), velocity=(0.0, -V_ROT, 0.0))
p5 = sim.add_particle(mass=MASS, position=(L1, -L2, 0.0), velocity=(0.0, 0.0, -V_ROT))
p6 = sim.add_particle(mass=MASS, position=(L1, 0.0, -L2), velocity=(0.0, V_ROT, 0.0))

# Principal axle
sim.add_spring(p1, p2, K)
# Spokes around axle tip
for rim in (p3, p4, p5, p6):
    sim.add_spring(p2, rim, K)
    sim.add_spring(p1, rim, K)
# Outer ring
sim.add_spring(p3, p4, K)
sim.add_spring(p4, p5, K)
sim.add_spring(p5, p6, K)
sim.add_spring(p6, p3, K)

rotor_particles = (p2, p3, p4, p5, p6)

# ------------------------------------------------------------------
# Analytical rigid-body gyroscope prediction (Euler-Lagrange equations)
# ------------------------------------------------------------------
M_TOTAL = 5.0 * MASS
I3 = 4.0 * MASS * L2**2
I1 = I3 / 2.0 + M_TOTAL * L1**2
OMEGA_SPIN = V_ROT / L2

state = {"theta": 0.0, "theta1": 0.0, "phi": 0.0, "phi1": 0.0}


def reset_analytical(_sim: Simulation) -> None:
    state["theta"] = 0.0
    state["theta1"] = 0.0
    state["phi"] = 0.0
    state["phi1"] = 0.0


def advance_analytical(_sim: Simulation, dt_frame: float) -> None:
    g_mag = float(-gravity.g[2])
    n_sub = max(1, int(round(dt_frame / DT)))
    sub_dt = dt_frame / n_sub
    theta = state["theta"]
    theta1 = state["theta1"]
    phi = state["phi"]
    phi1 = state["phi1"]

    for _ in range(n_sub):
        c_phi = cos(phi)
        s_phi = sin(phi)
        theta2 = (phi1 / c_phi) * (2.0 * theta1 * s_phi - (I3 / I1) * OMEGA_SPIN)
        phi2 = (
            theta1 * c_phi * (-theta1 * s_phi + (I3 / I1) * OMEGA_SPIN)
            - (M_TOTAL * g_mag * L1 / I1) * c_phi
        )
        phi1 += phi2 * sub_dt
        phi += phi1 * sub_dt
        theta1 += theta2 * sub_dt
        theta += theta1 * sub_dt

    state["theta"] = theta
    state["theta1"] = theta1
    state["phi"] = phi
    state["phi1"] = phi1


def analytical_pos() -> np.ndarray:
    th, ph = state["theta"], state["phi"]
    return np.array(
        [L1 * cos(ph) * cos(th), L1 * cos(ph) * sin(th), L1 * sin(ph)],
        dtype=np.float64,
    )


viewer = Viewer(
    sim,
    title="Gyroscope Precession & Nutation",
    particle_radius=0.08,
    spring_radius=0.02,
    on_frame=advance_analytical,
    on_reset=reset_analytical,
)
viewer.add_trail(p2, max_points=600, color=(0.15, 0.85, 0.35), radius=0.015)
viewer.add_marker(
    "analytical_prediction",
    analytical_pos,
    color=(0.95, 0.22, 0.18),
    radius=0.09,
    trail=True,
    trail_points=600,
    trail_color=(1.0, 0.55, 0.10),
    trail_radius=0.015,
)
viewer.add_marker(
    "center_of_mass",
    lambda: sim.center_of_mass(rotor_particles),
    color=(0.90, 0.15, 0.85),
    radius=0.07,
)
viewer.add_vector_overlay(
    "angular_momentum",
    origin_getter=lambda: np.zeros(3),
    vector_getter=lambda: sim.angular_momentum(),
    color=(0.0, 0.85, 0.95),
    scale=0.015,
    radius=0.025,
)
viewer.add_slider(
    "Gravity |g|",
    0.0,
    25.0,
    G,
    on_change=lambda val: setattr(gravity, "g", (0.0, 0.0, -val)),
)

viewer.show()
