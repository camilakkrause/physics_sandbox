"""3D Elastic Cube Vibration & Energy Analysis.

Builds a 3D mass-spring lattice with face-diagonal bracing, kicks one face
node, and visualizes internal stress waves (blue = compression, red = tension)
alongside live kinetic/potential energy and particle speed plots.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_sandbox import Simulation, Viewer, add_lattice_3d

DT = 5e-4
K_SPRING = 500.0
DAMPING = 0.25
INIT_V = 4.0

sim = Simulation(dt=DT, integrator="verlet")

# 4x4x4 elastic cube (64 masses, 300+ springs)
lattice = add_lattice_3d(
    sim,
    nx=4,
    ny=4,
    nz=4,
    spacing=1.0,
    mass=1.0,
    k=K_SPRING,
    b=DAMPING,
    connect_face_diagonals=True,
)

# Kick a node on the -x face and monitor both ends
kicked = lattice[0][1][1]
opposite = lattice[3][1][1]
kicked.vel = (INIT_V, 0.0, 0.0)

viewer = Viewer(
    sim,
    title="3D Lattice Vibration",
    particle_radius=0.09,
    spring_radius=0.018,
    color_springs_by_tension=True,
    ground_plane="none",
)
viewer.add_plot("Kicked v_x", lambda s: float(kicked.vel[0]))
viewer.add_plot("Opposite v_x", lambda s: float(opposite.vel[0]))
viewer.add_slider(
    "Stiffness k",
    50.0,
    2000.0,
    K_SPRING,
    on_change=sim.set_all_spring_stiffness,
    fmt="%.0f",
)
viewer.add_slider(
    "Damping b",
    0.0,
    3.0,
    DAMPING,
    on_change=sim.set_all_spring_damping,
)

viewer.show()
