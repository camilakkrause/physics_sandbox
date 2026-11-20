"""Intermediate-Axis Theorem (Dzhanibekov Effect).

Five point masses connected purely by stiff Hookean springs exhibit periodic
180-degree mid-air flips when spun near their intermediate principal axis—
without any rigid-body equations or inertia tensors.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_sandbox import Simulation, Viewer, connect_all_pairs

DT = 5e-5
D1 = 1.0  # +/-x offset for light masses
D2 = 1.0  # +/-y offset for heavy masses
M1 = 1.0  # central & side masses
M2 = 2.0  # heavy rotor masses
K_SPRING = 1_000_000.0
V_Z = 10.0
DV_XY = 0.001  # tiny perturbation away from exact intermediate axis

sim = Simulation(dt=DT, integrator="verlet")

p1 = sim.add_particle(mass=M1, position=(0.0, 0.0, 0.0))
p2 = sim.add_particle(mass=M1, position=(D1, 0.0, 0.0))
p3 = sim.add_particle(mass=M1, position=(-D1, 0.0, 0.0))
p4 = sim.add_particle(mass=M2, position=(0.0, D2, 0.0), velocity=(DV_XY, DV_XY, V_Z))
p5 = sim.add_particle(mass=M2, position=(0.0, -D2, 0.0), velocity=(-DV_XY, -DV_XY, -V_Z))

# Fully brace the 5-mass T-handle using pure Hookean springs
connect_all_pairs(sim, [p1, p2, p3, p4, p5], k=K_SPRING)

viewer = Viewer(
    sim,
    title="Dzhanibekov Effect (Intermediate-Axis Theorem)",
    particle_radius=0.09,
    spring_radius=0.02,
    ground_plane="none",
)
viewer.add_trail(p2, max_points=300, color=(0.15, 0.85, 0.35), radius=0.015)
viewer.add_vector_overlay(
    "angular_momentum",
    origin_getter=sim.center_of_mass,
    vector_getter=lambda: sim.angular_momentum(origin=sim.center_of_mass()),
    color=(0.0, 0.85, 0.95),
    scale=0.05,
    radius=0.025,
)
viewer.add_plot("Tip x-position", lambda s: float(p2.pos[0]))

viewer.show()
