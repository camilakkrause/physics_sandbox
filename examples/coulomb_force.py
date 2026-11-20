"""Inverse-Square Electrostatic Orbits (Coulomb Force).

Demonstrates pairwise symmetric Coulomb interactions (respecting Newton's 3rd
law and conserving total energy & momentum) with live orbital trails.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_sandbox import Coulomb, Simulation, Viewer

DT = 1e-4
K_COULOMB = 0.5

sim = Simulation(dt=DT, integrator="yoshida4")
coulomb = Coulomb(k_e=K_COULOMB, softening=1e-4)
sim.add_force(coulomb)

p1 = sim.add_particle(mass=5.0, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), charge=10.0)
p2 = sim.add_particle(mass=1.0, position=(3.0, -1.0, 0.0), velocity=(-0.5, 1.0, 0.0), charge=-0.5)
p3 = sim.add_particle(mass=1.0, position=(3.0, 1.0, 0.0), velocity=(-0.5, -1.0, 0.0), charge=-0.5)

viewer = Viewer(
    sim,
    title="Coulomb Orbits",
    particle_radius=0.12,
    spring_radius=0.01,
    ground_plane="none",
)
viewer.add_trail(p1, max_points=400, color=(0.95, 0.25, 0.25), radius=0.015)
viewer.add_trail(p2, max_points=400, color=(0.15, 0.85, 0.35), radius=0.015)
viewer.add_trail(p3, max_points=400, color=(0.25, 0.65, 1.0), radius=0.015)
viewer.add_slider(
    "Coulomb k_e",
    0.0,
    2.0,
    K_COULOMB,
    on_change=lambda val: setattr(coulomb, "k_e", val),
)

viewer.show()
