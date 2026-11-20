"""Simple Pendulum — minimal Physics Sandbox quickstart.

Simulates a pendulum using two point masses connected by a very stiff spring
(approximating a rigid rod) under uniform gravity.
"""
from physics_sandbox import Gravity, Simulation, Viewer

sim = Simulation(dt=5e-5, integrator="verlet")

pivot = sim.add_particle(mass=1.0, position=(0.0, 0.0, 1.0), movable=False)
bob = sim.add_particle(mass=1.0, position=(0.0, 0.0, 0.0), velocity=(2.5, 0.0, 0.0))

# Stiff spring acts like a rigid rod—no pendulum equations needed
rod = sim.add_spring(pivot, bob, k=100_000.0, b=0.0)

gravity = Gravity((0.0, 0.0, -9.8))
sim.add_force(gravity)

viewer = Viewer(
    sim,
    title="Simple Pendulum",
    particle_radius=0.08,
    spring_radius=0.02,
    color_springs_by_tension=True,
)
viewer.add_trail(bob, max_points=200, color=(0.15, 0.85, 0.35))
viewer.add_slider(
    "Gravity |g|",
    0.0,
    25.0,
    9.81,
    on_change=lambda val: setattr(gravity, "g", (0.0, 0.0, -val)),
)
viewer.add_slider(
    "Spring Damping",
    0.0,
    5.0,
    0.0,
    on_change=lambda val: setattr(rod, "b", val),
)

viewer.show()
