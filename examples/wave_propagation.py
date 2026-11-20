"""2D Wave Propagation in a Pre-Tensioned Mass-Spring Membrane.

Demonstrates a 45x45 (2,025-particle, ~3,960-spring) membrane driven at its
center by a built-in `HarmonicDriver`, with live tension/compression wave
coloring and interactive frequency/amplitude controls.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_sandbox import Simulation, Viewer, add_grid_2d

DT = 2e-3
NX, NY = 45, 45
LX, LY = 4.0, 4.0
K = 200.0
DAMPING = 0.05
AMPLITUDE = 0.15
FREQUENCY = 1.2

sim = Simulation(dt=DT, integrator="verlet")

# Pre-tensioned membrane (rest_length_scale < 1 creates surface tension)
grid = add_grid_2d(
    sim,
    nx=NX,
    ny=NY,
    lx=LX,
    ly=LY,
    mass=1.0,
    k=K,
    b=DAMPING,
    rest_length_scale=0.1,
    pin_boundary=True,
)

center = grid[NX // 2][NY // 2]
driver = sim.add_harmonic_driver(
    center,
    axis=(0.0, 0.0, 1.0),
    amplitude=AMPLITUDE,
    frequency=FREQUENCY,
)

viewer = Viewer(
    sim,
    title="2D Wave Propagation (2,025 Particles)",
    particle_radius=0.0,  # Hide spheres to highlight the spring mesh
    spring_radius=0.018,
    color_springs_by_tension=True,
    ground_plane="none",
)
viewer.add_slider(
    "Drive Frequency (Hz)",
    0.1,
    4.0,
    FREQUENCY,
    on_change=lambda val: setattr(driver, "frequency", val),
)
viewer.add_slider(
    "Drive Amplitude",
    0.0,
    0.4,
    AMPLITUDE,
    on_change=lambda val: setattr(driver, "amplitude", val),
)
viewer.add_slider(
    "Damping b",
    0.0,
    1.0,
    DAMPING,
    on_change=sim.set_all_spring_damping,
)

viewer.show()
