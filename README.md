# Physics Sandbox

**Physics Sandbox** is a fast, minimal, educational Python engine for exploring classical mechanics using only Newton's laws. Build particle–spring systems, add force laws, and watch complex rotational and continuum phenomena—like gyroscopic precession, nutation, the Dzhanibekov intermediate-axis flip, and 2D/3D elastic waves—emerge naturally without rigid-body equations or algebraic constraints.

---

## Features

* **Pure Newtonian Core:** Model rigid bodies, chains, membranes, and 3D elastic solids using point masses and Hookean springs—no rotational mechanics formulas or algebraic constraints needed.
* **Fast Structure-of-Arrays Engine + Optional Numba JIT:** Contiguous NumPy storage with automatic Numba acceleration (`sim.step(n_steps)`) handles thousands of particles and stiff-spring rigid bodies smoothly, while keeping an intuitive `Particle` and `Spring` object interface.
* **Single-Step Symplectic Integrators:** Choose between **Velocity Verlet** (default, 2nd-order symplectic), **Yoshida 4th-Order Symplectic** (high accuracy for stiff springs and orbits), and **Symplectic Euler**—all strictly advancing from the current state to the next with no past-step history.
* **Interactive GPU 3D Viewer (Polyscope):** Real-time 3D visualization with right-handed Z-up coordinates, spring tension/compression colormapping, particle speed colormapping, smooth trajectory trails, 3D vector overlays, live energy & custom plots, and an interactive sidebar with Play/Pause, Step, Reset, Time Scale, Integrator switching, and custom **"What-If" sliders**.
* **Built-in Force Laws & Geometry Builders:** Ready-to-use forces (gravity, drag, Coulomb, N-body gravity, electromagnetic fields, contact planes), harmonic drivers, custom Python fields, and 1D/2D/3D mesh builders.
* **Headless & Scriptable:** Use `Simulation` on its own for batch experiments, parameter sweeps, or data analysis with zero visualization dependencies.

---

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/camilakkrause/physics_sandbox.git
cd physics_sandbox

# Recommended: install editable package with Polyscope viewer, Numba JIT, and pytest
pip install -e ".[all]"
```

You can also install only the extras you need:

* Headless core only (NumPy): `pip install -e .`
* With 3D visualization: `pip install -e ".[vis]"`
* With Numba acceleration: `pip install -e ".[fast]"`

---

## Quickstart

**Example: Simple Pendulum** (`examples/simple_pendulum.py`)

Simulate a pendulum as two masses joined by a very stiff spring under gravity, with interactive sliders to tweak gravity and damping live:

```python
from physics_sandbox import Gravity, Simulation, Viewer

sim = Simulation(dt=5e-5, integrator="verlet")

pivot = sim.add_particle(mass=1.0, position=(0.0, 0.0, 1.0), movable=False)
bob = sim.add_particle(mass=1.0, position=(0.0, 0.0, 0.0), velocity=(2.5, 0.0, 0.0))

# A very stiff spring acts like a rigid rod—no pendulum formula required
rod = sim.add_spring(pivot, bob, k=100_000.0)

gravity = Gravity((0.0, 0.0, -9.81))
sim.add_force(gravity)

viewer = Viewer(sim, title="Simple Pendulum", color_springs_by_tension=True)
viewer.add_trail(bob, max_points=200)
viewer.add_slider(
    "Gravity |g|", 0.0, 25.0, 9.81,
    on_change=lambda val: setattr(gravity, "g", (0.0, 0.0, -val)),
)
viewer.show()
```

---

## Example Demos

Run any example from the repository root:

| Example | Demonstrates |
| :--- | :--- |
| `python examples/simple_pendulum.py` | Quickstart pendulum with live gravity & damping sliders |
| `python examples/dzhanibekov.py` | Intermediate-axis theorem (Dzhanibekov flip) + angular momentum arrow |
| `python examples/gyroscope.py` | Gyroscopic precession & nutation vs. analytical rigid-body prediction |
| `python examples/coulomb_force.py` | Inverse-square Coulomb orbits with 4th-order symplectic energy conservation |
| `python examples/cube_vibration.py` | 4×4×4 3D elastic lattice with tension/compression wave coloring & live plots |
| `python examples/wave_propagation.py` | 45×45 (2,025-particle) 2D wave membrane with live frequency/amplitude controls |

---

## Documentation

### 1. Simulation, Particles & Springs

Create a simulation with a timestep `dt` and an integrator:

```python
from physics_sandbox import Simulation

sim = Simulation(
    dt=1e-4,                 # Integration timestep (seconds)
    integrator="verlet",     # "verlet" (default), "yoshida4", or "euler"
    damping=False,           # Default damping on newly added springs
    dissipation_coefficient=0.5,
    use_numba=True,          # Automatically uses Numba JIT when available
)
```

**Choosing an integrator** (all are single-step Newtonian integrators using only the current state):
* **`"verlet"` (Velocity Verlet, default):** 2nd-order symplectic and time-reversible. Fast (1 force evaluation per step) with bounded energy oscillation—a great default for most systems.
* **`"yoshida4"` (Yoshida 4th-Order Symplectic):** 4th-order symplectic (3 sub-kicks per step). Ideal for stiff-spring rigid bodies or orbital dynamics where ultra-low energy drift matters.
* **`"euler"` (Symplectic Euler):** 1st-order symplectic. Simple baseline, useful for comparing energy stability against Verlet and Yoshida.

**Adding particles and springs:**

```python
p1 = sim.add_particle(
    mass=1.0,
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),  # optional, defaults to (0, 0, 0)
    movable=True,              # set False to pin a particle in place
    charge=0.0,                # used by Coulomb and UniformEMField
    properties={"label": "A"}, # optional custom attributes
)

s = sim.add_spring(
    p1, p2,
    k=10_000.0,         # Hooke stiffness constant
    L0=None,            # Rest length (defaults to initial distance |p2 - p1|)
    b=None,             # Per-spring axial damping (defaults to 0, or dissipation_coefficient if damping=True)
    tension_only=False, # If True, acts like a rope/string (zero force when slack/compressed)
)
```

`Particle` (`p.pos`, `p.vel`, `p.acc`, `p.mass`, `p.movable`, `p.charge`) and `Spring` (`s.k`, `s.L0`, `s.b`, `s.tension_only`, `s.length()`, `s.extension()`, `s.strain()`, `s.energy()`) are lightweight handles backed directly by `Simulation`'s contiguous NumPy arrays (`sim.positions`, `sim.velocities`, `sim.masses`, etc.). Editing attributes on a handle updates the underlying arrays immediately.

**Stepping & resetting (headless usage):**

```python
sim.step(10_000)         # Advance by N substeps (runs in Numba when available)
sim.advance(0.5)         # Advance by approximately 0.5 seconds
sim.update()             # Alias for sim.step(1)

sim.save_initial_state() # Snapshot current positions, velocities, and time
sim.reset()              # Restore the saved snapshot
```

---

### 2. Built-in Force Laws

Attach any built-in force with `sim.add_force(...)`. Built-in forces are vectorized in NumPy and automatically compiled inside the Numba substep loop when `numba` is installed. Conservative forces also contribute to `sim.potential_energy()`.

| Force | Signature & Defaults | Physics / Behavior |
| :--- | :--- | :--- |
| **`Gravity`** | `Gravity(g=(0.0, 0.0, -9.81))` | Uniform gravitational field $\vec{F}_i = m_i \vec{g}$. Includes gravitational potential energy. Mutate `gravity.g` live to change gravity mid-run. |
| **`LinearDrag`** | `LinearDrag(gamma=0.1)` | Viscous damping $\vec{F}_i = -\gamma \vec{v}_i$, proportional to velocity (low-speed fluid resistance). |
| **`QuadraticDrag`** | `QuadraticDrag(c=0.01)` | Aerodynamic drag $\vec{F}_i = -c \|\vec{v}_i\| \vec{v}_i$, proportional to speed squared. |
| **`Coulomb`** | `Coulomb(k_e=1.0, softening=1e-6)` | Pairwise inverse-square electrostatic force between charged particles (`p.charge`): $\vec{F}_{ij} = k_e \frac{q_i q_j}{(r^2 + \epsilon^2)^{3/2}} \vec{r}_{ij}$. Evaluated symmetrically (conserves momentum) and includes electrostatic potential energy. `softening` ($\epsilon$) avoids singularities at near-zero separation. |
| **`NBodyGravity`** | `NBodyGravity(G=1.0, softening=1e-6)` | Pairwise Newtonian gravitation between all particles: $\vec{F}_{ij} = -G \frac{m_i m_j}{(r^2 + \epsilon^2)^{3/2}} \vec{r}_{ij}$. Symmetric and includes gravitational potential energy. |
| **`UniformEMField`** | `UniformEMField(E=(0,0,0), B=(0,0,0))` | Uniform electromagnetic Lorentz force $\vec{F}_i = q_i (\vec{E} + \vec{v}_i \times \vec{B})$ acting on charged particles. Includes electric potential energy from $\vec{E}$. |
| **`PlaneBoundary`** | `PlaneBoundary(normal=(0,0,1), offset=0.0, stiffness=1e5, damping=10.0)` | Pure Newtonian spring-damper floor or wall keeping particles in the half-space $\hat{n} \cdot \vec{r} \ge \text{offset}$. Penetrating particles feel a linear restoring spring force plus normal damping (no hard impulse discontinuities). |

**Example:**

```python
from physics_sandbox import Gravity, LinearDrag, PlaneBoundary

sim.add_force(Gravity((0.0, 0.0, -9.81)))
sim.add_force(LinearDrag(gamma=0.05))
sim.add_force(PlaneBoundary(normal=(0.0, 0.0, 1.0), offset=-2.0, stiffness=5e4, damping=10.0))
```

---

### 3. Custom "What-If" Fields & Kinematic Drivers

#### Custom Force Fields (`sim.add_field`)
Write any custom force law as a Python function:

```python
# Simple per-particle force function: returns (Fx, Fy, Fz) in Newtons
sim.add_field(lambda p: -0.5 * p.pos)

# Vectorized over the entire Simulation for faster NumPy execution:
sim.add_field(
    lambda s: -0.5 * s.positions,
    vectorized=True,
    potential=lambda s: float(0.25 * (s.positions ** 2).sum()),  # optional U(x)
)
```

*(Pass `returns_acceleration=True` if your function returns acceleration $\vec{a}$ instead of force $\vec{F}$.)*

#### Kinematic Drivers
Use drivers to prescribe the motion of specific particles over time (e.g., oscillating a membrane node):

* **`sim.add_harmonic_driver(particle, axis=(0,0,1), amplitude=0.1, frequency=1.0, phase=0.0)`:** Pins `particle` and moves it sinusoidally along `axis` (`pos = base + A sin(2πft + φ)`). Runs natively inside Numba and lets you tweak `driver.frequency` or `driver.amplitude` live.
* **`sim.add_driver(fn)`:** Registers an arbitrary Python callback `fn(sim)` called on every substep.

---

### 4. Geometry Builders

Builders construct common multi-particle structures in a single call and return the created `Particle` handles so you can easily kick, pin, or monitor individual nodes.

#### `add_chain(...)` — 1D chains, strings, and rods
```python
from physics_sandbox import add_chain

chain = add_chain(
    sim,
    n=20,                     # Number of particles
    start=(0.0, 0.0, 2.0),    # Start endpoint
    end=(3.0, 0.0, 2.0),      # End endpoint
    mass=0.2,
    k=5_000.0,
    b=0.1,                    # Optional per-spring damping
    L0=None,                  # Optional custom rest length
    pin_start=True,           # Pin first particle
    pin_end=False,            # Pin last particle
    tension_only=True,        # Rope/string behavior (no compression resistance)
)
# Returns: list[Particle] of length n
```

#### `add_grid_2d(...)` — 2D wave membranes & cloth sheets
```python
from physics_sandbox import add_grid_2d

grid = add_grid_2d(
    sim,
    nx=40, ny=40,             # Grid resolution in X and Y
    lx=4.0, ly=4.0,           # Physical dimensions
    origin=(0.0, 0.0, 0.0),   # Center of the sheet
    mass=1.0,
    k=200.0,
    b=0.05,
    rest_length_scale=0.1,    # < 1.0 pre-tensions the sheet (like a drumhead)
    pin_boundary=True,        # Clamp all 4 outer edges
    pin_corners=False,        # Or pin only the 4 corners (useful for hanging cloth)
    diagonals=False,          # Add shear diagonal springs in each cell
    tension_only=False,
)
# Returns: 2D list[list[Particle]] indexed as grid[i][j]
```

#### `add_lattice_3d(...)` — 3D elastic solids
```python
from physics_sandbox import add_lattice_3d

cube = add_lattice_3d(
    sim,
    nx=4, ny=4, nz=4,           # Grid dimensions
    spacing=1.0,                # Distance between adjacent nodes
    origin=(0.0, 0.0, 0.0),
    mass=1.0,
    k=500.0,
    b=0.2,
    connect_face_diagonals=True,  # Shear bracing on cube faces
    connect_body_diagonals=False, # Optional internal diagonal bracing
)
# Returns: 3D list[list[list[Particle]]] indexed as cube[i][j][m]
```

#### `connect_all_pairs(...)` — Pure-spring rigid clusters
```python
from physics_sandbox import connect_all_pairs

# Braces every pair among the given particles with stiff springs (great for rigid bodies)
connect_all_pairs(sim, [p1, p2, p3, p4, p5], k=1_000_000.0, b=0.0)
```

---

### 5. Interactive 3D Viewer (`Viewer`)

`Viewer` launches a GPU-accelerated Polyscope window with built-in playback and inspection controls (**Play/Pause**, **Step**, **Reset**, **Time Scale** slow-mo/fast-forward, live **Integrator** switcher, **Spring Tension** & **Particle Speed** colormaps, **Velocity/Acceleration** arrows, and live **Energy** sparklines):

```python
from physics_sandbox import Viewer

viewer = Viewer(
    sim,
    title="My Experiment",
    particle_radius=0.08,            # Set 0.0 to hide spheres on dense spring meshes
    scale_radius_by_mass=True,       # Scales radius by mass^(1/3)
    spring_radius=0.02,
    color_springs_by_tension=True,   # Blue = compression, Red = tension
    ground_plane="shadow_only",      # "shadow_only", "tile_reflection", "tile", or "none"
    time_scale=1.0,                  # Initial playback speed multiplier
    start_paused=False,
)
```

Extend any experiment with trails, overlays, live plots, and interactive **"What-If" controls**:

* `viewer.add_trail(particle, max_points=300, color=(0.15, 0.85, 0.35), radius=0.015)` — Attaches a 3D trajectory trail to a particle (or any `() -> np.ndarray` position callback).
* `viewer.add_marker(name, position_getter, color=..., radius=..., trail=False)` — Draws a custom 3D sphere (e.g., center of mass or an analytical prediction).
* `viewer.add_vector_overlay(name, origin_getter, vector_getter, color=..., scale=1.0)` — Draws a live 3D vector arrow (e.g., total angular momentum $\vec{L}$).
* `viewer.add_plot(label, lambda sim: ...)` — Adds a live sparkline plot to the sidebar.
* `viewer.add_slider(label, min_val, max_val, initial, on_change)` / `viewer.add_checkbox(label, initial, on_change)` — Exposes live parameters right in the sidebar (`sim.set_all_spring_stiffness` and `sim.set_all_spring_damping` pair nicely with sliders).

---

### 6. Diagnostics & Analysis

All diagnostics work both headless and inside viewer callbacks/plots:

* **Energy:** `sim.kinetic_energy()`, `sim.spring_energy()`, `sim.field_potential_energy()`, `sim.potential_energy()`, `sim.total_energy()`, `sim.get_energy()`
* **Momenta & Center of Mass:** `sim.center_of_mass(particles=None)`, `sim.center_of_mass_velocity(particles=None)`, `sim.linear_momentum(particles=None)`, `sim.angular_momentum(origin=None, particles=None)`
* **Spring Mechanics (vectorized arrays):** `sim.spring_lengths()`, `sim.spring_extensions()`, `sim.spring_strains()`, `sim.spring_tensions()`

---

## Running Tests

```bash
pytest
# or without pytest installed:
python -m unittest discover -s tests -v
```

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or an issue for suggestions, bug reports, or new demos.

---

## License

This project is licensed under the [MIT License](LICENSE).
