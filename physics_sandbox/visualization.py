"""Interactive GPU-accelerated 3D visualization powered by Polyscope.

Features:
- Right-handed Z-up coordinate system (matches physics directly—no axis mirroring)
- GPU-rendered point clouds & spring curve networks (handles thousands of objects)
- Live spring tension/compression colormapping, particle speed colormap, and
  velocity/acceleration vector overlays
- Smooth particle trails and custom analytical markers / vector overlays
- Built-in Dear ImGui sidebar with Play/Pause, Step, Reset, Time Scale,
  Integrator switcher, Live Energy telemetry + sparklines, and custom "what-if"
  sliders registered by examples
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Callable, Optional, Sequence

import numpy as np

from .simulation import Particle, Simulation


@dataclass
class _SliderSpec:
    label: str
    min_val: float
    max_val: float
    value: float
    on_change: Callable[[float], None]
    fmt: str = "%.3f"


@dataclass
class _CheckboxSpec:
    label: str
    value: bool
    on_change: Callable[[bool], None]


@dataclass
class _PlotSpec:
    label: str
    getter: Callable[[Simulation], float]
    history_len: int = 300
    data: deque = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.data = deque(maxlen=self.history_len)


@dataclass
class _TrailSpec:
    name: str
    getter: Callable[[], np.ndarray]
    max_points: int
    color: tuple[float, float, float]
    radius: float
    buffer: np.ndarray = field(init=False)
    initialized: bool = False
    ps_net: object = None

    def __post_init__(self) -> None:
        self.buffer = np.zeros((self.max_points, 3), dtype=np.float64)

    def reset(self) -> None:
        p = np.asarray(self.getter(), dtype=np.float64).reshape(3)
        self.buffer[:] = p
        self.initialized = True
        if self.ps_net is not None:
            self.ps_net.update_node_positions(self.buffer)

    def push(self) -> None:
        p = np.asarray(self.getter(), dtype=np.float64).reshape(3)
        if not self.initialized:
            self.buffer[:] = p
            self.initialized = True
        else:
            self.buffer[:-1] = self.buffer[1:]
            self.buffer[-1] = p
        if self.ps_net is not None:
            self.ps_net.update_node_positions(self.buffer)


@dataclass
class _MarkerSpec:
    name: str
    getter: Callable[[], np.ndarray]
    color: tuple[float, float, float]
    radius: float
    ps_cloud: object = None


@dataclass
class _VectorOverlaySpec:
    name: str
    origin_getter: Callable[[], np.ndarray]
    vec_getter: Callable[[], np.ndarray]
    color: tuple[float, float, float]
    scale: float
    radius: float
    ps_cloud: object = None


class Viewer:
    """Interactive Polyscope 3D viewer for a `Simulation`.

    Parameters
    ----------
    simulation : Simulation
        The simulation instance to visualize and drive.
    title : str
        Window/panel title.
    particle_radius : float
        Base radius for particles in world units. Set to `0.0` to hide particles
        initially (useful for dense wave sheets).
    scale_radius_by_mass : bool
        Scale each particle's radius by `mass ** (1/3)`.
    particle_color : tuple[float, float, float]
        RGB color in `[0, 1]`.
    spring_radius : float
        Spring cylinder radius in world units.
    spring_color : tuple[float, float, float]
        Default RGB color for springs when tension coloring is disabled.
    color_springs_by_tension : bool
        Start with blue/red compression/tension coloring enabled on springs.
    ground_plane : {"tile_reflection", "tile", "shadow_only", "none"}
        Polyscope ground plane style.
    time_scale : float
        Initial playback speed multiplier (`1.0` = real time).
    max_substeps_per_frame : int
        Upper bound on physics substeps executed per rendered frame so stiff
        or large scenes stay responsive.
    start_paused : bool
        If True, open the window paused until the user clicks Play.
    history_len : int
        Number of frames retained in the live energy and custom plots.
    """

    def __init__(
        self,
        simulation: Simulation,
        *,
        title: str = "Physics Sandbox",
        particle_radius: float = 0.08,
        scale_radius_by_mass: bool = True,
        particle_color: Sequence[float] = (0.18, 0.45, 0.92),
        spring_radius: float = 0.02,
        spring_color: Sequence[float] = (0.68, 0.70, 0.74),
        color_springs_by_tension: bool = False,
        ground_plane: str = "shadow_only",
        time_scale: float = 1.0,
        max_substeps_per_frame: int = 2000,
        start_paused: bool = False,
        history_len: int = 300,
        on_frame: Optional[Callable[[Simulation, float], None]] = None,
        on_reset: Optional[Callable[[Simulation], None]] = None,
    ) -> None:
        self.sim = simulation
        self.title = title
        self.particle_radius = float(particle_radius)
        self.scale_radius_by_mass = bool(scale_radius_by_mass)
        self.particle_color = tuple(float(c) for c in particle_color)
        self.spring_radius = float(spring_radius)
        self.spring_color = tuple(float(c) for c in spring_color)
        self.ground_plane = ground_plane
        self.time_scale = float(time_scale)
        self.max_substeps_per_frame = int(max_substeps_per_frame)
        self.paused = bool(start_paused)
        self.history_len = int(history_len)
        self.on_frame = on_frame
        self.on_reset = on_reset

        # Visual toggle state
        self._show_particles = self.particle_radius > 0.0
        self._show_springs = self.sim.n_springs > 0
        self._color_by_tension = bool(color_springs_by_tension)
        self._color_by_speed = False
        self._show_velocity_arrows = False
        self._show_acc_arrows = False
        self._show_trails = True
        self._tension_scale = 1.0

        # Registered extras
        self._sliders: list[_SliderSpec] = []
        self._checkboxes: list[_CheckboxSpec] = []
        self._plots: list[_PlotSpec] = []
        self._trails: list[_TrailSpec] = []
        self._markers: list[_MarkerSpec] = []
        self._vectors: list[_VectorOverlaySpec] = []

        # Telemetry buffers
        self._kin_hist: deque[float] = deque(maxlen=self.history_len)
        self._pot_hist: deque[float] = deque(maxlen=self.history_len)
        self._tot_hist: deque[float] = deque(maxlen=self.history_len)
        self._initial_energy: Optional[float] = None

        # Runtime handles created in `show()`
        self._ps = None
        self._psim = None
        self._ps_particles = None
        self._ps_springs = None
        self._last_wall_time: Optional[float] = None
        self._last_fps: float = 0.0
        self._last_substeps: int = 0
        self._accumulator: float = 0.0
        self._integrator_names = ("verlet", "yoshida4", "euler")

    # ------------------------------------------------------------------
    # Public registration API for examples & "what-if" exploration
    # ------------------------------------------------------------------

    def add_trail(
        self,
        target: Particle | Callable[[], np.ndarray],
        *,
        max_points: int = 250,
        color: Sequence[float] = (0.15, 0.82, 0.35),
        radius: float = 0.012,
        name: Optional[str] = None,
    ) -> None:
        """Attach a smooth 3D trajectory trail to a `Particle` or position callback."""
        if isinstance(target, Particle):
            p = target
            getter = lambda: p.pos  # noqa: E731
            trail_name = name or f"trail_p{p.index}"
        else:
            getter = target
            trail_name = name or f"trail_{len(self._trails)}"

        self._trails.append(
            _TrailSpec(
                name=trail_name,
                getter=getter,
                max_points=max(2, int(max_points)),
                color=tuple(float(c) for c in color),
                radius=float(radius),
            )
        )

    def add_marker(
        self,
        name: str,
        position_getter: Callable[[], np.ndarray],
        *,
        color: Sequence[float] = (0.95, 0.20, 0.20),
        radius: float = 0.09,
        trail: bool = False,
        trail_points: int = 250,
        trail_color: Optional[Sequence[float]] = None,
        trail_radius: float = 0.012,
    ) -> None:
        """Register a custom 3D marker sphere (e.g., center of mass or analytical prediction)."""
        rgb = tuple(float(c) for c in color)
        self._markers.append(
            _MarkerSpec(name=name, getter=position_getter, color=rgb, radius=float(radius))
        )
        if trail:
            self.add_trail(
                position_getter,
                max_points=trail_points,
                color=trail_color if trail_color is not None else rgb,
                radius=trail_radius,
                name=f"{name}_trail",
            )

    def add_vector_overlay(
        self,
        name: str,
        origin_getter: Callable[[], np.ndarray],
        vector_getter: Callable[[], np.ndarray],
        *,
        color: Sequence[float] = (0.0, 0.85, 0.90),
        scale: float = 1.0,
        radius: float = 0.025,
    ) -> None:
        """Display a live 3D vector arrow (e.g., total angular momentum)."""
        self._vectors.append(
            _VectorOverlaySpec(
                name=name,
                origin_getter=origin_getter,
                vec_getter=vector_getter,
                color=tuple(float(c) for c in color),
                scale=float(scale),
                radius=float(radius),
            )
        )

    def add_slider(
        self,
        label: str,
        min_val: float,
        max_val: float,
        initial: float,
        on_change: Callable[[float], None],
        fmt: str = "%.3f",
    ) -> None:
        """Add an interactive ImGui float slider to the sidebar for live 'what-if' tuning."""
        self._sliders.append(
            _SliderSpec(
                label=label,
                min_val=float(min_val),
                max_val=float(max_val),
                value=float(initial),
                on_change=on_change,
                fmt=fmt,
            )
        )

    def add_checkbox(
        self,
        label: str,
        initial: bool,
        on_change: Callable[[bool], None],
    ) -> None:
        """Add an interactive ImGui checkbox to the sidebar."""
        self._checkboxes.append(
            _CheckboxSpec(label=label, value=bool(initial), on_change=on_change)
        )

    def add_plot(
        self,
        label: str,
        getter: Callable[[Simulation], float],
        history_len: Optional[int] = None,
    ) -> None:
        """Add a live sparkline plot to the sidebar tracking `getter(sim)`."""
        self._plots.append(
            _PlotSpec(
                label=label,
                getter=getter,
                history_len=history_len or self.history_len,
            )
        )

    # ------------------------------------------------------------------
    # Polyscope initialization & per-frame synchronization
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        try:
            import polyscope as ps
            import polyscope.imgui as psim
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Polyscope is required for 3D visualization. "
                "Install it with: pip install polyscope"
            ) from exc

        self._ps = ps
        self._psim = psim

        ps.set_program_name(self.title)
        ps.set_verbosity(0)
        ps.init()
        ps.set_up_dir("z_up")
        ps.set_ground_plane_mode(self.ground_plane)
        ps.set_automatically_compute_scene_extents(True)

        # Save initial simulation snapshot for Reset
        self.sim.save_initial_state()

        # Register particles
        if self.sim.n_particles > 0:
            self._ps_particles = ps.register_point_cloud(
                "particles",
                self.sim.positions,
                color=self.particle_color,
                enabled=self._show_particles,
            )
            base_r = self.particle_radius if self.particle_radius > 0.0 else 0.05
            if self.scale_radius_by_mass and np.ptp(self.sim.masses) > 1e-12:
                radii = base_r * np.cbrt(self.sim.masses)
                self._ps_particles.add_scalar_quantity("radius", radii, enabled=False)
                self._ps_particles.set_point_radius_quantity("radius", autoscale=False)
            else:
                self._ps_particles.set_radius(base_r, relative=False)

        # Register springs
        if self.sim.n_springs > 0:
            self._ps_springs = ps.register_curve_network(
                "springs",
                self.sim.positions,
                self.sim.spring_edges,
                color=self.spring_color,
                enabled=self._show_springs,
            )
            self._ps_springs.set_radius(self.spring_radius, relative=False)

        # Register trails
        for trail in self._trails:
            edges = np.column_stack(
                (np.arange(trail.max_points - 1), np.arange(1, trail.max_points))
            )
            trail.reset()
            trail.ps_net = ps.register_curve_network(
                trail.name,
                trail.buffer,
                edges,
                color=trail.color,
                enabled=self._show_trails,
            )
            trail.ps_net.set_radius(trail.radius, relative=False)

        # Register markers
        for marker in self._markers:
            pos = np.asarray(marker.getter(), dtype=np.float64).reshape(1, 3)
            marker.ps_cloud = ps.register_point_cloud(
                marker.name,
                pos,
                color=marker.color,
            )
            marker.ps_cloud.set_radius(marker.radius, relative=False)

        # Register vector overlays
        for ov in self._vectors:
            origin = np.asarray(ov.origin_getter(), dtype=np.float64).reshape(1, 3)
            ov.ps_cloud = ps.register_point_cloud(ov.name, origin, color=ov.color)
            ov.ps_cloud.set_radius(ov.radius * 0.5, relative=False)

        self._record_telemetry(reset_baseline=True)
        self._sync_visuals()

    def _record_telemetry(self, reset_baseline: bool = False) -> None:
        kin, pot = self.sim.get_energy()
        tot = kin + pot
        if reset_baseline or self._initial_energy is None:
            self._initial_energy = tot
        self._kin_hist.append(kin)
        self._pot_hist.append(pot)
        self._tot_hist.append(tot)
        for plot in self._plots:
            plot.data.append(float(plot.getter(self.sim)))

    def _sync_visuals(self) -> None:
        if self._ps_particles is not None:
            self._ps_particles.set_enabled(self._show_particles)
            if self._show_particles:
                self._ps_particles.update_point_positions(self.sim.positions)
                if self._color_by_speed:
                    speeds = np.linalg.norm(self.sim.velocities, axis=1)
                    self._ps_particles.add_scalar_quantity(
                        "speed", speeds, cmap="viridis", enabled=True
                    )
                if self._show_velocity_arrows:
                    self._ps_particles.add_vector_quantity(
                        "velocity",
                        self.sim.velocities,
                        color=(0.15, 0.85, 0.45),
                        enabled=True,
                    )
                if self._show_acc_arrows:
                    self._ps_particles.add_vector_quantity(
                        "acceleration",
                        self.sim.accelerations,
                        color=(0.95, 0.35, 0.15),
                        enabled=True,
                    )

        if self._ps_springs is not None:
            self._ps_springs.set_enabled(self._show_springs)
            if self._show_springs:
                self._ps_springs.update_node_positions(self.sim.positions)
                if self._color_by_tension:
                    tensions = self.sim.spring_tensions()
                    peak = float(np.max(np.abs(tensions))) if len(tensions) else 1.0
                    self._tension_scale = max(self._tension_scale * 0.995, peak, 1e-6)
                    self._ps_springs.add_scalar_quantity(
                        "spring_tension",
                        tensions,
                        defined_on="edges",
                        cmap="coolwarm",
                        vminmax=(-self._tension_scale, self._tension_scale),
                        enabled=True,
                    )

        for marker in self._markers:
            if marker.ps_cloud is not None:
                pos = np.asarray(marker.getter(), dtype=np.float64).reshape(1, 3)
                marker.ps_cloud.update_point_positions(pos)

        for ov in self._vectors:
            if ov.ps_cloud is not None:
                origin = np.asarray(ov.origin_getter(), dtype=np.float64).reshape(1, 3)
                vec = (
                    np.asarray(ov.vec_getter(), dtype=np.float64).reshape(1, 3) * ov.scale
                )
                ov.ps_cloud.update_point_positions(origin)
                ov.ps_cloud.add_vector_quantity(
                    ov.name,
                    vec,
                    vectortype="ambient",
                    color=ov.color,
                    radius=ov.radius,
                    enabled=True,
                )

    def _reset_all(self) -> None:
        self.sim.reset()
        self._accumulator = 0.0
        self._tension_scale = 1.0
        self._kin_hist.clear()
        self._pot_hist.clear()
        self._tot_hist.clear()
        for plot in self._plots:
            plot.data.clear()
        if self.on_reset is not None:
            self.on_reset(self.sim)
        for trail in self._trails:
            trail.reset()
        self._record_telemetry(reset_baseline=True)
        self._sync_visuals()

    def _step_one_frame(self, frame_dt: float) -> None:
        self._accumulator += frame_dt * self.time_scale
        n_steps = int(self._accumulator / self.sim.dt)
        if n_steps <= 0:
            n_steps = 1
            self._accumulator = 0.0
        else:
            if n_steps > self.max_substeps_per_frame:
                n_steps = self.max_substeps_per_frame
                self._accumulator = 0.0
            else:
                self._accumulator -= n_steps * self.sim.dt

        self.sim.step(n_steps)
        self._last_substeps = n_steps

        if self.on_frame is not None:
            self.on_frame(self.sim, n_steps * self.sim.dt)

        for trail in self._trails:
            trail.push()

        self._record_telemetry()
        self._sync_visuals()

    # ------------------------------------------------------------------
    # ImGui frame callback
    # ------------------------------------------------------------------

    def _callback(self) -> None:
        psim = self._psim
        now = time.perf_counter()
        if self._last_wall_time is None:
            wall_dt = 1.0 / 60.0
        else:
            wall_dt = max(1e-4, now - self._last_wall_time)
        self._last_wall_time = now
        self._last_fps = 0.9 * self._last_fps + 0.1 * (1.0 / wall_dt)
        clamped_dt = min(wall_dt, 0.05)

        # --- Playback Controls ---
        psim.TextUnformatted(f"{self.title}  [{self.sim.backend}]")
        psim.TextUnformatted(
            f"t = {self.sim.time:.4f} s | {self._last_fps:.0f} FPS | "
            f"{self._last_substeps} substeps/frame"
        )
        psim.TextUnformatted(
            f"Particles: {self.sim.n_particles}   Springs: {self.sim.n_springs}"
        )
        psim.Separator()

        if psim.Button("Play" if self.paused else "Pause"):
            self.paused = not self.paused
        psim.SameLine()
        step_clicked = psim.Button("Step")
        psim.SameLine()
        if psim.Button("Reset"):
            self._reset_all()

        _, self.time_scale = psim.SliderFloat(
            "Time Scale", self.time_scale, 0.01, 5.0, format="%.2fx"
        )

        # --- Integrator Selector ---
        psim.TextUnformatted("Integrator (single-step Newtonian):")
        current_idx = self._integrator_names.index(self.sim.integrator)
        for idx, name in enumerate(self._integrator_names):
            if idx > 0:
                psim.SameLine()
            changed, current_idx = psim.RadioButton(name, current_idx, idx)
            if changed:
                self.sim.integrator = self._integrator_names[current_idx]

        # --- Visual Toggles ---
        psim.Separator()
        if self._ps_particles is not None:
            c, self._show_particles = psim.Checkbox("Show Particles", self._show_particles)
            if c:
                self._ps_particles.set_enabled(self._show_particles)
        if self._ps_springs is not None:
            c, self._show_springs = psim.Checkbox("Show Springs", self._show_springs)
            if c:
                self._ps_springs.set_enabled(self._show_springs)
            c, self._color_by_tension = psim.Checkbox(
                "Color Springs by Tension", self._color_by_tension
            )
            if c and not self._color_by_tension:
                self._ps_springs.set_color(self.spring_color)
                try:
                    self._ps_springs.remove_quantity("spring_tension")
                except Exception:
                    pass
        if self._ps_particles is not None:
            c, self._color_by_speed = psim.Checkbox(
                "Color Particles by Speed", self._color_by_speed
            )
            if c and not self._color_by_speed:
                self._ps_particles.set_color(self.particle_color)
                try:
                    self._ps_particles.remove_quantity("speed")
                except Exception:
                    pass
            c, self._show_velocity_arrows = psim.Checkbox(
                "Velocity Arrows", self._show_velocity_arrows
            )
            if c and not self._show_velocity_arrows:
                try:
                    self._ps_particles.remove_quantity("velocity")
                except Exception:
                    pass
            c, self._show_acc_arrows = psim.Checkbox(
                "Acceleration Arrows", self._show_acc_arrows
            )
            if c and not self._show_acc_arrows:
                try:
                    self._ps_particles.remove_quantity("acceleration")
                except Exception:
                    pass
        if self._trails:
            c, self._show_trails = psim.Checkbox("Show Trails", self._show_trails)
            if c:
                for trail in self._trails:
                    if trail.ps_net is not None:
                        trail.ps_net.set_enabled(self._show_trails)

        # --- Custom "What-If" Sliders & Checkboxes ---
        if self._sliders or self._checkboxes:
            psim.Separator()
            psim.TextUnformatted("What-If Parameters:")
            for slider in self._sliders:
                changed, new_val = psim.SliderFloat(
                    slider.label,
                    slider.value,
                    slider.min_val,
                    slider.max_val,
                    format=slider.fmt,
                )
                if changed:
                    slider.value = new_val
                    slider.on_change(new_val)
                    self.sim.invalidate()
            for cb in self._checkboxes:
                changed, new_val = psim.Checkbox(cb.label, cb.value)
                if changed:
                    cb.value = new_val
                    cb.on_change(new_val)
                    self.sim.invalidate()

        # --- Live Energy & Custom Plots ---
        psim.Separator()
        kin = self._kin_hist[-1] if self._kin_hist else 0.0
        pot = self._pot_hist[-1] if self._pot_hist else 0.0
        tot = self._tot_hist[-1] if self._tot_hist else 0.0
        drift_str = ""
        if self._initial_energy is not None and abs(self._initial_energy) > 1e-9:
            drift_pct = 100.0 * (tot - self._initial_energy) / abs(self._initial_energy)
            drift_str = f" ({drift_pct:+.3f}%)"
        psim.TextUnformatted(f"E_total = {tot:.4f}{drift_str}")
        psim.TextUnformatted(f"E_kin = {kin:.4f} | E_pot = {pot:.4f}")

        if len(self._tot_hist) >= 2:
            psim.PlotLines("Total E", np.asarray(self._tot_hist, dtype=np.float32))
            psim.PlotLines("Kinetic E", np.asarray(self._kin_hist, dtype=np.float32))
            psim.PlotLines("Potential E", np.asarray(self._pot_hist, dtype=np.float32))

        for plot in self._plots:
            if len(plot.data) >= 2:
                psim.PlotLines(plot.label, np.asarray(plot.data, dtype=np.float32))

        # --- Advance Simulation ---
        if not self.paused:
            self._step_one_frame(clamped_dt)
        elif step_clicked:
            self._step_one_frame(1.0 / 60.0)

    def show(self) -> None:
        """Initialize the Polyscope window and enter the interactive event loop."""
        self._setup_scene()
        self._ps.set_user_callback(self._callback)
        self._ps.show()


# Alias matching the previous class name
Visualization = Viewer
