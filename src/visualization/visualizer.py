"""
src/visualization/visualizer.py

Step 6: Visualization

Three interactive Plotly views:
    1. orbit_plot()     — 3D satellite trajectories in ECI space
    2. risk_dashboard() — conjunction event table colored by Pc risk level
    3. timeline_view()  — TCA events across 24-hour window sized by Pc

All outputs are standalone HTML files — open in any browser, no server needed.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.models import Trajectory
from src.conjunction.conjunction_models import ConjunctionEvent

logger = logging.getLogger(__name__)

# ── Color scheme ──────────────────────────────────────────────────────────────
RISK_COLORS = {
    "red":    "#FF4B4B",
    "yellow": "#FFD700",
    "green":  "#00CC96",
    "none":   "#636EFA",
}

ORBIT_COLORS = [
    "#58A6FF", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF",
    "#FECB52", "#00B5F7", "#E4E4E4", "#A2A2A2",
]

BG_COLOR    = "#0D1117"
PANEL_COLOR = "#161B22"
GRID_COLOR  = "#21262D"
TEXT_COLOR  = "#E6EDF3"
MUTED_COLOR = "#8B949E"


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Convert hex color to rgba string for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _event_risk_level(event: ConjunctionEvent) -> str:
    """Get risk level string from event — handles None Pc gracefully."""
    if event.pc is None:
        return "none"
    if event.pc >= 1e-4:
        return "red"
    elif event.pc >= 1e-5:
        return "yellow"
    else:
        return "green"


def _position_at_tca(
    traj: Trajectory,
    tca: datetime,
) -> Optional[np.ndarray]:
    """Find the position in the trajectory closest to the TCA time."""
    if not traj.states:
        return None
    min_dt = float("inf")
    closest_pos = None
    for sv in traj.states:
        dt = abs((sv.epoch - tca).total_seconds())
        if dt < min_dt:
            min_dt = dt
            closest_pos = sv.position
    return closest_pos


class SatelliteVisualizer:
    """
    Generates interactive HTML visualizations from pipeline outputs.

    Usage:
        viz = SatelliteVisualizer(output_dir="data/visualizations")
        viz.orbit_plot(trajectories, events)
        viz.risk_dashboard(events)
        viz.timeline_view(events)
    """

    def __init__(self, output_dir: str = "data/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. 3D Orbit Plot ──────────────────────────────────────────────────────

    def orbit_plot(
        self,
        trajectories: dict[int, Trajectory],
        events: Optional[list[ConjunctionEvent]] = None,
        max_satellites: int = 20,
        output_file: str = "orbit_plot.html",
    ) -> str:
        """3D interactive plot of satellite trajectories in ECI space."""
        fig = go.Figure()

        # ── Earth sphere ──────────────────────────────────────────────────────
        R_EARTH = 6371.0
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 40)
        xe = R_EARTH * np.outer(np.cos(u), np.sin(v))
        ye = R_EARTH * np.outer(np.sin(u), np.sin(v))
        ze = R_EARTH * np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(go.Surface(
            x=xe, y=ye, z=ze,
            colorscale=[[0, "#0A3060"], [0.5, "#1A5FA8"], [1, "#2E86C1"]],
            showscale=False,
            opacity=0.85,
            name="Earth",
            hoverinfo="skip",
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3),
        ))

        # ── Satellite orbits ──────────────────────────────────────────────────
        traj_list = list(trajectories.values())[:max_satellites]

        for idx, traj in enumerate(traj_list):
            color = ORBIT_COLORS[idx % len(ORBIT_COLORS)]
            positions = traj.positions()[::5]
            altitudes = np.linalg.norm(positions, axis=1) - R_EARTH
            speeds = [np.linalg.norm(sv.velocity) for sv in traj.states[::5]]

            hover_text = [
                f"<b>{traj.name}</b><br>Alt: {alt:.1f} km<br>Speed: {spd:.2f} km/s"
                for alt, spd in zip(altitudes, speeds)
            ]

            fig.add_trace(go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                mode="lines",
                name=traj.name,
                line=dict(color=color, width=2),
                hovertext=hover_text,
                hoverinfo="text",
            ))

            fig.add_trace(go.Scatter3d(
                x=[positions[0, 0]], y=[positions[0, 1]], z=[positions[0, 2]],
                mode="markers",
                marker=dict(size=4, color=color),
                showlegend=False,
                hovertext=f"<b>{traj.name}</b> — current position",
                hoverinfo="text",
            ))

        # ── Conjunction event markers ─────────────────────────────────────────
        if events:
            for event in events:
                risk = _event_risk_level(event)
                color = RISK_COLORS[risk]
                size = 10 if risk == "red" else 7 if risk == "yellow" else 5
                traj = trajectories.get(event.primary_id)
                if traj is None:
                    continue
                pos = _position_at_tca(traj, event.tca)
                if pos is None:
                    continue
                pc_str = f"{event.pc:.2e}" if event.pc else "N/A"
                hover = (
                    f"<b>⚠ CONJUNCTION</b><br>"
                    f"{event.primary_name} / {event.secondary_name}<br>"
                    f"TCA: {event.tca.strftime('%H:%M:%S UTC')}<br>"
                    f"Miss: {event.miss_distance:.3f} km<br>"
                    f"Pc: {pc_str}<br>Risk: {risk.upper()}"
                )
                fig.add_trace(go.Scatter3d(
                    x=[pos[0]], y=[pos[1]], z=[pos[2]],
                    mode="markers",
                    marker=dict(size=size, color=color, symbol="diamond",
                                line=dict(color="white", width=1)),
                    name=f"⚠ {event.primary_name}/{event.secondary_name}",
                    hovertext=hover,
                    hoverinfo="text",
                ))

        fig.update_layout(
            title=dict(text="Satellite Collision Risk System — 3D Orbit View",
                       font=dict(size=18, color=TEXT_COLOR), x=0.5),
            scene=dict(
                bgcolor=BG_COLOR,
                xaxis=dict(title="X (km)", color=MUTED_COLOR,
                           gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
                yaxis=dict(title="Y (km)", color=MUTED_COLOR,
                           gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
                zaxis=dict(title="Z (km)", color=MUTED_COLOR,
                           gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            ),
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=PANEL_COLOR,
            font=dict(color=TEXT_COLOR),
            legend=dict(bgcolor=PANEL_COLOR, bordercolor=GRID_COLOR,
                        font=dict(color=TEXT_COLOR, size=10)),
            margin=dict(l=0, r=0, t=50, b=0),
            height=700,
        )

        output_path = self.output_dir / output_file
        fig.write_html(str(output_path))
        logger.info("Orbit plot saved → %s", output_path)
        return str(output_path)

    # ── 2. Risk Dashboard ─────────────────────────────────────────────────────

    def risk_dashboard(
        self,
        events: list[ConjunctionEvent],
        output_file: str = "risk_dashboard.html",
    ) -> str:
        """Interactive risk dashboard — Pc bar chart + conjunction event table."""
        if not events:
            logger.warning("No events to display in risk dashboard")
            return ""

        n_red    = sum(1 for e in events if _event_risk_level(e) == "red")
        n_yellow = sum(1 for e in events if _event_risk_level(e) == "yellow")
        n_green  = sum(1 for e in events if _event_risk_level(e) == "green")

        # ── Two-panel layout: bar chart on top, table below ───────────────────
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.45, 0.55],
            subplot_titles=["Collision Probability (Pc) by Event",
                            "Conjunction Event Details"],
            vertical_spacing=0.12,
            specs=[[{"type": "xy"}], [{"type": "table"}]],
        )

        # ── Bar chart ─────────────────────────────────────────────────────────
        labels = [f"{e.primary_name[:12]}/{e.secondary_name[:12]}" for e in events]
        pc_values  = [e.pc if e.pc else 0.0 for e in events]
        bar_colors = [RISK_COLORS[_event_risk_level(e)] for e in events]

        fig.add_trace(go.Bar(
            x=labels,
            y=pc_values,
            marker_color=bar_colors,
            marker_line=dict(color=GRID_COLOR, width=0.5),
            hovertemplate="<b>%{x}</b><br>Pc: %{y:.3e}<br><extra></extra>",
            name="Pc",
        ), row=1, col=1)

        fig.add_hline(y=1e-4, line_dash="dash", line_color=RISK_COLORS["red"],
                      line_width=1.5, annotation_text="Red threshold (1e-4)",
                      annotation_font_color=RISK_COLORS["red"], row=1, col=1)
        fig.add_hline(y=1e-5, line_dash="dash", line_color=RISK_COLORS["yellow"],
                      line_width=1.5, annotation_text="Yellow threshold (1e-5)",
                      annotation_font_color=RISK_COLORS["yellow"], row=1, col=1)

        # ── Table ─────────────────────────────────────────────────────────────
        risk_icons = {"red": "🔴", "yellow": "🟡", "green": "🟢", "none": "⚪"}

        row_colors = [_hex_to_rgba(RISK_COLORS[_event_risk_level(e)]) for e in events]
        table_colors = [row_colors for _ in range(7)]

        fig.add_trace(go.Table(
            header=dict(
                values=["Risk", "Primary", "Secondary", "TCA (UTC)",
                        "Miss (km)", "V_rel (km/s)", "Pc"],
                fill_color=PANEL_COLOR,
                font=dict(color=TEXT_COLOR, size=12),
                line_color=GRID_COLOR,
                align="left",
            ),
            cells=dict(
                values=[
                    [risk_icons[_event_risk_level(e)] for e in events],
                    [e.primary_name for e in events],
                    [e.secondary_name for e in events],
                    [e.tca.strftime("%Y-%m-%d %H:%M:%S") for e in events],
                    [f"{e.miss_distance:.3f}" for e in events],
                    [f"{e.relative_velocity:.3f}" for e in events],
                    [f"{e.pc:.3e}" if e.pc else "N/A" for e in events],
                ],
                fill_color=table_colors,
                font=dict(color=TEXT_COLOR, size=11),
                line_color=GRID_COLOR,
                align="left",
                height=28,
            ),
        ), row=2, col=1)

        fig.update_layout(
            title=dict(
                text=(f"Conjunction Risk Dashboard  —  "
                      f"🔴 {n_red} Red  🟡 {n_yellow} Yellow  🟢 {n_green} Green"),
                font=dict(size=17, color=TEXT_COLOR),
                x=0.5,
            ),
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=PANEL_COLOR,
            font=dict(color=TEXT_COLOR),
            yaxis=dict(type="log", title="Pc (log scale)",
                       gridcolor=GRID_COLOR, color=MUTED_COLOR),
            xaxis=dict(tickangle=-35, gridcolor=GRID_COLOR, color=MUTED_COLOR),
            showlegend=False,
            height=850,
            margin=dict(l=60, r=40, t=80, b=20),
        )

        output_path = self.output_dir / output_file
        fig.write_html(str(output_path))
        logger.info("Risk dashboard saved → %s", output_path)
        return str(output_path)

    # ── 3. Timeline View ──────────────────────────────────────────────────────

    def timeline_view(
        self,
        events: list[ConjunctionEvent],
        output_file: str = "timeline.html",
    ) -> str:
        """24-hour timeline of conjunction events sized by Pc."""
        if not events:
            logger.warning("No events for timeline")
            return ""

        tca_times   = [e.tca for e in events]
        miss_dists  = [e.miss_distance for e in events]
        pc_values   = [e.pc if e.pc else 1e-10 for e in events]
        risk_levels = [_event_risk_level(e) for e in events]
        labels      = [f"{e.primary_name} / {e.secondary_name}" for e in events]

        log_pc  = [-np.log10(max(pc, 1e-12)) for pc in pc_values]
        max_log = max(log_pc) if log_pc else 1
        sizes   = [max(8, 40 * (1 - lp / (max_log + 1))) + 8 for lp in log_pc]

        hover_texts = [
            f"<b>{label}</b><br>"
            f"TCA: {tca.strftime('%Y-%m-%d %H:%M:%S UTC')}<br>"
            f"Miss: {miss:.3f} km<br>Pc: {pc:.3e}<br>Risk: {risk.upper()}"
            for label, tca, miss, pc, risk
            in zip(labels, tca_times, miss_dists, pc_values, risk_levels)
        ]

        fig = go.Figure()

        for risk in ["red", "yellow", "green", "none"]:
            mask = [r == risk for r in risk_levels]
            if not any(mask):
                continue
            fig.add_trace(go.Scatter(
                x=[t for t, m in zip(tca_times, mask) if m],
                y=[d for d, m in zip(miss_dists, mask) if m],
                mode="markers+text",
                marker=dict(
                    size=[s for s, m in zip(sizes, mask) if m],
                    color=RISK_COLORS[risk],
                    opacity=0.85,
                    line=dict(color="white", width=1),
                ),
                text=[l.split("/")[0].strip() for l, m in zip(labels, mask) if m],
                textposition="top center",
                textfont=dict(size=9, color=TEXT_COLOR),
                hovertext=[h for h, m in zip(hover_texts, mask) if m],
                hoverinfo="text",
                name=f"{risk.capitalize()} risk",
            ))

        fig.add_hline(y=0.2, line_dash="dot", line_color=RISK_COLORS["red"],
                      line_width=1, annotation_text="200m hard-body threshold",
                      annotation_font_color=RISK_COLORS["red"],
                      annotation_position="right")

        fig.update_layout(
            title=dict(text="Conjunction Timeline — 24h Screening Window",
                       font=dict(size=17, color=TEXT_COLOR), x=0.5),
            xaxis=dict(title="Time (UTC)", gridcolor=GRID_COLOR,
                       color=MUTED_COLOR, showgrid=True),
            yaxis=dict(title="Miss Distance (km)", gridcolor=GRID_COLOR,
                       color=MUTED_COLOR, showgrid=True),
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=PANEL_COLOR,
            font=dict(color=TEXT_COLOR),
            legend=dict(bgcolor=PANEL_COLOR, bordercolor=GRID_COLOR,
                        font=dict(color=TEXT_COLOR)),
            height=550,
            margin=dict(l=60, r=40, t=70, b=60),
        )

        output_path = self.output_dir / output_file
        fig.write_html(str(output_path))
        logger.info("Timeline saved → %s", output_path)
        return str(output_path)