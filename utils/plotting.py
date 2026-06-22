"""Reusable Plotly chart builders for the Streamlit dashboard.

All builders return ``plotly.graph_objects.Figure`` objects styled with a dark
template so the dashboard has a consistent professional look.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

try:  # plotly is only needed for the dashboard / notebooks
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency
    go = None

_TEMPLATE = "plotly_dark"
_PALETTE = ["#4C9BE8", "#E8794C", "#6BCB77", "#FFD93D", "#C780E8", "#FF6B6B"]


def _require_plotly():
    if go is None:  # pragma: no cover
        raise ImportError("plotly is required for plotting; pip install plotly")


def smile_figure(
    strikes: np.ndarray,
    smiles: Dict[str, np.ndarray],
    title: str = "Implied Volatility Smile",
    x_title: str = "Strike",
) -> "go.Figure":
    """Overlay several IV smiles (one trace per dict entry)."""
    _require_plotly()
    fig = go.Figure()
    for i, (label, iv) in enumerate(smiles.items()):
        fig.add_trace(
            go.Scatter(
                x=strikes,
                y=np.asarray(iv) * 100,
                mode="lines+markers",
                name=label,
                line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            )
        )
    fig.update_layout(
        template=_TEMPLATE,
        title=title,
        xaxis_title=x_title,
        yaxis_title="Implied Vol (%)",
        hovermode="x unified",
    )
    return fig


def surface_figure(
    strikes: np.ndarray,
    maturities: np.ndarray,
    iv: np.ndarray,
    title: str = "Implied Volatility Surface",
) -> "go.Figure":
    """3-D IV surface."""
    _require_plotly()
    fig = go.Figure(
        data=[
            go.Surface(
                x=strikes,
                y=maturities,
                z=np.asarray(iv) * 100,
                colorscale="Viridis",
                colorbar=dict(title="IV %"),
            )
        ]
    )
    fig.update_layout(
        template=_TEMPLATE,
        title=title,
        scene=dict(xaxis_title="Strike", yaxis_title="Maturity", zaxis_title="IV (%)"),
        height=600,
    )
    return fig


def fan_chart(
    t: np.ndarray,
    mean: np.ndarray,
    bands: Sequence[np.ndarray],
    sample_paths: Optional[np.ndarray] = None,
    title: str = "Fan Chart",
    y_title: str = "Value",
) -> "go.Figure":
    """Mean +/- sigma bands with optional sample paths."""
    _require_plotly()
    fig = go.Figure()
    for k, band in enumerate(bands):
        lo, hi = band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself",
                fillcolor=f"rgba(76,155,232,{0.15 - 0.04 * k})",
                line=dict(width=0),
                name=f"±{k + 1}σ",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(x=t, y=mean, line=dict(color="#FFD93D", width=3), name="E[·]")
    )
    if sample_paths is not None:
        for p in sample_paths:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=p,
                    line=dict(color="rgba(255,255,255,0.18)", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    fig.update_layout(
        template=_TEMPLATE, title=title, xaxis_title="Time (years)", yaxis_title=y_title
    )
    return fig


def line_figure(x, y, title="", x_title="x", y_title="y", log_x=False, log_y=False):
    """Simple single-line chart."""
    _require_plotly()
    fig = go.Figure(
        go.Scatter(
            x=x, y=y, mode="lines+markers", line=dict(color=_PALETTE[0], width=2)
        )
    )
    fig.update_layout(
        template=_TEMPLATE, title=title, xaxis_title=x_title, yaxis_title=y_title
    )
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def heatmap_figure(x, y, z, title="", colorscale="RdBu", zmid=0.0):
    """Residual / error heatmap."""
    _require_plotly()
    fig = go.Figure(go.Heatmap(x=x, y=y, z=z, colorscale=colorscale, zmid=zmid))
    fig.update_layout(
        template=_TEMPLATE, title=title, xaxis_title="Strike", yaxis_title="Maturity"
    )
    return fig


def waterfall_figure(labels, values, title="P&L Attribution"):
    """P&L attribution waterfall."""
    _require_plotly()
    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=["relative"] * len(labels),
            connector=dict(line=dict(color="rgba(255,255,255,0.3)")),
        )
    )
    fig.update_layout(template=_TEMPLATE, title=title, yaxis_title="P&L")
    return fig
