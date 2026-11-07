"""Streamlit dashboard for the stochastic-vol-models library.

Run with::

    streamlit run app.py

Eight tabs cover parameter exploration, variance simulation, Fourier pricing,
the IV surface, calibration, model comparison, risk/Greeks and documentation.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from calibration.heston_calibration import HestonCalibrator
from calibration.market_data import MarketIVSurface
from characteristic_functions.cf_properties import CFAnalyzer
from diagnostics.variance_process import VarianceProcessDiagnostics
from fourier_pricing.carr_madan import CarrMadanPricer
from fourier_pricing.cos_method import COSMethodPricer
from fourier_pricing.fourier_diagnostics import FourierDiagnostics
from implied_vol.arbitrage_checks import ArbitrageFreeChecker
from implied_vol.smile_analytics import SmileAnalytics
from implied_vol.surface import IVSurface
from models.heston import HestonModel, HestonParams
from risk.greeks_heston import HestonGreeks
from risk.pnl_attribution import VolPnLAttribution
from simulation.heston_qe import HestonQESimulator
from utils import plotting as plt

st.set_page_config(page_title="Stochastic Vol Models", layout="wide", page_icon="📈")

# ---------------------------------------------------------------------------
# Cached compute helpers
# ---------------------------------------------------------------------------


def _params(kappa, theta, xi, rho, V0) -> HestonParams:
    return HestonParams(kappa, theta, xi, rho, V0)


@st.cache_data(show_spinner=False)
def cached_surface(pv, S0, r, q, strikes, mats):
    p = HestonParams(*pv)
    return IVSurface.from_heston(p, S0, r, q, np.array(strikes), np.array(mats)).iv


@st.cache_data(show_spinner=False)
def cached_smile(pv, S0, r, q, strikes, T):
    p = HestonParams(*pv)
    return IVSurface.from_heston(p, S0, r, q, np.array(strikes), np.array([T])).iv[0]


@st.cache_data(show_spinner=True)
def cached_paths(pv, S0, T, r, q, n_paths, n_steps, scheme, seed):
    p = HestonParams(*pv)
    from simulation.path_manager import PathManager

    res = PathManager().simulate(S0, p.V0, T, r, q, p, n_paths, n_steps, scheme, seed)
    return res.S_paths, res.V_paths


# ---------------------------------------------------------------------------
# Sidebar — global Heston parameters
# ---------------------------------------------------------------------------
st.sidebar.title("Heston Parameters")
kappa = st.sidebar.slider("κ  (mean reversion)", 0.1, 10.0, 2.0, 0.1)
theta = st.sidebar.slider("θ  (long-run variance)", 0.005, 0.25, 0.04, 0.005)
xi = st.sidebar.slider("ξ  (vol-of-vol)", 0.05, 2.0, 0.5, 0.05)
rho = st.sidebar.slider("ρ  (spot-vol corr)", -0.99, 0.99, -0.7, 0.01)
V0 = st.sidebar.slider("V₀  (initial variance)", 0.005, 0.25, 0.04, 0.005)
st.sidebar.markdown("---")
S0 = st.sidebar.number_input("Spot S₀", 1.0, 100000.0, 100.0)
r = st.sidebar.number_input("Rate r", -0.05, 0.20, 0.02, 0.005, format="%.3f")
q = st.sidebar.number_input("Dividend q", 0.0, 0.20, 0.0, 0.005, format="%.3f")

pv = (kappa, theta, xi, rho, V0)
model = HestonModel(HestonParams(*pv), validate=False)

feller = model.feller_ratio
if model.feller_satisfied:
    st.sidebar.success(f"Feller SATISFIED  (2κθ/ξ² = {feller:.2f})")
else:
    st.sidebar.error(f"Feller VIOLATED  (2κθ/ξ² = {feller:.2f})")

st.title("📈 Stochastic Volatility Modeling — Heston Engine")

tabs = st.tabs(
    [
        "1 · Parameter Explorer",
        "2 · Variance Simulator",
        "3 · Fourier Pricing",
        "4 · IV Surface",
        "5 · Calibration",
        "6 · Model Comparison",
        "7 · Risk & Greeks",
        "8 · Documentation",
    ]
)

# ---------------------------------------------------------------------------
# Tab 1 — Parameter Explorer
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Live model readouts")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Long-run vol √θ", f"{model.long_run_vol:.1%}")
    c2.metric("Initial vol √V₀", f"{model.initial_vol:.1%}")
    c3.metric("Norm. vol-of-vol", f"{model.vol_of_vol:.2f}")
    c4.metric("Var half-life", f"{model.half_life:.2f} y")

    strikes = np.linspace(0.7 * S0, 1.3 * S0, 31)
    st.markdown("#### Implied-volatility smile by maturity")
    smiles = {}
    for label, T in [
        ("1M", 1 / 12),
        ("3M", 0.25),
        ("6M", 0.5),
        ("1Y", 1.0),
        ("2Y", 2.0),
    ]:
        smiles[label] = cached_smile(pv, S0, r, q, tuple(strikes), T)
    st.plotly_chart(
        plt.smile_figure(strikes, smiles, "Smile term structure"),
        use_container_width=True,
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### ρ effect on skew (T = 6M)")
        rho_smiles = {}
        for rr in [-0.9, -0.5, 0.0, 0.5, 0.9]:
            rho_smiles[f"ρ={rr:+.1f}"] = cached_smile(
                (kappa, theta, xi, rr, V0), S0, r, q, tuple(strikes), 0.5
            )
        st.plotly_chart(
            plt.smile_figure(strikes, rho_smiles, "Correlation → skew"),
            use_container_width=True,
        )
    with colB:
        st.markdown("#### ξ effect on curvature (T = 6M)")
        xi_smiles = {}
        for xx in [0.1, 0.4, 0.8, 1.2]:
            xi_smiles[f"ξ={xx:.1f}"] = cached_smile(
                (kappa, theta, xx, rho, V0), S0, r, q, tuple(strikes), 0.5
            )
        st.plotly_chart(
            plt.smile_figure(strikes, xi_smiles, "Vol-of-vol → curvature"),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Tab 2 — Variance Simulator
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Variance process simulation")
    cc = st.columns(4)
    Tsim = cc[0].slider("Horizon T (y)", 0.25, 5.0, 1.0, 0.25)
    n_paths = cc[1].select_slider("Paths", [1000, 5000, 10000, 20000], 5000)
    n_steps = cc[2].select_slider("Steps", [50, 100, 250, 500], 100)
    scheme = cc[3].selectbox("Scheme", ["qe", "euler", "milstein", "exact"])

    S, V = cached_paths(pv, S0, Tsim, r, q, n_paths, n_steps, scheme, 0)
    t = np.linspace(0, Tsim, n_steps + 1)
    vp = VarianceProcessDiagnostics(HestonParams(*pv))
    cm = vp.conditional_moments(V0, t)
    mean, sd = cm["E_V"].values, np.sqrt(cm["Var_V"].values)
    bands = [(mean - sd, mean + sd), (mean - 2 * sd, mean + 2 * sd)]

    st.plotly_chart(
        plt.fan_chart(
            t,
            mean,
            bands,
            sample_paths=V[:15],
            title="Variance V(t)",
            y_title="Variance",
        ),
        use_container_width=True,
    )

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Terminal variance vs stationary Gamma")
        import plotly.graph_objects as go

        vgrid = np.linspace(0, V[:, -1].max() * 1.1, 200)
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=V[:, -1],
                histnorm="probability density",
                name="Simulated",
                marker_color="#4C9BE8",
                opacity=0.6,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=vgrid,
                y=vp.stationary_distribution(vgrid),
                name="Stationary Γ",
                line=dict(color="#FFD93D", width=3),
            )
        )
        fig.update_layout(template="plotly_dark", title="Terminal V(T)")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.markdown("#### Conditional moments")
        tg = np.array([0.25, 0.5, 1.0, 2.0, 5.0])
        st.dataframe(vp.conditional_moments(V0, tg).round(5), use_container_width=True)
        st.metric(
            "Variance-swap rate (vol)",
            f"{np.sqrt(vp.variance_swap_rate(V0, Tsim)):.2%}",
        )

    st.markdown("#### Spot–vol correlation (empirical leverage)")
    from diagnostics.correlation_analysis import CorrelationEffects

    emp = CorrelationEffects.leverage_effect_from_paths(S, V)
    st.write(f"Empirical Corr(dlnS, dV) = **{emp:.3f}**  (input ρ = {rho:.2f})")

# ---------------------------------------------------------------------------
# Tab 3 — Fourier Pricing
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Fourier pricing lab")
    cc = st.columns(4)
    K = cc[0].number_input("Strike K", 1.0, 100000.0, float(S0))
    T = cc[1].number_input("Maturity T", 0.05, 5.0, 1.0, 0.05)
    alpha = cc[2].slider("Carr-Madan α", 0.5, 3.0, 1.5, 0.1)
    otype = cc[3].selectbox("Option", ["call", "put"])

    fd = FourierDiagnostics()
    p = HestonParams(*pv)
    table = fd.method_comparison(S0, K, T, r, q, p, otype)
    from implied_vol.bsm_inversion import BSMInverter

    inv = BSMInverter()
    table["implied_vol"] = [
        inv.safe_implied_vol(pr, S0, K, T, r, q, otype) for pr in table["price"]
    ]
    m1, m2 = st.columns(2)
    m1.metric(
        f"{otype.title()} price (COS)",
        f"{COSMethodPricer().price(S0, K, T, r, q, p, otype):.4f}",
    )
    m2.metric("Implied vol", f"{table['implied_vol'].iloc[1]:.2%}")
    st.dataframe(table.round(6), use_container_width=True)

    colA, colB = st.columns(2)
    with colA:
        conv = fd.cos_convergence_plot(S0, K, T, r, q, p, otype)
        st.plotly_chart(
            plt.line_figure(
                conv["N"],
                np.maximum(conv["abs_error"], 1e-16),
                "COS convergence",
                "N terms",
                "abs error",
                log_y=True,
            ),
            use_container_width=True,
        )
    with colB:
        cfa = CFAnalyzer()
        u = np.linspace(-40, 40, 600)
        d = cfa.cf_plot_data(u, S0, V0, T, r, q, kappa, theta, xi, rho)
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=u, y=d["real"], name="Re φ(u)", line=dict(color="#4C9BE8"))
        )
        fig.add_trace(
            go.Scatter(x=u, y=d["imag"], name="Im φ(u)", line=dict(color="#E8794C"))
        )
        fig.add_trace(
            go.Scatter(x=u, y=d["abs"], name="|φ(u)|", line=dict(color="#FFD93D"))
        )
        fig.update_layout(template="plotly_dark", title="Characteristic function")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4 — IV Surface
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Implied-volatility surface")
    cc = st.columns(3)
    lo = cc[0].slider("Min strike (% spot)", 50, 100, 70)
    hi = cc[1].slider("Max strike (% spot)", 100, 200, 130)
    n_mat = cc[2].slider("Maturities", 4, 12, 8)
    strikes = np.linspace(lo / 100 * S0, hi / 100 * S0, 25)
    mats = np.linspace(1 / 12, 2.0, n_mat)
    iv = cached_surface(pv, S0, r, q, tuple(strikes), tuple(mats))
    st.plotly_chart(plt.surface_figure(strikes, mats, iv), use_container_width=True)

    surf = IVSurface(strikes, mats, iv, S0)
    metrics = SmileAnalytics().compute_smile_metrics(surf, S0, r, q)
    st.markdown("#### Smile metrics")
    st.dataframe(metrics.round(4), use_container_width=True)

    chk = ArbitrageFreeChecker(S0, r, q)
    report = chk.full_arbitrage_report(strikes, mats, iv)
    if report["arbitrage_free"]:
        st.success("No static-arbitrage violations detected.")
    else:
        st.warning(f"Arbitrage report: {report}")

# ---------------------------------------------------------------------------
# Tab 5 — Calibration
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Calibration engine")
    src = st.radio(
        "Market data", ["Built-in SPX sample", "Upload CSV"], horizontal=True
    )
    surface = None
    if src == "Upload CSV":
        up = st.file_uploader("IV surface CSV", type="csv")
        spot_in = st.number_input("Spot for uploaded surface", 1.0, 100000.0, float(S0))
        if up is not None:
            surface = MarketIVSurface(pd.read_csv(up), spot_in)
    else:
        try:
            surface = MarketIVSurface.from_csv("data/sample_surface.csv", spot=4500.0)
        except Exception as e:  # pragma: no cover
            st.error(f"Could not load sample: {e}")

    if surface is not None:
        st.write(f"Loaded **{len(surface.data)}** quotes, spot {surface.spot:g}.")
        loss_type = st.selectbox("Loss", ["bid_ask", "rmse", "vwrmse"])
        maxiter = st.slider("DE generations", 20, 200, 60, 10)
        cal_spot = surface.spot
        cal_r = st.number_input(
            "Calibration r", -0.05, 0.20, 0.045, 0.005, format="%.3f"
        )
        cal_q = st.number_input("Calibration q", 0.0, 0.20, 0.018, 0.005, format="%.3f")
        if st.button("Run calibration", type="primary"):
            with st.spinner("Calibrating (differential evolution + L-BFGS-B)…"):
                cal = HestonCalibrator(seed=0)
                res = cal.calibrate_joint(
                    surface, cal_spot, cal_r, cal_q, loss_type, de_maxiter=maxiter
                )
            st.session_state["cal_res"] = res
        if "cal_res" in st.session_state:
            res = st.session_state["cal_res"]
            cc = st.columns(5)
            for col, (k, v) in zip(cc, res.params.as_dict().items()):
                col.metric(k, f"{v:.4f}")
            m1, m2, m3 = st.columns(3)
            m1.metric("RMSE (bps)", f"{res.rmse_bps:.1f}")
            m2.metric("Feller ratio", f"{res.extra['feller_ratio']:.2f}")
            m3.metric("Time (s)", f"{res.elapsed_seconds:.1f}")
            hist = res.extra.get("history", [])
            if hist:
                st.plotly_chart(
                    plt.line_figure(
                        np.arange(len(hist)), hist, "Loss history", "evaluation", "loss"
                    ),
                    use_container_width=True,
                )

# ---------------------------------------------------------------------------
# Tab 6 — Model Comparison
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Heston vs SABR vs flat BSM")
    try:
        surface = MarketIVSurface.from_csv("data/sample_surface.csv", spot=4500.0)
        from diagnostics.model_comparison import ModelComparison

        p = HestonParams(*pv)
        if st.button("Run comparison"):
            with st.spinner("Comparing models…"):
                mc = ModelComparison()
                tbl = mc.comparison_table(surface, p, 4500.0, 0.045, 0.018)
            st.dataframe(tbl.round(2), use_container_width=True)
            slice_T = surface.maturities[len(surface.maturities) // 2]
            ov = mc.smile_overlay_all_models(surface, slice_T, p, 4500.0, 0.045, 0.018)
            st.plotly_chart(
                plt.smile_figure(
                    ov["strikes"],
                    {
                        "Market": ov["market"],
                        "Heston": ov["heston"],
                        "SABR": ov["sabr"],
                    },
                    f"Smile overlay (T={ov['maturity']:.2f})",
                ),
                use_container_width=True,
            )
    except Exception as e:  # pragma: no cover
        st.info(f"Load the sample surface to compare models. ({e})")

# ---------------------------------------------------------------------------
# Tab 7 — Risk & Greeks
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Risk & Greeks dashboard")
    cc = st.columns(3)
    Kg = cc[0].number_input("Strike", 1.0, 100000.0, float(S0), key="gk")
    Tg = cc[1].number_input("Maturity", 0.05, 5.0, 1.0, 0.05, key="gt")
    otg = cc[2].selectbox("Option", ["call", "put"], key="go")
    p = HestonParams(*pv)
    g = HestonGreeks().compute_all(S0, Kg, Tg, r, q, p, otg)
    cols = st.columns(4)
    for col, key in zip(
        cols * 2,
        ["delta", "gamma", "vega", "theta", "vanna", "volga", "rho_r", "rho_corr"],
    ):
        col.metric(key, f"{g[key]:.4f}")

    st.markdown("#### P&L attribution (dS = 1%, dσ = +1 vol pt)")
    att = VolPnLAttribution().explain(
        S0, Kg, Tg, r, q, p, dS=0.01 * S0, dSigma=0.01, option_type=otg
    )
    wf = VolPnLAttribution().waterfall_data(att)
    st.plotly_chart(
        plt.waterfall_figure(wf["labels"], wf["values"]), use_container_width=True
    )

# ---------------------------------------------------------------------------
# Tab 8 — Documentation
# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Model documentation")
    st.markdown("### The Heston (1993) model")
    st.latex(r"""dS_t = (r-q)S_t\,dt + \sqrt{V_t}\,S_t\,dW^S_t""")
    st.latex(r"""dV_t = \kappa(\theta - V_t)\,dt + \xi\sqrt{V_t}\,dW^V_t,
                 \quad d\langle W^S,W^V\rangle_t = \rho\,dt""")
    st.markdown(
        "**Feller condition** $2\\kappa\\theta \\ge \\xi^2$ keeps "
        "the variance strictly positive."
    )
    st.markdown("### Characteristic function")
    st.latex(r"""\phi(u)=\exp\big(C(u,T)\theta + D(u,T)V_0 + iu\ln F\big)""")
    st.latex(r"""d=\sqrt{(\rho\xi iu-\kappa)^2+\xi^2(iu+u^2)},\quad
                 g=\frac{\kappa-\rho\xi iu-d}{\kappa-\rho\xi iu+d}""")
    st.markdown("### Fourier pricing methods")
    st.table(
        pd.DataFrame(
            {
                "Method": ["Carr-Madan FFT", "COS", "Lewis"],
                "Convergence": ["Algebraic in N", "Exponential in N", "Adaptive quad"],
                "Best for": [
                    "Full smile in one FFT",
                    "Fast & accurate (default)",
                    "Single-strike accuracy",
                ],
            }
        )
    )
    st.markdown("### QE simulation scheme (Andersen 2007)")
    st.markdown(r"""The QE scheme switches at $\psi = \mathrm{Var}[V]/E[V]^2$:
        a quadratic-Gaussian branch for $\psi \le 1.5$ and an exponential
        branch for $\psi > 1.5$, guaranteeing non-negative variance.""")
    st.markdown("### References")
    st.markdown("""
- Heston (1993), *A Closed-Form Solution for Options with Stochastic Volatility*
- Carr & Madan (1999), *Option Valuation Using the FFT*
- Fang & Oosterlee (2008), *A Novel Pricing Method Based on Fourier-Cosine Series*
- Andersen (2007), *Efficient Simulation of the Heston Model*
- Albrecher et al. (2007), *The Little Heston Trap*
- Gatheral et al. (2018), *Volatility is Rough*
""")
