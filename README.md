# stochastic-vol-models

[![CI](https://github.com/rxj0102/stochastic-vol-models/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/stochastic-vol-models/actions/workflows/ci.yml)

**A stochastic volatility modelling library: Heston, SABR and related models, with Fourier and Monte Carlo pricing and calibration.**

A comprehensive implementation of the **Heston (1993)** model with full
characteristic-function machinery, three independent **Fourier pricing** engines
(Carr-Madan FFT, COS, Lewis), **implied-volatility surface calibration**,
**smile/skew analytics**, low-bias **variance-process simulation**, and
semi-analytical **Heston Greeks** - packaged as a Python library, a live
**Streamlit** dashboard, and an SSRN-ready research paper.

[![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![tests](https://img.shields.io/badge/tests-33%20passing-brightgreen)](tests/)

---

## Highlights

- **Characteristic function** in both the original and the numerically stable
  *Little Heston Trap* formulations, vectorised over the frequency argument.
- **Three Fourier pricers** that agree to **< 1 bp** of implied volatility:
  Carr-Madan FFT (whole smile per transform), COS (exponential convergence,
  the default), and Lewis (single contour integral).
- **Simulation schemes**: full-truncation Euler, Milstein, Andersen's
  **Quadratic-Exponential (QE)** (non-negative variance, low bias) and a
  near-exact **Broadie-Kaya** scheme built on the exact CIR transition.
- **Calibration** via a two-stage **differential evolution → L-BFGS-B**
  pipeline; recovers known parameters from a synthetic surface to within a
  fraction of a percent.
- **Arbitrage checks** (calendar, butterfly, call-spread) and
  Breeden-Litzenberger **risk-neutral density** extraction.
- **Risk**: delta, gamma, vega, theta, rho plus the Heston cross-Greeks
  **vanna** and **volga**, with full P&L attribution.

## Installation

```bash
git clone https://github.com/rxj0102/stochastic-vol-models.git
cd stochastic-vol-models
pip install -r requirements.txt
# or: pip install -e ".[app,paper,dev]"
```

## Quick start

```python
import numpy as np
from models.heston import HestonModel, HestonParams
from fourier_pricing.cos_method import COSMethodPricer

params = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, V0=0.04)
model = HestonModel(params)

print(model.feller_ratio)        # 2*kappa*theta / xi^2
print(model.long_run_vol)        # sqrt(theta)

# Price a 1Y ATM call three independent ways - they agree to < 1 bp of IV.
price = COSMethodPricer().price(S0=100, K=100, T=1.0, r=0.02, q=0.0, params=params)
print(f"COS call price: {price:.4f}")
```

### Calibrate to a market surface

```python
from calibration.market_data import MarketIVSurface
from calibration.heston_calibration import HestonCalibrator

surface = MarketIVSurface.from_csv("data/sample_surface.csv", spot=4500.0)
result = HestonCalibrator(seed=0).calibrate_joint(surface, S0=4500, r=0.045, q=0.018)
print(result.params, f"RMSE = {result.rmse_bps:.1f} bps")
```

## The Streamlit dashboard

```bash
streamlit run app.py
```

Eight tabs: **Parameter Explorer**, **Variance Simulator**, **Fourier Pricing
Lab**, **IV Surface**, **Calibration Engine**, **Model Comparison**,
**Risk & Greeks**, and **Documentation**.

## Repository layout

```
models/                  Abstract base + Heston, SABR, Dupire local vol, rough vol
characteristic_functions/ Heston CF (two formulations), cumulants, damping
fourier_pricing/         Carr-Madan FFT, COS, Lewis, cross-method diagnostics
simulation/              Euler, Milstein, QE, Broadie-Kaya, rBergomi, path manager
calibration/             Market data, loss functions, optimiser, Heston/SABR pipelines
implied_vol/             Surface, smile analytics, BSM inversion, arbitrage checks
risk/                    Heston Greeks, P&L attribution, vol-aware VaR
diagnostics/             Variance process, correlation, moment matching, comparison
utils/                   Plotly chart builders, statistics helpers
tests/                   33 pytest tests across the full stack
data/                    Synthetic SPX-style sample surface
paper/                   ReportLab SSRN-ready PDF generator
app.py                   Streamlit dashboard
```

## The model

$$dS_t = (r-q)\,S_t\,dt + \sqrt{V_t}\,S_t\,dW^S_t$$
$$dV_t = \kappa(\theta - V_t)\,dt + \xi\sqrt{V_t}\,dW^V_t, \qquad d\langle W^S, W^V\rangle_t = \rho\,dt$$

The **Feller condition** $2\kappa\theta \ge \xi^2$ keeps the variance strictly
positive. The parameters: $\kappa$ mean-reversion speed, $\theta$ long-run
variance, $\xi$ vol-of-vol, $\rho$ spot-vol correlation (drives skew),
$V_0$ initial variance.

## Testing

```bash
pytest -m "not slow"     # fast suite (~45s)
pytest                   # full suite incl. calibration round-trip
```

## Generate the research paper

```bash
python paper/generate_paper.py   # writes paper/stochastic_volatility_heston.pdf
```

## Reproducibility

### Data availability

Inputs ship in the `data/` folder of this repository. No external downloads are needed.

### Computational requirements

- Python 3.12.3. Every dependency version is pinned in `requirements.lock` (compiled from `requirements.txt` with uv).
- Checked on Linux x86_64 with 2 CPU cores and 8 GB RAM. No GPU is needed.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock   # exact pinned environment
pip install -e . --no-deps         # this package
pytest -q                          # runs the test suite
```

### Runtime

The test suite (43 tests) finishes in about 3 minutes on the machine above, including any data downloads it triggers.

### Citation

Citation metadata is in `CITATION.cff`. GitHub turns it into a Cite this repository button on the repo page.


## References

- Heston (1993), *A Closed-Form Solution for Options with Stochastic Volatility*
- Carr & Madan (1999), *Option Valuation Using the Fast Fourier Transform*
- Fang & Oosterlee (2008), *A Novel Pricing Method Based on Fourier-Cosine Series*
- Andersen (2007), *Efficient Simulation of the Heston Stochastic Volatility Model*
- Broadie & Kaya (2006), *Exact Simulation of Stochastic Volatility Processes*
- Albrecher, Mayer, Schoutens & Tistaert (2007), *The Little Heston Trap*
- Hagan, Kumar, Lesniewski & Woodward (2002), *Managing Smile Risk*
- Gatheral, Jaisson & Rosenbaum (2018), *Volatility Is Rough*

## License

MIT - see [LICENSE](LICENSE). The bundled data are synthetic and for
demonstration only; nothing here is investment advice.

## Topics

`heston-model` · `stochastic-volatility` · `implied-volatility` ·
`characteristic-functions` · `fourier-pricing` · `carr-madan` · `cos-method` ·
`calibration` · `volatility-surface` · `quantitative-finance` · `python` ·
`streamlit`
