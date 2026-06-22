# Production Code Review — `stochastic-vol-models`

**Reviewer role:** senior quantitative analyst
**Scope:** mathematical correctness, numerical stability, code quality, test
coverage, Streamlit app, GitHub presentation.
**Verdict:** The numerical core is **correct and well-validated**. None of the
audit's *CRITICAL* candidates turned out to be real defects — each was verified
empirically and already satisfied. Remediation therefore concentrated on
defensive guards, a few coverage gaps, and presentation.

---

## Audit method

Every claim was checked against the running code (not assumed). A verification
harness confirmed: `little_trap` is the default CF; `CF(0)=1` to machine
epsilon; `max|CF(u)|=1` on `[0,100]`; the martingale condition; the Carr-Madan
denominator sign `+i(2α+1)u` and Simpson weights; the strike grid
`K_j=exp(k_j)`; both QE conditional-variance terms; `psi ≤ ψ_c` routing to the
quadratic branch; and calibration bounds expressed in **variance** space.

### Claims that were raised as CRITICAL but are NOT defects

| Audit claim | Finding |
|---|---|
| "Verify Little Trap, not original" | `formulation='little_trap'` is the default; `original` is opt-in. They agree to 1e-10 away from the branch cut (`test_formulations_agree`). |
| "CF(-i) = exp((r-q)T)" | Our CF is of the **log-spot** `ln S_T`, so `CF(-i)=E[S_T]=S0·e^{(r-q)T}` ⇒ `CF(-i)/F = 1`. The audit's identity assumes an `S0=1` log-return CF. `test_martingale_condition` already checks `CF(-i)/F = 1`. ✅ correct as written. |
| "Little Trap uses `g̃=(b+d)/(b−d)`, `exp(d·T)`" | That is an *algebraically equivalent* rewriting. Our stable form uses `g=(b−d)/(b+d)`, `exp(−d·T)` (Albrecher et al. 2007), which is well-defined at `u=0` (where `b+d=2κ≠0`). Adopting the prompt's variant would reintroduce a `0/0` at `u=0`. **Rejected** — would regress passing tests. |
| "COS: N must be even" | The COS method is a direct cosine sum; it does **not** require even `N`. **Rejected.** |
| "QE missing 2nd variance term" | Both terms present: `s2 = v*c1 + c2`. ✅ |
| "Calibration bounds in vol space" | Bounds are already in variance space: `theta,V0 ∈ [1e-3, 1]`. ✅ |

---

## CRITICAL
None. The core math is correct and covered by tests.

## HIGH — implemented this pass

1. **Carr-Madan damping guard.** `price`/`price_fft` now raise `ValueError`
   for `alpha ≤ 0` (integrability fails for calls) and for non-power-of-two
   `N`. *(fourier_pricing/carr_madan.py)*
2. **Carr-Madan intrinsic floor.** Interpolated call prices are floored at the
   discounted-intrinsic lower bound to remove cubic-spline artefacts in the
   deep wings. *(fourier_pricing/carr_madan.py)*
3. **CF parameter guard.** The raw `cf()` now rejects non-positive
   `theta, xi, V0` immediately rather than producing `NaN`. *(heston_cf.py)*
4. **SABR `beta` validation.** `SABRParams` validates `beta ∈ [0,1]`,
   `alpha,nu > 0`, `rho ∈ (−1,1)` with a clear message. *(models/sabr.py)*
5. **COS degenerate-interval fallback + `|c4|`.** Uses `|c2|`,`|c4|`
   (Fang-Oosterlee) for the truncation width and falls back to the
   Black-Scholes price when `b−a < 1e-6` (near-expiry / near-zero variance).
   *(fourier_pricing/cos_method.py)*

### Tests added (8)
- `test_carr_madan_rejects_nonpositive_alpha`, `…_non_power_of_two_N`
- `test_cf_rejects_negative_variance_params`
- `test_sabr_beta_out_of_range_raises`, `test_sabr_invalid_alpha_nu_rho_raise`,
  `test_hagan_atm_limit_is_finite_and_smooth`
- `test_safe_iv_above_maximum_returns_nan`, `test_batch_inversion_performance`
- `test_bk_exact_lower_variance_bias_than_euler` (retargeted from a flaky
  `E[V_T]`-mean comparison to a `Var[V_T]` comparison — full-truncation Euler
  is nearly mean-unbiased, so the mean was not a valid discriminator).

**Suite: 41 fast tests + 1 slow calibration roundtrip, all green.**

## MEDIUM — tracked as GitHub issues
- COS: cache `_c2/_c4` cumulants per `(params,T)` to avoid recomputing the
  finite-difference CGF on every strike (perf).
- `IVSurface`: optionally store a labelled `pd.DataFrame`/`MultiIndex` view
  alongside the numpy grid (currently labels live on the axes arrays).
- BSM inversion: surface an explicit warning when the Brent bracket is
  same-signed (today it silently returns `NaN`).
- Calibration: add an out-of-sample / overfitting guard and a
  `calibrated Feller ratio > 0.5` sanity check in diagnostics.
- Streamlit Tab 5: live DE iteration callback + residual heatmap (RdBu, centered
  at 0) + per-parameter ±20% sensitivity panel.

## LOW — tracked as GitHub issues
- README: embed a 3-D IV-surface PNG, a FFT/COS/Lewis comparison table, and the
  Lord & Kahl (2010) "Complex logarithms in Heston-like models" citation.
- Risk Tab 7: annotate vanna sign flip across `K=F` and volga wing peak.
- Add `differential_evolution` reproducibility note to docs (seed already wired).

## Estimated effort
| Tier | Effort |
|---|---|
| CRITICAL | 0 h (none) |
| HIGH | ~3 h (done) |
| MEDIUM | ~6–8 h |
| LOW | ~3–4 h |
