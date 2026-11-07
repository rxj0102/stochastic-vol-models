# Sample data

## `sample_surface.csv`

A synthetic SPX-style implied-volatility surface with a pronounced negative
skew, generated from the library's own Heston engine and lightly perturbed with
noise so that it resembles a real market snapshot.

| column          | meaning                                              |
|-----------------|------------------------------------------------------|
| `maturity`      | option maturity in years                             |
| `strike`        | absolute strike                                      |
| `bid_iv`        | bid implied volatility (decimal, e.g. `0.20` = 20%)  |
| `ask_iv`        | ask implied volatility                               |
| `mid_iv`        | mid implied volatility (used as the calibration target) |
| `volume`        | traded contracts (liquidity weight)                  |
| `open_interest` | open interest (liquidity filter)                     |

**Reference spot:** `S0 = 4500`, rates `r = 4.5%`, dividend yield `q = 1.8%`.

The surface was produced with the "true" parameters
`kappa=1.8, theta=0.045, xi=0.55, rho=-0.72, V0=0.032`, so a successful
calibration should recover values close to these.

```python
from calibration.market_data import MarketIVSurface
surface = MarketIVSurface.from_csv("data/sample_surface.csv", spot=4500.0)
```

> These data are **synthetic** and provided for demonstration and testing only.
> They are not real market quotes and must not be used for trading.
