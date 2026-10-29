# exomass

[![DOI](https://zenodo.org/badge/1361279113.svg)](https://doi.org/10.5281/zenodo.22658907)

Mass inference for exoplanets from catalogue observables.

One LightGBM regressor of log mass, trained on 1449 transiting planets with
measured masses from the NASA Exoplanet Archive (snapshot of 28 October 2026,
cleaned as described in the paper) plus the eight Solar System planets. A single
model covers all radii; results are labelled small or giant at 8.36 Earth radii,
and the label sets the typical error that comes with each prediction.

Only the planet radius is required. Anything derivable from what you supply is
computed from physics first, so five ordinary quantities already fill thirteen of
the seventeen predictors.

## Install

```bash
pip install git+https://github.com/ycanatilgan/exomass.git
```

To work on a local checkout instead, use `pip install -e .` from the repository
root.

The models ship as LightGBM text files, so the only dependencies are numpy, pandas
and LightGBM 4.x. The tests pin the predicted masses of 40 reference planets to
1e-9 dex.

## Use

```python
from exomass import predict

result = predict(radius=2.4, period=11.2, stellar_mass=0.91,
                 stellar_radius=0.88, stellar_teff=5320)

result["pred_mass_earth"]       # 10.68
result["typical_error_factor"]  # 1.93
result["derived"]               # the eight predictors filled from physics
result["absent"]                # the four that stayed unknown
```

Same thing from a shell:

```bash
exomass predict --radius 2.4 --period 11.2 --stellar-mass 0.91 --stellar-radius 0.88 --stellar-teff 5320
```

A whole table at once:

```bash
exomass predict --csv planets.csv --out masses.csv
```

`exomass info` prints the model contract and the training coverage.

## What comes back

| Field | Meaning |
| --- | --- |
| `pred_mass_earth`, `pred_mass_jup` | predicted mass |
| `pred_density_gcm3` | bulk density implied by the predicted mass and the given radius |
| `cv_eps`, `typical_error_factor` | cross-validated mean log error of the regime, and its exponential |
| `regime`, `model_used` | small or giant label, and the model file that produced it |
| `derived` | predictors the physics cascade filled in |
| `absent` | predictors that stayed unknown and reached the model as missing values |
| `out_of_range` | predictors sitting outside the training sample's 2nd to 98th percentile |
| `in_domain` | whether the radius is within the recommended 1.10 to 18.49 Earth radii |
| `notes` | anything worth reading before trusting the number |

`notes` collects anything the caller should know about the inputs, such as a
quantity that arrived absent or outside the training coverage.

## The physics cascade

Missing predictors are filled in dependency order, repeatedly, until nothing more
can be derived: semi-major axis from Kepler's third law, then scaled separation,
equilibrium temperature, insolation, transit depth and duration, stellar surface
gravity, density and luminosity. Formulas and units are documented in
`exomass/physics.py`.

Pass `cascade=False` to skip it and leave every unsupplied predictor absent.
Absent predictors reach the model as missing values, as they did in training;
nothing is imputed.

Two predictors are never derivable: metallicity, and the projected rotational
velocity unless you also supply a rotation period. Transit duration additionally
needs an impact parameter.

## Notes on use

Predictions are point estimates. The error is given as the mean log error
eps = sqrt(<ln(M_pred/M_true)^2>), so a typical prediction is off by a factor of
about exp(eps). In 10x5 cross-validation grouped by host star:

| Model | Small | Giant | All |
| --- | --- | --- | --- |
| `full` (17 predictors) | 0.655 | 0.729 | 0.695 |
| `no_vsini` (16 predictors) | 0.656 | 0.758 | 0.711 |

On 119 planets that received masses after the training snapshot, the full model
gave 0.505 for small planets, 0.714 for giants and 0.600 overall. The giants were
predicted about 0.17 dex too heavy on median. Planets outside the bulk-density
range of the training sample (0.16 to 18.25 g/cm³) are predicted much worse.
`exomass info` prints these numbers.

These numbers come from the model trained on the October 2025 snapshot, as in the
paper. The shipped model is the same procedure retrained on the October 2026
archive, so it has no separate test set of its own.

`in_domain` marks whether the radius falls inside the recommended domain of 1.10
to 18.49 Earth radii. Predictions outside it are returned and flagged.

`variant="no_vsini"` (`--model no_vsini` on the command line) uses the model
trained without the projected rotational velocity. It is meant for catalogues
that never carry that quantity, such as the *Kepler* candidate table. With the
default model an absent v sin i is also fine; about a quarter of the training
planets lack it.

## Published predictions

`predictions/Mass-Predictions.csv` holds the catalogue of predicted masses
described in the accompanying paper. It lives in the repository rather than inside
the installed package, so it does not travel with `pip install`.

5133 objects:

| `source` | Count | What it is |
| --- | --- | --- |
| `NEA` | 3258 | transiting planets in the NASA Exoplanet Archive (28 October 2026) without a published mass, cleaned like the training sample; full model |
| `Kepler candidate` | 1875 | entries in the *Kepler* planet-candidate table; `no_vsini` model |

| Column | Meaning |
| --- | --- |
| `planet_name`, `host_name`, `source` | identifier, host, and which catalogue it came from |
| `regime`, `model` | small or giant label, and the model file used |
| `planet_radius_earth`, `orbital_period_days`, `system_planet_count`, `stellar_rot_vel_kms` | some of the inputs, blank where absent |
| `n_missing_features` | how many of the model's predictors were absent |
| `pred_log10_mass_earth`, `pred_mass_earth`, `pred_mass_jup` | predicted mass |
| `pred_density_gcm3` | bulk density implied by the predicted mass and the measured radius |
| `cv_eps`, `typical_error_factor` | cross-validated mean log error of the regime and model, and its exponential |

Every object with a radius was predicted, including those outside the recommended
radius domain; filter on `planet_radius_earth` if you need to.

## Earlier versions

Version 1.0.0 replaces the models and catalogue of 0.1.0 and 0.2.0, which stay
available on Zenodo and as git tags.

## Reference

If this is useful in published work, please cite the accompanying paper; see
`CITATION.cff`.
