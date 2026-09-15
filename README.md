# exomass

Mass inference for exoplanets from catalogue observables.

Two gradient-boosted and tree-ensemble regressors, trained on a provenance-filtered
sample of 1347 planets from the NASA Exoplanet Archive plus the eight Solar System
planets, split into a small-planet and a giant-planet regime at 8.36 Earth radii.

Only the planet radius is required. Anything derivable from what you supply is
computed from physics first, so five ordinary quantities already fill fourteen of
the seventeen predictors.

## Install

```bash
pip install git+https://github.com/ycanatilgan/exomass.git
```

To work on a local checkout instead, use `pip install -e .` from the repository
root.

The shipped models were built with scikit-learn 1.7.2 and xgboost 3.1.1, and the
dependency bounds in `pyproject.toml` are pinned around those versions.

## Use

```python
from exomass import predict

result = predict(radius=2.4, period=11.2, stellar_mass=0.91,
                 stellar_radius=0.88, stellar_teff=5320)

result["pred_mass_earth"]  # 10.06
result["derived"]          # the eight predictors filled from physics
result["absent"]           # the three that stayed unknown
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
| `regime`, `model_used` | which model produced it |
| `derived` | predictors the physics cascade filled in |
| `absent` | predictors that stayed unknown and reached the model as its sentinel |
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

Two predictors are never derivable: metallicity, and the projected rotational
velocity unless you also supply a rotation period. Transit duration additionally
needs an impact parameter.

## Notes on use

Predictions are point estimates. Model accuracy is reported by `exomass info` as
out-of-fold `R2_log` and `MAE_log`, which characterise each model across the
sample rather than any single planet.

`in_domain` marks whether the radius falls inside the recommended domain of 1.10
to 18.49 Earth radii. Predictions outside it are returned and flagged.

`regime` selects a model explicitly instead of routing on radius, which is useful
for planets sitting close to the 8.36 Earth-radius boundary.

## Published predictions

`predictions/Mass-Predictions.csv` holds the catalogue-scale predictions reported
in the accompanying paper, and is the same file as the article's supplementary
table. It lives in the repository rather than inside the installed package, so it
does not travel with `pip install`.

4148 objects, every one inside the recommended domain of 1.10 to 18.49 Earth
radii:

| `source` | Count | What it is |
| --- | --- | --- |
| `NEA` | 2782 | transiting planets in the NASA Exoplanet Archive with no published mass |
| `Kepler Candidate` | 1366 | entries in the *Kepler* planet-candidate table |

| Column | Meaning |
| --- | --- |
| `planet_name`, `source` | identifier, and which catalogue it came from |
| `regime`, `model_used` | which model produced the row |
| `planet_radius_earth`, `orbital_period_days`, `system_planet_count`, `stellar_rot_vel_kms`, `stellar_metallicity_dex` | the measured inputs that were available, blank where absent |
| `pred_mass_earth`, `pred_mass_jup` | predicted mass |
| `pred_density_gcm3` | bulk density implied by the predicted mass and the measured radius |

The *Kepler* candidate table carries no projected rotational velocity, so those
rows were predicted by champions retrained with that predictor dropped. The
`model_used` column records which model produced each row.

Predictions are point estimates. Eighteen of these objects acquired dynamical
masses after the training snapshot: they came back with a median absolute error
of 0.19 dex, twelve of the eighteen within a factor of two. The residuals track
bulk density. The eleven denser than 1 g/cm³ were essentially unbiased, while the
seven below it were over-predicted by 0.43 dex on average, five of them orbiting
stars younger than 0.25 Gyr. Treat predicted masses for inflated planets with
caution.

## Reference

If this is useful in published work, please cite the accompanying paper; see
`CITATION.cff`.
