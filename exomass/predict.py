"""Mass prediction from whatever catalogue quantities the caller happens to have.

The caller is not expected to supply all 17 predictors.  Give the package a
planet radius plus whatever else is at hand and the physics cascade fills in
everything it can derive, which for a typical transiting system means most of
the vector.  What cannot be derived falls back to the sentinel the models were
trained to recognise, and every prediction reports which of the three cases
each predictor ended up in.

Absent, non-physical and off-coverage inputs are all recorded in the returned
notes, so a caller can see what the model was actually given.
"""

from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .features import build_feature_frame, classify_regime, select_for_regime
from .physics import CASCADE_COLUMNS, derive_missing_physical, ensure_columns
from .schema import (
    BASE_COLUMNS,
    FEATURE_RANGES,
    MODEL_PATHS,
    RADIUS_DOMAIN_EARTH,
    REGIME_FEATURES,
)

#: How many out-of-range quantities to spell out in the notes before
#: summarising the rest as a count.
_NOTE_DETAIL_LIMIT = 3

MJUP_TO_MEARTH = 317.828
RHO_EARTH_G_CM3 = 5.514

#: Columns the caller may supply: the 17 predictors plus the two extra
#: quantities the cascade can use as sources.
ACCEPTED_COLUMNS = tuple(dict.fromkeys(BASE_COLUMNS + CASCADE_COLUMNS))

#: Friendlier names for the quantities people actually have to hand.
ALIASES = {
    "radius": "planet_radius_earth",
    "planet_radius": "planet_radius_earth",
    "period": "orbital_period_days",
    "orbital_period": "orbital_period_days",
    "stellar_mass": "stellar_mass_solar",
    "stellar_radius": "stellar_radius_solar",
    "stellar_teff": "stellar_temp_k",
    "teff": "stellar_temp_k",
    "metallicity": "stellar_metallicity_dex",
    "feh": "stellar_metallicity_dex",
    "planet_count": "system_planet_count",
    "vsini": "stellar_rot_vel",
    "rotation_period": "stellar_rot_period",
    "semi_major_axis": "orbit_semi_major_axis_au",
    "eq_temp": "equilibrium_temp_k",
    "insolation": "insolation_flux_earth",
}

#: Quantities that are meaningless at or below zero.  A non-positive value
#: here is treated as absent and reported, rather than being fed to a
#: logarithm.  Note the deliberate omissions: metallicity and luminosity are
#: logarithms and are routinely negative, log g is a logarithm too, and an
#: impact parameter of zero is simply a central transit.
POSITIVE_DEFINITE = (
    "planet_radius_earth",
    "orbital_period_days",
    "stellar_mass_solar",
    "stellar_radius_solar",
    "stellar_temp_k",
    "equilibrium_temp_k",
    "insolation_flux_earth",
    "orbit_semi_major_axis_au",
    "ratio_distance_star_radius",
    "stellar_density",
    "transit_depth",
    "transit_duration",
    "stellar_rot_vel",
    "stellar_rot_period",
)


@lru_cache(maxsize=None)
def load_model(regime: str):
    """Load and cache one regime's trained pipeline."""
    try:
        path = MODEL_PATHS[regime]
    except KeyError:
        raise ValueError(
            f"unknown regime {regime!r}, expected one of {sorted(MODEL_PATHS)}"
        ) from None
    return joblib.load(path)


def resolve_names(mapping: dict) -> dict:
    """Map caller-supplied names onto canonical column names."""
    resolved: dict = {}
    for key, value in mapping.items():
        canonical = ALIASES.get(key, key)
        if canonical not in ACCEPTED_COLUMNS:
            raise TypeError(
                f"unknown quantity {key!r}. Accepted names are "
                f"{sorted(ACCEPTED_COLUMNS)} plus the aliases {sorted(ALIASES)}"
            )
        if canonical in resolved:
            raise TypeError(f"{canonical!r} supplied twice (once as {key!r})")
        resolved[canonical] = value
    return resolved


def predict_frame(
    df: pd.DataFrame,
    cascade: bool = True,
    regime: str | None = None,
) -> pd.DataFrame:
    """Predict masses for a frame of physical quantities.

    Parameters
    ----------
    df
        One row per planet.  Columns are canonical physical quantities; any
        the caller does not have may simply be absent.
    cascade
        Fill derivable predictors from physics before predicting.  On by
        default, matching how the training sample was prepared.  Set to
        ``False`` to leave every unsupplied predictor absent instead.
    regime
        Force every row through one regime's model instead of routing on
        radius.  Mostly useful for inspecting behaviour near the boundary.
    """
    work = df.copy()
    renamed = [ALIASES.get(c, c) for c in work.columns]

    # Two input columns can resolve to the same canonical name, for instance
    # "radius" alongside "planet_radius_earth".  Caught here so the caller is
    # told which columns clashed.
    collisions = {name for name in renamed if renamed.count(name) > 1}
    if collisions:
        raise ValueError(
            "these columns resolve to the same quantity: "
            + ", ".join(
                f"{sorted(c for c in work.columns if ALIASES.get(c, c) == name)} "
                f"-> {name!r}"
                for name in sorted(collisions)
            )
        )
    work.columns = renamed

    if "planet_radius_earth" not in work.columns:
        raise ValueError(
            "planet_radius_earth is required: it selects the regime and is the "
            "dominant predictor in both"
        )

    notes: list[list[str]] = [[] for _ in range(len(work))]

    # The planet count is used without a log, so an absent one reaches the
    # model as 0 rather than as the usual -9 sentinel.  Every planet in the
    # training sample carries a real count, so a zero is worth reporting.
    if "system_planet_count" not in work.columns:
        absent_count = np.ones(len(work), dtype=bool)
    else:
        absent_count = (
            pd.to_numeric(work["system_planet_count"], errors="coerce")
            .isna()
            .to_numpy()
        )
    for pos, missing in enumerate(absent_count):
        if missing:
            notes[pos].append(
                "system_planet_count absent, model sees 0, a value it was never "
                "trained on"
            )

    for column in POSITIVE_DEFINITE:
        if column not in work.columns:
            continue
        values = pd.to_numeric(work[column], errors="coerce")
        rejected = values.notna() & ~(values > 0)
        if rejected.any():
            work[column] = values.where(~rejected)
            for pos, bad in enumerate(rejected.to_numpy()):
                if bad:
                    notes[pos].append(f"{column} was not positive, treated as absent")
        else:
            work[column] = values

    absent_before = {
        column: (
            pd.to_numeric(work[column], errors="coerce").isna().to_numpy()
            if column in work.columns
            else np.ones(len(work), dtype=bool)
        )
        for column in BASE_COLUMNS
    }

    if cascade:
        ensure_columns(work)
        derive_missing_physical(work, verbose=False)

    features, absent_after = build_feature_frame(work)

    derived_lists: list[list[str]] = []
    absent_lists: list[list[str]] = []
    after = {c: absent_after[c].to_numpy() for c in BASE_COLUMNS}
    for pos in range(len(work)):
        derived_lists.append(
            [c for c in BASE_COLUMNS if absent_before[c][pos] and not after[c][pos]]
        )
        absent_lists.append([c for c in BASE_COLUMNS if after[c][pos]])

    # Every predictor the model will see, checked against where the training
    # sample actually covers.  Derived values are checked too: a semi-major
    # axis computed from an implausible period is just as far off the training
    # distribution as one supplied directly.
    physical = {
        column: (
            pd.to_numeric(work[column], errors="coerce").to_numpy(dtype="float64")
            if column in work.columns
            else np.full(len(work), np.nan)
        )
        for column in BASE_COLUMNS
    }
    out_of_range_lists: list[list[str]] = []
    for pos in range(len(work)):
        flagged = []
        for column in BASE_COLUMNS:
            value = physical[column][pos]
            if not np.isfinite(value):
                continue
            low, high = FEATURE_RANGES[column]
            if value < low or value > high:
                flagged.append((column, value, low, high))
        out_of_range_lists.append([name for name, _, _, _ in flagged])
        if flagged:
            spelled = ", ".join(
                f"{name}={value:.4g} (trained on {low:.4g} to {high:.4g})"
                for name, value, low, high in flagged[:_NOTE_DETAIL_LIMIT]
            )
            extra = len(flagged) - _NOTE_DETAIL_LIMIT
            if extra > 0:
                spelled += f", and {extra} more"
            notes[pos].append(f"outside the training range: {spelled}")

    radius = pd.to_numeric(work["planet_radius_earth"], errors="coerce")
    if regime is None:
        regimes = classify_regime(radius.to_numpy())
    else:
        if regime not in REGIME_FEATURES:
            raise ValueError(
                f"unknown regime {regime!r}, expected one of {sorted(REGIME_FEATURES)}"
            )
        regimes = pd.Series(
            np.where(radius.notna() & (radius > 0), regime, None),
            index=range(len(work)),
            dtype="object",
        )
    regimes = pd.Series(regimes.to_numpy(), index=features.index, dtype="object")

    pred_log = np.full(len(work), np.nan)
    model_used = np.array([None] * len(work), dtype="object")

    for label in REGIME_FEATURES:
        mask = (regimes == label).to_numpy()
        if not mask.any():
            continue
        model = load_model(label)
        pred_log[mask] = model.predict(select_for_regime(features.loc[mask], label))
        model_used[mask] = f"{type(model[-1]).__name__}_{label}"

    low, high = RADIUS_DOMAIN_EARTH
    in_domain = (radius >= low) & (radius <= high)
    for pos, ok in enumerate(in_domain.to_numpy()):
        if regimes.iloc[pos] is None:
            notes[pos].append("no usable radius, not predicted")
        elif not ok:
            notes[pos].append(
                f"radius outside the recommended domain {low} to {high} R_earth, "
                f"prediction extrapolates"
            )

    mass_earth = 10.0**pred_log
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(
            (mass_earth > 0) & (radius.to_numpy() > 0),
            RHO_EARTH_G_CM3 * mass_earth / radius.to_numpy() ** 3,
            np.nan,
        )

    return pd.DataFrame(
        {
            "regime": regimes.to_numpy(),
            "model_used": model_used,
            "pred_log10_mass": pred_log,
            "pred_mass_earth": mass_earth,
            "pred_mass_jup": mass_earth / MJUP_TO_MEARTH,
            "pred_density_gcm3": density,
            "in_domain": in_domain.to_numpy(),
            "derived": derived_lists,
            "absent": absent_lists,
            "out_of_range": out_of_range_lists,
            "notes": ["; ".join(n) for n in notes],
        },
        index=df.index,
    )


def predict(cascade: bool = True, regime: str | None = None, **quantities) -> dict:
    """Predict the mass of a single planet.

    Accepts canonical column names or the aliases in :data:`ALIASES`, so::

        predict(radius=2.4, period=11.2, stellar_mass=0.91,
                stellar_radius=0.88, stellar_teff=5320)

    returns a dictionary holding the predicted mass in Earth and Jupiter
    masses, the implied bulk density, the regime and model that produced it,
    which predictors the cascade derived, which stayed absent, whether the
    radius sits inside the recommended domain, and any notes worth reading.
    """
    resolved = resolve_names(quantities)
    frame = pd.DataFrame([resolved])
    result = predict_frame(frame, cascade=cascade, regime=regime)
    return result.iloc[0].to_dict()
