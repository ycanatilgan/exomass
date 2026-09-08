"""Model contract, loaded from the metadata shipped with the trained models.

Nothing here is hard-coded.  Every value is read from the JSON file that was
written alongside the models at training time, so the package and the models
agree by construction rather than by convention.
"""

from __future__ import annotations

import json
from pathlib import Path

_MODELS_DIR = Path(__file__).parent / "models"
_METADATA_PATH = _MODELS_DIR / "metadata_generalist_champions_xgb16_rf17_v19.json"
_RANGES_PATH = _MODELS_DIR / "feature_ranges_v19.json"

with _METADATA_PATH.open(encoding="utf-8") as fh:
    METADATA = json.load(fh)

with _RANGES_PATH.open(encoding="utf-8") as fh:
    FEATURE_RANGE_METADATA = json.load(fh)

#: Where each predictor was measured in the training sample, as
#: ``{column: (low, high)}`` over its 2nd to 98th percentile.  Used to note
#: when an input lands where the models have no examples.  Changes no
#: prediction.
FEATURE_RANGES = {
    column: (float(bounds["low"]), float(bounds["high"]))
    for column, bounds in FEATURE_RANGE_METADATA["ranges"].items()
}

#: The 17 predictors, in the order the models expect them.
FEATURE_ORDER = tuple(METADATA["feature_order"])

#: Predictors that are used as-is instead of being log-transformed.
NO_LOG_COLS = frozenset(METADATA["no_log_cols"])

#: Radius that separates the two regimes, in Earth radii.
BOUNDARY_RADIUS_EARTH = float(METADATA["boundary_radius_earth"])

#: Feature subset each regime champion was trained on.
SMALL_FEATURES = tuple(METADATA["selected_features"]["small_top16"])
GIANT_FEATURES = tuple(METADATA["selected_features"]["giant_top17"])

REGIME_FEATURES = {"small": SMALL_FEATURES, "giant": GIANT_FEATURES}

MODEL_PATHS = {
    "small": _MODELS_DIR / "model_small_xgb_top16_v19.joblib",
    "giant": _MODELS_DIR / "model_giant_rf_top17_v19.joblib",
}

MODEL_INFO = METADATA["models"]

#: Recommended deployment domain: the 2nd to 98th percentile of the training
#: radii.  Outside this range the models extrapolate and can return
#: non-physical bulk densities.
RADIUS_DOMAIN_EARTH = (1.10, 18.49)

#: Absent values are set to this sentinel *before* the log transform.
MISSING_SENTINEL = 0.0

#: Floor applied inside the log transform.  Together with the sentinel above
#: this means an absent logged feature reaches the model as log10(1e-9) = -9,
#: which is what the models saw for absent entries during training.
LOG_FLOOR = 1e-9


def base_column(feature: str) -> str:
    """Return the source column a feature is built from.

    The ``log_`` prefix is the only marker of a log-transformed feature, so
    stripping it recovers the physical column name.

    >>> base_column("log_planet_radius_earth")
    'planet_radius_earth'
    >>> base_column("system_planet_count")
    'system_planet_count'
    """
    return feature[4:] if feature.startswith("log_") else feature


#: Physical columns behind FEATURE_ORDER, same order.
BASE_COLUMNS = tuple(base_column(f) for f in FEATURE_ORDER)


def _check_contract() -> None:
    """Guard against a metadata file that disagrees with itself."""
    for feature in FEATURE_ORDER:
        logged = not feature.startswith("log_")
        named = base_column(feature) in NO_LOG_COLS
        if logged != named:
            raise ValueError(
                f"metadata inconsistency: {feature!r} carries a log_ prefix "
                f"that disagrees with no_log_cols"
            )
    for regime, features in REGIME_FEATURES.items():
        unknown = set(features) - set(FEATURE_ORDER)
        if unknown:
            raise ValueError(f"{regime} model wants unknown features: {sorted(unknown)}")
    unranged = set(BASE_COLUMNS) - set(FEATURE_RANGES)
    if unranged:
        raise ValueError(f"no training range recorded for: {sorted(unranged)}")


_check_contract()
