"""Model contract, loaded from the metadata shipped with the trained models.

Nothing here is hard-coded.  Every value is read from the JSON files that were
written alongside the models, so the package and the models agree by
construction rather than by convention.
"""

from __future__ import annotations

import json
from pathlib import Path

_MODELS_DIR = Path(__file__).parent / "models"
_METADATA_PATH = _MODELS_DIR / "metadata.json"
_RANGES_PATH = _MODELS_DIR / "feature_ranges.json"

with _METADATA_PATH.open(encoding="utf-8") as fh:
    METADATA = json.load(fh)

with _RANGES_PATH.open(encoding="utf-8") as fh:
    FEATURE_RANGE_METADATA = json.load(fh)

#: Where each predictor was measured in the training sample, as
#: ``{column: (low, high)}`` over its 2nd to 98th percentile.  Used to note
#: when an input lands where the model has no examples.  Changes no
#: prediction.
FEATURE_RANGES = {
    column: (float(bounds["low"]), float(bounds["high"]))
    for column, bounds in FEATURE_RANGE_METADATA["ranges"].items()
}

#: The two shipped models.  ``full`` uses all 17 predictors; ``no_vsini`` was
#: trained without the projected rotational velocity, for catalogues that never
#: carry it, such as the *Kepler* candidate table.
MODEL_INFO = METADATA["models"]
VARIANTS = tuple(MODEL_INFO)
MODEL_PATHS = {name: _MODELS_DIR / info["file"] for name, info in MODEL_INFO.items()}

#: Predictors of each model, in the order the model expects them.
VARIANT_FEATURES = {name: tuple(info["feature_order"]) for name, info in MODEL_INFO.items()}

#: All 17 predictors, in the order of the full model.
FEATURE_ORDER = VARIANT_FEATURES["full"]

#: Predictors that are used as-is instead of being log-transformed.
NO_LOG_COLS = frozenset(METADATA["no_log_cols"])

#: Radius that separates the small and giant regimes, in Earth radii.  One
#: model covers both; the regime only labels the result and sets its typical
#: error.
BOUNDARY_RADIUS_EARTH = float(METADATA["boundary_radius_earth"])

#: Cross-validated mean log error of each model in each regime,
#: eps = sqrt(<ln(M_pred / M_true)^2>).  exp(eps) is the typical error factor.
CV_EPS = {name: info["cv_eps"] for name, info in MODEL_INFO.items()}

#: Recommended deployment domain: the 2nd to 98th percentile of the training
#: radii, as given in the paper.  Outside this range the model extrapolates
#: and can return non-physical bulk densities.
RADIUS_DOMAIN_EARTH = tuple(METADATA["radius_domain_earth"])


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
    for variant, features in VARIANT_FEATURES.items():
        unknown = set(features) - set(FEATURE_ORDER)
        if unknown:
            raise ValueError(f"{variant} model wants unknown features: {sorted(unknown)}")
    unranged = set(BASE_COLUMNS) - set(FEATURE_RANGES)
    if unranged:
        raise ValueError(f"no training range recorded for: {sorted(unranged)}")


_check_contract()
