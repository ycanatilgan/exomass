"""Mass inference for exoplanets from catalogue observables."""

from __future__ import annotations

__version__ = "0.1.0"

from .features import build_feature_frame, classify_regime, select_for_regime
from .physics import derive_missing_physical, ensure_columns
from .predict import ALIASES, load_model, predict, predict_frame
from .schema import (
    BOUNDARY_RADIUS_EARTH,
    FEATURE_ORDER,
    RADIUS_DOMAIN_EARTH,
    REGIME_FEATURES,
)

__all__ = [
    "__version__",
    "ALIASES",
    "BOUNDARY_RADIUS_EARTH",
    "FEATURE_ORDER",
    "RADIUS_DOMAIN_EARTH",
    "REGIME_FEATURES",
    "build_feature_frame",
    "classify_regime",
    "derive_missing_physical",
    "ensure_columns",
    "load_model",
    "predict",
    "predict_frame",
    "select_for_regime",
]
