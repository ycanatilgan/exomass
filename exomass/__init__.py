"""Mass inference for exoplanets from catalogue observables."""

from __future__ import annotations

__version__ = "1.0.0"

from .features import build_feature_frame, classify_regime, select_for_variant
from .physics import derive_missing_physical, ensure_columns
from .predict import ALIASES, load_model, predict, predict_frame
from .schema import (
    BOUNDARY_RADIUS_EARTH,
    CV_EPS,
    FEATURE_ORDER,
    RADIUS_DOMAIN_EARTH,
    VARIANT_FEATURES,
    VARIANTS,
)

__all__ = [
    "__version__",
    "ALIASES",
    "BOUNDARY_RADIUS_EARTH",
    "CV_EPS",
    "FEATURE_ORDER",
    "RADIUS_DOMAIN_EARTH",
    "VARIANT_FEATURES",
    "VARIANTS",
    "build_feature_frame",
    "classify_regime",
    "derive_missing_physical",
    "ensure_columns",
    "load_model",
    "predict",
    "predict_frame",
    "select_for_variant",
]
