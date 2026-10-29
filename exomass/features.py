"""Turn physical quantities into the feature matrix the models expect.

An absent value stays NaN and reaches the model as NaN; the trees were trained
that way and route it themselves.  A logged predictor that is zero or negative
has no logarithm, so it is treated as absent too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import (
    BASE_COLUMNS,
    BOUNDARY_RADIUS_EARTH,
    FEATURE_ORDER,
    NO_LOG_COLS,
    VARIANT_FEATURES,
    base_column,
)


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full 17-column feature frame from physical columns.

    Returns the feature frame and a boolean frame, aligned with
    :data:`~exomass.schema.BASE_COLUMNS`, marking which physical entries were
    absent and therefore reach the model as NaN.
    """
    absent = pd.DataFrame(index=df.index)
    features = pd.DataFrame(index=df.index)

    for feature in FEATURE_ORDER:
        source = base_column(feature)
        if source in df.columns:
            values = pd.to_numeric(df[source], errors="coerce").astype("float64")
        else:
            values = pd.Series(np.nan, index=df.index, dtype="float64")

        if source in NO_LOG_COLS:
            features[feature] = values
        else:
            features[feature] = np.log10(values.where(values > 0))
        absent[source] = features[feature].isna()

    return features[list(FEATURE_ORDER)], absent[list(BASE_COLUMNS)]


def classify_regime(radius_earth) -> pd.Series:
    """Label rows ``small`` or ``giant`` by the radius threshold.

    Radius is the one quantity a prediction needs, so a row without a usable
    one is labelled ``None`` and carries through as an absent answer.
    """
    radius = pd.to_numeric(pd.Series(radius_earth), errors="coerce")
    usable = radius.notna() & (radius > 0)
    labels = np.where(radius < BOUNDARY_RADIUS_EARTH, "small", "giant")
    return pd.Series(
        np.where(usable, labels, None), index=radius.index, dtype="object"
    )


def select_for_variant(features: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Cut the full feature frame down to one model's predictors, in its order."""
    try:
        wanted = VARIANT_FEATURES[variant]
    except KeyError:
        raise ValueError(
            f"unknown model {variant!r}, expected one of {sorted(VARIANT_FEATURES)}"
        ) from None
    return features[list(wanted)]
