"""Pin the prediction path against a stored set of reference planets.

``tests/data/reference_planets.csv`` holds 39 training planets drawn evenly
across both regimes, together with the mass each shipped model returns for
them.  They span radii from well below to well above
the recommended domain, and some carry absent entries, so the cascade, the NaN
path and the coverage flags are all exercised.

Nothing outside the package is needed, so these run anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REFERENCE = Path(__file__).parent / "data" / "reference_planets.csv"

from exomass.schema import BASE_COLUMNS

INPUT_COLUMNS = list(BASE_COLUMNS)


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE)


@pytest.mark.parametrize(
    "variant, column",
    [("full", "expected_log10_mass"), ("no_vsini", "expected_log10_mass_no_vsini")],
)
def test_reproduces_reference_masses(reference, variant, column):
    from exomass.predict import predict_frame

    result = predict_frame(reference[INPUT_COLUMNS], cascade=True, variant=variant)
    delta = np.abs(
        result["pred_log10_mass"].to_numpy() - reference[column].to_numpy()
    )
    worst = int(np.nanargmax(delta))
    assert np.nanmax(delta) < 1e-9, (
        f"{reference['planet_name'].iloc[worst]} disagrees by "
        f"{delta[worst]:.3e} dex"
    )


def test_labels_every_planet_with_the_recorded_regime(reference):
    from exomass.predict import predict_frame

    result = predict_frame(reference[INPUT_COLUMNS], cascade=True)
    assert list(result["regime"]) == list(reference["expected_regime"])


def test_flags_planets_beyond_the_recommended_radius(reference):
    from exomass.predict import predict_frame
    from exomass.schema import RADIUS_DOMAIN_EARTH

    low, high = RADIUS_DOMAIN_EARTH
    result = predict_frame(reference[INPUT_COLUMNS], cascade=True)
    radius = reference["planet_radius_earth"].to_numpy()
    expected = (radius >= low) & (radius <= high)
    assert list(result["in_domain"]) == list(expected)
    assert not expected.all(), "the reference set should straddle the domain edge"


def test_cascade_derives_the_expected_predictors():
    """Five ordinary quantities should fill everything that follows from them."""
    from exomass.predict import predict

    result = predict(
        radius=2.4,
        period=11.2,
        stellar_mass=0.91,
        stellar_radius=0.88,
        stellar_teff=5320,
    )

    assert set(result["derived"]) == {
        "transit_depth",
        "equilibrium_temp_k",
        "insolation_flux_earth",
        "stellar_luminosity",
        "ratio_distance_star_radius",
        "stellar_logg",
        "stellar_density",
        "orbit_semi_major_axis_au",
    }
    assert set(result["absent"]) == {
        "system_planet_count",
        "stellar_rot_vel",
        "stellar_metallicity_dex",
        "transit_duration",
    }
    assert result["regime"] == "small"
    assert result["typical_error_factor"] == pytest.approx(np.exp(result["cv_eps"]))
    assert result["in_domain"]
    assert np.isfinite(result["pred_mass_earth"])


def test_cascade_off_leaves_derivable_predictors_absent():
    from exomass.predict import predict

    filled = predict(radius=2.4, period=11.2, stellar_mass=0.91,
                     stellar_radius=0.88, stellar_teff=5320)
    plain = predict(radius=2.4, period=11.2, stellar_mass=0.91,
                    stellar_radius=0.88, stellar_teff=5320, cascade=False)

    assert filled["derived"] and not plain["derived"]
    assert len(plain["absent"]) > len(filled["absent"])


def test_rejects_a_planet_without_a_radius():
    from exomass.predict import predict

    result = predict(radius=float("nan"), period=11.2)
    assert result["regime"] is None
    assert not np.isfinite(result["pred_mass_earth"])


def test_rejects_two_columns_naming_the_same_quantity():
    from exomass.predict import predict_frame

    frame = pd.DataFrame({"planet_radius_earth": [2.4], "radius": [2.4]})
    with pytest.raises(ValueError, match="same quantity"):
        predict_frame(frame)


def test_treats_a_non_physical_value_as_absent():
    from exomass.predict import predict

    result = predict(radius=2.4, stellar_mass=-1.0)
    assert "stellar_mass_solar" in result["absent"]
    assert "not positive" in result["notes"]


def test_cascade_settles_on_non_physical_sources():
    """A formula fed an impossible source must not keep the loop going."""
    from exomass.physics import derive_missing_physical, ensure_columns

    frame = pd.DataFrame(
        [
            {
                "planet_radius_earth": 2.4,
                "orbital_period_days": 11.2,
                "stellar_mass_solar": -1.0,
                "stellar_radius_solar": 0.88,
                "stellar_temp_k": 5320,
            }
        ]
    )
    ensure_columns(frame)
    derive_missing_physical(frame)  # must return rather than spin
    assert pd.isna(frame["stellar_logg"].iloc[0])


def test_absent_and_non_positive_inputs_reach_the_model_as_nan():
    """A zero v sin i has no logarithm and must behave exactly like an absent one."""
    from exomass.predict import predict

    base = dict(radius=11.0, period=3.5, stellar_mass=1.1, stellar_radius=1.2,
                stellar_teff=6100, planet_count=1)
    absent = predict(**base)
    zero = predict(**base, vsini=0.0)
    assert zero["pred_log10_mass"] == absent["pred_log10_mass"]
    assert "stellar_rot_vel" in zero["absent"]


def test_no_vsini_model_ignores_a_supplied_vsini():
    from exomass.predict import predict

    base = dict(radius=11.0, period=3.5, stellar_mass=1.1, stellar_radius=1.2,
                stellar_teff=6100, planet_count=1)
    slow = predict(**base, vsini=2.0, variant="no_vsini")
    fast = predict(**base, vsini=40.0, variant="no_vsini")
    assert slow["pred_log10_mass"] == fast["pred_log10_mass"]
    assert "ignored" in slow["notes"]


def test_rejects_an_unknown_model():
    from exomass.predict import predict

    with pytest.raises(ValueError, match="unknown model"):
        predict(radius=2.4, variant="xgb")
