"""Pin the prediction path against a stored set of reference planets.

``tests/data/reference_planets.csv`` holds forty planets drawn evenly across
both regimes, together with the mass the released models return for each. They
span radii from well below to well above the recommended domain and carry
absent entries in nine of the predictors, so the cascade, the sentinel path
and the coverage flags are all exercised.

Nothing outside the package is needed, so these run anywhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REFERENCE = Path(__file__).parent / "data" / "reference_planets.csv"

INPUT_COLUMNS = [
    "planet_radius_earth",
    "transit_depth",
    "system_planet_count",
    "equilibrium_temp_k",
    "stellar_mass_solar",
    "insolation_flux_earth",
    "stellar_luminosity",
    "stellar_radius_solar",
    "stellar_rot_vel",
    "ratio_distance_star_radius",
    "stellar_temp_k",
    "stellar_logg",
    "stellar_density",
    "orbital_period_days",
    "stellar_metallicity_dex",
    "orbit_semi_major_axis_au",
    "transit_duration",
]


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return pd.read_csv(REFERENCE)


def test_reproduces_reference_masses(reference):
    from exomass.predict import predict_frame

    result = predict_frame(reference[INPUT_COLUMNS], cascade=True)
    delta = np.abs(
        result["pred_log10_mass"].to_numpy()
        - reference["expected_log10_mass"].to_numpy()
    )
    worst = int(np.nanargmax(delta))
    assert np.nanmax(delta) < 1e-9, (
        f"{reference['planet_name'].iloc[worst]} disagrees by "
        f"{delta[worst]:.3e} dex"
    )


def test_routes_every_planet_to_the_recorded_regime(reference):
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
