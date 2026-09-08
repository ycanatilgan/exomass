"""Physics-based cascade that fills derivable predictors.

The formulas and units here are the ones the training sample was prepared
with, so input supplied to a model goes through the same preparation.

The cascade is iterative on purpose.  Filling ``orbit_semi_major_axis_au``
from Kepler's third law unlocks the rules that depend on it, so the rules are
applied repeatedly until a pass changes nothing.

Units, by column:

    orbit_semi_major_axis_au    AU
    transit_depth               per cent
    transit_duration            hours
    stellar_logg                log10(g), cgs
    stellar_density             g/cm^3
    stellar_luminosity          log10(L/L_sun)
    ratio_distance_star_radius  dimensionless, a/R_star
    insolation_flux_earth       Earth fluxes
    equilibrium_temp_k          K, zero albedo
    stellar_rot_vel             km/s, from P_rot assuming sin(i) = 1
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Solar and astronomical constants, with the sources they were taken from.
_R_SUN_R_EARTH = 109.120   # Serenelli+2016: R_sun 6.9598e8 m / R_earth 6.3781e6 m
_AU_R_SUN = 214.946        # IAU2012: 1 AU 1.495978707e11 m / Serenelli+2016 R_sun
_LOGG_SUN = 4.438          # Heiter+2015 Tbl.2: log g_sun, log cm/s^2
_RHO_SUN = 1.4082          # Serenelli+2016 R_sun + IAU2009 M_sun, g/cm^3
_T_SUN = 5778.0            # Heiter+2015 Tbl.2: T_eff,sun, K
_R_SUN_KM = 695700.0       # Serenelli+2016 R_sun, km


# (target, [sources], vectorised formula)
RULES = [
    # Independent rules: their sources are never themselves derived.
    (
        "orbit_semi_major_axis_au",
        ["orbital_period_days", "stellar_mass_solar"],
        lambda d: (d["orbital_period_days"] / 365.25) ** (2 / 3)
        * d["stellar_mass_solar"] ** (1 / 3),
    ),
    (
        "transit_depth",
        ["planet_radius_earth", "stellar_radius_solar"],
        lambda d: (
            d["planet_radius_earth"] / (d["stellar_radius_solar"] * _R_SUN_R_EARTH)
        )
        ** 2
        * 100.0,
    ),
    (
        "stellar_logg",
        ["stellar_mass_solar", "stellar_radius_solar"],
        lambda d: _LOGG_SUN
        + np.log10(d["stellar_mass_solar"])
        - 2.0 * np.log10(d["stellar_radius_solar"]),
    ),
    (
        "stellar_density",
        ["stellar_mass_solar", "stellar_radius_solar"],
        lambda d: _RHO_SUN * d["stellar_mass_solar"] / d["stellar_radius_solar"] ** 3,
    ),
    (
        "stellar_luminosity",
        ["stellar_temp_k", "stellar_radius_solar"],
        lambda d: 2.0 * np.log10(d["stellar_radius_solar"])
        + 4.0 * np.log10(d["stellar_temp_k"] / _T_SUN),
    ),
    # Dependent rules: these can activate once a semi-major axis or a
    # luminosity has been filled by the pass above.
    (
        "transit_duration",
        [
            "orbital_period_days",
            "stellar_radius_solar",
            "planet_radius_earth",
            "orbit_semi_major_axis_au",
            "impact_parameter",
        ],
        lambda d: (d["orbital_period_days"] * 24.0 / np.pi)
        * np.arcsin(
            np.clip(
                (
                    d["stellar_radius_solar"]
                    / (d["orbit_semi_major_axis_au"] * _AU_R_SUN)
                )
                * np.sqrt(
                    np.maximum(
                        0.0,
                        (
                            1.0
                            + (
                                d["planet_radius_earth"]
                                / (d["stellar_radius_solar"] * _R_SUN_R_EARTH)
                            )
                        )
                        ** 2
                        - d["impact_parameter"] ** 2,
                    )
                ),
                -1.0,
                1.0,
            )
        ),
    ),
    (
        "ratio_distance_star_radius",
        ["orbit_semi_major_axis_au", "stellar_radius_solar"],
        lambda d: d["orbit_semi_major_axis_au"]
        * _AU_R_SUN
        / d["stellar_radius_solar"],
    ),
    (
        "insolation_flux_earth",
        ["stellar_luminosity", "orbit_semi_major_axis_au"],
        lambda d: 10.0 ** d["stellar_luminosity"]
        / d["orbit_semi_major_axis_au"] ** 2,
    ),
    (
        "equilibrium_temp_k",
        ["stellar_temp_k", "stellar_radius_solar", "orbit_semi_major_axis_au"],
        lambda d: d["stellar_temp_k"]
        * np.sqrt(
            d["stellar_radius_solar"]
            / (2.0 * d["orbit_semi_major_axis_au"] * _AU_R_SUN)
        ),
    ),
    # v [km/s] = 2 pi R_star [km] / P_rot [s], assuming sin(i) = 1, which is
    # reasonable for transiting systems.
    (
        "stellar_rot_vel",
        ["stellar_radius_solar", "stellar_rot_period"],
        lambda d: (2.0 * np.pi * d["stellar_radius_solar"] * _R_SUN_KM)
        / (d["stellar_rot_period"] * 86400.0),
    ),
]


#: Every column the cascade reads or writes.
CASCADE_COLUMNS = tuple(
    dict.fromkeys(
        [target for target, _, _ in RULES]
        + [src for _, sources, _ in RULES for src in sources]
    )
)


def ensure_columns(df: pd.DataFrame, columns=CASCADE_COLUMNS) -> pd.DataFrame:
    """Add any missing column as all-NaN so the rules can be evaluated.

    A caller supplying a handful of quantities will not carry the full archive
    schema, and every column a rule reads has to exist before the rule can be
    tried.
    """
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df


#: Safety net on the cascade loop.  The loop cannot exceed the dependency
#: depth of RULES, so this ceiling is never reached in normal use.
MAX_ITERATIONS = 32


def derive_missing_physical(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Fill NaN entries of the derivable predictors, in place.

    Returns the same frame for convenience.  Rows where a rule's sources are
    themselves absent are left untouched, and it is the caller's job to decide
    what an unfilled entry should become (see :mod:`exomass.features`).

    Progress is measured by the cells that end up holding a usable number, so
    a rule whose sources are present but non-physical, a negative stellar mass
    for instance, contributes nothing and the loop settles.
    """
    filled: dict[str, int] = {}
    iteration = 0

    while True:
        iteration += 1
        if iteration > MAX_ITERATIONS:
            raise RuntimeError(
                f"cascade failed to converge in {MAX_ITERATIONS} iterations; "
                f"filled so far: {filled}"
            )
        round_total = 0

        for target, sources, formula in RULES:
            if target not in df.columns:
                continue
            if any(src not in df.columns for src in sources):
                continue

            nan_mask = df[target].isna()
            if not nan_mask.any():
                continue

            can_fill = nan_mask.copy()
            for src in sources:
                can_fill &= df[src].notna()

            if not can_fill.any():
                continue

            with np.errstate(invalid="ignore", divide="ignore"):
                values = pd.to_numeric(
                    pd.Series(
                        np.asarray(formula(df.loc[can_fill]), dtype="float64"),
                        index=df.index[can_fill],
                    ),
                    errors="coerce",
                )
            # Only finite results count as derived; a division by a vanishing
            # radius yields an infinity, which is treated as absent.
            values = values.where(np.isfinite(values))
            df.loc[can_fill, target] = values

            # Only count cells that genuinely became usable numbers.
            gained = int((nan_mask & df[target].notna()).sum())
            if gained == 0:
                continue

            round_total += gained
            filled[target] = filled.get(target, 0) + gained
            if verbose:
                print(f"  iter {iteration} | {target:<35}: {gained} rows filled")

        if round_total == 0:
            break

    if verbose:
        print(f"  cascade finished after {iteration} iterations")

    df.attrs["cascade_filled"] = filled
    return df
