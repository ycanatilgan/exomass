"""Command line access to the same predictions the Python API returns."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .predict import ACCEPTED_COLUMNS, ALIASES, predict_frame
from .schema import (
    BOUNDARY_RADIUS_EARTH,
    FEATURE_RANGES,
    METADATA,
    MODEL_INFO,
    RADIUS_DOMAIN_EARTH,
    VARIANTS,
)

_FLAG_HELP = {
    "planet_radius_earth": "planet radius in Earth radii (required)",
    "orbital_period_days": "orbital period in days",
    "stellar_mass_solar": "host mass in solar masses",
    "stellar_radius_solar": "host radius in solar radii",
    "stellar_temp_k": "host effective temperature in K",
}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exomass",
        description=(
            "Predict exoplanet masses from whatever catalogue quantities you "
            "have. Only the planet radius is required; derivable quantities "
            "are filled from physics and the rest are reported as absent."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("predict", help="predict one planet or a table of them")
    for column in ACCEPTED_COLUMNS:
        run.add_argument(
            _flag(column),
            type=float,
            default=None,
            metavar="X",
            help=_FLAG_HELP.get(column, argparse.SUPPRESS),
        )
    for alias, column in sorted(ALIASES.items()):
        run.add_argument(
            _flag(alias),
            dest=f"alias_{alias}",
            type=float,
            default=None,
            metavar="X",
            help=f"alias for {_flag(column)}",
        )
    run.add_argument("--csv", help="read a table of planets instead of using flags")
    run.add_argument("--out", help="write results to this CSV instead of stdout")
    run.add_argument(
        "--no-cascade",
        action="store_true",
        help="skip the physics cascade and leave derivable quantities absent",
    )
    run.add_argument(
        "--model",
        choices=VARIANTS,
        default="full",
        help="'no_vsini' for catalogues without projected rotational velocity",
    )
    run.add_argument("--json", action="store_true", help="emit JSON")

    sub.add_parser("info", help="show the model contract and training coverage")
    return parser


def _frame_from_args(args: argparse.Namespace) -> pd.DataFrame:
    supplied: dict[str, float] = {}
    for column in ACCEPTED_COLUMNS:
        value = getattr(args, column, None)
        if value is not None:
            supplied[column] = value
    for alias, column in ALIASES.items():
        value = getattr(args, f"alias_{alias}", None)
        if value is None:
            continue
        if column in supplied:
            raise SystemExit(
                f"error: {_flag(alias)} and {_flag(column)} both set the same quantity"
            )
        supplied[column] = value
    if not supplied:
        raise SystemExit(
            "error: give at least --radius, or a table with --csv. "
            "Run 'exomass predict --help' for the full list."
        )
    return pd.DataFrame([supplied])


def _print_info() -> None:
    low, high = RADIUS_DOMAIN_EARTH
    print(f"training sample        : {METADATA['training_sample']}")
    print(f"algorithm              : {METADATA['algorithm']}")
    print(f"regime boundary        : {BOUNDARY_RADIUS_EARTH} Earth radii (labels only)")
    print(f"recommended radius     : {low} to {high} Earth radii")
    print()
    print("mean log error eps, 10x5 cross-validation grouped by host star:")
    for name, info in MODEL_INFO.items():
        eps = info["cv_eps"]
        print(
            f"  {name:<9} {len(info['feature_order'])} features, "
            f"small {eps['small']:.3f}, giant {eps['giant']:.3f}, all {eps['all']:.3f}"
        )
    test = METADATA["temporal_test_eps"]
    print(
        f"later-measured planets (full model): small {test['small']:.3f}, "
        f"giant {test['giant']:.3f}, all {test['all']:.3f}"
    )
    print()
    print("training coverage, 2nd to 98th percentile:")
    width = max(len(c) for c in FEATURE_RANGES)
    for column, (lo, hi) in FEATURE_RANGES.items():
        print(f"  {column:<{width}}  {lo:>12.4g}  to {hi:>12.4g}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "info":
        _print_info()
        return 0

    if args.csv:
        frame = pd.read_csv(args.csv)
    else:
        frame = _frame_from_args(args)

    result = predict_frame(
        frame, cascade=not args.no_cascade, variant=args.model
    )

    if args.out:
        flat = result.copy()
        for column in ("derived", "absent", "out_of_range"):
            flat[column] = flat[column].apply(lambda names: ";".join(names))
        flat.to_csv(args.out, index=False)
        print(f"wrote {len(flat)} rows to {args.out}")
        return 0

    if args.json:
        print(json.dumps(result.to_dict(orient="records"), indent=2, default=str))
        return 0

    if len(result) == 1:
        row = result.iloc[0]
        if row["regime"] is None:
            print("not predicted: no usable planet radius")
            print(f"notes: {row['notes']}")
            return 1
        print(f"mass      : {row['pred_mass_earth']:.3f} Earth masses "
              f"({row['pred_mass_jup']:.4f} Jupiter masses)")
        print(f"error     : typical factor {row['typical_error_factor']:.2f} "
              f"(eps {row['cv_eps']:.3f})")
        print(f"density   : {row['pred_density_gcm3']:.3f} g/cm3")
        print(f"regime    : {row['regime']}, model {row['model_used']}")
        if row["derived"]:
            print(f"derived   : {', '.join(row['derived'])}")
        if row["absent"]:
            print(f"absent    : {', '.join(row['absent'])}")
        if not row["in_domain"]:
            print("warning   : radius outside the recommended domain")
        if row["out_of_range"]:
            print(f"off cover : {', '.join(row['out_of_range'])}")
        if row["notes"]:
            print(f"notes     : {row['notes']}")
        return 0

    columns = ["regime", "pred_mass_earth", "typical_error_factor", "pred_density_gcm3", "in_domain"]
    print(result[columns].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
