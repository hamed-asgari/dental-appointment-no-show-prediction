"""Build and write the approved exploratory-data-analysis artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.artifacts import write_eda_tables
from src.analysis.drift import (
    summarize_categorical_drift_features,
    summarize_categorical_drift_levels,
    summarize_numeric_drift,
)
from src.analysis.figure_artifacts import write_eda_figures
from src.analysis.figures import build_eda_figures
from src.analysis.relationships import summarize_numeric_relationships
from src.analysis.summaries import (
    summarize_categorical_features,
    summarize_cohort_target,
    summarize_missingness,
    summarize_numeric_by_target,
    summarize_numeric_features,
    summarize_temporal_coverage,
    summarize_temporal_monthly,
)
from src.data import build_dataset as bd


__all__ = (
    "build_eda_tables",
    "generate_eda_artifacts",
)


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _REPOSITORY_ROOT / "data" / "raw"
_DRIFT_COLUMNS = (
    "appointment_id",
    "prediction_time",
    *bd.FEATURE_COLUMNS,
)
_SUPERVISED_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "target",
    *bd.FEATURE_COLUMNS,
)
_MATURITY_AUDIT_COLUMNS = (
    "appointment_id",
    "prediction_time",
    "split",
    "development_fit_eligible",
)
_DEVELOPMENT_REQUIRED_COLUMNS = tuple(
    dict.fromkeys(
        (
            *_SUPERVISED_COLUMNS,
            "split",
            "development_fit_eligible",
        )
    )
)


def _require_no_nulls(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
) -> None:
    missing_counts = frame.loc[:, list(columns)].isna().sum()
    missing = missing_counts[missing_counts.gt(0)].to_dict()
    if missing:
        raise ValueError(f"{context} contains null values: {missing}")


def _require_expected_dtypes(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    context: str,
) -> None:
    for column in columns:
        expected = bd.EXPECTED_DTYPES[column]
        actual = frame[column].dtype
        if expected == "string":
            matches = isinstance(actual, pd.StringDtype)
        else:
            matches = str(actual) == expected
        if not matches:
            raise ValueError(
                f"{context}.{column} dtype must be {expected}; got {actual}"
            )


def _validate_temporal_split_consistency(
    frame: pd.DataFrame,
    *,
    context: str,
) -> None:
    prediction_time = frame["prediction_time"]
    split = frame["split"]
    inconsistent = (
        (split.eq("train") & prediction_time.ge(bd.VALIDATION_START))
        | (
            split.eq("validation")
            & (
                prediction_time.lt(bd.VALIDATION_START)
                | prediction_time.ge(bd.TEST_START)
            )
        )
        | (split.eq("test") & prediction_time.lt(bd.TEST_START))
    )
    if inconsistent.any():
        raise ValueError(f"{context} contains temporal split inconsistency")


def _validate_maturity_implications(
    frame: pd.DataFrame,
    *,
    include_pretest: bool,
    context: str,
) -> None:
    development_eligible = frame["development_fit_eligible"]
    if (development_eligible & ~frame["split"].eq("train")).any():
        raise ValueError(
            f"{context}: development_fit_eligible may be true only for train rows"
        )
    if not include_pretest:
        return

    pretest_eligible = frame["pretest_fit_eligible"]
    if (pretest_eligible & ~frame["split"].isin(("train", "validation"))).any():
        raise ValueError(
            f"{context}: pretest_fit_eligible may be true only for train or "
            "validation rows"
        )
    if (development_eligible & ~pretest_eligible).any():
        raise ValueError(
            f"{context}: development_fit_eligible=True requires "
            "pretest_fit_eligible=True"
        )


def _validate_predictor_domains(
    frame: pd.DataFrame,
    *,
    context: str,
) -> None:
    lead_time = frame["booking_lead_time_hours"]
    domain_checks = {
        "planned_duration_min": frame["planned_duration_min"].gt(0),
        "booking_lead_time_hours": pd.Series(
            np.isfinite(lead_time.to_numpy()) & lead_time.ge(24).to_numpy(),
            index=frame.index,
        ),
        "scheduled_weekday": frame["scheduled_weekday"].between(0, 6),
        "scheduled_hour": frame["scheduled_hour"].between(0, 23),
        "scheduled_month": frame["scheduled_month"].between(1, 12),
        "approximate_age_at_prediction": frame[
            "approximate_age_at_prediction"
        ].ge(0),
        "patient_registration_tenure_days": frame[
            "patient_registration_tenure_days"
        ].ge(0),
        "dentist_tenure_days": frame["dentist_tenure_days"].ge(0),
    }
    for column, valid in domain_checks.items():
        if not valid.all():
            raise ValueError(f"{context}.{column} contains values outside its domain")


def _validate_canonical_semantics(canonical: pd.DataFrame) -> None:
    """Validate canonical semantics without observing protected target totals."""

    bd._validate_canonical_structure(canonical)
    _require_no_nulls(canonical, canonical.columns, context="Canonical dataset")
    if not canonical["appointment_id"].is_unique:
        raise ValueError("Canonical appointment_id values must be unique")
    if not canonical["target"].isin((0, 1)).all():
        raise ValueError("Canonical target values must be binary 0 or 1")

    observed_splits = set(canonical["split"].unique())
    if observed_splits != bd.ALLOWED_SPLITS:
        raise ValueError(
            "Canonical split values must be exactly train, validation, and test"
        )
    _validate_temporal_split_consistency(canonical, context="Canonical dataset")
    _validate_maturity_implications(
        canonical,
        include_pretest=True,
        context="Canonical dataset",
    )

    if len(canonical) != bd.EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Canonical row count must be {bd.EXPECTED_TOTAL_ROWS:,}; "
            f"got {len(canonical):,}"
        )
    for split_name, expected in bd.EXPECTED_SPLIT_ROWS.items():
        actual = int(canonical["split"].eq(split_name).sum())
        if actual != expected:
            raise ValueError(
                f"Canonical {split_name} row count must be {expected:,}; "
                f"got {actual:,}"
            )
    for maturity_column, expected in bd.EXPECTED_MATURITY_ROWS.items():
        actual = int(canonical[maturity_column].sum())
        if actual != expected:
            raise ValueError(
                f"Canonical {maturity_column} row count must be {expected:,}; "
                f"got {actual:,}"
            )
    _validate_predictor_domains(canonical, context="Canonical dataset")


def _validate_development_boundary(development: pd.DataFrame) -> None:
    """Validate the selector return before any EDA mask or projection."""

    missing = [
        column
        for column in _DEVELOPMENT_REQUIRED_COLUMNS
        if column not in development.columns
    ]
    if missing:
        raise ValueError(
            f"Development selection is missing required columns: {missing}"
        )
    _require_expected_dtypes(
        development,
        _DEVELOPMENT_REQUIRED_COLUMNS,
        context="Development selection",
    )
    _require_no_nulls(
        development,
        _DEVELOPMENT_REQUIRED_COLUMNS,
        context="Development selection",
    )
    if not development["appointment_id"].is_unique:
        raise ValueError("Development appointment_id values must be unique")
    if development["split"].eq("test").any():
        raise ValueError("Development selection unexpectedly contains test rows")
    if not set(development["split"].unique()).issubset({"train", "validation"}):
        raise ValueError(
            "Development split values must be limited to train and validation"
        )
    if development["prediction_time"].ge(bd.TEST_START).any():
        raise ValueError("Development selection contains a test-period timestamp")

    _validate_temporal_split_consistency(
        development,
        context="Development selection",
    )
    _validate_maturity_implications(
        development,
        include_pretest=False,
        context="Development selection",
    )
    mature_train = development["split"].eq("train") & development[
        "development_fit_eligible"
    ]
    if not development.loc[mature_train, "target"].isin((0, 1)).all():
        raise ValueError("Development supervised target values must be binary 0 or 1")
    _validate_predictor_domains(development, context="Development selection")


def _project_population(
    development: pd.DataFrame,
    *,
    mask: pd.Series,
    columns: Sequence[str],
    forbidden_columns: Collection[str],
    population_name: str,
) -> pd.DataFrame:
    """Project one population and enforce its output-column boundary."""

    missing = [column for column in columns if column not in development.columns]
    if missing:
        raise ValueError(
            f"{population_name} is missing required projection columns: {missing}"
        )
    projected = development.loc[mask, list(columns)].copy(deep=True)
    leaked = set(projected.columns) & set(forbidden_columns)
    if leaked:
        raise ValueError(
            f"{population_name} unexpectedly contains forbidden columns: "
            f"{sorted(leaked)}"
        )
    if tuple(projected.columns) != tuple(columns):
        raise ValueError(f"{population_name} column projection is not exact")
    return projected


def select_eda_populations(
    canonical: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return four defensive, leakage-safe EDA populations.

    Mature training rows are the only target-bearing population. Validation is
    feature-only, test rows are never exposed, and nominal training rows are
    available only through a target-free label-maturity audit.
    """

    _validate_canonical_semantics(canonical)
    development = bd.select_development_rows(canonical)
    _validate_development_boundary(development)

    mature_train = development["split"].eq("train") & development[
        "development_fit_eligible"
    ]
    validation = development["split"].eq("validation")
    nominal_train = development["split"].eq("train")

    predictor_forbidden = {
        "patient_id",
        "dentist_id",
        "split",
        "development_fit_eligible",
        "pretest_fit_eligible",
        *bd.PROHIBITED_COLUMNS,
    }
    target_free_forbidden = {"target", *predictor_forbidden}
    maturity_forbidden = {
        "target",
        "patient_id",
        "dentist_id",
        "pretest_fit_eligible",
        *bd.FEATURE_COLUMNS,
        *bd.PROHIBITED_COLUMNS,
    }

    populations = {
        "supervised_train": _project_population(
            development,
            mask=mature_train,
            columns=_SUPERVISED_COLUMNS,
            forbidden_columns=predictor_forbidden,
            population_name="supervised_train",
        ),
        "train_drift": _project_population(
            development,
            mask=mature_train,
            columns=_DRIFT_COLUMNS,
            forbidden_columns=target_free_forbidden,
            population_name="train_drift",
        ),
        "validation_drift": _project_population(
            development,
            mask=validation,
            columns=_DRIFT_COLUMNS,
            forbidden_columns=target_free_forbidden,
            population_name="validation_drift",
        ),
        "maturity_audit": _project_population(
            development,
            mask=nominal_train,
            columns=_MATURITY_AUDIT_COLUMNS,
            forbidden_columns=maturity_forbidden,
            population_name="maturity_audit",
        ),
    }
    return populations


def build_eda_tables(
    canonical: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build the deterministic in-memory bundle of approved EDA tables."""

    populations = select_eda_populations(canonical)
    supervised_train = populations["supervised_train"]
    train_drift = populations["train_drift"]
    validation_drift = populations["validation_drift"]
    maturity_audit = populations["maturity_audit"]
    return {
        "cohort_target": summarize_cohort_target(supervised_train),
        "missingness": summarize_missingness(supervised_train),
        "numeric_features": summarize_numeric_features(supervised_train),
        "numeric_by_target": summarize_numeric_by_target(supervised_train),
        "categorical_features": summarize_categorical_features(supervised_train),
        "temporal_coverage": summarize_temporal_coverage(
            supervised_train,
            maturity_audit,
        ),
        "temporal_monthly": summarize_temporal_monthly(
            supervised_train,
            maturity_audit,
        ),
        "numeric_drift": summarize_numeric_drift(
            train_drift,
            validation_drift,
        ),
        "categorical_drift_levels": summarize_categorical_drift_levels(
            train_drift,
            validation_drift,
        ),
        "categorical_drift_features": summarize_categorical_drift_features(
            train_drift,
            validation_drift,
        ),
        "numeric_relationships": summarize_numeric_relationships(train_drift),
    }


def _build_canonical_dataset() -> pd.DataFrame:
    """Build the canonical dataset in memory from verified approved inputs."""

    bd.validate_raw_hashes(_RAW_DIR)
    raw_tables = bd.load_raw_data(_RAW_DIR)
    return bd.build_analytical_dataset(raw_tables)


def generate_eda_artifacts(
    output_dir: str | Path,
) -> dict[str, dict[str, Path]]:
    """Build and write the complete deterministic EDA artifact bundle."""

    output_path = Path(output_dir)
    canonical = _build_canonical_dataset()
    tables = build_eda_tables(canonical)
    figures = build_eda_figures(tables)
    table_paths = write_eda_tables(tables, output_path)
    figure_paths = write_eda_figures(figures, output_path)
    return {"tables": table_paths, "figures": figure_paths}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the approved deterministic EDA artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/eda"),
        help="Directory for the eleven CSV and five PNG artifacts.",
    )
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = generate_eda_artifacts(args.output_dir)
    for path in paths["tables"].values():
        print(f"Wrote table: {path}")
    for path in paths["figures"].values():
        print(f"Wrote figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
