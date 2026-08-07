"""Version 2 rolling-origin model development under the frozen R2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data.build_v2_dataset import calculate_sha256, load_verified_v2_raw_tables
from src.data.export_v2_processed import (
    DEFAULT_V2_PROCESSED_DIR,
    load_frozen_v2_processed_feature_dataset,
)
from src.data.v2_targets import build_mature_v2_target_table
from src.features.schema import V2_MODEL_FEATURE_COLUMNS
from src.modeling.evaluation import evaluate_binary_probabilities


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V2_MODEL_CONFIG_PATH = (
    _REPOSITORY_ROOT / "configs" / "v2_model_development.json"
)
DEFAULT_V2_MODEL_CONTRACT_PATH = (
    _REPOSITORY_ROOT
    / "docs"
    / "v2_model_development_and_selection_contract.md"
)
DEFAULT_V2_MODELING_OUTPUT_DIR = (
    _REPOSITORY_ROOT / "reports" / "modeling" / "v2"
)

FROZEN_V2_MODEL_CONFIG_SHA256 = (
    "0b39dbe9b15c64579a81e7b9dbeaad5e5a6694fc5066698fbcfd4623a1bd1dd6"
)
FROZEN_V2_MODEL_CONTRACT_SHA256 = (
    "735953523db15e36b82bacfb022915c3eff0c4f4329f16c72877dc32f8ff597f"
)

CANDIDATE_NAMES = (
    "population_prior",
    "logistic_regression",
    "random_forest",
)

FOLD_METRIC_COLUMNS = (
    "fold",
    "model",
    "fit_time",
    "validation_label_cutoff",
    "training_rows",
    "training_positive_count",
    "training_prevalence",
    "sample_size",
    "positive_count",
    "positive_rate",
    "average_precision",
    "roc_auc",
    "brier_score",
    "log_loss",
)

PREDICTION_COLUMNS = (
    "fold",
    "model",
    "appointment_id",
    "prediction_time",
    "evaluation_partition",
    "target",
    "no_show_probability",
)

MACRO_SUMMARY_COLUMNS = (
    "model",
    "fold_count",
    "mean_average_precision",
    "mean_roc_auc",
    "mean_brier_score",
    "mean_log_loss",
    "mean_validation_positive_rate",
    "ap_uplift_vs_population_prior",
    "folds_with_positive_ap_uplift_vs_prior",
    "passes_minimum_usefulness_gate",
)

POOLED_SUMMARY_COLUMNS = (
    "model",
    "sample_size",
    "positive_count",
    "positive_rate",
    "average_precision",
    "roc_auc",
    "brier_score",
    "log_loss",
)


@dataclass(frozen=True, slots=True)
class V2RollingOriginResult:
    """Deterministic Version 2 rolling-origin ranking result."""

    fold_metrics: pd.DataFrame
    macro_summary: pd.DataFrame
    pooled_summary: pd.DataFrame
    predictions: pd.DataFrame
    selection: Mapping[str, Any]


def _load_frozen_model_config(
    config_path: Path = DEFAULT_V2_MODEL_CONFIG_PATH,
    contract_path: Path = DEFAULT_V2_MODEL_CONTRACT_PATH,
) -> dict[str, Any]:
    config_path = Path(config_path)
    contract_path = Path(contract_path)
    if calculate_sha256(config_path) != FROZEN_V2_MODEL_CONFIG_SHA256:
        raise ValueError("Frozen Version 2 model-development config SHA-256 mismatch")
    if calculate_sha256(contract_path) != FROZEN_V2_MODEL_CONTRACT_SHA256:
        raise ValueError("Frozen Version 2 model-development contract SHA-256 mismatch")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Could not load frozen Version 2 model config") from exc
    if config.get("status") != "frozen_before_any_v2_model_metric":
        raise ValueError("Version 2 model config is not in the frozen pre-metric state")
    return config


def _to_float64(values: object) -> np.ndarray:
    """Cast boolean feature blocks to dense float64 arrays."""

    if isinstance(values, pd.DataFrame):
        return values.astype(np.float64).to_numpy(dtype=np.float64, copy=True)
    array = np.asarray(values)
    return array.astype(np.float64, copy=True)


def _build_preprocessor(
    config: Mapping[str, Any],
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    roles = config["feature_roles"]
    categorical = list(roles["categorical"])
    boolean = list(roles["boolean"])
    numeric = list(roles["numeric"])

    combined = categorical + boolean + numeric
    if len(combined) != len(V2_MODEL_FEATURE_COLUMNS):
        raise ValueError("Frozen feature-role count does not match Version 2 allowlist")
    if set(combined) != set(V2_MODEL_FEATURE_COLUMNS):
        raise ValueError("Frozen feature roles do not match Version 2 allowlist")

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=config["preprocessing"]["categorical"]["fill_value"],
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    boolean_transformer = FunctionTransformer(
        _to_float64,
        validate=False,
        feature_names_out="one-to-one",
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical),
            ("boolean", boolean_transformer, boolean),
            ("numeric", numeric_pipeline, numeric),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def build_v2_candidate_estimator(
    model_name: str,
    config: Mapping[str, Any],
) -> Pipeline:
    """Build one fixed non-constant candidate from the frozen contract."""

    if model_name not in {"logistic_regression", "random_forest"}:
        raise ValueError(
            "model_name must be logistic_regression or random_forest"
        )

    candidate = config["candidates"][model_name]
    if model_name == "logistic_regression":
        estimator = LogisticRegression(
            C=float(candidate["C"]),
            penalty=str(candidate["penalty"]),
            solver=str(candidate["solver"]),
            max_iter=int(candidate["max_iter"]),
            class_weight=candidate["class_weight"],
            random_state=int(candidate["random_state"]),
        )
        preprocessor = _build_preprocessor(config, scale_numeric=True)
    else:
        estimator = RandomForestClassifier(
            n_estimators=int(candidate["n_estimators"]),
            max_depth=int(candidate["max_depth"]),
            min_samples_leaf=int(candidate["min_samples_leaf"]),
            min_samples_split=int(candidate["min_samples_split"]),
            max_features=str(candidate["max_features"]),
            bootstrap=bool(candidate["bootstrap"]),
            class_weight=candidate["class_weight"],
            random_state=int(candidate["random_state"]),
            n_jobs=int(candidate["n_jobs"]),
        )
        preprocessor = _build_preprocessor(config, scale_numeric=False)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def _select_features_by_ids(
    feature_dataset: pd.DataFrame,
    appointment_ids: pd.Series,
) -> pd.DataFrame:
    if appointment_ids.empty:
        raise ValueError("appointment_ids must not be empty")
    indexed = feature_dataset.set_index("appointment_id", verify_integrity=True)
    try:
        selected = indexed.loc[
            appointment_ids.to_numpy(dtype=np.int64),
            list(V2_MODEL_FEATURE_COLUMNS),
        ].copy(deep=True)
    except KeyError as exc:
        raise ValueError("Target appointment IDs are missing feature rows") from exc
    selected.index = pd.RangeIndex(len(selected))
    return selected


def _validation_metadata_by_ids(
    feature_dataset: pd.DataFrame,
    appointment_ids: pd.Series,
) -> pd.DataFrame:
    indexed = feature_dataset.set_index("appointment_id", verify_integrity=True)
    try:
        selected = indexed.loc[
            appointment_ids.to_numpy(dtype=np.int64),
            ["prediction_time", "evaluation_partition"],
        ].copy(deep=True)
    except KeyError as exc:
        raise ValueError("Validation target IDs are missing feature metadata") from exc
    selected = selected.reset_index()
    selected["appointment_id"] = selected["appointment_id"].astype("int64")
    selected["prediction_time"] = pd.to_datetime(
        selected["prediction_time"],
        errors="raise",
    ).astype("datetime64[ns]")
    selected["evaluation_partition"] = selected[
        "evaluation_partition"
    ].astype("string")
    return selected


def _evaluate(
    target: pd.Series,
    probability: np.ndarray,
) -> dict[str, float | int]:
    metrics = evaluate_binary_probabilities(
        target.astype("int8"),
        probability.astype(np.float64, copy=True),
    )
    return {
        "sample_size": int(len(target)),
        "positive_count": int(target.sum()),
        "positive_rate": float(target.mean()),
        **metrics,
    }


def _positive_class_probability(estimator: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probability = estimator.predict_proba(features)
    classes = np.asarray(estimator.named_steps["classifier"].classes_)
    positive = np.flatnonzero(classes == 1)
    if len(positive) != 1:
        raise RuntimeError("Candidate estimator does not expose exactly one positive class")
    result = probability[:, int(positive[0])].astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise RuntimeError("Candidate produced non-finite probabilities")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise RuntimeError("Candidate produced probabilities outside [0, 1]")
    return result


def _build_macro_summary(
    fold_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prior_by_fold = (
        fold_metrics.loc[
            fold_metrics["model"].eq("population_prior"),
            ["fold", "average_precision"],
        ]
        .set_index("fold")["average_precision"]
    )

    grouped = fold_metrics.groupby("model", sort=False)
    gate = config["ranking_selection"]["minimum_usefulness_gate"]
    for model_name in CANDIDATE_NAMES:
        model_rows = grouped.get_group(model_name).copy()
        mean_ap = float(model_rows["average_precision"].mean())
        mean_roc = float(model_rows["roc_auc"].mean())
        mean_brier = float(model_rows["brier_score"].mean())
        mean_log_loss = float(model_rows["log_loss"].mean())
        mean_positive_rate = float(model_rows["positive_rate"].mean())

        aligned = model_rows.set_index("fold")["average_precision"]
        uplift = aligned - prior_by_fold
        positive_uplift_folds = int(uplift.gt(0.0).sum())
        macro_prior_ap = float(prior_by_fold.mean())
        absolute_uplift = float(mean_ap - macro_prior_ap)

        passes = False
        if model_name != "population_prior":
            passes = (
                absolute_uplift
                >= float(gate["mean_average_precision_absolute_uplift_vs_prior"])
                and mean_roc >= float(gate["mean_roc_auc_minimum"])
                and positive_uplift_folds
                >= int(gate["minimum_folds_with_positive_ap_uplift_vs_prior"])
            )

        rows.append(
            {
                "model": model_name,
                "fold_count": int(len(model_rows)),
                "mean_average_precision": mean_ap,
                "mean_roc_auc": mean_roc,
                "mean_brier_score": mean_brier,
                "mean_log_loss": mean_log_loss,
                "mean_validation_positive_rate": mean_positive_rate,
                "ap_uplift_vs_population_prior": absolute_uplift,
                "folds_with_positive_ap_uplift_vs_prior": positive_uplift_folds,
                "passes_minimum_usefulness_gate": bool(passes),
            }
        )

    return pd.DataFrame(rows, columns=list(MACRO_SUMMARY_COLUMNS))


def _build_pooled_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name in CANDIDATE_NAMES:
        subset = predictions.loc[predictions["model"].eq(model_name)]
        metrics = _evaluate(
            subset["target"].astype("int8"),
            subset["no_show_probability"].to_numpy(dtype=np.float64),
        )
        rows.append({"model": model_name, **metrics})
    return pd.DataFrame(rows, columns=list(POOLED_SUMMARY_COLUMNS))


def _choose_ranking_model(
    macro_summary: pd.DataFrame,
) -> dict[str, Any]:
    eligible = macro_summary.loc[
        macro_summary["passes_minimum_usefulness_gate"]
    ].copy()

    if eligible.empty:
        return {
            "selected_ranking_model": "population_prior",
            "fallback_to_population_prior": True,
            "eligible_nonconstant_candidates": [],
            "selection_reason": (
                "No non-constant candidate passed the frozen minimum usefulness gate."
            ),
        }

    preference = {
        "logistic_regression": 0,
        "random_forest": 1,
    }
    eligible["_tie_preference"] = eligible["model"].map(preference)
    eligible = eligible.sort_values(
        [
            "mean_average_precision",
            "mean_roc_auc",
            "mean_brier_score",
            "mean_log_loss",
            "_tie_preference",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    selected = str(eligible.iloc[0]["model"])
    return {
        "selected_ranking_model": selected,
        "fallback_to_population_prior": False,
        "eligible_nonconstant_candidates": [
            str(value) for value in eligible["model"].tolist()
        ],
        "selection_reason": (
            "Selected by the frozen AP, ROC-AUC, Brier, log-loss, and "
            "Logistic Regression tie-break ordering."
        ),
    }


def run_v2_rolling_origin_development(
    *,
    config_path: Path = DEFAULT_V2_MODEL_CONFIG_PATH,
    contract_path: Path = DEFAULT_V2_MODEL_CONTRACT_PATH,
    processed_dir: Path = DEFAULT_V2_PROCESSED_DIR,
) -> V2RollingOriginResult:
    """Run the frozen three-fold Version 2 ranking comparison."""

    config = _load_frozen_model_config(config_path, contract_path)

    protected = config["protected_final_test"]
    if any(
        bool(protected[key])
        for key in (
            "target_access_permitted_during_r2",
            "probability_vector_generation_permitted_during_r2",
            "metric_computation_permitted_during_r2",
        )
    ):
        raise RuntimeError("Frozen R2 contract no longer protects final_test")

    feature_dataset = load_frozen_v2_processed_feature_dataset(
        Path(processed_dir)
    )
    if "target" in feature_dataset.columns:
        raise RuntimeError("Target reached the frozen feature artifact")

    tables = load_verified_v2_raw_tables()
    fold_metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold in config["rolling_origin"]:
        fold_name = str(fold["name"])
        fit_time = pd.Timestamp(fold["fit_time"])
        validation_cutoff = pd.Timestamp(fold["validation_label_cutoff"])
        training_partitions = tuple(fold["training_partitions"])
        validation_partition = str(fold["validation_partition"])

        if "final_test" in training_partitions or validation_partition == "final_test":
            raise RuntimeError("Rolling-origin contract attempted to use final_test")

        training_targets = build_mature_v2_target_table(
            feature_dataset,
            tables.appointments,
            model_fit_time=fit_time,
            allowed_partitions=training_partitions,
        )
        validation_targets = build_mature_v2_target_table(
            feature_dataset,
            tables.appointments,
            model_fit_time=validation_cutoff,
            allowed_partitions=(validation_partition,),
        )

        if training_targets["evaluation_partition"].eq("final_test").any():
            raise RuntimeError("Training target access exposed final_test")
        if validation_targets["evaluation_partition"].eq("final_test").any():
            raise RuntimeError("Validation target access exposed final_test")

        training_features = _select_features_by_ids(
            feature_dataset,
            training_targets["appointment_id"],
        )
        validation_features = _select_features_by_ids(
            feature_dataset,
            validation_targets["appointment_id"],
        )
        training_target = training_targets["target"].astype("int8").reset_index(
            drop=True
        )
        validation_target = validation_targets["target"].astype("int8").reset_index(
            drop=True
        )

        if set(training_target.unique()) != {0, 1}:
            raise RuntimeError(f"{fold_name} training target must contain both classes")
        if set(validation_target.unique()) != {0, 1}:
            raise RuntimeError(f"{fold_name} validation target must contain both classes")

        validation_metadata = _validation_metadata_by_ids(
            feature_dataset,
            validation_targets["appointment_id"],
        )
        if not validation_metadata["evaluation_partition"].eq(
            validation_partition
        ).all():
            raise RuntimeError("Validation metadata partition mismatch")
        if validation_metadata["prediction_time"].ge(validation_cutoff).any():
            # The configured validation windows should already end before this cutoff.
            raise RuntimeError("Validation feature row reached its label cutoff boundary")

        training_prevalence = float(training_target.mean())
        model_probabilities: dict[str, np.ndarray] = {
            "population_prior": np.full(
                len(validation_target),
                training_prevalence,
                dtype=np.float64,
            )
        }

        for model_name in ("logistic_regression", "random_forest"):
            estimator = build_v2_candidate_estimator(model_name, config)
            estimator.fit(training_features, training_target)
            model_probabilities[model_name] = _positive_class_probability(
                estimator,
                validation_features,
            )

        for model_name in CANDIDATE_NAMES:
            probability = model_probabilities[model_name]
            metrics = _evaluate(validation_target, probability)
            fold_metric_rows.append(
                {
                    "fold": fold_name,
                    "model": model_name,
                    "fit_time": fit_time,
                    "validation_label_cutoff": validation_cutoff,
                    "training_rows": int(len(training_target)),
                    "training_positive_count": int(training_target.sum()),
                    "training_prevalence": training_prevalence,
                    **metrics,
                }
            )

            prediction_frame = validation_metadata.copy(deep=True)
            prediction_frame.insert(0, "model", model_name)
            prediction_frame.insert(0, "fold", fold_name)
            prediction_frame["target"] = validation_target.to_numpy(
                dtype=np.int8
            )
            prediction_frame["no_show_probability"] = probability
            prediction_frame = prediction_frame.loc[
                :,
                list(PREDICTION_COLUMNS),
            ]
            prediction_frames.append(prediction_frame)

    fold_metrics = pd.DataFrame(
        fold_metric_rows,
        columns=list(FOLD_METRIC_COLUMNS),
    )
    fold_metrics["fit_time"] = pd.to_datetime(
        fold_metrics["fit_time"]
    ).astype("datetime64[ns]")
    fold_metrics["validation_label_cutoff"] = pd.to_datetime(
        fold_metrics["validation_label_cutoff"]
    ).astype("datetime64[ns]")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions["fold"] = predictions["fold"].astype("string")
    predictions["model"] = predictions["model"].astype("string")
    predictions["appointment_id"] = predictions["appointment_id"].astype("int64")
    predictions["prediction_time"] = pd.to_datetime(
        predictions["prediction_time"]
    ).astype("datetime64[ns]")
    predictions["evaluation_partition"] = predictions[
        "evaluation_partition"
    ].astype("string")
    predictions["target"] = predictions["target"].astype("int8")
    predictions["no_show_probability"] = predictions[
        "no_show_probability"
    ].astype("float64")

    if predictions["evaluation_partition"].eq("final_test").any():
        raise RuntimeError("Rolling-origin predictions exposed final_test rows")
    if predictions.duplicated(["fold", "model", "appointment_id"]).any():
        raise RuntimeError("Rolling-origin prediction keys are not unique")

    macro_summary = _build_macro_summary(fold_metrics, config)
    pooled_summary = _build_pooled_summary(predictions)
    selection = _choose_ranking_model(macro_summary)
    selection = {
        **selection,
        "contract_config_sha256": FROZEN_V2_MODEL_CONFIG_SHA256,
        "contract_document_sha256": FROZEN_V2_MODEL_CONTRACT_SHA256,
        "final_test_target_accessed": False,
        "final_test_probabilities_generated": False,
    }

    return V2RollingOriginResult(
        fold_metrics=fold_metrics,
        macro_summary=macro_summary,
        pooled_summary=pooled_summary,
        predictions=predictions,
        selection=selection,
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.12g",
        date_format="%Y-%m-%dT%H:%M:%S",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_v2_rolling_origin_results(
    result: V2RollingOriginResult,
    *,
    output_dir: Path = DEFAULT_V2_MODELING_OUTPUT_DIR,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write deterministic R2 rolling-origin results and a hash manifest."""

    destination = Path(output_dir)
    paths = {
        "fold_metrics": destination / "rolling_origin_fold_metrics.csv",
        "macro_summary": destination / "rolling_origin_macro_summary.csv",
        "pooled_summary": destination / "rolling_origin_pooled_summary.csv",
        "predictions": destination / "rolling_origin_predictions.csv",
        "selection": destination / "ranking_selection.json",
        "manifest": destination / "rolling_origin_manifest.json",
    }

    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "Rolling-origin outputs already exist; use overwrite=True to replace them"
        )

    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(result.fold_metrics, paths["fold_metrics"])
    _write_csv(result.macro_summary, paths["macro_summary"])
    _write_csv(result.pooled_summary, paths["pooled_summary"])
    _write_csv(result.predictions, paths["predictions"])

    paths["selection"].write_text(
        json.dumps(dict(result.selection), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    artifact_entries: dict[str, dict[str, object]] = {}
    for key in (
        "fold_metrics",
        "macro_summary",
        "pooled_summary",
        "predictions",
        "selection",
    ):
        path = paths[key]
        artifact_entries[path.name] = {
            "sha256": _sha256(path),
            "size_bytes": int(path.stat().st_size),
        }

    manifest = {
        "schema_version": "1.0.0",
        "phase": "R2",
        "stage": "rolling_origin_ranking",
        "model_config_sha256": FROZEN_V2_MODEL_CONFIG_SHA256,
        "model_contract_sha256": FROZEN_V2_MODEL_CONTRACT_SHA256,
        "processed_dataset_sha256": (
            "08a2c16ca6cc66f91fda1cd09a2549a3e2d5357c2b975eb2f55f4ade66a46b53"
        ),
        "processed_manifest_sha256": (
            "2ee3f7d42f2d73fdcde71fd601fd0423d5e610767ac5162afd38c33bf2fb8073"
        ),
        "processed_dataset_fingerprint": (
            "0f3243e4ef3d832ae5562a98354828bad481a6691a0e17e6a5271307143b5787"
        ),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "selected_ranking_model": result.selection["selected_ranking_model"],
        "fallback_to_population_prior": result.selection[
            "fallback_to_population_prior"
        ],
        "final_test_target_accessed": False,
        "final_test_probabilities_generated": False,
        "artifacts": artifact_entries,
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Version 2 rolling-origin model ranking comparison "
            "without accessing protected final-test targets."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_V2_MODELING_OUTPUT_DIR,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_v2_rolling_origin_development()
    manifest = export_v2_rolling_origin_results(
        result,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    print("Version 2 rolling-origin ranking complete.")
    print(result.fold_metrics.to_string(index=False))
    print("\nMacro fold summary:")
    print(result.macro_summary.to_string(index=False))
    print("\nPooled validation summary:")
    print(result.pooled_summary.to_string(index=False))
    print(
        "\nSelected ranking model: "
        f"{result.selection['selected_ranking_model']}"
    )
    print(
        "Fallback to population prior: "
        f"{result.selection['fallback_to_population_prior']}"
    )
    print("Final-test target accessed: false")
    print("Final-test probabilities generated: false")
    print(
        "Rolling-origin manifest: "
        f"{Path(args.output_dir) / 'rolling_origin_manifest.json'}"
    )
    print(
        "Manifest stage: "
        f"{manifest['stage']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CANDIDATE_NAMES",
    "DEFAULT_V2_MODELING_OUTPUT_DIR",
    "DEFAULT_V2_MODEL_CONFIG_PATH",
    "DEFAULT_V2_MODEL_CONTRACT_PATH",
    "FOLD_METRIC_COLUMNS",
    "FROZEN_V2_MODEL_CONFIG_SHA256",
    "FROZEN_V2_MODEL_CONTRACT_SHA256",
    "MACRO_SUMMARY_COLUMNS",
    "POOLED_SUMMARY_COLUMNS",
    "PREDICTION_COLUMNS",
    "V2RollingOriginResult",
    "build_v2_candidate_estimator",
    "export_v2_rolling_origin_results",
    "run_v2_rolling_origin_development",
)
