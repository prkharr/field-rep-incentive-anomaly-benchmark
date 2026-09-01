"""Finalized PCA scoring and controlled-benchmark evaluation at rep-period grain.

The module is deliberately additive: it does not select a model zoo, mutate its
inputs, or write project artifacts.  The PCA model and preprocessing pipeline are
fit on clean training rows only.  A manager-review threshold is frozen from the
unlabelled validation score distribution; controlled labels are joined only after
that threshold has been selected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve as sklearn_precision_recall_curve
from sklearn.metrics import roc_curve as sklearn_roc_curve

from ..evaluation import classification_metrics, ranking_metrics
from ..extended_scoring import PercentileCalibrator
from ..models.pca_reconstruction import PCAReconstruction
from ..preprocessing import fit_preprocessor


MODEL_NAME = "PCA Reconstruction"
DEFAULT_CUTOFFS = (0.01, 0.05, 0.10)
DEFAULT_GROUP_COLUMNS = ("manager_id", "team_id", "territory_id")
PEER_LABEL_ALIASES = {
    "net_sales": "sales",
    "final_incentive_paid": "incentive",
    "average_discount_pct": "discount",
    "claimed_expense_amount": "expense",
}
KNOWN_TRUTH_COLUMNS = {
    "injection_id",
    "entity_type",
    "entity_id",
    "anomaly_type",
    "anomaly_category",
    "severity",
    "correlated_case_flag",
    "ground_truth_label",
    "injected_anomaly_flag",
    "affected_dataset",
    "affected_record_ids",
    "injection_description",
    "original_value",
    "injected_value",
    "expected_detection_signals",
}
CONTEXT_COLUMNS = {
    "observation_id",
    "source_row_id",
    "rep_id",
    "rep_name",
    "manager_id",
    "manager_name",
    "team_id",
    "team_name",
    "territory_id",
    "territory_name",
    "period",
    "date",
    "split",
    "data_lineage",
    "currency_code",
}
OUTPUT_FILES = {
    "clean_scores": "pca_clean_scores.csv",
    "injected_scores": "pca_injected_scores.csv",
    "metrics_summary": "pca_metrics_summary.csv",
    "top_k_metrics": "pca_top_k_metrics.csv",
    "group_metrics": "pca_group_metrics.csv",
    "period_stability": "pca_period_stability.csv",
    "score_distributions": "pca_score_distributions.csv",
    "roc_curve": "pca_roc_curve.csv",
    "pr_curve": "pca_pr_curve.csv",
    "lift_curve": "pca_lift_curve.csv",
    "feature_contributions": "pca_feature_contributions.csv",
    "false_positive_review": "pca_false_positive_review.csv",
}


def _load_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    path = Path(config)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise TypeError("PCA config must contain a mapping.")
    return dict(loaded)


def _setting(root: Mapping[str, Any], model: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for source in (model, root):
        for name in names:
            if name in source:
                return source[name]
    return default


def _model_settings(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    model = config.get("model", {})
    if model is None:
        model = {}
    if not isinstance(model, Mapping):
        raise TypeError("config['model'] must be a mapping.")
    return dict(config), dict(model)


def _period_column(clean: pd.DataFrame, injected: pd.DataFrame, root: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    configured = _setting(root, model, ("period_column", "date_column"))
    if configured:
        column = str(configured)
        if column not in clean or column not in injected:
            raise ValueError(f"Configured period column is missing: {column}")
        return column
    for candidate in ("period", "date"):
        if candidate in clean and candidate in injected:
            return candidate
    raise ValueError("Inputs require a shared 'period' or 'date' column.")


def _normalise_period(frame: pd.DataFrame, period_column: str, label: str) -> pd.DataFrame:
    result = frame.copy(deep=True)
    result[period_column] = pd.to_datetime(result[period_column], errors="coerce")
    if result[period_column].isna().any():
        raise ValueError(f"{label} contains invalid or missing periods.")
    return result


def _join_keys(
    clean: pd.DataFrame,
    injected: pd.DataFrame,
    truth: pd.DataFrame,
    period_column: str,
    root: Mapping[str, Any],
    model: Mapping[str, Any],
) -> list[str]:
    configured = _setting(root, model, ("id_columns", "join_keys", "grain_columns"))
    if configured:
        keys = [str(value) for value in configured]
    elif "observation_id" in clean and "observation_id" in injected and "observation_id" in truth:
        keys = ["observation_id"]
    elif all(column in clean and column in injected and column in truth for column in ("rep_id", period_column)):
        keys = ["rep_id", period_column]
    else:
        common = set(clean) & set(injected) & set(truth)
        keys = [column for column in ("rep_id", "entity_id", period_column) if column in common]
    if not keys or period_column not in keys and "observation_id" not in keys:
        raise ValueError("Could not infer a unique rep-period join key; configure id_columns.")
    for label, frame in (("clean_features", clean), ("injected_features", injected), ("ground_truth", truth)):
        missing = [column for column in keys if column not in frame]
        if missing:
            raise ValueError(f"{label} is missing join columns: {missing}")
    return keys


def _ensure_unique(frame: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    if frame[list(keys)].isna().any().any():
        raise ValueError(f"{label} contains null join keys: {list(keys)}")
    if frame.duplicated(list(keys)).any():
        raise ValueError(f"{label} is not unique at rep-period grain: {list(keys)}")


def _observation_ids(frame: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    if "observation_id" in frame:
        values = frame["observation_id"].astype("string")
        if values.isna().any() or values.duplicated().any():
            raise ValueError("observation_id must be complete and unique.")
        return values

    def encode(values: tuple[Any, ...]) -> str:
        payload = "|".join(str(value) for value in values)
        return "rp_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    return pd.Series((encode(row) for row in frame[list(keys)].itertuples(index=False, name=None)), index=frame.index, dtype="string")


def _truth_column(truth: pd.DataFrame, configured: Any, candidates: Sequence[str], required: bool = False) -> str | None:
    if configured:
        column = str(configured)
        if column not in truth:
            raise ValueError(f"Configured ground-truth column is missing: {column}")
        return column
    for column in candidates:
        if column in truth:
            return column
    if required:
        raise ValueError(f"Ground truth requires one of: {list(candidates)}")
    return None


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    numeric = pd.to_numeric(values, errors="coerce")
    text = values.astype("string").str.strip().str.lower()
    mapped = text.map({"true": True, "yes": True, "y": True, "1": True, "false": False, "no": False, "n": False, "0": False})
    result = mapped.where(mapped.notna(), numeric.ne(0).where(numeric.notna()))
    return result.astype("boolean").fillna(False).astype(bool)


def _joined_strings(values: pd.Series) -> str:
    clean = sorted({str(value) for value in values.dropna() if str(value).strip() and str(value).lower() not in {"none", "nan"}})
    return "|".join(clean) if clean else "none"


def _prepare_truth(
    ground_truth: pd.DataFrame,
    keys: Sequence[str],
    period_column: str,
    root: Mapping[str, Any],
    model: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    truth = _normalise_period(ground_truth, period_column, "ground_truth") if period_column in ground_truth else ground_truth.copy(deep=True)
    label_column = _truth_column(
        truth,
        _setting(root, model, ("label_column", "ground_truth_column")),
        ("ground_truth_label", "injected_anomaly_flag", "anomaly_flag", "label"),
        required=True,
    )
    type_column = _truth_column(truth, _setting(root, model, ("anomaly_type_column",)), ("anomaly_type",))
    category_column = _truth_column(truth, _setting(root, model, ("anomaly_category_column",)), ("anomaly_category",))
    severity_column = _truth_column(truth, _setting(root, model, ("severity_column",)), ("severity", "anomaly_severity"))
    truth = truth.copy()
    truth["ground_truth_label"] = _as_bool(truth[label_column])
    aggregations: dict[str, Any] = {"ground_truth_label": "max"}
    canonical = {
        "anomaly_type": type_column,
        "anomaly_category": category_column,
        "severity": severity_column,
    }
    for target, source in canonical.items():
        if source:
            truth[target] = truth[source]
            aggregations[target] = _joined_strings
    for column in DEFAULT_GROUP_COLUMNS:
        if column in truth and column not in keys:
            aggregations[column] = "first"
    result = truth.groupby(list(keys), dropna=False, observed=True).agg(aggregations).reset_index()
    for column in ("anomaly_type", "anomaly_category", "severity"):
        if column not in result:
            result[column] = "none"
    return result, {
        "label": label_column,
        "anomaly_type": type_column,
        "anomaly_category": category_column,
        "severity": severity_column,
    }


def _feature_allowlist(
    clean: pd.DataFrame,
    injected: pd.DataFrame,
    truth: pd.DataFrame,
    keys: Sequence[str],
    period_column: str,
    root: Mapping[str, Any],
    model: Mapping[str, Any],
) -> list[str]:
    configured = _setting(root, model, ("feature_columns", "features", "model_features"))
    if configured is not None:
        features = [str(column) for column in configured]
        if len(features) != len(set(features)):
            raise ValueError("Feature allowlist contains duplicates.")
    else:
        excluded = set(keys) | CONTEXT_COLUMNS | set(truth.columns) | KNOWN_TRUTH_COLUMNS | {period_column}
        features = [
            column
            for column in clean.columns
            if column in injected and column not in excluded and pd.api.types.is_numeric_dtype(clean[column])
        ]
    if not features:
        raise ValueError("No numeric PCA features were configured or inferred.")
    missing_clean = [column for column in features if column not in clean]
    missing_injected = [column for column in features if column not in injected]
    if missing_clean or missing_injected:
        raise ValueError(f"Model features missing; clean={missing_clean}, injected={missing_injected}")
    truth_overlap = set(features) & (set(truth.columns) | KNOWN_TRUTH_COLUMNS)
    suspicious = {
        column
        for column in features
        if any(token in column.lower() for token in ("ground_truth", "anomaly_type", "anomaly_category", "severity", "injection_id"))
    }
    if truth_overlap or suspicious:
        raise ValueError(f"Ground-truth leakage in PCA feature allowlist: {sorted(truth_overlap | suspicious)}")
    non_numeric = [column for column in features if not pd.api.types.is_numeric_dtype(clean[column]) or not pd.api.types.is_numeric_dtype(injected[column])]
    if non_numeric:
        raise TypeError(f"PCA features must be numeric in both populations: {non_numeric}")
    empty_train_candidates = [column for column in features if clean[column].notna().sum() == 0]
    if empty_train_candidates:
        raise ValueError(f"PCA features cannot be entirely missing: {empty_train_candidates}")
    return features


def _split_labels(periods: pd.Series, train_end: pd.Timestamp, validation_end: pd.Timestamp) -> pd.Series:
    return pd.Series(
        np.select(
            [periods.le(train_end), periods.le(validation_end)],
            ["train", "validation"],
            default="test",
        ),
        index=periods.index,
        dtype="string",
    )


def _raw_review_threshold(scores: np.ndarray, fraction: float) -> tuple[float, int]:
    values = np.asarray(scores, dtype=float)
    if not 0 < fraction <= 1:
        raise ValueError("manager_review_fraction must be in (0, 1].")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Validation scores must be non-empty and finite.")
    count = max(1, int(math.ceil(values.size * fraction)))
    ordered = np.sort(values)[::-1]
    return float(ordered[count - 1]), count


def _exact_review_flags(scores: np.ndarray, observation_ids: Sequence[Any], fraction: float) -> np.ndarray:
    """Exact review budget with observation id as the deterministic tie-breaker."""
    values = np.asarray(scores, dtype=float)
    identifiers = np.asarray(observation_ids, dtype=str)
    if len(values) != len(identifiers):
        raise ValueError("Scores and observation ids must align for review selection.")
    if not 0 < fraction <= 1:
        raise ValueError("Review fraction must be in (0, 1].")
    count = max(1, int(math.ceil(len(values) * fraction)))
    order = np.lexsort((identifiers, -np.nan_to_num(values, nan=-np.inf)))
    flags = np.zeros(len(values), dtype=bool)
    flags[order[:count]] = True
    return flags


def _friendly_feature(feature: str) -> str:
    return feature.removeprefix("simulated_").replace("_", " ").strip().capitalize()


def _number(value: Any) -> str:
    if pd.isna(value):
        return "unavailable"
    try:
        return f"{float(value):,.4g}"
    except (TypeError, ValueError):
        return str(value)


def _peer_column(frame: pd.DataFrame, feature: str) -> str | None:
    base = re.sub(r"_(peer_z|peer_percentile|history_deviation)$", "", feature)
    alias = PEER_LABEL_ALIASES.get(base)
    candidates = (
        f"{feature}_peer_value",
        f"{feature}_peer_median",
        f"peer_{feature}_median",
        f"{base}_peer_median",
        f"{base}_peer_value",
        f"{alias}_peer_median" if alias else "",
    )
    return next((column for column in candidates if column in frame and column != feature), None)


def _percentile_series(frame: pd.DataFrame, feature: str, period_column: str) -> pd.Series:
    alias = PEER_LABEL_ALIASES.get(feature)
    explicit = next(
        (
            column
            for column in (
                f"{feature}_peer_percentile",
                f"{feature}_percentile",
                f"{alias}_peer_percentile" if alias else "",
            )
            if column in frame and column != feature
        ),
        None,
    )
    if explicit:
        values = pd.to_numeric(frame[explicit], errors="coerce")
        finite = values[np.isfinite(values)]
        if not finite.empty and finite.max() > 1 and finite.max() <= 100:
            values = values / 100.0
        return values.clip(0, 1)
    return frame.groupby(period_column, observed=True)[feature].rank(method="average", pct=True)


def _recommended_action(feature: str) -> str:
    name = feature.lower()
    if any(token in name for token in ("incentive", "payout", "attainment", "target", "quota", "accelerator", "adjustment")):
        return "Validate the target, policy tier, adjustment approval, and payout reconciliation before action."
    if any(token in name for token in ("discount", "return", "cancel", "order", "sales")):
        return "Validate the underlying orders, discounts, returns, period timing, and customer context before action."
    if any(token in name for token in ("visit", "crm", "interaction", "travel", "expense", "distance")):
        return "Validate visit, CRM, route, and expense records with the representative and manager before action."
    if any(token in name for token in ("capacity", "utilization", "workload", "fte", "coverage")):
        return "Review workload assumptions, customer coverage, and available capacity; do not treat this as an automated staffing decision."
    return "Validate the source record, peer context, and relevant commercial activity before action."


def _add_explanations(
    source: pd.DataFrame,
    scored: pd.DataFrame,
    transformed: np.ndarray,
    model: PCAReconstruction,
    features: Sequence[str],
    period_column: str,
    population: str,
    top_count: int,
    keys: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contributions = np.asarray(model.contributions(transformed), dtype=float)
    if contributions.shape != (len(source), len(features)):
        raise RuntimeError("PCA contribution matrix does not align with the feature allowlist.")
    top_count = min(max(int(top_count), 1), len(features))
    top = np.argsort(-contributions, axis=1, kind="stable")[:, :top_count]
    percentiles = {feature: _percentile_series(source, feature, period_column) for feature in features}
    peer_columns = {feature: _peer_column(source, feature) for feature in features}
    result = scored.copy()
    long_rows: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    primary_reasons: list[str] = []
    secondary_reasons: list[str] = []
    actions: list[str] = []
    driver_payload: dict[int, dict[str, list[Any]]] = {
        rank: {name: [] for name in ("feature", "name", "value", "peer_value", "percentile", "contribution")}
        for rank in range(1, top_count + 1)
    }
    for row_position, indices in enumerate(top):
        details: list[dict[str, Any]] = []
        for rank, feature_index in enumerate(indices, start=1):
            feature = str(features[int(feature_index)])
            peer_column = peer_columns[feature]
            value = source.iloc[row_position][feature]
            peer_value = source.iloc[row_position][peer_column] if peer_column else np.nan
            percentile = percentiles[feature].iloc[row_position]
            contribution = float(contributions[row_position, feature_index])
            detail = {
                "feature": feature,
                "name": _friendly_feature(feature),
                "value": value,
                "peer_value": peer_value,
                "percentile": percentile,
                "contribution": contribution,
            }
            details.append(detail)
            for name, payload in driver_payload[rank].items():
                payload.append(detail[name])
        first = details[0]
        token = re.sub(r"[^A-Z0-9]+", "_", first["feature"].upper()).strip("_")
        code = f"PCA_RECONSTRUCTION_{token}"
        if pd.notna(first["peer_value"]):
            primary = (
                f"{first['name']} was {_number(first['value'])} versus a comparable-representative "
                f"reference of {_number(first['peer_value'])}; it was the largest PCA reconstruction deviation."
            )
        else:
            primary = f"{first['name']} was the largest PCA reconstruction deviation (observed {_number(first['value'])})."
        if len(details) > 1:
            second = details[1]
            secondary = f"{second['name']} was the next-largest reconstruction deviation (observed {_number(second['value'])})."
        else:
            secondary = "No secondary reconstruction driver was available."
        action = _recommended_action(first["feature"])
        reason_codes.append(code)
        primary_reasons.append(primary)
        secondary_reasons.append(secondary)
        actions.append(action)
        context = list(dict.fromkeys([*keys, "rep_id", "manager_id", "team_id", "territory_id"]))
        common = {column: source.iloc[row_position][column] for column in context if column in source}
        common.update(
            observation_id=result.iloc[row_position]["observation_id"],
            population=population,
            period=source.iloc[row_position][period_column],
            anomaly_score=result.iloc[row_position]["anomaly_score"],
            primary_reason_code=code,
            primary_reason=primary,
            secondary_reason=secondary,
            recommended_review_action=action,
        )
        for rank, detail in enumerate(details, start=1):
            long_rows.append({**common, "driver_rank": rank, **detail})
    result["primary_reason_code"] = reason_codes
    result["primary_reason"] = primary_reasons
    result["secondary_reason"] = secondary_reasons
    result["recommended_review_action"] = actions
    for rank, values in driver_payload.items():
        for name, payload in values.items():
            result[f"driver_{rank}_{name}"] = payload
    return result, pd.DataFrame(long_rows)


def _score_population(
    source: pd.DataFrame,
    transformed: np.ndarray,
    model: PCAReconstruction,
    calibrator: PercentileCalibrator,
    raw_threshold: float,
    manager_fraction: float,
    split: pd.Series,
    observation_ids: pd.Series,
) -> pd.DataFrame:
    raw = np.asarray(model.raw_score(transformed), dtype=float)
    percentile = np.asarray(calibrator.transform(raw), dtype=float)
    if not np.isfinite(raw).all() or not np.isfinite(percentile).all():
        raise ValueError("PCA scoring produced non-finite values.")
    result = source.copy(deep=True)
    result["observation_id"] = observation_ids.to_numpy()
    result["split"] = split.to_numpy()
    result["model_name"] = MODEL_NAME
    result["raw_score"] = raw
    result["anomaly_score"] = percentile
    result["anomaly_percentile"] = percentile * 100.0
    result["raw_threshold"] = raw_threshold
    result["threshold_flag"] = raw >= raw_threshold
    # Operational membership always uses the one validation-frozen raw threshold.
    # Split-local top-k selection is confined to explicitly named ranking metrics.
    result["manager_review_flag"] = result["threshold_flag"]
    result["review_budget_flag"] = result["threshold_flag"]
    return result


def _top_k_metrics(test: pd.DataFrame, cutoffs: Sequence[float]) -> tuple[pd.DataFrame, dict[float, np.ndarray]]:
    truth = test["ground_truth_label"].to_numpy(bool)
    scores = test["raw_score"].to_numpy(float)
    prevalence = float(truth.mean()) if len(truth) else np.nan
    flags: dict[float, np.ndarray] = {}
    rows = []
    for cutoff in cutoffs:
        selected = _exact_review_flags(scores, test["observation_id"], float(cutoff))
        flags[float(cutoff)] = selected
        captured = int(np.sum(selected & truth))
        precision = captured / max(int(selected.sum()), 1)
        recall = captured / int(truth.sum()) if truth.any() else np.nan
        rows.append(
            {
                "model": MODEL_NAME,
                "split": "test",
                "review_fraction": float(cutoff),
                "review_count": int(selected.sum()),
                "positive_count": int(truth.sum()),
                "captured_count": captured,
                "precision": float(precision),
                "recall": float(recall) if np.isfinite(recall) else np.nan,
                "lift": float(precision / prevalence) if prevalence > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows), flags


def _summary_metrics(
    test: pd.DataFrame,
    raw_threshold: float,
    manager_fraction: float,
    validation_rows: int,
    validation_target_count: int,
) -> pd.DataFrame:
    ordered = test.sort_values(["raw_score", "observation_id"], ascending=[False, True], kind="stable")
    truth = ordered["ground_truth_label"].to_numpy(bool)
    scores = ordered["raw_score"].to_numpy(float)
    selected = ordered["threshold_flag"].to_numpy(bool)
    classification = classification_metrics(truth, scores, selected)
    ranked = ranking_metrics(truth, scores, cutoffs=DEFAULT_CUTOFFS)
    return pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "split": "test",
                "threshold": raw_threshold,
                "threshold_basis": "unlabelled injected validation top manager-review fraction",
                "manager_review_fraction": manager_fraction,
                "validation_rows": validation_rows,
                "validation_target_review_count": validation_target_count,
                "test_rows": len(test),
                "test_prevalence": float(truth.mean()) if len(truth) else np.nan,
                **classification,
                **ranked,
            }
        ]
    )


def _binary_group_row(part: pd.DataFrame, grouping: str, value: Any, top5_column: str) -> dict[str, Any]:
    truth = part["ground_truth_label"].to_numpy(bool)
    selected = part["threshold_flag"].to_numpy(bool)
    top5 = part[top5_column].to_numpy(bool)
    tp = int(np.sum(truth & selected))
    fp = int(np.sum(~truth & selected))
    positives = int(truth.sum())
    return {
        "grouping": grouping,
        "value": str(value),
        "observations": len(part),
        "positive_support": positives,
        "selected_at_threshold": int(selected.sum()),
        "captured_at_threshold": tp,
        "false_positives_at_threshold": fp,
        "precision_at_threshold": tp / max(tp + fp, 1),
        "recall_at_threshold": tp / positives if positives else np.nan,
        "detection_rate_at_threshold": tp / positives if positives else np.nan,
        "captured_at_top5pct": int(np.sum(truth & top5)),
        "recall_at_top5pct": float(np.sum(truth & top5) / positives) if positives else np.nan,
        "detection_rate_at_top5pct": float(np.sum(truth & top5) / positives) if positives else np.nan,
    }


def _group_metrics(
    test: pd.DataFrame,
    group_columns: Sequence[str],
    top5_flags: np.ndarray,
    expected_truth_values: Mapping[str, Sequence[str]] | None = None,
    overall_truth_support: Mapping[tuple[str, str], int] | None = None,
) -> pd.DataFrame:
    work = test.copy()
    top5_column = "__top5_flag"
    work[top5_column] = top5_flags
    rows: list[dict[str, Any]] = []
    for column in ("anomaly_type", "anomaly_category", "severity"):
        if column not in work:
            continue
        observed_values = {
            item
            for value in work.loc[work.ground_truth_label, column].fillna("none").astype(str)
            for item in value.split("|")
            if item and item.lower() not in {"none", "nan"}
        }
        values = sorted(
            observed_values
            | set((expected_truth_values or {}).get(column, []))
        )
        for value in values:
            mask = work.ground_truth_label & work[column].fillna("none").astype(str).str.split("|").map(lambda items: value in items)
            part = work.loc[mask]
            row = _binary_group_row(part, column, value, top5_column)
            row["false_positives_at_threshold"] = np.nan
            row["precision_at_threshold"] = np.nan
            row["group_kind"] = "ground_truth"
            row["overall_truth_support"] = int(
                (overall_truth_support or {}).get((column, value), len(part))
            )
            row["support_status"] = (
                "evaluated_on_final_test" if len(part) else "no_final_test_support"
            )
            row["evaluation_scope"] = "commercial PCA final holdout"
            rows.append(row)
    available_groups = [column for column in group_columns if column in work]
    for column in available_groups:
        for value, part in work.groupby(column, dropna=False, observed=True, sort=True):
            row = _binary_group_row(part, column, value, top5_column)
            row["group_kind"] = "organization"
            row["overall_truth_support"] = int(row["positive_support"])
            row["support_status"] = "evaluated_on_final_test"
            row["evaluation_scope"] = "commercial PCA final holdout"
            rows.append(row)
    if all(column in work for column in DEFAULT_GROUP_COLUMNS):
        combined = list(DEFAULT_GROUP_COLUMNS)
        for values, part in work.groupby(combined, dropna=False, observed=True, sort=True):
            if not isinstance(values, tuple):
                values = (values,)
            label = " / ".join(str(value) for value in values)
            row = _binary_group_row(part, "manager_team_territory", label, top5_column)
            row["group_kind"] = "organization"
            row["overall_truth_support"] = int(row["positive_support"])
            row["support_status"] = "evaluated_on_final_test"
            row["evaluation_scope"] = "commercial PCA final holdout"
            rows.append(row)
    columns = [
        "group_kind",
        "grouping",
        "value",
        "observations",
        "positive_support",
        "overall_truth_support",
        "selected_at_threshold",
        "captured_at_threshold",
        "false_positives_at_threshold",
        "precision_at_threshold",
        "recall_at_threshold",
        "detection_rate_at_threshold",
        "captured_at_top5pct",
        "recall_at_top5pct",
        "detection_rate_at_top5pct",
        "support_status",
        "evaluation_scope",
    ]
    return pd.DataFrame(rows, columns=columns)


def _period_stability(
    clean: pd.DataFrame,
    injected: pd.DataFrame,
    period_column: str,
    entity_column: str | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for population, frame in (("clean", clean), ("injected", injected)):
        previous_selected: set[str] | None = None
        for period, part in frame.sort_values(period_column).groupby(period_column, observed=True, sort=True):
            selected_entities: set[str] = set()
            if entity_column and entity_column in part:
                selected_entities = set(part.loc[part.threshold_flag, entity_column].dropna().astype(str))
            overlap = np.nan
            if previous_selected is not None:
                overlap = len(previous_selected & selected_entities) / max(len(previous_selected), 1)
            row = {
                "population": population,
                "period": period,
                "observations": len(part),
                "mean_raw_score": float(part.raw_score.mean()),
                "median_raw_score": float(part.raw_score.median()),
                "std_raw_score": float(part.raw_score.std(ddof=0)),
                "mean_anomaly_score": float(part.anomaly_score.mean()),
                "p95_anomaly_score": float(part.anomaly_score.quantile(0.95)),
                "review_rate": float(part.threshold_flag.mean()),
                "selected_entity_overlap_previous_period": overlap,
                "ground_truth_prevalence": np.nan,
                "precision_at_threshold": np.nan,
                "recall_at_threshold": np.nan,
            }
            if population == "injected" and "ground_truth_label" in part:
                truth = part.ground_truth_label.to_numpy(bool)
                selected = part.threshold_flag.to_numpy(bool)
                tp = int(np.sum(truth & selected))
                row["ground_truth_prevalence"] = float(truth.mean())
                row["precision_at_threshold"] = tp / max(int(selected.sum()), 1)
                row["recall_at_threshold"] = tp / int(truth.sum()) if truth.any() else np.nan
            rows.append(row)
            previous_selected = selected_entities
    result = pd.DataFrame(rows).sort_values(["population", "period"], kind="stable").reset_index(drop=True)
    result["mean_score_change_from_previous_period"] = result.groupby("population", observed=True).mean_anomaly_score.diff()
    return result


def _score_distributions(
    clean: pd.DataFrame,
    injected: pd.DataFrame,
    bins: int,
) -> pd.DataFrame:
    if bins < 2:
        raise ValueError("distribution_bins must be at least 2.")
    populations: list[tuple[str, pd.DataFrame]] = [("clean", clean), ("injected_all", injected)]
    if "ground_truth_label" in injected:
        populations.extend(
            [
                ("injected_normal", injected.loc[~injected.ground_truth_label]),
                ("injected_anomaly", injected.loc[injected.ground_truth_label]),
            ]
        )
    rows: list[dict[str, Any]] = []
    for population, frame in populations:
        for split, part in frame.groupby("split", observed=True, sort=True):
            indices = np.minimum((part.anomaly_score.to_numpy(float) * bins).astype(int), bins - 1)
            for index in range(bins):
                selected = part.iloc[np.flatnonzero(indices == index)]
                rows.append(
                    {
                        "population": population,
                        "split": split,
                        "bin": index + 1,
                        "score_lower": index / bins,
                        "score_upper": (index + 1) / bins,
                        "count": len(selected),
                        "share": len(selected) / len(part) if len(part) else 0.0,
                        "mean_raw_score": float(selected.raw_score.mean()) if len(selected) else np.nan,
                        "mean_anomaly_score": float(selected.anomaly_score.mean()) if len(selected) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _curve_tables(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = test.ground_truth_label.to_numpy(bool)
    scores = test.raw_score.to_numpy(float)
    if len(np.unique(truth)) == 2:
        false_positive_rate, true_positive_rate, roc_thresholds = sklearn_roc_curve(truth, scores)
        roc_thresholds = np.where(np.isfinite(roc_thresholds), roc_thresholds, np.nan)
        roc = pd.DataFrame(
            {
                "false_positive_rate": false_positive_rate,
                "true_positive_rate": true_positive_rate,
                "threshold": roc_thresholds,
            }
        )
        precision, recall, pr_thresholds = sklearn_precision_recall_curve(truth, scores)
        padded = np.append(pr_thresholds, np.nan)
        pr = pd.DataFrame({"precision": precision, "recall": recall, "threshold": padded})
    else:
        roc = pd.DataFrame(columns=["false_positive_rate", "true_positive_rate", "threshold"])
        pr = pd.DataFrame(columns=["precision", "recall", "threshold"])
    order = np.lexsort((test.observation_id.astype(str).to_numpy(), -scores))
    ranked_truth = truth[order]
    cumulative = np.cumsum(ranked_truth)
    rank = np.arange(1, len(test) + 1)
    prevalence = float(truth.mean()) if len(truth) else np.nan
    precision = cumulative / rank if len(rank) else np.array([])
    lift = precision / prevalence if prevalence > 0 else np.full(len(rank), np.nan)
    lift_frame = pd.DataFrame(
        {
            "review_rank": rank,
            "review_fraction": rank / len(test) if len(test) else np.array([]),
            "threshold": scores[order] if len(test) else np.array([]),
            "captured_count": cumulative,
            "precision": precision,
            "recall": cumulative / int(truth.sum()) if truth.any() else np.full(len(rank), np.nan),
            "lift": lift,
        }
    )
    return roc, pr, lift_frame


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_results(output_dir: str | Path, results: Mapping[str, Any]) -> None:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for key, filename in OUTPUT_FILES.items():
        table = results[key]
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"Result {key} must be a DataFrame before persistence.")
        table.to_csv(destination / filename, index=False, lineterminator="\n")
    (destination / "pca_metadata.json").write_text(
        json.dumps(results["pca_metadata"], indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def run_finalized_pca(
    clean_features: pd.DataFrame,
    injected_features: pd.DataFrame,
    ground_truth: pd.DataFrame,
    config: Mapping[str, Any] | str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fit and evaluate the finalized PCA reconstruction architecture.

    Parameters
    ----------
    clean_features:
        Clean rep-period feature store.  Only rows through ``train_end`` are used
        to fit preprocessing, PCA, and score calibration.
    injected_features:
        Separate controlled evaluation copy at the same grain.
    ground_truth:
        Separate labels/evidence keyed by ``observation_id`` or ``rep_id, period``.
        Its columns are categorically excluded from the model feature allowlist.
    config:
        Mapping (or YAML path).  Settings may live at the root or under ``model``.
        Important fields are ``train_end``, ``validation_end``,
        ``manager_review_fraction``, ``pca_retained_variance`` and optionally
        ``feature_columns`` / ``id_columns``.
    output_dir:
        No files are written when omitted.  When supplied, only static result
        filenames inside this directory are created.

    Returns
    -------
    dict
        Scored clean/injected frames, evaluation tables, top PCA contributions
        with deterministic manager reasons, and PCA/preprocessing metadata.
    """
    if not isinstance(clean_features, pd.DataFrame) or not isinstance(injected_features, pd.DataFrame):
        raise TypeError("clean_features and injected_features must be pandas DataFrames.")
    if not isinstance(ground_truth, pd.DataFrame):
        raise TypeError("ground_truth must be a pandas DataFrame.")
    if clean_features.empty or injected_features.empty or ground_truth.empty:
        raise ValueError("Clean features, injected features, and ground truth must be non-empty.")

    root, model_cfg = _model_settings(_load_config(config))
    finalized = str(_setting(root, model_cfg, ("finalized_model", "model_name"), MODEL_NAME))
    if finalized.casefold() != MODEL_NAME.casefold():
        raise ValueError(f"This runner preserves the finalized {MODEL_NAME} architecture; received {finalized!r}.")
    period_column = _period_column(clean_features, injected_features, root, model_cfg)
    clean = _normalise_period(clean_features, period_column, "clean_features")
    injected = _normalise_period(injected_features, period_column, "injected_features")
    truth_for_keys = _normalise_period(ground_truth, period_column, "ground_truth") if period_column in ground_truth else ground_truth.copy(deep=True)
    keys = _join_keys(clean, injected, truth_for_keys, period_column, root, model_cfg)
    _ensure_unique(clean, keys, "clean_features")
    _ensure_unique(injected, keys, "injected_features")
    clean_keys = pd.MultiIndex.from_frame(clean[list(keys)])
    injected_keys = pd.MultiIndex.from_frame(injected[list(keys)])
    if len(clean_keys.difference(injected_keys)) or len(injected_keys.difference(clean_keys)):
        raise ValueError("Clean and injected feature stores must contain the same rep-period keys.")
    embedded_truth = (set(clean.columns) | set(injected.columns)) & KNOWN_TRUTH_COLUMNS
    if embedded_truth:
        raise ValueError(
            "Ground-truth leakage: label/evaluation columns must remain in the separate ground_truth table: "
            f"{sorted(embedded_truth)}"
        )
    features = _feature_allowlist(clean, injected, truth_for_keys, keys, period_column, root, model_cfg)

    train_end = pd.Timestamp(_setting(root, model_cfg, ("train_end",)))
    validation_end = pd.Timestamp(_setting(root, model_cfg, ("validation_end",)))
    if pd.isna(train_end) or pd.isna(validation_end) or validation_end <= train_end:
        raise ValueError("Config requires validation_end later than train_end.")
    clean_split = _split_labels(clean[period_column], train_end, validation_end)
    injected_split = _split_labels(injected[period_column], train_end, validation_end)
    train_mask = clean_split.eq("train").to_numpy()
    validation_mask = injected_split.eq("validation").to_numpy()
    test_mask = injected_split.eq("test").to_numpy()
    if min(int(train_mask.sum()), int(validation_mask.sum()), int(test_mask.sum())) == 0:
        raise ValueError("Chronological train, validation, and test partitions must all be non-empty.")
    all_missing_train = [feature for feature in features if clean.loc[train_mask, feature].notna().sum() == 0]
    if all_missing_train:
        raise ValueError(f"Features entirely missing in clean training data: {all_missing_train}")

    manager_fraction = float(_setting(root, model_cfg, ("manager_review_fraction", "review_fraction"), 0.05))
    retained_variance = _setting(root, model_cfg, ("pca_retained_variance", "n_components", "pca_variance"), 0.95)
    if isinstance(retained_variance, Sequence) and not isinstance(retained_variance, (str, bytes)):
        retained_variance = retained_variance[-1]
    retained_variance = float(retained_variance) if not isinstance(retained_variance, int) else int(retained_variance)
    preprocessing = {
        "features": features,
        "scaler": str(_setting(root, model_cfg, ("scaler",), "robust")),
        "signed_log1p": bool(_setting(root, model_cfg, ("signed_log1p",), True)),
        "clip_outliers": bool(_setting(root, model_cfg, ("preprocessing_clip", "clip_outliers"), False)),
        "clip_lower_quantile": float(_setting(root, model_cfg, ("clip_lower_quantile",), 0.01)),
        "clip_upper_quantile": float(_setting(root, model_cfg, ("clip_upper_quantile",), 0.99)),
    }
    fitted_preprocessor, clean_train_matrix = fit_preprocessor(clean.loc[train_mask], preprocessing)
    clean_matrix = fitted_preprocessor.transform(clean)
    injected_matrix = fitted_preprocessor.transform(injected)
    pca = PCAReconstruction(
        n_components=retained_variance,
        random_state=int(_setting(root, model_cfg, ("seed", "random_state"), root.get("project", {}).get("seed", 42) if isinstance(root.get("project"), Mapping) else 42)),
    ).fit(clean_train_matrix)
    explained = np.asarray(pca.model.explained_variance_ratio_, dtype=float)
    if not np.isfinite(explained).all() or explained.sum() <= 0:
        raise ValueError("Clean training features produce a degenerate PCA variance space.")
    clean_raw = pca.raw_score(clean_matrix)
    injected_raw = pca.raw_score(injected_matrix)
    calibrator = PercentileCalibrator().fit(clean_raw[train_mask])

    # The threshold is deliberately frozen before label values are prepared or joined.
    raw_threshold, validation_target_count = _raw_review_threshold(injected_raw[validation_mask], manager_fraction)

    clean_scored = _score_population(
        clean,
        clean_matrix,
        pca,
        calibrator,
        raw_threshold,
        manager_fraction,
        clean_split,
        _observation_ids(clean, keys),
    )
    injected_scored = _score_population(
        injected,
        injected_matrix,
        pca,
        calibrator,
        raw_threshold,
        manager_fraction,
        injected_split,
        _observation_ids(injected, keys),
    )
    truth, truth_columns = _prepare_truth(truth_for_keys, keys, period_column, root, model_cfg)
    truth_payload = [column for column in truth.columns if column not in keys and column not in injected_scored]
    injected_scored = injected_scored.merge(
        truth[list(keys) + truth_payload],
        on=list(keys),
        how="left",
        validate="one_to_one",
        sort=False,
    )
    injected_scored["ground_truth_label"] = _as_bool(injected_scored.get("ground_truth_label", pd.Series(False, index=injected_scored.index)))
    for column in ("anomaly_type", "anomaly_category", "severity"):
        if column not in injected_scored:
            injected_scored[column] = "none"
        injected_scored[column] = injected_scored[column].fillna("none").astype(str)

    top_count = int(_setting(root, model_cfg, ("top_contribution_count", "top_driver_count"), 3))
    clean_scored, clean_contributions = _add_explanations(
        clean,
        clean_scored,
        clean_matrix,
        pca,
        features,
        period_column,
        "clean",
        top_count,
        keys,
    )
    injected_scored, injected_contributions = _add_explanations(
        injected,
        injected_scored,
        injected_matrix,
        pca,
        features,
        period_column,
        "injected",
        top_count,
        keys,
    )
    feature_contributions = pd.concat([clean_contributions, injected_contributions], ignore_index=True)

    test = injected_scored.loc[injected_scored.split.eq("test")].copy()
    top_k, top_flags = _top_k_metrics(test, DEFAULT_CUTOFFS)
    metrics_summary = _summary_metrics(
        test,
        raw_threshold,
        manager_fraction,
        int(validation_mask.sum()),
        validation_target_count,
    )
    configured_groups = _setting(root, model_cfg, ("group_columns",), list(DEFAULT_GROUP_COLUMNS))
    group_columns = [str(column) for column in configured_groups]
    expected_truth_values: dict[str, list[str]] = {}
    overall_truth_support: dict[tuple[str, str], int] = {}
    truth_positive = ground_truth.loc[
        _as_bool(
            ground_truth.get(
                truth_columns["label"],
                pd.Series(True, index=ground_truth.index),
            )
        )
    ].copy()
    for canonical, source_column in (
        ("anomaly_type", truth_columns["anomaly_type"]),
        ("anomaly_category", truth_columns["anomaly_category"]),
        ("severity", truth_columns["severity"]),
    ):
        if source_column is None or source_column not in truth_positive:
            continue
        values = sorted(
            {
                item
                for raw in truth_positive[source_column].dropna().astype(str)
                for item in raw.split("|")
                if item and item.lower() not in {"none", "nan"}
            }
        )
        expected_truth_values[canonical] = values
        for value in values:
            mask = truth_positive[source_column].astype(str).str.split("|").map(
                lambda items: value in items
            )
            overall_truth_support[(canonical, value)] = int(
                truth_positive.loc[mask, list(keys)].drop_duplicates().shape[0]
            )
    group_metrics = _group_metrics(
        test,
        group_columns,
        top_flags[0.05],
        expected_truth_values,
        overall_truth_support,
    )
    entity_column = str(_setting(root, model_cfg, ("entity_column",), "rep_id"))
    period_stability = _period_stability(clean_scored, injected_scored, period_column, entity_column)
    distribution_bins = int(_setting(root, model_cfg, ("distribution_bins",), 20))
    score_distributions = _score_distributions(clean_scored, injected_scored, distribution_bins)
    roc, pr, lift = _curve_tables(test)
    false_positives = test.loc[
        test["threshold_flag"].astype(bool)
        & ~test["ground_truth_label"].astype(bool)
    ].copy()
    if false_positives.empty:
        false_positive_review = pd.DataFrame(
            columns=[
                "driver_feature",
                "driver_name",
                "false_positive_count",
                "mean_raw_score",
                "median_anomaly_score",
                "example_observation_ids",
                "review_interpretation",
            ]
        )
    else:
        false_positive_review = (
            false_positives.groupby(
                ["driver_1_feature", "driver_1_name"], observed=True, dropna=False
            )
            .agg(
                false_positive_count=("observation_id", "size"),
                mean_raw_score=("raw_score", "mean"),
                median_anomaly_score=("anomaly_score", "median"),
                example_observation_ids=(
                    "observation_id",
                    lambda values: "|".join(values.astype(str).head(5)),
                ),
            )
            .reset_index()
            .rename(
                columns={
                    "driver_1_feature": "driver_feature",
                    "driver_1_name": "driver_name",
                }
            )
            .sort_values(
                ["false_positive_count", "mean_raw_score"],
                ascending=[False, False],
                kind="mergesort",
            )
        )
        false_positive_review["review_interpretation"] = (
            "Clean-labeled review candidate; validate the driver and source context, "
            "not evidence of misconduct."
        )
    cumulative = np.cumsum(explained)
    pca_metadata = {
        "model_name": MODEL_NAME,
        "finalized_model": True,
        "scoring_grain": list(keys),
        "period_column": period_column,
        "feature_count": len(features),
        "feature_columns": list(features),
        "ground_truth_columns_excluded": sorted(set(truth_for_keys.columns) | KNOWN_TRUTH_COLUMNS),
        "ground_truth_source_columns": truth_columns,
        "train_end": train_end.isoformat(),
        "validation_end": validation_end.isoformat(),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "fit_population": "clean train only",
        "threshold": raw_threshold,
        "threshold_basis": "kth highest raw PCA score in injected validation at configured manager review fraction; labels unused",
        "manager_review_fraction": manager_fraction,
        "validation_target_review_count": validation_target_count,
        "validation_threshold_flag_count": int(np.sum(injected_raw[validation_mask] >= raw_threshold)),
        "raw_score_definition": "mean per-feature squared reconstruction error in fitted transformed space",
        "anomaly_score_definition": "clean-TRAIN empirical CDF percentile with bounded monotone tails (0-1)",
        "contribution_definition": "unnormalized per-feature squared reconstruction error in fitted transformed space",
        "n_components_parameter": retained_variance,
        "retained_components": int(pca.model.n_components_),
        "explained_variance_ratio": explained.tolist(),
        "cumulative_explained_variance": cumulative.tolist(),
        "total_explained_variance": float(cumulative[-1]),
        "preprocessing": {key: value for key, value in preprocessing.items() if key != "features"},
        "responsible_use": "Review signal only; PCA contributions are not causal explanations and do not establish misconduct.",
    }
    results: dict[str, Any] = {
        "clean_scores": clean_scored.reset_index(drop=True),
        "injected_scores": injected_scored.reset_index(drop=True),
        "metrics_summary": metrics_summary,
        "top_k_metrics": top_k,
        "group_metrics": group_metrics,
        "period_stability": period_stability,
        "score_distributions": score_distributions,
        "roc_curve": roc,
        "pr_curve": pr,
        "lift_curve": lift,
        "feature_contributions": feature_contributions,
        "false_positive_review": false_positive_review,
        "pca_metadata": pca_metadata,
    }
    if output_dir is not None:
        _write_results(output_dir, results)
    return results


__all__ = ["run_finalized_pca"]
