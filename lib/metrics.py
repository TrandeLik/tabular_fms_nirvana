import warnings
from typing import Any, Literal, cast, overload

import numpy as np
import scipy.special
import sklearn.metrics
from sklearn.exceptions import UndefinedMetricWarning

from .util import PredictionType, TaskType

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def _get_labels_and_probs(
    prediction: np.ndarray,
    task_type: TaskType,
    prediction_type: PredictionType,
) -> tuple[np.ndarray, None | np.ndarray]:
    """Obtain labels and probabilities from raw predictions."""
    assert task_type in (TaskType.BINCLASS, TaskType.MULTICLASS)

    if prediction_type == PredictionType.LABELS:
        return prediction, None
    elif prediction_type == PredictionType.PROBS:
        probs = prediction
    elif prediction_type == PredictionType.LOGITS:
        probs = (
            scipy.special.expit(prediction)
            if task_type == TaskType.BINCLASS
            else scipy.special.softmax(prediction, axis=1)
        )
    else:
        raise ValueError(f'Unknown prediction type: {prediction_type}')

    assert probs is not None
    labels = np.round(probs) if task_type == TaskType.BINCLASS else probs.argmax(axis=1)
    return labels.astype(np.int64), probs


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str | TaskType,
    prediction_type: str | PredictionType,
) -> dict[str, Any]:
    task_type = TaskType(task_type)
    prediction_type = PredictionType(prediction_type)

    if task_type == TaskType.REGRESSION:
        assert prediction_type == PredictionType.LABELS
        result = {
            'rmse': float(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5),
            'mae': float(sklearn.metrics.mean_absolute_error(y_true, y_pred)),
            'r2': float(sklearn.metrics.r2_score(y_true, y_pred)),
        }

    else:
        assert prediction_type is not None
        labels, probs = _get_labels_and_probs(y_pred, task_type, prediction_type)
        result = cast(
            dict[str, Any],
            sklearn.metrics.classification_report(y_true, labels, output_dict=True),
        )
        if probs is not None:
            result['cross-entropy'] = float(sklearn.metrics.log_loss(y_true, probs))
        if task_type == TaskType.BINCLASS and probs is not None:
            result['roc-auc'] = float(sklearn.metrics.roc_auc_score(y_true, probs))
            result['ap'] = float(sklearn.metrics.average_precision_score(y_true, probs))

    return result


@overload
def composite_score(
    predictions: np.ndarray,
    baseline_labels: np.ndarray,
    assessor_labels: np.ndarray,
    threshold: float | None = ...,
    return_curve: Literal[False] = ...,
) -> tuple[float, float | None]: ...
@overload
def composite_score(
    predictions: np.ndarray,
    baseline_labels: np.ndarray,
    assessor_labels: np.ndarray,
    threshold: float | None = ...,
    *,
    return_curve: Literal[True],
) -> tuple[float, float | None, list[dict[str, float]]]: ...


def composite_score(
    predictions: np.ndarray,
    baseline_labels: np.ndarray,
    assessor_labels: np.ndarray,
    threshold: float | None = None,
    return_curve: bool = False,
) -> tuple[float, float | None] | tuple[float, float | None, list[dict[str, float]]]:
    """Composite metric measuring NEW fraud detection beyond a known baseline.

    A predicted node is counted as "new fraud" when the model flags it as fraud
    AND it is not already flagged in ``baseline_labels``. The precision of those
    new-fraud predictions is then checked against ``assessor_labels`` (external
    ground truth). The score is the largest new-fraud ratio across thresholds
    whose precision against assessors is above 0.9; if no threshold reaches
    that precision, the score is ``best_precision - 0.9`` (negative).

    Args:
        predictions: model predictions in [0, 1] for the selected nodes.
        baseline_labels: reference labels of already-known fraud for the same
            nodes (may contain NaN). A prediction equal to 1 here is treated
            as not-new.
        assessor_labels: externally verified ground-truth labels for the same
            nodes (NaN means not assessed; only assessed nodes contribute to
            precision).
        threshold: if given, only this threshold is evaluated; otherwise the
            range [0, 1] is swept.
        return_curve: if True, also return per-threshold metrics as a third
            element.

    Returns:
        A scalar score (higher = better) and selected threshold. If
        ``return_curve`` is True, also returns the per-threshold curve as a
        third element.
    """
    assert ((predictions <= 1) & (predictions >= 0)).all()
    if threshold is None:
        thresholds = np.linspace(
            0,
            1,
            21,
            endpoint=True,
        )
    else:
        thresholds = [threshold]

    assessor_mask = ~np.isnan(assessor_labels)
    assessed_labels = assessor_labels[assessor_mask]

    max_new_fraud_ratio = 0
    best_precision = -np.inf
    best_precision_threshold: float | None = None
    curve: list[dict[str, float]] = []
    for t in thresholds:
        fraud_pred = (predictions >= t).astype(int)
        new_fraud_pred = ((fraud_pred == 1) & (baseline_labels != 1)).astype(int)
        new_fraud_ratio = new_fraud_pred.sum() / new_fraud_pred.shape[0]

        # Subset of new-fraud predictions for nodes that have assessor markup.
        assessed_new_fraud_pred = new_fraud_pred[assessor_mask]
        assessed_new_fraud_count = (assessed_new_fraud_pred == 1).sum()
        if assessed_new_fraud_count == 0:
            # No assessed predictions — treat as perfect precision.
            new_fraud_precision = 1
        else:
            new_fraud_precision = (
                (assessed_new_fraud_pred == 1) & (assessed_labels == 1)
            ).sum() / assessed_new_fraud_count
            # Track best non-vacuous precision for the sub-0.9 fallback.
            if new_fraud_precision > best_precision:
                best_precision = new_fraud_precision
                best_precision_threshold = t
        if new_fraud_precision > 0.9 and max_new_fraud_ratio < new_fraud_ratio:
            threshold = t
            max_new_fraud_ratio = new_fraud_ratio
        if return_curve:
            curve.append(
                {
                    'threshold': float(t),
                    'new_fraud_ratio': float(new_fraud_ratio),
                    'new_fraud_precision': float(new_fraud_precision),
                    'new_fraud_count': float(new_fraud_pred.sum()),
                    'assessed_new_fraud_count': float(assessed_new_fraud_count),
                }
            )

    if best_precision_threshold is not None and best_precision < 0.9:
        score, sel = float(best_precision - 0.9), best_precision_threshold
    else:
        score, sel = float(max_new_fraud_ratio), threshold

    if return_curve:
        return score, sel, curve
    return score, sel
