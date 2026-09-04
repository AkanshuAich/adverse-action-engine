"""Model persistence.

Nothing here writes or reads a pickle. That is a security requirement, not a
style preference: unpickling executes arbitrary code embedded in the file, so a
pickled model is a remote code execution vector wearing a lab coat. Bank
security teams block the format outright, and a model artifact is exactly the
kind of file that gets copied between environments by people who did not create
it.

A saved model is a directory of three pure-data files:

* ``booster.ubj`` - XGBoost's native UBJSON. A data format, not a code format,
  and the only one that preserves native categorical splits.
* ``calibrator.json`` - the isotonic knot table.
* ``metadata.json`` - feature order, version, metrics, and provenance.

ONNX was the original intent and was abandoned on evidence: its XGBoost
converter requires generic ``f0``-style feature names and cannot represent
native categorical splits. Serving is Python-only here, so ONNX's portability
buys nothing, while degrading the model to fit the serialiser would have cost
the category names that denial reasons are written from.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from xgboost import XGBClassifier

from aae.domain.errors import ModelError
from aae.logging import get_logger
from aae.ml.calibration import Calibrator
from aae.ml.features import DERIVED_FEATURES, FeatureSpec
from aae.ml.train import ModelMetrics, TrainedModel

logger = get_logger(__name__)

BOOSTER_FILENAME: Final[str] = "booster.ubj"
CALIBRATOR_FILENAME: Final[str] = "calibrator.json"
METADATA_FILENAME: Final[str] = "metadata.json"

ARTIFACT_SCHEMA_VERSION: Final[int] = 1


def save_model(model: TrainedModel, directory: Path) -> Path:
    """Write a trained model as pure data.

    Args:
        model: The trained, calibrated model.
        directory: Destination directory, created if absent.

    Returns:
        The directory written to.
    """
    directory.mkdir(parents=True, exist_ok=True)

    model.booster.save_model(str(directory / BOOSTER_FILENAME))

    (directory / CALIBRATOR_FILENAME).write_text(
        json.dumps(model.calibrator.to_dict(), indent=2), encoding="utf-8"
    )

    metadata: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": model.model_version,
        "trained_at": model.trained_at.isoformat(),
        "data_source": model.data_source,
        "feature_spec": {
            "numeric": list(model.spec.numeric),
            "categorical": list(model.spec.categorical),
            # Derived features are code, versioned by git, not data. Only their
            # names are stored, so a mismatch on load is detected rather than
            # silently reconstructing different arithmetic.
            "derived_names": list(model.spec.derived_names),
        },
        "feature_order": list(model.spec.feature_names),
        "metrics": {
            "auc": model.metrics.auc,
            "ks": model.metrics.ks,
            "brier_uncalibrated": model.metrics.brier_uncalibrated,
            "brier_calibrated": model.metrics.brier_calibrated,
            "ece_uncalibrated": model.metrics.ece_uncalibrated,
            "ece_calibrated": model.metrics.ece_calibrated,
            "n_train": model.metrics.n_train,
            "n_calibrate": model.metrics.n_calibrate,
            "n_test": model.metrics.n_test,
            "positive_rate": model.metrics.positive_rate,
            "best_iteration": model.metrics.best_iteration,
            "calibration_applied": model.metrics.calibration_applied,
        },
    }
    (directory / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("model_saved", directory=str(directory), version=model.model_version)
    return directory


def load_model(directory: Path) -> TrainedModel:
    """Load a model saved by :func:`save_model`.

    Args:
        directory: Directory containing the three artifact files.

    Returns:
        The reconstructed model.

    Raises:
        ModelError: If files are missing, the artifact schema is unknown, or
            the stored feature specification disagrees with the current code.
    """
    for filename in (BOOSTER_FILENAME, CALIBRATOR_FILENAME, METADATA_FILENAME):
        if not (directory / filename).is_file():
            msg = f"Model artifact {filename!r} is missing from {directory}."
            raise ModelError(msg)

    metadata = json.loads((directory / METADATA_FILENAME).read_text(encoding="utf-8"))

    stored_schema = metadata.get("artifact_schema_version")
    if stored_schema != ARTIFACT_SCHEMA_VERSION:
        msg = (
            f"Artifact schema version {stored_schema} is not supported "
            f"(expected {ARTIFACT_SCHEMA_VERSION})."
        )
        raise ModelError(msg)

    spec_payload = metadata["feature_spec"]
    stored_derived = tuple(spec_payload["derived_names"])
    current_derived = tuple(d.name for d in DERIVED_FEATURES)
    if stored_derived != current_derived:
        msg = (
            "Derived feature set has changed since this model was trained "
            f"(stored {stored_derived}, current {current_derived}). Retrain "
            "rather than scoring with mismatched feature arithmetic."
        )
        raise ModelError(msg)

    # Reconstructing through FeatureSpec re-runs the fair-lending validation, so
    # a model saved before a protected attribute was added to the list cannot be
    # loaded and quietly used.
    spec = FeatureSpec(
        numeric=tuple(spec_payload["numeric"]),
        categorical=tuple(spec_payload["categorical"]),
        derived=DERIVED_FEATURES,
    )

    if list(spec.feature_names) != list(metadata["feature_order"]):
        msg = "Reconstructed feature order does not match the stored order."
        raise ModelError(msg)

    booster = XGBClassifier()
    booster.load_model(str(directory / BOOSTER_FILENAME))

    calibrator = Calibrator.from_dict(
        json.loads((directory / CALIBRATOR_FILENAME).read_text(encoding="utf-8"))
    )

    metrics = ModelMetrics(**metadata["metrics"])

    logger.info("model_loaded", directory=str(directory), version=metadata["model_version"])
    return TrainedModel(
        booster=booster,
        calibrator=calibrator,
        spec=spec,
        metrics=metrics,
        model_version=metadata["model_version"],
        trained_at=datetime.fromisoformat(metadata["trained_at"]).astimezone(UTC),
        data_source=metadata["data_source"],
    )
