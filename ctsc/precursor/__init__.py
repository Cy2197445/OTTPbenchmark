'Public package surface for precursor utilities.'

from __future__ import annotations

__all__: list[str] = []

try:
    from ctsc.precursor.metrics import (
        binary_average_precision,
        binary_brier_score,
        binary_roc_auc,
        expected_calibration_error,
    )
except Exception:
    pass
else:
    __all__.extend(
        [
            "binary_average_precision",
            "binary_brier_score",
            "binary_roc_auc",
            "expected_calibration_error",
        ]
    )

try:
    from ctsc.precursor.model import (
        FlatMLPPrecursorModel,
        GRUPrecursorModel,
        LinearPrecursorModel,
        PrevalenceBaseline,
        TemporalCNNPrecursorModel,
        TemperatureScaler,
        build_precursor_model,
        get_precursor_model_spec,
    )
except Exception:
    pass
else:
    __all__.extend(
        [
            "build_precursor_model",
            "get_precursor_model_spec",
            "FlatMLPPrecursorModel",
            "GRUPrecursorModel",
            "LinearPrecursorModel",
            "PrevalenceBaseline",
            "TemperatureScaler",
            "TemporalCNNPrecursorModel",
        ]
    )

try:
    from ctsc.precursor.data import (
        FeatureMatrixSequenceDataset,
        FeatureMatrixWindowFlatDataset,
        PrecursorSequenceDataset,
        SPLIT_FAMILY_REGISTRY,
        build_group_splits,
        build_splits,
        build_transition_random_splits,
        build_feature_profile_matrix,
        build_precursor_dataset,
        compute_profile_normalization_stats,
        compute_label_thresholds,
        compute_normalization_stats,
        feature_profile_current_cost_index,
        load_precursor_dataset,
        save_precursor_dataset,
        sequence_view_for_profile,
        select_periodic_checkpoints,
        summarize_split_diagnostics,
    )
except Exception:
    pass
else:
    __all__.extend(
        [
            "FeatureMatrixSequenceDataset",
            "FeatureMatrixWindowFlatDataset",
            "PrecursorSequenceDataset",
            "SPLIT_FAMILY_REGISTRY",
            "build_feature_profile_matrix",
            "build_group_splits",
            "build_splits",
            "build_transition_random_splits",
            "build_precursor_dataset",
            "compute_label_thresholds",
            "compute_normalization_stats",
            "compute_profile_normalization_stats",
            "feature_profile_current_cost_index",
            "load_precursor_dataset",
            "save_precursor_dataset",
            "sequence_view_for_profile",
            "select_periodic_checkpoints",
            "summarize_split_diagnostics",
        ]
    )


try:
    from ctsc.precursor.public_offline import (
        PublicOfflineProtocolConfig,
        audit_public_task_protocol,
        build_public_precursor_dataset,
        load_public_offline_raw_dataset,
        summarize_public_precursor_dataset,
    )
except Exception:
    pass
else:
    __all__.extend(
        [
            "PublicOfflineProtocolConfig",
            "audit_public_task_protocol",
            "build_public_precursor_dataset",
            "load_public_offline_raw_dataset",
            "summarize_public_precursor_dataset",
        ]
    )
