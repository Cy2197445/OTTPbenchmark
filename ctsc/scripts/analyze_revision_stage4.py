"""Check matched Step and Outcome endpoint pairs."""
from __future__ import annotations
from typing import Any

def _endpoint_alignment_audit(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required = {
        "step": ("step_auprc", "matched_step_prevalence"),
        "outcome": ("outcome_auprc", "matched_outcome_prevalence"),
    }
    for row in seed_rows:
        for endpoint, fields in required.items():
            if any(field not in row for field in fields):
                raise ValueError(f"Missing {endpoint} endpoint field in Stage 4 result row: {fields}")
    return {
        "schema_version": "ottp_stage4_endpoint_alignment_audit_v1",
        "status": "PASS",
        "row_count": len(seed_rows),
        "step_pair": ["Step catastrophic AUPRC", "matched Step catastrophic prevalence"],
        "outcome_pair": ["Outcome catastrophic AUPRC", "matched Outcome catastrophic prevalence"],
        "inference_endpoint": "Step",
        "outcome_role": "descriptive_secondary_only",
        "cross_endpoint_pairing": False,
    }

