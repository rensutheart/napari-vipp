from __future__ import annotations

from dataclasses import replace

import pytest

from napari_vipp.core.compute_policy import validate_spec_policy_references
from napari_vipp.core.compute_specs import compute_specs_for


def test_synthesized_cpu_spec_uses_registered_policies():
    validate_spec_policy_references(compute_specs_for("gaussian_blur")[0])


def test_unknown_policy_reference_fails_declaration_validation():
    cpu_spec = compute_specs_for("median_filter")[0]

    with pytest.raises(ValueError, match="unknown parity policy"):
        validate_spec_policy_references(
            replace(cpu_spec, parity_policy_id="missing-policy-v1")
        )
