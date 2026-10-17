"""One explicitly authorized refinement with a product-owned protected region.

The structure and region selectors identify the model's own accepted baseline
outputs. They do not supply an algorithm, threshold, hand-drawn region or oracle.
The ordinary product preservation controller must freeze and check the overlap.
"""

from run_workflow_feedback_refinement import main

PROTECTED_GOAL = (
    "In this z-stack, segment two separate structures: the wavy band and the bright "
    "circular object in the middle. Produce a complete 3D mask and a separate, "
    "closed-ended mesh for each, then overlay the meshes in different colours. "
    "Where the bright object obscures the band, reconstruct the band's continuous "
    "path through the obscured centre using the visible band as evidence; do not "
    "substitute a ring or detour around the circular object. It is acceptable for "
    "the two independently represented masks to overlap there. Treat reconstructed "
    "geometry as an inference, not observed anatomy, and report the assumption. "
    "For this refinement, preserve the accepted band foreground in the protected "
    "central region while jointly adjusting background correction and threshold "
    "to reduce background-related width variation elsewhere, without forcing "
    "genuinely varying parts to uniform width. Choose the parameters yourself from "
    "the supplied input and output evidence. Keep the mesh ends closed at the "
    "acquired image borders."
)

PROTECTED_FEEDBACK = (
    "Tune background correction and threshold together. Preserve the accepted "
    "central band reconstruction in the selected protected region, and keep the "
    "mesh ends closed. Use the input and output previews to compare the result; "
    "report any trade-offs or unmet requirements."
)

PROTECTED_SELECTION = {
    "mask_model_key": "refine_reference_labels",
    "mask_port": 0,
    "region_model_key": "refine_circle_labels",
    "region_port": 0,
    "policy": "preserve_all_selected_positive_foreground",
    "proposal_key": "protected_structure",
}


if __name__ == "__main__":
    raise SystemExit(
        main(
            feedback=PROTECTED_FEEDBACK,
            objective=PROTECTED_GOAL,
            require_own_baseline=True,
            preservation=PROTECTED_SELECTION,
            entry_source=__file__,
        )
    )
