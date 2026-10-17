"""One user-authorized joint background/threshold refinement of its own result."""

from run_workflow_feedback_refinement import main

JOINT_FEEDBACK = (
    "Go ahead: fix the recorder, then let the model tune background correction "
    "and threshold together, keeping the mesh ends closed and preserving the "
    "reconstructed middle. Rerun and show me the comparison."
)


if __name__ == "__main__":
    raise SystemExit(
        main(feedback=JOINT_FEEDBACK, require_own_baseline=True, entry_source=__file__)
    )
