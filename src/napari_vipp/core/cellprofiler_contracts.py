"""Provider-free identities for the explicit 2D CellProfiler workflows."""

CELLPROFILER_COMPARTMENT_OPERATION_IDS = frozenset(
    {
        "cellprofiler_smooth",
        "cellprofiler_threshold",
        "cellprofiler_primary_objects",
        "cellprofiler_propagation_seeds",
        "cellprofiler_finish_cells",
        "cellprofiler_cytoplasm",
    }
)
CELLPROFILER_2D_OPERATION_IDS = CELLPROFILER_COMPARTMENT_OPERATION_IDS | {
    "cellprofiler_propagation"
}
