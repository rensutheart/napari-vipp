# Context-aware controls audit

This audit covers all 108 operation specifications and their parameter
declarations. Context-aware visibility is presentation only: a hidden value
remains in the node, workflow JSON, generated Python, batch execution, cache
key, and undo/redo snapshots. No visibility rule changes an operation default
or scientific algorithm.

Automatic hiding uses explicit carried metadata. Shape-only axis guesses,
disconnected inputs, missing dtypes, unresolved ports, and partially resolved
multi-input nodes keep potentially relevant controls visible. In the inspector,
one concise note explains either that values were hidden without being changed
or that unresolved context is keeping controls available.

## Catalog audit

| Operation or family | Parameter(s) | Controlling context | Relevant / irrelevant / unresolved behavior | Scientific justification | Status |
| --- | --- | --- | --- | --- | --- |
| Otsu, Triangle, Yen, Isodata, Minimum | `histogram_bins` | Primary input dtype | Float: show. Integer/bool: hide. Unknown: show. | These operations use exact integer levels; the bin count affects floating histograms only. | Implemented: `floating_input`. |
| Prepare/Validate PSF; RL; RL-TV | `clip_negatives`, `clip_negative_input` | Primary input dtype | Signed/float: show. Unsigned/bool: hide. Unknown: show. | Unsigned and boolean arrays cannot contain negative samples. | Implemented: `negative_values_possible`. |
| Crop Stack, Average Blur, Gaussian Blur, Gaussian Blur 3D, Median, Bilateral, Non-Local Means, Rolling Ball, Subtract Background, Difference of Gaussians, Unsharp Mask | `channel_axis` | Explicit ordinary channel semantics | Explicit channel axis, including encoded colour: show. Explicit scalar: hide. Unknown: show. | Channel-wise filtering differs from scalar spatial filtering. No channel is inferred from length 3 or 4. | Implemented: shared `multichannel_input`, including the three former custom duplicates. |
| Sobel, Canny, Laplace, global/local thresholds | encoded-colour `channel_axis` | Explicit RGB/RGBA semantics | Explicit `rgb`/`rgba`: show. Explicit scalar or fluorescence `C`: hide. Unknown: show. | Luminance conversion is correct for encoded colour, not for fluorescence channels. | Implemented: `rgb_or_rgba_input`. |
| Global thresholds | `threshold_scope` | Explicit non-channel stack axis | Z/T/series stack: show. Resolved YX or channel-only CYX/RGB: hide. Unknown: show. | Stack and slice histograms are identical without a distinct stack axis. | Implemented: `stack_scope_relevant`. |
| Shared spatial operations | `spatial_mode` | All declared image/label/mask ports, explicit axes, stored mode | Pure scalar YX with Auto/2D: hide. Three spatial axes, leading non-spatial axes, unresolved/disconnected ports, or stored 3D on YX: show. | Plane-wise and volumetric processing must not be selected from rank alone; an invalid stored 3D choice must remain repairable. | Implemented catalog-wide: `spatial_mode_relevant`. |
| RL and RL-TV | `spatial_mode` | Image and PSF axes | Both resolved YX: hide if stored mode is equivalent. Either volumetric, mismatched, missing, or unresolved: show. | PSF/image compatibility depends on both ports. | Implemented through named multi-input context; grid validation remains specialized. |
| Born-Wolf PSF | `spatial_mode`, optical fields | Connected image acquisition metadata and Auto/manual mode | Spatial mode remains visible. Auto fields remain visible but disabled with resolved/missing status; manual mode enables them. | A YX image can provide metadata for deliberately configured 3D PSF generation, and visible resolved values explain provenance. | Intentionally specialized; documented exception to the shared spatial hide. |
| Measure 3D Mesh Morphology | `spatial_mode` | Required true 3D label input | Always show with the 3D requirement status. | Hiding on invalid 2D input would conceal why execution cannot proceed. | Intentionally always visible; documented exception. |
| Gaussian Blur 3D; Set Pixel Size; Rescale Axes | `sigma_z`, `z_size`, `z_scale`/optional `z_size` | Explicit Z spatial axis | Explicit Z: show. Explicit no-Z: hide. Unknown: show. | Z-only parameters have no effect without a Z spatial axis. | Implemented: `three_spatial_dimensions`; Rescale retains its specialized scale/size renderer. |
| Measure Objects; Measure Objects + Intensity | 2D boundary/moment flags | Selected/resolved spatial processing | 2D: show. 3D: hide. Unknown: show. | These descriptors are defined only for 2D processing. | Implemented: `two_dimensional_processing`. |
| Measure Objects; Measure Objects + Intensity | axis descriptors, derived ratios | Resolved spatial dimensionality | At least 2D: show. Explicit 1D: hide. Unknown: show. | The descriptors require at least two spatial dimensions. | Implemented: `at_least_two_spatial_dimensions`. |
| Clear Border Objects | `boundary_mode` | Spatial axes | Pure scalar YX: hide. Volumetric/leading/unresolved: show. | All-spatial and lateral-only borders are equivalent on a single YX plane. | Implemented with spatial-choice relevance. |
| Rescale Intensity | value and percentile cutoff pairs | `cutoff_mode` | Show only the selected representation. Missing mode: show conservatively. | The unselected pair has no effect. | Implemented: `parameter_in`. |
| Clip | `minimum`, `maximum` | `cutoff_mode` | Values: show. Data range: hide. Missing mode: show. | Explicit cutoffs are unused in data-range mode. | Implemented: `parameter_in`. |
| Seven colocalization families, including Object Colocalization | manual channel thresholds | `threshold_mode` | Manual: show. Costes auto: hide. Unknown: show. | Costes derives both thresholds. The scatter/status still exposes resolved guides; dragging a guide switches back to Manual. | Implemented across all 14 threshold declarations. |
| RL-TV | `tv_epsilon`, `denominator_floor` | `tv_regularization` | Non-zero: show. Zero: hide. Missing value: show. | Both guards are unused when TV regularization is off and RL-TV reduces to ordinary RL. | Implemented: `parameter_not_in`. |
| Colocalized Voxels, masked variant | channel colours | `display_mode` | White-on-black: hide. Colour-bearing modes: show. | Channel colours do not contribute to a white-on-black display. | Implemented. |
| Skeleton Graph Overlay | `node_size` | `display_mode` | Coloured-edges-only: hide. Node-bearing modes: show. | No nodes are drawn in the edges-only mode. | Implemented. |
| Convert Dtype | `scaling` | `output_dtype` | Boolean output: hide. Other outputs: show. | Boolean conversion ignores numeric scaling strategy. | Implemented. |
| Image Source | source/layer/file/sample/series/binding/channel-colour fields | Source mode, series metadata, channel metadata | Renderer shows the selected source representation, series only when useful, and a compact source status. Unknown metadata remains explicit. | Source selection and acquisition validation need richer status than row predicates. | Intentionally specialized. |
| Composite to RGB | axis mode, mapping mode, dynamic channel assignments, legacy RGB fields | Explicit channel metadata, Auto/manual modes, channel count | Dynamic renderer shows resolved read-only Auto choices and editable Manual choices; legacy stored fields remain compatible. | Fluorescence pseudo-colours and encoded RGB/RGBA mappings are scientifically distinct. | Intentionally specialized. |
| Split Channels / Extract Channel / Assign Channel Colors | channel metadata and dynamic output use | Explicit channel axis, channel count, used output ports | Specialized renderers preserve output/status behavior; missing semantics report an error or remain manually resolvable. | Dynamic graph ports and inspection output selection require more than row hiding. | Intentionally specialized. |
| Rescale Axes | axis roles, scale/output-size mode, interpolation | Explicit axes and selected representation | Scale versus output-size controls follow the chosen mode. Z hides only from explicit no-Z metadata; unresolved axes retain X/Y/Z fields and a status note. | Axis roles must not be inferred solely from array rank. | Specialized renderer integrated with shared Z rule. |
| Dynamic-input math/logical/combine/merge nodes | `input_count`, weights/colour lists | Saved arity and dynamic ports | Keep visible even when fewer ports are connected. | Arity is workflow configuration, not a reflection of current connection count; auto-hiding would prevent preconfiguration. | Intentionally visible. |
| Save Image / Batch Output | path, format, overwrite, enabled state | User save intent | Keep visible while disabled. | Preconfiguring a destination before enabling save is meaningful product behavior. | Intentionally visible. |
| Orthogonal Projection / skeleton physical-length controls | physical-scale options | Calibration metadata | Keep visible with validation/status when calibration is missing. | Unknown calibration must not silently remove a deliberate physical-units choice. | Intentionally visible pending a richer calibration-status rule. |
| Rescale interpolation / anti-aliasing | data kind, selected interpolation, actual downsampling | Multiple unresolved facts | Keep visible. | Labels, masks, images, and unresolved output geometry need different validation; hiding before the complete resampling contract is known is unsafe. | Intentionally visible pending a product decision. |
| Boolean intensity inputs | intensity-operation controls | Input dtype and operation output contract | Keep visible. | Several operations convert boolean input to a numeric output or otherwise retain meaningful parameters; a catalog-wide boolean hide would change discoverability without a single safe rule. | Intentionally visible. |
| Remaining morphology, table, metadata and batch controls | operation-specific parameters | Valid typed input | Keep visible. | Their values remain meaningful whenever the typed node can execute; connection type validation supplies the relevant context. | Audited, no row-visibility rule required. |

## Rule and compatibility contract

`ParameterVisibilityContext` contains operation ID, stored parameter values,
input data and `ImageState` by declared port name, connected ports, and used
output ports. It is Qt-free. Rules are validated when tested; unknown or
incompletely declared rules raise `ValueError` rather than silently hiding a
row.

The workflow schema remains version 3. Visibility fields belong to immutable
catalog specifications and are not serialized as node state. Presentation
refreshes do not call `set_param`, normalize a value through a control, dirty a
node, schedule execution, alter caches, or capture history. Older version-3
workflows therefore restore with the same stored values, including values that
the current context hides.
