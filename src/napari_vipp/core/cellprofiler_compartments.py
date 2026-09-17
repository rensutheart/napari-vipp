"""CPU 2D stages of the published CellProfiler 4.2.6 compartment profile.

These functions implement the settings in the statistics-paper author pipelines:
https://github.com/FrancisCrickInstitute/Enhancing-Reproducibility
Algorithms are adapted from CellProfiler 4.2.6 Smooth, Threshold,
IdentifyPrimaryObjects, IdentifySecondaryObjects and IdentifyTertiaryObjects.
https://github.com/CellProfiler/CellProfiler/tree/v4.2.6/cellprofiler/modules

The BSD 3-Clause License

Copyright © 2003 - 2021 Broad Institute, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1.  Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

    2.  Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

    3.  Neither the name of the Broad Institute, Inc. nor the names of its
        contributors may be used to endorse or promote products derived from
        this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED “AS IS.”  BROAD MAKES NO EXPRESS OR IMPLIED
REPRESENTATIONS OR WARRANTIES OF ANY KIND REGARDING THE SOFTWARE AND
COPYRIGHT, INCLUDING, BUT NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, CONFORMITY WITH ANY DOCUMENTATION,
NON-INFRINGEMENT, OR THE ABSENCE OF LATENT OR OTHER DEFECTS, WHETHER OR NOT
DISCOVERABLE. IN NO EVENT SHALL BROAD, THE COPYRIGHT HOLDERS, OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF, HAVE REASON TO KNOW, OR IN
FACT SHALL KNOW OF THE POSSIBILITY OF SUCH DAMAGE.

If, by operation of law or otherwise, any of the aforementioned warranty
disclaimers are determined inapplicable, your sole remedy, regardless of the
form of action, including, but not limited to, negligence and strict
liability, shall be replacement of the software with an updated version if one
exists.

Development of CellProfiler has been funded in whole or in part with federal
funds from the National Institutes of Health, the National Science Foundation,
and the Human Frontier Science Program.

Guidance is explicitly normalized float32 in [0, 1], matching CP's Image
container; one float32 ULP above 1 is accepted, unchanged, because CP's
normalized Gaussian can produce that rounding overshoot. These functions never
infer a normalization from image contents or clip guidance intensities.
Distances and diameters use pixels, independent of carried physical spacing.
Only full, unmasked YX planes are supported. The execution layer verifies axes
and aligned physical grids. All returned arrays own their buffers. Cancellation
is checked around blocking library calls, not during those calls.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from centrosome import cpmorphology
from centrosome.outline import outline
from centrosome.smooth import smooth_with_function_and_mask
from scipy import ndimage as ndi
from skimage.filters import threshold_li
from skimage.segmentation import watershed

from .cellprofiler_contracts import (
    CELLPROFILER_COMPARTMENT_OPERATION_IDS,
)

_MAX_LABEL = np.iinfo(np.int32).max
_NORMALIZED_UPPER = np.nextafter(np.float32(1), np.float32(np.inf))
_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def _checkpoint(progress, completed=False):
    if progress is not None:
        progress.check_cancelled()
        progress.report(int(completed), 1, "CellProfiler compartment profile")
        progress.check_cancelled()


def _plane(data):
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError("CellProfiler compartment stages require one 2D YX plane.")
    if array.size > _MAX_LABEL or any(n > _MAX_LABEL for n in array.shape):
        raise ValueError(
            "CellProfiler plane dimensions and pixel count must fit int32."
        )
    return array


def _guidance(data):
    array = _plane(data)
    if array.dtype != np.dtype(np.float32):
        raise ValueError(
            "CellProfiler guidance must be float32 normalized to [0, 1]; "
            "convert and scale explicitly upstream."
        )
    if not np.isfinite(array).all() or (
        array.size and (array.min() < 0 or array.max() > _NORMALIZED_UPPER)
    ):
        raise ValueError(
            "CellProfiler guidance must contain finite values in [0, 1], "
            "allowing one float32 ULP above 1 for reference Gaussian roundoff."
        )
    return np.array(array, dtype=np.float32, order="C", copy=True)


def _labels(data):
    array = _plane(data)
    if array.dtype.kind not in "ui" or (
        array.size and (int(array.min()) < 0 or int(array.max()) > _MAX_LABEL)
    ):
        raise ValueError(
            "CellProfiler labels must be nonnegative integers fitting int32."
        )
    return np.array(array, dtype=np.int32, order="C", copy=True)


def _label_pair(inputs):
    try:
        if len(inputs) != 2:
            raise ValueError("CellProfiler stage requires exactly two label inputs.")
    except TypeError as error:
        raise ValueError(
            "CellProfiler stage requires exactly two label inputs."
        ) from error
    a, b = (_labels(value) for value in inputs)
    if a.shape != b.shape:
        raise ValueError("CellProfiler label inputs must have matching 2D shapes.")
    return a, b


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number.")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite nonnegative number.") from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite nonnegative number.")
    return number


def _diameter(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer pixel diameter.")
    if not 1 <= int(value) <= _MAX_LABEL:
        raise ValueError(
            f"{name} must be a positive integer pixel diameter fitting int32."
        )
    return int(value)


def _normalized_gaussian(image, sigma):
    if sigma > (_MAX_LABEL - 1) / 8:
        raise ValueError("Gaussian kernel width must fit int32; reduce smoothing.")
    return smooth_with_function_and_mask(
        image,
        lambda value: ndi.gaussian_filter(value, sigma, mode="constant", cval=0),
        np.ones(image.shape, dtype=bool),
    )


def cellprofiler_smooth(data, artifact_diameter=2.0, progress=None):
    """CP Gaussian Smooth: sigma=artifact diameter/2.35, normalized edges.

    Return float32, matching the explicit Image conversion after CP Smooth.
    The fixed Gaussian profile has no clipping, background correction or mask.
    """
    _checkpoint(progress)
    image = _guidance(data)
    diameter = _nonnegative(artifact_diameter, "Artifact diameter")
    if image.size:
        result = np.array(
            _normalized_gaussian(image, diameter / 2.35),
            dtype=np.float32,
            order="C",
            copy=True,
        )
    else:
        result = image.copy()
    _checkpoint(progress, True)
    return result


def _threshold_value(image):
    if not image.size:
        return 0.0
    if np.all(image == image.flat[0]):
        return min(1.0, float(image.flat[0]))
    tolerance = max(float(np.min(np.diff(np.unique(image)))) / 2, 0.5 / 65536)
    # CP estimates the threshold BEFORE applying threshold smoothing.
    return min(1.0, max(0.0, float(threshold_li(image.ravel(), tolerance=tolerance))))


def _threshold_mask(image, scale):
    value = _threshold_value(image)
    if not image.size or scale == 0:
        return image >= value
    blurred = _normalized_gaussian(image, scale / 0.6744 / 2.0)
    return blurred >= value


def cellprofiler_threshold(data, smoothing_scale=0.0, progress=None):
    """CP global Minimum Cross-Entropy mask, fixed correction1/bounds[0,1].

    Li tolerance=max(minimum distinct intensity step/2, .5/65536); no log
    transform. Optional normalized Gaussian is applied only after estimating
    the threshold. The foreground comparison includes equality (>=).
    """
    _checkpoint(progress)
    image = _guidance(data)
    scale = _nonnegative(smoothing_scale, "Threshold smoothing scale")
    result = np.array(_threshold_mask(image, scale), dtype=bool, order="C", copy=True)
    _checkpoint(progress, True)
    return result


def _shape_maxima(image, labels, minimum_diameter):
    resize = 10.0 / minimum_diameter if minimum_diameter > 10 else 1.0
    suppression = 7.0 if resize < 1.0 else minimum_diameter / 1.5
    footprint = cpmorphology.strel_disk(max(1, suppression - 0.5))
    if resize < 1.0:
        shape = np.array(image.shape) * resize
        coordinates = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float) / resize
        reduced = ndi.map_coordinates(image, coordinates)
        reduced_labels = ndi.map_coordinates(labels, coordinates, order=0).astype(
            labels.dtype
        )
    else:
        reduced, reduced_labels = image, labels
    maxima = cpmorphology.is_local_maximum(reduced, reduced_labels, footprint)
    maxima[reduced <= 0] = False
    if resize < 1.0:
        inverse_resize = float(image.shape[0]) / float(maxima.shape[0])
        coordinates = (
            np.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(float)
            / inverse_resize
        )
        maxima = ndi.map_coordinates(maxima.astype(float), coordinates) > 0.5
    return cpmorphology.binary_shrink(maxima)


def cellprofiler_primary_objects(
    data,
    min_diameter=15,
    max_diameter=50,
    threshold_smoothing=1.3488,
    progress=None,
):
    """Return retained nuclei and pre-exclusion nuclei for CP Shape/Shape.

    Fixed published profile: global Li; fill holes before and after declumping;
    automatic maxima suppression with low-resolution maxima; shape maxima and
    8-neighbor shape watershed; remove border objects and out-of-range areas;
    continue regardless of object count. Diameter bounds specify equivalent
    circular areas. Unedited labels precede exclusions and relabeling. Tiny EDT
    tie-breaking noise uses local RandomState(0), leaving global RNG untouched.
    """
    _checkpoint(progress)
    image = _guidance(data)
    minimum = _diameter(min_diameter, "Minimum diameter")
    maximum = _diameter(max_diameter, "Maximum diameter")
    if maximum < minimum:
        raise ValueError("Maximum diameter must be at least the minimum diameter.")
    scale = _nonnegative(threshold_smoothing, "Threshold smoothing scale")
    if not image.size:
        _checkpoint(progress, True)
        return np.zeros(image.shape, np.int32), np.zeros(image.shape, np.int32)
    foreground = _threshold_mask(image, scale)
    foreground = cpmorphology.fill_labeled_holes(
        foreground, size_fn=lambda size, _is_foreground: size < maximum * maximum
    )
    connected, count = ndi.label(foreground, _EIGHT_CONNECTED)
    if count == 0:
        _checkpoint(progress, True)
        return np.zeros(image.shape, np.int32), np.zeros(image.shape, np.int32)
    distance = ndi.distance_transform_edt(connected > 0)
    distance += np.random.RandomState(0).uniform(0, 0.001, distance.shape)
    maxima = _shape_maxima(distance, connected, minimum)
    labeled_maxima, count = ndi.label(maxima, _EIGHT_CONNECTED)
    elevation = -distance
    elevation -= elevation.min()
    marker_dtype = np.int16 if count < np.iinfo(np.int16).max else np.int32
    markers = -labeled_maxima.astype(marker_dtype)
    segmented = -watershed(
        elevation, markers=markers, connectivity=_EIGHT_CONNECTED, mask=connected != 0
    )
    unedited = np.array(segmented, dtype=np.int32, order="C", copy=True)
    border_ids = np.unique(
        np.concatenate((segmented[0], segmented[-1], segmented[:, 0], segmented[:, -1]))
    )
    segmented[np.isin(segmented, border_ids)] = 0
    if count:
        areas = np.bincount(segmented.ravel(), minlength=count + 1)
        allowed = (areas >= np.pi * minimum**2 / 4) & (areas <= np.pi * maximum**2 / 4)
        allowed[0] = False
        segmented[~allowed[segmented]] = 0
    segmented = cpmorphology.fill_labeled_holes(segmented)
    retained, _ = cpmorphology.relabel(segmented)
    retained = np.array(retained, dtype=np.int32, order="C", copy=True)
    _checkpoint(progress, True)
    return retained, unedited


def cellprofiler_propagation_seeds(inputs, progress=None):
    """Prepare [unedited nuclei, retained nuclei], keeping edge competitors.

    Output IDs remain those of the unedited nuclei. Edge-touching excluded
    nuclei remain competitors until cell regions are finished and remapped.
    """
    _checkpoint(progress)
    unedited, retained = _label_pair(inputs)
    if unedited.size:
        border = np.unique(
            np.concatenate((unedited[0], unedited[-1], unedited[:, 0], unedited[:, -1]))
        )
        unedited[(~np.isin(unedited, border)) & (retained == 0)] = 0
    _checkpoint(progress, True)
    return unedited


def cellprofiler_finish_cells(inputs, fill_holes=True, progress=None):
    """Finish [grown regions, retained nuclei] and return accepted nucleus IDs.

    Fill labeled holes, then map each region to the maximum retained nuclear
    ID overlapping it, exactly as CP's IdentifySecondaryObjects filter_labels.
    Regions with no accepted nucleus are removed. Border cells are retained.
    """
    _checkpoint(progress)
    grown, retained = _label_pair(inputs)
    if not isinstance(fill_holes, (bool, np.bool_)):
        raise ValueError("Fill holes must be Boolean.")
    if not grown.size:
        _checkpoint(progress, True)
        return grown
    # Compact temporary IDs to avoid allocating max-ID arrays for sparse labels.
    ids = np.unique(grown[grown != 0])
    if not len(ids):
        _checkpoint(progress, True)
        return grown
    compact = np.zeros(grown.shape, np.int32)
    compact[grown != 0] = np.searchsorted(ids, grown[grown != 0]) + 1
    if fill_holes:
        compact = cpmorphology.fill_labeled_holes(compact, mask=compact == 0)
    lookup = np.zeros(len(ids) + 1, np.int32)
    np.maximum.at(lookup, compact.ravel(), retained.ravel())
    lookup[0] = 0
    result = np.array(lookup[compact], dtype=np.int32, order="C", copy=True)
    _checkpoint(progress, True)
    return result


def cellprofiler_cytoplasm(inputs, shrink_nuclei=True, progress=None):
    """Subtract nuclei from cells; optionally retain the CP nuclear outline.

    Inputs are [cells, nuclei]. With the paper's shrink setting, nuclear
    boundary pixels belong to both the measured Nuclei and Cytoplasm regions.
    Uses Centrosome's exact outline convention, not a generic binary erosion.
    """
    _checkpoint(progress)
    cells, nuclei = _label_pair(inputs)
    if not isinstance(shrink_nuclei, (bool, np.bool_)):
        raise ValueError("Shrink nuclei must be Boolean.")
    if cells.size:
        keep = nuclei == 0
        if shrink_nuclei:
            keep |= outline(nuclei) != 0
        cells[~keep] = 0
    _checkpoint(progress, True)
    return cells


__all__ = [
    *sorted(CELLPROFILER_COMPARTMENT_OPERATION_IDS),
    "CELLPROFILER_COMPARTMENT_OPERATION_IDS",
]
