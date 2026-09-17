"""Audit/recompute Marcotti et al. (2026) reference measurements, without VIPP.

Run with Python containing NumPy, pandas and SciPy. Original downloaded files are
read only. Default output is
D:/VIPP-paper-reproductions/validation/statistics-paper/reference;
--output selects a separate directory for repeat runs.
The authors' four preparation helpers are executed unchanged from their pinned
source using AST selection (avoids importing optional plotting dependencies).
Independent narrow-table joins and sampling calculations verify those helpers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import re
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

AUTHOR_COMMIT = "a81d20c9393485e57395cd4a1be9e7ff83c7f11d"
HELPER_SHA256 = "a74256fee27517fdf39f994e1b4e9ece70198e60942af501020a8f406afa63aa"
PAPER = "https://doi.org/10.1242/jcs.264367"
PRIMARY_0139 = ["J05", "O02", "E22", "L08"]
ALL_0139 = ["J05", "I19", "G15", "O02", "B02", "N12", "L08", "L18", "H13", "E22"]
ALL_0028 = {
    "1A": ["I23", "J23", "M23", "N23", "B06"],
    "2A": ["C13", "J23", "N23", "K16"],
    "2B": ["C13", "C17", "P13"],
}
DISPLAY_ORDER = {
    "idr0139": ["Untreated", "DMSO", "SN0212398523", "Leptomycin b"],
    "idr0028": ["LATS1", "ARAP2", "MOCK", "YAP"],
}
EFFECT_ORDER = {
    "idr0139": ["SN0212398523", "DMSO", "Leptomycin b"],
    "idr0028": ["ARAP2", "YAP", "LATS1"],
}
CONTROL = {"idr0139": "Untreated", "idr0028": "MOCK"}
TABLE1 = {
    "J05": {
        "table_index": 1,
        "mean": 0.465,
        "sem_cell": 0.013,
        "median": 0.511,
        "q25": 0.418,
        "q75": 0.547,
        "std_sample": 0.131,
        "n_cells": 1362,
    },
    "E22": {
        "table_index": 4,
        "mean": 0.488,
        "sem_cell": 0.017,
        "median": 0.533,
        "q25": 0.448,
        "q75": 0.558,
        "std_sample": 0.140,
        "n_cells": 844,
    },
    "L08": {
        "table_index": 5,
        "mean": 0.579,
        "sem_cell": 0.024,
        "median": 0.569,
        "q25": 0.543,
        "q75": 0.626,
        "std_sample": 0.108,
        "n_cells": 606,
    },
    "O02": {
        "table_index": 8,
        "mean": 0.476,
        "sem_cell": 0.013,
        "median": 0.524,
        "q25": 0.441,
        "q75": 0.554,
        "std_sample": 0.131,
        "n_cells": 1299,
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv(path: Path, rows: Any) -> pd.DataFrame:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
    return frame


def author_helpers(source: Path) -> dict[str, Any]:
    assert sha256(source) == HELPER_SHA256, "Unexpected author helper revision"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {
        "normalize_well_format",
        "load_and_prepare_data",
        "prepare_data",
        "map_wells_to_treatments",
        "exp_decay",
    }
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"np": np, "pd": pd, "re": re, "Dict": dict}
    exec(compile(tree, str(source), "exec"), namespace)  # noqa: S102 -- hash-verified reviewed author helpers
    return namespace


def describe(values: pd.Series) -> dict[str, float | int]:
    return {
        "n_cells": len(values),
        "n_finite": int(np.isfinite(values).sum()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q25": float(values.quantile(0.25)),
        "q75": float(values.quantile(0.75)),
        "std_sample": float(values.std(ddof=1)),
        "std_population": float(values.std(ddof=0)),
        "sem_cell": float(values.sem(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def group_summaries(data: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in data.groupby(keys, sort=True, dropna=False):
        if len(keys) == 1:
            values = (values,) if not isinstance(values, tuple) else values
        rows.append(
            {
                **dict(zip(keys, values, strict=False)),
                **describe(group.Ratio),
                "n_wells": int(group.Replicate.nunique()),
                "n_images": int(
                    group[["Plate", "ImageNumber"]].drop_duplicates().shape[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def prepare_plate(
    base: Path, case: str, plate: str, helpers: dict
) -> tuple[pd.DataFrame, dict]:
    subset = base / "data_subsets"
    if case == "idr0139":
        protein, filename = "Fascin", "FileName_DNA"
        table_dir = subset / "cell_profiler_outputs/idr0139"
        annotation = helpers["load_and_prepare_data"](
            str(subset / "idr/idr0139-screenA-annotation.csv"),
            int(plate),
            "Control Type",
            "Treated",
        )
        selected = ALL_0139
        mapping = {
            "Treated": "Treated",
            "Negative Control": "Untreated",
            "Neutral Control": "DMSO",
            "Stimulator Control": "Leptomycin b",
        }
        compounds = (
            annotation[annotation["Control Type"] == "Treated"]
            .set_index("Well")["Proprietary Compound"]
            .to_dict()
        )
        treatments = annotation.set_index("Well")["Control Type"].to_dict()
        index = None
    else:
        protein, filename = "YAPTAZ", "FileName_Hoechst"
        table_dir = (
            subset
            / f"cell_profiler_outputs/idr0028/screenB/LM2_GEFGAP_ONTARGETPlus_{plate}"
        )
        annotation = helpers["load_and_prepare_data"](
            str(subset / "idr/idr0028-screenB-annotation.csv"),
            f"LM2_ONTARGETPlus_{plate}",
            "Gene Symbol",
            "MOCK",
        )
        selected, mapping = ALL_0028[plate], None
        compounds = (
            annotation[annotation["Control Type"] == "Treated"]
            .set_index("Well")["Gene Symbol"]
            .to_dict()
        )
        treatments = annotation.set_index("Well")["Gene Symbol"].to_dict()
        index = pd.read_csv(
            subset / f"idr/LM2_GEFGAP_ONTARGETPlus_{plate}_ImageIndex.ColumbusIDX.csv",
            sep="\t",
        )
    nuclei, cyto, images = [
        pd.read_csv(table_dir / f"{name}.csv")
        for name in ("Nuclei", "Cytoplasm", "Image")
    ]
    keys = ["ImageNumber", "ObjectNumber"]
    assert not nuclei.duplicated(keys).any()
    assert not cyto.duplicated(keys).any()
    assert not images.duplicated("ImageNumber").any()
    assert not annotation.duplicated("Well").any()
    check = nuclei[keys].merge(
        cyto[keys], on=keys, how="outer", indicator=True, validate="one_to_one"
    )
    assert check._merge.eq("both").all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        prepared = helpers["prepare_data"](
            nuclei,
            cyto,
            images,
            index,
            treatments,
            mapping,
            compounds,
            selected,
            [protein],
        )
    intensity = f"Intensity_MeanIntensity_{protein}"
    # Independent minimal join, with explicit relational cardinality assertions.
    independent = (
        nuclei[keys + [intensity]]
        .merge(
            cyto[keys + [intensity, "Parent_Nuclei", "Parent_Cells"]],
            on=keys,
            how="left",
            suffixes=("_nuclear", "_cytoplasmic"),
            validate="one_to_one",
        )
        .merge(
            images[["ImageNumber", filename]],
            on="ImageNumber",
            how="left",
            validate="many_to_one",
        )
    )
    independent["ratio"] = independent[intensity + "_nuclear"] / (
        independent[intensity + "_cytoplasmic"] + independent[intensity + "_nuclear"]
    )
    pd.testing.assert_series_equal(
        prepared.set_index(keys)[f"{protein}_Ratio"].sort_index(),
        independent.set_index(keys).ratio.sort_index(),
        check_names=False,
        check_exact=True,
    )
    assert len(prepared) == len(nuclei)
    assert (cyto.Parent_Nuclei == cyto.ObjectNumber).all()
    assert (cyto.Parent_Cells == cyto.ObjectNumber).all()
    assert not prepared.Treatment.eq("Unknown").any()
    assert prepared[filename].notna().all()
    count = (
        nuclei.groupby("ImageNumber")
        .size()
        .reindex(images.ImageNumber, fill_value=0)
        .to_numpy()
    )
    assert np.array_equal(count, images.Count_Nuclei.to_numpy())
    assert np.array_equal(count, images.Count_Cytoplasm.to_numpy())
    assert np.array_equal(count, images.Count_Cells.to_numpy())
    normalized = pd.DataFrame(
        {
            "Case": case,
            "Plate": plate,
            "Well": prepared.Well,
            "Replicate": (plate + "_" + prepared.Well)
            if case == "idr0028"
            else prepared.Well,
            "Treatment": prepared.Treatment,
            "ImageNumber": prepared.ImageNumber,
            "ObjectNumber": prepared.ObjectNumber,
            "ReferenceRowIndex": prepared.index,
            "FileName": prepared[filename],
            "NuclearMean": prepared[f"Nuclear_{intensity}"],
            "CytoplasmicMean": prepared[f"Cyto_{intensity}"],
            "Ratio": prepared[f"{protein}_Ratio"],
        }
    )
    assert (
        np.isfinite(normalized[["NuclearMean", "CytoplasmicMean", "Ratio"]]).all().all()
    )
    assert ((normalized.NuclearMean + normalized.CytoplasmicMean) > 0).all()
    assert normalized.Ratio.between(0, 1).all()
    assert len(images) == normalized.ImageNumber.nunique()
    checks = {
        "case": case,
        "plate": plate,
        "protein": protein,
        "cells": len(normalized),
        "images": len(images),
        "wells": len(selected),
        "duplicate_object_keys": 0,
        "unmatched_object_keys": 0,
        "parent_nuclei_and_cells_equal_object_number": True,
        "duplicate_image_keys": 0,
        "all_image_counts_equal_object_row_counts": True,
        "nonfinite_ratios": 0,
        "nonpositive_denominators": 0,
        "unknown_treatments": 0,
        "independent_join_exact": True,
        "reference_row_order_preserved": True,
        "quality_control_annotations": annotation["Quality Control"]
        .fillna("<missing>")
        .value_counts()
        .to_dict(),
    }
    if index is not None:
        # Four rows per filename reflect channels of one multichannel TIFF.
        ambiguous = index.groupby("sourcefilename").WellName.nunique().gt(1)
        assert not ambiguous.any()
        checks["index_rows"] = len(index)
        checks["index_unique_filenames"] = int(index.sourcefilename.nunique())
        checks["ambiguous_filename_to_well_mappings"] = 0
    return normalized, checks


def sample_members(group: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    return group.sample(n=n, replace=False, random_state=seed)


def sampling_tables(
    case: str, primary: pd.DataFrame, all_cells: pd.DataFrame, output: Path
) -> dict:
    output.mkdir(exist_ok=True)
    groups = {t: primary[primary.Treatment == t] for t in primary.Treatment.unique()}
    member_rows, summaries = [], []
    for n, seeds in ((50, range(42, 45)), (200, range(45, 48))):
        for panel, seed in enumerate(seeds, 1):
            for treatment in DISPLAY_ORDER[case]:
                sampled = sample_members(groups[treatment], n, seed)
                summaries.append(
                    {
                        "n_requested": n,
                        "panel": panel,
                        "seed": seed,
                        "Treatment": treatment,
                        **describe(sampled.Ratio),
                    }
                )
                member_rows.append(sampled.assign(SampleSize=n, Panel=panel, Seed=seed))
    csv(output / "swarm_sample_summary.csv", summaries)
    csv(output / "swarm_sample_members.csv", pd.concat(member_rows, ignore_index=True))
    summaries, member_rows = [], []
    for n in [50, 200, 500]:
        for treatment in DISPLAY_ORDER[case]:
            subset = all_cells[all_cells.Treatment == treatment]
            for replicate_index, replicate in enumerate(subset.Replicate.unique()):
                sampled = sample_members(subset[subset.Replicate == replicate], n, 42)
                summaries.append(
                    {
                        "n_requested": n,
                        "seed": 42,
                        "Treatment": treatment,
                        "Replicate": replicate,
                        "ReplicateIndex": replicate_index,
                        **describe(sampled.Ratio),
                    }
                )
                member_rows.append(
                    sampled.assign(
                        SampleSize=n, ReplicateIndex=replicate_index, Seed=42
                    )
                )
    csv(output / "superplot_sample_summary.csv", summaries)
    csv(
        output / "superplot_sample_members.csv",
        pd.concat(member_rows, ignore_index=True),
    )

    # Cumulative batches: sample ten remaining cells each time, advancing seed.
    # This is NOT one shuffle followed by prefixes.
    control = groups[CONTROL[case]]
    reference = describe(control.Ratio)
    reference["iqr"] = reference["q75"] - reference["q25"]
    cumulative, summaries, cumulative_members = [], [], []
    for step, n in enumerate(range(10, 501, 10)):
        available = control[~control.index.isin(cumulative)]
        new = sample_members(available, min(10, len(available)), 42 + step)
        cumulative.extend(new.index.tolist())
        cumulative_members.append(new.assign(AddedAtN=n, Seed=42 + step))
        desc = describe(control.loc[cumulative, "Ratio"])
        desc["iqr"] = desc["q75"] - desc["q25"]
        summaries.append(
            {
                "sample_size": n,
                **desc,
                **{
                    "difference_" + key: desc[key] - reference[key]
                    for key in ["mean", "median", "std_sample", "iqr"]
                },
            }
        )
    csv(output / "cumulative_summary.csv", summaries)
    csv(
        output / "cumulative_members.csv",
        pd.concat(cumulative_members, ignore_index=True),
    )
    histogram_rows = []
    for n in [20, 50, 100, 200, 300, 500]:
        # The notebook uses 50 automatic-range bins, density=True (not percent).
        heights, edges = np.histogram(
            control.loc[cumulative[:n], "Ratio"], bins=50, density=True
        )
        for idx, height in enumerate(heights):
            histogram_rows.append(
                {
                    "sample_size": n,
                    "bin": idx,
                    "left": edges[idx],
                    "right": edges[idx + 1],
                    "density": height,
                }
            )
    csv(output / "cumulative_histograms.csv", histogram_rows)

    iqr_trials, effect_trials = [], []
    iqr_seed, effect_seed = 42, 42
    for n in range(10, 500, 10):
        for iteration in range(100):
            for treatment, group in groups.items():
                values = group.Ratio.sample(n=n, replace=False, random_state=iqr_seed)
                q25, q75 = np.percentile(values, [25, 75])
                iqr_trials.append(
                    {
                        "sample_size": n,
                        "iteration": iteration,
                        "seed": iqr_seed,
                        "Treatment": treatment,
                        "iqr": q75 - q25,
                    }
                )
            iqr_seed += 1
            for treatment in EFFECT_ORDER[case]:
                values = groups[treatment].Ratio.sample(
                    n=n, replace=False, random_state=effect_seed
                )
                ref = control.Ratio.sample(n=n, replace=False, random_state=effect_seed)
                delta = (values.mean() - ref.mean()) / ref.std(ddof=1)
                effect_trials.append(
                    {
                        "sample_size": n,
                        "iteration": iteration,
                        "seed": effect_seed,
                        "Treatment": treatment,
                        "standardized_mean_difference": float(delta),
                    }
                )
                effect_seed += 1
    iqr = csv(output / "iqr_trials.csv", iqr_trials)
    effects = csv(output / "effect_trials.csv", effect_trials)
    ranges = (
        iqr.groupby(["Treatment", "sample_size"]).iqr.agg(["min", "max"]).reset_index()
    )
    ranges["range"] = ranges["max"] - ranges["min"]
    fit_rows = []
    for treatment in groups:
        subset = ranges[ranges.Treatment == treatment]
        parameters, _ = curve_fit(
            lambda x, a, b, c: a * np.exp(-b * x) + c,
            subset.sample_size,
            subset["range"],
            p0=[1, 0.01, np.median(subset["range"])],
            maxfev=5000,
        )
        fit_rows.append(
            {
                "Treatment": treatment,
                "a": parameters[0],
                "b": parameters[1],
                "c": parameters[2],
            }
        )
        ranges.loc[subset.index, "exponential_fit"] = (
            parameters[0] * np.exp(-parameters[1] * subset.sample_size) + parameters[2]
        )
    csv(output / "iqr_ranges.csv", ranges)
    csv(output / "iqr_exponential_fit.csv", fit_rows)
    effect_summary = []
    for (treatment, n), group in effects.groupby(["Treatment", "sample_size"]):
        values = group.standardized_mean_difference.to_numpy()
        effect_summary.append(
            {
                "Treatment": treatment,
                "sample_size": n,
                "median": np.nanmedian(values),
                "q25": np.nanpercentile(values, 25),
                "q75": np.nanpercentile(values, 75),
            }
        )
    csv(output / "effect_summary.csv", effect_summary)
    full_effects = [
        {
            "Treatment": t,
            "Control": CONTROL[case],
            "standardized_mean_difference": (
                groups[t].Ratio.mean() - control.Ratio.mean()
            )
            / control.Ratio.std(),
        }
        for t in EFFECT_ORDER[case]
    ]
    csv(output / "full_population_effects.csv", full_effects)
    return {
        "iqr_trial_count": len(iqr),
        "effect_trial_count": len(effects),
        "iqr_seed_first": 42,
        "iqr_seed_last": iqr_seed - 1,
        "effect_seed_first": 42,
        "effect_seed_last": effect_seed - 1,
        "full_population_effects": full_effects,
    }


class NullProgress:
    """Permit unchanged author's numerical loops to run without notebook UI."""

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def update(self, _n):
        pass


def verify_author_sampling(source: Path, data: pd.DataFrame, case: str) -> dict:
    """Run original nested loops at bounded n/iteration values, bypassing plots.

    Only function tails are replaced with `return` of their accumulated trial
    arrays. Original filtering, sampling, seed advancement and statistics remain.
    """
    tree = ast.parse(source.read_text(encoding="utf-8"))
    targets = {
        "plot_iqr_v_sample_size": ("iqr_values_min", "iqr_values"),
        "plot_effect_size_v_sample_size": ("median_values_mean", "mean_values"),
    }
    chosen = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in targets:
            boundary, result = targets[node.name]
            idx = next(
                i
                for i, sub in enumerate(node.body)
                if isinstance(sub, ast.Assign)
                and isinstance(sub.targets[0], ast.Name)
                and sub.targets[0].id == boundary
            )
            node.body = node.body[:idx] + [
                ast.Return(value=ast.Name(id=result, ctx=ast.Load()))
            ]
            chosen.append(node)
    tree.body = chosen
    ast.fix_missing_locations(tree)
    namespace = {"np": np, "pd": pd, "tqdm": NullProgress}
    exec(compile(tree, str(source), "exec"), namespace)  # noqa: S102 -- hash-verified reviewed author loops
    sizes, iterations = [10, 50, 200, 490], 3
    values_iqr = namespace["plot_iqr_v_sample_size"](
        sizes, iterations, data, "Treatment", "Ratio", ""
    )
    values_effect = namespace["plot_effect_size_v_sample_size"](
        sizes,
        iterations,
        data,
        "Treatment",
        "Ratio",
        "",
        EFFECT_ORDER[case],
        CONTROL[case],
    )
    groups = {t: data[data.Treatment == t].Ratio for t in data.Treatment.unique()}
    expected_iqr = {t: [[] for _ in sizes] for t in groups}
    expected_effect = {t: [[] for _ in sizes] for t in EFFECT_ORDER[case]}
    seed_iqr, seed_effect = 42, 42
    for idx, n in enumerate(sizes):
        for _ in range(iterations):
            for treatment, series in groups.items():
                q25, q75 = np.percentile(
                    series.sample(n=n, replace=False, random_state=seed_iqr), [25, 75]
                )
                expected_iqr[treatment][idx].append(q75 - q25)
            seed_iqr += 1
            for treatment in EFFECT_ORDER[case]:
                sample = groups[treatment].sample(
                    n=n, replace=False, random_state=seed_effect
                )
                ref = groups[CONTROL[case]].sample(
                    n=n, replace=False, random_state=seed_effect
                )
                expected_effect[treatment][idx].append(
                    (sample.mean() - ref.mean()) / ref.std()
                )
                seed_effect += 1
    for key, expected in expected_iqr.items():
        np.testing.assert_array_equal(expected, values_iqr[key])
    for key, expected in expected_effect.items():
        np.testing.assert_array_equal(expected, values_effect[key])
    return {
        "unchanged_original_loops_match_exactly": True,
        "sample_sizes": sizes,
        "iterations_per_size": iterations,
        "iqr_values_checked": 48,
        "effect_values_checked": 36,
        "scope": (
            "Bounded cross-check of original loops; full curves recomputed "
            "independently."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("D:/VIPP-paper-reproductions/statistics/reference"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "D:/VIPP-paper-reproductions/validation/statistics-paper/reference"
        ),
    )
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    source = args.reference / "authors-code/notebooks/utility_functions.py"
    helper = author_helpers(source)
    frames, checks = [], []
    for case, plate in [
        ("idr0139", "1093711385"),
        ("idr0028", "1A"),
        ("idr0028", "2A"),
        ("idr0028", "2B"),
    ]:
        data, check = prepare_plate(args.reference, case, plate, helper)
        frames.append(data)
        checks.append(check)
    all_cells = pd.concat(frames, ignore_index=True)
    assert not all_cells.duplicated(
        ["Case", "Plate", "ImageNumber", "ObjectNumber"]
    ).any()
    csv(output / "cells.csv", all_cells)
    images = group_summaries(
        all_cells,
        ["Case", "Plate", "Well", "Replicate", "Treatment", "ImageNumber", "FileName"],
    )
    csv(output / "image_summary.csv", images)
    wells = group_summaries(
        all_cells, ["Case", "Plate", "Well", "Replicate", "Treatment"]
    )
    for idx, row in wells.iterrows():
        field_means = images[
            (images.Case == row.Case)
            & (images.Plate == row.Plate)
            & (images.Well == row.Well)
        ]["mean"]
        wells.loc[idx, "unweighted_fov_mean"] = field_means.mean()
        wells.loc[idx, "sem_of_fov_means"] = field_means.sem(ddof=1)
    csv(output / "well_summary.csv", wells)
    primary_by_case = {}
    for case in ["idr0139", "idr0028"]:
        # Restore the original dataframe index for exact cumulative membership.
        frame = frames[0] if case == "idr0139" else frames[2]
        primary_by_case[case] = (
            frame[frame.Well.isin(PRIMARY_0139)].copy()
            if case == "idr0139"
            else frame.copy()
        )
    primary = pd.concat(primary_by_case.values(), ignore_index=True)
    csv(output / "primary_cells.csv", primary)
    csv(
        output / "primary_treatment_summary.csv",
        group_summaries(primary, ["Case", "Treatment"]),
    )
    csv(
        output / "all_wells_treatment_summary.csv",
        group_summaries(all_cells, ["Case", "Treatment"]),
    )
    comparison = []
    quantile_checks = []
    quantile_methods = [
        "inverted_cdf",
        "averaged_inverted_cdf",
        "closest_observation",
        "interpolated_inverted_cdf",
        "hazen",
        "weibull",
        "linear",
        "median_unbiased",
        "normal_unbiased",
        "lower",
        "higher",
        "midpoint",
        "nearest",
    ]
    for well, published in TABLE1.items():
        row = wells[(wells.Case == "idr0139") & (wells.Well == well)].iloc[0]
        values = all_cells.loc[
            (all_cells.Case == "idr0139") & (all_cells.Well == well), "Ratio"
        ]
        for method in quantile_methods:
            q25, q75 = np.percentile(values, [25, 75], method=method)
            quantile_checks.append(
                {
                    "Well": well,
                    "method": method,
                    "q25": q25,
                    "q75": q75,
                    "q25_matches_3dp": abs(q25 - published["q25"]) <= 0.0005,
                    "q75_matches_3dp": abs(q75 - published["q75"]) <= 0.0005,
                }
            )
        for statistic, value in published.items():
            if statistic == "table_index":
                continue
            calculated = float(row[statistic])
            comparison.append(
                {
                    "Well": well,
                    "Treatment": row.Treatment,
                    "statistic": statistic,
                    "published": value,
                    "reference": calculated,
                    "reference_rounded_3dp": round(calculated, 3),
                    "reference_minus_published": calculated - value,
                    "matches_published_precision": calculated == value
                    if statistic == "n_cells"
                    else abs(calculated - value) <= 0.0005,
                    "sem_of_fov_means": float(row.sem_of_fov_means)
                    if statistic == "sem_cell"
                    else None,
                }
            )
    comparison_frame = csv(output / "table1_comparison.csv", comparison)
    quantile_frame = csv(output / "table1_quantile_conventions.csv", quantile_checks)
    weibull = quantile_frame[quantile_frame.method == "weibull"]
    assert weibull.q25_matches_3dp.all() and weibull.q75_matches_3dp.all()
    sampling, validations = {}, {}
    for case, data in primary_by_case.items():
        print(f"Recomputing {case}: {len(data)} primary cells", flush=True)
        sampling[case] = sampling_tables(
            case, data, all_cells[all_cells.Case == case], output / case
        )
        validations[case] = verify_author_sampling(source, data, case)
    inputs = []
    for path in sorted(args.reference.rglob("*")):
        if path.is_file() and (
            path.suffix in {".csv", ".py", ".ipynb", ".cppipe"}
            or path.name == "manifest.json"
        ):
            inputs.append(
                {
                    "path": path.relative_to(args.reference).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    source_scan = {}
    for path in sorted((args.reference / "authors-code/notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = [
            {"cell_index": idx, "source": "".join(cell["source"])}
            for idx, cell in enumerate(notebook["cells"])
            if cell["cell_type"] == "code"
        ]
        csv(output / (path.stem + "_code_inventory.csv"), cells)
        source_scan[path.name] = {
            "code_cells": len(cells),
            "sem_or_describe_calls": [
                c["cell_index"]
                for c in cells
                if re.search(r"\.sem\(|\.describe\(|stats\.sem\(", c["source"])
            ],
        }
    acquisition = {}
    for case in ["idr0139", "idr0028"]:
        path = args.reference.parent / case / "manifest.json"
        acquisition[case] = {
            "manifest_sha256_at_audit": sha256(path),
            "manifest_path": str(path),
        }
    # Snapshot names from manifests. The acquisition report checks raw integrity.
    manifest0139 = json.loads(
        (args.reference.parent / "idr0139/manifest.json").read_text(encoding="utf-8")
    )
    acquired_names = {Path(item["path"]).name for item in manifest0139["files"]}
    subset_images = images[images.Case == "idr0139"]
    acquisition["idr0139"]["reference_nuclear_files_in_manifest"] = int(
        subset_images.FileName.isin(acquired_names).sum()
    )
    acquisition["idr0139"]["reference_nuclear_files_total"] = len(subset_images)
    acquisition["idr0139"]["missing_nuclear_filenames_at_audit"] = subset_images.loc[
        ~subset_images.FileName.isin(acquired_names), "FileName"
    ].tolist()
    manifest0028 = pd.read_csv(args.reference.parent / "idr0028/image_manifest.csv")
    observed = set(
        zip(
            manifest0028.plate.str.replace("LM2_GEFGAP_ONTARGETPlus_", "", regex=False),
            manifest0028.path.map(lambda x: Path(x).name),
            strict=False,
        )
    )
    expected = set(
        zip(
            images.loc[images.Case == "idr0028", "Plate"],
            images.loc[images.Case == "idr0028", "FileName"],
            strict=False,
        )
    )
    assert observed == expected
    acquisition["idr0028"]["all_336_reference_files_in_manifest"] = True
    audit = {
        "created_utc": datetime.now(UTC).isoformat(),
        "paper": PAPER,
        "author_commit": AUTHOR_COMMIT,
        "author_dataset_record": "https://zenodo.org/records/15584545",
        "scope": (
            "Authors' deposited measurements and notebook numerical recipes only. No "
            "image resegmentation or VIPP agreement claim."
        ),
        "grain": (
            "One nucleus/cytoplasm object pair per "
            "Case,Plate,ImageNumber,ObjectNumber; fields nested in wells; wells "
            "nested in plates."
        ),
        "ratio": (
            "nuclear compartment mean / (cytoplasmic compartment mean + nuclear "
            "compartment mean)"
        ),
        "join": (
            "Nuclei left join Cytoplasm on ImageNumber,ObjectNumber; left join Image "
            "on ImageNumber. No cross-plate joins."
        ),
        "filters": (
            "Only selected wells and plate. No applied QC, cell-area, zero-intensity "
            "or outlier filter in authors' preparation helpers."
        ),
        "statistics": {
            "quantiles": "linear interpolation",
            "sd": "pandas sample SD, ddof=1",
            "sem_cell": (
                "sample SD/sqrt(number of finite cell values); descriptive only, not "
                "biological-replicate SEM"
            ),
            "fov_sem": (
                "sample SD of unweighted field means/sqrt(number of fields); fields "
                "are not independent biological replicates"
            ),
        },
        "table1_mismatches": comparison_frame.loc[
            ~comparison_frame.matches_published_precision
        ]
        .drop(columns=["sem_of_fov_means"])
        .to_dict("records"),
        "table1_sem_conclusion": (
            "No SEM-generating code in available notebook cells/helper. All four "
            "reported SEM values disagree with both cell SEM and SEM of four FOV "
            "means. Changing ddof=1 to ddof=0 cannot reconcile them. Unresolved "
            "published-reference discrepancy; never use reported SEM as validation "
            "ground truth."
        ),
        "table1_rounding": (
            "All counts, means, medians and SD agree at displayed precision. "
            "Notebook-default linear quantiles differ in two places: J05 "
            "q75=.546433587 vs published .547; E22 q25=.448801630 vs published .448. "
            "All eight Table1 quartiles agree at three decimals using "
            "Weibull/exclusive percentiles. This is compatible with a different "
            "quantile convention, not evidence of a numerical error; Table1 "
            "generation code is unavailable."
        ),
        "table1_quantile_convention_check": {
            "methods_checked": quantile_methods,
            "all_eight_quartiles_match_using_weibull": True,
            "notebook_method": "linear",
            "table1_actual_method": (
                "not documented; Weibull compatibility is an inference"
            ),
        },
        "replicate_scope": {
            "idr0139": (
                "Ten wells on one plate: three each Untreated, DMSO, Leptomycin b; "
                "one SN0212398523. Primary figures/Table1 use one well per condition."
            ),
            "idr0028": (
                "Twelve wells across three plates. Three wells per condition, but "
                "not one well per condition per plate: LATS1 occurs twice on 1A, "
                "once 2A; MOCK once1A,once2A,once2B; ARAP2 once2A,twice2B; YAP "
                "twice1A,once2A. Primary figures use four wells on2A."
            ),
        },
        "rng": {
            "backend": (
                "pandas.sample(replace=False, random_state=integer), NumPy "
                "RandomState path; input row order preserved from source merges then "
                "Treatment,Well sorting"
            ),
            "swarm": (
                "n50 seeds42,43,44; n200 seeds45,46,47; same seed separately in each "
                "condition per panel"
            ),
            "superplots": (
                "n50,200,500 per well, seed42 restarted for each well and each "
                "sample size; means per well; boxplot over well means"
            ),
            "iqr": (
                "n10..490 step10;100 iterations;seed42 increments once per "
                "iteration, same seed separately per condition; plotted "
                "error=max(IQR)-min(IQR) over100 samples, exponential a*exp(-b*n)+c "
                "fit"
            ),
            "effect": (
                "n10..490 step10;100 iterations;seed42 increments per treatment "
                "comparison; same seed for treatment and control within comparison; "
                "(sample_mean_treatment-sample_mean_control)/sample_sd_control, then "
                "median and25th/75th percentiles over100 trials; Glass-type "
                "standardized difference, not pooled-SD Cohen d"
            ),
            "cumulative": (
                "Add ten previously unselected cells at each n10..500; seed42 "
                "increments each batch; histogram n20,50,100,200,300,500 with50 "
                "bins,density=True; signed differences from full control-well "
                "statistics"
            ),
        },
        "additional_audit_notes": [
            (
                "Author requirements are unpinned; exact historical NumPy/pandas "
                "environment is unavailable. Runtime versions recorded below."
            ),
            (
                "Histogram y label says Frequency (%) but density=True produces "
                "probability density, not percentages."
            ),
            (
                "IDR0028 effect-size y label says relative to Untreated although "
                "actual code control is MOCK."
            ),
            (
                "Kruskal-Wallis/Dunn exist as optional helper branches; primary "
                "notebooks never enable p_values=True; no inferential p-values "
                "claimed here."
            ),
            (
                "The separate generic framework notebook targets newer "
                "Zenodo15727751 and tidy_data/tidy_idr0028_data.csv, absent from "
                "pinned15584545; it cannot run against this pinned subset unchanged. "
                "This audit follows the two study-specific notebooks."
            ),
            (
                "Paper subtraction .579-.488=.091 quotes uncertainty .041, equal to "
                "arithmetic sum .024+.017. This audit records but does not endorse "
                "that as an independently justified uncertainty estimate."
            ),
        ],
        "join_and_quality_checks": checks,
        "sampling": sampling,
        "author_loop_crosschecks": validations,
        "notebook_scan": source_scan,
        "acquisition_snapshot": acquisition,
        "source_inputs": inputs,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy"]
            },
        },
        "script_sha256": sha256(Path(__file__)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output / "audit.json", audit)
    write_json(
        output / "descriptive_tables.json",
        {
            "primary_treatment_summary": group_summaries(
                primary, ["Case", "Treatment"]
            ).to_dict("records"),
            "well_summary": wells.to_dict("records"),
            "table1_comparison": comparison,
            "all_wells_treatment_summary": group_summaries(
                all_cells, ["Case", "Treatment"]
            ).to_dict("records"),
        },
    )
    # Fixed recipe, compact notebook companion; execution lives in the script.
    notebook_arguments = [
        str(Path(__file__).resolve()),
        "--reference",
        str(args.reference.resolve()),
        "--output",
        str(output.resolve()),
    ]
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {
                "id": "purpose",
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    (
                        "# Reference-only audit\nRecompute the authors' deposited "
                        "measurements, not VIPP segmentation. See audit.json for "
                        "provenance, exact sampling recipes, and published SEM "
                        "discrepancies. Execute from this directory.\n"
                    )
                ],
            },
            {
                "id": "run",
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    (
                        "import subprocess, sys\n"
                        f"subprocess.run([sys.executable] + {notebook_arguments!r}, "
                        "check=True)\n"
                    )
                ],
            },
            {
                "id": "inspect",
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    (
                        "import pandas as "
                        "pd\npd.read_csv('primary_treatment_summary.csv')\n"
                    )
                ],
            },
        ],
    }
    write_json(output / "reference_audit.ipynb", notebook)
    print(
        json.dumps(
            {
                "output": str(output),
                "cells": len(all_cells),
                "images": len(images),
                "wells": len(wells),
                "elapsed_seconds": audit["elapsed_seconds"],
                "table1_discrepancies": len(audit["table1_mismatches"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
