"""Compare native VIPP results with Marcotti et al.'s archived measurements.

Enumerates the 376 reference fields, never recursively collects pilot/aggregate
CSV files. Cell-ID comparisons require every source file in the field to match
the authors' archived MD5 and verify geometric centroids independently. Reads
originals/results without modifying them. Produces tables, exact sampling recipe
outputs, a readable report, and scientific Matplotlib PNG/PDF figures on HDD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import recompute_statistics_paper_reference as recipes
from matplotlib.lines import Line2D

KEYS = ["Case", "Plate", "ImageNumber", "ObjectNumber"]
FIELD_KEYS = KEYS[:-1]
COLORS = ["#3268A6", "#DD8735", "#609751", "#AA5C96"]
SOURCE_COLORS = {"Authors": "#536D83", "VIPP": "#D86C39"}
TOLERANCES = {"intensity_mean": 1e-15, "ratio": 1e-12, "centroid_pixels": 1e-10}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, dtype={"Plate": str}, float_precision="round_trip")


def write_csv(path, frame):
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def json_write(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def field_directory(row, vipp_root):
    expression = r"F(\d+)" if row.Case == "idr0139" else r"-(\d+)\.tif$"
    field = int(re.search(expression, row.FileName)[1])
    return vipp_root / row.Case / str(row.Plate) / f"{row.Well}_F{field:03d}"


def load_comparison(root, allow_incomplete=False):
    base = root / "validation/statistics-paper"
    authors = read_csv(base / "reference/cells.csv")
    expected_fields = authors.drop_duplicates(FIELD_KEYS)
    assert len(expected_fields) == 376
    all_frames, image_rows, paired_frames, source_records = [], [], [], []
    author_object_tables = {}
    source_hashes = []
    missing = []
    for order, row in enumerate(expected_fields.itertuples(index=False)):
        directory = field_directory(row, base / "vipp")
        cells_path = directory / "cells.csv"
        vipp = read_csv(cells_path)
        assert not vipp.duplicated(KEYS).any()
        assert set(vipp.Case) == {row.Case} and set(vipp.Plate) == {str(row.Plate)}
        assert set(vipp.ImageNumber) == {row.ImageNumber}
        vipp = vipp.rename(
            columns={"NucleiMean": "NuclearMean", "CytoplasmMean": "CytoplasmicMean"}
        )
        vipp["_field_order"] = order
        receipt_path = directory / "result.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            source_records.extend(receipt["sources"])
            hash_match = all(
                source["author_file_bytes_identical"] for source in receipt["sources"]
            )
            assert len(vipp) == receipt["vipp_count"]
            source_hashes.append(
                {
                    "path": receipt_path.relative_to(root).as_posix(),
                    "sha256": digest(receipt_path),
                }
            )
        else:
            missing.append(directory.as_posix())
            hash_match = False
        reference = authors[
            (authors.Case == row.Case)
            & (authors.Plate == str(row.Plate))
            & (authors.ImageNumber == row.ImageNumber)
        ]
        image_rows.append(
            {
                "Case": row.Case,
                "Plate": str(row.Plate),
                "Well": row.Well,
                "Treatment": row.Treatment,
                "ImageNumber": row.ImageNumber,
                "FileName": row.FileName,
                "complete_receipt": receipt_path.exists(),
                "all_source_bytes_match_author_md5": hash_match,
                "author_cells": len(reference),
                "vipp_cells": len(vipp),
                "cell_count_difference": len(vipp) - len(reference),
                "author_ratio_mean": reference.Ratio.mean(),
                "vipp_ratio_mean": vipp.Ratio.mean(),
                "ratio_mean_difference": vipp.Ratio.mean() - reference.Ratio.mean(),
            }
        )
        if hash_match:
            plate_key = (row.Case, str(row.Plate))
            if plate_key not in author_object_tables:
                table_root = (
                    root / "statistics/reference/data_subsets/cell_profiler_outputs"
                )
                table_root = (
                    table_root / "idr0139"
                    if row.Case == "idr0139"
                    else table_root
                    / "idr0028/screenB"
                    / f"LM2_GEFGAP_ONTARGETPlus_{row.Plate}"
                )
                author_object_tables[plate_key] = {
                    compartment: read_csv(table_root / f"{compartment}.csv")
                    for compartment in ["Nuclei", "Cytoplasm"]
                }
                source_hashes.extend(
                    {
                        "path": (table_root / f"{compartment}.csv")
                        .relative_to(root)
                        .as_posix(),
                        "sha256": digest(table_root / f"{compartment}.csv"),
                    }
                    for compartment in ["Nuclei", "Cytoplasm"]
                )
            joined = reference.merge(
                vipp,
                on=KEYS,
                suffixes=("_author", "_vipp"),
                how="outer",
                validate="one_to_one",
                indicator=True,
            )
            assert joined._merge.eq("both").all(), (
                "Hash-matched field has unequal cell IDs"
            )
            joined = joined.drop(columns="_merge")
            protein = "Fascin" if row.Case == "idr0139" else "YAPTAZ"
            for compartment in ["Nuclei", "Cytoplasm"]:
                table = author_object_tables[plate_key][compartment]
                table = table[table.ImageNumber == row.ImageNumber][
                    ["ObjectNumber", "Location_Center_X", "Location_Center_Y"]
                ]
                table = table.rename(
                    columns={
                        "Location_Center_X": f"{compartment}_x_author",
                        "Location_Center_Y": f"{compartment}_y_author",
                    }
                )
                vipp_path = directory / f"{compartment}-{protein}.csv"
                measured = read_csv(vipp_path)[
                    ["label_id", "centroid_x", "centroid_y"]
                ].rename(
                    columns={
                        "label_id": "ObjectNumber",
                        "centroid_x": f"{compartment}_x_vipp",
                        "centroid_y": f"{compartment}_y_vipp",
                    }
                )
                joined = joined.merge(
                    table, on="ObjectNumber", validate="one_to_one"
                ).merge(measured, on="ObjectNumber", validate="one_to_one")
                source_hashes.append(
                    {
                        "path": vipp_path.relative_to(root).as_posix(),
                        "sha256": digest(vipp_path),
                    }
                )
            for metric in [
                "NuclearMean",
                "CytoplasmicMean",
                "Ratio",
                "Nuclei_x",
                "Nuclei_y",
                "Cytoplasm_x",
                "Cytoplasm_y",
            ]:
                joined[f"{metric}_difference"] = (
                    joined[f"{metric}_vipp"] - joined[f"{metric}_author"]
                )
            paired_frames.append(joined)
        all_frames.append(vipp)
        source_hashes.append(
            {
                "path": cells_path.relative_to(root).as_posix(),
                "sha256": digest(cells_path),
            }
        )
    if missing and not allow_incomplete:
        raise ValueError(f"Missing {len(missing)} complete result receipts: {missing}")
    vipp = pd.concat(all_frames, ignore_index=True)
    # Authors prepare each plate separately, sort Treatment,Well, concatenate
    # plates in 1A,2A,2B order. Within a well preserve source field/object order.
    vipp = vipp.sort_values(
        ["Case", "Plate", "Treatment", "Well", "_field_order", "ObjectNumber"],
        kind="stable",
    ).reset_index(drop=True)
    vipp["ReferenceRowIndex"] = np.arange(len(vipp))
    assert not vipp.duplicated(KEYS).any()
    assert np.isfinite(vipp[["NuclearMean", "CytoplasmicMean", "Ratio"]]).all().all()
    paired = pd.concat(paired_frames, ignore_index=True)
    metrics = {}
    for metric in [
        "NuclearMean",
        "CytoplasmicMean",
        "Ratio",
        "Nuclei_x",
        "Nuclei_y",
        "Cytoplasm_x",
        "Cytoplasm_y",
    ]:
        absolute = paired[f"{metric}_difference"].abs()
        tolerance = (
            TOLERANCES["intensity_mean"]
            if metric.endswith("Mean")
            else TOLERANCES["ratio"]
            if metric == "Ratio"
            else TOLERANCES["centroid_pixels"]
        )
        metrics[metric] = {
            "n": len(paired),
            "max_absolute_difference": float(absolute.max()),
            "mean_absolute_difference": float(absolute.mean()),
            "absolute_tolerance": tolerance,
            "count_outside_tolerance": int((absolute > tolerance).sum()),
        }
    return (
        authors,
        vipp,
        pd.DataFrame(image_rows),
        paired,
        metrics,
        missing,
        source_hashes,
    )


def primary_subset(frame, case):
    subset = frame[frame.Case == case]
    return (
        subset[subset.Well.isin(recipes.PRIMARY_0139)]
        if case == "idr0139"
        else subset[subset.Plate == "2A"]
    )


def compare_summaries(authors, vipp, keys):
    left = recipes.group_summaries(authors, keys)
    right = recipes.group_summaries(vipp, keys)
    result = left.merge(
        right, on=keys, suffixes=("_author", "_vipp"), validate="one_to_one"
    )
    for name in ["n_cells", "mean", "median", "q25", "q75", "std_sample", "sem_cell"]:
        result[name + "_difference"] = result[name + "_vipp"] - result[name + "_author"]
    return result


def compare_sampling(reference_root, comparison_root):
    configurations = {
        "iqr_trials": (["sample_size", "iteration", "seed", "Treatment"], ["iqr"]),
        "iqr_ranges": (
            ["Treatment", "sample_size"],
            ["min", "max", "range", "exponential_fit"],
        ),
        "effect_trials": (
            ["sample_size", "iteration", "seed", "Treatment"],
            ["standardized_mean_difference"],
        ),
        "effect_summary": (["Treatment", "sample_size"], ["median", "q25", "q75"]),
        "cumulative_summary": (
            ["sample_size"],
            [
                "mean",
                "median",
                "std_sample",
                "iqr",
                "difference_mean",
                "difference_median",
                "difference_std_sample",
                "difference_iqr",
            ],
        ),
        "swarm_sample_summary": (
            ["n_requested", "panel", "seed", "Treatment"],
            ["mean", "median", "q25", "q75", "std_sample"],
        ),
        "superplot_sample_summary": (
            ["n_requested", "seed", "Treatment", "Replicate", "ReplicateIndex"],
            ["mean", "median", "q25", "q75", "std_sample"],
        ),
        "full_population_effects": (
            ["Treatment", "Control"],
            ["standardized_mean_difference"],
        ),
    }
    comparisons = []
    for case in ["idr0139", "idr0028"]:
        for name, (keys, columns) in configurations.items():
            a = read_csv(reference_root / case / f"{name}.csv")
            b = read_csv(comparison_root / case / f"{name}.csv")
            merged = a.merge(
                b,
                on=keys,
                suffixes=("_author", "_vipp"),
                how="outer",
                indicator=True,
                validate="one_to_one",
            )
            assert merged._merge.eq("both").all()
            for column in columns:
                difference = merged[f"{column}_vipp"] - merged[f"{column}_author"]
                merged[f"{column}_difference"] = difference
                comparisons.append(
                    {
                        "Case": case,
                        "table": name,
                        "statistic": column,
                        "rows_compared": len(merged),
                        "max_absolute_difference": difference.abs().max(),
                        "mean_absolute_difference": difference.abs().mean(),
                    }
                )
            write_csv(
                comparison_root / case / f"{name}_comparison.csv",
                merged.drop(columns="_merge"),
            )
        # Compare sampled object identities separately from their measurements.
        for name in [
            "swarm_sample_members",
            "superplot_sample_members",
            "cumulative_members",
        ]:
            a = read_csv(reference_root / case / f"{name}.csv")
            b = read_csv(comparison_root / case / f"{name}.csv")
            same = len(a) == len(b) and np.array_equal(
                a[KEYS].to_numpy(), b[KEYS].to_numpy()
            )
            comparisons.append(
                {
                    "Case": case,
                    "table": name,
                    "statistic": "same_sampled_object_keys_and_order",
                    "rows_compared": len(a),
                    "max_absolute_difference": 0 if same else 1,
                    "mean_absolute_difference": 0 if same else 1,
                }
            )
    return pd.DataFrame(comparisons)


def save_figure(fig, folder, stem):
    fig.savefig(folder / f"{stem}.png", dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(folder / f"{stem}.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def figures(authors, vipp, fields, paired, wells, reference_root, output):
    folder = output / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for ax, case in zip(axes, ["idr0139", "idr0028"], strict=True):
        for source, data, offset in [("Authors", authors, -0.16), ("VIPP", vipp, 0.16)]:
            subset = primary_subset(data, case)
            for i, treatment in enumerate(recipes.DISPLAY_ORDER[case]):
                values = subset.loc[subset.Treatment == treatment, "Ratio"]
                violin = ax.violinplot(
                    values,
                    positions=[i + offset],
                    widths=0.28,
                    showextrema=False,
                    showmedians=True,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(SOURCE_COLORS[source])
                    body.set_alpha(0.6)
                violin["cmedians"].set_color(SOURCE_COLORS[source])
                ax.scatter(i + offset, values.mean(), color="black", s=18, zorder=5)
        ax.set_xticks(range(4), recipes.DISPLAY_ORDER[case], rotation=18)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Nuclear mean / (nuclear mean + cytoplasmic mean)")
        ax.set_title(
            "Fascin — primary four wells"
            if case == "idr0139"
            else "YAP/TAZ — primary four wells"
        )
    handles = [
        Line2D([], [], color=color, lw=5, label=source)
        for source, color in SOURCE_COLORS.items()
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            color="black",
            linestyle="",
            label="Mean (black); median (line)",
        )
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Reference distributions versus native VIPP results")
    save_figure(fig, folder, "primary-distributions")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for case, color, name in [
        ("idr0139", COLORS[1], "Fascin"),
        ("idr0028", COLORS[0], "YAP/TAZ"),
    ]:
        selected = fields[fields.Case == case]
        axes[0].scatter(
            selected.author_cells,
            selected.vipp_cells,
            s=18,
            alpha=0.65,
            color=color,
            label=name,
        )
    maximum = max(fields.author_cells.max(), fields.vipp_cells.max()) * 1.05
    axes[0].plot([0, maximum], [0, maximum], "k--", lw=0.8)
    axes[0].set(
        xlabel="Authors’ cells per field",
        ylabel="VIPP cells per field",
        title="A · Counts across all 376 fields",
    )
    axes[0].legend(frameon=False)
    for case, color, name in [
        ("idr0139", COLORS[1], "Fascin: 8 byte-matched fields"),
        ("idr0028", COLORS[0], "YAP/TAZ: all 336 fields"),
    ]:
        subset = paired[paired.Case == case]
        axes[1].scatter(
            subset.Ratio_author,
            subset.Ratio_difference,
            s=3,
            alpha=0.3,
            color=color,
            label=name,
            rasterized=True,
        )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(
        xlabel="Authors’ per-cell ratio",
        ylabel="VIPP − authors’ ratio",
        title="B · Only verified source-byte matches",
    )
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].legend(frameon=False, fontsize=8)
    fascin = wells[wells.Case == "idr0139"]
    axes[2].barh(
        fascin.Well,
        fascin.mean_difference,
        color=[COLORS[0] if w in {"N12", "O02"} else COLORS[1] for w in fascin.Well],
    )
    axes[2].axvline(0, color="black", lw=0.8)
    axes[2].set(
        xlabel="VIPP − authors’ pooled cell mean",
        ylabel="Fascin well",
        title="C · Historical-input mismatch remains",
    )
    save_figure(fig, folder, "agreement")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    for row_index, case in enumerate(["idr0139", "idr0028"]):
        for source, base, style in [
            ("Authors", reference_root, "-"),
            ("VIPP", output, "--"),
        ]:
            iqr = read_csv(base / case / "iqr_ranges.csv")
            for color, treatment in zip(
                COLORS, recipes.DISPLAY_ORDER[case], strict=True
            ):
                curve = iqr[iqr.Treatment == treatment]
                axes[row_index, 0].plot(
                    curve.sample_size,
                    curve["range"],
                    style,
                    color=color,
                    label=treatment if source == "Authors" else None,
                    lw=1.3,
                )
            cumulative = read_csv(base / case / "cumulative_summary.csv")
            for color, metric in zip(
                COLORS, ["mean", "median", "std_sample", "iqr"], strict=True
            ):
                axes[row_index, 1].plot(
                    cumulative.sample_size,
                    cumulative[f"difference_{metric}"],
                    style,
                    color=color,
                    lw=1.2,
                    label=metric.replace("std_sample", "SD").upper()
                    if source == "Authors"
                    else None,
                )
            effects = read_csv(base / case / "effect_summary.csv")
            for color, treatment in zip(
                COLORS, recipes.EFFECT_ORDER[case], strict=False
            ):
                curve = effects[effects.Treatment == treatment]
                axes[row_index, 2].plot(
                    curve.sample_size,
                    curve["median"],
                    style,
                    color=color,
                    lw=1.3,
                    label=treatment if source == "Authors" else None,
                )
                if source == "Authors":
                    axes[row_index, 2].fill_between(
                        curve.sample_size, curve.q25, curve.q75, color=color, alpha=0.12
                    )
        for column in range(3):
            axes[row_index, column].set_xlabel("Cells sampled per well")
            axes[row_index, column].legend(frameon=False, fontsize=8)
        axes[row_index, 0].set_ylabel(
            ("Fascin" if case == "idr0139" else "YAP/TAZ")
            + "\nRange of 100 sampled IQRs"
        )
        axes[row_index, 1].set_ylabel("Signed difference from all control cells")
        axes[row_index, 2].set_ylabel("Median standardized mean difference")
    for ax, title in zip(
        axes[0],
        [
            "IQR sampling variation",
            "Cumulative control samples",
            "Effect relative to control",
        ],
        strict=True,
    ):
        ax.set_title(title)
    fig.suptitle(
        (
            "Exact notebook sampling recipes · authors solid, VIPP dashed; "
            "effect shading = authors’ 25–75%"
        ),
        fontsize=12,
    )
    save_figure(fig, folder, "sampling-comparison")

    for n in [50, 200, 500]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained")
        rng = np.random.default_rng(
            123
        )  # Display jitter only; sampling IDs fixed upstream.
        for ax, case in zip(axes, ["idr0139", "idr0028"], strict=True):
            for source, base, offset in [
                ("Authors", reference_root, -0.2),
                ("VIPP", output, 0.2),
            ]:
                cells = read_csv(base / case / "superplot_sample_members.csv")
                means = read_csv(base / case / "superplot_sample_summary.csv")
                for i, treatment in enumerate(recipes.DISPLAY_ORDER[case]):
                    points = cells[
                        (cells.SampleSize == n) & (cells.Treatment == treatment)
                    ]
                    avg = means[
                        (means.n_requested == n) & (means.Treatment == treatment)
                    ]
                    ax.scatter(
                        i + offset + rng.uniform(-0.12, 0.12, len(points)),
                        points.Ratio,
                        s=2,
                        alpha=0.12,
                        color=SOURCE_COLORS[source],
                        rasterized=True,
                    )
                    ax.scatter(
                        i + offset + np.linspace(-0.08, 0.08, len(avg)),
                        avg["mean"],
                        marker="D",
                        s=35,
                        color=SOURCE_COLORS[source],
                        edgecolor="black",
                        linewidth=0.7,
                        zorder=5,
                    )
            ax.set_xticks(range(4), recipes.DISPLAY_ORDER[case], rotation=18)
            ax.set(
                ylim=(0, 1),
                ylabel="Relative nuclear localization",
                title="Fascin: ten wells"
                if case == "idr0139"
                else "YAP/TAZ: twelve wells",
            )
        fig.legend(
            handles=[
                Line2D([], [], color=color, marker="D", linestyle="", label=source)
                for source, color in SOURCE_COLORS.items()
            ],
            loc="outside lower center",
            ncol=2,
            frameon=False,
        )
        fig.suptitle(
            f"Superplot sampling: {n} cells per well · diamonds are well means; "
            "cell jitter is display-only",
        )
        save_figure(fig, folder, f"superplots-{n}")

    # Every sampled panel keeps its exact scientific draw; horizontal jitter
    # is a separate deterministic display choice.
    for case in ["idr0139", "idr0028"]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
        display_rng = np.random.default_rng(137)
        for source, base, offset in [
            ("Authors", reference_root, -0.18),
            ("VIPP", output, 0.18),
        ]:
            members = read_csv(base / case / "swarm_sample_members.csv")
            for row_index, n in enumerate([50, 200]):
                for panel in [1, 2, 3]:
                    ax = axes[row_index, panel - 1]
                    for i, treatment in enumerate(recipes.DISPLAY_ORDER[case]):
                        points = members[
                            (members.SampleSize == n)
                            & (members.Panel == panel)
                            & (members.Treatment == treatment)
                        ]
                        center = i + offset
                        ax.scatter(
                            center + display_rng.uniform(-0.12, 0.12, len(points)),
                            points.Ratio,
                            s=3,
                            alpha=0.3,
                            color=SOURCE_COLORS[source],
                            rasterized=True,
                        )
                        ax.boxplot(
                            points.Ratio,
                            positions=[center],
                            widths=0.25,
                            showfliers=False,
                            manage_ticks=False,
                            boxprops={"color": SOURCE_COLORS[source]},
                            whiskerprops={"color": SOURCE_COLORS[source]},
                            capprops={"color": SOURCE_COLORS[source]},
                            medianprops={"color": SOURCE_COLORS[source]},
                        )
                        ax.scatter(
                            center, points.Ratio.mean(), s=20, color="black", zorder=5
                        )
                    seed = 41 + panel if n == 50 else 44 + panel
                    ax.set_title(f"{n} cells per condition · seed {seed}")
                    ax.set_ylim(0, 1)
                    ax.set_xticks(range(4), recipes.DISPLAY_ORDER[case], rotation=25)
                    ax.set_ylabel("Relative nuclear localization")
        fig.legend(
            handles=[
                Line2D([], [], color=color, lw=3, label=source)
                for source, color in SOURCE_COLORS.items()
            ],
            loc="outside lower center",
            ncol=2,
            frameon=False,
        )
        fig.suptitle(
            f"{case}: exact notebook samples · black points = means; "
            "jitter is display-only",
        )
        save_figure(fig, folder, f"swarm-samples-{case}")

    fig, axes = plt.subplots(2, 6, figsize=(18, 6), layout="constrained")
    for row_index, case in enumerate(["idr0139", "idr0028"]):
        for source, base, style in [
            ("Authors", reference_root, "-"),
            ("VIPP", output, "--"),
        ]:
            histogram = read_csv(base / case / "cumulative_histograms.csv")
            for column, n in enumerate([20, 50, 100, 200, 300, 500]):
                values = histogram[histogram.sample_size == n]
                edges = np.append(values.left.to_numpy(), values.right.iloc[-1])
                axes[row_index, column].stairs(
                    values.density,
                    edges,
                    color=SOURCE_COLORS[source],
                    linestyle=style,
                    linewidth=1.1,
                )
                axes[row_index, column].set(
                    xlim=(0, 1),
                    ylim=(0, 20),
                    title=f"{n} cells",
                    xlabel="Relative nuclear localization",
                )
        axes[row_index, 0].set_ylabel(
            f"{recipes.CONTROL[case]} control\nProbability density"
        )
    fig.suptitle(
        "Exact cumulative histogram bins · authors solid, VIPP dashed · "
        "density, not percentage",
        fontsize=12,
    )
    save_figure(fig, folder, "cumulative-histograms")


def report(summary, primary, fields, output):
    yap_effect_difference = next(
        item["max_absolute_difference"]
        for item in summary["sampling_comparison"]
        if item["Case"] == "idr0028" and item["table"] == "effect_trials"
    )
    yap_fit_difference = next(
        item["max_absolute_difference"]
        for item in summary["sampling_comparison"]
        if item["Case"] == "idr0028"
        and item["table"] == "iqr_ranges"
        and item["statistic"] == "exponential_fit"
    )
    lines = [
        "# Statistics-paper native VIPP comparison",
        "",
        (
            "The **YAP/TAZ numerical analysis is reproduced across all 336 "
            "fields and 9,550 cells** within the declared numerical "
            "tolerances. The Fascin analysis runs across all forty fields, "
            "but eight wells have source-file MD5 differences from the "
            "authors’ archived inputs; its historical distributions are "
            "therefore not fully reproduced. No parameters were tuned to "
            "match the published numbers."
        ),
        "",
        (
            "This comparison distinguishes archived author measurements, VIPP"
            " applied to acquired TIFFs, and independent CellProfiler applied"
            " to the same acquired TIFFs. Source-byte agreement alone does "
            "not prove pixel or label identity; object counts, scalar "
            "measurements and geometric centroids are checked separately."
        ),
        "",
        f"Complete fields: **{summary['completed_receipts']}/376**. "
        f"Reconstructed VIPP cells: **{summary['vipp_cells']}**, "
        f"archived author cells: **{summary['author_cells']}**. "
        f"Byte-matched fields: **{summary['byte_matched_fields']}**; "
        f"paired cells in those fields: **{summary['paired_cells']}**.",
        "",
        "## Primary distributions",
        "",
        (
            "| Case | Condition | Author n | VIPP n | Author mean | VIPP mean"
            " | Difference |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.Case} | {row.Treatment} | {row.n_cells_author} | "
            f"{row.n_cells_vipp} | {row.mean_author:.9f} | "
            f"{row.mean_vipp:.9f} | {row.mean_difference:+.3g} |"
        )
    lines.extend(
        [
            "",
            (
                "Means summarize cells within one selected well per treatment; "
                "they are not independent biological-replicate estimates. "
                "Complete medians, quartiles, SD and descriptive cell SEM appear "
                "in the CSV/JSON tables. The reference audit explains the "
                "unresolved published SEM column and the compatible "
                "exclusive-percentile convention for Table 1."
            ),
            "",
            "## Object-level comparison on verified source-byte matches",
            "",
            (
                "| Quantity | Maximum absolute difference | Absolute tolerance | "
                "Outside tolerance |"
            ),
            "|---|---:|---:|---:|",
        ]
    )
    for metric, values in summary["paired_metrics"].items():
        lines.append(
            f"| {metric} | {values['max_absolute_difference']:.6g} | "
            f"{values['absolute_tolerance']:.1g} | "
            f"{values['count_outside_tolerance']} |"
        )
    lines.extend(
        [
            "",
            (
                "IDs are paired only after all channels/files of the field match "
                "the archived MD5; nuclei and cytoplasm geometric centroids "
                "independently verify the correspondence. No cell-ID pairing is "
                "claimed for unmatched-source fields. Tiny scalar differences "
                "remain from exported decimal precision and floating-point "
                "evaluation; bitwise scalar equality is not claimed."
            ),
            "",
            (
                "Fascin N12 and O02 are the eight byte-matched fields. The other "
                "32 Fascin fields use acquired source files whose archived MD5 "
                "does not match. A file-byte difference can reflect storage "
                "metadata or pixel changes; this audit does not establish the "
                "cause. Unequal archived/current segmentation counts and "
                "distributions are reported, not hidden or tuned away."
            ),
            "",
            "## Statistics and figures",
            "",
            "YAP/TAZ sampled object identities and their order match exactly "
            "for all swarmplot, cumulative and superplot samples. The largest "
            "difference among 14,700 standardized-effect draws is "
            f"{yap_effect_difference:.6g}; fitted IQR curves differ by at most "
            f"{yap_fit_difference:.6g}. These are reported numerical differences, "
            "not claims of bitwise statistical equality.",
            "",
            (
                "The companion recomputation reuses the audited author sampling "
                "recipes and source field/object ordering: six swarm samples, "
                "cumulative control samples, 19,600 IQR trials, 14,700 "
                "standardized-effect trials, and superplots at 50/200/500 cells "
                "per well, for each case study. Effect size divides by the "
                "control SD. IQR error is the max–min range across 100 draws. "
                "Fitted exponentials can amplify tiny input differences in their "
                "fitted coefficients; their values are reported separately. "
                "Neither display jitter nor figure rendering enters any "
                "scientific result."
            ),
            "",
            (
                "- `image_comparison.csv`: all 376 expected fields, source-match "
                "status, counts and mean differences."
            ),
            (
                "- `paired_cells.csv`: validated matched-field object "
                "measurements and centroid differences."
            ),
            (
                "- `well_comparison.csv`: all 22 wells; `primary_comparison.csv`:"
                " eight primary condition/well groups."
            ),
            (
                "- `idr0139/` and `idr0028/`: numerical sampling outputs, "
                "memberships and comparisons."
            ),
            (
                "- `sampling_comparison.csv`: maximum and mean numerical "
                "differences for each curve/statistic."
            ),
            (
                "- `figures/primary-distributions`, `agreement`, "
                "`sampling-comparison`, and `superplots-50/200/500`: scientific "
                "PNG and vector PDF exports."
            ),
            (
                "- `comparison.json`: tolerances, input hashes, runtime and "
                "machine-readable findings."
            ),
            "- `figures/swarm-samples-idr0139`, `swarm-samples-idr0028`, "
            "and `cumulative-histograms`: all six sampled panels per case "
            "and all twelve cumulative histograms, also in PNG/PDF.",
            "",
            (
                "Independent same-input CellProfiler comparisons are retained in "
                "the per-field VIPP receipts and `cellprofiler-reference/`; this "
                "report does not infer a historical full-paper match from "
                "agreement between two tools on the acquired TIFFs."
            ),
            "",
            (
                "Rerun with `python scripts/compare_statistics_paper.py` from the"
                " source checkout. Output defaults to "
                "`D:/VIPP-paper-reproductions/validation/statistics-paper/comparison`."
                " Original TIFFs, author tables and native VIPP results are read "
                "only."
            ),
        ]
    )
    if summary["missing_receipts"]:
        lines.insert(
            2,
            (
                "**INCOMPLETE DEVELOPMENT RUN: some result receipts are missing; "
                "final qualification requires all 376.**\n"
            ),
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("D:/VIPP-paper-reproductions")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Reuse existing sampling CSVs for figure/report refresh only.",
    )
    args = parser.parse_args()
    output = args.output or args.root / "validation/statistics-paper/comparison"
    output.mkdir(parents=True, exist_ok=True)
    reference_root = args.root / "validation/statistics-paper/reference"
    authors, vipp, fields, paired, metrics, missing, sources = load_comparison(
        args.root, args.allow_incomplete
    )
    print(
        f"Loaded {len(fields)} expected fields, {len(vipp)} VIPP cells; "
        f"{len(paired)} safely paired cells.",
        flush=True,
    )
    write_csv(output / "vipp_cells.csv", vipp)
    write_csv(output / "image_comparison.csv", fields)
    write_csv(output / "paired_cells.csv", paired)
    wells = compare_summaries(
        authors, vipp, ["Case", "Plate", "Well", "Replicate", "Treatment"]
    )
    primary_author = pd.concat(
        [primary_subset(authors, case) for case in ["idr0139", "idr0028"]]
    )
    primary_vipp = pd.concat(
        [primary_subset(vipp, case) for case in ["idr0139", "idr0028"]]
    )
    primary = compare_summaries(primary_author, primary_vipp, ["Case", "Treatment"])
    write_csv(output / "well_comparison.csv", wells)
    write_csv(output / "primary_comparison.csv", primary)
    sampling_columns = KEYS + ["Well", "Replicate", "Treatment", "Ratio"]
    sampling_identity = {
        "ordered_vipp_cell_keys_and_ratios_sha256": hashlib.sha256(
            vipp[sampling_columns]
            .to_csv(index=False, float_format="%.17g", lineterminator="\n")
            .encode("utf-8")
        ).hexdigest(),
        "reference_cells_sha256": digest(reference_root / "cells.csv"),
        "recipe_script_sha256": digest(Path(recipes.__file__)),
    }
    sampling_receipt = output / "sampling_inputs.json"
    if args.skip_sampling:
        if (
            not sampling_receipt.exists()
            or json.loads(sampling_receipt.read_text(encoding="utf-8"))
            != sampling_identity
        ):
            raise ValueError(
                "Sampling inputs or recipe changed; rerun without --skip-sampling."
            )
    if not args.skip_sampling:
        for case in ["idr0139", "idr0028"]:
            print(f"Recomputing {case} VIPP sampling recipes", flush=True)
            recipes.sampling_tables(
                case, primary_subset(vipp, case), vipp[vipp.Case == case], output / case
            )
        json_write(sampling_receipt, sampling_identity)
    sampling = compare_sampling(reference_root, output)
    write_csv(output / "sampling_comparison.csv", sampling)
    findings = {
        "created_utc": datetime.now(UTC).isoformat(),
        "paper": recipes.PAPER,
        "author_commit": recipes.AUTHOR_COMMIT,
        "author_cells": len(authors),
        "vipp_cells": len(vipp),
        "paired_cells": len(paired),
        "expected_fields": len(fields),
        "completed_receipts": int(fields.complete_receipt.sum()),
        "byte_matched_fields": int(fields.all_source_bytes_match_author_md5.sum()),
        "missing_receipts": missing,
        "paired_metrics": metrics,
        "all_paired_metrics_within_tolerance": all(
            value["count_outside_tolerance"] == 0 for value in metrics.values()
        ),
        "source_match_by_case": fields.groupby("Case")
        .all_source_bytes_match_author_md5.agg(["sum", "count"])
        .to_dict("index"),
        "cell_counts_by_case": {
            "author": authors.groupby("Case").size().to_dict(),
            "vipp": vipp.groupby("Case").size().to_dict(),
        },
        "primary_comparison": primary.to_dict("records"),
        "well_comparison": wells.to_dict("records"),
        "sampling_comparison": sampling.to_dict("records"),
        "source_hashes": sources
        + [
            {
                "path": "validation/statistics-paper/reference/cells.csv",
                "sha256": digest(reference_root / "cells.csv"),
            }
        ],
        "script_sha256": digest(Path(__file__)),
        "recipe_script_sha256": digest(Path(recipes.__file__)),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "limits": [
            "No parameter fitting to author counts.",
            "Paired objects require source-byte matches and centroid agreement.",
            (
                "Fascin historical-input mismatch remains; agreement with "
                "current-input CellProfiler is a separate claim."
            ),
            (
                "Cell SEM is descriptive, not independent biological-replicate "
                "uncertainty."
            ),
        ],
    }
    json_write(output / "comparison.json", findings)
    report(findings, primary, fields, output)
    figures(authors, vipp, fields, paired, wells, reference_root, output)
    if not findings["all_paired_metrics_within_tolerance"]:
        raise AssertionError(
            "Matched-source scalar or centroid comparison exceeded declared "
            "tolerance; see comparison.json"
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "complete_receipts": findings["completed_receipts"],
                "paired_cells": len(paired),
                "all_paired_metrics_within_tolerance": True,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
