import argparse
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


INDEX_ORDER = ["NDVI", "GNDVI", "NDRE", "OSAVI", "WDRVI"]
SELECTOR_ORDER = ["enet", "ga"]
SELECTOR_LABELS = {"enet": "ENet", "ga": "GA"}
SELECTOR_COLORS = {
    "reference": "#555555",
    "NDVI": "#4C78A8",
    "GNDVI": "#F58518",
    "NDRE": "#54A24B",
    "OSAVI": "#E45756",
    "WDRVI": "#B279A2",
    "enet": "#72B7B2",
    "ga": "#FF9DA6",
}
SEASON_COLORS = {2024: "#4C78A8", 2025: "#F58518"}


def validate_columns(frame, required, description):
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{description} is missing columns: {missing}")


def get_unique_row(frame, selector, model, description):
    rows = frame[(frame["selector"] == selector) & (frame["model"] == model)]
    if len(rows) != 1:
        raise ValueError(f"Expected one {description} row for {selector}/{model}; found {len(rows)}")
    return rows.iloc[0]


def select_best(summary, selector):
    candidates = summary[
        (summary["selector"] == selector)
        & (~summary["model"].isin(["Ref_year_mean", "Ref_global_mean"]))
    ]
    if candidates.empty:
        raise ValueError(f"No model rows found for selector {selector!r}")
    return candidates.sort_values(["RMSE_mean", "model"], kind="stable").iloc[0]


def build_candidates(pgroup_summary, index_summary):
    candidates = [{"selector": "none", "model": "Ref_year_mean", "label": "Ref_year_mean", "color": SELECTOR_COLORS["reference"]}]
    for selector in INDEX_ORDER:
        row = select_best(index_summary, selector)
        candidates.append(
            {
                "selector": selector,
                "model": row["model"],
                "label": f"{selector}\n{row['model']}",
                "color": SELECTOR_COLORS[selector],
                "pooled_rmse": row["RMSE_mean"],
            }
        )
    for selector in SELECTOR_ORDER:
        row = select_best(pgroup_summary, selector)
        candidates.append(
            {
                "selector": selector,
                "model": row["model"],
                "label": f"{SELECTOR_LABELS[selector]}\n{row['model']}",
                "color": SELECTOR_COLORS[selector],
                "pooled_rmse": row["RMSE_mean"],
            }
        )
    return candidates


def validate_reference_agreement(pgroup_summary, index_summary, pgroup_loyo, index_loyo):
    for season in (2024, 2025):
        pg = get_unique_row(pgroup_summary, "none", "Ref_year_mean", "pgroup reference summary")
        im = get_unique_row(index_summary, "none", "Ref_year_mean", "index reference summary")
        for metric in (f"RMSE_{season}_mean", f"RMSE_{season}_sd"):
            if not np.isclose(float(pg[metric]), float(im[metric]), rtol=1e-8, atol=1e-8):
                raise ValueError(f"Reference mismatch between analyses for {metric}")

        pg_loyo = pgroup_loyo[
            (pgroup_loyo["selector"] == "none")
            & (pgroup_loyo["model"] == "Ref_year_mean")
            & (pgroup_loyo["held_out_season"].astype(str) == str(season))
        ]
        im_loyo = index_loyo[
            (index_loyo["selector"] == "none")
            & (index_loyo["model"] == "Ref_year_mean")
            & (index_loyo["held_out_season"].astype(str) == str(season))
        ]
        if len(pg_loyo) != 1 or len(im_loyo) != 1:
            raise ValueError(f"Expected one LOYO reference per source for held-out season {season}")
        if not np.isclose(float(pg_loyo.iloc[0]["RMSE"]), float(im_loyo.iloc[0]["RMSE"]), rtol=1e-8, atol=1e-8):
            raise ValueError(f"LOYO reference mismatch between analyses for held-out season {season}")


def find_metric_row(frame, candidate, season):
    rows = frame[
        (frame["selector"] == candidate["selector"])
        & (frame["model"] == candidate["model"])
        & (frame["held_out_season"].astype(str) == str(season))
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one LOYO row for {candidate['selector']}/{candidate['model']}/{season}; found {len(rows)}"
        )
    return rows.iloc[0]


def create_figure(pgroup_summary, index_summary, pgroup_loyo, index_loyo, output_path):
    candidates = build_candidates(pgroup_summary, index_summary)
    validate_reference_agreement(pgroup_summary, index_summary, pgroup_loyo, index_loyo)
    positions = np.arange(len(candidates))
    labels = [candidate["label"] for candidate in candidates]

    # constrained_layout=True használata a rácsos elrendezéshez
    figure = plt.figure(figsize=(12, 11), constrained_layout=True)
    gs = figure.add_gridspec(2, 2, hspace=0.05, wspace=0.1)
    
    axes = [
        figure.add_subplot(gs[0, 0]),
        figure.add_subplot(gs[0, 1]),
        figure.add_subplot(gs[1, :])  # (c) ábra alul teljes szélességben
    ]

    for axis, season in zip(axes[:2], (2024, 2025)):
        reference = get_unique_row(pgroup_summary, "none", "Ref_year_mean", "pgroup reference summary")
        means = []
        errors = []
        for candidate in candidates:
            row = get_unique_row(
                index_summary if candidate["selector"] in INDEX_ORDER else pgroup_summary,
                candidate["selector"],
                candidate["model"],
                "selected model summary",
            )
            means.append(float(row[f"RMSE_{season}_mean"]))
            errors.append(float(row[f"RMSE_{season}_sd"]))
        axis.bar(
            positions,
            means,
            yerr=errors,
            color=[candidate["color"] for candidate in candidates],
            edgecolor="white",
            linewidth=0.5,
            capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
        axis.axhline(
            float(reference[f"RMSE_{season}_mean"]),
            color="#222222",
            linestyle="--",
            linewidth=1.3,
        )
        axis.set_title(f"({'a' if season == 2024 else 'b'}) Within-season RMSE, {season}", fontsize=16)
        axis.set_ylabel("RMSE (kg ha$^{-1}$)", fontsize=14)
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
        axis.tick_params(axis='y', labelsize=12)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        axis.set_axisbelow(True)

    axis = axes[2]
    width = 0.36
    for offset, season in enumerate((2024, 2025)):
        values = []
        for candidate in candidates:
            if candidate["selector"] == "none":
                rows = pgroup_loyo[
                    (pgroup_loyo["selector"] == "none")
                    & (pgroup_loyo["model"] == "Ref_year_mean")
                    & (pgroup_loyo["held_out_season"].astype(str) == str(season))
                ]
                if len(rows) != 1:
                    raise ValueError(f"Expected one LOYO reference for held-out season {season}")
                values.append(float(rows.iloc[0]["RMSE"]))
            else:
                source = index_loyo if candidate["selector"] in INDEX_ORDER else pgroup_loyo
                values.append(float(find_metric_row(source, candidate, season)["RMSE"]))
        axis.bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            color=SEASON_COLORS[season],
            edgecolor="white",
            linewidth=0.5,
        )
        ref_rows = pgroup_loyo[
            (pgroup_loyo["selector"] == "none")
            & (pgroup_loyo["model"] == "Ref_year_mean")
            & (pgroup_loyo["held_out_season"].astype(str) == str(season))
        ]
        axis.axhline(
            float(ref_rows.iloc[0]["RMSE"]),
            color=SEASON_COLORS[season],
            linestyle="--",
            linewidth=1.2,
        )

    axis.set_title("(c) Leave-one-year-out (LOYO) RMSE", fontsize=16)
    axis.set_ylabel("RMSE (kg ha$^{-1}$)", fontsize=14)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    axis.tick_params(axis='y', labelsize=12)
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    axis.set_axisbelow(True)

    legend_handles = [
        Patch(facecolor=SEASON_COLORS[2024], label="Held out: 2024"),
        Patch(facecolor=SEASON_COLORS[2025], label="Held out: 2025"),
        Line2D([0], [0], color=SEASON_COLORS[2024], linestyle="--", label="Ref_year_mean, held out: 2024"),
        Line2D([0], [0], color=SEASON_COLORS[2025], linestyle="--", label="Ref_year_mean, held out: 2025"),
    ]
    
    # Jelmagyarázat alulra helyezve, vízszintes elrendezésben
    figure.legend(
        handles=legend_handles,
        fontsize=12,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Plot selected models against Ref_year_mean using saved CSV outputs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"c:\dg\kepelemzes01\cikk_kozos\kozoscikk_jav\ujcikk2\abracodes\fig06_performance_vs_reference.png"),
    )
    args = parser.parse_args()

    pgroup_summary = pd.read_csv(r"c:\dg\kepelemzes01\cikk_kozos\kozoscikk_jav\komondorout\pgroup01_out_2026okt20_21h\run_20261020_180820\pgroup01_metrics_summary.csv", sep=";", decimal=",", encoding="utf-8-sig")
    pgroup_loyo = pd.read_csv(r"c:\dg\kepelemzes01\cikk_kozos\kozoscikk_jav\komondorout\pgroup01_out_2026okt20_21h\run_20261020_180820\pgroup01_metrics_loyo.csv", sep=";", decimal=",", encoding="utf-8-sig")
    index_summary = pd.read_csv(r"c:\dg\kepelemzes01\cikk_kozos\kozoscikk_jav\kimenetek\indexmod4out\run_20261025_032437\\im04_metrics_summary.csv", sep=";", decimal=",", encoding="utf-8-sig")
    index_loyo = pd.read_csv(r"c:\dg\kepelemzes01\cikk_kozos\kozoscikk_jav\kimenetek\indexmod4out\run_20261025_032437\\im04_metrics_loyo.csv", sep=";", decimal=",", encoding="utf-8-sig")

    validate_columns(
        pgroup_summary,
        ["selector", "model", "RMSE_mean", "RMSE_2024_mean", "RMSE_2024_sd", "RMSE_2025_mean", "RMSE_2025_sd"],
        "pgroup summary",
    )
    validate_columns(
        index_summary,
        ["selector", "model", "RMSE_mean", "RMSE_2024_mean", "RMSE_2024_sd", "RMSE_2025_mean", "RMSE_2025_sd"],
        "index-model summary",
    )
    validate_columns(pgroup_loyo, ["selector", "model", "held_out_season", "RMSE"], "pgroup LOYO")
    validate_columns(index_loyo, ["selector", "model", "held_out_season", "RMSE"], "index-model LOYO")

    candidates = create_figure(pgroup_summary, index_summary, pgroup_loyo, index_loyo, args.output)
    print(f"Figure saved: {args.output}")
    print("Model selection: lowest pooled RMSE_mean within each selector")
    for candidate in candidates[1:]:
        print(f"{candidate['selector']}: {candidate['model']} (pooled RMSE={candidate['pooled_rmse']:.3f})")
    print("LOYO bars are single held-out-season estimates; no repeat-based SD is available.")


if __name__ == "__main__":
    main()
