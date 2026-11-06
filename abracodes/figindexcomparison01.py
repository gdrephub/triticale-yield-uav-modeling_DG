import argparse
import io
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COMPARISON_NAME = "fig09_index_comparison.csv"
SEED_METRICS_NAME = "im02_metrics_per_seed.csv"
INDEX_ORDER = ["NDVI", "GNDVI", "NDRE", "OSAVI", "WDRVI"]
INDEX_COLORS = {
    "NDVI": "#4C78A8",
    "GNDVI": "#F58518",
    "NDRE": "#54A24B",
    "OSAVI": "#E45756",
    "WDRVI": "#B279A2",
}
MODEL_ORDER = [
    "MLR", "Ridge", "Elastic Net", "SVR", "PLSR",
    "Random Forest", "Gradient Boosting", "KNN", "XGBoost",
]
REFERENCE_NAMES = ["Ref_year_mean", "Ref_global_mean"]
REQUIRED_COLUMNS = {"selector", "model", "RMSE_mean", "RMSE_sd"}


def _read_data(source, filename, figure_data=False):
    source = Path(source)
    if source.suffix.lower() == ".zip":
        suffix = f"/figure_data/{filename}" if figure_data else f"/{filename}"
        with ZipFile(source) as archive:
            candidates = [name for name in archive.namelist() if name.endswith(suffix)]
            if len(candidates) != 1:
                raise ValueError(f"Expected one {filename} in {source}; found {len(candidates)}")
            content = archive.read(candidates[0])
        return pd.read_csv(io.BytesIO(content), sep=";", decimal=",", encoding="utf-8-sig")

    if source.is_dir():
        candidates = [path for path in source.rglob(filename) if ("figure_data" in path.parts) == figure_data]
        if len(candidates) != 1:
            raise ValueError(f"Expected one {filename} below {source}; found {len(candidates)}")
        source = candidates[0]
    elif source.suffix.lower() == ".csv":
        if not figure_data:
            raise ValueError(f"Cannot load {filename} from a standalone comparison CSV")
    return pd.read_csv(source, sep=";", decimal=",", encoding="utf-8-sig")


def load_comparison_data(source, expected_repeats):
    summary = _read_data(source, COMPARISON_NAME, figure_data=True)
    missing = REQUIRED_COLUMNS - set(summary.columns)
    if missing:
        raise ValueError(f"Comparison table is missing columns: {sorted(missing)}")
    summary["selector"] = summary["selector"].astype(str).str.strip()
    summary["model"] = summary["model"].astype(str).str.strip()
    for column in ["RMSE_mean", "RMSE_sd"]:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")
    if summary[list(REQUIRED_COLUMNS - {"selector", "model"})].isna().any().any():
        raise ValueError("Missing or nonnumeric RMSE means/standard deviations")
    if not np.isfinite(summary[["RMSE_mean", "RMSE_sd"]].to_numpy()).all():
        raise ValueError("RMSE means and standard deviations must be finite")
    if (summary[["RMSE_mean", "RMSE_sd"]] < 0).any().any():
        raise ValueError("RMSE means and standard deviations must be nonnegative")

    indices = summary[summary["selector"].isin(INDEX_ORDER) & summary["model"].isin(MODEL_ORDER)].copy()
    references = summary[(summary["selector"] == "none") & summary["model"].isin(REFERENCE_NAMES)].copy()
    if len(indices) != len(INDEX_ORDER) * len(MODEL_ORDER):
        raise ValueError(f"Expected {len(INDEX_ORDER) * len(MODEL_ORDER)} index/model rows; found {len(indices)}")
    if indices.duplicated(["selector", "model"]).any():
        raise ValueError("Duplicate index/model summary rows found")
    expected_pairs = {(index, model) for index in INDEX_ORDER for model in MODEL_ORDER}
    actual_pairs = set(indices[["selector", "model"]].itertuples(index=False, name=None))
    if actual_pairs != expected_pairs:
        raise ValueError("The index/model comparison table is incomplete")
    if set(references["model"]) != set(REFERENCE_NAMES) or references["model"].duplicated().any():
        raise ValueError("Expected one pooled RMSE summary for each reference model")

    try:
        seeds = _read_data(source, SEED_METRICS_NAME)
    except (FileNotFoundError, ValueError):
        seeds = None
    if seeds is not None:
        if not {"selector", "model", "seed"}.issubset(seeds.columns):
            raise ValueError("Per-seed metrics table is missing selector/model/seed columns")
        seed_counts = seeds.groupby(["selector", "model"])["seed"].nunique()
        if len(seed_counts) != len(INDEX_ORDER) * len(MODEL_ORDER) + len(REFERENCE_NAMES):
            raise ValueError("Unexpected model/set coverage in per-seed metric table")
        if not (seed_counts == expected_repeats).all():
            raise ValueError(f"Expected {expected_repeats} repeats for every index/model and reference; found {seed_counts.unique().tolist()}")
    return indices, references


def create_figure(indices, references, output_path, expected_repeats):
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 11})
    figure, axis = plt.subplots(figsize=(16.5, 8.5))
    positions = np.arange(len(MODEL_ORDER))
    bar_width = 0.15
    group_width = bar_width * len(INDEX_ORDER)

    for index_number, index in enumerate(INDEX_ORDER):
        values = indices[indices["selector"] == index].set_index("model").reindex(MODEL_ORDER)
        centers = positions - group_width / 2 + (index_number + 0.5) * bar_width
        axis.bar(
            centers,
            values["RMSE_mean"].to_numpy(dtype=float),
            width=bar_width,
            yerr=values["RMSE_sd"].to_numpy(dtype=float),
            capsize=2.5,
            color=INDEX_COLORS[index],
            edgecolor="white",
            linewidth=0.5,
            label=index,
            zorder=3,
        )

    reference_values = references.set_index("model")["RMSE_mean"]
    axis.axhline(
        reference_values["Ref_year_mean"],
        color="#222222",
        linestyle="--",
        linewidth=1.7,
        label=f"Season-mean reference ({reference_values['Ref_year_mean']:.0f})",
        zorder=4,
    )
    axis.axhline(
        reference_values["Ref_global_mean"],
        color="#6B6B6B",
        linestyle=":",
        linewidth=1.9,
        label=f"Global-mean reference ({reference_values['Ref_global_mean']:.0f})",
        zorder=4,
    )

    axis.set_xticks(positions, MODEL_ORDER, rotation=25, ha="right")
    axis.set_ylabel("Pooled out-of-fold RMSE (kg ha$^{-1}$)")
    axis.set_xlabel("Regression algorithm")
    axis.set_ylim(0, max(float((indices["RMSE_mean"] + indices["RMSE_sd"]).max()), float(reference_values.max())) * 1.12)
    axis.set_title("Single-index model performance by algorithm and vegetation index", loc="left", fontsize=15, fontweight="bold", pad=14)
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    axis.set_axisbelow(True)

    handles = [Patch(facecolor=INDEX_COLORS[index], label=index) for index in INDEX_ORDER]
    handles.extend([
        Line2D([0], [0], color="#222222", linestyle="--", linewidth=1.7,
               label=f"Season-mean reference ({reference_values['Ref_year_mean']:.0f})"),
        Line2D([0], [0], color="#6B6B6B", linestyle=":", linewidth=1.9,
               label=f"Global-mean reference ({reference_values['Ref_global_mean']:.0f})"),
    ])
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, fontsize=10.5)
    figure.text(0.5, 0.02, f"Bars show mean ± SD across {expected_repeats} repeats.", ha="center", fontsize=10.5)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.23)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Plot pooled RMSE for single-index models and references.")
    parser.add_argument("--input", type=Path, default=Path("data02/indexmodout.zip"))
    parser.add_argument("--output", type=Path, default=Path("data05/figS6_index_comparison.png"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    indices, references = load_comparison_data(args.input, args.repeats)
    create_figure(indices, references, args.output, args.repeats)
    print(f"Figure saved: {args.output}")
    print("Reference pooled RMSE means:")
    print(references[["model", "RMSE_mean", "RMSE_sd"]].to_string(index=False))


if __name__ == "__main__":
    main()