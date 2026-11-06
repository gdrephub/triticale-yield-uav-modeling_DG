import argparse
import io
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


FIGURE_DATA_NAME = "fig02_observed_vs_predicted.csv"
SEASONS = [2024, 2025]
SEASON_COLORS = {2024: "#4C78A8", 2025: "#F58518"}
REQUIRED_COLUMNS = {
    "row_id",
    "season",
    "observed",
    "predicted_mean_over_seeds",
    "selector",
    "model",
}


def _read_table(source):
    source = Path(source)
    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            candidates = [
                name for name in archive.namelist()
                if name.endswith(f"/figure_data/{FIGURE_DATA_NAME}")
            ]
            if len(candidates) != 1:
                raise ValueError(f"Expected one {FIGURE_DATA_NAME} in {source}; found {len(candidates)}")
            content = archive.read(candidates[0])
        return pd.read_csv(io.BytesIO(content), sep=";", decimal=",", encoding="utf-8-sig")

    if source.is_dir():
        candidates = list(source.rglob(FIGURE_DATA_NAME))
        if len(candidates) != 1:
            raise ValueError(f"Expected one {FIGURE_DATA_NAME} below {source}; found {len(candidates)}")
        source = candidates[0]
    return pd.read_csv(source, sep=";", decimal=",", encoding="utf-8-sig")


def _select_model(source, selector, model, label):
    table = _read_table(source)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"{label} prediction table is missing columns: {sorted(missing)}")
    selected = table.loc[
        table["selector"].astype(str).str.casefold().eq(selector.casefold())
        & table["model"].astype(str).str.casefold().eq(model.casefold())
    ].copy()
    if selected.empty:
        raise ValueError(f"No {selector}/{model} records found for {label}")
    for column in ["row_id", "season", "observed", "predicted_mean_over_seeds"]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if selected[["row_id", "season", "observed", "predicted_mean_over_seeds"]].isna().any().any():
        raise ValueError(f"Missing or nonnumeric values in {label} {selector}/{model} records")
    if not np.isfinite(selected[["observed", "predicted_mean_over_seeds"]].to_numpy()).all():
        raise ValueError(f"Non-finite observed or predicted values in {label} {selector}/{model} records")
    if sorted(selected["season"].unique().astype(int).tolist()) != SEASONS:
        raise ValueError(f"Expected seasons {SEASONS} in {label}; found {sorted(selected['season'].unique().tolist())}")
    if selected.duplicated(["row_id", "season"]).any():
        raise ValueError(f"Duplicate row_id/season pairs in {label} {selector}/{model} records")
    return selected.sort_values(["season", "row_id"]).reset_index(drop=True)


def _r_squared(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    total_sum_squares = np.sum((observed - observed.mean()) ** 2)
    if len(observed) < 2 or total_sum_squares == 0:
        raise ValueError("Within-season R² requires at least two observations with nonzero variance")
    return 1 - np.sum((observed - predicted) ** 2) / total_sum_squares


def _season_r_squared(data):
    return {
        season: _r_squared(
            group["observed"],
            group["predicted_mean_over_seeds"],
        )
        for season, group in data.groupby("season", sort=True)
    }


def _draw_panel(axis, data, title, limits):
    for season in SEASONS:
        subset = data[data["season"] == season]
        axis.scatter(
            subset["observed"],
            subset["predicted_mean_over_seeds"],
            s=55,
            color=SEASON_COLORS[season],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
            zorder=3,
        )

    axis.plot(limits, limits, color="#333333", linestyle="--", linewidth=1.5, zorder=2)
    scores = _season_r_squared(data)
    score_text = "Within-season R² of mean OOF predictions\n" + "\n".join(
        f"{season}: {scores[season]:.2f}" for season in SEASONS
    )
    axis.text(
        0.04,
        0.96,
        score_text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
    )
    axis.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=10)
    axis.set_xlabel("Observed grain yield (kg ha$^{-1}$)")
    axis.set_ylabel("Predicted grain yield (kg ha$^{-1}$)")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#E7E7E7", linewidth=0.8)
    axis.set_axisbelow(True)


def create_figure(index_data, multi_data, output_path):
    observed_values = np.concatenate([
        index_data["observed"].to_numpy(),
        multi_data["observed"].to_numpy(),
    ])
    predicted_values = np.concatenate([
        index_data["predicted_mean_over_seeds"].to_numpy(),
        multi_data["predicted_mean_over_seeds"].to_numpy(),
    ])
    lower = min(observed_values.min(), predicted_values.min())
    upper = max(observed_values.max(), predicted_values.max())
    padding = (upper - lower) * 0.04
    limits = (lower - padding, upper + padding)

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11})
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 7.0), constrained_layout=True)
    _draw_panel(axes[0], index_data, "(a) WDRVI–KNN (single-index)", limits)
    _draw_panel(axes[1], multi_data, "(b) GA–KNN (multi-index)", limits)

    season_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="None",
            markerfacecolor=SEASON_COLORS[season], markeredgecolor="white",
            markersize=8, label=str(season)
        )
        for season in SEASONS
    ]
    for axis in axes:
        axis.legend(handles=season_handles, title="Season", loc="lower right", frameon=True, framealpha=0.92)
    figure.suptitle("Observed versus predicted grain yield", fontsize=17, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(figure)


def _check_same_observations(index_data, multi_data):
    key = ["row_id", "season"]
    comparison = index_data[key + ["observed"]].merge(
        multi_data[key + ["observed"]],
        on=key,
        how="outer",
        suffixes=("_single", "_multi"),
        indicator=True,
        validate="one_to_one",
    )
    if not (comparison["_merge"] == "both").all():
        raise ValueError("The two model files do not contain the same row_id/season observations")
    if not np.allclose(comparison["observed_single"], comparison["observed_multi"]):
        raise ValueError("Observed yields differ between the two model files")


def main():
    parser = argparse.ArgumentParser(description="Create a two-panel observed-versus-predicted yield figure.")
    parser.add_argument("--index-input", type=Path, default=Path("data02/indexmodout.zip"))
    parser.add_argument("--multi-input", type=Path, default=Path("data03/pgroup01_out_2026okt20_21h.zip"))
    parser.add_argument("--output", type=Path, default=Path("data05/figS2_observed_vs_predicted.png"))
    args = parser.parse_args()

    index_data = _select_model(args.index_input, "WDRVI", "KNN", "single-index")
    multi_data = _select_model(args.multi_input, "ga", "KNN", "multi-index")
    _check_same_observations(index_data, multi_data)
    create_figure(index_data, multi_data, args.output)
    print(f"Figure saved: {args.output}")
    print(f"WDRVI–KNN within-season R²: {_season_r_squared(index_data)}")
    print(f"GA–KNN within-season R²: {_season_r_squared(multi_data)}")


if __name__ == "__main__":
    main()