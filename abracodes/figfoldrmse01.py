import argparse
import io
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DATA_NAME = "fig01_fold_rmse_boxplot.csv"
METRICS_DATA_NAME = "pgroup01_metrics_per_fold.csv"
SELECTORS = ["enet", "ga"]
SELECTOR_LABELS = {"enet": "(a) Elastic Net stability selection (enet)", "ga": "(b) Consensus genetic algorithm (ga)"}
SELECTOR_COLORS = {"enet": "#4C78A8", "ga": "#E58535"}
MODEL_ORDER = ["Elastic Net", "SVR", "PLSR", "Random Forest", "Gradient Boosting", "KNN", "XGBoost"]


def _read_run_table(source, filename, figure_data=False):
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
            raise ValueError(f"Cannot load {filename} from a standalone figure-data CSV")
    return pd.read_csv(source, sep=";", decimal=",", encoding="utf-8-sig")


def load_fold_data(source, expected_folds, expected_fold_size):
    table = _read_run_table(source, FIGURE_DATA_NAME, figure_data=True)
    required = {"selector", "model", "outer_id", "seed", "fold", "RMSE"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Fold RMSE table is missing columns: {sorted(missing)}")
    table["selector"] = table["selector"].astype(str).str.strip().str.casefold()
    table["model"] = table["model"].astype(str).str.strip()
    table = table[table["selector"].isin(SELECTORS)].copy()
    for column in ["outer_id", "seed", "fold", "RMSE"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[["outer_id", "seed", "fold", "RMSE"]].isna().any().any():
        raise ValueError("Missing or nonnumeric fold metadata or RMSE")
    if not np.isfinite(table[["RMSE"]].to_numpy()).all() or (table["RMSE"] < 0).any():
        raise ValueError("RMSE values must be finite and nonnegative")
    if table.duplicated(["selector", "model", "outer_id"]).any():
        raise ValueError("Duplicate selector/model/outer-fold records found")
    if set(table["model"].unique()) != set(MODEL_ORDER):
        raise ValueError(f"Unexpected model list: {sorted(table['model'].unique())}")

    counts = table.groupby(["selector", "model"]).agg(rows=("RMSE", "size"), folds=("outer_id", "nunique"))
    if not (counts["rows"] == expected_folds).all() or not (counts["folds"] == expected_folds).all():
        raise ValueError(f"Expected {expected_folds} distinct outer folds per selector/model; found:\n{counts}")

    try:
        metrics = _read_run_table(source, METRICS_DATA_NAME)
    except (FileNotFoundError, ValueError):
        metrics = None
    if metrics is not None:
        metric_required = {"selector", "model", "outer_id", "n"}
        if not metric_required.issubset(metrics.columns):
            raise ValueError(f"Per-fold metrics table is missing columns: {sorted(metric_required - set(metrics.columns))}")
        metrics["selector"] = metrics["selector"].astype(str).str.strip().str.casefold()
        metrics["model"] = metrics["model"].astype(str).str.strip()
        metrics = metrics[metrics["selector"].isin(SELECTORS)].copy()
        metrics["n"] = pd.to_numeric(metrics["n"], errors="coerce")
        if metrics["n"].isna().any():
            raise ValueError("Missing test-fold sample sizes in per-fold metrics")
        if set(metrics["n"].unique()) != {expected_fold_size}:
            raise ValueError(f"Expected {expected_fold_size} records per test fold; found {sorted(metrics['n'].unique())}")
        keys = ["selector", "model", "outer_id"]
        check = table[keys].merge(metrics[keys + ["n"]], on=keys, how="outer", indicator=True, validate="one_to_one")
        if not (check["_merge"] == "both").all():
            raise ValueError("Figure data and per-fold metrics have different selector/model/fold rows")

    return table.sort_values(["selector", "model", "outer_id"]).reset_index(drop=True)


def _draw_panel(axis, data, selector, models, expected_folds):
    values = [data.loc[(data["selector"] == selector) & (data["model"] == model), "RMSE"].to_numpy() for model in models]
    positions = np.arange(1, len(models) + 1)
    boxes = axis.boxplot(
        values,
        positions=positions,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.5},
        whiskerprops={"color": "#555555", "linewidth": 1.1},
        capprops={"color": "#555555", "linewidth": 1.1},
    )
    for box in boxes["boxes"]:
        box.set_facecolor(SELECTOR_COLORS[selector])
        box.set_alpha(0.3)
        box.set_edgecolor(SELECTOR_COLORS[selector])
        box.set_linewidth(1.4)

    random = np.random.default_rng(2026)
    for position, model in zip(positions, models):
        points = data.loc[(data["selector"] == selector) & (data["model"] == model), "RMSE"].to_numpy()
        jitter = random.uniform(-0.12, 0.12, size=len(points))
        axis.scatter(
            position + jitter,
            points,
            s=25,
            color=SELECTOR_COLORS[selector],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.78,
            zorder=3,
        )

    axis.set_xticks(positions, models, rotation=28, ha="right")
    axis.set_xlim(0.45, len(models) + 0.55)
    axis.set_ylabel("Outer-fold RMSE (kg ha$^{-1}$)")
    axis.set_title(SELECTOR_LABELS[selector], loc="left", fontsize=14, fontweight="bold", pad=10)
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.8)
    axis.set_axisbelow(True)


def create_figure(data, output_path, expected_folds):
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 11})
    figure, axes = plt.subplots(1, 2, figsize=(15.2, 7.6), sharey=True, constrained_layout=True)
    for axis, selector in zip(axes, SELECTORS):
        _draw_panel(axis, data, selector, MODEL_ORDER, expected_folds)
    axes[1].set_ylabel("")
    figure.suptitle("RMSE distributions across outer test folds", fontsize=17, fontweight="bold")
    figure.text(0.5, 0.015, "Each outer test fold contains n = 8 records.", ha="center", fontsize=10.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Plot outer-fold RMSE distributions for enet and ga selectors.")
    parser.add_argument("--input", type=Path, default=Path("data03/pgroup01_out_2026okt20_21h.zip"))
    parser.add_argument("--output", type=Path, default=Path("data05/figS5_fold_rmse.png"))
    parser.add_argument("--folds", type=int, default=25)
    parser.add_argument("--fold-size", type=int, default=8)
    args = parser.parse_args()
    if args.folds < 1 or args.fold_size < 1:
        parser.error("--folds and --fold-size must be positive")

    data = load_fold_data(args.input, args.folds, args.fold_size)
    create_figure(data, args.output, args.folds)
    print(f"Figure saved: {args.output}")
    print(f"Selectors: {SELECTORS}; models per selector: {len(MODEL_ORDER)}; folds per model: {args.folds}; records per fold: {args.fold_size}")


if __name__ == "__main__":
    main()