import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INDEX_ORDER = ["GNDVI", "NDRE", "NDVI", "OSAVI", "WDRVI"]
MODEL_ORDER = [
    "MLR", "Ridge", "Elastic Net", "SVR", "PLSR",
    "Random Forest", "Gradient Boosting", "KNN", "XGBoost",
]



def prepare_heatmap_data(metrics):
    required = {"selector", "model", "RMSE_mean"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics table is missing columns: {sorted(missing)}")

    metrics = metrics.copy()
    metrics["selector"] = metrics["selector"].astype(str).str.strip()
    metrics["model"] = metrics["model"].astype(str).str.strip()
    r2_columns = sorted(
        [column for column in metrics.columns if re.fullmatch(r"R2_\d+_mean", column)],
        key=lambda column: int(re.search(r"\d+", column).group()),
    )
    if not r2_columns:
        raise ValueError("Metrics table contains no season-specific R2 mean columns")

    for column in ["RMSE_mean", *r2_columns]:
        metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    metrics = metrics[metrics["model"].isin(MODEL_ORDER)].copy()
    metrics["R2_within"] = metrics[r2_columns].mean(axis=1)

    selectors = set(metrics["selector"]) - {"none"}
    if selectors != set(INDEX_ORDER):
        raise ValueError(f"Expected the five vegetation-index sets; found {sorted(selectors)}")
    metrics = metrics[metrics["selector"].isin(INDEX_ORDER)]
    if len(metrics) != len(INDEX_ORDER) * len(MODEL_ORDER):
        raise ValueError(f"Expected 45 index/model rows; found {len(metrics)}")
    if metrics.duplicated(["selector", "model"]).any():
        raise ValueError("Duplicate index/model records found")

    values = metrics[["RMSE_mean", "R2_within"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (metrics["RMSE_mean"] < 0).any():
        raise ValueError("Heatmap metrics contain missing, nonnumeric, or invalid values")
    return metrics


def create_figure(metrics, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 7.2))
    panels = [
        (axes[0], "RMSE_mean", "Pooled RMSE (kg ha$^{-1}$)", "viridis_r"),
        (axes[1], "R2_within", "Within-season R$^2$ (mean of seasons)", "viridis"),
    ]
    for axis, column, title, colormap in panels:
        matrix = metrics.pivot(index="selector", columns="model", values=column)
        matrix = matrix.reindex(index=INDEX_ORDER, columns=MODEL_ORDER)
        image = axis.imshow(matrix.to_numpy(dtype=float), aspect="equal", cmap=colormap)
        axis.set_xticks(range(len(MODEL_ORDER)))
        axis.set_xticklabels(MODEL_ORDER, rotation=35, ha="right", fontsize=10)
        axis.set_yticks(range(len(INDEX_ORDER)))
        axis.set_yticklabels(INDEX_ORDER, fontsize=10)
        for row in range(len(INDEX_ORDER)):
            for col in range(len(MODEL_ORDER)):
                value = matrix.iat[row, col]
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                text_color = "white" if luminance < 0.48 else "#1A1A1A"
                axis.text(col, row, f"{value:.0f}" if column == "RMSE_mean" else f"{value:.2f}",
                          ha="center", va="center", fontsize=9,
                          fontweight="semibold", color=text_color)
        axis.set_title(title, fontsize=12, fontweight="semibold", pad=12)
        colorbar = figure.colorbar(image, ax=axis, shrink=0.5)
        colorbar.ax.tick_params(labelsize=10)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    metrics = pd.read_csv(r"C:\dg\kepelemzes01\cikk_kozos\kozoscikk_jav\kimenetek\indexmod4out\run_20261025_032437\figure_data\fig10_index_model_heatmap.csv", sep=";", decimal=",", encoding="utf-8-sig")
    metrics = prepare_heatmap_data(metrics)
    output_path = Path(r"C:\dg\kepelemzes01\cikk_kozos\kozoscikk_jav\ujcikk2\abracodes\fig10_index_model_heatmap.png")
    create_figure(metrics, output_path)
    print(f"Figure saved: {output_path}")
    print(f"Metrics rows used: {len(metrics)}")


if __name__ == "__main__":
    main()
