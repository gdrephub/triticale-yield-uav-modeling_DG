import argparse
import io
import re
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


FIGURE_DATA_NAME = "fig05_rf_importance_oof.csv"
SELECTORS = ["enet", "ga"]
SELECTOR_LABELS = {"enet": "Elastic Net selection", "ga": "Consensus GA selection"}
SELECTOR_COLORS = {"enet": "#4C78A8", "ga": "#E58535"}
METHODS = ["perm_mse_increase", "gini", "shap_mean_abs", "rfe_rank"]
METHOD_TITLES = {
    "perm_mse_increase": "(a) Permutation importance (MSE increase)",
    "gini": "(b) Impurity-based importance (Gini)",
    "shap_mean_abs": "(c) Mean absolute SHAP value",
    "rfe_rank": "(d) Recursive feature elimination rank (1 = best)",
}
X_LABELS = {
    "perm_mse_increase": "Increase in mean squared error",
    "gini": "Gini importance",
    "shap_mean_abs": "Mean absolute SHAP value (kg ha$^{-1}$)",
    "rfe_rank": "Mean RFE rank (lower = better)",
}
INDEX_ORDER = {"NDVI": 0, "GNDVI": 1, "NDRE": 2, "OSAVI": 3, "WDRVI": 4}
STAT_ORDER = {"mean": 0, "median": 1, "std": 2, "auc": 3}
PREDICTOR_PATTERN = re.compile(r"^(?P<index>NDVI|GNDVI|NDRE|OSAVI|WDRVI)_t_(?P<stat>mean|median|std|auc)$")
REQUIRED_COLUMNS = {"selector", "method", "predictor", "mean", "sd", "n_folds_selected"}


def _read_importance_table(source):
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


def load_importance_data(source):
    table = _read_importance_table(source)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Importance table is missing columns: {sorted(missing)}")
    table["selector"] = table["selector"].astype(str).str.strip().str.casefold()
    table["method"] = table["method"].astype(str).str.strip()
    table["predictor"] = table["predictor"].astype(str).str.strip()
    table = table[table["selector"].isin(SELECTORS) & table["method"].isin(METHODS)].copy()

    for column in ["mean", "sd", "n_folds_selected"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    if table[["mean", "n_folds_selected"]].isna().any().any():
        raise ValueError("Missing or nonnumeric importance means/fold counts")
    if not np.isfinite(table["mean"].to_numpy()).all():
        raise ValueError("Non-finite importance means")
    finite_sd = table["sd"].dropna().to_numpy(dtype=float)
    if not np.isfinite(finite_sd).all() or (finite_sd < 0).any():
        raise ValueError("Standard deviations must be finite and nonnegative when available")
    if (table["n_folds_selected"] < 1).any() or (table["n_folds_selected"] % 1 != 0).any():
        raise ValueError("Fold counts must be positive integers")
    if table.duplicated(["selector", "method", "predictor"]).any():
        raise ValueError("Duplicate selector/method/predictor rows found")

    predictor_sets = {}
    for selector in SELECTORS:
        predictor_sets[selector] = set(table.loc[table["selector"] == selector, "predictor"])
    if len(predictor_sets["ga"]) != 20 or len(predictor_sets["enet"] | predictor_sets["ga"]) != 20:
        raise ValueError(f"Expected 20 candidate predictors; found {len(predictor_sets['enet'] | predictor_sets['ga'])}")

    for method in METHODS:
        method_data = table[table["method"] == method]
        method_sets = {
            selector: set(method_data.loc[method_data["selector"] == selector, "predictor"])
            for selector in SELECTORS
        }
        if len(method_sets["ga"]) != 20 or method_sets["enet"] != predictor_sets["enet"]:
            raise ValueError(f"Unexpected predictor coverage for importance method {method}")
        if method_data.loc[method_data["sd"].isna(), "n_folds_selected"].ne(1).any():
            raise ValueError(f"Missing SD for a predictor selected in multiple folds ({method})")
        if method_data.loc[method_data["n_folds_selected"] == 1, "sd"].notna().any():
            raise ValueError(f"Unexpected SD for a predictor selected in one fold ({method})")

    predictor_names = sorted(
        predictor_sets["ga"],
        key=lambda predictor: (
            INDEX_ORDER[PREDICTOR_PATTERN.fullmatch(predictor).group("index")],
            STAT_ORDER[PREDICTOR_PATTERN.fullmatch(predictor).group("stat")],
        ) if PREDICTOR_PATTERN.fullmatch(predictor) else (999, predictor),
    )
    if any(PREDICTOR_PATTERN.fullmatch(predictor) is None for predictor in predictor_names):
        invalid = [predictor for predictor in predictor_names if PREDICTOR_PATTERN.fullmatch(predictor) is None]
        raise ValueError(f"Unexpected predictor labels: {invalid}")
    return table, predictor_names


def _draw_method(axis, table, predictors, method, y_positions):
    method_data = table[table["method"] == method]
    offsets = {"enet": -0.19, "ga": 0.19}
    bar_height = 0.34
    all_extents = [0.0]
    for selector in SELECTORS:
        selected = method_data[method_data["selector"] == selector].set_index("predictor")
        for predictor in predictors:
            if predictor not in selected.index:
                continue
            row = selected.loc[predictor]
            mean = float(row["mean"])
            sd = float(row["sd"]) if pd.notna(row["sd"]) else np.nan
            y = y_positions[predictor] + offsets[selector]
            axis.barh(
                y,
                mean,
                height=bar_height,
                color=SELECTOR_COLORS[selector],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            if np.isfinite(sd):
                axis.errorbar(
                    mean,
                    y,
                    xerr=sd,
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=1.0,
                    capsize=2.5,
                    zorder=4,
                )
                all_extents.extend([mean - sd, mean + sd])
            else:
                all_extents.append(mean)

    lower, upper = min(all_extents), max(all_extents)
    span = upper - lower
    if span == 0:
        span = max(abs(upper), 1.0)
    axis.set_xlim(lower - 0.06 * span, upper + 0.09 * span)
    axis.set_title(METHOD_TITLES[method], loc="left", fontsize=13, fontweight="bold", pad=9)
    axis.set_xlabel(X_LABELS[method])
    axis.grid(axis="x", color="#E5E5E5", linewidth=0.8)
    axis.set_axisbelow(True)
    if method == "rfe_rank":
        axis.axvline(1, color="#777777", linewidth=0.9, linestyle=":", zorder=1)


def create_figure(table, predictors, output_path):
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5})
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 17.2), sharey=True)
    y_positions = {predictor: position for position, predictor in enumerate(predictors)}

    for axis, method in zip(axes.flat, METHODS):
        _draw_method(axis, table, predictors, method, y_positions)
        axis.set_yticks(range(len(predictors)))
        axis.set_ylim(len(predictors) - 0.5, -0.5)

    for axis in axes[:, 1]:
        axis.tick_params(axis="y", labelleft=False)
    for axis in axes[:, 0]:
        axis.set_yticklabels(predictors)
        axis.set_ylabel("Candidate predictor")

    legend_handles = [
        Patch(facecolor=SELECTOR_COLORS[selector], label=SELECTOR_LABELS[selector])
        for selector in SELECTORS
    ]
    figure.suptitle("Random Forest predictor importance", fontsize=18, fontweight="bold", y=0.985)
    figure.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.957), ncol=2, frameon=False, fontsize=12)
    figure.text(
        0.5,
        0.018,
        "Means ± SD are computed over folds in which each predictor was selected. No ENet bar means never selected; SD is unavailable when selected in one fold.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.17, right=0.98, top=0.925, bottom=0.07, hspace=0.24, wspace=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=350, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Create a four-panel Random Forest importance figure.")
    parser.add_argument("--input", type=Path, default=Path("data03/pgroup01_out_2026okt20_21h.zip"))
    parser.add_argument("--output", type=Path, default=Path("data05/figS4_rf_importance.png"))
    args = parser.parse_args()
    table, predictors = load_importance_data(args.input)
    create_figure(table, predictors, args.output)
    print(f"Figure saved: {args.output}")
    for method in METHODS:
        subset = table[table["method"] == method]
        selected = subset.groupby("selector")["predictor"].nunique().to_dict()
        print(f"{method}: predictors with importance values by selector = {selected}")


if __name__ == "__main__":
    main()