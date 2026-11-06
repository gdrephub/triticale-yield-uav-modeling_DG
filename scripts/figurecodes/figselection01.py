import argparse
import io
import re
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE_DATA_NAME = "fig04_selection_frequency.csv"
SELECTORS = ["enet", "ga"]
SELECTOR_LABELS = {
    "enet": "(a) Elastic Net stability selection (enet)",
    "ga": "(b) Consensus genetic algorithm (ga)",
}
SELECTOR_COLORS = {"enet": "#4C78A8", "ga": "#E07B24"}
INDEX_ORDER = {"NDVI": 0, "GNDVI": 1, "NDRE": 2, "OSAVI": 3, "WDRVI": 4}
STAT_ORDER = {"mean": 0, "median": 1, "std": 2, "auc": 3}
PREDICTOR_PATTERN = re.compile(r"^(?P<index>NDVI|GNDVI|NDRE|OSAVI|WDRVI)_t_(?P<stat>mean|median|std|auc)$")
REQUIRED_COLUMNS = {"selector", "predictor", "n_selected", "frequency"}


def _read_frequency_table(source):
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


def load_frequency_data(source, n_folds):
    table = _read_frequency_table(source)
    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Selection-frequency table is missing columns: {sorted(missing)}")
    table["selector"] = table["selector"].astype(str).str.strip().str.casefold()
    table = table[table["selector"].isin(SELECTORS)].copy()
    if set(table["selector"].unique()) != set(SELECTORS):
        raise ValueError(f"Expected selectors {SELECTORS}; found {sorted(table['selector'].unique())}")

    table["n_selected"] = pd.to_numeric(table["n_selected"], errors="coerce")
    table["frequency"] = pd.to_numeric(table["frequency"], errors="coerce")
    if table[["n_selected", "frequency"]].isna().any().any():
        raise ValueError("Missing or nonnumeric selection counts/frequencies")
    if not np.isfinite(table[["n_selected", "frequency"]].to_numpy()).all():
        raise ValueError("Non-finite selection counts/frequencies")
    if (table["n_selected"] % 1 != 0).any() or (table["n_selected"] < 0).any() or (table["n_selected"] > n_folds).any():
        raise ValueError(f"Selection counts must be integers from 0 to {n_folds}")
    if ((table["frequency"] < 0) | (table["frequency"] > 1)).any():
        raise ValueError("Selection frequencies must be between 0 and 1")
    if not np.allclose(table["frequency"], table["n_selected"] / n_folds, atol=1e-8):
        raise ValueError(f"Frequencies do not match n_selected / {n_folds}")
    if table.duplicated(["selector", "predictor"]).any():
        raise ValueError("Duplicate selector/predictor rows found")

    predictor_sets = {
        selector: set(table.loc[table["selector"] == selector, "predictor"])
        for selector in SELECTORS
    }
    if any(len(predictor_sets[selector]) != 20 for selector in SELECTORS):
        counts = {selector: len(values) for selector, values in predictor_sets.items()}
        raise ValueError(f"Expected 20 predictors per selector; found {counts}")
    if predictor_sets["enet"] != predictor_sets["ga"]:
        raise ValueError("enet and ga do not contain the same predictor set")

    parsed = {}
    for predictor in predictor_sets["enet"]:
        match = PREDICTOR_PATTERN.fullmatch(predictor)
        if match is None:
            raise ValueError(f"Unexpected predictor label: {predictor!r}")
        parsed[predictor] = (match.group("index"), match.group("stat"))
    ordered_predictors = sorted(
        predictor_sets["enet"],
        key=lambda name: (INDEX_ORDER[parsed[name][0]], STAT_ORDER[parsed[name][1]]),
    )
    return table, ordered_predictors


def create_figure(table, predictors, output_path, n_folds):
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 10.5})
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 10.0), sharey=True, constrained_layout=True)
    positions = np.arange(len(predictors))

    for axis, selector in zip(axes, SELECTORS):
        subset = table[table["selector"] == selector].set_index("predictor").reindex(predictors)
        frequencies = subset["frequency"].to_numpy(dtype=float)
        counts = subset["n_selected"].to_numpy(dtype=int)
        axis.barh(
            positions,
            frequencies,
            height=0.72,
            color=SELECTOR_COLORS[selector],
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        for y_position, frequency, count in zip(positions, frequencies, counts):
            label_x = max(frequency + 0.018, 0.018)
            axis.text(label_x, y_position, f"{count}/{n_folds}", va="center", ha="left", fontsize=9, color="#333333")
        axis.set_title(SELECTOR_LABELS[selector], loc="left", fontsize=14, fontweight="bold", pad=11)
        axis.set_xlim(0, 1.22)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"])
        axis.set_xlabel("Selection frequency across 25 outer folds")
        axis.grid(axis="x", color="#E5E5E5", linewidth=0.8)
        axis.set_axisbelow(True)

    axes[0].set_yticks(positions, predictors)
    axes[0].set_ylabel("Candidate predictor")
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", labelleft=False)
    figure.suptitle("Predictor selection frequency", fontsize=17, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Plot enet and ga predictor-selection frequencies.")
    parser.add_argument("--input", type=Path, default=Path("data03/pgroup01_out_2026okt20_21h.zip"))
    parser.add_argument("--output", type=Path, default=Path("data05/figS3_selection_frequency.png"))
    parser.add_argument("--n-folds", type=int, default=25)
    args = parser.parse_args()
    if args.n_folds < 1:
        parser.error("--n-folds must be positive")

    table, predictors = load_frequency_data(args.input, args.n_folds)
    create_figure(table, predictors, args.output, args.n_folds)
    print(f"Figure saved: {args.output}")
    for selector in SELECTORS:
        values = table.loc[table["selector"] == selector, "n_selected"]
        print(f"{selector}: {len(values)} predictors across {args.n_folds} folds; selected counts={values.sum()}")


if __name__ == "__main__":
    main()