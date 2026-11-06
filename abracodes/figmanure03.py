import argparse
import posixpath
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
SEASONS = [2024, 2025]
TREATMENTS = [0, 1]
TREATMENT_LABELS = {0: "Control", 1: "Manured"}
TREATMENT_COLORS = {0: "#4C78A8", 1: "#E07A3F"}


def _column_index(cell_reference):
    value = 0
    for character in cell_reference:
        if not character.isalpha():
            break
        value = value * 26 + ord(character.upper()) - ord("A") + 1
    return value - 1


def read_xlsx_sheet(path, sheet_name):
    with ZipFile(path) as workbook:
        workbook_xml = ET.fromstring(workbook.read("xl/workbook.xml"))
        relationships_xml = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        relationships = {
            relationship.attrib["Id"]: relationship.attrib["Target"]
            for relationship in relationships_xml.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
        }
        sheet = next(
            (
                item
                for item in workbook_xml.findall(f"{{{MAIN_NS}}}sheets/{{{MAIN_NS}}}sheet")
                if item.attrib["name"] == sheet_name
            ),
            None,
        )
        if sheet is None:
            available = [
                item.attrib["name"]
                for item in workbook_xml.findall(f"{{{MAIN_NS}}}sheets/{{{MAIN_NS}}}sheet")
            ]
            raise ValueError(f"Worksheet {sheet_name!r} not found; available: {available}")

        target = relationships[sheet.attrib[f"{{{REL_NS}}}id"]]
        sheet_path = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_xml = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared_strings = [
                "".join(node.text or "" for node in item.iter(f"{{{MAIN_NS}}}t"))
                for item in shared_xml.findall(f"{{{MAIN_NS}}}si")
            ]

        worksheet = ET.fromstring(workbook.read(sheet_path))
        rows = []
        for row in worksheet.findall(f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row"):
            cells = {}
            for cell in row.findall(f"{{{MAIN_NS}}}c"):
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    value = "".join(node.text or "" for node in cell.iter(f"{{{MAIN_NS}}}t"))
                else:
                    value_node = cell.find(f"{{{MAIN_NS}}}v")
                    value = value_node.text if value_node is not None else ""
                    if cell_type == "s" and value:
                        value = shared_strings[int(value)]
                cells[_column_index(cell.attrib["r"])] = value
            if cells:
                rows.append([cells.get(index, "") for index in range(max(cells) + 1)])

    if not rows:
        raise ValueError(f"Worksheet {sheet_name!r} is empty in {path}")
    headers = [str(value).strip() for value in rows[0]]
    return [dict(zip(headers, row + [""] * (len(headers) - len(row)))) for row in rows[1:]]


def load_plot_data(predictor_path, treatment_path, sheet_name):
    predictors = pd.read_csv(predictor_path, sep=";", decimal=",", encoding="utf-8-sig")
    treatments = pd.DataFrame(read_xlsx_sheet(treatment_path, sheet_name))
    required_predictors = {"season", "n_parcella", "hozam", "NDRE_t_std"}
    required_treatments = {"season", "n_parcella", "kezeles"}
    if not required_predictors.issubset(predictors.columns):
        raise ValueError(f"Predictor CSV is missing columns: {sorted(required_predictors - set(predictors.columns))}")
    if not required_treatments.issubset(treatments.columns):
        raise ValueError(f"Treatment worksheet is missing columns: {sorted(required_treatments - set(treatments.columns))}")

    for frame, columns in (
        (predictors, ["season", "n_parcella", "hozam", "NDRE_t_std"]),
        (treatments, ["season", "n_parcella", "kezeles"]),
    ):
        for column in columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[columns].isna().any().any():
            raise ValueError(f"Missing or nonnumeric values in required columns: {columns}")

    key = ["season", "n_parcella"]
    if predictors.duplicated(key).any() or treatments.duplicated(key).any():
        raise ValueError("Duplicate season/plot keys prevent an unambiguous merge")
    data = predictors.merge(
        treatments[key + ["kezeles"]], on=key, how="outer", indicator=True, validate="one_to_one"
    )
    if not (data["_merge"] == "both").all():
        unmatched = data.loc[data["_merge"] != "both", key + ["_merge"]]
        raise ValueError(f"Predictor and treatment records do not match: {unmatched.to_dict('records')}")
    data = data.drop(columns="_merge")
    if sorted(data["season"].unique().tolist()) != SEASONS:
        raise ValueError(f"Expected seasons {SEASONS}; found {sorted(data['season'].unique().tolist())}")
    if not set(data["kezeles"].unique()).issubset(TREATMENTS):
        raise ValueError(f"Unexpected treatment values: {sorted(data['kezeles'].unique().tolist())}")

    treatment_by_plot = data.pivot(index="n_parcella", columns="season", values="kezeles")
    if treatment_by_plot[SEASONS].isna().any().any():
        raise ValueError("Every plot must have records in both seasons for the connected-plot panel")
    if (treatment_by_plot[SEASONS[0]] != treatment_by_plot[SEASONS[1]]).any():
        raise ValueError("Treatment assignments changed between seasons; do not connect plot trajectories")
    return data.sort_values(["season", "kezeles", "n_parcella"]).reset_index(drop=True)


def _draw_box_panel(axis, data, value_column, ylabel, panel_label):
    positions = {
        (season, treatment): season_position + treatment_offset
        for season_position, season in enumerate(SEASONS)
        for treatment, treatment_offset in zip(TREATMENTS, [-0.19, 0.19])
    }
    grouped_values = [
        data.loc[(data["season"] == season) & (data["kezeles"] == treatment), value_column].to_numpy()
        for season in SEASONS
        for treatment in TREATMENTS
    ]
    grouped_positions = [positions[(season, treatment)] for season in SEASONS for treatment in TREATMENTS]
    boxes = axis.boxplot(
        grouped_values,
        positions=grouped_positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.5},
        whiskerprops={"color": "#555555", "linewidth": 1.1},
        capprops={"color": "#555555", "linewidth": 1.1},
    )
    for box, treatment in zip(boxes["boxes"], [0, 1, 0, 1]):
        box.set_facecolor(TREATMENT_COLORS[treatment])
        box.set_alpha(0.28)
        box.set_edgecolor(TREATMENT_COLORS[treatment])
        box.set_linewidth(1.4)

    for season in SEASONS:
        for treatment in TREATMENTS:
            subset = data.loc[(data["season"] == season) & (data["kezeles"] == treatment)]
            position = positions[(season, treatment)]
            jitter = subset["n_parcella"].map(lambda plot: ((int(plot) * 37) % 9 - 4) * 0.012)
            axis.scatter(
                np.full(len(subset), position) + jitter,
                subset[value_column],
                s=37,
                color=TREATMENT_COLORS[treatment],
                edgecolor="white",
                linewidth=0.55,
                alpha=0.9,
                zorder=3,
            )

    axis.set_xticks(grouped_positions)
    axis.set_xticklabels([f"{season}\n{TREATMENT_LABELS[treatment]}" for season in SEASONS for treatment in TREATMENTS])
    axis.set_xlim(-0.52, 1.52)
    axis.set_ylabel(ylabel)
    axis.set_title(panel_label, loc="left", fontweight="bold", pad=9)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    axis.set_axisbelow(True)


def _draw_interaction_panel(axis, data):
    plot_data = data.pivot(index="n_parcella", columns="season", values="hozam")
    for _, values in plot_data.iterrows():
        axis.plot(SEASONS, [values[season] for season in SEASONS], color="#A8A8A8", linewidth=0.8, alpha=0.38, zorder=1)
        axis.scatter(SEASONS, [values[season] for season in SEASONS], color="#A8A8A8", s=16, alpha=0.45, zorder=2)

    summary = data.groupby(["season", "kezeles"])["hozam"].agg(["mean", "std", "count"])
    offsets = {0: -0.035, 1: 0.035}
    for treatment in TREATMENTS:
        means = []
        errors = []
        x_values = []
        for season in SEASONS:
            row = summary.loc[(season, treatment)]
            means.append(row["mean"])
            errors.append(row["std"] / np.sqrt(row["count"]))
            x_values.append(season + offsets[treatment])
        axis.errorbar(
            x_values,
            means,
            yerr=errors,
            color=TREATMENT_COLORS[treatment],
            marker="o",
            markersize=8,
            linewidth=2.3,
            capsize=5,
            elinewidth=1.6,
            label=TREATMENT_LABELS[treatment],
            zorder=4,
        )

    axis.set_xticks(SEASONS, [str(season) for season in SEASONS])
    axis.set_xlim(2023.55, 2025.45)
    axis.set_ylabel("Mean grain yield (kg ha$^{-1}$) ± SE")
    axis.set_title("(b) Treatment × season interaction", loc="left", fontweight="bold", pad=9)
    axis.grid(axis="y", color="#E6E6E6", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(title="Treatment", frameon=False, loc="best")


def create_figure(data, output_path):
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11})
    figure = plt.figure(figsize=(13.2, 10.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 0.95])
    yield_axis = figure.add_subplot(grid[0, 0])
    interaction_axis = figure.add_subplot(grid[0, 1])
    index_axis = figure.add_subplot(grid[1, :])

    _draw_box_panel(yield_axis, data, "hozam", "Grain yield (kg ha$^{-1}$)", "(a) Grain yield")
    _draw_interaction_panel(interaction_axis, data)
    _draw_box_panel(index_axis, data, "NDRE_t_std", "NDRE temporal standard deviation", "(c) NDRE$_{t\\_std}$")

    figure.suptitle("Management groups by season", fontsize=17, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Plot season-specific management-group outcomes.")
    parser.add_argument("--input", type=Path, default=Path("prediktor5_190_245_hozam.csv"))
    parser.add_argument("--treatment", type=Path, default=Path("data04/parcella_kezeles.xlsx"))
    parser.add_argument("--sheet", default="Munka1")
    parser.add_argument("--output", type=Path, default=Path("data05/fig07_management_groups.png"))
    args = parser.parse_args()

    data = load_plot_data(args.input, args.treatment, args.sheet)
    create_figure(data, args.output)
    print(f"Figure saved: {args.output}")
    for season in SEASONS:
        season_data = data[data["season"] == season]
        counts = season_data.groupby("kezeles").size().to_dict()
        print(f"{season}: plots per treatment = {counts}")


if __name__ == "__main__":
    main()