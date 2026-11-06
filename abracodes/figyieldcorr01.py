import argparse
import posixpath
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
STAT_ORDER = {"mean": 0, "median": 1, "std": 2, "auc": 3}
INDEX_ORDER = {"NDVI": 0, "GNDVI": 1, "NDRE": 2, "OSAVI": 3, "WDRVI": 4}
SUBSET_ORDER = ["Pooled", "Season 2024", "Season 2025"]
SUBSET_COLORS = {
    "Pooled": "#4C78A8",
    "Season 2024": "#F58518",
    "Season 2025": "#54A24B",
}
PREDICTOR_PATTERN = re.compile(r"^(?P<index>.+)_t_(?P<stat>mean|median|std|auc)$")


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
            raise ValueError(f"Worksheet {sheet_name!r} not found in {path}; available: {available}")

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


def load_figure_data(path, window):
    rows = read_xlsx_sheet(path, "Yield_Summary")
    required = {"Window", "Subset", "Predictor", "Pearson_r", "Q_BH", "N"}
    if not rows or not required.issubset(rows[0]):
        columns = list(rows[0]) if rows else []
        raise ValueError(f"Yield_Summary is missing required columns: {sorted(required - set(columns))}")

    rows = [row for row in rows if str(row["Window"]).strip() == window]
    if not rows:
        raise ValueError(f"No records found for the requested DAS window {window!r}")

    records = []
    for row in rows:
        subset = str(row["Subset"]).strip()
        predictor = str(row["Predictor"]).strip()
        match = PREDICTOR_PATTERN.fullmatch(predictor)
        if subset not in SUBSET_ORDER:
            raise ValueError(f"Unexpected subset label: {subset!r}")
        if match is None:
            raise ValueError(f"Unexpected predictor label: {predictor!r}")
        index = match.group("index")
        stat = match.group("stat")
        if index not in INDEX_ORDER:
            raise ValueError(f"Unexpected vegetation index: {index!r}")
        try:
            correlation = float(row["Pearson_r"])
            q_value = float(row["Q_BH"])
            sample_size = int(float(row["N"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric result for {subset}, {predictor}") from exc
        if not -1 <= correlation <= 1 or not 0 <= q_value <= 1 or sample_size < 1:
            raise ValueError(f"Out-of-range value for {subset}, {predictor}")
        records.append(
            {
                "subset": subset,
                "predictor": predictor,
                "index": index,
                "stat": stat,
                "r": correlation,
                "q": q_value,
                "n": sample_size,
            }
        )

    predictor_names = {
        record["predictor"]
        for record in records
    }
    if len(predictor_names) != 20:
        raise ValueError(f"Expected 20 unique predictors, found {len(predictor_names)}")
    keys = [(record["subset"], record["predictor"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate subset/predictor records found")
    counts = {subset: sum(record["subset"] == subset for record in records) for subset in SUBSET_ORDER}
    if any(count != 20 for count in counts.values()) or len(records) != 60:
        raise ValueError(f"Expected 20 predictors per subset (60 rows total); got {counts}")

    records.sort(
        key=lambda record: (
            INDEX_ORDER[record["index"]],
            STAT_ORDER[record["stat"]],
            SUBSET_ORDER.index(record["subset"]),
        )
    )
    return records


def create_figure(records, output_path, window):
    predictor_names = list(dict.fromkeys(record["predictor"] for record in records))
    predictor_y = {predictor: position for position, predictor in enumerate(predictor_names)}
    offsets = {"Pooled": 0.22, "Season 2024": 0.0, "Season 2025": -0.22}
    sample_sizes = {}
    for subset in SUBSET_ORDER:
        values = {record["n"] for record in records if record["subset"] == subset}
        sample_sizes[subset] = next(iter(values)) if len(values) == 1 else None

    figure, axis = plt.subplots(figsize=(10.8, 9.0))
    for position in range(3, len(predictor_names) - 1, 4):
        axis.axhline(position + 0.5, color="#D9D9D9", linewidth=0.8, zorder=0)
    axis.axvline(0, color="#444444", linewidth=1.0, linestyle="--", zorder=1)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#E8E8E8", linewidth=0.7)
## **** **** ****
## Például 4-es blokkonként váltakozó háttérsávok
#    for i in range(0, len(predictor_names), 4):
#        if (i // 4) % 2 == 1:  # Minden második blokk
#            axis.axhspan(i - 0.5, i + 3.5, color="#F2F2F2", zorder=0)
    colors = ["#FFFFFF", "#F7F7F7", "#EFEFEF", "#E7E7E7"]

    for i in range(len(predictor_names)):
        color_index = i % 4
        axis.axhspan(i - 0.5, i + 0.5, color=colors[color_index], zorder=0)

## **** **** ****            
    for record in records:
        subset = record["subset"]
        is_significant = record["q"] < 0.05
        axis.scatter(
            record["r"],
            predictor_y[record["predictor"]] + offsets[subset],
            s=43,
            marker="o",
            facecolors=SUBSET_COLORS[subset] if is_significant else "white",
            edgecolors=SUBSET_COLORS[subset],
            linewidths=1.35,
            zorder=3,
        )

    axis.set_yticks(range(len(predictor_names)))
    axis.set_yticklabels(predictor_names, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlim(-1.02, 1.02)
    axis.set_xticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    axis.set_xlabel("Pearson correlation coefficient (r)")
    axis.set_ylabel("Temporal vegetation-index descriptor")
    axis.set_title(f"Correlation with grain yield ({window} DAS)", loc="left", fontsize=12, pad=14)

    subset_handles = []
    for subset in SUBSET_ORDER:
        sample_size = sample_sizes[subset]
        label = subset if sample_size is None else f"{subset} (n = {sample_size})"
        subset_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=SUBSET_COLORS[subset],
                markeredgecolor=SUBSET_COLORS[subset],
                markersize=6,
                label=label,
            )
        )
    axis.legend(
        handles=subset_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    figure.text(
        0.5,
        0.115,
        "Filled markers: BH-adjusted q < 0.05; correction across unique off-diagonal pairs within each subset.",
        ha="center",
        fontsize=8.5,
    )
    figure.subplots_adjust(left=0.24, right=0.98, top=0.93, bottom=0.21)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Plot pooled and season-specific yield correlations.")
    parser.add_argument("--input", type=Path, default=Path("data05/idopears04_FDR_N.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("data05/fig04_yield_correlations.png"))
    parser.add_argument("--window", default="190-245")
    args = parser.parse_args()

    records = load_figure_data(args.input, args.window)
    create_figure(records, args.output, args.window)
    print(f"Figure saved: {args.output}")
    for subset in SUBSET_ORDER:
        subset_rows = [record for record in records if record["subset"] == subset]
        n_values = sorted({record["n"] for record in subset_rows})
        significant = sum(record["q"] < 0.05 for record in subset_rows)
        print(f"{subset}: {len(subset_rows)} predictors; N={n_values}; q<0.05: {significant}")


if __name__ == "__main__":
    main()
