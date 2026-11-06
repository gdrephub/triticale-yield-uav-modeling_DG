import os
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# ==========================================
# FELHASZNÁLÓI PARAMÉTEREK ÉS ÚTVONALAK
# ==========================================
WDIR = r"c:\dg\kepelemzes01\rawdata\kimenet\\"
INPUT_CSV = os.path.join(WDIR, "osszes_v30a.csv")
HOZAM_XLSX = os.path.join(WDIR, "hozam.xlsx")
EXCEL_OUTPUT = os.path.join(WDIR, "idopears04.xlsx")
FDR_EXCEL_OUTPUT = os.path.join(WDIR, "idopears04_FDR_N.xlsx")

# Vetési dátumok évjáratonként
VETESI_DATUMOK = {
    2024: "2023-10-17",
    2025: "2024-09-26"
}

# Vizsgálandó időablakok listája (kezdő és záró nap a vetéstől számítva)
IDOABLAKOK = [
#    (190, 237),
    (190, 245)
    # Ide további ablakok is felvehetők, pl. (190, 220), (200, 245) stb.
]

# Kiválasztott 5 core index
SELECTED_INDICES = ['NDVI', 'GNDVI', 'NDRE', 'OSAVI', 'WDRVI']


def calculate_pearson_matrix(data, numeric_cols):
    """
    Kiszámítja a Pearson-korrelációs, p-érték- és elemszámmátrixot.
    """
    corr_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols, dtype=float)
    pval_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols, dtype=float)
    n_matrix = pd.DataFrame(index=numeric_cols, columns=numeric_cols, dtype="Int64")
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            if col1 == col2:
                valid_count = data[col1].dropna().count()
                n_matrix.loc[col1, col2] = valid_count
                if valid_count > 2 and float(data[col1].std()) > 0.0:
                    corr_matrix.loc[col1, col2] = 1.0
                    pval_matrix.loc[col1, col2] = 0.0
                else:
                    corr_matrix.loc[col1, col2] = np.nan
                    pval_matrix.loc[col1, col2] = np.nan
                continue
                
            valid_data = data[[col1, col2]].dropna()
            n_matrix.loc[col1, col2] = len(valid_data)
            if len(valid_data) > 2:
                x = valid_data[col1].to_numpy(dtype=float)
                y = valid_data[col2].to_numpy(dtype=float)
                
                std1 = np.std(x, ddof=1)
                std2 = np.std(y, ddof=1)
                
                if std1 > 0.0 and std2 > 0.0:
                    res = stats.pearsonr(x, y)
                    if hasattr(res, 'statistic'):
                        r_val = res.statistic
                        p_val = res.pvalue
                    else:
                        r_val, p_val = res[0], res[1]
                    
                    corr_matrix.loc[col1, col2] = float(np.asarray(r_val).item())
                    pval_matrix.loc[col1, col2] = float(np.asarray(p_val).item())
                    continue
            
            corr_matrix.loc[col1, col2] = np.nan
            pval_matrix.loc[col1, col2] = np.nan
                
    return corr_matrix, pval_matrix, n_matrix


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))

    if finite_indices.size == 0:
        return q_values

    finite_p_values = p_values[finite_indices]
    order = np.argsort(finite_p_values)
    ranked_p_values = finite_p_values[order]
    test_count = len(ranked_p_values)
    adjusted = ranked_p_values * test_count / np.arange(1, test_count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q_values[finite_indices[order]] = np.minimum(adjusted, 1.0)
    return q_values


def build_fdr_pair_table(corr_matrix, pval_matrix, n_matrix, start_nap, end_nap, subset_label):
    columns = ["Window", "Subset", "Variable_1", "Variable_2", "Pearson_r", "P_raw", "Q_BH", "N"]
    rows = []
    variables = list(corr_matrix.columns)

    for first_index in range(len(variables)):
        for second_index in range(first_index + 1, len(variables)):
            pair_n = n_matrix.iloc[first_index, second_index]
            rows.append({
                "Window": f"{start_nap}-{end_nap}",
                "Subset": subset_label,
                "Variable_1": variables[first_index],
                "Variable_2": variables[second_index],
                "Pearson_r": corr_matrix.iloc[first_index, second_index],
                "P_raw": pval_matrix.iloc[first_index, second_index],
                "N": int(pair_n) if pd.notna(pair_n) else 0
            })
    pair_table = pd.DataFrame(rows, columns=columns)
    pair_table["Q_BH"] = benjamini_hochberg(pair_table["P_raw"].to_numpy(dtype=float))
    return pair_table


def build_bh_q_matrix(pval_matrix):
    """A teljes off-diagonális korrelációs mátrix BH-korrigált q-mátrixa."""
    p_values = pval_matrix.to_numpy(dtype=float)
    q_values = np.full(p_values.shape, np.nan, dtype=float)
    upper_indices = np.triu_indices_from(p_values, k=1)
    upper_q_values = benjamini_hochberg(p_values[upper_indices])
    q_values[upper_indices] = upper_q_values
    q_values[(upper_indices[1], upper_indices[0])] = upper_q_values
    return pd.DataFrame(q_values, index=pval_matrix.index, columns=pval_matrix.columns)



def save_correlation_heatmap(corr_matrix, pval_matrix, title, filename_prefix, start_nap, end_nap, subset_label):
    """
    Elmenti az eredeti Pearson-r heatmapet és a külön BH-q heatmapet.
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    plot_corr_matrix = corr_matrix.rename(index={"hozam": "yield"}, columns={"hozam": "yield"})
    mask = np.zeros_like(plot_corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=0)] = True
    
    sns.heatmap(plot_corr_matrix.astype(float), mask=mask, annot=False, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    
    n = len(plot_corr_matrix)
    for i in range(n):
        rect = Rectangle((i + 0.1, i + 0.1), 0.8, 0.8, facecolor='white', edgecolor='#666666', linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        
    plt.title(f"Pearson Correlation Heatmap (Lower Triangle)\n{title} (Window: {start_nap}-{end_nap} DAS)", fontsize=12, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    heatmap_filename = os.path.join(WDIR, f"{filename_prefix}_{start_nap}_{end_nap}_{subset_label}.png")
    plt.savefig(heatmap_filename, dpi=300)
    plt.close()
    print(f"Heatmap saved: {heatmap_filename}")

    q_matrix = build_bh_q_matrix(pval_matrix)
    q_values = q_matrix.to_numpy(dtype=float)
    plot_q_matrix = q_matrix.rename(index={"hozam": "yield"}, columns={"hozam": "yield"})
    q_mask = np.zeros_like(q_values, dtype=bool)
    q_mask[np.triu_indices_from(q_mask, k=0)] = True

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    sns.heatmap(plot_q_matrix, mask=q_mask, annot=False, cmap="coolwarm", vmin=0, vmax=1, center=0.5,
                square=True, linewidths=.5, cbar_kws={"shrink": .8, "label": "BH-adjusted q-value"}, ax=ax)

    n = len(plot_q_matrix)
    for i in range(n):
        rect = Rectangle((i + 0.1, i + 0.1), 0.8, 0.8, facecolor='white', edgecolor='#666666', linewidth=1.2, zorder=3)
        ax.add_patch(rect)

    for i in range(n):
        for j in range(i):
            q_value = q_values[i, j]
            if not np.isfinite(q_value):
                continue
            if q_value < 0.001:
                stars = "***"
            elif q_value < 0.01:
                stars = "**"
            elif q_value < 0.05:
                stars = "*"
            else:
                stars = ""
            if stars:
                ax.text(j + 0.5, i + 0.5, stars, ha="center", va="center", color="white",
                        fontsize=10, fontweight="bold", zorder=4)

    plt.title(f"Benjamini–Hochberg FDR-Adjusted q-Value Heatmap (Lower Triangle)\n{title} (Window: {start_nap}-{end_nap} DAS)", fontsize=12, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    q_filename = os.path.join(WDIR, f"heatmap_q_values_{start_nap}_{end_nap}_{subset_label}.png")
    plt.savefig(q_filename, dpi=300)
    plt.close()
    print(f"BH-adjusted q-value heatmap saved: {q_filename}")

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    sns.heatmap(plot_corr_matrix.astype(float), mask=mask, annot=False, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)

    for diagonal_index in range(n):
        rect = Rectangle((diagonal_index + 0.1, diagonal_index + 0.1), 0.8, 0.8, facecolor='white', edgecolor='#666666', linewidth=1.2, zorder=3)
        ax.add_patch(rect)

    for row_index in range(n):
        for column_index in range(row_index):
            q_value = q_values[row_index, column_index]
            if not np.isfinite(q_value):
                continue
            if q_value < 0.001:
                stars = "***"
            elif q_value < 0.01:
                stars = "**"
            elif q_value < 0.05:
                stars = "*"
            else:
                stars = ""
            if stars:
                ax.text(column_index + 0.5, row_index + 0.5, stars, ha="center", va="center", color="white",
                        fontsize=10, fontweight="bold", zorder=4)

    plt.title(f"Pearson's r Heatmap with BH-Adjusted q-Value Significance (Lower Triangle)\n{title} (Window: {start_nap}-{end_nap} DAS)\nStars: q < 0.05 (*), q < 0.01 (**), q < 0.001 (***)", fontsize=11, pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    q_star_filename = os.path.join(WDIR, f"{filename_prefix}_q_stars_{start_nap}_{end_nap}_{subset_label}.png")
    plt.savefig(q_star_filename, dpi=300)
    plt.close()
    print(f"Pearson-r heatmap with q-value significance stars saved: {q_star_filename}")


def save_combined_comparison_heatmap(corr_dict, start_nap, end_nap):
    """
    Létrehozza és elmenti a pooled, 2024 és 2025 adatsorok heatmapjeit 
    egy közös ábrán (1 sor, 3 oszlop elrendezésben), megnövelt mérettel és optimalizált tengelyfeliratokkal.
    """
    fig, axes = plt.subplots(1, 3, figsize=(25, 8.5))
    
    titles = ["Pooled Dataset", "Season 2024", "Season 2025"]
    keys = ["Pooled", "2024", "2025"]
    
    for idx, (key, title) in enumerate(zip(keys, titles)):
        ax = axes[idx]
        corr_matrix = corr_dict.get(key)
        
        if corr_matrix is None or corr_matrix.empty:
            ax.set_visible(False)
            continue
            
        display_matrix = corr_matrix.rename(index={"hozam": "yield"}, columns={"hozam": "yield"})
        mask = np.zeros_like(display_matrix, dtype=bool)
        mask[np.triu_indices_from(mask, k=0)] = True
        
        cbar_flag = (idx == 2)
        
        sns.heatmap(display_matrix.astype(float), mask=mask, annot=False, cmap="coolwarm", 
                    vmin=-1, vmax=1, center=0, square=True, linewidths=.5, 
                    cbar=cbar_flag, cbar_kws={"shrink": .8}, ax=ax)
        
        n = len(display_matrix)
        for i in range(n):
            rect = Rectangle((i + 0.1, i + 0.1), 0.8, 0.8, facecolor='white', edgecolor='#666666', linewidth=1.2, zorder=3)
            ax.add_patch(rect)
            
        ax.set_title(f"{title}", fontsize=13, pad=15)
        ax.tick_params(axis='x', labelsize=11, rotation=45)
        ax.tick_params(axis='y', labelsize=11)
        plt.setp(ax.get_xticklabels(), ha='right')

    plt.suptitle(f"Pearson Correlation Heatmaps Comparison (Window: {start_nap}-{end_nap} DAS)", fontsize=15, y=0.93, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.87])
    
    combined_filename = os.path.join(WDIR, f"pcorrheatmap_{start_nap}_{end_nap}_combined_comparison.png")
    plt.savefig(combined_filename, dpi=300)
    plt.close()
    print(f"Combined comparison heatmap saved: {combined_filename}")


def process_window_data(df, start_nap, end_nap, date_col, parcel_col, season_col, mean_cols, hozam_df):
    """
    Kiszámítja az adott időablakra vonatkozó parcellánkénti statisztikákat,
    ahol az időbeli aggregációt a '_t_' jelölés (pl. NDVI_t_std) mutatja, majd összefésüli a hozam adatokkal,
    és beállítja a 'season', 'n_parcella', 'hozam' oszlopokat az elejére.
    """
    results = []
    
    for parcella_id, group in df.groupby(parcel_col):
        if season_col in group.columns:
            seasons = group[season_col].dropna().unique()
        else:
            seasons = group[date_col].dt.year.unique()
            
        for season in seasons:
            season_val = int(season)
            if season_col in group.columns:
                season_group = group[group[season_col].astype(str).str.contains(str(season_val))].copy()
            else:
                season_group = group[group[date_col].dt.year == season_val].copy()
                
            if season_group.empty:
                continue
                
            vetesi_str = VETESI_DATUMOK.get(season_val, "2023-10-17")
            vetesi_date = pd.to_datetime(vetesi_str)
            
            season_group['days_after_sowing'] = (season_group[date_col] - vetesi_date).dt.days
            
            window_group = season_group[
                (season_group['days_after_sowing'] >= start_nap) & 
                (season_group['days_after_sowing'] <= end_nap)
            ].sort_values(by=date_col)
            
            if window_group.empty:
                continue
                
            row_data = {
                'season': season_val,
                'n_parcella': parcella_id
            }
            
            for col in mean_cols:
                base_name = col[:-5] if col.endswith('_mean') else col
                
                # Csak a kiválasztott 5 core index statisztikáit számoljuk ki/tartjuk meg
                if base_name not in SELECTED_INDICES:
                    continue
                
                series = window_group[col].dropna()
                if len(series) == 0:
                    row_data[f"{base_name}_t_mean"] = np.nan
                    row_data[f"{base_name}_t_median"] = np.nan
                    row_data[f"{base_name}_t_std"] = np.nan
                    row_data[f"{base_name}_t_auc"] = np.nan
                    continue
                    
                vals = series.values
                days = window_group.loc[series.index, 'days_after_sowing'].values
                
                row_data[f"{base_name}_t_mean"] = np.mean(vals)
                row_data[f"{base_name}_t_median"] = np.median(vals)
                row_data[f"{base_name}_t_std"] = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
                
                if len(vals) > 1:
                    raw_auc = np.trapezoid(vals, days) if hasattr(np, 'trapezoid') else np.trapz(vals, days)
                    window_length = days[-1] - days[0]
                    row_data[f"{base_name}_t_auc"] = raw_auc / window_length if window_length > 0 else raw_auc
                else:
                    row_data[f"{base_name}_t_auc"] = vals[0] if len(vals) == 1 else 0.0
                    
            results.append(row_data)
            
    stats_df = pd.DataFrame(results)
    if stats_df.empty:
        return stats_df
        
    merged_df = pd.merge(stats_df, hozam_df, on=['season', 'n_parcella'], how='left')
    
    # Oszlopok átrendezése: season, n_parcella, hozam legyenek elöl
    if 'hozam' in merged_df.columns:
        fixed_cols = ['season', 'n_parcella', 'hozam']
        other_cols = [c for c in merged_df.columns if c not in fixed_cols]
        merged_df = merged_df[fixed_cols + other_cols]
        
    return merged_df


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: File not found: {INPUT_CSV}")
        return
    if not os.path.exists(HOZAM_XLSX):
        print(f"Error: Yield file not found: {HOZAM_XLSX}")
        return
        
    hozam_df = pd.read_excel(HOZAM_XLSX)
    hozam_df.columns = hozam_df.columns.str.replace('\ufeff', '').str.strip()
    
    df = pd.read_csv(INPUT_CSV, sep=None, engine='python')
    df.columns = df.columns.str.replace('\ufeff', '').str.strip()
    
    date_col = next((c for c in df.columns if 'date' in c.lower() or 'datum' in c.lower()), 'date')
    parcel_col = next((c for c in df.columns if 'parcella' in c.lower() or 'parcel' in c.lower() or 'id' in c.lower()), 'parcel_id')
    season_col = next((c for c in df.columns if 'season' in c.lower() or 'ev' in c.lower()), 'season')
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    mean_cols = [c for c in df.columns if c.endswith('_mean')]
    for col in mean_cols:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    yield_summary_rows = []
    fdr_tables = {}
    
    with pd.ExcelWriter(EXCEL_OUTPUT, engine='openpyxl') as writer:
        for start_nap, end_nap in IDOABLAKOK:
            print(f"\n--- Processing window: {start_nap} - {end_nap} days ---")
            window_df = process_window_data(df, start_nap, end_nap, date_col, parcel_col, season_col, mean_cols, hozam_df)
            
            if window_df.empty:
                print(f"Warning: No data found for window {start_nap}-{end_nap}.")
                continue
                
            # Excel-barát CSV mentése a kért formátumban
            csv_output_filename = os.path.join(WDIR, f"prediktor5_{start_nap}_{end_nap}_hozam.csv")
            window_df.to_csv(
                csv_output_filename,
                index=False,
                sep=";",
                decimal=",",
                encoding="utf-8-sig"
            )
            print(f"Excel-friendly CSV saved: {csv_output_filename}")
                
            predictor_cols = []
            for col in window_df.columns:
                if any(col.startswith(idx + '_t_') for idx in SELECTED_INDICES):
                    predictor_cols.append(col)
                    
            numeric_cols = predictor_cols + ['hozam'] if 'hozam' in window_df.columns else predictor_cols
            seasons = sorted(window_df['season'].unique())
            
            corr_matrices_for_comparison = {}
            
            # 1. Pooled correlations and lower-triangle heatmap
            corr_pool, pval_pool, n_pool = calculate_pearson_matrix(window_df, numeric_cols)
            corr_pool.to_excel(writer, sheet_name=f'P_{start_nap}_{end_nap}_Corr')
            pval_pool.to_excel(writer, sheet_name=f'P_{start_nap}_{end_nap}_Pval')
            pooled_fdr_table = build_fdr_pair_table(corr_pool, pval_pool, n_pool, start_nap, end_nap, "Pooled")
            fdr_tables[f'Pooled_{start_nap}_{end_nap}'] = pooled_fdr_table
            
            save_correlation_heatmap(corr_pool, pval_pool, "Pooled Dataset", "pcorrheatmap", start_nap, end_nap, "Pooled")
            corr_matrices_for_comparison["Pooled"] = corr_pool
            
            if 'hozam' in corr_pool.columns:
                for pred in predictor_cols:
                    yield_summary_rows.append({
                        'Window': f"{start_nap}-{end_nap}",
                        'Season': 'Pooled',
                        'Predictor': pred,
                        'Pearson_r': corr_pool.loc[pred, 'hozam'],
                        'P_value': pval_pool.loc[pred, 'hozam']
                    })
            
            # 2. Seasonal correlations and lower-triangle heatmaps
            for season in seasons:
                season_subset = window_df[window_df['season'] == season]
                corr_season, pval_season, n_season = calculate_pearson_matrix(season_subset, numeric_cols)
                season_fdr_table = build_fdr_pair_table(corr_season, pval_season, n_season, start_nap, end_nap, f"Season {season}")
                fdr_tables[f'S{season}_{start_nap}_{end_nap}'] = season_fdr_table
                corr_season.to_excel(writer, sheet_name=f'S{season}_{start_nap}_{end_nap}_C')
                pval_season.to_excel(writer, sheet_name=f'S{season}_{start_nap}_{end_nap}_P')
                
                save_correlation_heatmap(corr_season, pval_season, f"Season {season}", "pcorrheatmap", start_nap, end_nap, f"Season_{season}")
                corr_matrices_for_comparison[str(season)] = corr_season
                
                if 'hozam' in corr_season.columns:
                    for pred in predictor_cols:
                        yield_summary_rows.append({
                            'Window': f"{start_nap}-{end_nap}",
                            'Season': str(season),
                            'Predictor': pred,
                            'Pearson_r': corr_season.loc[pred, 'hozam'],
                            'P_value': pval_season.loc[pred, 'hozam']
                        })
            
            # 3. Egyesített (Pooled, 2024, 2025) összehasonlító ábra mentése
            save_combined_comparison_heatmap(corr_matrices_for_comparison, start_nap, end_nap)
                        
        # 4. Dedicated yield correlation summary sheet
        if yield_summary_rows:
            summary_df = pd.DataFrame(yield_summary_rows)
            summary_df.to_excel(writer, sheet_name='Yield_Correlations_Summary', index=False)
            print("\nYield correlation summary sheet successfully created in Excel!")
            
    all_fdr_table = pd.concat(fdr_tables.values(), ignore_index=True) if fdr_tables else pd.DataFrame()
    if all_fdr_table.empty:
        yield_fdr_table = pd.DataFrame(columns=["Window", "Subset", "Predictor", "Pearson_r", "P_raw", "Q_BH", "N"])
    else:
        yield_mask = all_fdr_table["Variable_1"].eq("hozam") | all_fdr_table["Variable_2"].eq("hozam")
        yield_fdr_table = all_fdr_table.loc[yield_mask].copy()
        yield_fdr_table["Predictor"] = np.where(yield_fdr_table["Variable_1"].eq("hozam"), yield_fdr_table["Variable_2"], yield_fdr_table["Variable_1"])
        yield_fdr_table = yield_fdr_table[["Window", "Subset", "Predictor", "Pearson_r", "P_raw", "Q_BH", "N"]]

    fdr_method_df = pd.DataFrame([
        {"Field": "Correction family", "Description": "Benjamini-Hochberg correction is applied separately within each window and dataset subset (pooled or season), across all unique off-diagonal pairs in that correlation matrix. Symmetric duplicates and diagonal values are excluded."},
        {"Field": "N", "Description": "Number of pairwise-complete observations after dropping rows missing either variable."},
        {"Field": "P_raw", "Description": "Unadjusted two-sided Pearson correlation p-value."},
        {"Field": "Q_BH", "Description": "Benjamini-Hochberg FDR-adjusted p-value within the stated correction family; undefined raw p-values remain blank."},
        {"Field": "Yield_Summary", "Description": "Yield-correlation rows filtered from the pair tables; Q_BH uses the full matrix correction family, not a separate yield-only family."}
    ])
    with pd.ExcelWriter(FDR_EXCEL_OUTPUT, engine="openpyxl") as fdr_writer:
        fdr_method_df.to_excel(fdr_writer, sheet_name="Method", index=False)
        for sheet_name, pair_table in fdr_tables.items():
            pair_table.to_excel(fdr_writer, sheet_name=sheet_name[:31], index=False)
        yield_fdr_table.to_excel(fdr_writer, sheet_name="Yield_Summary", index=False)

    print(f"FDR-corrected p-values and pairwise N saved to: {FDR_EXCEL_OUTPUT}")
    print(f"\nAll analyses and heatmap generations completed successfully! Results saved to: {EXCEL_OUTPUT}")


if __name__ == "__main__":
    main()
