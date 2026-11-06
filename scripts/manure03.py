#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
manure03.py - Trágyázási kezelés (kezeles) hatásának elemzése
=============================================================
A manure02.py TELJES tartalmát megtartja (1-5. lépés, azonos kimenetek), és kiegészíti
az alábbi elemzésekkel (6-11. lépés), amelyek a bírálói kérésekre és a projektbeli
megbeszélésre épülnek.

Az eredeti manure02.py lépései (változatlanok):
1. Beolvassa a parcellák trágyázási kezelési adatait és a prediktorokat.
2. Összefésüli a két adatállományt a szezon és parcellaszám alapján.
3. Student t-próba, Welch-féle t-próba és Mann-Whitney U-próba (az évek ÖSSZEVONVA).
4. Excel-barát CSV formátumban menti az eredményeket.
5. Minden prediktorra és a hozamra doboz- és hegedűdiagram + CSV, interakciós ábra,
   szignifikancia hőtérkép.

ÚJ lépések (manure03):
6. Elrendezés-ellenőrzés: a kezelés mintázata a parcellasorrendben, évek közötti
   egyezés, 4-es blokkok (0,0,1,1) vizsgálata.
7. Évenkénti összehasonlítás (kontroll vs. trágyázott): átlag, szórás, különbség,
   95%-os CI, Hedges-féle g, Welch és Mann-Whitney p, Holm- és BH-korrekció.
8. Évhatással korrigált összehasonlítás: additív OLS (season + kezeles), interakciós
   teszt (season x kezeles), évenként rétegzett permutációs teszt.
9. Évenkénti (within-season) kapcsolat a hozam és a prediktorok között:
   Pearson/Spearman r, parciális r a kezelés kiszűrésével, összevont "within" korreláció,
   összehasonlításul a nyers (évek közötti különbséget is tartalmazó) pooled r.
10. Blokkon belüli (szomszédos parcellák) páros összehasonlítás. FELTEVÉS: a parcellaszám
    a térbeli sorrendet követi, és a 4-es blokkokon belül 2 kontroll + 2 trágyázott van.
    Ezt a tényleges kísérleti elrendezés alapján ellenőrizni kell!
11. Új ábrák: hatásméret-erdőábra évenként, évenkénti szórásdiagram (hozam vs. NDVI_t_mean).

Jelölés: a prediktorok a cikk új jelölésével szerepelnek az új kimenetekben
(pl. NDVI_t_mean); a bemeneti CSV régi oszlopnevei (NDVI_mean_mean) is elfogadottak.
A szkript a 20 "core" prediktorral (5 index x 4 időbeli statisztika) és a hozammal
számol a bővített elemzésekben; a többszörös összehasonlítás korrekciója a 20 prediktor
családján belül történik, a hozam külön (előre meghatározott) próba.

Használat:
    python manure03.py
    python manure03.py --input-dir <mappa> --output-dir <mappa>
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_DIR = r"c:\dg\kepelemzes01\rawdata\kimenet"
OUTPUT_DIR = r"c:\dg\kepelemzes01\rawdata\kimenet\manure3out"

if not os.path.exists(INPUT_DIR):
    INPUT_DIR = "."
    OUTPUT_DIR = "manure3out"

PRED_CSV_NAME = "prediktor5_190_245_hozam.csv"
KEZELES_XLSX_NAME = "parcella_kezeles.xlsx"

# --- Bővített elemzések beállításai ---------------------------------------------------
INDICES = ["NDVI", "GNDVI", "NDRE", "OSAVI", "WDRVI"]   # a 20 core prediktor indexei
STATS = ["mean", "median", "std", "auc"]                  # időbeli statisztikák
N_PERM = 10000                                            # permutációk száma
PERM_SEED = 2026
BLOCK_SIZE = 4                                            # a (0,0,1,1) mintázat blokkmérete


def _set_paths(input_dir, output_dir):
    """Elérési utak beállítása (a parancssori argumentumok felülírhatják az alapértékeket)."""
    global INPUT_DIR, OUTPUT_DIR
    global PRED_CSV, KEZELES_XLSX
    global OUT_SUMMARY_CSV, OUT_PRED_DIFF_CSV, OUT_SEASON_YIELD_CSV, OUT_ADVANCED_TESTS_CSV
    if input_dir:
        INPUT_DIR = input_dir
    if output_dir:
        OUTPUT_DIR = output_dir
    PRED_CSV = os.path.join(INPUT_DIR, PRED_CSV_NAME)
    KEZELES_XLSX = os.path.join(INPUT_DIR, KEZELES_XLSX_NAME)
    OUT_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "manure_treatment_summary.csv")
    OUT_PRED_DIFF_CSV = os.path.join(OUTPUT_DIR, "manure_treatment_predictor_diffs.csv")
    OUT_SEASON_YIELD_CSV = os.path.join(OUTPUT_DIR, "manure_treatment_season_yield.csv")
    OUT_ADVANCED_TESTS_CSV = os.path.join(OUTPUT_DIR, "manure_advanced_tests.csv")


_set_paths(None, None)


def save_excel_csv(df, filepath):
    """Excel-barát CSV fájl mentése pontosvesszővel és vessző tizedesjellel."""
    df.to_csv(
        filepath,
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )
    print(f"Successfully saved: {filepath}")


def get_significance_stars(p):
    """Nemzetközi gyakorlat szerinti szignifikancia csillagok kerek zárójelben p-érték alapján."""
    if pd.isna(p):
        return ""
    elif p < 0.001:
        return "(***)"
    elif p < 0.01:
        return "(**)"
    elif p < 0.05:
        return "(*)"
    else:
        return "(ns)"


def to_numeric_frame(df):
    """Minden oszlop numerikussá alakítása (tizedesvessző és szöveges típus kezelése)."""
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(",", ".", regex=False).str.strip(), errors="coerce")
    return out


# =====================================================================================
# Segédfüggvények a bővített elemzésekhez
# =====================================================================================
def holm_adjust(p):
    """Holm-féle lépcsős korrekció (NaN-ok kihagyva)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pv[i])
        adj[i] = min(1.0, running)
    out[ok] = adj
    return out


def bh_adjust(p):
    """Benjamini-Hochberg FDR (q-érték)."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(1.0, ranked)
    out[ok] = adj
    return out


def resolve_core_columns(df):
    """A 20 core prediktor oszlopa: [(cikk-jelölés, tényleges oszlopnév), ...].
    Elfogadja az új (NDVI_t_mean) és a régi (NDVI_mean_mean) oszlopneveket is."""
    core = []
    for idx in INDICES:
        for st in STATS:
            label = f"{idx}_t_{st}"
            for cand in (label, f"{idx}_mean_{st}"):
                if cand in df.columns:
                    core.append((label, cand))
                    break
            else:
                print(f"  ! Figyelmeztetés: nincs oszlop a(z) {label} prediktorhoz.")
    return core


def welch_summary(a, b):
    """a = kontroll, b = trágyázott. Visszaad: különbség (b-a), 95% CI, Welch p, Hedges g (+CI)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    res = {"n_control": na, "n_manured": nb, "mean_control": np.nan, "sd_control": np.nan,
           "mean_manured": np.nan, "sd_manured": np.nan, "diff_manured_minus_control": np.nan,
           "diff_ci_low": np.nan, "diff_ci_high": np.nan, "welch_p": np.nan,
           "hedges_g": np.nan, "g_ci_low": np.nan, "g_ci_high": np.nan}
    if na < 2 or nb < 2:
        return res
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    res.update(mean_control=ma, sd_control=np.sqrt(va), mean_manured=mb, sd_manured=np.sqrt(vb),
               diff_manured_minus_control=mb - ma)
    se = np.sqrt(va / na + vb / nb)
    if se > 0:
        dfw = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
        tcrit = stats.t.ppf(0.975, dfw)
        res["diff_ci_low"] = (mb - ma) - tcrit * se
        res["diff_ci_high"] = (mb - ma) + tcrit * se
        res["welch_p"] = 2 * stats.t.sf(abs((mb - ma) / se), dfw)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp > 0:
        d = (mb - ma) / sp
        j = 1 - 3 / (4 * (na + nb) - 9)
        g = d * j
        se_g = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
        res.update(hedges_g=g, g_ci_low=g - 1.96 * se_g, g_ci_high=g + 1.96 * se_g)
    return res


def mannwhitney_p(a, b):
    try:
        return stats.mannwhitneyu(b, a, alternative="two-sided")[1]
    except ValueError:
        return np.nan


def ols_fit(X, y):
    """Egyszerű OLS: visszaad együtthatókat, SE-t és kétoldali p-értéket."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof)
    return beta, se, p, dof


def corr_with_dof(x, y, dof):
    """Pearson r és p-érték megadott szabadsági fokkal."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 4 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    r = float(np.corrcoef(x, y)[0, 1])
    if dof <= 0 or abs(r) >= 1:
        return r, np.nan
    t = r * np.sqrt(dof / (1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), dof))


def residualize_on_cells(v, cells):
    """A változóból kivonja a cellaátlagokat (pl. évenként/kezelésenként)."""
    v = pd.Series(np.asarray(v, float))
    return (v - v.groupby(np.asarray(cells)).transform("mean")).to_numpy()


# =====================================================================================
# 6. Elrendezés-ellenőrzés
# =====================================================================================
def layout_check(df):
    seasons = sorted(df["season"].unique())
    piv = df.pivot(index="n_parcella", columns="season", values="kezeles")
    piv.columns = [f"kezeles_{c}" for c in piv.columns]
    piv = piv.reset_index()
    piv["block"] = (piv["n_parcella"] - 1) // BLOCK_SIZE + 1
    same = bool((piv[[f"kezeles_{s}" for s in seasons]].nunique(axis=1) == 1).all())
    piv["same_in_all_seasons"] = (piv[[f"kezeles_{s}" for s in seasons]].nunique(axis=1) == 1).astype(int)
    save_excel_csv(piv, os.path.join(OUTPUT_DIR, "manure_layout_check.csv"))
    for s in seasons:
        pattern = "".join(str(int(v)) for v in piv[f"kezeles_{s}"])
        print(f"  Kezelés-mintázat a parcellasorrendben ({s}): {pattern}")
    print(f"  A kezelés minden évben azonos a parcellákon: {same}")
    bal = piv.groupby("block")[f"kezeles_{seasons[0]}"].agg(["sum", "count"])
    balanced = bool(((bal["sum"] * 2) == bal["count"]).all())
    print(f"  A {BLOCK_SIZE}-es blokkok kiegyensúlyozottak (fele kontroll, fele trágyázott): {balanced}")
    print("  FIGYELEM: a blokk-értelmezés csak akkor érvényes, ha a parcellaszám a térbeli sorrendet követi.")
    return piv


# =====================================================================================
# 7. Évenkénti összehasonlítás
# =====================================================================================
def tests_by_season(df, core):
    seasons = sorted(df["season"].unique())
    rows = []
    items = [("yield", "hozam", "hozam")] + [("core20", lab, col) for lab, col in core]
    for s in seasons:
        x = df[df["season"] == s]
        for fam, lab, col in items:
            a = x.loc[x["kezeles"] == 0, col].dropna().to_numpy(float)
            b = x.loc[x["kezeles"] == 1, col].dropna().to_numpy(float)
            r = welch_summary(a, b)
            r.update(season=s, family=fam, variable=lab, source_column=col, mannwhitney_p=mannwhitney_p(a, b))
            rows.append(r)
    out = pd.DataFrame(rows)
    out["holm_p_welch"] = np.nan
    out["bh_q_welch"] = np.nan
    out["holm_p_mannwhitney"] = np.nan
    for (s, fam), idx in out.groupby(["season", "family"]).groups.items():
        out.loc[idx, "holm_p_welch"] = holm_adjust(out.loc[idx, "welch_p"].to_numpy())
        out.loc[idx, "bh_q_welch"] = bh_adjust(out.loc[idx, "welch_p"].to_numpy())
        out.loc[idx, "holm_p_mannwhitney"] = holm_adjust(out.loc[idx, "mannwhitney_p"].to_numpy())
    first = ["season", "family", "variable", "source_column"]
    out = out[first + [c for c in out.columns if c not in first]]
    save_excel_csv(out, os.path.join(OUTPUT_DIR, "manure_by_season_tests.csv"))
    return out


# =====================================================================================
# 8. Évhatással korrigált összehasonlítás
# =====================================================================================
def year_adjusted_tests(df, core):
    rng = np.random.default_rng(PERM_SEED)
    seasons = sorted(df["season"].unique())
    sea = df["season"].map({s: (-0.5 if i == 0 else 0.5) for i, s in enumerate(seasons)}).to_numpy(float)
    kez = df["kezeles"].to_numpy(float)
    items = [("yield", "hozam", "hozam")] + [("core20", lab, col) for lab, col in core]
    Y = np.column_stack([df[col].to_numpy(float) for _, _, col in items])
    ones = np.ones(len(df))
    X_add = np.column_stack([ones, sea, kez])
    X_int = np.column_stack([ones, sea, kez, sea * kez])
    # évenként rétegzett permutációs teszt: a kezelés-címkék az évek EGYES belül keverednek
    idx_by_season = [np.where(df["season"].to_numpy() == s)[0] for s in seasons]

    def adj_stat(k):
        d = []
        for ix in idx_by_season:
            kk = k[ix]
            d.append(Y[ix][kk == 1].mean(axis=0) - Y[ix][kk == 0].mean(axis=0))
        return np.mean(d, axis=0)

    obs = adj_stat(kez)
    count = np.zeros(Y.shape[1])
    for _ in range(N_PERM):
        k = kez.copy()
        for ix in idx_by_season:
            k[ix] = rng.permutation(k[ix])
        count += np.abs(adj_stat(k)) >= np.abs(obs) - 1e-12
    p_perm = (1 + count) / (1 + N_PERM)

    rows = []
    for j, (fam, lab, col) in enumerate(items):
        y = Y[:, j]
        beta, se, p, dof = ols_fit(X_add, y)
        tcrit = stats.t.ppf(0.975, dof)
        _, _, p_int, _ = ols_fit(X_int, y)
        beta_i, _, _, _ = ols_fit(X_int, y)
        rows.append({"family": fam, "variable": lab, "source_column": col,
                     "effect_year_adjusted": beta[2], "se": se[2],
                     "ci_low": beta[2] - tcrit * se[2], "ci_high": beta[2] + tcrit * se[2],
                     "p_ols": p[2], "p_permutation_stratified": p_perm[j],
                     "interaction_season_x_manure": beta_i[3], "p_interaction": p_int[3],
                     "n": len(y), "n_permutations": N_PERM})
    out = pd.DataFrame(rows)
    for col_in, col_out in [("p_ols", "holm_p_ols"), ("p_permutation_stratified", "holm_p_perm"),
                            ("p_interaction", "holm_p_interaction")]:
        out[col_out] = np.nan
        m = out["family"] == "core20"
        out.loc[m, col_out] = holm_adjust(out.loc[m, col_in].to_numpy())
        out.loc[~m, col_out] = out.loc[~m, col_in]
    save_excel_csv(out, os.path.join(OUTPUT_DIR, "manure_year_adjusted_tests.csv"))
    return out


# =====================================================================================
# 9. Évenkénti (within-season) kapcsolat a hozammal
# =====================================================================================
def within_season_correlations(df, core):
    seasons = sorted(df["season"].unique())
    rows = []
    groups = [(str(s), df[df["season"] == s].copy(), "season") for s in seasons]
    groups.append(("within_pooled", df.copy(), "within"))   # évenként centrált, n = 40
    groups.append(("pooled_raw", df.copy(), "raw"))          # nyers pooled (az évek közti különbséggel)
    for gname, sub, mode in groups:
        y_raw = sub["hozam"].to_numpy(float)
        cells = (sub["season"].astype(str) + "_" + sub["kezeles"].astype(int).astype(str)).to_numpy()
        seas = sub["season"].astype(str).to_numpy()
        n = len(sub)
        ncell = len(np.unique(cells))
        for lab, col in core:
            x_raw = sub[col].to_numpy(float)
            if mode == "season":
                x, y = x_raw, y_raw
                dof_r = n - 2
            elif mode == "within":
                x, y = residualize_on_cells(x_raw, seas), residualize_on_cells(y_raw, seas)
                dof_r = n - len(np.unique(seas)) - 1
            else:
                x, y = x_raw, y_raw
                dof_r = n - 2
            r, p = corr_with_dof(x, y, dof_r)
            try:
                rs = float(stats.spearmanr(x, y)[0])
            except Exception:
                rs = np.nan
            # parciális r: a kezelés (és az év) hatásának kiszűrése cellaátlagokkal
            if mode in ("season", "within"):
                xr = residualize_on_cells(x_raw, cells)
                yr = residualize_on_cells(y_raw, cells)
                r_part, p_part = corr_with_dof(xr, yr, n - ncell - 1)
            else:
                r_part, p_part = np.nan, np.nan
            rows.append({"group": gname, "variable": lab, "source_column": col, "n": n,
                         "pearson_r": r, "pearson_p": p, "spearman_rho": rs,
                         "partial_r_given_treatment": r_part, "partial_p": p_part})
    out = pd.DataFrame(rows)
    out["holm_p_pearson"] = np.nan
    out["holm_p_partial"] = np.nan
    for g, idx in out.groupby("group").groups.items():
        out.loc[idx, "holm_p_pearson"] = holm_adjust(out.loc[idx, "pearson_p"].to_numpy())
        out.loc[idx, "holm_p_partial"] = holm_adjust(out.loc[idx, "partial_p"].to_numpy())
    save_excel_csv(out, os.path.join(OUTPUT_DIR, "manure_within_season_correlations.csv"))
    return out


# =====================================================================================
# 10. Blokkon belüli (szomszédos parcellák) páros összehasonlítás
# =====================================================================================
def block_paired_tests(df, core):
    d = df.copy()
    d["block"] = (d["n_parcella"] - 1) // BLOCK_SIZE + 1
    seasons = sorted(d["season"].unique())
    items = [("yield", "hozam", "hozam")] + [("core20", lab, col) for lab, col in core]
    rows = []
    for s in seasons:
        x = d[d["season"] == s]
        for fam, lab, col in items:
            diffs = []
            for b, g in x.groupby("block"):
                a = g.loc[g["kezeles"] == 0, col].dropna()
                c = g.loc[g["kezeles"] == 1, col].dropna()
                if len(a) > 0 and len(c) > 0:
                    diffs.append(c.mean() - a.mean())
            diffs = np.array(diffs, float)
            nb = len(diffs)
            if nb >= 3 and np.std(diffs, ddof=1) > 0:
                t_p = stats.ttest_1samp(diffs, 0.0).pvalue
            else:
                t_p = np.nan
            try:
                w_p = stats.wilcoxon(diffs).pvalue if nb >= 3 else np.nan
            except ValueError:
                w_p = np.nan
            rows.append({"season": s, "family": fam, "variable": lab, "source_column": col,
                         "n_blocks": nb, "mean_block_diff_manured_minus_control": diffs.mean() if nb else np.nan,
                         "sd_block_diff": np.std(diffs, ddof=1) if nb > 1 else np.nan,
                         "n_blocks_positive": int((diffs > 0).sum()) if nb else 0,
                         "paired_t_p": t_p, "wilcoxon_p": w_p,
                         "note": "feltevés: a parcellaszám a térbeli sorrendet követi; 5 blokknál az egzakt Wilcoxon p >= 0,0625"})
    out = pd.DataFrame(rows)
    out["holm_p_paired_t"] = np.nan
    for (s, fam), idx in out.groupby(["season", "family"]).groups.items():
        out.loc[idx, "holm_p_paired_t"] = holm_adjust(out.loc[idx, "paired_t_p"].to_numpy())
    save_excel_csv(out, os.path.join(OUTPUT_DIR, "manure_block_paired_tests.csv"))
    return out


# =====================================================================================
# 11. Új ábrák
# =====================================================================================
def plot_effect_sizes(by_season):
    d = by_season.copy()
    d["label"] = d["variable"].replace({"hozam": "Yield"})
    labels = ["Yield"] + [v for v in d["label"].unique() if v != "Yield"]
    seasons = sorted(d["season"].unique())
    save_excel_csv(d[["season", "variable", "hedges_g", "g_ci_low", "g_ci_high", "welch_p", "holm_p_welch"]],
                   os.path.join(OUTPUT_DIR, "fig_effect_sizes_by_season_data.csv"))
    fig, ax = plt.subplots(figsize=(8, max(6, len(labels) * 0.32)))
    offsets = np.linspace(-0.18, 0.18, len(seasons))
    colors = sns.color_palette("Set2", len(seasons))
    for k, s in enumerate(seasons):
        ds = d[d["season"] == s].set_index("label").reindex(labels)
        ypos = np.arange(len(labels)) + offsets[k]
        sig = (ds["holm_p_welch"] < 0.05).to_numpy()
        ax.errorbar(ds["hedges_g"], ypos, xerr=[ds["hedges_g"] - ds["g_ci_low"], ds["g_ci_high"] - ds["hedges_g"]],
                    fmt="none", ecolor=colors[k], elinewidth=1, capsize=2, alpha=0.8)
        ax.scatter(ds["hedges_g"][~sig], ypos[~sig], facecolors="white", edgecolors=colors[k], s=36, zorder=3, label=f"{s}")
        ax.scatter(ds["hedges_g"][sig], ypos[sig], color=colors[k], s=36, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Hedges' g (manured - control), 95% CI")
    ax.set_title("Treatment effect size by season\n(filled: Holm-adjusted p < 0.05 within the 20-predictor family)", fontsize=11)
    ax.legend(title="Season", loc="best")
    ax.grid(True, axis="x", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_effect_sizes_by_season.png"), dpi=300)
    plt.close()


def plot_within_scatter(df, core):
    lab, col = core[0]   # NDVI_t_mean
    seasons = sorted(df["season"].unique())
    save_excel_csv(df[["season", "n_parcella", "kezeles", "hozam", col]].rename(columns={col: lab}),
                   os.path.join(OUTPUT_DIR, "fig_within_season_scatter_data.csv"))
    fig, axes = plt.subplots(1, len(seasons), figsize=(5.5 * len(seasons), 4.8), sharey=False)
    axes = np.atleast_1d(axes)
    pal = {0: "#e74c3c", 1: "#2ecc71"}
    for ax, s in zip(axes, seasons):
        x = df[df["season"] == s]
        for g, gg in x.groupby("kezeles"):
            ax.scatter(gg[col], gg["hozam"], s=55, edgecolor="k", color=pal[int(g)],
                       label="Manured (1)" if g == 1 else "Control (0)")
        r, p = stats.pearsonr(x[col], x["hozam"])
        if x[col].std() > 0:
            b = np.polyfit(x[col], x["hozam"], 1)
            xs = np.linspace(x[col].min(), x[col].max(), 50)
            ax.plot(xs, np.polyval(b, xs), "k--", lw=1)
        ax.set_title(f"Season {s}: r = {r:.2f} (p = {p:.2f}, n = {len(x)})", fontsize=11)
        ax.set_xlabel(lab)
        ax.set_ylabel("Yield (kg/ha)")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(loc="best", fontsize=9)
    plt.suptitle("Within-season relationship between yield and NDVI_t_mean", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_within_season_scatter.png"), dpi=300)
    plt.close()


def run_extended_analysis(df):
    """6-11. lépés: bővített elemzések."""
    print("\n=== Bővített elemzések (manure03) ===")
    core = resolve_core_columns(df)
    if not core:
        print("Nincs core prediktor oszlop, a bővített elemzések kimaradnak.")
        return
    print("6. Elrendezés-ellenőrzés")
    layout_check(df)
    print("7. Évenkénti összehasonlítás")
    by_season = tests_by_season(df, core)
    print("8. Évhatással korrigált összehasonlítás")
    adj = year_adjusted_tests(df, core)
    print("9. Évenkénti kapcsolat a hozammal")
    corr = within_season_correlations(df, core)
    print("10. Blokkon belüli páros összehasonlítás")
    block_paired_tests(df, core)
    print("11. Új ábrák")
    plot_effect_sizes(by_season)
    plot_within_scatter(df, core)

    # --- tömör összefoglaló a konzolra ---
    print("\n--- ÖSSZEFOGLALÓ ---")
    y = by_season[by_season["variable"] == "hozam"]
    for _, r in y.iterrows():
        print(f"Hozam {int(r['season'])}: kontroll {r['mean_control']:.0f}, trágyázott {r['mean_manured']:.0f} kg/ha; "
              f"különbség {r['diff_manured_minus_control']:+.0f} [{r['diff_ci_low']:.0f}; {r['diff_ci_high']:.0f}], "
              f"Welch p = {r['welch_p']:.3f}, g = {r['hedges_g']:.2f}")
    ya = adj[adj["variable"] == "hozam"].iloc[0]
    print(f"Hozam, évhatással korrigált különbség: {ya['effect_year_adjusted']:+.0f} kg/ha "
          f"(p_OLS = {ya['p_ols']:.3f}, p_perm = {ya['p_permutation_stratified']:.3f}, "
          f"interakció p = {ya['p_interaction']:.3f})")
    for s in sorted(df["season"].unique()):
        c = by_season[(by_season["season"] == s) & (by_season["family"] == "core20")]
        print(f"{int(s)}: core prediktorok, p<0,05 (korrekció nélkül): {int((c['welch_p'] < 0.05).sum())}/{len(c)}; "
              f"Holm-korrekció után: {int((c['holm_p_welch'] < 0.05).sum())}/{len(c)}")
    lab0 = core[0][0]
    for g in [str(s) for s in sorted(df["season"].unique())] + ["within_pooled", "pooled_raw"]:
        c = corr[(corr["group"] == g) & (corr["variable"] == lab0)]
        if len(c):
            c = c.iloc[0]
            print(f"r(hozam, {lab0}) [{g}]: r = {c['pearson_r']:.2f} (p = {c['pearson_p']:.3f}, n = {int(c['n'])})")


# =====================================================================================
# Fő program (a manure02.py 1-5. lépése változatlan)
# =====================================================================================
def main():
    ap = argparse.ArgumentParser(description="manure03: trágyázási kezelés hatásának elemzése")
    ap.add_argument("--input-dir", type=str, default=None, help="a bemeneti fájlok mappája")
    ap.add_argument("--output-dir", type=str, default=None, help="a kimeneti mappa")
    args = ap.parse_args()
    if args.input_dir or args.output_dir:
        _set_paths(args.input_dir, args.output_dir)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(PRED_CSV) or not os.path.exists(KEZELES_XLSX):
        print(f"Error: Input files not found in {INPUT_DIR}!")
        return

    # 1. Adatok beolvasása és összefésülése
    df_pred = pd.read_csv(PRED_CSV, sep=";", decimal=",", encoding="utf-8-sig")
    df_pred = to_numeric_frame(df_pred)
    df_kezeles = pd.read_excel(KEZELES_XLSX)

    df = pd.merge(df_pred, df_kezeles, on=["season", "n_parcella"], how="inner")
    print(f"Merged dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Hozam statisztikák évjárat szerint
    season_yield_stats = []
    for (season, kezeles), group in df.groupby(["season", "kezeles"]):
        season_yield_stats.append({
            "season": season,
            "kezeles": kezeles,
            "n_parcella": len(group),
            "hozam_mean": group["hozam"].mean(),
            "hozam_std": group["hozam"].std(),
            "hozam_median": group["hozam"].median(),
            "hozam_min": group["hozam"].min(),
            "hozam_max": group["hozam"].max(),
        })
    df_season_yield = pd.DataFrame(season_yield_stats)
    save_excel_csv(df_season_yield, OUT_SEASON_YIELD_CSV)

    # Összesített hozam statisztika
    yield_g0 = df[df["kezeles"] == 0]["hozam"].dropna()
    yield_g1 = df[df["kezeles"] == 1]["hozam"].dropna()
    t_yield, p_yield = stats.ttest_ind(yield_g1, yield_g0)

    summary_rows = [
        {"csoport": "Control (0)", "n": len(yield_g0), "hozam_atlag": yield_g0.mean(), "hozam_szoras": yield_g0.std(), "hozam_median": yield_g0.median()},
        {"csoport": "Manured (1)", "n": len(yield_g1), "hozam_atlag": yield_g1.mean(), "hozam_szoras": yield_g1.std(), "hozam_median": yield_g1.median()},
        {"csoport": "Difference (1 - 0)", "n": len(df), "hozam_atlag": yield_g1.mean() - yield_g0.mean(), "hozam_szoras": np.nan, "hozam_median": yield_g1.median() - yield_g0.median(), "t_stat": t_yield, "p_val": p_yield}
    ]
    df_summary = pd.DataFrame(summary_rows)
    save_excel_csv(df_summary, OUT_SUMMARY_CSV)

    # 3. Prediktorok statisztikai tesztelése (Student, Welch, Mann-Whitney)
    # MEGJEGYZÉS (manure03): ez a tábla az éveket ÖSSZEVONVA hasonlítja a kezeléseket, ezért
    # az évhatás (a két év közti nagy különbség) torzíthatja. Az évenkénti és az évhatással
    # korrigált eredményeket lásd a 7-8. lépésben (manure_by_season_tests.csv,
    # manure_year_adjusted_tests.csv). A p-értékek itt korrekció nélküliek.
    numeric_cols = [c for c in df.columns if c not in ["season", "n_parcella", "kezeles"]]
    pred_results = []

    for col in numeric_cols:
        g0 = df[df["kezeles"] == 0][col].dropna()
        g1 = df[df["kezeles"] == 1][col].dropna()

        t_stat, p_val = stats.ttest_ind(g1, g0)
        welch_stat, welch_p = stats.ttest_ind(g1, g0, equal_var=False)
        try:
            mw_stat, mw_p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        except ValueError:
            mw_stat, mw_p = np.nan, np.nan

        pred_results.append({
            "prediktor": col,
            "mean_control_0": g0.mean(),
            "mean_manured_1": g1.mean(),
            "diff_1_minus_0": g1.mean() - g0.mean(),
            "student_t_stat": t_stat,
            "student_p_val": p_val,
            "welch_t_stat": welch_stat,
            "welch_p_val": welch_p,
            "mannwhitney_u_stat": mw_stat,
            "mannwhitney_p_val": mw_p,
            "significant_005": int(p_val < 0.05),
        })

    df_pred_diff = pd.DataFrame(pred_results).sort_values(by="student_p_val")
    save_excel_csv(df_pred_diff, OUT_PRED_DIFF_CSV)
    save_excel_csv(df_pred_diff, OUT_ADVANCED_TESTS_CSV)

    # 4. VIZUALIZÁCIÓK: DOBOZDIAGRAMOK ÉS HEGEDŰDIAGRAMOK MINDEN EGYES VÁLTOZÓRA
    sns.set_theme(style="whitegrid")

    all_vars = numeric_cols

    for col in all_vars:
        if col == "hozam":
            pval_val = p_yield
            mw_pval = stats.mannwhitneyu(yield_g1, yield_g0, alternative="two-sided")[1]
            title_name = "Yield"
            y_label = "Yield (kg/ha)"
            palette_box = {0: "#e74c3c", 1: "#2ecc71"}
            palette_violin = {0: "#e74c3c", 1: "#2ecc71"}
        else:
            match_row = df_pred_diff[df_pred_diff["prediktor"] == col]
            pval_val = match_row["student_p_val"].values[0] if len(match_row) > 0 else 1.0
            mw_pval = match_row["mannwhitney_p_val"].values[0] if len(match_row) > 0 else 1.0
            title_name = col
            y_label = f"{col} Value"
            palette_box = {0: "#3498db", 1: "#9b59b6"}
            palette_violin = {0: "#3498db", 1: "#9b59b6"}

        # --- A. DOBOZDIAGRAM (Box Plot) ---
        fig_box_csv = os.path.join(OUTPUT_DIR, f"fig_{col}_data.csv")
        df_fig_box = df[["season", "n_parcella", "kezeles", col]].copy()
        save_excel_csv(df_fig_box, fig_box_csv)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=df, x="season", y=col, hue="kezeles", palette=palette_box, ax=ax)
        ax.set_title(f"Effect of Manure Treatment on {title_name}\n(Student t p = {pval_val:.4f})", fontsize=12, pad=12)
        ax.set_xlabel("Growing Season", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=["Control (0)", "Manured (1)"], title="Treatment")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fig_{col}_impact.png"), dpi=300)
        plt.close()

        # --- B. HEGEDŰDIAGRAM (Violin Plot) ---
        fig_violin_csv = os.path.join(OUTPUT_DIR, f"fig_{col}_violin_data.csv")
        save_excel_csv(df_fig_box, fig_violin_csv)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.violinplot(data=df, x="season", y=col, hue="kezeles", split=False, palette=palette_violin, ax=ax)
        ax.set_title(f"Violin Distribution: {title_name}\n(Mann-Whitney U p = {mw_pval:.4f})", fontsize=12, pad=12)
        ax.set_xlabel("Growing Season", fontsize=11)
        ax.set_ylabel(f"{title_name} Distribution", fontsize=11)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=["Control (0)", "Manured (1)"], title="Treatment")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"fig_{col}_violin.png"), dpi=300)
        plt.close()

    # --- 5. TOVÁBBI SPECIÁLIS ÁBRÁK (Interaction Plot & Significance Heatmap) ---
    print("Generating interaction plot and significance heatmap...")
    # CSV adatfájl mentése az interakciós ábrához
    fig_inter_csv = os.path.join(OUTPUT_DIR, "fig_interaction_yield_data.csv")
    df_fig_inter = df[["season", "n_parcella", "kezeles", "hozam"]].copy()
    save_excel_csv(df_fig_inter, fig_inter_csv)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.pointplot(data=df, x="season", y="hozam", hue="kezeles", palette={0: "#e74c3c", 1: "#2ecc71"}, markers=["o", "s"], linestyles=["-", "--"], errorbar="se", ax=ax)
    ax.set_title("Two-Way Interaction Plot: Yield (Treatment x Season)", fontsize=12, pad=12)
    ax.set_xlabel("Growing Season", fontsize=11)
    ax.set_ylabel("Mean Yield (kg/ha) ± SE", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=["Control (0)", "Manured (1)"], title="Treatment")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_interaction_yield.png"), dpi=300)
    plt.close()

    heatmap_data = df_pred_diff.set_index("prediktor")[["student_p_val", "welch_p_val", "mannwhitney_p_val"]]
    heatmap_data.columns = ["Student's t", "Welch's t", "Mann-Whitney U"]

    # Ha a hőtérkép indexeiben szerepel a 'hozam', cseréljük 'Yield'-re a megjelenítés kedvéért
    heatmap_data_display = heatmap_data.rename(index={"hozam": "Yield"})

    annot_matrix = heatmap_data_display.map(lambda p: f"{p:.4f} {get_significance_stars(p)}")

    save_excel_csv(heatmap_data.reset_index(), os.path.join(OUTPUT_DIR, "fig_significance_heatmap_data.csv"))

    fig, ax = plt.subplots(figsize=(9, max(6, len(heatmap_data) * 0.35)))
    sns.heatmap(heatmap_data_display, annot=annot_matrix, fmt="", cmap="YlGnBu_r", cbar_kws={'label': 'p-value'}, ax=ax, annot_kws={"size": 9})
    ax.set_title("P-value Comparison Across Statistical Tests\n((*): p<0.05, (**): p<0.01, (***): p<0.001, (ns): not significant)", fontsize=12, pad=12)
    ax.set_xlabel("Statistical Test", fontsize=11)
    ax.set_ylabel("Predictor / Metric", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig_significance_heatmap.png"), dpi=300)
    plt.close()

    print("\nAll boxplots, violin plots, CSV files, and advanced figures generated successfully with Yield labeling!")

    # 6-11. ÚJ: bővített elemzések
    run_extended_analysis(df)

    print("\nmanure03: minden elemzés és ábra elkészült.")


if __name__ == "__main__":
    main()
