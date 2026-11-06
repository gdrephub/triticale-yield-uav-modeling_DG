#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pgroup05.py - szivárgásmentes (nested) keresztvalidáció a triticale hozammodellezéshez
=====================================================================================

Alapelv: MINDEN adatvezérelt lépés (prediktorszelekció, modell-/hiperparaméter-választás,
standardizálás) kizárólag a külső tanítóhalmazon (train_outer) fut. A külső teszthalmaz
(test_outer) csak a legvégén kap predikciót.

Párhuzamosítási architektúra (4 magos optimalizált változat):
- Külső ciklus (outer folds): SZEKVENCIÁLIS (egyszerre 1 fold fut), elkerülve a farkas-hatást (tail effect).
- Belső GA (select_ga): PÁRHUZAMOS (a populáció kiértékelése PG_N_JOBS magon fut).
- Belső GridSearch (modellek): PÁRHUZAMOS (a rácskeresés PG_N_JOBS magon fut).

Külső kör : 5 seed x 5-fold, PARCELLA szerint csoportosított (a parcella mindkét éve együtt marad)
            -> 25 külső fold. A felosztás minden seedre más.
Szelektorok: "enet"  = Elastic Net bootstrap stabilitási szelekció (parcella-szintű bootstrap)
             "ga"    = genetikus algoritmus konszenzusos top-K kiválasztással (csak ha RUN_GA = True)
             "leaky_enet" / "leaky_ga" = a TELJES adaton előre kiválasztott 8 prediktor
                         (CSV kimenetekben szerepel, de az ábrákról szigorúan lekerült)
Modellek  : Elastic Net, SVR, PLSR, Random Forest, Gradient Boosting, KNN, XGBoost
            (belső, parcella-csoportosított GridSearchCV hangolással)
Referencia: Ref_year_mean (a tanító évek átlaga évenként), Ref_global_mean (teljes tanító átlag)
Külön futások: LOYO (év szerinti kihagyás), permutációs nullteszt (RUN_NULL_TEST),
               végső feltáró szelekció a teljes adaton.

Használat:
    python pgroup05.py                    # teljes futás
    python pgroup05.py --quick            # gyors próbafutás (kis beállításokkal)
    python pgroup05.py --replot RUN_MAPPA # ábrák újragenerálása a mentett ábra-CSV-kből
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import platform
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

try:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.feature_selection import RFE
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import ElasticNet, ElasticNetCV, enet_path
    from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Hiányzó csomag: {exc}. Szükséges: scikit-learn, xgboost, pandas, numpy, matplotlib, joblib.") from exc

try:
    import shap
except ModuleNotFoundError:  # a SHAP opcionális
    shap = None


def env_bool(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "igen", "on")


def env_int(name, default):
    v = os.environ.get(name)
    return default if v is None or v == "" else int(v)


# =====================================================================================
# 1. KAPCSOLÓK ÉS BEÁLLÍTÁSOK
# =====================================================================================
RUN_GA = env_bool("PG_RUN_GA", True)                   # LOGIKAI KAPCSOLÓ: futtassa-e a genetikus algoritmust?
RUN_LEAKY_COMPARISON = env_bool("PG_RUN_LEAKY", False) # szivárgásos viszonyítás (CSV-kben benne marad, ábrákról lekerül)
RUN_LOYO = env_bool("PG_RUN_LOYO", True)               # év szerinti kihagyás (leave-one-year-out)
RUN_NULL_TEST = env_bool("PG_RUN_NULL", False)         # permutációs nullteszt (időigényes)
RUN_IMPORTANCE = env_bool("PG_RUN_IMPORTANCE", True)   # RF fontosságok (permutáció, Gini, SHAP, RFE) külső foldonként
RUN_REFERENCE_MODELS = env_bool("PG_RUN_REF", True)    # évátlag / globális átlag referencia
RUN_FINAL_EXPLORATORY = env_bool("PG_RUN_FINAL", False)# végső feltáró szelekció a teljes adaton

PG_N_JOBS = env_int("PG_N_JOBS", 4)                    # belső párhuzamos magok száma - általában kívülről adom meg
CV_SEEDS = [2026, 2027, 2028, 2029, 2030]              # 5 ismétlés
N_OUTER_SPLITS = 5                                     # 5-fold -> 25 külső fold
RANDOM_STATE = 2026
N_BOOTSTRAP_CI = env_int("PG_N_BOOT_CI", 2000)         # parcella-bootstrap a konfidenciaintervallumhoz
N_NULL_PERMUTATIONS = env_int("PG_N_NULL", 20)
NULL_SELECTOR = "enet"

TARGET_COLUMN = "hozam"
PARCEL_COLUMN = "n_parcella"
SEASON_COLUMN = "season"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = Path(os.environ.get("PG_INPUT", str(SCRIPT_DIR / "prediktor5_190_245_hozam.csv")))
OUTPUT_BASE = Path(os.environ.get("PG_OUTPUT_BASE", str(SCRIPT_DIR / "pgroup01_out")))

CFG = {
    "n_select": 8,
    "random_state": RANDOM_STATE,
    "inner_folds": 3,                       # belső GroupKFold a hangoláshoz
    "enet_l1_ratios": [0.3, 0.5, 0.7, 0.9],
    "enet_alpha_n": 60,
    "enet_alpha_log_span": 3.0,             # alpha_max -> alpha_max*10^-3
    "n_boot_nested": 300,                   # bootstrap-szám a külső foldokon belüli szelekcióhoz
    "n_boot_final": 2000,                   # bootstrap-szám a végső feltáró szelekcióhoz
    "perm_repeats": 10,
    "ga": {
        "population": 96, "max_generations": 50, "patience": 10, "elite": 8, "tournament": 3,
        "immigrants": 10, "base_mutation": 0.15, "high_mutation": 0.45, "swap_min": 1, "swap_max": 2,
        "low_div_unique_fraction": 0.60, "high_similarity": 0.55, "high_pair_fraction": 0.25,
        "inner_folds": 4, "inner_repeats": 2, "rf_trees": 100, "pls_max": 6,
        "std_penalty": 0.5,                 # célfüggvény = legjobb modell RMSE-átlaga + std_penalty * RMSE-szórása
        "rel_tol": 1e-4, "top_fraction": 0.15, "keep_top": 50,
        "enet_l1": [0.1, 0.5, 0.9, 1.0], "enet_alphas": np.logspace(-3, 0, 12).tolist(),
    },
}

MODEL_NAMES = ["Elastic Net", "SVR", "PLSR", "Random Forest", "Gradient Boosting", "KNN", "XGBoost"]
REF_NAMES = ["Ref_year_mean", "Ref_global_mean"]
IMPORTANCE_METHODS = ["perm_mse_increase", "gini", "shap_mean_abs", "rfe_rank"]


def apply_quick_mode():
    global CV_SEEDS, N_BOOTSTRAP_CI, N_NULL_PERMUTATIONS
    CV_SEEDS = CV_SEEDS[:1]
    N_BOOTSTRAP_CI = 100
    N_NULL_PERMUTATIONS = 2
    CFG["n_boot_nested"] = 30
    CFG["n_boot_final"] = 100
    CFG["perm_repeats"] = 3
    CFG["ga"].update(population=12, max_generations=3, patience=2, elite=2, immigrants=2,
                     inner_repeats=1, rf_trees=30, keep_top=10)


# =====================================================================================
# 2. KIMENETI SEGÉDEK (Excel-barát CSV, nyilvántartás)
# =====================================================================================
class Out:
    """Kimeneti mappa + fájlnyilvántartás (manifest)."""

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.fig_dir = run_dir / "figures"
        self.figdata_dir = run_dir / "figure_data"
        self.log_dir = run_dir / "task_logs"
        for d in (self.dir, self.fig_dir, self.figdata_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.manifest = []

    def table(self, df: pd.DataFrame, name: str, desc: str, sub: Path | None = None) -> Path:
        path = (sub or self.dir) / f"{name}.csv"
        save_csv(df, path)
        self.manifest.append({"file": str(path.relative_to(self.dir)), "type": "table", "rows": len(df), "description": desc})
        return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, sep=";", decimal=",", encoding="utf-8-sig")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("pgroup05")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 3600:d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


# =====================================================================================
# 3. ADAT
# =====================================================================================
def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Nem található a bemeneti fájl: {path}")
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    df.columns = df.columns.str.replace("\ufeff", "").str.strip()
    for col in (SEASON_COLUMN, PARCEL_COLUMN, TARGET_COLUMN):
        if col not in df.columns:
            raise ValueError(f"Hiányzó kötelező oszlop: {col}")
    predictors = [c for c in df.columns if c not in (SEASON_COLUMN, PARCEL_COLUMN, TARGET_COLUMN)]
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df[predictors] = df[predictors].apply(pd.to_numeric, errors="coerce")
    valid = df[TARGET_COLUMN].notna() & df[predictors].notna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df, predictors


# =====================================================================================
# 4. SZELEKTOROK (csak a megadott tanítóadatot látják, GA konszenzusos felülvizsgálattal)
# =====================================================================================
def select_enet(X: pd.DataFrame, y, groups, seed: int, cfg: dict, n_boot: int):
    """Elastic Net bootstrap stabilitási szelekció, parcella-szintű (cluster) bootstrappel."""
    names = list(X.columns)
    Xv = X.to_numpy(float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    n_sel = cfg["n_select"]
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows_by_g = {g: np.where(groups == g)[0] for g in uniq}
    counts = np.zeros(len(names))
    abs_coef = np.zeros(len(names))
    n_exact = 0
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_g[g] for g in chosen])
        Xb, yb = Xv[idx], y[idx]
        sd = Xb.std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (Xb - Xb.mean(axis=0)) / sd
        ysd = yb.std() or 1.0
        ys = (yb - yb.mean()) / ysd
        best = None
        for l1 in cfg["enet_l1_ratios"]:
            amax = np.max(np.abs(Xs.T @ ys)) / (len(ys) * l1)
            if amax <= 0:
                continue
            alphas = amax * np.logspace(0, -cfg["enet_alpha_log_span"], cfg["enet_alpha_n"])
            _, coefs, _ = enet_path(Xs, ys, l1_ratio=l1, alphas=alphas, max_iter=20000)
            nz = (coefs != 0).sum(axis=0)
            diff = np.abs(nz - n_sel)
            j = int(np.argmin(diff))
            if best is None or diff[j] < best[0]:
                best = (diff[j], coefs[:, j].copy())
            if diff[j] == 0:
                break
        if best is None:
            continue
        n_exact += int(best[0] == 0)
        counts += (best[1] != 0)
        abs_coef += np.abs(best[1])
    prob = counts / max(1, n_boot)
    mean_abs = abs_coef / max(1, n_boot)
    order = sorted(range(len(names)), key=lambda i: (-counts[i], -mean_abs[i]))
    feats = [names[i] for i in order[:n_sel]]
    prob_rows = [{"predictor": names[i], "selection_probability": prob[i], "mean_abs_std_coef": mean_abs[i],
                  "n_boot": n_boot, "fraction_boot_exact_n_select": n_exact / max(1, n_boot),
                  "selected": int(names[i] in feats)} for i in order]
    return feats, {"enet_prob": prob_rows}


def select_ga(X: pd.DataFrame, y, groups, seed: int, cfg: dict, log):
    """Genetikus algoritmus konszenzusos top-K frekvencia alapú 8-as prediktorszelekcióval."""
    g = cfg["ga"]
    names = list(X.columns)
    P = len(names)
    size = cfg["n_select"]
    rng = np.random.default_rng(seed)
    Xv = X.to_numpy(float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    splits = []
    for r in range(g["inner_repeats"]):
        kf = KFold(n_splits=min(g["inner_folds"], len(uniq)), shuffle=True, random_state=seed * 1000 + r)
        for a, b in kf.split(uniq):
            splits.append((np.where(np.isin(groups, uniq[a]))[0], np.where(np.isin(groups, uniq[b]))[0]))
    cache: dict = {}
    t_ga = time.perf_counter()

    def evaluate(key):
        cols = list(key)
        rm = {"Random Forest": [], "Elastic Net": [], "PLSR": []}
        for tr, te in splits:
            sc = StandardScaler().fit(Xv[tr][:, cols])
            xtr, xte = sc.transform(Xv[tr][:, cols]), sc.transform(Xv[te][:, cols])
            ytr, yte = y[tr], y[te]
            p = RandomForestRegressor(n_estimators=g["rf_trees"], max_features="sqrt", min_samples_leaf=2,
                                      random_state=seed, n_jobs=1).fit(xtr, ytr).predict(xte)
            rm["Random Forest"].append(math.sqrt(np.mean((p - yte) ** 2)))
            mu, sd = ytr.mean(), (ytr.std() or 1.0)
            gtr = groups[tr]
            cvs = list(GroupKFold(n_splits=min(3, len(np.unique(gtr)))).split(xtr, ytr, groups=gtr))
            en = ElasticNetCV(l1_ratio=g["enet_l1"], alphas=g["enet_alphas"], cv=cvs, max_iter=20000,
                              random_state=seed).fit(xtr, (ytr - mu) / sd)
            p = en.predict(xte) * sd + mu
            rm["Elastic Net"].append(math.sqrt(np.mean((p - yte) ** 2)))
            nc = max(1, min(g["pls_max"], xtr.shape[1], xtr.shape[0] - 2))
            p = PLSRegression(n_components=nc, scale=False).fit(xtr, ytr).predict(xte).ravel()
            rm["PLSR"].append(math.sqrt(np.mean((p - yte) ** 2)))
        means = {k: float(np.mean(v)) for k, v in rm.items()}
        best_model = min(means, key=means.get)
        b_std = float(np.std(rm[best_model]))
        return {"obj": means[best_model] + g["std_penalty"] * b_std, "best_model": best_model,
                "best_rmse_mean": means[best_model], "best_rmse_std": b_std,
                "rf_rmse": means["Random Forest"], "enet_rmse": means["Elastic Net"], "pls_rmse": means["PLSR"]}

    def rand_ind():
        return tuple(sorted(rng.choice(P, size=size, replace=False).tolist()))

    def complete(sel):
        sel = list(dict.fromkeys(sel))[:size]
        avail = [i for i in range(P) if i not in sel]
        while len(sel) < size:
            c = int(rng.choice(avail))
            sel.append(c)
            avail.remove(c)
        return tuple(sorted(sel))

    def jaccard(a, b):
        return len(set(a) & set(b)) / len(set(a) | set(b))

    def diversity(pop):
        uniq_p = {i for ind in pop for i in ind}
        sims = [jaccard(a, b) for a, b in itertools.combinations(pop, 2)]
        pc: dict = {}
        for ind in pop:
            for pr in itertools.combinations(ind, 2):
                pc[pr] = pc.get(pr, 0) + 1
        return {"unique_predictor_fraction": len(uniq_p) / P, "mean_jaccard": float(np.mean(sims)) if sims else 1.0,
                "max_pair_frequency": max(pc.values()) if pc else 0}

    def tournament(ranked):
        k = min(g["tournament"], len(ranked))
        cand = rng.choice(len(ranked), size=k, replace=False)
        return ranked[int(min(cand))]

    pop = []
    seen = set()
    att = 0
    while len(pop) < g["population"] and att < g["population"] * 200:
        att += 1
        c = rand_ind()
        if c not in seen:
            pop.append(c)
            seen.add(c)
    best_obj, no_imp, gen_rows, stopped = math.inf, 0, [], "max_generations"
    for gen in range(g["max_generations"] + 1):
        t_gen = time.perf_counter()
        new = [k for k in pop if k not in cache]
        if new:
            eval_results = Parallel(n_jobs=PG_N_JOBS, backend="loky")(
                delayed(evaluate)(k) for k in new
            )
            for k, res in zip(new, eval_results):
                cache[k] = res

        ranked = sorted(pop, key=lambda k: cache[k]["obj"])
        objs = np.array([cache[k]["obj"] for k in ranked])
        top_n = max(1, math.ceil(len(ranked) * g["top_fraction"]))
        div = diversity(pop)
        high = (div["unique_predictor_fraction"] < g["low_div_unique_fraction"] or div["mean_jaccard"] > g["high_similarity"]
                or div["max_pair_frequency"] > len(pop) * g["high_pair_fraction"])
        mut = g["high_mutation"] if high else g["base_mutation"]
        gen_rows.append({"generation": gen, "best_objective": float(objs[0]), "mean_objective": float(objs.mean()),
                         "top15_mean_objective": float(objs[:top_n].mean()),
                         "best_model": cache[ranked[0]]["best_model"], "best_rmse_mean": cache[ranked[0]]["best_rmse_mean"],
                         "n_new_evaluations": len(new), "mutation_rate": mut, **div,
                         "generation_seconds": time.perf_counter() - t_gen})
        log(f"    GA gen {gen:02d}/{g['max_generations']}: legjobb célfv={objs[0]:.1f} ({cache[ranked[0]]['best_model']}), "
            f"új egyed={len(new)}, mutáció={mut:.2f}")
        if objs[0] < best_obj * (1 - g["rel_tol"]):
            best_obj, no_imp = float(objs[0]), 0
        else:
            no_imp += 1
        if gen >= 1 and no_imp >= g["patience"]:
            stopped = "patience"
            break
        if gen == g["max_generations"]:
            break
        nxt = list(ranked[:g["elite"]])
        seen = set(nxt)
        target = max(g["elite"], g["population"] - g["immigrants"])
        att = 0
        while len(nxt) < target and att < g["population"] * 200:
            att += 1
            p1, p2 = tournament(ranked), tournament(ranked)
            half = size // 2
            child = list(rng.choice(p1, size=half, replace=False)) + list(rng.choice(p2, size=size - half, replace=False))
            child = complete([int(c) for c in child])
            if rng.random() <= mut:
                sel = list(child)
                for _ in range(int(rng.integers(g["swap_min"], g["swap_max"] + 1))):
                    sel.pop(int(rng.integers(0, len(sel))))
                    sel.append(int(rng.choice([i for i in range(P) if i not in sel])))
                child = tuple(sorted(sel))
            if child not in seen:
                nxt.append(child)
                seen.add(child)
        att = 0
        while len(nxt) < g["population"] and att < g["population"] * 200:
            att += 1
            c = rand_ind()
            if c not in seen:
                nxt.append(c)
                seen.add(c)
        pop = nxt

    ranked_all = sorted(cache, key=lambda k: cache[k]["obj"])
    
    # --- KONSZENZUSOS KIVÁLASZTÁS A KEEP_TOP ALAPJÁN ---
    k_top = min(g.get("keep_top", 10), len(ranked_all))
    top_individuals = ranked_all[:k_top]
    
    # Összegyűjtjük az összes prediktor indexet a top_individuals-ből
    all_pred_indices = [idx for ind in top_individuals for idx in ind]
    
    # Megszámoljuk a frekvenciákat és másodlagos rendezéshez az 1. helyezettbeli pozíciót
    from collections import Counter
    freq_counts = Counter(all_pred_indices)
    rank1_ind = ranked_all[0]
    
    # Rendezési kulcs: (-frekvencia, ha benne van az 1. helyezettben 0 különben 1, index értéke)
    sorted_pred_indices = sorted(
        freq_counts.keys(),
        key=lambda idx: (-freq_counts[idx], 0 if idx in rank1_ind else 1, idx)
    )
    
    consensus_best_indices = sorted_pred_indices[:size]
    feats = [names[i] for i in consensus_best_indices]
    
    top_rows = [{"rank": r + 1, "predictors": ";".join(names[i] for i in k), "objective": cache[k]["obj"],
                 "best_model": cache[k]["best_model"], "best_rmse_mean": cache[k]["best_rmse_mean"],
                 "best_rmse_std": cache[k]["best_rmse_std"], "rf_rmse": cache[k]["rf_rmse"],
                 "enet_rmse": cache[k]["enet_rmse"], "pls_rmse": cache[k]["pls_rmse"]}
                for r, k in enumerate(ranked_all[:k_top])]
    
    best = ranked_all[0]
    info = {"generations_run": len(gen_rows) - 1, "stopped_by": stopped, "n_evaluations": len(cache),
            "best_objective": cache[best]["obj"], "best_model": cache[best]["best_model"],
            "best_rmse_mean": cache[best]["best_rmse_mean"], "best_rmse_std": cache[best]["best_rmse_std"],
            "seconds": time.perf_counter() - t_ga}
    return feats, {"ga_info": info, "ga_gen": gen_rows, "ga_top": top_rows}


# =====================================================================================
# 5. MODELLEK
# =====================================================================================
def build_model(name: str, seed: int):
    """Pipeline (StandardScaler + modell) és hiperparaméter-rács."""
    if name == "Elastic Net":
        est = TransformedTargetRegressor(regressor=ElasticNet(max_iter=50000, random_state=seed), transformer=StandardScaler())
        grid = {"model__regressor__alpha": [0.001, 0.01, 0.1, 0.5, 1.0],
                "model__regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
    elif name == "SVR":
        est = TransformedTargetRegressor(regressor=SVR(kernel="rbf"), transformer=StandardScaler())
        grid = {"model__regressor__C": [0.1, 1, 10, 100], "model__regressor__gamma": ["scale", "auto", 0.01, 0.1]}
    elif name == "PLSR":
        est = PLSRegression(scale=False)
        grid = {"model__n_components": [1, 2, 3, 4, 5]}
    elif name == "Random Forest":
        est = RandomForestRegressor(random_state=seed, n_jobs=1)
        grid = {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 5, 10], "model__min_samples_split": [2, 5]}
    elif name == "Gradient Boosting":
        est = GradientBoostingRegressor(random_state=seed)
        grid = {"model__n_estimators": [50, 100, 150], "model__learning_rate": [0.01, 0.05, 0.1], "model__max_depth": [3, 5]}
    elif name == "KNN":
        est = KNeighborsRegressor(n_jobs=1)
        grid = {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"]}
    elif name == "XGBoost":
        est = XGBRegressor(random_state=seed, n_jobs=1, verbosity=0)
        grid = {"model__n_estimators": [50, 100, 150], "model__learning_rate": [0.01, 0.05, 0.1], "model__max_depth": [3, 5]}
    else:
        raise ValueError(name)
    return Pipeline([("scaler", StandardScaler()), ("model", est)]), grid


def clean_params(params: dict) -> dict:
    return {k.replace("model__regressor__", "").replace("model__", ""): v for k, v in params.items()}


def rf_importances(pipe, Xtr, ytr, Xte, yte, cfg, seed):
    """RF fontosságok külső foldonként: permutációs, Gini, SHAP, RFE."""
    feats = list(Xtr.columns)
    rf, sc = pipe.named_steps["model"], pipe.named_steps["scaler"]
    out = [("gini", f, float(v)) for f, v in zip(feats, rf.feature_importances_)]
    pi = permutation_importance(pipe, Xte, yte, scoring="neg_mean_squared_error", n_repeats=cfg["perm_repeats"], random_state=seed)
    out += [("perm_mse_increase", f, float(v)) for f, v in zip(feats, pi.importances_mean)]
    if shap is not None:
        try:
            sv = shap.TreeExplainer(rf).shap_values(sc.transform(Xte))
            out += [("shap_mean_abs", f, float(v)) for f, v in zip(feats, np.abs(sv).mean(axis=0))]
        except Exception:  # noqa: BLE001
            pass
    try:
        rfe = RFE(RandomForestRegressor(**rf.get_params()), n_features_to_select=1, step=1).fit(sc.transform(Xtr), ytr)
        out += [("rfe_rank", f, float(v)) for f, v in zip(feats, rfe.ranking_)]
    except Exception:  # noqa: BLE001
        pass
    return out


# =====================================================================================
# 6. EGY KÜLSŐ FOLD / TASK FUTTATÁSA
# =====================================================================================
class TaskLog:
    def __init__(self, path: Path, echo: bool):
        self.path, self.echo = path, echo

    def __call__(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if self.echo:
            print(line, flush=True)


def run_task(task: dict) -> dict:
    warnings.filterwarnings("ignore")
    t0 = time.perf_counter()
    cfg = task["cfg"]
    log = TaskLog(Path(task["log_dir"]) / f"{task['task_id']}.log", task["echo"])
    X, y, groups, seasons, parcels = task["X"], np.asarray(task["y"], float), task["groups"], task["seasons"], task["parcels"]
    tr, te = task["train_idx"], task["test_idx"]
    Xtr_all, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
    yte = y[te]
    meta = {"task_id": task["task_id"], "kind": task["kind"], "outer_id": task["outer_id"], "seed": task["seed"],
            "fold": task["fold"], "perm_id": task["perm_id"]}
    res = {k: [] for k in ("preds", "params", "selection", "enet_prob", "ga_info", "ga_gen", "ga_top", "importance", "timing")}
    res["meta"] = meta
    log(f"TASK {task['task_id']} indul: train={len(tr)} sor ({len(np.unique(gtr))} parcella), test={len(te)} sor")

    def add_preds(selector, model, pred):
        for j, p in zip(te, pred):
            res["preds"].append({**meta, "selector": selector, "model": model, "row_id": int(task["row_ids"][j]),
                                 "n_parcella": parcels[j], "season": seasons[j], "observed": float(y[j]), "predicted": float(p)})

    for sel in task["selectors"]:
        ts = time.perf_counter()
        if sel == "enet":
            feats, extra = select_enet(Xtr_all, ytr, gtr, task["sel_seed"], cfg, cfg["n_boot_nested"])
        elif sel == "ga":
            feats, extra = select_ga(Xtr_all, ytr, gtr, task["sel_seed"], cfg, log)
        elif sel.startswith("leaky_"):
            feats, extra = list(task["fixed_sets"][sel]), {}
        else:
            raise ValueError(sel)
        sec_sel = time.perf_counter() - ts
        log(f"  [{sel}] kiválasztott prediktorok ({sec_sel:.1f} s): {feats}")
        for r, f in enumerate(feats, 1):
            res["selection"].append({**meta, "selector": sel, "rank": r, "predictor": f})
        for row in extra.get("enet_prob", []):
            res["enet_prob"].append({**meta, "selector": sel, **row})
        if "ga_info" in extra:
            res["ga_info"].append({**meta, "selector": sel, **extra["ga_info"]})
            res["ga_gen"] += [{**meta, "selector": sel, **r} for r in extra["ga_gen"]]
            res["ga_top"] += [{**meta, "selector": sel, **r} for r in extra["ga_top"]]
        Xtr_s, Xte_s = Xtr_all[feats], X.iloc[te][feats]
        n_splits = min(cfg["inner_folds"], len(np.unique(gtr)))
        tm = time.perf_counter()
        for mname in MODEL_NAMES:
            t1 = time.perf_counter()
            pipe, grid = build_model(mname, cfg["random_state"])
            gs = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=n_splits), scoring="neg_mean_squared_error", n_jobs=PG_N_JOBS, refit=True)
            gs.fit(Xtr_s, ytr, groups=gtr)
            pred = np.ravel(gs.predict(Xte_s))
            add_preds(sel, mname, pred)
            res["params"].append({**meta, "selector": sel, "model": mname, "best_params": json.dumps(clean_params(gs.best_params_), default=str),
                                  "inner_cv_rmse": math.sqrt(max(0.0, -gs.best_score_)), "n_train": len(tr), "n_test": len(te),
                                  "fit_seconds": time.perf_counter() - t1})
            if task["run_importance"] and mname == "Random Forest":
                for method, f, v in rf_importances(gs.best_estimator_, Xtr_s, ytr, Xte_s, yte, cfg, task["sel_seed"]):
                    res["importance"].append({**meta, "selector": sel, "method": method, "predictor": f, "value": v})
            log(f"    {mname:18s} belső CV RMSE={math.sqrt(max(0.0, -gs.best_score_)):8.1f} | teszt RMSE={math.sqrt(np.mean((pred - yte) ** 2)):8.1f} | {time.perf_counter() - t1:5.1f} s")
        res["timing"].append({**meta, "selector": sel, "selection_seconds": sec_sel, "models_seconds": time.perf_counter() - tm})
    if task["run_ref"]:
        tr_seasons = seasons[tr]
        ymeans = {s: float(ytr[tr_seasons == s].mean()) for s in np.unique(tr_seasons)}
        gmean = float(ytr.mean())
        add_preds("none", "Ref_year_mean", [ymeans.get(seasons[j], gmean) for j in te])
        add_preds("none", "Ref_global_mean", [gmean] * len(te))
    res["total_seconds"] = time.perf_counter() - t0
    log(f"TASK {task['task_id']} kész ({res['total_seconds']:.1f} s)")
    return res


def run_tasks(tasks: list, n_jobs: int, label: str, logger: logging.Logger) -> list:
    n = len(tasks)
    if n == 0:
        return []
    results, t0 = [], time.perf_counter()

    def report(i, r):
        el = time.perf_counter() - t0
        eta = el / i * (n - i)
        pr = pd.DataFrame(r["preds"])
        line = f"[{label}] {i}/{n} kész | task={r['meta']['task_id']} | eltelt {fmt_time(el)} | hátralévő ~{fmt_time(eta)}"
        logger.info(line)
        if not pr.empty:
            for sel, gsel in pr[pr.model.isin(MODEL_NAMES)].groupby("selector"):
                rm = gsel.groupby("model").apply(lambda d: math.sqrt(np.mean((d.predicted - d.observed) ** 2)), include_groups=False)
                feats = [s["predictor"] for s in r["selection"] if s["selector"] == sel]
                logger.info(f"      {sel}: legjobb fold-RMSE = {rm.min():.0f} ({rm.idxmin()}); prediktorok: {', '.join(feats)}")

    for i, t in enumerate(tasks, 1):
        r = run_task(t)
        results.append(r)
        report(i, r)

    return results


def collect(results: list, key: str) -> pd.DataFrame:
    rows = [row for r in results for row in r[key]]
    return pd.DataFrame(rows)


# =====================================================================================
# 7. METRIKÁK, BOOTSTRAP
# =====================================================================================
def calc_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    sse, sst = float((err ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    rmse = math.sqrt(sse / len(y))
    return {"R2": 1 - sse / sst if sst > 0 else np.nan, "RMSE": rmse, "NRMSE_pct": rmse / y.mean() * 100,
            "MAE": float(np.abs(err).mean()), "Bias": float(err.mean()), "n": len(y)}


METRIC_KEYS = ("R2", "RMSE", "NRMSE_pct", "MAE", "Bias")


def per_seed_metrics(pred: pd.DataFrame, seasons) -> pd.DataFrame:
    rows = []
    for (sel, mod, seed), g in pred.groupby(["selector", "model", "seed"], sort=True):
        m = calc_metrics(g.observed, g.predicted)
        row = {"selector": sel, "model": mod, "seed": seed, **{k: m[k] for k in METRIC_KEYS}, "n": m["n"]}
        for s in seasons:
            gs = g[g.season == s]
            if len(gs) > 1:
                ms = calc_metrics(gs.observed, gs.predicted)
                for k in METRIC_KEYS:
                    row[f"{k}_{s}"] = ms[k]
        rows.append(row)
    return pd.DataFrame(rows)


def per_fold_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sel, mod, oid), g in pred.groupby(["selector", "model", "outer_id"], sort=True):
        m = calc_metrics(g.observed, g.predicted)
        rows.append({"selector": sel, "model": mod, "outer_id": oid, "seed": g.seed.iloc[0], "fold": g.fold.iloc[0],
                     **{k: m[k] for k in METRIC_KEYS}, "n": m["n"]})
    return pd.DataFrame(rows)


def summarize(df_seed: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df_seed.columns if c not in ("selector", "model", "seed", "n")]
    g = df_seed.groupby(["selector", "model"])[cols]
    out = pd.concat([g.mean().add_suffix("_mean"), g.std(ddof=1).add_suffix("_sd")], axis=1).reset_index()
    out["rank_RMSE_in_selector"] = out.groupby("selector")["RMSE_mean"].rank(method="min")
    return out


def bootstrap_ci(pred: pd.DataFrame, data: pd.DataFrame, seasons, B: int, seed: int):
    keys = sorted(pred[["selector", "model"]].drop_duplicates().itertuples(index=False, name=None))
    seeds = sorted(pred.seed.unique())
    n = len(data)
    P = np.full((len(keys), len(seeds), n), np.nan)
    for ki, (sel, mod) in enumerate(keys):
        sub = pred[(pred.selector == sel) & (pred.model == mod)]
        for si, sd in enumerate(seeds):
            ss = sub[sub.seed == sd]
            P[ki, si, ss.row_id.values] = ss.predicted.values
    y = data[TARGET_COLUMN].to_numpy(float)
    par = data[PARCEL_COLUMN].to_numpy()
    sea = data[SEASON_COLUMN].to_numpy()
    uniq = np.unique(par)
    rows_by_p = {p: np.where(par == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)

    def block(idx):
        Pb, yb, sb = P[:, :, idx], y[idx], sea[idx]
        out = {}
        for tag, mask in [("", np.ones(len(idx), bool))] + [(f"_{s}", sb == s) for s in seasons]:
            if mask.sum() < 2:
                continue
            e2 = (Pb[:, :, mask] - yb[mask]) ** 2
            sse = e2.sum(axis=2)
            sst = ((yb[mask] - yb[mask].mean()) ** 2).sum()
            rmse = np.sqrt(sse / mask.sum())
            out["RMSE" + tag] = rmse.mean(axis=1)
            out["R2" + tag] = (1 - sse / sst).mean(axis=1) if sst > 0 else np.full(len(keys), np.nan)
            out["NRMSE_pct" + tag] = out["RMSE" + tag] / yb[mask].mean() * 100
        return out

    point = block(np.arange(n))
    boots = {k: np.zeros((B, len(keys))) for k in point}
    for b in range(B):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        bl = block(np.concatenate([rows_by_p[p] for p in chosen]))
        for k in boots:
            boots[k][b] = bl.get(k, np.nan)
    ci_rows = []
    for ki, (sel, mod) in enumerate(keys):
        row = {"selector": sel, "model": mod, "n_boot": B}
        for k in point:
            lo, hi = np.nanpercentile(boots[k][:, ki], [2.5, 97.5])
            row[f"{k}_est"], row[f"{k}_ci_low"], row[f"{k}_ci_high"] = point[k][ki], lo, hi
        ci_rows.append(row)
    pair_rows = []
    for sel in sorted({k[0] for k in keys}):
        ids = [i for i, k in enumerate(keys) if k[0] == sel]
        for a, b in itertools.combinations(ids, 2):
            d = boots["RMSE"][:, a] - boots["RMSE"][:, b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            pair_rows.append({"selector": sel, "model_A": keys[a][1], "model_B": keys[b][1],
                              "RMSE_diff_A_minus_B": point["RMSE"][a] - point["RMSE"][b], "ci_low": lo, "ci_high": hi,
                              "share_A_better": float((d < 0).mean()), "ci_excludes_zero": int(lo > 0 or hi < 0)})
    return pd.DataFrame(ci_rows), pd.DataFrame(pair_rows)


# =====================================================================================
# 8. ÁBRÁK
# =====================================================================================
def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _model_order(models):
    return [m for m in MODEL_NAMES + REF_NAMES if m in set(models)]


def plot_fold_rmse(df, path):
    d_clean = df[~df.selector.str.startswith("leaky_")] if "selector" in df.columns else df
    sels = list(dict.fromkeys(d_clean.selector)) if "selector" in d_clean.columns else ["enet"]
    for sel in sels:
        d = d_clean[d_clean.selector == sel] if "selector" in d_clean.columns else d_clean
        ms = _model_order(d.model)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot([d[d.model == m].RMSE.values for m in ms])
        ax.set_xticks(range(1, len(ms) + 1))
        ax.set_xticklabels(ms, rotation=20, ha="right")
        ax.set_ylabel("RMSE of outer test fold (kg ha$^{-1}$)")
        ax.set_title(f"Outer fold RMSE distribution ({sel})")
        ax.grid(True, ls=":", alpha=0.5)
        p = path if len(sels) == 1 else path.parent / f"{path.stem}_{sel}{path.suffix}"
        _finish(fig, p)


def plot_obs_pred(df, path):
    d_clean = df[~df.selector.str.startswith("leaky_")] if "selector" in df.columns else df
    combos = list(dict.fromkeys(zip(d_clean.selector, d_clean.model)))
    for sel, mod in combos:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        d = d_clean[(d_clean.selector == sel) & (d_clean.model == mod)]
        for s, gs in d.groupby("season"):
            ax.scatter(gs.observed, gs.predicted_mean_over_seeds, s=45, edgecolor="k", alpha=0.8, label=str(s))
        lo, hi = min(d.observed.min(), d.predicted_mean_over_seeds.min()) * 0.95, max(d.observed.max(), d.predicted_mean_over_seeds.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_xlabel("Observed yield (kg ha$^{-1}$)")
        ax.set_ylabel("Predicted yield, mean of repeats (kg ha$^{-1}$)")
        ax.set_title(f"Observed vs Predicted: {mod} ({sel})")
        ax.text(0.04, 0.96, f"R$^2$ = {d.R2_mean_over_seeds.iloc[0]:.2f}\nRMSE = {d.RMSE_mean_over_seeds.iloc[0]:.0f}", transform=ax.transAxes,
                va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.85))
        ax.legend(title="Season", loc="lower right")
        ax.grid(True, ls=":", alpha=0.5)
        p = path.parent / f"{path.stem}_{sel}_{mod.lower().replace(' ', '_')}{path.suffix}"
        _finish(fig, p)


def plot_season_metrics(df, path):
    d_clean = df[~df.selector.str.startswith("leaky_")] if "selector" in df.columns else df
    sels = list(dict.fromkeys(d_clean.selector)) if "selector" in d_clean.columns else ["enet"]
    for sel in sels:
        d = d_clean[d_clean.selector == sel] if "selector" in d_clean.columns else d_clean
        ms, seas = _model_order(d.model), list(dict.fromkeys(d.season))
        w = 0.8 / max(1, len(seas))
        for col, lab in [("R2_mean", "Within-season R$^2$"), ("NRMSE_pct_mean", "NRMSE (%)")]:
            fig, ax = plt.subplots(figsize=(9, 5))
            for k, s in enumerate(seas):
                dd = d[d.season == s].set_index("model").reindex(ms)
                ax.bar(np.arange(len(ms)) + k * w, dd[col].values, w, yerr=dd[col.replace("_mean", "_sd")].values, capsize=2, label=str(s))
            ax.set_xticks(np.arange(len(ms)) + w * (len(seas) - 1) / 2)
            ax.set_xticklabels(ms, rotation=25, ha="right")
            ax.set_ylabel(lab)
            ax.set_title(f"{lab} by season ({sel})")
            ax.axhline(0, color="k", lw=0.6)
            ax.legend(title="Season")
            ax.grid(True, axis="y", ls=":", alpha=0.5)
            metric_tag = "r2" if "R2" in col else "nrmse"
            p = path.parent / f"{path.stem}_{sel}_{metric_tag}{path.suffix}"
            _finish(fig, p)


def plot_selection_freq(df, path):
    d_clean = df[~df.selector.str.startswith("leaky_")] if "selector" in df.columns else df
    sels = list(dict.fromkeys(d_clean.selector)) if "selector" in d_clean.columns else ["enet"]
    for sel in sels:
        fig, ax = plt.subplots(figsize=(8, 7))
        d = d_clean[d_clean.selector == sel].sort_values("frequency") if "selector" in d_clean.columns else d_clean.sort_values("frequency")
        ax.barh(d.predictor, d.frequency, color="#2b7a78")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Selection frequency over outer folds")
        ax.set_title(f"Predictor selection stability ({sel})")
        ax.grid(True, axis="x", ls=":", alpha=0.5)
        p = path.parent / f"{path.stem}_{sel}{path.suffix}" if len(sels) > 1 else path
        _finish(fig, p)


def plot_importance(df, path):
    d_clean = df[~df.selector.str.startswith("leaky_")] if "selector" in df.columns else df
    methods = [m for m in IMPORTANCE_METHODS if m in set(d_clean.method)]
    sels = list(dict.fromkeys(d_clean.selector)) if "selector" in d_clean.columns else ["enet"]
    for meth in methods:
        fig, ax = plt.subplots(figsize=(8, 7))
        d = d_clean[d_clean.method == meth]
        asc = meth == "rfe_rank"
        order = d.groupby("predictor")["mean"].mean().sort_values(ascending=asc).index.tolist()[:12][::-1]
        h = 0.8 / len(sels)
        for k, sel in enumerate(sels):
            dd = d[d.selector == sel].set_index("predictor").reindex(order) if "selector" in d.columns else d.set_index("predictor").reindex(order)
            ax.barh(np.arange(len(order)) + k * h, dd["mean"].values, h, xerr=dd["sd"].values, label=sel, capsize=2)
        ax.set_yticks(np.arange(len(order)) + h * (len(sels) - 1) / 2)
        ax.set_yticklabels(order)
        title_map = {"perm_mse_increase": "Permutation Importance (MSE increase)", "gini": "Gini Impurity Importance",
                     "shap_mean_abs": "SHAP Mean Absolute Value", "rfe_rank": "RFE Rank (1 = best)"}
        ax.set_title(title_map.get(meth, meth))
        if len(sels) > 1:
            ax.legend()
        ax.grid(True, axis="x", ls=":", alpha=0.5)
        p = path.parent / f"{path.stem}_{meth}{path.suffix}"
        _finish(fig, p)


def plot_loyo(df, path):
    d_clean = df[~df.selector.str.startswith("leaky_")] if "selector" in df.columns else df
    sels = list(dict.fromkeys(d_clean.selector)) if "selector" in d_clean.columns else ["enet"]
    for sel in sels:
        fig, ax = plt.subplots(figsize=(9, 5))
        d = d_clean[d_clean.selector == sel] if "selector" in d_clean.columns else d_clean
        ms, seas = _model_order(d.model), list(dict.fromkeys(d.held_out_season))
        w = 0.8 / max(1, len(seas))
        for k, s in enumerate(seas):
            dd = d[d.held_out_season == s].set_index("model").reindex(ms)
            ax.bar(np.arange(len(ms)) + k * w, dd.RMSE.values, w, label=f"held-out {s}")
        ax.set_xticks(np.arange(len(ms)) + w * (len(seas) - 1) / 2)
        ax.set_xticklabels(ms, rotation=25, ha="right")
        ax.set_ylabel("RMSE (kg ha$^{-1}$)")
        ax.set_title(f"Leave-one-year-out cross-validation RMSE ({sel})")
        ax.legend()
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        p = path.parent / f"{path.stem}_{sel}{path.suffix}" if len(sels) > 1 else path
        _finish(fig, p)


def null_metrics_row(gp: pd.DataFrame, seasons) -> dict:
    m = calc_metrics(gp.observed, gp.predicted)
    row = {"R2_pooled": m["R2"], "RMSE": m["RMSE"], "NRMSE_pct": m["NRMSE_pct"]}
    r2s, rmses = [], []
    for s in seasons:
        gs = gp[gp.season == s]
        if len(gs) > 1:
            ms = calc_metrics(gs.observed, gs.predicted)
            row[f"R2_{s}"], row[f"RMSE_{s}"], row[f"NRMSE_pct_{s}"] = ms["R2"], ms["RMSE"], ms["NRMSE_pct"]
            r2s.append(ms["R2"])
            rmses.append(ms["RMSE"])
    row["R2_within_season_mean"] = float(np.nanmean(r2s)) if r2s else np.nan
    row["RMSE_within_season_mean"] = float(np.mean(rmses)) if rmses else np.nan
    return row


def null_summary(nd: pd.DataFrame, seasons) -> pd.DataFrame:
    metrics = [("R2_pooled", "higher"), ("R2_within_season_mean", "higher"), ("RMSE_within_season_mean", "lower")]
    for s in seasons:
        metrics += [(f"R2_{s}", "higher"), (f"RMSE_{s}", "lower")]
    rows = []
    for mod, gp in nd.groupby("model"):
        ob = gp[gp.is_observed == 1]
        nl = gp[gp.is_observed == 0]
        for col, direction in metrics:
            if col not in gp.columns or ob.empty or nl[col].dropna().empty:
                continue
            o, v = float(ob[col].iloc[0]), nl[col].dropna()
            hits = (v >= o).sum() if direction == "higher" else (v <= o).sum()
            rows.append({"model": mod, "metric": col, "better_is": direction, "observed": o, "null_mean": v.mean(),
                         "null_sd": v.std(ddof=1), "null_p05": np.percentile(v, 5), "null_p95": np.percentile(v, 95),
                         "n_perm": len(v), "empirical_p": (1 + hits) / (1 + len(v))})
    return pd.DataFrame(rows)


def plot_null(df, path):
    ms = _model_order(df.model)
    for m in ms:
        fig, ax = plt.subplots(figsize=(7, 5))
        d = df[df.model == m]
        series = [("R2_pooled", "pooled R$^2$", "#999999", "k"), ("R2_within_season_mean", "within-season R$^2$ (mean)", "#6fa8dc", "b")]
        for col, lab, hc, lc in series:
            if col not in d.columns:
                continue
            ax.hist(d[d.is_observed == 0][col].dropna(), bins=12, color=hc, alpha=0.6, label=f"null {lab}")
            ob = d[d.is_observed == 1][col].dropna()
            if len(ob):
                ax.axvline(ob.iloc[0], color=lc, lw=2, ls="--", label=f"observed {lab}")
        ax.set_title(f"Permutation test null distribution: {m}")
        ax.set_xlabel("R$^2$ (yields shuffled within season)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(True, ls=":", alpha=0.5)
        p = path.parent / f"{path.stem}_{m.lower().replace(' ', '_')}{path.suffix}"
        _finish(fig, p)


def plot_ga(df, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for tid, d in df.groupby("task_id"):
        ax.plot(d.generation, d.best_objective, lw=1, alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best objective (inner-CV RMSE + penalty, kg ha$^{-1}$)")
    ax.set_title("GA convergence across outer folds")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(True, ls=":", alpha=0.5)
    _finish(fig, path)


def plot_selector_comparison(df, path):
    d_clean = df[~df.selector.str.startswith("leaky_")] if "selector" in df.columns else df
    ms = _model_order(d_clean.model)
    sels = list(dict.fromkeys(d_clean.selector)) if "selector" in d_clean.columns else ["enet"]
    w = 0.8 / len(sels)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k, sel in enumerate(sels):
        d = d_clean[d_clean.selector == sel].set_index("model").reindex(ms) if "selector" in d.columns else d_clean.set_index("model").reindex(ms)
        ax.bar(np.arange(len(ms)) + k * w, d.RMSE_mean.values, w, yerr=d.RMSE_sd.values, capsize=2, label=sel)
    ax.set_xticks(np.arange(len(ms)) + w * (len(sels) - 1) / 2)
    ax.set_xticklabels(ms, rotation=25, ha="right")
    ax.set_ylabel("Pooled RMSE, mean of repeats (kg ha$^{-1}$)")
    ax.set_title("Nested feature selection vs. reference models")
    ax.legend()
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    _finish(fig, path)


FIG_SPECS = {
    "fig01_fold_rmse_boxplot": (plot_fold_rmse, "RMSE of outer test folds per model and selector (standalone)"),
    "fig02_observed_vs_predicted": (plot_obs_pred, "Observed vs predicted yield per model and selector (standalone)"),
    "fig03_within_season_metrics": (plot_season_metrics, "Within-season R2 and NRMSE per model (standalone)"),
    "fig04_selection_frequency": (plot_selection_freq, "Predictor selection frequency over outer folds (standalone)"),
    "fig05_rf_importance_oof": (plot_importance, "Random Forest importances aggregated over outer folds (standalone)"),
    "fig06_loyo_rmse": (plot_loyo, "Leave-one-year-out RMSE (standalone)"),
    "fig07_null_test": (plot_null, "Permutation null distribution vs observed pooled R2 per model (standalone)"),
    "fig08_ga_convergence": (plot_ga, "GA convergence per outer fold"),
    "fig09_selector_comparison": (plot_selector_comparison, "Pooled RMSE comparison (nested selection vs references)"),
}


def emit_figure(out: Out, name: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    csv_path = out.figdata_dir / f"{name}.csv"
    save_csv(df, csv_path)
    fn, desc = FIG_SPECS[name]
    try:
        fn(df, out.fig_dir / f"{name}.png")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Az ábra ({name}) rajzolása nem sikerült: {exc}", flush=True)
    out.manifest.append({"file": str(csv_path.relative_to(out.dir)), "type": "figure_data", "rows": len(df), "description": desc})
    out.manifest.append({"file": f"figures/{name}.png", "type": "figure", "rows": "", "description": desc})


def replot(run_dir: Path):
    figdata = run_dir / "figure_data"
    (run_dir / "figures").mkdir(exist_ok=True)
    for name, (fn, _) in FIG_SPECS.items():
        p = figdata / f"{name}.csv"
        if p.exists():
            print(f"Újrarajzolás: {name}", flush=True)
            fn(read_csv(p), run_dir / "figures" / f"{name}.png")
    print("Kész.", flush=True)


# =====================================================================================
# 9. FŐPROGRAM
# =====================================================================================
def make_task(kind, task_id, tdata, train_idx, test_idx, selectors, **kw):
    base = {"kind": kind, "task_id": task_id, "cfg": CFG, "X": tdata["X"], "y": tdata["y"], "groups": tdata["groups"],
            "seasons": tdata["seasons"], "parcels": tdata["groups"], "row_ids": tdata["row_ids"], "train_idx": train_idx,
            "test_idx": test_idx, "selectors": selectors, "log_dir": str(tdata["log_dir"]), "echo": True,
            "outer_id": -1, "seed": -1, "fold": -1, "perm_id": -1, "sel_seed": RANDOM_STATE, "fixed_sets": {},
            "run_ref": False, "run_importance": False}
    base.update(kw)
    return base


def main():
    ap = argparse.ArgumentParser(description="pgroup05: szivárgásmentes nested CV GA konszenzussal")
    ap.add_argument("--quick", action="store_true", help="gyors próbafutás kis beállításokkal")
    ap.add_argument("--replot", type=str, default=None, help="ábrák újragenerálása egy korábbi futás mappájából")
    ap.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    ap.add_argument("--out", type=str, default=str(OUTPUT_BASE))
    args = ap.parse_args()
    if args.replot:
        replot(Path(args.replot))
        return
    if args.quick:
        apply_quick_mode()

    t_start = time.perf_counter()
    run_dir = Path(args.out) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    out = Out(run_dir)
    logger = setup_logger(run_dir / "pgroup05.log")
    logger.info("=== pgroup05 indul: szivárgásmentes nested CV (GA konszenzusos szelekcióval) ===")

    config_dump = {"RUN_GA": RUN_GA, "RUN_LEAKY_COMPARISON": RUN_LEAKY_COMPARISON, "RUN_LOYO": RUN_LOYO, "RUN_NULL_TEST": RUN_NULL_TEST,
                   "RUN_IMPORTANCE": RUN_IMPORTANCE, "RUN_REFERENCE_MODELS": RUN_REFERENCE_MODELS, "RUN_FINAL_EXPLORATORY": RUN_FINAL_EXPLORATORY,
                   "PG_N_JOBS": PG_N_JOBS, "CV_SEEDS": CV_SEEDS, "N_OUTER_SPLITS": N_OUTER_SPLITS, "N_BOOTSTRAP_CI": N_BOOTSTRAP_CI,
                   "N_NULL_PERMUTATIONS": N_NULL_PERMUTATIONS, "quick_mode": bool(args.quick), "input": args.input, "CFG": CFG}
    for k, v in config_dump.items():
        logger.info("  %s = %r", k, v)
    import sklearn, xgboost, joblib
    soft = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "joblib": joblib.__version__,
            "shap": getattr(shap, "__version__", "nincs telepítve"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"), "SLURM_CPUS_ON_NODE": os.environ.get("SLURM_CPUS_ON_NODE")}
    for k, v in soft.items():
        logger.info("  SOFTWARE/%s = %s", k, v)
    (run_dir / "pgroup05_run_config.json").write_text(json.dumps({"config": config_dump, "software": soft}, indent=2, default=str), encoding="utf-8")

    # ---------------- adat ----------------
    data, predictors = load_data(Path(args.input))
    seasons = sorted(data[SEASON_COLUMN].unique())
    parcel_ids = np.sort(data[PARCEL_COLUMN].unique())
    logger.info("Adat: %d sor, %d prediktor, %d parcella, évjáratok: %s", len(data), len(predictors), len(parcel_ids), seasons)
    if len(predictors) != 20:
        logger.warning("A prediktorok száma %d (20-at vártunk)", len(predictors))
    tdata = {"X": data[predictors].copy(), "y": data[TARGET_COLUMN].to_numpy(float), "groups": data[PARCEL_COLUMN].to_numpy(),
             "seasons": data[SEASON_COLUMN].to_numpy(), "row_ids": data["row_id"].to_numpy(), "log_dir": out.log_dir}
    seasons_arr, groups_arr = tdata["seasons"], tdata["groups"]

    ds = data.groupby(SEASON_COLUMN)[TARGET_COLUMN].agg(n="count", mean="mean", sd="std", min="min", max="max").reset_index()
    out.table(ds, "pgroup01_data_summary", "Yield descriptive statistics per season")
    cor_rows = []
    for p in predictors:
        row = {"predictor": p, "r_pooled": data[p].corr(data[TARGET_COLUMN])}
        for s in seasons:
            d = data[data[SEASON_COLUMN] == s]
            row[f"r_season_{s}"] = d[p].corr(d[TARGET_COLUMN])
        cor_rows.append(row)
    out.table(pd.DataFrame(cor_rows), "pgroup01_predictor_yield_correlations", "Pearson r of each predictor with yield")
    out.table(data[predictors].corr().reset_index().rename(columns={"index": "predictor"}), "pgroup01_predictor_correlation_matrix",
              "Pearson correlation matrix of candidate predictors")
    logger.info("Hozam évenként:\n%s", ds.to_string(index=False))

    # ---------------- végső feltáró szelekció (a teljes adaton) ----------------
    main_selectors = ["enet"] + (["ga"] if RUN_GA else [])
    fixed_sets, final_rows = {}, []
    if RUN_FINAL_EXPLORATORY or RUN_LEAKY_COMPARISON:
        logger.info("Végső FELTÁRÓ szelekció a teljes adaton...")
        log_stdout = TaskLog(out.log_dir / "final_exploratory.log", True)
        feats, extra = select_enet(tdata["X"], tdata["y"], groups_arr, RANDOM_STATE, CFG, CFG["n_boot_final"])
        fixed_sets["leaky_enet"] = feats
        for r, row in enumerate(extra["enet_prob"], 1):
            final_rows.append({"selector": "enet", "rank": r, **row})
        logger.info("  enet (teljes adat): %s", feats)
        if RUN_GA:
            feats_ga, extra_ga = select_ga(tdata["X"], tdata["y"], groups_arr, RANDOM_STATE, CFG, log_stdout)
            fixed_sets["leaky_ga"] = feats_ga
            for r, f in enumerate(feats_ga, 1):
                final_rows.append({"selector": "ga", "rank": r, "predictor": f, "selected": 1, **extra_ga["ga_info"]})
            out.table(pd.DataFrame(extra_ga["ga_gen"]), "pgroup01_final_ga_generations", "GA generation statistics on full data")
            out.table(pd.DataFrame(extra_ga["ga_top"]), "pgroup01_final_ga_top_individuals", "Top GA individuals on full data")
            logger.info("  ga (teljes adat): %s", feats_ga)
        out.table(pd.DataFrame(final_rows), "pgroup01_final_exploratory_selection",
                  "Predictors selected on FULL data (exploratory, CSV reference only)")

    selectors_outer = list(main_selectors)
    if RUN_LEAKY_COMPARISON:
        selectors_outer += [f"leaky_{s}" for s in main_selectors if f"leaky_{s}" in fixed_sets]

    # ---------------- külső felosztások ----------------
    outer_tasks, split_rows, oid = [], [], 0
    for seed in CV_SEEDS:
        kf = KFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=seed)
        for fold, (tp, ep) in enumerate(kf.split(parcel_ids), 1):
            oid += 1
            tr_idx = np.where(np.isin(groups_arr, parcel_ids[tp]))[0]
            te_idx = np.where(np.isin(groups_arr, parcel_ids[ep]))[0]
            split_rows.append({"outer_id": oid, "seed": seed, "fold": fold, "n_train_rows": len(tr_idx), "n_test_rows": len(te_idx),
                               "train_parcels": ";".join(map(str, parcel_ids[tp])), "test_parcels": ";".join(map(str, parcel_ids[ep]))})
            outer_tasks.append(make_task("outer", f"outer_{oid:02d}", tdata, tr_idx, te_idx, selectors_outer, outer_id=oid, seed=seed, fold=fold,
                                         sel_seed=RANDOM_STATE + oid, fixed_sets=fixed_sets, run_ref=RUN_REFERENCE_MODELS,
                                         run_importance=RUN_IMPORTANCE))
    out.table(pd.DataFrame(split_rows), "pgroup01_outer_splits", "Outer CV splits (parcel-grouped)")
    logger.info("Külső CV: %d seed x %d-fold = %d külső fold | szelektorok: %s | modellek: %s | belső magok: %d",
                len(CV_SEEDS), N_OUTER_SPLITS, len(outer_tasks), selectors_outer, MODEL_NAMES, PG_N_JOBS)

    # ---------------- külső kör ----------------
    res_outer = run_tasks(outer_tasks, 1, "KÜLSŐ CV", logger)
    pred = collect(res_outer, "preds")
    params = collect(res_outer, "params")
    selection = collect(res_outer, "selection")
    timing = collect(res_outer, "timing")
    out.table(pred, "pgroup01_predictions_outer", "All out-of-fold predictions per selector/model/outer fold")
    out.table(params, "pgroup01_fold_model_params", "Tuned hyperparameters and inner-CV RMSE per outer fold")
    out.table(selection, "pgroup01_selection_per_fold", "Predictors selected in each outer fold")
    out.table(timing, "pgroup01_timing_per_fold", "Selection and model-fitting time per outer fold")
    enet_prob = collect(res_outer, "enet_prob")
    if not enet_prob.empty:
        out.table(enet_prob, "pgroup01_enet_probabilities_per_fold", "Elastic Net stability-selection probabilities")
    ga_info, ga_gen, ga_top = collect(res_outer, "ga_info"), collect(res_outer, "ga_gen"), collect(res_outer, "ga_top")
    if not ga_info.empty:
        out.table(ga_info, "pgroup01_ga_info_per_fold", "GA run summary per outer fold")
        out.table(ga_gen, "pgroup01_ga_generations_per_fold", "GA generation statistics per outer fold")
        out.table(ga_top, "pgroup01_ga_top_individuals_per_fold", "Top GA individuals per outer fold")

    n_folds = len(outer_tasks)
    sel_freq = (selection.groupby(["selector", "predictor"]).size().rename("n_selected").reset_index())
    full_idx = pd.MultiIndex.from_product([sorted(selection.selector.unique()), predictors], names=["selector", "predictor"])
    sel_freq = sel_freq.set_index(["selector", "predictor"]).reindex(full_idx, fill_value=0).reset_index()
    sel_freq["n_outer_folds"] = n_folds
    sel_freq["frequency"] = sel_freq.n_selected / n_folds
    out.table(sel_freq.sort_values(["selector", "frequency"], ascending=[True, False]), "pgroup01_selection_frequency",
              "Predictor selection frequency over outer folds")
    emit_figure(out, "fig04_selection_frequency", sel_freq[["selector", "predictor", "n_selected", "frequency"]])

    # metrikák
    logger.info("Metrikák számítása...")
    seed_m = per_seed_metrics(pred, seasons)
    fold_m = per_fold_metrics(pred)
    summ = summarize(seed_m)
    out.table(seed_m, "pgroup01_metrics_per_seed", "Pooled and per-season metrics for each repeat")
    out.table(fold_m, "pgroup01_metrics_per_fold", "Metrics on each outer test fold")
    out.table(summ, "pgroup01_metrics_summary", "Mean and SD of metrics over repeats")
    ci, pairs = bootstrap_ci(pred, data, seasons, N_BOOTSTRAP_CI, RANDOM_STATE)
    out.table(ci, "pgroup01_metrics_bootstrap_ci", "Parcel-bootstrap 95% CIs for metrics")
    out.table(pairs, "pgroup01_pairwise_rmse_differences", "Paired parcel-bootstrap RMSE differences between models")

    emit_figure(out, "fig01_fold_rmse_boxplot", fold_m[["selector", "model", "outer_id", "seed", "fold", "RMSE", "R2", "NRMSE_pct"]])
    real = summ[summ.model.isin(MODEL_NAMES)]
    best_rows = real.loc[real.groupby("selector")["RMSE_mean"].idxmin()]
    op_rows = []
    for _, br in best_rows.iterrows():
        d = pred[(pred.selector == br.selector) & (pred.model == br.model)]
        a = d.groupby(["row_id", "n_parcella", "season"]).agg(observed=("observed", "first"), predicted_mean_over_seeds=("predicted", "mean")).reset_index()
        a["selector"], a["model"], a["R2_mean_over_seeds"], a["RMSE_mean_over_seeds"] = br.selector, br.model, br.R2_mean, br.RMSE_mean
        op_rows.append(a)
    emit_figure(out, "fig02_observed_vs_predicted", pd.concat(op_rows, ignore_index=True))
    sm_rows = []
    for _, r in summ.iterrows():
        for s in seasons:
            if f"R2_{s}_mean" in summ.columns:
                sm_rows.append({"selector": r.selector, "model": r.model, "season": s, "R2_mean": r[f"R2_{s}_mean"], "R2_sd": r[f"R2_{s}_sd"],
                                "NRMSE_pct_mean": r[f"NRMSE_pct_{s}_mean"], "NRMSE_pct_sd": r[f"NRMSE_pct_{s}_sd"],
                                "RMSE_mean": r[f"RMSE_{s}_mean"], "RMSE_sd": r[f"RMSE_{s}_sd"]})
    emit_figure(out, "fig03_within_season_metrics", pd.DataFrame(sm_rows))
    emit_figure(out, "fig09_selector_comparison", summ[["selector", "model", "RMSE_mean", "RMSE_sd", "R2_mean", "R2_sd", "NRMSE_pct_mean"]])
    if not ga_gen.empty:
        emit_figure(out, "fig08_ga_convergence", ga_gen[ga_gen.kind == "outer"][["task_id", "outer_id", "generation", "best_objective", "top15_mean_objective", "mean_objective"]])

    # fontosságok
    imp = collect(res_outer, "importance")
    if not imp.empty:
        out.table(imp, "pgroup01_importance_oof_raw", "RF importances per outer fold")
        g = imp.groupby(["selector", "method", "predictor"])["value"]
        imp_s = pd.concat([g.mean().rename("mean"), g.std(ddof=1).rename("sd"), g.size().rename("n_folds_selected")], axis=1).reset_index()
        out.table(imp_s, "pgroup01_importance_oof_summary", "RF importances aggregated over outer folds")
        emit_figure(out, "fig05_rf_importance_oof", imp_s)

    # ---------------- LOYO ----------------
    pred_loyo = pd.DataFrame()
    if RUN_LOYO:
        loyo_tasks = []
        for k, s in enumerate(seasons):
            tr_idx, te_idx = np.where(seasons_arr != s)[0], np.where(seasons_arr == s)[0]
            loyo_tasks.append(make_task("loyo", f"loyo_{s}", tdata, tr_idx, te_idx, main_selectors, sel_seed=RANDOM_STATE + 5000 + k,
                                        run_ref=RUN_REFERENCE_MODELS, fold=k + 1))
        res_loyo = run_tasks(loyo_tasks, 1, "LOYO", logger)
        pred_loyo = collect(res_loyo, "preds")
        pred_loyo["held_out_season"] = pred_loyo["season"]
        out.table(pred_loyo, "pgroup01_predictions_loyo", "Leave-one-year-out predictions")
        out.table(collect(res_loyo, "selection"), "pgroup01_selection_loyo", "Predictors selected in LOYO runs")
        out.table(collect(res_loyo, "params"), "pgroup01_params_loyo", "Tuned hyperparameters in LOYO runs")
        lo_rows = []
        for (sel, mod, hs), gp in pred_loyo.groupby(["selector", "model", "held_out_season"]):
            m = calc_metrics(gp.observed, gp.predicted)
            lo_rows.append({"selector": sel, "model": mod, "held_out_season": hs, **{k: m[k] for k in METRIC_KEYS}, "n": m["n"]})
        lo = pd.DataFrame(lo_rows)
        pooled = []
        for (sel, mod), gp in pred_loyo.groupby(["selector", "model"]):
            m = calc_metrics(gp.observed, gp.predicted)
            pooled.append({"selector": sel, "model": mod, "held_out_season": "pooled", **{k: m[k] for k in METRIC_KEYS}, "n": m["n"]})
        out.table(pd.concat([lo, pd.DataFrame(pooled)], ignore_index=True), "pgroup01_metrics_loyo", "LOYO metrics per held-out season")
        emit_figure(out, "fig06_loyo_rmse", lo)

    # ---------------- permutációs nullteszt ----------------
    if RUN_NULL_TEST:
        rng = np.random.default_rng(RANDOM_STATE + 99)
        seed0 = CV_SEEDS[0]
        base_splits = [t for t in outer_tasks if t["seed"] == seed0]
        null_tasks = []
        for pidx in range(N_NULL_PERMUTATIONS):
            yp = tdata["y"].copy()
            for s in seasons:
                ii = np.where(seasons_arr == s)[0]
                yp[ii] = yp[rng.permutation(ii)]
            for bt in base_splits:
                t = dict(bt)
                t.update(task_id=f"null{pidx + 1:03d}_f{bt['fold']}", kind="null", perm_id=pidx + 1, y=yp, selectors=[NULL_SELECTOR],
                         run_ref=False, run_importance=False, fixed_sets={}, sel_seed=RANDOM_STATE + 9000 + pidx * 10 + bt["fold"])
                null_tasks.append(t)
        logger.info("Permutációs nullteszt: %d permutáció x %d fold", N_NULL_PERMUTATIONS, len(base_splits))
        res_null = run_tasks(null_tasks, 1, "NULLTESZT", logger)
        pn = collect(res_null, "preds")
        out.table(pn, "pgroup01_predictions_null", "Predictions of permutation null runs")
        rows = []
        for (pid, mod), gp in pn.groupby(["perm_id", "model"]):
            rows.append({"perm_id": pid, "model": mod, "is_observed": 0, **null_metrics_row(gp, seasons)})
        ob = pred[(pred.selector == NULL_SELECTOR) & (pred.seed == seed0) & pred.model.isin(MODEL_NAMES)]
        for mod, gp in ob.groupby("model"):
            rows.append({"perm_id": 0, "model": mod, "is_observed": 1, **null_metrics_row(gp, seasons)})
        nd = pd.DataFrame(rows)
        out.table(nd, "pgroup01_null_test_metrics", "Null test metrics for permutations")
        out.table(null_summary(nd, seasons), "pgroup01_null_test_summary", "Permutation null test summary and empirical p-values")
        emit_figure(out, "fig07_null_test", nd)

    # ---------------- lezárás ----------------
    try:
        with pd.ExcelWriter(run_dir / "pgroup01_key_tables.xlsx", engine="openpyxl") as xw:
            for nm, df_ in [("summary", summ), ("bootstrap_ci", ci), ("pairwise", pairs), ("sel_freq", sel_freq)]:
                df_.to_excel(xw, sheet_name=nm, index=False)
        out.manifest.append({"file": "pgroup01_key_tables.xlsx", "type": "table", "rows": "", "description": "Key tables in Excel workbook"})
    except Exception as exc:  # noqa: BER001
        logger.info("Excel-munkafüzet kihagyva: %s", exc)
    total = time.perf_counter() - t_start
    out.manifest.append({"file": "pgroup05.log", "type": "log", "rows": "", "description": "Main log"})
    out.table(pd.DataFrame(out.manifest), "pgroup01_output_manifest", "List of all output files")
    top = summ[summ.model.isin(MODEL_NAMES)].sort_values(["selector", "RMSE_mean"]).groupby("selector").head(3)
    logger.info("Legjobb modellek szelektoronként:\n%s", top[["selector", "model", "R2_mean", "RMSE_mean", "NRMSE_pct_mean"]].to_string(index=False))
    logger.info("=== KÉSZ. Teljes futási idő: %s | kimenet: %s ===", fmt_time(total), run_dir)


if __name__ == "__main__":
    main()
