#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
indexmodel04.py - egyindexes modellek szivárgásmentes nested CV értékeléssel
===========================================================================================

Az indexmodel01.py továbbfejlesztése. A prediktorkészletek előre rögzítettek; nincs teljes adaton végzett szelekció vagy validációs kimenetbe keverés.
  * Az értékelés ugyanaz, mint a pgroup05.py-ban: 5 seed x 5-fold, PARCELLA szerint csoportosított külső CV
    (a seedek, a KFold és a parcellasorrend azonos -> a külső felosztások BETŰRE ugyanazok, a predikciók
    fold-szinten párosíthatók), 25 külső fold. A tesztparcellák nem vesznek részt az illesztésben vagy a hiperparaméter-hangolásban.
  * Belső hangolás: GridSearchCV + GroupKFold(3) (parcella-csoportok), negatív MSE; a modellek hiperparaméter-
    rácsa megegyezik a pgroup05.py-éval (a PLSR komponensszáma a prediktorszámhoz igazítva).
  * Modellek: MLR, Ridge, Elastic Net, SVR, PLSR, Random Forest, Gradient Boosting, KNN, XGBoost
    (a pgroup05.py 7 modellje + az indexmodel01.py MLR és Ridge modellje).
  * Nincs adatvezérelt prediktorszelekció: minden indexhez előre rögzítetten a saját 4 időbeli leírója tartozik
    (<INDEX>_t_mean, _t_median, _t_std, _t_auc). A pgroup05.py "selector" oszlopának itt a prediktorkészlet
    (index neve) felel meg: selector = "NDVI", "GNDVI", ... (opcionálisan "ALL" = mind a 20, szelekció nélkül).
  * Referencia: Ref_year_mean, Ref_global_mean (selector = "none"), mint a pgroup05.py-ban.
  * Metrikák: R2, RMSE, NRMSE_pct, MAE, Bias; pooled és évenkénti; ismétlésenként, átlag +- SD; parcella-
    bootstrap 95%-os CI; páros RMSE-különbségek (modellek között, indexek között, referenciához képest).
  * Fontosságok külső foldonként: permutációs (minden modellre), standardizált együttható (lineáris
    modellekre), Gini + SHAP + RFE (Random Forest), SHAP (Gradient Boosting, XGBoost).
  * Kiegészítő LOYO (év szerinti kihagyás) validáció.
  * Kimenet: indexmodout/run_<időbélyeg>/ (Excel-barát CSV, ábrák + ábra-adatok, key_tables.xlsx, manifest,
    napló, futási konfiguráció), a pgroup05.py kimeneti rendszerének mintájára (a fájlnevek előtagja: im04_).

Használat:
    python indexmodel04.py                          # teljes futás
    python indexmodel04.py --quick                  # gyors próbafutás
    python indexmodel04.py --replot <indexmodout/run_...>    # ábrák újragenerálása az ábra-adatokból

Környezeti változók (mind opcionális): IM_INPUT, IM_OUTPUT_BASE, IM_N_JOBS, IM_RUN_LOYO, IM_RUN_IMPORTANCE, IM_RUN_REF, IM_ALL_PREDICTORS, IM_INDICES (pl. "NDVI,NDRE"), IM_N_BOOT_CI.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import platform
import re
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

try:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.feature_selection import RFE
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from xgboost import XGBRegressor
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"Hiányzó csomag: {exc}. Szükséges: scikit-learn, xgboost, pandas, numpy, matplotlib.") from exc

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
RUN_LOYO = env_bool("IM_RUN_LOYO", True)
RUN_IMPORTANCE = env_bool("IM_RUN_IMPORTANCE", True)
RUN_REFERENCE_MODELS = env_bool("IM_RUN_REF", True)
INCLUDE_ALL_PREDICTORS = env_bool("IM_ALL_PREDICTORS", False)   # "ALL": mind a 20 prediktor, szelekció nélkül
INDEX_FILTER = [s.strip() for s in os.environ.get("IM_INDICES", "").split(",") if s.strip()]

N_JOBS = env_int("IM_N_JOBS", env_int("PG_N_JOBS", 1))
CV_SEEDS = [2026, 2027, 2028, 2029, 2030]      # azonos a pgroup05.py-val
N_OUTER_SPLITS = 5                             # 5-fold -> 25 külső fold
RANDOM_STATE = 2026
N_BOOTSTRAP_CI = env_int("IM_N_BOOT_CI", 2000)

TARGET_COLUMN = "hozam"
PARCEL_COLUMN = "n_parcella"
SEASON_COLUMN = "season"
PREFIX = "im04_"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = Path(os.environ.get("IM_INPUT", str(SCRIPT_DIR / "prediktor5_190_245_hozam.csv")))
OUTPUT_BASE = Path(os.environ.get("IM_OUTPUT_BASE", str(SCRIPT_DIR / "indexmod4out")))

CFG = {"random_state": RANDOM_STATE, "inner_folds": 3, "perm_repeats": 10}

MODEL_NAMES = ["MLR", "Ridge", "Elastic Net", "SVR", "PLSR", "Random Forest", "Gradient Boosting", "KNN", "XGBoost"]
REF_NAMES = ["Ref_year_mean", "Ref_global_mean"]
IMPORTANCE_METHODS = ["perm_mse_increase", "std_coef", "gini", "shap_mean_abs", "rfe_rank"]
STAT_ORDER = ["mean", "median", "std", "auc"]
STAT_RE = re.compile(r"^(?P<index>.+)_t_(?P<stat>mean|median|std|auc)$")


def apply_quick_mode():
    global CV_SEEDS, N_BOOTSTRAP_CI
    CV_SEEDS = CV_SEEDS[:1]
    N_BOOTSTRAP_CI = 100
    CFG["perm_repeats"] = 3


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

    def table(self, df: pd.DataFrame, name: str, desc: str) -> Path:
        path = self.dir / f"{PREFIX}{name}.csv"
        save_csv(df, path)
        self.manifest.append({"file": str(path.relative_to(self.dir)), "type": "table", "rows": len(df), "description": desc})
        return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, sep=";", decimal=",", encoding="utf-8-sig")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("indexmodel04")
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
# 3. ADAT ÉS PREDIKTORKÉSZLETEK
# =====================================================================================
def load_data(path: Path):
    """Ugyanaz a beolvasás és szűrés, mint a pgroup05.py-ban (a sorrend és a row_id azonos)."""
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


def make_predictor_sets(predictors, logger):
    """Indexenként a 4 időbeli leíró (<INDEX>_t_mean/_median/_std/_auc); nincs szelekció."""
    by_index = {}
    for c in predictors:
        m = STAT_RE.match(c)
        if m:
            by_index.setdefault(m["index"], []).append(c)
        else:
            logger.warning("A(z) %s oszlop nem illik az <INDEX>_t_<stat> mintára, kihagyva az indexkészletekből.", c)
    sets = {}
    for idx, cols in by_index.items():
        if INDEX_FILTER and idx not in INDEX_FILTER:
            continue
        cols = sorted(cols, key=lambda c: STAT_ORDER.index(STAT_RE.match(c)["stat"]))
        if len(cols) != 4:
            logger.warning("A(z) %s indexhez %d prediktor tartozik (4-et vártunk): %s", idx, len(cols), cols)
        sets[idx] = cols
    if INCLUDE_ALL_PREDICTORS:
        sets["ALL"] = list(predictors)
    if not sets:
        raise ValueError("Nincs használható prediktorkészlet (az oszlopnevek: <INDEX>_t_mean/median/std/auc).")
    return sets


# =====================================================================================
# 4. MODELLEK
# =====================================================================================
def build_model(name: str, seed: int, n_features: int):
    """Pipeline (StandardScaler + modell) és hiperparaméter-rács (a pgroup05.py rácsai)."""
    if name == "MLR":
        est, grid = LinearRegression(), {}
    elif name == "Ridge":
        est = Ridge()
        grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    elif name == "Elastic Net":
        est = TransformedTargetRegressor(regressor=ElasticNet(max_iter=50000, random_state=seed), transformer=StandardScaler())
        grid = {"model__regressor__alpha": [0.001, 0.01, 0.1, 0.5, 1.0],
                "model__regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
    elif name == "SVR":
        est = TransformedTargetRegressor(regressor=SVR(kernel="rbf"), transformer=StandardScaler())
        grid = {"model__regressor__C": [0.1, 1, 10, 100], "model__regressor__gamma": ["scale", "auto", 0.01, 0.1]}
    elif name == "PLSR":
        est = PLSRegression(scale=False)
        grid = {"model__n_components": list(range(1, min(5, n_features) + 1))}
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


def model_importances(name, pipe, Xtr, ytr, Xte, yte, cfg, seed):
    """Fontosságok egy külső foldon. Permutációs: minden modell; std_coef: lineáris modellek;
    Gini, SHAP, RFE: Random Forest; SHAP: Gradient Boosting és XGBoost."""
    feats = list(Xtr.columns)
    est, sc = pipe.named_steps["model"], pipe.named_steps["scaler"]
    out = []
    try:
        pi = permutation_importance(pipe, Xte, yte, scoring="neg_mean_squared_error",
                                    n_repeats=cfg["perm_repeats"], random_state=seed)
        out += [("perm_mse_increase", f, float(v)) for f, v in zip(feats, pi.importances_mean)]
    except Exception:  # noqa: BLE001
        pass
    try:  # standardizált együtthatók: kg/ha változás 1 SD prediktorváltozásra
        coefs = None
        if name in ("MLR", "Ridge", "PLSR"):
            coefs = np.ravel(est.coef_)
        elif name == "Elastic Net":
            coefs = np.ravel(est.regressor_.coef_) * float(np.ravel(est.transformer_.scale_)[0])
        if coefs is not None and len(coefs) == len(feats):
            out += [("std_coef", f, float(v)) for f, v in zip(feats, coefs)]
    except Exception:  # noqa: BLE001
        pass
    if name == "Random Forest":
        out += [("gini", f, float(v)) for f, v in zip(feats, est.feature_importances_)]
    if name in ("Random Forest", "Gradient Boosting", "XGBoost") and shap is not None:
        try:
            sv = shap.TreeExplainer(est).shap_values(sc.transform(Xte))
            out += [("shap_mean_abs", f, float(v)) for f, v in zip(feats, np.abs(sv).mean(axis=0))]
        except Exception:  # noqa: BLE001
            pass
    if name == "Random Forest":
        try:
            rfe = RFE(RandomForestRegressor(**est.get_params()), n_features_to_select=1, step=1).fit(sc.transform(Xtr), ytr)
            out += [("rfe_rank", f, float(v)) for f, v in zip(feats, rfe.ranking_)]
        except Exception:  # noqa: BLE001
            pass
    return out


# =====================================================================================
# 5. EGY KÜLSŐ FOLD / TASK FUTTATÁSA
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
    ytr, gtr, yte = y[tr], groups[tr], y[te]
    meta = {"task_id": task["task_id"], "kind": task["kind"], "outer_id": task["outer_id"], "seed": task["seed"],
            "fold": task["fold"], "perm_id": task["perm_id"]}
    res = {k: [] for k in ("preds", "params", "importance", "timing")}
    res["meta"] = meta
    log(f"TASK {task['task_id']} indul: train={len(tr)} sor ({len(np.unique(gtr))} parcella), test={len(te)} sor")

    def add_preds(sel, model, pred):
        for j, p in zip(te, pred):
            res["preds"].append({**meta, "selector": sel, "model": model, "row_id": int(task["row_ids"][j]),
                                 "n_parcella": parcels[j], "season": seasons[j], "observed": float(y[j]), "predicted": float(p)})

    n_splits = min(cfg["inner_folds"], len(np.unique(gtr)))
    for sel, cols in task["pred_sets"].items():
        Xtr_s, Xte_s = X.iloc[tr][cols], X.iloc[te][cols]
        tm = time.perf_counter()
        for mname in task["models"]:
            t1 = time.perf_counter()
            pipe, grid = build_model(mname, cfg["random_state"], len(cols))
            gs = GridSearchCV(pipe, grid, cv=GroupKFold(n_splits=n_splits), scoring="neg_mean_squared_error",
                              n_jobs=N_JOBS, refit=True)
            gs.fit(Xtr_s, ytr, groups=gtr)
            pred = np.ravel(gs.predict(Xte_s))
            add_preds(sel, mname, pred)
            res["params"].append({**meta, "selector": sel, "model": mname,
                                  "best_params": json.dumps(clean_params(gs.best_params_), default=str),
                                  "inner_cv_rmse": math.sqrt(max(0.0, -gs.best_score_)), "n_train": len(tr), "n_test": len(te),
                                  "fit_seconds": time.perf_counter() - t1})
            if task["run_importance"]:
                for method, f, v in model_importances(mname, gs.best_estimator_, Xtr_s, ytr, Xte_s, yte, cfg, task["sel_seed"]):
                    res["importance"].append({**meta, "selector": sel, "model": mname, "method": method, "predictor": f, "value": v})
            log(f"  [{sel}] {mname:18s} belső CV RMSE={math.sqrt(max(0.0, -gs.best_score_)):8.1f} | "
                f"teszt RMSE={math.sqrt(np.mean((pred - yte) ** 2)):8.1f} | {time.perf_counter() - t1:5.1f} s")
        res["timing"].append({**meta, "selector": sel, "models_seconds": time.perf_counter() - tm})
    if task["run_ref"]:
        tr_seasons = seasons[tr]
        ymeans = {s: float(ytr[tr_seasons == s].mean()) for s in np.unique(tr_seasons)}
        gmean = float(ytr.mean())
        add_preds("none", "Ref_year_mean", [ymeans.get(seasons[j], gmean) for j in te])
        add_preds("none", "Ref_global_mean", [gmean] * len(te))
    res["total_seconds"] = time.perf_counter() - t0
    log(f"TASK {task['task_id']} kész ({res['total_seconds']:.1f} s)")
    return res


def run_tasks(tasks: list, label: str, logger: logging.Logger) -> list:
    n = len(tasks)
    results, t0 = [], time.perf_counter()
    for i, t in enumerate(tasks, 1):
        r = run_task(t)
        results.append(r)
        el = time.perf_counter() - t0
        eta = el / i * (n - i)
        logger.info(f"[{label}] {i}/{n} kész | task={r['meta']['task_id']} | eltelt {fmt_time(el)} | hátralévő ~{fmt_time(eta)}")
    return results


def collect(results: list, key: str) -> pd.DataFrame:
    return pd.DataFrame([row for r in results for row in r[key]])


def make_task(kind, task_id, tdata, train_idx, test_idx, pred_sets, **kw):
    base = {"kind": kind, "task_id": task_id, "cfg": CFG, "X": tdata["X"], "y": tdata["y"], "groups": tdata["groups"],
            "seasons": tdata["seasons"], "parcels": tdata["groups"], "row_ids": tdata["row_ids"], "train_idx": train_idx,
            "test_idx": test_idx, "pred_sets": pred_sets, "models": MODEL_NAMES, "log_dir": str(tdata["log_dir"]),
            "echo": False, "outer_id": -1, "seed": -1, "fold": -1, "perm_id": -1, "sel_seed": RANDOM_STATE,
            "run_ref": False, "run_importance": False}
    base.update(kw)
    return base


# =====================================================================================
# 6. METRIKÁK, BOOTSTRAP (a pgroup05.py-val azonos számítások)
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
    return out


def _ci_row(d, point_val):
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return lo, hi


def bootstrap_ci(pred: pd.DataFrame, data: pd.DataFrame, seasons, B: int, seed: int):
    """Parcella-bootstrap az out-of-fold predikciókon. Visszaad: CI, páros (modellek egy készleten belül),
    páros (készletek egy modellen belül), páros (modell vs. Ref_year_mean, pooled és évenként)."""
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

    def pair_row(a, b, metric="RMSE"):
        d = boots[metric][:, a] - boots[metric][:, b]
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return {"diff_A_minus_B": point[metric][a] - point[metric][b], "ci_low": lo, "ci_high": hi,
                "share_A_better": float(np.nanmean(d < 0)), "ci_excludes_zero": int(lo > 0 or hi < 0)}

    pm_rows, ps_rows, pr_rows = [], [], []
    for sel in sorted({k[0] for k in keys}):                      # modellek egy prediktorkészleten belül
        ids = [i for i, k in enumerate(keys) if k[0] == sel]
        for a, b in itertools.combinations(ids, 2):
            r = pair_row(a, b)
            pm_rows.append({"selector": sel, "model_A": keys[a][1], "model_B": keys[b][1],
                            "RMSE_diff_A_minus_B": r["diff_A_minus_B"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                            "share_A_better": r["share_A_better"], "ci_excludes_zero": r["ci_excludes_zero"]})
    for mod in sorted({k[1] for k in keys if k[0] != "none"}):    # prediktorkészletek egy modellen belül
        ids = [i for i, k in enumerate(keys) if k[1] == mod and k[0] != "none"]
        for a, b in itertools.combinations(ids, 2):
            r = pair_row(a, b)
            ps_rows.append({"model": mod, "set_A": keys[a][0], "set_B": keys[b][0],
                            "RMSE_diff_A_minus_B": r["diff_A_minus_B"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                            "share_A_better": r["share_A_better"], "ci_excludes_zero": r["ci_excludes_zero"]})
    if ("none", "Ref_year_mean") in keys:                         # modell vs. évátlag-referencia
        ref = keys.index(("none", "Ref_year_mean"))
        metrics = ["RMSE"] + [f"RMSE_{s}" for s in seasons if f"RMSE_{s}" in point]
        for ki, (sel, mod) in enumerate(keys):
            if sel == "none":
                continue
            for metric in metrics:
                r = pair_row(ki, ref, metric)
                pr_rows.append({"selector": sel, "model": mod, "reference": "Ref_year_mean", "metric": metric,
                                "diff_model_minus_ref": r["diff_A_minus_B"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                                "share_model_better": r["share_A_better"], "ci_excludes_zero": r["ci_excludes_zero"]})
    return pd.DataFrame(ci_rows), pd.DataFrame(pm_rows), pd.DataFrame(ps_rows), pd.DataFrame(pr_rows)


# =====================================================================================
# 7. ÁBRASEGÉDEK
# =====================================================================================
def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _model_order(models):
    return [m for m in MODEL_NAMES if m in set(models)]


def _sets(df):
    return [s for s in dict.fromkeys(df.selector) if s != "none"]


def _for_selector(df, sel):
    return df[df.selector == sel]


def plot_fold_rmse(df, path):
    for sel in _sets(df):
        d = _for_selector(df, sel)
        ms = _model_order(d.model)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot([d[d.model == m].RMSE.values for m in ms])
        ax.set_xticks(range(1, len(ms) + 1))
        ax.set_xticklabels(ms, rotation=25, ha="right")
        ax.set_ylabel("RMSE of outer test fold (kg ha$^{-1}$)")
        ax.set_title(f"Outer fold RMSE distribution ({sel})")
        ax.grid(True, ls=":", alpha=0.5)
        _finish(fig, path.parent / f"{path.stem}_{sel}{path.suffix}")


def plot_season_metrics(df, path):
    for sel in _sets(df):
        d = _for_selector(df, sel)
        ms, seas = _model_order(d.model), list(dict.fromkeys(d.season))
        w = 0.8 / max(1, len(seas))
        for col, lab in [("R2_mean", "Within-season R$^2$"), ("NRMSE_pct_mean", "NRMSE (%)")]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for k, s in enumerate(seas):
                dd = d[d.season == s].set_index("model").reindex(ms)
                ax.bar(np.arange(len(ms)) + k * w, dd[col].values, w, yerr=dd[col.replace("_mean", "_sd")].values,
                       capsize=2, label=str(s))
            ax.set_xticks(np.arange(len(ms)) + w * (len(seas) - 1) / 2)
            ax.set_xticklabels(ms, rotation=25, ha="right")
            ax.set_ylabel(lab)
            ax.set_title(f"{lab} by season ({sel})")
            ax.axhline(0, color="k", lw=0.6)
            ax.legend(title="Season")
            ax.grid(True, axis="y", ls=":", alpha=0.5)
            _finish(fig, path.parent / f"{path.stem}_{sel}_{'r2' if 'R2' in col else 'nrmse'}{path.suffix}")


def plot_importance(df, path):
    d0 = df[df.selector != "ALL"]
    stats = [s for s in STAT_ORDER if s in set(d0["stat"])]
    for (method, model), d in d0.groupby(["method", "model"]):
        sets = list(dict.fromkeys(d.selector))
        h = 0.8 / max(1, len(sets))
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, sel in enumerate(sets):
            dd = d[d.selector == sel].set_index("stat").reindex(stats)
            ax.bar(np.arange(len(stats)) + k * h, dd["mean"].values, h, yerr=dd["sd"].values, capsize=2, label=sel)
        ax.set_xticks(np.arange(len(stats)) + h * (len(sets) - 1) / 2)
        ax.set_xticklabels([f"t_{s}" for s in stats])
        title_map = {"perm_mse_increase": "Permutation importance (MSE increase)", "gini": "Gini impurity importance",
                     "shap_mean_abs": "SHAP mean |value|", "rfe_rank": "RFE rank (1 = best)",
                     "std_coef": "Standardized coefficient (kg ha$^{-1}$ per SD)"}
        ax.set_title(f"{title_map.get(method, method)}: {model}")
        ax.axhline(0, color="k", lw=0.6)
        ax.legend(title="Index", fontsize=8)
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        _finish(fig, path.parent / f"{path.stem}_{method}_{_slug(model)}{path.suffix}")


def plot_loyo(df, path):
    for sel in _sets(df):
        d = _for_selector(df, sel)
        ms, seas = _model_order(d.model), list(dict.fromkeys(d.held_out_season))
        w = 0.8 / max(1, len(seas))
        fig, ax = plt.subplots(figsize=(10, 5))
        for k, s in enumerate(seas):
            dd = d[d.held_out_season == s].set_index("model").reindex(ms)
            ax.bar(np.arange(len(ms)) + k * w, dd.RMSE.values, w, label=f"held-out {s}")
        ax.set_xticks(np.arange(len(ms)) + w * (len(seas) - 1) / 2)
        ax.set_xticklabels(ms, rotation=25, ha="right")
        ax.set_ylabel("RMSE (kg ha$^{-1}$)")
        ax.set_title(f"Leave-one-year-out RMSE ({sel})")
        ax.legend()
        ax.grid(True, axis="y", ls=":", alpha=0.5)
        _finish(fig, path.parent / f"{path.stem}_{sel}{path.suffix}")


def plot_set_comparison(df, path):
    sets, ms = _sets(df), [m for m in MODEL_NAMES if m in set(df.model)]
    w = 0.8 / max(1, len(sets))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k, sel in enumerate(sets):
        d = df[df.selector == sel].set_index("model").reindex(ms)
        ax.bar(np.arange(len(ms)) + k * w, d.RMSE_mean.values, w, yerr=d.RMSE_sd.values, capsize=2, label=sel)
    ax.set_xticks(np.arange(len(ms)) + w * (len(sets) - 1) / 2)
    ax.set_xticklabels(ms, rotation=25, ha="right")
    ax.set_ylabel("Pooled RMSE, mean of repeats (kg ha$^{-1}$)")
    ax.set_title("Single-index models: pooled out-of-fold RMSE by model")
    ax.legend(title="Predictor set", fontsize=8)
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    _finish(fig, path)


def plot_heatmap(df, path):
    d = df[df.model.isin(MODEL_NAMES)].copy()
    scols = [c for c in d.columns if re.match(r"^R2_\d+_mean$", c)]
    d["R2_within"] = d[scols].mean(axis=1) if scols else np.nan
    sets, ms = _sets(d), [m for m in MODEL_NAMES if m in set(d.model)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.9 * len(sets) + 2.5))
    for ax, col, title, cmap in [(axes[0], "RMSE_mean", "Pooled RMSE (kg ha$^{-1}$)", "viridis_r"),
                                 (axes[1], "R2_within", "Within-season R$^2$ (mean of seasons)", "viridis")]:
        M = d.pivot(index="selector", columns="model", values=col).reindex(index=sets, columns=ms)
        im = ax.imshow(M.values.astype(float), aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels(ms, rotation=35, ha="right")
        ax.set_yticks(range(len(sets)))
        ax.set_yticklabels(sets)
        for i in range(len(sets)):
            for j in range(len(ms)):
                v = M.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.0f}" if col == "RMSE_mean" else f"{v:.2f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    _finish(fig, path)


FIG_SPECS = {
    "fig01_fold_rmse_boxplot": (plot_fold_rmse, "RMSE of outer test folds per model and predictor set"),
    "fig03_within_season_metrics": (plot_season_metrics, "Within-season R2 and NRMSE per model and predictor set"),
    "fig05_importance_oof": (plot_importance, "Importances aggregated over outer folds, per method and model"),
    "fig06_loyo_rmse": (plot_loyo, "Leave-one-year-out RMSE"),
    "fig09_index_comparison": (plot_set_comparison, "Pooled RMSE by model and predictor set"),
    "fig10_index_model_heatmap": (plot_heatmap, "Predictor set x model heatmap: pooled RMSE and within-season R2"),
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
    (run_dir / "figures").mkdir(exist_ok=True)
    for name, (fn, _) in FIG_SPECS.items():
        p = run_dir / "figure_data" / f"{name}.csv"
        if p.exists():
            print(f"Újrarajzolás: {name}", flush=True)
            fn(read_csv(p), run_dir / "figures" / f"{name}.png")
    print("Kész.", flush=True)


# =====================================================================================
# 10. FŐPROGRAM
# =====================================================================================
def main():
    ap = argparse.ArgumentParser(description="indexmodel04: egyindexes modellek a pgroup05.py-val összevethető rendszerben")
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
    logger = setup_logger(run_dir / "indexmodel04.log")
    logger.info("=== indexmodel04 indul: előre rögzített indexkészletek, parcel-grouped nested CV ===")

    import joblib
    import sklearn
    import xgboost
    config_dump = {"RUN_LOYO": RUN_LOYO, "RUN_IMPORTANCE": RUN_IMPORTANCE,
                   "RUN_REFERENCE_MODELS": RUN_REFERENCE_MODELS, "INCLUDE_ALL_PREDICTORS": INCLUDE_ALL_PREDICTORS,
                   "INDEX_FILTER": INDEX_FILTER, "N_JOBS": N_JOBS, "CV_SEEDS": CV_SEEDS, "N_OUTER_SPLITS": N_OUTER_SPLITS,
                   "N_BOOTSTRAP_CI": N_BOOTSTRAP_CI, "MODEL_NAMES": MODEL_NAMES,
                   "quick_mode": bool(args.quick), "input": args.input, "CFG": CFG}
    for k, v in config_dump.items():
        logger.info("  %s = %r", k, v)
    soft = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "xgboost": getattr(xgboost, "__version__", "?"), "joblib": joblib.__version__,
            "shap": getattr(shap, "__version__", "nincs telepítve")}
    for k, v in soft.items():
        logger.info("  SOFTWARE/%s = %s", k, v)
    (run_dir / "indexmodel04_run_config.json").write_text(json.dumps({"config": config_dump, "software": soft}, indent=2, default=str),
                                                          encoding="utf-8")

    # ---------------- adat ----------------
    data, predictors = load_data(Path(args.input))
    pred_sets = make_predictor_sets(predictors, logger)
    seasons = sorted(data[SEASON_COLUMN].unique())
    parcel_ids = np.sort(data[PARCEL_COLUMN].unique())
    logger.info("Adat: %d sor, %d prediktor, %d parcella, évjáratok: %s", len(data), len(predictors), len(parcel_ids), seasons)
    for s, c in pred_sets.items():
        logger.info("  prediktorkészlet %-6s: %s", s, c)
    tdata = {"X": data[predictors].copy(), "y": data[TARGET_COLUMN].to_numpy(float), "groups": data[PARCEL_COLUMN].to_numpy(),
             "seasons": data[SEASON_COLUMN].to_numpy(), "row_ids": data["row_id"].to_numpy(), "log_dir": out.log_dir}
    seasons_arr, groups_arr = tdata["seasons"], tdata["groups"]

    ds = data.groupby(SEASON_COLUMN)[TARGET_COLUMN].agg(n="count", mean="mean", sd="std", min="min", max="max").reset_index()
    out.table(ds, "data_summary", "Yield descriptive statistics per season")
    logger.info("Hozam évenként:\n%s", ds.to_string(index=False))

    # ---------------- külső felosztások (azonosak a pgroup05.py-éval) ----------------
    outer_tasks, split_rows, oid = [], [], 0
    for seed in CV_SEEDS:
        kf = KFold(n_splits=N_OUTER_SPLITS, shuffle=True, random_state=seed)
        for fold, (tp, ep) in enumerate(kf.split(parcel_ids), 1):
            oid += 1
            tr_idx = np.where(np.isin(groups_arr, parcel_ids[tp]))[0]
            te_idx = np.where(np.isin(groups_arr, parcel_ids[ep]))[0]
            split_rows.append({"outer_id": oid, "seed": seed, "fold": fold, "n_train_rows": len(tr_idx), "n_test_rows": len(te_idx),
                               "train_parcels": ";".join(map(str, parcel_ids[tp])), "test_parcels": ";".join(map(str, parcel_ids[ep]))})
            outer_tasks.append(make_task("outer", f"outer_{oid:02d}", tdata, tr_idx, te_idx, pred_sets, outer_id=oid, seed=seed,
                                         fold=fold, sel_seed=RANDOM_STATE + oid, run_ref=RUN_REFERENCE_MODELS,
                                         run_importance=RUN_IMPORTANCE))
    out.table(pd.DataFrame(split_rows), "outer_splits", "Outer CV splits (parcel-grouped; identical to pgroup05.py)")
    logger.info("Külső CV: %d seed x %d-fold = %d külső fold | prediktorkészletek: %s | modellek: %s | belső magok: %d",
                len(CV_SEEDS), N_OUTER_SPLITS, len(outer_tasks), list(pred_sets), MODEL_NAMES, N_JOBS)

    res_outer = run_tasks(outer_tasks, "KÜLSŐ CV", logger)
    pred = collect(res_outer, "preds")
    out.table(pred, "predictions_outer", "All out-of-fold predictions per predictor set/model/outer fold")
    out.table(collect(res_outer, "params"), "fold_model_params", "Tuned hyperparameters and inner-CV RMSE per outer fold")
    out.table(collect(res_outer, "timing"), "timing_per_fold", "Model-fitting time per outer fold and predictor set")

    # ---------------- metrikák ----------------
    logger.info("Metrikák számítása...")
    seed_m, fold_m = per_seed_metrics(pred, seasons), per_fold_metrics(pred)
    summ = summarize(seed_m)
    out.table(seed_m, "metrics_per_seed", "Pooled and per-season metrics for each repeat")
    out.table(fold_m, "metrics_per_fold", "Metrics on each outer test fold")
    out.table(summ, "metrics_summary", "Mean and SD of metrics over repeats")
    ci, pairs_models, pairs_sets, pairs_ref = bootstrap_ci(pred, data, seasons, N_BOOTSTRAP_CI, RANDOM_STATE)
    out.table(ci, "metrics_bootstrap_ci", "Parcel-bootstrap 95% CIs for metrics")
    out.table(pairs_models, "pairwise_rmse_differences", "Paired parcel-bootstrap RMSE differences between models (within predictor set)")
    out.table(pairs_sets, "pairwise_rmse_differences_between_sets", "Paired parcel-bootstrap RMSE differences between predictor sets (within model)")
    if not pairs_ref.empty:
        out.table(pairs_ref, "pairwise_vs_reference", "Model vs Ref_year_mean: pooled and within-season RMSE differences (bootstrap CI)")

    emit_figure(out, "fig01_fold_rmse_boxplot", fold_m[["selector", "model", "outer_id", "seed", "fold", "RMSE", "R2", "NRMSE_pct"]])
    sm_rows = []
    for _, r in summ.iterrows():
        for s in seasons:
            if f"R2_{s}_mean" in summ.columns:
                sm_rows.append({"selector": r.selector, "model": r.model, "season": s, "R2_mean": r[f"R2_{s}_mean"], "R2_sd": r[f"R2_{s}_sd"],
                                "NRMSE_pct_mean": r[f"NRMSE_pct_{s}_mean"], "NRMSE_pct_sd": r[f"NRMSE_pct_{s}_sd"],
                                "RMSE_mean": r[f"RMSE_{s}_mean"], "RMSE_sd": r[f"RMSE_{s}_sd"]})
    emit_figure(out, "fig03_within_season_metrics", pd.DataFrame(sm_rows))
    emit_figure(out, "fig09_index_comparison", summ)
    emit_figure(out, "fig10_index_model_heatmap", summ)

    # ---------------- fontosságok ----------------
    imp = collect(res_outer, "importance")
    imp_s = pd.DataFrame()
    if not imp.empty:
        out.table(imp, "importance_oof_raw", "Importances per outer fold (permutation, standardized coefficient, Gini, SHAP, RFE)")
        g = imp.groupby(["selector", "model", "method", "predictor"])["value"]
        imp_s = pd.concat([g.mean().rename("mean"), g.std(ddof=1).rename("sd"), g.size().rename("n_folds")], axis=1).reset_index()
        imp_s["stat"] = imp_s.predictor.map(lambda p: STAT_RE.match(p)["stat"] if STAT_RE.match(p) else "all")
        out.table(imp_s, "importance_oof_summary", "Importances aggregated over outer folds")
        emit_figure(out, "fig05_importance_oof", imp_s)

    # ---------------- LOYO ----------------
    if RUN_LOYO:
        loyo_tasks = []
        for k, s in enumerate(seasons):
            tr_idx, te_idx = np.where(seasons_arr != s)[0], np.where(seasons_arr == s)[0]
            loyo_tasks.append(make_task("loyo", f"loyo_{s}", tdata, tr_idx, te_idx, pred_sets, sel_seed=RANDOM_STATE + 5000 + k,
                                        run_ref=RUN_REFERENCE_MODELS, fold=k + 1))
        res_loyo = run_tasks(loyo_tasks, "LOYO", logger)
        pred_loyo = collect(res_loyo, "preds")
        pred_loyo["held_out_season"] = pred_loyo["season"]
        out.table(pred_loyo, "predictions_loyo", "Leave-one-year-out predictions")
        out.table(collect(res_loyo, "params"), "params_loyo", "Tuned hyperparameters in LOYO runs")
        lo_rows = []
        for (sel, mod, hs), gp in pred_loyo.groupby(["selector", "model", "held_out_season"]):
            m = calc_metrics(gp.observed, gp.predicted)
            lo_rows.append({"selector": sel, "model": mod, "held_out_season": hs, **{k: m[k] for k in METRIC_KEYS}, "n": m["n"]})
        lo = pd.DataFrame(lo_rows)
        pooled = []
        for (sel, mod), gp in pred_loyo.groupby(["selector", "model"]):
            m = calc_metrics(gp.observed, gp.predicted)
            pooled.append({"selector": sel, "model": mod, "held_out_season": "pooled", **{k: m[k] for k in METRIC_KEYS}, "n": m["n"]})
        out.table(pd.concat([lo, pd.DataFrame(pooled)], ignore_index=True), "metrics_loyo", "LOYO metrics per held-out season")
        emit_figure(out, "fig06_loyo_rmse", lo)

    # ---------------- lezárás ----------------
    try:
        with pd.ExcelWriter(run_dir / f"{PREFIX}key_tables.xlsx", engine="openpyxl") as xw:
            for nm, df_ in [("summary", summ), ("bootstrap_ci", ci), ("pairwise_models", pairs_models), ("pairwise_sets", pairs_sets),
                            ("vs_reference", pairs_ref), ("importance", imp_s)]:
                if df_ is not None and not df_.empty:
                    df_.to_excel(xw, sheet_name=nm, index=False)
        out.manifest.append({"file": f"{PREFIX}key_tables.xlsx", "type": "table", "rows": "", "description": "Key tables in Excel workbook"})
    except Exception as exc:  # noqa: BLE001
        logger.info("Excel-munkafüzet kihagyva: %s", exc)
    out.manifest.append({"file": "indexmodel04.log", "type": "log", "rows": "", "description": "Main log"})
    out.table(pd.DataFrame(out.manifest), "output_manifest", "List of all output files")
    ref_rows = summ[summ.model.isin(REF_NAMES)]
    if not ref_rows.empty:
        logger.info("Referenciák:\n%s", ref_rows[["model", "R2_mean", "RMSE_mean", "NRMSE_pct_mean"]].to_string(index=False))
    logger.info("=== KÉSZ. Teljes futási idő: %s | kimenet: %s ===", fmt_time(time.perf_counter() - t_start), run_dir)


if __name__ == "__main__":
    main()
