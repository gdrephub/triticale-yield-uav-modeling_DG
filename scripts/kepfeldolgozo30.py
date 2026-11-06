###
#   kepfeldolgozo30.py
#
#   Ortomozaikok -> parcella x időpont vegetációs index statisztikák (Dezső et al., triticálé UAV cikk)
#
#   A kepfeldolgozo29.py-hoz képest a változások:
#   - A textúraelemzés kikerült (mahotas nem kell). Az elforgatás kódja (a szög számítása és a sávonkénti
#     forgatás) szó szerint a v29-ből van átmásolva, változtatás nélkül.
#   - Egységes érvényességi maszk: egy pixel akkor érvényes, ha a parcellapoligonon belül van, és mind a négy
#     sávban érvényes (nem NaN, nem 0, nem -10000, nem nodata). Minden index ugyanezeken a pixeleken készül.
#   - Index-készlet: NDVI, GNDVI, NDRE, CIre, CIgreen, OSAVI, EVI2, WDRVI, SR, MCARI2.
#     (Kikerült: a RECI, mert azonos a CIre-vel; a PSRI, mert módosított változat volt; és az MTVI2, mert a
#     képlete algebrailag azonos az MCARI2-ével: mindkét számláló 1,2*NIR - 2,5*R + 1,3*G.)
#   - Indexenként: n, mean, median, std (ddof=0), cv, p10, p25, p75, p90, min, max.
#     Sávonként: mean, median, std (a későbbi, utólagos index-számításhoz és a reflektancia-ellenőrzéshez).
#   - Új oszlopok: kezelés, parcella középpont (UTM) és terület, érvényes pixelarány, reflektancia-ellenőrzés.
#   - Az időpontok rendezettek, az eltelt napok közvetlenül a dátumból számolódnak (nem listaindexből).
#   - Hibás parcella/időpont nem marad észrevétlen: hibalista és darabszám-ellenőrzés a végén.
#   - Új kimenetek: repülési napló (repulesi_naplo_*.csv), összevont CSV (Excel-barát), évenkénti xlsx.
#
#   A parcella azonosítója a shapefile sorrendje (1-től), a hozam CSV ugyanebben a sorrendben kell, hogy legyen.
###

import os
import re
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import cv2
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from shapely.ops import transform
from pyproj import Transformer

###
### Beállítások
###

TIDOSZAK = ["2024", "2025"]  ###!!! nagyon fontos: a termesztési időszakok (mozaikok<év> könyvtárak)
TIFF_FOLDER = r"c:\dg\kepelemzes01\rawdata"
EGYEDICIMKE = "v30a"  # a kimeneti fájlok és a napló nevébe kerül

MASZK_PATH = os.path.join(TIFF_FOLDER, "maszk", "kombajn.shp")  # az aratott sávok = a maszkolt terület
# Opcionális kezelés-hozzárendelés: UTF-8 CSV, oszlopai: parcel_id;kezeles  (parcel_id = a shapefile sorrendje 1-től)
# Ha nincs ilyen fájl, a shapefile attribútumai között keres "kezeles"/"treatment" oszlopot.
KEZELES_CSV = os.path.join(TIFF_FOLDER, "maszk", "kezeles.csv")
OUTPUT_DIR = os.path.join(TIFF_FOLDER, "kimenet")

VART_PARCELLASZAM = 20
VART_CRS = "EPSG:32634"
WDRVI_ALPHA = 0.2  # a WDRVI súlyozó együtthatója (irodalmi tartomány: 0,1-0,2)

STAT_INDEX = ["n", "mean", "median", "std", "cv", "p10", "p25", "p75", "p90", "min", "max"]
STAT_SAV = ["mean", "median", "std"]
SAV_NEVEK = ["G", "R", "RE", "NIR"]  # a TIFF sávsorrendje: 1=zöld, 2=vörös, 3=red-edge, 4=NIR

CSV_SEP = ";"      # magyar Excel: pontosvessző
CSV_DECIMAL = ","  # magyar Excel: tizedesvessző (pandas-ban: decimal=",")

FILE_PATTERN = re.compile(r"Orthomosaic2_(\d{8})_(data|rgb)\.tif$", re.IGNORECASE)

EPSILON = 1e-10

###
### Segédfüggvények
###


def safe_divide(numerator, denominator):
    """Osztás, ahol a 0 (vagy nagyon kicsi) nevező eredménye NaN. Nem torzít epsilon hozzáadásával."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(denominator) > EPSILON, numerator / denominator, np.nan)
    result = np.asarray(result, dtype=np.float64)
    result[~np.isfinite(result)] = np.nan
    return result


def calculate_vegetation_indices(g, r, re, nir):
    """Vegetációs indexek az érvényes pixelek 1D vektoraiból (reflektancia, 0-1 skála)."""
    idx = {}
    idx["NDVI"] = safe_divide(nir - r, nir + r)
    idx["GNDVI"] = safe_divide(nir - g, nir + g)
    idx["NDRE"] = safe_divide(nir - re, nir + re)
    idx["CIre"] = safe_divide(nir, re) - 1
    idx["CIgreen"] = safe_divide(nir, g) - 1
    idx["OSAVI"] = 1.16 * safe_divide(nir - r, nir + r + 0.16)
    idx["EVI2"] = 2.5 * safe_divide(nir - r, nir + 2.4 * r + 1)
    idx["WDRVI"] = safe_divide(WDRVI_ALPHA * nir - r, WDRVI_ALPHA * nir + r)
    idx["SR"] = safe_divide(nir, r)
    with np.errstate(invalid="ignore"):
        gyok = np.sqrt((2 * nir + 1) ** 2 - (6 * nir - 5 * np.sqrt(np.abs(r))) - 0.5)
    idx["MCARI2"] = safe_divide(1.5 * (2.5 * (nir - r) - 1.3 * (nir - g)), gyok)
    return idx


def vector_statistics(values, names):
    """Statisztikák egy 1D vektorból; a nem véges értékeket kihagyja. Üres bemenetnél NaN (n=0)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = int(v.size)
    full = {"n": n}
    keys = ("mean", "median", "std", "cv", "p10", "p25", "p75", "p90", "min", "max")
    if n == 0:
        for k in keys:
            full[k] = np.nan
    else:
        mean = float(v.mean())
        std = float(v.std())  # ddof=0, mint a v29-ben
        p10, p25, p50, p75, p90 = np.percentile(v, [10, 25, 50, 75, 90])
        full.update(
            mean=mean,
            median=float(p50),
            std=std,
            cv=(std / abs(mean)) if abs(mean) > EPSILON else np.nan,
            p10=float(p10),
            p25=float(p25),
            p75=float(p75),
            p90=float(p90),
            min=float(v.min()),
            max=float(v.max()),
        )
    return {k: full[k] for k in names}


def load_treatment(parcellak, logger):
    """Kezelés-hozzárendelés: 1) KEZELES_CSV, 2) shapefile attribútum, 3) üres (figyelmeztetéssel)."""
    n = len(parcellak)
    if os.path.exists(KEZELES_CSV):
        df = pd.read_csv(KEZELES_CSV, sep=None, engine="python", encoding="utf-8-sig")
        df.columns = [str(c).strip().lower() for c in df.columns]
        if {"parcel_id", "kezeles"} <= set(df.columns):
            mp = dict(zip(df["parcel_id"].astype(int), df["kezeles"].astype(str).str.strip()))
            logger.info("Kezelés-hozzárendelés a %s fájlból.", KEZELES_CSV)
            return [mp.get(i + 1, "") for i in range(n)]
        logger.warning("A %s fájlban nincs parcel_id és kezeles oszlop, figyelmen kívül hagyva.", KEZELES_CSV)
    for col in parcellak.columns:
        if str(col).lower() in ("kezeles", "kezelés", "treatment"):
            logger.info("Kezelés-hozzárendelés a shapefile '%s' oszlopából.", col)
            return parcellak[col].astype(str).str.strip().tolist()
    logger.warning(
        "NINCS kezelés-hozzárendelés (sem %s, sem shapefile oszlop): a 'kezeles' oszlop üres lesz!", KEZELES_CSV
    )
    return [""] * n


def load_yield(path, n_parcella):
    hozam = np.atleast_1d(np.genfromtxt(path, skip_header=1))
    if hozam.ndim != 1 or hozam.size != n_parcella:
        raise ValueError(
            f"A hozam fájl ({path}) alakja {hozam.shape}, de {n_parcella} parcellához egy oszlop kellene."
        )
    return hozam


def process_parcel(src_data, geom, i, base_info, parcel_meta, treatment):
    """Egy parcella kivágása és statisztikái egy időpontra. Visszaad egy rekordot (dict)."""
    out_image_data, out_transform_data = mask(src_data, [geom], crop=True)

    # A sávok kiolvasása (v29)
    # Itt fontos a sorrend, az elnevezés és a számozás is
    savok = []
    keplista = []
    # msband_G
    savok.append(out_image_data[0])
    # msband_R
    savok.append(out_image_data[1])
    # msband_Edge
    savok.append(out_image_data[2])
    # msband_NIR
    savok.append(out_image_data[3])

    # a parcella érvényes pixeleinek száma forgatás előtt (az érvényes pixelarány viszonyítási alapja)
    n_poly = int(np.sum(np.isfinite(savok[3]) & (savok[3] != 0) & (savok[3] != -10000)))

    # Sáv elforgatása, hogy majd a textúra elemzéshez megfelelő legyen
    for irot in range(len(savok)):
        # logger.debug("a tömb típusa: ",irot, savok[irot].dtype)
        if savok[irot].dtype != np.float64:
            imr = (savok[irot]).astype(np.float64).copy()
        else:
            imr = (savok[irot]).copy()
        # logger.debug("imr típusa: ",irot, imr.dtype)
        # imrgbR[imrgbR==0]=np.nan
        (h, w) = imr.shape[:2]
        center = (w, 0)

        # Forgatási mátrix
        M = cv2.getRotationMatrix2D(center, angle, scale)
        # Kép elforgatása
        imrot = cv2.warpAffine(
            imr, M, (2 * w, 2 * h), flags=cv2.INTER_NEAREST
        )
        imr[imr == 0] = np.nan
        imrot[imrot == 0] = np.nan
        imr[imr == -10000] = np.nan
        imrot[imrot == -10000] = np.nan
        # relevans_pix_eredeti = np.sum(~np.isnan(imrgbR))
        # logger.debug("Releváns pixelek az eredeti képen:", relevans_pix_eredeti)
        # relevans_pix_forgatott = np.sum(~np.isnan(rotated_imrgbR))
        # logger.debug("Releváns pixelek az elforgatott képen:", relevans_pix_forgatott)
        rotmask = ~np.isnan(imrot)  # releváns pixelek
        if not np.any(rotmask):
            logger.debug("return None  # ha minden NaN, nincs mit kivágni")
        rows = np.any(rotmask, axis=1)
        cols = np.any(rotmask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        keplista.append(imrot[rmin : rmax + 1, cmin : cmax + 1])

    # Az elforgatott, kivágott sávok (2D, NaN = érvénytelen) -> érvényes pixelek 1D vektorai
    if len({k.shape for k in keplista}) != 1:
        raise ValueError(f"A forgatott sávok mérete eltér: {[k.shape for k in keplista]}")
    bands = keplista
    valid = np.ones(bands[0].shape, dtype=bool)
    for b in bands:
        valid &= np.isfinite(b)
        if src_data.nodata is not None:
            valid &= b != src_data.nodata
    n_valid = int(valid.sum())

    g, r, re, nir = (b[valid] for b in bands)
    vec = dict(zip(SAV_NEVEK, (g, r, re, nir)))

    record = dict(base_info)
    record.update(
        {
            "parcel_id": i + 1,
            "kezeles": treatment,
            "yield_kg_ha": base_info["_hozam"][i],
            "parcel_x_utm": parcel_meta["x"][i],
            "parcel_y_utm": parcel_meta["y"][i],
            "parcel_area_m2": parcel_meta["area"][i],
            "px_polygon": n_poly,
            "px_valid": n_valid,
            "valid_frac": (n_valid / n_poly) if n_poly > 0 else np.nan,
        }
    )
    record.pop("_hozam", None)

    # Sávok
    for name in SAV_NEVEK:
        for stat_name, stat_value in vector_statistics(vec[name], STAT_SAV).items():
            record[f"{name}_{stat_name}"] = stat_value

    # Reflektancia-ellenőrzés: a sávmediánok 0-1 körüli reflektanciát jelentsenek (DN esetén nagyobbak)
    medians = [record[f"{name}_median"] for name in SAV_NEVEK]
    record["refl_check"] = (
        "OK" if all(np.isfinite(m) and 0 <= m <= 1.5 for m in medians) else "CHECK"
    )

    # Indexek
    if n_valid > 0:
        indices = calculate_vegetation_indices(g, r, re, nir)
    else:
        indices = {k: np.array([]) for k in calculate_vegetation_indices(
            np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])
        )}
    for index_name, index_values in indices.items():
        for stat_name, stat_value in vector_statistics(index_values, STAT_INDEX).items():
            record[f"{index_name}_{stat_name}"] = stat_value
    return record


###
### Főprogram
###

starttime = time.time()
starttimep = time.process_time()
print("A program elindult. DG")

os.makedirs(OUTPUT_DIR, exist_ok=True)

logfile = os.path.join(OUTPUT_DIR, f"naplo_{EGYEDICIMKE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(logfile, mode="w", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("DG_kronikas")
logging.getLogger("rasterio").setLevel(logging.WARNING)
logger.info("A program naplózásának kezdete: %s", datetime.now())

# Parcellák
parcellak = gpd.read_file(MASZK_PATH).reset_index(drop=True)
n_parcella = len(parcellak)
logger.info("Parcellák száma a shapefile-ban: %s, CRS: %s", n_parcella, parcellak.crs)
logger.info("A shapefile oszlopai: %s", list(parcellak.columns))
logger.debug("A shapefile első sorai (a sorrend ellenőrzéséhez):\n%s", parcellak.drop(columns="geometry").head(25))
if n_parcella != VART_PARCELLASZAM:
    logger.warning("A parcellák száma %s, a várt %s!", n_parcella, VART_PARCELLASZAM)
if parcellak.crs is None:
    raise ValueError("A shapefile-nak nincs koordináta-rendszere (.prj hiányzik?).")

parcellak_utm = parcellak.to_crs(VART_CRS)
maszk_crs = parcellak.crs
logger.info(
    " Feltételezem, hogy a tif képfájlok EPSG:32634 KR-ben vannak. \n Ezt ellenőrizni kell."
)
tif_crs = "EPSG:32634"
tlap = parcellak.loc[0, "geometry"]
geojson_tlap = [mapping(tlap)]
# KR átváltás !!! fontos !!!
if maszk_crs != tif_crs:
    tformer = Transformer.from_crs(maszk_crs, tif_crs, always_xy=True)
    tlap = transform(tformer.transform, tlap)
    geojson_tlap = [mapping(tlap)]
# Az elforgatási szög kiszámítása
# coords = list(teglalap.exterior.coords)
coords = np.array(list(tlap.exterior.coords))
bal_also = coords[np.argmin(coords[:, 0])]
bal_felso = coords[np.argmax(coords[:, 1])]
dx = bal_felso[0] - bal_also[0]
dy = bal_felso[1] - bal_also[1]
angle = 90 - np.degrees(np.arctan2(dy, dx))
scale = 1.0
logger.info("coords: \n %s", coords)
logger.info("bal_felso: %s", bal_felso)
logger.info("bal_also: %s", bal_also)
logger.info(f"dx,dy: {dx},{dy}")
logger.info("angle: %s", angle)
parcel_meta = {
    "x": [float(g.centroid.x) for g in parcellak_utm.geometry],
    "y": [float(g.centroid.y) for g in parcellak_utm.geometry],
    "area": [float(g.area) for g in parcellak_utm.geometry],
}
kezeles = load_treatment(parcellak, logger)

geom_cache = {}


def geometries_for(src_crs):
    """A parcellák GeoJSON-geometriái a raszter koordináta-rendszerében (CRS-enként egyszer számolva)."""
    key = src_crs.to_wkt()
    if key not in geom_cache:
        geom_cache[key] = [mapping(g) for g in parcellak.to_crs(key).geometry]
    return geom_cache[key]


records = []
hibak = []
kezdo_datumok = {}
varhato_sorok = 0

for ev in TIDOSZAK:
    folder = os.path.join(TIFF_FOLDER, f"mozaikok{ev}")
    logger.info("=== Termesztési időszak: %s (%s) ===", ev, folder)

    with open(os.path.join(folder, f"kezdo_datum_{ev}.txt"), "r") as fkezdo:
        kezdo_datum = pd.to_datetime(fkezdo.read().strip(), format="%Y%m%d")
    kezdo_datumok[ev] = kezdo_datum
    logger.info("A %s időszak kezdő dátuma: %s", ev, kezdo_datum.date())

    fajlok = {}
    for filename in os.listdir(folder):
        match = FILE_PATTERN.match(filename)
        if match and match.group(2).lower() == "data":
            fajlok[match.group(1)] = os.path.join(folder, filename)
    datumok = sorted(fajlok)  # időrendi sorrend
    logger.info("%s időszak: %s db 'data' ortomozaik: %s", ev, len(datumok), datumok)
    varhato_sorok += len(datumok) * n_parcella

    hozam = load_yield(os.path.join(folder, f"hozam{ev}.csv"), n_parcella)
    logger.debug("hozam %s:\n%s", ev, hozam)

    for date_str in datumok:
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            logger.error("Hibás dátumformátum: %s", date_str)
            continue
        eltelt_napok = (pd.Timestamp(date_obj) - kezdo_datum).days
        logger.info("OOOOOOOO  %s időszak, %s, a kezdő dátum óta eltelt napok: %s", ev, date_str, eltelt_napok)

        base_info = {
            "season": ev,
            "date": date_obj.strftime("%Y-%m-%d"),
            "year": date_obj.year,
            "month": date_obj.month,
            "day": date_obj.day,
            "doy": date_obj.timetuple().tm_yday,
            "days_after_start": int(eltelt_napok),
            "source_tif": os.path.basename(fajlok[date_str]),
            "_hozam": hozam,
        }

        try:
            with rasterio.open(fajlok[date_str]) as src:
                logger.debug("Sávok száma: %s, méret: %s x %s, CRS: %s", src.count, src.width, src.height, src.crs)
                logger.debug("Sávleírások: %s, nodata: %s, dtypes: %s", src.descriptions, src.nodata, src.dtypes)
                if src.count < 4:
                    raise ValueError(f"Csak {src.count} sáv van a fájlban, legalább 4 kellene (G, R, RE, NIR).")
                if str(src.crs) != VART_CRS:
                    logger.warning("A raszter CRS-e %s, nem %s.", src.crs, VART_CRS)
                geoms = geometries_for(src.crs)

                for i, geom in enumerate(geoms):
                    try:
                        records.append(process_parcel(src, geom, i, base_info, parcel_meta, kezeles[i]))
                    except Exception as e:
                        logger.error("Hiba: %s, parcella %s: %s", date_str, i + 1, e)
                        hibak.append({"season": ev, "date": date_str, "parcel_id": i + 1, "hiba": str(e)})
        except Exception as e:
            logger.error("Hiba a fájl megnyitásakor: %s\n%s", fajlok[date_str], e)
            hibak.append({"season": ev, "date": date_str, "parcel_id": "mind", "hiba": str(e)})

###
### Ellenőrzések és kimenetek
###

df_all = pd.DataFrame(records)
logger.info("Összesen %s rekord (várt: %s).", len(df_all), varhato_sorok)
if len(df_all) != varhato_sorok:
    logger.warning("A rekordok száma eltér a várttól (%s vs %s): lásd a hibalistát!", len(df_all), varhato_sorok)

if hibak:
    df_hibak = pd.DataFrame(hibak)
    hibak_path = os.path.join(OUTPUT_DIR, f"hibak_{EGYEDICIMKE}.csv")
    df_hibak.to_csv(hibak_path, sep=CSV_SEP, decimal=CSV_DECIMAL, index=False, encoding="utf-8-sig")
    logger.warning("%s hiba történt, lásd: %s", len(hibak), hibak_path)

if not df_all.empty:
    # Oszlopok sorrendje: azonosítók, majd sávok, majd indexek
    info_cols = [
        "season", "date", "year", "month", "day", "doy", "days_after_start",
        "parcel_id", "kezeles", "yield_kg_ha", "parcel_x_utm", "parcel_y_utm", "parcel_area_m2",
        "px_polygon", "px_valid", "valid_frac", "refl_check", "source_tif",
    ]
    df_all = df_all[info_cols + [c for c in df_all.columns if c not in info_cols]]

    # Évenkénti xlsx (parcella x időpont, széles formátum)
    for ev in TIDOSZAK:
        out_xlsx = os.path.join(OUTPUT_DIR, f"longout_{EGYEDICIMKE}_{ev}.xlsx")
        df_all[df_all["season"] == ev].to_excel(out_xlsx, sheet_name="longout", index=False)
        logger.info("Kimeneti fájl elkészült: %s", out_xlsx)

    # Összevont CSV, magyar Excel-barát formátumban (UTF-8 BOM, ';' elválasztó, ',' tizedes)
    out_csv = os.path.join(OUTPUT_DIR, f"osszes_{EGYEDICIMKE}.csv")
    df_all.to_csv(out_csv, sep=CSV_SEP, decimal=CSV_DECIMAL, index=False, encoding="utf-8-sig")
    logger.info("Összevont CSV elkészült: %s", out_csv)

    # Repülési napló: időpontonként (a cikk repülési dátum-táblázatához és az ellenőrzéshez)
    naplo = (
        df_all.groupby(["season", "date", "days_after_start", "source_tif"])
        .agg(
            n_parcella=("parcel_id", "count"),
            valid_frac_atlag=("valid_frac", "mean"),
            G_median=("G_median", "median"),
            R_median=("R_median", "median"),
            RE_median=("RE_median", "median"),
            NIR_median=("NIR_median", "median"),
            refl_check=("refl_check", lambda s: "OK" if (s == "OK").all() else "CHECK"),
        )
        .reset_index()
    )
    naplo["kezdo_datum"] = naplo["season"].map(lambda s: kezdo_datumok[s].strftime("%Y-%m-%d"))
    out_naplo = os.path.join(OUTPUT_DIR, f"repulesi_naplo_{EGYEDICIMKE}.csv")
    naplo.to_csv(out_naplo, sep=CSV_SEP, decimal=CSV_DECIMAL, index=False, encoding="utf-8-sig")
    logger.info("Repülési napló elkészült: %s", out_naplo)
    for ev in TIDOSZAK:
        logger.info("%s időszak: %s repülés.", ev, int((naplo["season"] == ev).sum()))
    if (naplo["refl_check"] == "CHECK").any():
        logger.warning(
            "Néhány időpontban a sávmediánok kívül esnek a 0-1,5 tartományon: lehet, hogy a TIFF nem reflektancia!"
        )

endtime = time.time()
endtimep = time.process_time()
logger.info("A program futásának falióra időtartama: %s s", endtime - starttime)
logger.info("A program futásának CPU időtartama: %s s", endtimep - starttimep)
