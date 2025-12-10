import pandas as pd
import numpy as np
from pathlib import Path
import re
import unicodedata
from datetime import datetime
import warnings

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_BPS_DIR = BASE_DIR / "data" / "raw" / "Data Penduduk BPS"
MASTER_OUT = BASE_DIR / "data" / "processed" / "bps_kecamatan_master.csv"
AGG_OUT = BASE_DIR / "data" / "processed" / "bps_kabkota_yearly.csv"

KEYWORDS = ("kecamatan", "distrik", "jumlah", "penduduk")
NUMERIC_COLS = ["population", "density", "growth_rate", "sex_ratio"]
REQUIRED_COLS = ["kecamatan", *NUMERIC_COLS]
COLUMN_PATTERNS = [
    ("kecamatan", "kecamatan"),
    ("district", "kecamatan"),
    ("distrik", "kecamatan"),
    ("kec", "kecamatan"),
    ("kepadatan", "density"),
    ("density", "density"),
    ("laju pertumbuhan", "growth_rate"),
    ("pertumbuhan", "growth_rate"),
    ("growth", "growth_rate"),
    ("rasio jenis kelamin", "sex_ratio"),
    ("rasio kelamin", "sex_ratio"),
    ("sex ratio", "sex_ratio"),
    ("sexratio", "sex_ratio"),
    ("jumlah penduduk", "population"),
    ("total penduduk", "population"),
    ("penduduk", "population"),
]
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")

warnings.filterwarnings("ignore", category=UserWarning)


def detect_header_row(df_preview: pd.DataFrame) -> int:
    if df_preview is None or df_preview.empty:
        return 0
    for idx, row in df_preview.iterrows():
        joined = " ".join(
            str(value).lower() for value in row.tolist() if pd.notna(value)
        )
        if any(keyword in joined for keyword in KEYWORDS):
            return idx
    return 0


def normalize_column_label(label: object) -> str:
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return ""
    text = str(label)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.strip()


def map_standard_name(label: str) -> str:
    for pattern, target in COLUMN_PATTERNS:
        if pattern in label:
            if target == "population" and any(
                marker in label for marker in ["laki", "perempuan", "lk", "pr"]
            ):
                continue
            return target
    return label.replace(" ", "_") or ""


def standardize_columns(columns) -> list:
    new_cols = []
    counters = {}
    for idx, col in enumerate(columns):
        normalized = normalize_column_label(col)
        mapped = map_standard_name(normalized)
        if not mapped:
            mapped = f"unnamed_{idx}"
        counters[mapped] = counters.get(mapped, 0) + 1
        if counters[mapped] > 1:
            mapped = f"{mapped}_{counters[mapped]}"
        new_cols.append(mapped)
    return new_cols


def clean_string_value(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    lowered = text.lower()
    if lowered in {"nan", "none", "-", "", "jumlah", "total", "grand total"}:
        return pd.NA
    return re.sub(r"\s+", " ", text)


def coerce_numeric(value: object) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text or text.lower() in {"nan", "na", "-", ""}:
        return np.nan
    text = text.replace("\xa0", "")
    text = re.sub(r"[^0-9,.-]", "", text)
    if not text or text in {"-", ""}:
        return np.nan
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    else:
        if text.count(".") > 1:
            text = text.replace(".", "")
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return np.nan


def extract_year_from_path(path: Path) -> object:
    match = YEAR_PATTERN.search(str(path))
    if match:
        year = int(match.group(0))
        current_year = datetime.now().year + 5
        if 1900 <= year <= current_year:
            return year
    return pd.NA


def clean_region_name(name: str) -> str:
    if not name:
        return ""
    name = str(name)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name)
    return name.strip()


def parse_metadata(path: Path) -> dict:
    rel_path = path.relative_to(RAW_BPS_DIR)
    dir_parts = rel_path.parts[:-1]
    province = clean_region_name(dir_parts[0]) if dir_parts else ""
    kabkota = ""
    for part in dir_parts[1:]:
        candidate = clean_region_name(part)
        low = candidate.lower()
        if any(keyword in low for keyword in ["kab", "kota", "administratif", "regency"]):
            kabkota = candidate
            break
    if not kabkota and len(dir_parts) >= 2:
        kabkota = clean_region_name(dir_parts[1])
    if not kabkota:
        stem = clean_region_name(path.stem)
        tokens = re.split(r"[_-]", stem)
        for token in tokens:
            cand = clean_region_name(token)
            low = cand.lower()
            if any(keyword in low for keyword in ["kab", "kota", "administratif", "reg"]):
                kabkota = cand
                break
    metadata = {
        "provinsi": province,
        "kabkota": kabkota,
        "year": extract_year_from_path(rel_path),
    }
    return metadata


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, header=None, dtype=str, encoding=enc, engine="python")
        except Exception as exc:
            last_error = exc
    raise last_error


def load_raw_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm", ".xlsb"}:
        return pd.read_excel(path, sheet_name=0, header=None, dtype=str)
    if suffix == ".csv":
        return read_csv_with_fallback(path)
    return pd.DataFrame()


def process_file(path: Path) -> pd.DataFrame:
    try:
        df_raw = load_raw_table(path)
    except Exception as exc:
        print(f"[WARN] Gagal membaca {path.name}: {exc}")
        return pd.DataFrame()
    if df_raw.empty:
        return pd.DataFrame()
    preview = df_raw.head(20)
    header_idx = detect_header_row(preview)
    if header_idx >= len(df_raw) - 1:
        return pd.DataFrame()
    headers = df_raw.iloc[header_idx].fillna("").tolist()
    data = df_raw.iloc[header_idx + 1 :].copy()
    data.columns = standardize_columns(headers)
    data = data.dropna(how="all")
    data = data.loc[:, data.notna().any()].copy()
    for col in REQUIRED_COLS:
        if col not in data.columns:
            data[col] = pd.NA
    data = data[REQUIRED_COLS]
    data = data.applymap(lambda v: np.nan if isinstance(v, str) and not v.strip() else v)
    data["kecamatan"] = data["kecamatan"].apply(clean_string_value)
    data = data.dropna(subset=["kecamatan"])
    for col in NUMERIC_COLS:
        data[col] = data[col].apply(coerce_numeric)
    pop_median = data["population"].median(skipna=True)
    if pd.notna(pop_median) and pop_median < 1000:
        data["population"] = data["population"].apply(lambda v: v * 1000 if pd.notna(v) else v)
    metadata = parse_metadata(path)
    data["provinsi"] = metadata.get("provinsi", "")
    data["kabkota"] = metadata.get("kabkota", "")
    data["year"] = metadata.get("year", pd.NA)
    data["source_file"] = str(path.relative_to(RAW_BPS_DIR).as_posix())
    ordered_cols = [
        "provinsi",
        "kabkota",
        "kecamatan",
        "year",
        "population",
        "density",
        "growth_rate",
        "sex_ratio",
        "source_file",
    ]
    return data[ordered_cols]


def gather_data_files() -> list:
    if not RAW_BPS_DIR.exists():
        return []
    files = [
        path
        for path in RAW_BPS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".xlsm", ".xlsb", ".csv"}
    ]
    return sorted(files)


def sum_with_nan(series: pd.Series) -> float:
    return series.sum(min_count=1)


def main():
    files = gather_data_files()
    if not files:
        print("Tidak ada file BPS ditemukan.")
        return
    frames = []
    processed = 0
    for path in files:
        df = process_file(path)
        if df.empty:
            continue
        frames.append(df)
        processed += 1
    if not frames:
        print("Tidak ada data BPS yang berhasil diproses.")
        return
    master_df = pd.concat(frames, ignore_index=True)
    master_df["year"] = pd.to_numeric(master_df["year"], errors="coerce").astype("Int64")
    MASTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(MASTER_OUT, index=False)
    agg_df = (
        master_df.groupby(["provinsi", "kabkota", "year"], dropna=False)
        .agg(
            population=("population", sum_with_nan),
            density=("density", "mean"),
            growth_rate=("growth_rate", "mean"),
            sex_ratio=("sex_ratio", "mean"),
        )
        .reset_index()
    )
    agg_df.to_csv(AGG_OUT, index=False)
    print(f"Total file dibaca: {processed}")
    print(f"Total baris master: {len(master_df)}")
    print("Contoh 5 baris:")
    print(master_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
