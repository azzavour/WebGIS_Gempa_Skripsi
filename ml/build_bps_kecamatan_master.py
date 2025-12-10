from pathlib import Path
import pandas as pd

# --- PATH PROJECT & DATA ---
# Asumsi: file ini ada di folder "ml/" di root repo
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BPS_ROOT = PROJECT_ROOT / "data" / "raw" / "Data Penduduk BPS"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_XLSX = OUTPUT_DIR / "bps_kecamatan_master.xlsx"
OUTPUT_CSV = OUTPUT_DIR / "bps_kecamatan_master.csv"

# File daftar kabupaten yang kamu kirim tadi
DAFTAR_FILE = PROJECT_ROOT / "Daftar_Kabupaten_Kota_Indonesia_UPDATED.xlsx"

ALLOWED_EXT = {".xlsx", ".xls", ".csv"}


# ========== 1. BACA & SIAPKAN TABEL TAHUN PER KAB/KOTA ==========

def load_tahun_mapping():
    """
    Baca Daftar_Kabupaten_Kota_Indonesia_UPDATED.xlsx
    lalu buat mapping: (Provinsi, Nama Kab/Kota) -> Tahun
    """
    if not DAFTAR_FILE.exists():
        print(f"[WARNING] File daftar kabupaten tidak ditemukan: {DAFTAR_FILE}")
        return {}

    df = pd.read_excel(DAFTAR_FILE)

    # Isi provinsi yang kosong dengan nilai di atasnya
    if "Provinsi" in df.columns:
        df["Provinsi"] = df["Provinsi"].ffill()
    else:
        raise ValueError("Kolom 'Provinsi' tidak ditemukan di daftar kabupaten.")

    # Bersihkan nama
    df["Provinsi"] = df["Provinsi"].astype(str).str.strip()
    df["Nama"] = df["Nama"].astype(str).str.strip()

    # Tahun -> integer (boleh NaN kalau kosong)
    if "Tahun" in df.columns:
        df["Tahun"] = pd.to_numeric(df["Tahun"], errors="coerce").astype("Int64")
    else:
        df["Tahun"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    mapping = {}
    for _, row in df.iterrows():
        prov = row["Provinsi"]
        kab = row["Nama"]
        tahun = row["Tahun"]

        if pd.isna(prov) or pd.isna(kab) or pd.isna(tahun):
            continue

        key = (prov.strip(), kab.strip())
        mapping[key] = int(tahun)

    return mapping


def normalize_kab_folder_name(folder_name: str) -> str:
    """
    Nama folder kabupaten di BPS biasanya:
      - 'Kabupaten Aceh Barat Daya' atau
      - 'Kota Bekasi'
    Sedangkan di daftar kabupaten, kolom Nama = 'Aceh Barat Daya', 'Bekasi'.

    Fungsi ini menghapus prefix 'Kabupaten ' / 'Kota '.
    """
    name = folder_name.strip()
    for prefix in ["Kabupaten ", "Kota "]:
        if name.startswith(prefix):
            return name[len(prefix):].strip()
    return name


# ========== 2. FUNGSI BANTU DETEKSI KOLOM ==========

def detect_col(cols, keywords, exact=False):
    """
    Cari nama kolom yang mengandung semua keyword (case-insensitive).
    - exact=True: nama kolom harus persis (setelah lower & strip)
    - exact=False: asal keyword-nya ada di mana saja di nama kolom
    """
    cols_list = list(cols)
    cols_lower = [str(c).lower() for c in cols_list]

    if exact:
        for c, cl in zip(cols_list, cols_lower):
            for kw in keywords:
                if cl.strip() == kw.lower().strip():
                    return c
        return None

    for c, cl in zip(cols_list, cols_lower):
        if all(kw.lower() in cl for kw in keywords):
            return c
    return None


def read_bps_file(file_path: Path) -> pd.DataFrame:
    """Baca file BPS (xls/xlsx/csv) apa adanya."""
    ext = file_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df


# ========== 3. PROSES SATU FILE MENJADI FORMAT STANDAR ==========

def process_one_file(
    file_path: Path,
    provinsi: str,
    kabupaten_folder_name: str,
    tahun_mapping: dict,
) -> pd.DataFrame | None:
    """Konversi satu file menjadi DF standar per kecamatan."""
    try:
        df = read_bps_file(file_path)
    except Exception as e:
        print(f"[ERROR] Gagal baca {file_path}: {e}")
        return None

    if df.empty:
        print(f"[WARNING] File kosong: {file_path}")
        return None

    # --- Wilayah: 'Kecamatan' atau 'Distrik' ---
    wilayah_col = detect_col(df.columns, ["kecamatan"])
    if wilayah_col is None:
        wilayah_col = detect_col(df.columns, ["distrik"])
    if wilayah_col is None:
        print(f"[WARNING] Tidak menemukan kolom kecamatan/distrik di {file_path}, dilewati.")
        return None

    # --- Jumlah penduduk ---
    penduduk_col = detect_col(df.columns, ["jumlah", "penduduk"])
    if penduduk_col is None:
        penduduk_col = detect_col(df.columns, ["penduduk"])
    if penduduk_col is None:
        print(f"[WARNING] Tidak menemukan kolom jumlah penduduk di {file_path}, dilewati.")
        return None

    # --- Optional: kepadatan, rasio JK, laju, persentase ---
    kepadatan_col = detect_col(df.columns, ["kepadatan"])
    rasio_col = detect_col(df.columns, ["rasio", "jenis kelamin"]) \
        or detect_col(df.columns, ["jenis kelamin"])
    laju_col = detect_col(df.columns, ["laju", "pertumbuhan"])
    persentase_col = detect_col(df.columns, ["persentase", "penduduk"])

    # --- Siapkan output standar ---
    out = pd.DataFrame()

    # Kecamatan (atau distrik disamakan jadi kecamatan)
    out["kecamatan"] = df[wilayah_col].astype(str).str.strip()

    # Jumlah penduduk -> numeric
    penduduk = pd.to_numeric(df[penduduk_col], errors="coerce")

    # Kalau nama kolom penduduk mengandung '(Ribu)', berarti satuannya ribuan
    if "(ribu" in str(penduduk_col).lower():
        penduduk = penduduk * 1000

    out["jumlah_penduduk"] = penduduk

    # Optional info (kalau tidak ada, isi NaN)
    if kepadatan_col is not None:
        out["kepadatan_penduduk_km2"] = pd.to_numeric(df[kepadatan_col], errors="coerce")
    else:
        out["kepadatan_penduduk_km2"] = pd.NA

    if rasio_col is not None:
        out["rasio_jenis_kelamin"] = pd.to_numeric(df[rasio_col], errors="coerce")
    else:
        out["rasio_jenis_kelamin"] = pd.NA

    if laju_col is not None:
        out["laju_pertumbuhan"] = pd.to_numeric(df[laju_col], errors="coerce")
    else:
        out["laju_pertumbuhan"] = pd.NA

    if persentase_col is not None:
        out["persentase_penduduk"] = pd.to_numeric(df[persentase_col], errors="coerce")
    else:
        out["persentase_penduduk"] = pd.NA

    # --- Info daerah untuk WebGIS ---
    prov_clean = provinsi.strip()
    kab_folder = kabupaten_folder_name.strip()
    kab_clean = normalize_kab_folder_name(kab_folder)

    out["provinsi"] = prov_clean           # contoh: 'Kalimantan Tengah'
    out["kabupaten"] = kab_folder          # contoh: 'Kabupaten Katingan'

    # --- Tahun: diambil dari daftar kabupaten ---
    tahun = tahun_mapping.get((prov_clean, kab_clean))
    if tahun is None:
        print(
            f"[WARNING] Tidak menemukan tahun untuk "
            f"({prov_clean}, {kab_clean}) di daftar kabupaten. Tahun = NaN. File: {file_path}"
        )
        out["tahun"] = pd.NA
    else:
        out["tahun"] = int(tahun)

    # Simpan info file sumber
    out["source_file"] = str(file_path.relative_to(PROJECT_ROOT))

    return out


# ========== 4. LOOP SEMUA PROVINSI / KABUPATEN / FILE ==========

def build_bps_kecamatan_master():
    if not BPS_ROOT.exists():
        raise FileNotFoundError(f"Folder BPS tidak ditemukan: {BPS_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tahun_mapping = load_tahun_mapping()
    print(f"Loaded mapping tahun untuk {len(tahun_mapping)} kabupaten/kota.")

    all_rows = []

    # Struktur folder:
    # data/raw/Data Penduduk BPS/<Provinsi>/<Kabupaten>/*.xlsx
    for prov_path in sorted(BPS_ROOT.iterdir()):
        if not prov_path.is_dir():
            continue

        provinsi = prov_path.name

        for kab_path in sorted(prov_path.iterdir()):
            if not kab_path.is_dir():
                continue

            kabupaten_folder_name = kab_path.name

            data_files = [
                f for f in kab_path.iterdir()
                if f.is_file() and f.suffix.lower() in ALLOWED_EXT
            ]

            if not data_files:
                print(f"[INFO] Tidak ada file data di folder {kab_path}")
                continue

            for fpath in data_files:
                print(f"Proses file: {fpath}")
                df_std = process_one_file(
                    fpath,
                    provinsi=provinsi,
                    kabupaten_folder_name=kabupaten_folder_name,
                    tahun_mapping=tahun_mapping,
                )
                if df_std is not None:
                    all_rows.append(df_std)

    if not all_rows:
        raise RuntimeError("Tidak ada data yang berhasil diproses dari folder BPS.")

    master_df = pd.concat(all_rows, ignore_index=True)

    master_df.to_excel(OUTPUT_XLSX, index=False)
    master_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Selesai. Baris total: {len(master_df)}")
    print(f"Disimpan ke: {OUTPUT_XLSX}")
    print(f"dan:         {OUTPUT_CSV}")


if __name__ == "__main__":
    build_bps_kecamatan_master()
