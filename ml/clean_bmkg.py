import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "processed" / "bmkg_all_raw.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "bmkg_clean.csv"

# nama kolom yang kita inginkan di output
COLUMN_MAP = {
    "DATE (GMT)": "datetime",
    "LINTANG (°)": "lat",
    "BUJUR (°)": "lon",
    "MAGNITUDO (M)": "mag",
    "KEDALAMAN (KM)": "depth",
}

def main():
    print("Baca data dari:", RAW_PATH)

    # BACA TANPA HEADER dulu, karena header asli ada di salah satu baris
    df_raw = pd.read_csv(RAW_PATH, header=None)

    # CARI baris yang berisi teks "DATE (GMT)" di kolom pertama
    header_rows = df_raw.index[df_raw.iloc[:, 0] == "DATE (GMT)"].tolist()
    if not header_rows:
        print("Tidak menemukan baris header 'DATE (GMT)' di file CSV. Cek kembali bmkg_all_raw.csv.")
        return

    header_idx = header_rows[0]
    print("Header ditemukan di baris index:", header_idx)

    # SET baris itu sebagai header, dan data mulai dari baris setelahnya
    header = df_raw.iloc[header_idx]
    df = df_raw.iloc[header_idx + 1 :].copy()
    df.columns = header

    print("Kolom setelah set header:", list(df.columns))

    # Pilih & rename kolom yang kita butuhkan
    missing = [c for c in COLUMN_MAP.keys() if c not in df.columns]
    if missing:
        print("Masih ada kolom yang belum ketemu:", missing)
        return

    df = df[list(COLUMN_MAP.keys())].rename(columns=COLUMN_MAP)

    # UBAH TIPE DATA
    # 1) datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # 2) angka
    for col in ["lat", "lon", "mag", "depth"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # BUANG baris yang tidak punya datetime / lat / lon / mag
    df = df.dropna(subset=["datetime", "lat", "lon", "mag"]).reset_index(drop=True)

    # TAMBAH kolom turunan
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day

    # FILTER magnitudo minimal (boleh ganti 4.5 sesuai rancanganmu)
    df = df[df["mag"] >= 4.5].reset_index(drop=True)

    print("Jumlah baris setelah dibersihkan:", len(df))

    # SIMPAN
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print("Data bersih disimpan di:", OUT_PATH)


if __name__ == "__main__":
    main()
