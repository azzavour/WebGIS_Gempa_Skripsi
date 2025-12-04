import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "Data Gempa BMKG"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    files = sorted(RAW_DIR.rglob("*.xlsx"))
    print(f"Jumlah file yang ditemukan: {len(files)}")

    if not files:
        print("Tidak ada file .xlsx di folder", RAW_DIR)
        return

    dfs = []
    for f in files:
        print("Baca:", f)

        # HEADER tabel ada di baris ke-9 Excel (index 8, 0-based)
        # jadi kita pakai header=8
        df = pd.read_excel(
            f,
            sheet_name=0,
            header=8,          # baris 9 jadi nama kolom
            usecols="A:E",     # DATE, LINTANG, BUJUR, KEDALAMAN, MAGNITUDO
        )

        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)

    out_path = OUT_DIR / "bmkg_all_raw.csv"
    all_data.to_csv(out_path, index=False)
    print("Selesai! File gabungan disimpan di:", out_path)


if __name__ == "__main__":
    main()
