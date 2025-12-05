import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

GRID_PATH = BASE_DIR / "data" / "processed" / "grid_yearly_dataset.csv"
OUT_PATH  = BASE_DIR / "data" / "processed" / "training_dataset.csv"


def main():
    print("Baca data grid tahunan dari:", GRID_PATH)
    df = pd.read_csv(GRID_PATH)

    print("Contoh kolom:", list(df.columns))

    # df berisi: grid_id, grid_lat, grid_lon, year, event_count, mean_mag, max_mag, mean_depth, event_occur

    # Kita ingin target = kejadian tahun berikutnya.
    # Caranya: buat df_next = data yang digeser satu tahun ke belakang.
    df_next = df[["grid_id", "year", "event_occur", "event_count"]].copy()
    # Tahun di df_next dikurangi 1, supaya nanti "tahun ini" ketemu dengan "tahun depan"
    df_next["year"] = df_next["year"] - 1
    df_next = df_next.rename(
        columns={
            "event_occur": "event_next",
            "event_count": "event_count_next",
        }
    )

    # Gabungkan df (tahun ini) dengan df_next (tahun depan) berdasarkan grid_id & tahun
    merged = df.merge(
        df_next,
        on=["grid_id", "year"],
        how="inner",
        validate="one_to_one",
    )

    print("Jumlah baris setelah digabung dengan tahun berikutnya:", len(merged))

    # Pilih kolom yang akan dipakai untuk training
    cols_order = [
        "grid_id",
        "grid_lat",
        "grid_lon",
        "year",
        "event_count",
        "mean_mag",
        "max_mag",
        "mean_depth",
        "event_occur",       # kondisi tahun ini (bisa jadi fitur)
        "event_count_next",
        "event_next",        # <- TARGET utama (0/1)
    ]

    merged = merged[cols_order].copy()

    # Simpan ke CSV
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print("Dataset training disimpan di:", OUT_PATH)


if __name__ == "__main__":
    main()
