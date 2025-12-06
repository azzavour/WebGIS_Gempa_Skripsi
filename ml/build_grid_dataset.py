import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "bmkg_clean.csv"
OUT_PATH   = BASE_DIR / "data" / "processed" / "grid_yearly_dataset.csv"

GRID_SIZE = 1.0  # derajat

LAT_MIN, LAT_MAX = -11, 6
LON_MIN, LON_MAX = 95, 142


def main():
    print("Baca data dari:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    # Filter area Indonesia
    df = df[
        (df["lat"].between(LAT_MIN, LAT_MAX)) &
        (df["lon"].between(LON_MIN, LON_MAX))
    ].copy()
    print("Jumlah kejadian setelah filter area:", len(df))

    # Hitung koordinat grid
    df["grid_lat"] = (np.floor(df["lat"] / GRID_SIZE) * GRID_SIZE).round(3)
    df["grid_lon"] = (np.floor(df["lon"] / GRID_SIZE) * GRID_SIZE).round(3)
    df["grid_id"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

    # Agregasi HANYA untuk grid-year yang ada gempa
    grouped = (
        df.groupby(["grid_id", "grid_lat", "grid_lon", "year"])
          .agg(
              event_count=("mag", "count"),
              mean_mag=("mag", "mean"),
              max_mag=("mag", "max"),
              mean_depth=("depth", "mean"),
          )
          .reset_index()
    )

    print("Jumlah baris grid-year yang ada gempanya:", len(grouped))

    # ==== BAGIAN PENTING: tambahkan grid-year yang TIDAK ada gempa ====

    # daftar semua grid yang pernah ada gempa
    grids = grouped[["grid_id", "grid_lat", "grid_lon"]].drop_duplicates()

    # rentang tahun dari data
    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    years = pd.DataFrame({"year": list(range(year_min, year_max + 1))})

    # produk kartesius grid x tahun
    grids["key"] = 1
    years["key"] = 1
    full = grids.merge(years, on="key").drop(columns="key")

    print("Jumlah kombinasi grid x tahun (full):", len(full))

    # merge dengan grouped (yang hanya punya baris kalau ada gempa)
    full = full.merge(
        grouped,
        on=["grid_id", "grid_lat", "grid_lon", "year"],
        how="left",
    )

    # isi NaN (tidak ada gempa) dengan 0
    full["event_count"] = full["event_count"].fillna(0).astype(int)
    full["mean_mag"] = full["mean_mag"].fillna(0.0)
    full["max_mag"] = full["max_mag"].fillna(0.0)
    full["mean_depth"] = full["mean_depth"].fillna(0.0)

    # flag apakah di tahun itu ada gempa
    full["event_occur"] = (full["event_count"] >= 1).astype(int)

    print("Jumlah baris grid-year SESUDAH dilengkapi:", len(full))

    # simpan
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUT_PATH, index=False)
    print("Dataset grid tahunan (lengkap) disimpan di:", OUT_PATH)


if __name__ == "__main__":
    main()
