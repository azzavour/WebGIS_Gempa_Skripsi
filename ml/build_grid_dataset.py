import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "bmkg_clean.csv"
OUT_PATH   = BASE_DIR / "data" / "processed" / "grid_yearly_dataset.csv"

# Parameter grid dan batas Indonesia (boleh kamu modifikasi nanti)
GRID_SIZE = 1.0  # derajat (1° x 1°)

LAT_MIN, LAT_MAX = -11, 6   # kira-kira batas lintang Indonesia
LON_MIN, LON_MAX = 95, 142  # kira-kira batas bujur Indonesia


def main():
    print("Baca data dari:", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)

    # Filter area Indonesia (optional, tapi bagus biar rapi)
    df = df[
        (df["lat"].between(LAT_MIN, LAT_MAX)) &
        (df["lon"].between(LON_MIN, LON_MAX))
    ].copy()

    print("Jumlah kejadian setelah filter area:", len(df))

    # Hitung koordinat grid (dibulatkan ke bawah tiap GRID_SIZE)
    df["grid_lat"] = (np.floor(df["lat"] / GRID_SIZE) * GRID_SIZE).round(3)
    df["grid_lon"] = (np.floor(df["lon"] / GRID_SIZE) * GRID_SIZE).round(3)

    # Buat ID unik grid
    df["grid_id"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

    # Group per grid + tahun
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

    # Target biner: ada gempa (≥1) atau tidak
    grouped["event_occur"] = (grouped["event_count"] >= 1).astype(int)

    print("Jumlah baris grid-year:", len(grouped))

    # Simpan ke CSV
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUT_PATH, index=False)
    print("Dataset grid tahunan disimpan di:", OUT_PATH)


if __name__ == "__main__":
    main()
