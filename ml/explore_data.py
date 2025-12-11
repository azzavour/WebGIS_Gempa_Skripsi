import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Konfigurasi path
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")

bmkg_path = os.path.join(DATA_PROCESSED, "bmkg_clean.csv")
training_path = os.path.join(DATA_PROCESSED, "training_dataset.csv")

output_dir = os.path.join(DATA_PROCESSED, "bab4_outputs")
os.makedirs(output_dir, exist_ok=True)

# =========================
# 1. Baca dataset
# =========================
print(">> Membaca bmkg_clean.csv ...")
bmkg = pd.read_csv(bmkg_path)

print(">> Membaca training_dataset.csv ...")
training = pd.read_csv(training_path)

print("\n=== INFO bmkg_clean ===")
print(bmkg.info())
print("\n=== INFO training_dataset ===")
print(training.info())


if "datetime" in bmkg.columns:
    # format di bmkg_clean kamu 8/1/2015 18:16, jadi kita coba parse dengan dayfirst=True
    bmkg["datetime"] = pd.to_datetime(bmkg["datetime"], dayfirst=True, errors="coerce")
    if "year" not in bmkg.columns:
        bmkg["year"] = bmkg["datetime"].dt.year

# =========================
# 2. Statistik deskriptif dasar
# =========================
desc_bmkg = bmkg[["mag", "depth"]].describe()
desc_training = training.describe()

desc_bmkg.to_csv(os.path.join(output_dir, "deskripsi_bmkg_clean.csv"))
desc_training.to_csv(os.path.join(output_dir, "deskripsi_training_dataset.csv"))

print("\n>> Deskripsi statistik disimpan ke:")
print("   - deskripsi_bmkg_clean.csv")
print("   - deskripsi_training_dataset.csv")

# =========================
# 3. Histogram magnitudo
# =========================
plt.figure()
bmkg["mag"].hist(bins=20)
plt.xlabel("Magnitudo")
plt.ylabel("Frekuensi")
plt.title("Distribusi Magnitudo Gempa (M ≥ 4.5)")
plt.tight_layout()
hist_path = os.path.join(output_dir, "hist_magnitudo.png")
plt.savefig(hist_path, dpi=300)
plt.close()
print(f">> Histogram magnitudo disimpan ke: {hist_path}")

# =========================
# 4. Tren jumlah gempa per tahun
# =========================
if "year" in bmkg.columns:
    gempa_per_tahun = bmkg.groupby("year").size().reset_index(name="jumlah_gempa")
    line_path_csv = os.path.join(output_dir, "gempa_per_tahun.csv")
    gempa_per_tahun.to_csv(line_path_csv, index=False)

    plt.figure()
    plt.plot(gempa_per_tahun["year"], gempa_per_tahun["jumlah_gempa"], marker="o")
    plt.xlabel("Tahun")
    plt.ylabel("Jumlah Gempa (M ≥ 4.5)")
    plt.title("Tren Jumlah Gempa per Tahun")
    plt.grid(True)
    plt.tight_layout()
    line_path_png = os.path.join(output_dir, "gempa_per_tahun.png")
    plt.savefig(line_path_png, dpi=300)
    plt.close()

    print(f">> Tren gempa per tahun disimpan ke:")
    print(f"   - {line_path_csv}")
    print(f"   - {line_path_png}")
else:
    print("!! Kolom 'year' tidak ditemukan di bmkg_clean.csv")

# =========================
# 5. Cek beberapa kolom penting di training_dataset
# =========================
kolom_penting = [c for c in training.columns if c.lower() in [
    "freq_1w", "freq_2w", "freq_3w", "freq_4w",
    "max_mag_prev", "mean_depth_prev",
    "grid_lat", "grid_lon",
    "year", "week", "target"
]]
print("\n=== Contoh 10 baris pertama training_dataset (kolom penting) ===")
print(training[kolom_penting].head(10))

sample_training_path = os.path.join(output_dir, "sample_training_dataset.csv")
training[kolom_penting].head(50).to_csv(sample_training_path, index=False)
print(f">> Sample 50 baris training_dataset disimpan ke: {sample_training_path}")

print("\nSelesai STEP 1: eksplorasi dasar untuk BAB 4.")
