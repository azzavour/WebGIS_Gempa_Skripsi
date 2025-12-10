import pandas as pd
from pathlib import Path

def clean_bps_master(
    input_path="data/processed/bps_kecamatan_master.xlsx",
    output_xlsx="data/processed/bps_kecamatan_master_clean.xlsx",
    output_csv="data/processed/bps_kecamatan_master_clean.csv",
):
    # Baca file master
    df = pd.read_excel(input_path)

    # Tentukan nama kolom kecamatan (sesuai script sebelumnya harusnya 'kecamatan')
    if "kecamatan" in df.columns:
        name_col = "kecamatan"
    elif "nama_wilayah" in df.columns:
        name_col = "nama_wilayah"
    else:
        # fallback: pakai kolom pertama
        name_col = df.columns[0]

    # --- 1. Buang baris yang jelas-jelas bukan data kecamatan ---

    # baris-baris yang kita anggap "sampah"
    junk_prefixes = (
        "catatan",
        "1 hasil sensus",
        "2 hasil sensus",
        "hasil sensus penduduk",
        "hasil long form sensus",
        "hasil long form population census",
        "laju pertumbuhan penduduk",
    )

    name_series = df[name_col].astype(str).str.strip()

    mask_not_empty = name_series.notna() & ~name_series.isin(["", "nan", "-", "None"])

    mask_not_junk = ~name_series.str.lower().str.startswith(junk_prefixes)

    # Gabungkan mask
    df = df[mask_not_empty & mask_not_junk].copy()

    # --- 2. (opsional) Wajib ada jumlah penduduk kalau kolomnya tersedia ---

    if "jumlah_penduduk" in df.columns:
        df = df[df["jumlah_penduduk"].notna()]

    # --- 3. (opsional) buang baris yang semua kolom numeriknya kosong ---

    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        df = df.dropna(subset=num_cols, how="all")

    # Reset index
    df = df.reset_index(drop=True)

    print(f"Setelah cleaning: {len(df)} baris")

    # Simpan
    Path(output_xlsx).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_xlsx, index=False)
    df.to_csv(output_csv, index=False)

    print("Disimpan ke:")
    print("  ", output_xlsx)
    print("  ", output_csv)


if __name__ == "__main__":
    clean_bps_master()
