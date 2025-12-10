import os
import csv

# 👉 SESUAIKAN kalau struktur kamu beda
BASE_DIR = os.path.join("data", "raw", "Data Penduduk BPS")

# Nama pola baru (tanpa nomor & ekstensi)
BASE_NAME = "data_penduduk_kabupate"

# File log biar tahu mapping nama lama -> nama baru
LOG_FILE = "renamed_bps_all.csv"

# Ubah ke False kalau sudah yakin
DRY_RUN = False   # True = cuma print, belum benar-benar rename


def main():
    mappings = []

    for dirpath, dirnames, filenames in os.walk(BASE_DIR):
        # Biar urut rapi
        filenames = sorted(filenames)

        counter = 1  # restart nomor per folder

        for fname in filenames:
            old_path = os.path.join(dirpath, fname)

            # Hanya rename file Excel/CSV
            name, ext = os.path.splitext(fname)
            if ext.lower() not in [".xlsx", ".xls", ".csv"]:
                continue

            new_name = f"{BASE_NAME}_{counter:03d}{ext}"
            new_path = os.path.join(dirpath, new_name)

            # Hindari overwrite kalau kebetulan sudah ada
            while os.path.exists(new_path) and old_path.lower() != new_path.lower():
                counter += 1
                new_name = f"{BASE_NAME}_{counter:03d}{ext}"
                new_path = os.path.join(dirpath, new_name)

            print(f"[{dirpath}]")
            print(f"  OLD: {fname}")
            print(f"  NEW: {new_name}\n")

            if not DRY_RUN:
                os.rename(old_path, new_path)

            mappings.append([old_path, new_path])
            counter += 1

    # Simpan log mapping
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["old_path", "new_path"])
        writer.writerows(mappings)

    print(f"\nSelesai. Mapping nama disimpan di: {LOG_FILE}")
    if DRY_RUN:
        print("DRY_RUN = True → belum ada file yang benar-benar di-rename.")


if __name__ == "__main__":
    main()
