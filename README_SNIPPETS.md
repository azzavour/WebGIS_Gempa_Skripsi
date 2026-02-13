# Cuplikan Kode Prediksi Bulanan (webgis_gempa)

File ini merangkum bagian kode yang diminta agar analisis/debugging bisa dilakukan tanpa membuka seluruh repo.

## A. Fungsi `prediksi_bulanan` (prediksi/views.py)
```python
@require_GET
def prediksi_bulanan(request):
    """
    Disagregasi probabilitas tahunan menjadi bulanan (tanpa retraining).
    Rumus: P_bulan = 1 - (1 - P_tahun) ** (1/12)
    """
    model_key = request.GET.get("model", "rf").lower()
    if model_key not in MODEL_FIELDS:
        model_key = "rf"

    probability_field = MODEL_FIELDS[model_key]
    if not PRED_PATH.exists():
        return JsonResponse({"error": "grid_predictions.csv tidak ditemukan."}, status=404)

    try:
        requested_bbox = _get_requested_bbox(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    df = pd.read_csv(PRED_PATH)
    required_cols = {"grid_id", "grid_lat", "grid_lon", probability_field, "year", "target_year"}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        return JsonResponse({"error": f"Kolom hilang: {', '.join(missing)}"}, status=500)

    df = _filter_predictions_by_bbox(df, requested_bbox)
    if df.empty:
        return JsonResponse({"error": "Tidak ada data dalam batas koordinat diminta."}, status=404)

    grid_lookup = _load_grid_lookup()
    pop_lookup = _get_population_lookup()

    monthly_records: list[dict] = []
    exposure_by_month = [0.0 for _ in range(12)]
    prob_sum_by_month = [0.0 for _ in range(12)]
    prob_count_by_month = [0 for _ in range(12)]
    peak_month_global = None
    peak_prob_global = -1.0
    hist_norm = _get_monthly_hist_norm()

    target_year_val = int(df["target_year"].dropna().max()) if "target_year" in df else None
```

## B. Memuat data BMKG & distribusi bulan (prediksi/views.py)
```python
def _load_bmkg_west_java() -> pd.DataFrame:
    global _BMKG_WEST_CACHE
    if _BMKG_WEST_CACHE is not None:
        return _BMKG_WEST_CACHE
    if not BMKG_CLEAN_PATH.exists():
        _BMKG_WEST_CACHE = pd.DataFrame(columns=["lat", "lon", "date", "month"])
        return _BMKG_WEST_CACHE
    try:
        df = pd.read_csv(BMKG_CLEAN_PATH)
    except Exception:
        _BMKG_WEST_CACHE = pd.DataFrame(columns=["lat", "lon", "date", "month"])
        return _BMKG_WEST_CACHE
    date_col = "date" if "date" in df.columns else "datetime" if "datetime" in df.columns else None
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["date"] = pd.NaT
    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month
    mask = (
        pd.to_numeric(df["lat"], errors="coerce").between(-8.5, -5.5)
        & pd.to_numeric(df["lon"], errors="coerce").between(106.0, 109.0)
    )
    df = df.loc[mask, ["lat", "lon", "month"]].dropna(subset=["month"])
    _BMKG_WEST_CACHE = df
    return _BMKG_WEST_CACHE

def _get_monthly_hist_norm() -> list[float]:
    global _MONTHLY_HIST_CACHE
    if _MONTHLY_HIST_CACHE is not None:
        return _MONTHLY_HIST_CACHE
    df = _load_bmkg_west_java()
    if df.empty:
        _MONTHLY_HIST_CACHE = [1 / 12] * 12
        return _MONTHLY_HIST_CACHE
    counts = df["month"].value_counts().reindex(range(1, 13), fill_value=0).tolist()
    total = sum(counts)
    _MONTHLY_HIST_CACHE = [c / total for c in counts] if total else [1 / 12] * 12
    return _MONTHLY_HIST_CACHE
```

## C. Penentuan `predicted_month` / puncak probabilitas (prediksi/views.py)
Potongan di dalam loop `prediksi_bulanan` yang menghitung bulan puncak per-grid:
```python
        monthly_raw = [n * prob_year for n in monthly_base]
        total_raw = sum(monthly_raw)
        scale = 1.0
        if total_raw > prob_year and total_raw > 0:
            scale = prob_year / total_raw
        monthly_probs = [max(0.0, min(1.0, val * scale)) for val in monthly_raw]
        ...
        for month_idx, month_name in enumerate(MONTH_NAMES, start=1):
            prob_month_weighted = monthly_probs[month_idx - 1]
            ...
            if prob_month_weighted > peak_prob_global:
                peak_prob_global = prob_month_weighted
                peak_month_global = month_name
```

Nilai `peak_month_global` dikirim kembali di respons JSON:
```python
    response_payload = {
        "region": "Jawa Barat",
        "year": target_year_val or 2026,
        "monthly": monthly_probs,
        "peak_month": peak_month_global,
        ...
    }
```

> Semua cuplikan di atas diambil langsung dari `prediksi/views.py` per 9 Februari 2026, tanpa modifikasi.
