from django.shortcuts import render
from django.http import JsonResponse


def home(request):
    # render template yang berisi Leaflet
    return render(request, 'prediksi/home.html')


def prediksi_geojson(request):
    """
    Untuk sementara: kirim GeoJSON dummy (2 grid saja).
    Nanti diganti dengan hasil perhitungan model Random Forest, SVM, Poisson.
    """
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "grid_id": "GRID-001",
                    "rf_prob": 0.75,
                    "svm_prob": 0.62,
                    "poisson_prob": 0.40,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [106.0, -6.5],
                        [107.0, -6.5],
                        [107.0, -5.5],
                        [106.0, -5.5],
                        [106.0, -6.5],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": {
                    "grid_id": "GRID-002",
                    "rf_prob": 0.30,
                    "svm_prob": 0.45,
                    "poisson_prob": 0.20,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [110.0, -7.5],
                        [111.0, -7.5],
                        [111.0, -6.5],
                        [110.0, -6.5],
                        [110.0, -7.5],
                    ]],
                },
            },
        ],
    }

    return JsonResponse(data)
