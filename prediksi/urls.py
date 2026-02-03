from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("api/prediksi/", views.prediksi_geojson, name="prediksi_geojson"),
    path("api/prediksi/points/", views.prediksi_points, name="prediksi_points"),
    path("api/predict-monthly/", views.predict_monthly, name="predict_monthly"),
]
