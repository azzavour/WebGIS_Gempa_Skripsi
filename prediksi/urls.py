from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("api/prediksi/", views.prediksi_geojson, name="prediksi_geojson"),
]
