from django.contrib import admin
from django.urls import path
from prediksi.views import home, prediksi_geojson

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('api/prediksi/', prediksi_geojson, name='prediksi_geojson'),
      path("", include("prediksi.urls")),
]
