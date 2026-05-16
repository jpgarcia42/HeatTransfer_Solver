from django.urls import path
from .views import fluid_props

urlpatterns = [
    path('props/', fluid_props),
]
