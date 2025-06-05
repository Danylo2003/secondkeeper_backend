from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import CameraViewSet
from .detection_views import (
    detection_service_status,
    start_detection_service,
    stop_detection_service,
    restart_detection_service,
    active_camera_processors,
    detection_statistics
)

router = DefaultRouter()
router.register(r'', CameraViewSet, basename='camera')

urlpatterns = [
    path('', include(router.urls)),
    
    # Detection service control endpoints
    path('detection/status/', detection_service_status, name='detection-status'),
    path('detection/start/', start_detection_service, name='detection-start'),
    path('detection/stop/', stop_detection_service, name='detection-stop'),
    path('detection/restart/', restart_detection_service, name='detection-restart'),
    path('detection/processors/', active_camera_processors, name='detection-processors'),
    path('detection/statistics/', detection_statistics, name='detection-statistics'),
]