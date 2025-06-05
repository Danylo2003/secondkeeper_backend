# cameras/detection_views.py - Views for controlling the detection service

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from utils.permissions import IsAdminUser
from utils.camera_detection_manager import detection_manager
from cameras.models import Camera
from alerts.models import Alert
from django.db.models import Count, Q
from django.utils import timezone
from datetime import timedelta
import logging

logger = logging.getLogger('security_ai')

@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def detection_service_status(request):
    """Get the status of the detection service"""
    try:
        # Get service status
        is_running = detection_manager.is_running
        active_cameras_count = len(detection_manager.active_cameras)
        
        # Get camera statistics
        total_cameras = Camera.objects.count()
        online_cameras = Camera.objects.filter(status='online').count()
        detection_enabled_cameras = Camera.objects.filter(
            status='online', 
            detection_enabled=True
        ).count()
        
        # Get recent alerts (last 24 hours)
        yesterday = timezone.now() - timedelta(days=1)
        recent_alerts = Alert.objects.filter(
            detection_time__gte=yesterday
        ).count()
        
        # Get alerts by type (last 24 hours)
        alerts_by_type = Alert.objects.filter(
            detection_time__gte=yesterday
        ).values('alert_type').annotate(count=Count('id'))
        
        return Response({
            'success': True,
            'data': {
                'service_running': is_running,
                'active_processors': active_cameras_count,
                'cameras': {
                    'total': total_cameras,
                    'online': online_cameras,
                    'detection_enabled': detection_enabled_cameras
                },
                'alerts_24h': recent_alerts,
                'alerts_by_type': {
                    item['alert_type']: item['count'] 
                    for item in alerts_by_type
                },
                'last_update': timezone.now().isoformat()
            },
            'message': 'Detection service status retrieved successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error getting detection service status: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error retrieving detection service status.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def start_detection_service(request):
    """Start the detection service"""
    try:
        if detection_manager.is_running:
            return Response({
                'success': True,
                'data': {'status': 'already_running'},
                'message': 'Detection service is already running.',
                'errors': []
            })
        
        detection_manager.start()
        
        return Response({
            'success': True,
            'data': {'status': 'started'},
            'message': 'Detection service started successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error starting detection service: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error starting detection service.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def stop_detection_service(request):
    """Stop the detection service"""
    try:
        if not detection_manager.is_running:
            return Response({
                'success': True,
                'data': {'status': 'already_stopped'},
                'message': 'Detection service is already stopped.',
                'errors': []
            })
        
        detection_manager.stop()
        
        return Response({
            'success': True,
            'data': {'status': 'stopped'},
            'message': 'Detection service stopped successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error stopping detection service: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error stopping detection service.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated, IsAdminUser])
def restart_detection_service(request):
    """Restart the detection service"""
    try:
        # Stop if running
        if detection_manager.is_running:
            detection_manager.stop()
            
        # Start again
        detection_manager.start()
        
        return Response({
            'success': True,
            'data': {'status': 'restarted'},
            'message': 'Detection service restarted successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error restarting detection service: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error restarting detection service.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def active_camera_processors(request):
    """Get list of active camera processors"""
    try:
        active_processors = []
        
        for camera_id, processor in detection_manager.active_cameras.items():
            active_processors.append({
                'camera_id': camera_id,
                'camera_name': processor.camera.name,
                'is_running': processor.is_running,
                'frame_count': processor.frame_count,
                'confidence_threshold': processor.confidence_threshold,
                'detectors_enabled': {
                    'fire_smoke': processor.camera.fire_smoke_detection,
                    'fall': processor.camera.fall_detection,
                    'violence': processor.camera.violence_detection,
                    'choking': processor.camera.choking_detection
                }
            })
        
        return Response({
            'success': True,
            'data': {
                'active_processors': active_processors,
                'total_count': len(active_processors)
            },
            'message': 'Active camera processors retrieved successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error getting active camera processors: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error retrieving active camera processors.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated, IsAdminUser])
def detection_statistics(request):
    """Get detection statistics"""
    try:
        # Get time range from query params
        days = int(request.GET.get('days', 7))
        start_date = timezone.now() - timedelta(days=days)
        
        # Total alerts in time range
        total_alerts = Alert.objects.filter(
            detection_time__gte=start_date
        ).count()
        
        # Alerts by severity
        alerts_by_severity = Alert.objects.filter(
            detection_time__gte=start_date
        ).values('severity').annotate(count=Count('id'))
        
        # Alerts by camera
        alerts_by_camera = Alert.objects.filter(
            detection_time__gte=start_date
        ).select_related('camera').values(
            'camera__id', 'camera__name'
        ).annotate(count=Count('id')).order_by('-count')[:10]
        
        # Alerts by status
        alerts_by_status = Alert.objects.filter(
            detection_time__gte=start_date
        ).values('status').annotate(count=Count('id'))
        
        # Daily alert counts
        daily_alerts = []
        for i in range(days):
            day_start = timezone.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=i)
            day_end = day_start + timedelta(days=1)
            
            day_count = Alert.objects.filter(
                detection_time__gte=day_start,
                detection_time__lt=day_end
            ).count()
            
            daily_alerts.append({
                'date': day_start.date().isoformat(),
                'count': day_count
            })
        
        daily_alerts.reverse()  # Oldest first
        
        return Response({
            'success': True,
            'data': {
                'time_range_days': days,
                'total_alerts': total_alerts,
                'alerts_by_severity': {
                    item['severity']: item['count']
                    for item in alerts_by_severity
                },
                'alerts_by_camera': [
                    {
                        'camera_id': item['camera__id'],
                        'camera_name': item['camera__name'],
                        'count': item['count']
                    }
                    for item in alerts_by_camera
                ],
                'alerts_by_status': {
                    item['status']: item['count']
                    for item in alerts_by_status
                },
                'daily_alerts': daily_alerts
            },
            'message': 'Detection statistics retrieved successfully.',
            'errors': []
        })
        
    except Exception as e:
        logger.error(f"Error getting detection statistics: {str(e)}")
        return Response({
            'success': False,
            'data': {},
            'message': 'Error retrieving detection statistics.',
            'errors': [str(e)]
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)