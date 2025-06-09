# utils/enhanced_video_processor_patch.py - Patch for the original EnhancedVideoProcessor

import os
import json
import uuid
import cv2
import numpy as np
from datetime import datetime
from django.conf import settings
from django.utils import timezone
from django.db import transaction
from alerts.models import Alert

def add_test_detection_support(video_processor_class):
    """
    Add test detection support to the original EnhancedVideoProcessor class
    """
    
    def create_test_detection_alert(self, camera, alert_type, confidence, detection_results, source_video_name):
        """
        Create alert for test video detection
        
        Args:
            camera: Camera model instance (test camera)
            alert_type: Type of detection
            confidence: Detection confidence
            detection_results: Detection results from model
            source_video_name: Name of the source video file
            
        Returns:
            Alert: Created alert instance or None if failed
        """
        try:
            with transaction.atomic():
                # Create output directory for test detections
                test_output_dir = os.path.join(settings.MEDIA_ROOT, 'test_detections')
                test_dir = os.path.join(test_output_dir, alert_type)
                os.makedirs(test_dir, exist_ok=True)
                
                # Create detection metadata
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]
                
                # Save detection frame as image (placeholder for test videos)
                detection_image_name = f"test_{alert_type}_{timestamp}_{unique_id}.jpg"
                detection_image_path = os.path.join(test_dir, detection_image_name)
                
                # Create placeholder frame with detection info
                placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_frame, f"Test Detection: {alert_type.replace('_', ' ').title()}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(placeholder_frame, f"Source: {source_video_name}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(placeholder_frame, f"Confidence: {confidence:.2f}", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(placeholder_frame, f"Time: {timestamp}", 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                cv2.imwrite(detection_image_path, placeholder_frame)
                
                # Create detection metadata file
                metadata = {
                    'detection_id': unique_id,
                    'alert_type': alert_type,
                    'confidence': confidence,
                    'source_video': source_video_name,
                    'detection_time': timestamp,
                    'camera_id': camera.id,
                    'camera_name': camera.name,
                    'user_id': camera.user.id,
                    'user_email': camera.user.email,
                    'severity': self._determine_test_severity(alert_type, confidence),
                    'detection_results': str(detection_results) if detection_results else 'N/A'
                }
                
                metadata_file_name = f"test_{alert_type}_{timestamp}_{unique_id}_metadata.json"
                metadata_file_path = os.path.join(test_dir, metadata_file_name)
                
                with open(metadata_file_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Determine severity
                severity = self._determine_test_severity(alert_type, confidence)
                
                # Get relative paths for database storage
                image_relative_path = os.path.relpath(detection_image_path, settings.MEDIA_ROOT)
                
                # Create alert with pending_review status
                alert = Alert.objects.create(
                    title=f"TEST {alert_type.replace('_', ' ').title()} Detection - {source_video_name}",
                    description=f"Test detection of {alert_type.replace('_', ' ')} from video file {source_video_name} with {confidence:.2f} confidence. Source: Test Video Processing.",
                    alert_type=alert_type,
                    severity=severity,
                    confidence=confidence,
                    camera=camera,
                    location=f"Test Video: {source_video_name}",
                    thumbnail=image_relative_path,
                    status='pending_review',
                    notes=f"Detected in test video: {source_video_name}. Unique ID: {unique_id}"
                )
                
                # Send notification to reviewers about test detection
                self._notify_test_reviewers(alert, source_video_name)
                
                return alert
                
        except Exception as e:
            import logging
            logger = logging.getLogger('security_ai')
            logger.error(f"Error creating test detection alert: {str(e)}")
            return None
    
    def _determine_test_severity(self, alert_type, confidence):
        """Determine alert severity for test detections"""
        # Base severity on confidence
        if confidence >= 0.9:
            base_severity = 'critical'
        elif confidence >= 0.7:
            base_severity = 'high'
        elif confidence >= 0.5:
            base_severity = 'medium'
        else:
            base_severity = 'low'
        
        # Adjust based on detection type
        high_priority_types = ['fire_smoke', 'choking']
        if alert_type in high_priority_types:
            if base_severity == 'medium':
                return 'high'
            elif base_severity == 'low':
                return 'medium'
        
        return base_severity
    
    def _notify_test_reviewers(self, alert, source_video_name):
        """Send notification to reviewers about test detection"""
        try:
            from django.contrib.auth import get_user_model
            from notifications.models import NotificationLog
            
            User = get_user_model()
            
            # Get all active reviewers
            reviewers = User.objects.filter(
                role__in=['reviewer', 'admin'],
                is_active=True,
                status='active'
            )
            
            if not reviewers.exists():
                import logging
                logger = logging.getLogger('security_ai')
                logger.warning("No active reviewers found for test alert notification")
                return
            
            title = f"TEST Alert Pending Review: {alert.get_alert_type_display()}"
            message = f"""
            A new TEST {alert.get_alert_type_display()} alert requires review:
            
            Source: Test Video - {source_video_name}
            Confidence: {alert.confidence:.2f}
            Time: {alert.detection_time.strftime('%Y-%m-%d %H:%M:%S')}
            Severity: {alert.get_severity_display()}
            
            This is a test detection from video file processing.
            Please review and confirm if this detection is accurate.
            """
            
            # Create notification logs for all reviewers
            for reviewer in reviewers:
                NotificationLog.objects.create(
                    user=reviewer,
                    title=title,
                    message=message,
                    notification_type='email',
                    alert=alert,
                    status='pending'
                )
            
            import logging
            logger = logging.getLogger('security_ai')
            logger.info(f"Notified {reviewers.count()} reviewers about test alert {alert.id}")
            
        except Exception as e:
            import logging
            logger = logging.getLogger('security_ai')
            logger.error(f"Error notifying reviewers about test detection: {str(e)}")
    
    # Add methods to the class
    video_processor_class.create_test_detection_alert = create_test_detection_alert
    video_processor_class._determine_test_severity = _determine_test_severity
    video_processor_class._notify_test_reviewers = _notify_test_reviewers
    
    return video_processor_class

# Apply the patch to the original EnhancedVideoProcessor
from utils.enhanced_video_processor import EnhancedVideoProcessor
add_test_detection_support(EnhancedVideoProcessor)