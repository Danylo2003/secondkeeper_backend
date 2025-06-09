# utils/enhanced_video_processor.py - Extension for handling test videos

import os
import time
import cv2
import numpy as np
import logging
import subprocess
import uuid
import json
from datetime import datetime
from pathlib import Path
from django.conf import settings
from django.utils import timezone
from django.db import transaction

from alerts.models import Alert
from cameras.models import Camera
from utils.model_manager import ModelManager

logger = logging.getLogger('security_ai')

class EnhancedVideoProcessor:
    """Extension to the original video processor for handling test videos"""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager or ModelManager()
        self.base_output_dir = os.path.join(settings.MEDIA_ROOT, 'detected_videos')
        self.test_output_dir = os.path.join(settings.MEDIA_ROOT, 'test_detections')
        
        # Create directories if they don't exist
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
    
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
                test_dir = os.path.join(self.test_output_dir, alert_type)
                os.makedirs(test_dir, exist_ok=True)
                
                # Create detection metadata
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]
                
                # Save detection frame as image
                detection_image_name = f"test_{alert_type}_{timestamp}_{unique_id}.jpg"
                detection_image_path = os.path.join(test_dir, detection_image_name)
                
                # For test videos, we don't have the actual frame, so we create a placeholder
                # In a real implementation, you would pass the actual detection frame
                placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder_frame, f"Test Detection: {alert_type}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(placeholder_frame, f"Source: {source_video_name}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(placeholder_frame, f"Confidence: {confidence:.2f}", 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
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
                    'severity': self._determine_severity(alert_type, confidence),
                    'detection_results': str(detection_results)
                }
                
                metadata_file_name = f"test_{alert_type}_{timestamp}_{unique_id}_metadata.json"
                metadata_file_path = os.path.join(test_dir, metadata_file_name)
                
                with open(metadata_file_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Determine severity based on confidence and detection type
                severity = self._determine_severity(alert_type, confidence)
                
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
                    notes=f"Detected in test video: {source_video_name}"
                )
                
                logger.info(f"Created test alert {alert.id} for {alert_type} detection in video {source_video_name}")
                
                # Send notification to reviewers about test detection
                self._notify_test_reviewers(alert, source_video_name)
                
                return alert
                
        except Exception as e:
            logger.error(f"Error creating test detection alert: {str(e)}")
            return None
    
    def _determine_severity(self, alert_type, confidence):
        """
        Determine alert severity based on detection type and confidence
        
        Args:
            alert_type: Type of detection
            confidence: Detection confidence
            
        Returns:
            str: Severity level
        """
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
        """
        Send notification to reviewers about test detection
        
        Args:
            alert: Alert instance
            source_video_name: Name of source video file
        """
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
            
            logger.info(f"Notified {reviewers.count()} reviewers about test alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Error notifying reviewers about test detection: {str(e)}")
    
    def get_test_detection_statistics(self):
        """
        Get statistics about test detections
        
        Returns:
            dict: Statistics about test detections
        """
        try:
            stats = {
                'total_test_detections': 0,
                'by_type': {},
                'recent_detections': [],
                'total_size_mb': 0
            }
            
            if not os.path.exists(self.test_output_dir):
                return stats
            
            # Walk through test detection directory
            for detection_type in os.listdir(self.test_output_dir):
                type_dir = os.path.join(self.test_output_dir, detection_type)
                if not os.path.isdir(type_dir):
                    continue
                
                type_count = 0
                for file in os.listdir(type_dir):
                    file_path = os.path.join(type_dir, file)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        stats['total_size_mb'] += file_size / (1024 * 1024)
                        
                        if file.endswith('.json'):
                            type_count += 1
                            stats['total_test_detections'] += 1
                            
                            # Add to recent detections
                            try:
                                with open(file_path, 'r') as f:
                                    metadata = json.load(f)
                                stats['recent_detections'].append(metadata)
                            except Exception as e:
                                logger.error(f"Error reading metadata file {file_path}: {str(e)}")
                
                if type_count > 0:
                    stats['by_type'][detection_type] = type_count
            
            # Sort recent detections by time (newest first)
            stats['recent_detections'].sort(
                key=lambda x: x.get('detection_time', ''), 
                reverse=True
            )
            
            # Limit to last 20 detections
            stats['recent_detections'] = stats['recent_detections'][:20]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting test detection statistics: {str(e)}")
            return stats
    
    def cleanup_old_test_detections(self, days_old=7):
        """
        Clean up old test detection files
        
        Args:
            days_old: Remove files older than this many days
        """
        try:
            if not os.path.exists(self.test_output_dir):
                return
            
            cutoff_time = time.time() - (days_old * 24 * 60 * 60)
            removed_count = 0
            
            for root, dirs, files in os.walk(self.test_output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if os.path.getmtime(file_path) < cutoff_time:
                            os.remove(file_path)
                            removed_count += 1
                            logger.debug(f"Removed old test detection file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {str(e)}")
            
            logger.info(f"Cleaned up {removed_count} old test detection files")
            
        except Exception as e:
            logger.error(f"Error cleaning up old test detections: {str(e)}")


# Add this method to the original EnhancedVideoProcessor class
def create_test_detection_alert(self, camera, alert_type, confidence, detection_results, source_video_name):
    """
    Create alert for test video detection - to be added to EnhancedVideoProcessor
    """
    extension = EnhancedVideoProcessorExtension(self.model_manager)
    return extension.create_test_detection_alert(
        camera, alert_type, confidence, detection_results, source_video_name
    )