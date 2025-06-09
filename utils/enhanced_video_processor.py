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
            return 
            
    def create_test_detection_alert_with_bbox(self, camera, alert_type, confidence, detection_results, source_video_name, frame=None):
        """
        Create alert for test video detection with bounding box
        
        Args:
            camera: Camera model instance (test camera)
            alert_type: Type of detection
            confidence: Detection confidence
            detection_results: Detection results from model
            source_video_name: Name of the source video file
            frame: Current frame for thumbnail
            
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
                
                # Extract bounding boxes
                bboxes = self._extract_bounding_boxes(detection_results)
                
                # Save detection frame with bounding boxes
                detection_image_name = f"test_{alert_type}_{timestamp}_{unique_id}.jpg"
                detection_image_path = os.path.join(test_dir, detection_image_name)
                
                if frame is not None:
                    annotated_frame = self._draw_bounding_boxes(frame.copy(), bboxes, alert_type)
                    cv2.imwrite(detection_image_path, annotated_frame)
                else:
                    # Create placeholder frame with detection info
                    placeholder_frame = self._create_placeholder_thumbnail(alert_type, confidence)
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
                    'bounding_boxes': bboxes,
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
                    description=f"Test detection of {alert_type.replace('_', ' ')} from video file {source_video_name} with {confidence:.2f} confidence. Bounding boxes detected and highlighted.",
                    alert_type=alert_type,
                    severity=severity,
                    confidence=confidence,
                    camera=camera,
                    location=f"Test Video: {source_video_name}",
                    thumbnail=image_relative_path,
                    status='pending_review',
                    notes=f"Detected in test video: {source_video_name}. Unique ID: {unique_id}. Bounding boxes: {len(bboxes)}"
                )
                
                # Send notification to reviewers about test detection
                self._notify_test_reviewers(alert, source_video_name)
                
                return alert
                
        except Exception as e:
            logger.error(f"Error creating test detection alert with bounding box: {str(e)}")
            return None
    
    def process_detection_with_video_and_bbox(self, camera, alert_type, confidence, detection_results, frame=None):
        """
        Process detection and create video with bounding box highlighting
        
        Args:
            camera: Camera model instance
            alert_type: Type of detection
            confidence: Detection confidence
            detection_results: Detection results from model
            frame: Current frame (optional)
            
        Returns:
            Alert: Created alert instance or None if failed
        """
        try:
            with transaction.atomic():
                # Create unique identifier
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_id = uuid.uuid4().hex[:8]
                
                # Create output directories
                video_dir = os.path.join(self.base_output_dir, alert_type)
                os.makedirs(video_dir, exist_ok=True)
                
                # Generate file names
                video_filename = f"{alert_type}_{timestamp}_{unique_id}.mp4"
                video_path = os.path.join(video_dir, video_filename)
                
                thumbnail_filename = f"{alert_type}_{timestamp}_{unique_id}_thumb.jpg"
                thumbnail_path = os.path.join(video_dir, thumbnail_filename)
                
                # Extract bounding boxes from detection results
                bboxes = self._extract_bounding_boxes(detection_results)
                
                # Start video recording with bounding box overlay
                recording_success = self._record_detection_video_with_bbox(
                    camera, video_path, bboxes, alert_type, duration=self.video_clip_duration
                )
                
                if recording_success:
                    # Create thumbnail from the current frame if available
                    if frame is not None:
                        annotated_frame = self._draw_bounding_boxes(frame.copy(), bboxes, alert_type)
                        cv2.imwrite(thumbnail_path, annotated_frame)
                    else:
                        # Create a placeholder thumbnail
                        placeholder = self._create_placeholder_thumbnail(alert_type, confidence)
                        cv2.imwrite(thumbnail_path, placeholder)
                    
                    # Determine severity
                    severity = self._determine_severity(alert_type, confidence)
                    
                    # Get relative paths for database storage
                    video_relative_path = os.path.relpath(video_path, settings.MEDIA_ROOT)
                    thumbnail_relative_path = os.path.relpath(thumbnail_path, settings.MEDIA_ROOT)
                    
                    # Create alert
                    alert = Alert.objects.create(
                        title=f"{alert_type.replace('_', ' ').title()} Detection",
                        description=f"Detected {alert_type.replace('_', ' ')} with {confidence:.2f} confidence on camera {camera.name}",
                        alert_type=alert_type,
                        severity=severity,
                        confidence=confidence,
                        camera=camera,
                        location=camera.name,
                        video_file=video_relative_path,
                        thumbnail=thumbnail_relative_path,
                        status='pending_review'
                    )
                    
                    logger.info(f"Created alert {alert.id} with bounding box video for {alert_type} detection")
                    return alert
                else:
                    logger.error(f"Failed to record video for {alert_type} detection")
                    return None
                    
        except Exception as e:
            logger.error(f"Error creating detection alert with video: {str(e)}")
            return None
    
    def _extract_bounding_boxes(self, detection_results):
        """
        Extract bounding boxes from YOLO detection results
        
        Args:
            detection_results: YOLO detection results
            
        Returns:
            list: List of bounding box dictionaries
        """
        bboxes = []
        
        try:
            for r in detection_results:
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
                    confidences = r.boxes.conf.cpu().numpy()
                    
                    if hasattr(r.boxes, 'cls'):
                        classes = r.boxes.cls.cpu().numpy()
                    else:
                        classes = [0] * len(boxes)  # Default class if not available
                    
                    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                        x1, y1, x2, y2 = box.astype(int)
                        bboxes.append({
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'confidence': float(conf),
                            'class': int(cls),
                            'class_name': r.names.get(int(cls), 'unknown') if hasattr(r, 'names') else 'unknown'
                        })
                        
        except Exception as e:
            logger.error(f"Error extracting bounding boxes: {str(e)}")
            
        return bboxes
    
    def _draw_bounding_boxes(self, frame, bboxes, alert_type):
        """
        Draw bounding boxes on frame
        
        Args:
            frame: OpenCV frame
            bboxes: List of bounding box dictionaries
            alert_type: Type of detection for color coding
            
        Returns:
            frame: Annotated frame
        """
        # Color mapping for different detection types
        color_map = {
            'fire_smoke': (0, 0, 255),    # Red
            'fall': (0, 255, 255),        # Yellow
            'violence': (0, 165, 255),    # Orange
            'choking': (255, 0, 0),       # Blue
            'person': (0, 255, 0)         # Green
        }
        
        color = color_map.get(alert_type, (255, 255, 255))  # Default white
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            confidence = bbox['confidence']
            class_name = bbox['class_name']
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def _record_detection_video_with_bbox(self, camera, output_path, bboxes, alert_type, duration=10):
        """
        Record video with bounding box overlay
        
        Args:
            camera: Camera model instance
            output_path: Path to save the video
            bboxes: Bounding boxes to draw
            alert_type: Type of detection
            duration: Duration in seconds
            
        Returns:
            bool: Success status
        """
        try:
            # Open camera stream
            cap = cv2.VideoCapture(camera.stream_url)
            if not cap.isOpened():
                logger.error(f"Failed to open camera stream: {camera.stream_url}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Calculate total frames to record
            total_frames = fps * duration
            frames_recorded = 0
            
            logger.info(f"Recording video with bounding boxes: {output_path}")
            
            while frames_recorded < total_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame during recording")
                    break
                
                # Draw bounding boxes on frame
                annotated_frame = self._draw_bounding_boxes(frame.copy(), bboxes, alert_type)
                
                # Write frame to video
                out.write(annotated_frame)
                frames_recorded += 1
            
            # Release resources
            cap.release()
            out.release()
            
            logger.info(f"Successfully recorded {frames_recorded} frames to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording video with bounding boxes: {str(e)}")
            return False
    
    def _create_placeholder_thumbnail(self, alert_type, confidence):
        """
        Create a placeholder thumbnail when no frame is available
        
        Args:
            alert_type: Type of detection
            confidence: Detection confidence
            
        Returns:
            numpy.ndarray: Placeholder image
        """
        # Create a blank image
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add text
        title = f"{alert_type.replace('_', ' ').title()} Detection"
        conf_text = f"Confidence: {confidence:.2f}"
        
        cv2.putText(placeholder, title, (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(placeholder, conf_text, (50, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return placeholder
    
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
    extension = EnhancedVideoProcessor(self.model_manager)
    return extension.create_test_detection_alert(
        camera, alert_type, confidence, detection_results, source_video_name
    )