# utils/enhanced_video_processor.py - Enhanced to create videos with bounding boxes

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
    """Enhanced video processor for handling test videos with bounding box detection"""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager or ModelManager()
        self.base_output_dir = os.path.join(settings.MEDIA_ROOT, 'detected_videos')
        self.test_output_dir = os.path.join(settings.MEDIA_ROOT, 'test_detections')
        
        # Create directories if they don't exist
        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Video settings
        self.video_clip_duration = 10  # seconds
        self.fps = 20
    
    def create_test_detection_alert_with_bbox(self, camera, alert_type, confidence, detection_results, source_video_name, frame=None):
        """
        Create alert for test video detection with bounding box video
        
        Args:
            camera: Camera model instance (test camera)
            alert_type: Type of detection
            confidence: Detection confidence
            detection_results: Detection results from model
            source_video_name: Name of the source video file
            frame: Current frame for processing
            
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
                
                # Generate file names
                video_filename = f"test_{alert_type}_{timestamp}_{unique_id}.mp4"
                video_path = os.path.join(test_dir, video_filename)
                
                thumbnail_filename = f"test_{alert_type}_{timestamp}_{unique_id}_thumb.jpg"
                thumbnail_path = os.path.join(test_dir, thumbnail_filename)
                
                # Extract bounding boxes from detection results
                bboxes = self._extract_bounding_boxes(detection_results)
                
                # Create video with bounding boxes from source video
                video_success = self._create_detection_video_from_source(
                    source_video_name, video_path, bboxes, alert_type, confidence
                )
                
                if video_success:
                    # Create thumbnail from detection frame
                    if frame is not None:
                        annotated_frame = self._draw_bounding_boxes(frame.copy(), bboxes, alert_type, confidence)
                        cv2.imwrite(thumbnail_path, annotated_frame)
                    else:
                        # Create thumbnail from first frame of output video
                        self._create_thumbnail_from_video(video_path, thumbnail_path, bboxes, alert_type, confidence)
                    
                    # Determine severity
                    severity = self._determine_test_severity(alert_type, confidence)
                    
                    # Get relative paths for database storage
                    video_relative_path = os.path.relpath(video_path, settings.MEDIA_ROOT)
                    thumbnail_relative_path = os.path.relpath(thumbnail_path, settings.MEDIA_ROOT)
                    
                    # Create alert with video file
                    alert = Alert.objects.create(
                        title=f"TEST {alert_type.replace('_', ' ').title()} Detection - {source_video_name}",
                        description=f"Test detection of {alert_type.replace('_', ' ')} from video file {source_video_name} with {confidence:.2f} confidence. Detection highlighted with bounding boxes.",
                        alert_type=alert_type,
                        severity=severity,
                        confidence=confidence,
                        camera=camera,
                        location=f"Test Video: {source_video_name}",
                        video_file=video_relative_path,
                        thumbnail=thumbnail_relative_path,
                        status='pending_review',
                        notes=f"Detected in test video: {source_video_name}. Unique ID: {unique_id}. Bounding boxes: {len(bboxes)}"
                    )
                    
                    logger.info(f"Created test alert {alert.id} with detection video for {alert_type} in {source_video_name}")
                    return alert
                else:
                    logger.error(f"Failed to create detection video for {alert_type}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error creating test detection alert with video: {str(e)}")
            return None
    
    def _create_detection_video_from_source(self, source_video_path, output_video_path, bboxes, alert_type, confidence):
        """
        Create a detection video with bounding boxes from source video
        
        Args:
            source_video_path: Path to source video
            output_video_path: Path to save output video
            bboxes: List of bounding boxes
            alert_type: Type of detection
            confidence: Detection confidence
            
        Returns:
            bool: Success status
        """
        try:
            # Get the full path to source video
            if not os.path.isabs(source_video_path):
                source_video_path = os.path.join(settings.MEDIA_ROOT, 'testvideo', source_video_path)
            
            if not os.path.exists(source_video_path):
                logger.error(f"Source video not found: {source_video_path}")
                return False
            
            # Open source video
            cap = cv2.VideoCapture(source_video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open source video: {source_video_path}")
                return False
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Calculate frames for the clip (10 seconds max)
            max_frames = min(total_frames, fps * self.video_clip_duration)
            
            logger.info(f"Creating detection video: {max_frames} frames at {fps} FPS")
            
            frame_count = 0
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Draw bounding boxes on frame
                annotated_frame = self._draw_bounding_boxes(frame.copy(), bboxes, alert_type, confidence)
                
                # Write frame to output video
                out.write(annotated_frame)
                frame_count += 1
            
            # Release resources
            cap.release()
            out.release()
            
            logger.info(f"Successfully created detection video: {output_video_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating detection video: {str(e)}")
            return False
    
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
                            'class_name': r.names.get(int(cls), 'detection') if hasattr(r, 'names') else 'detection'
                        })
                        
        except Exception as e:
            logger.error(f"Error extracting bounding boxes: {str(e)}")
            
        return bboxes
    
    def _draw_bounding_boxes(self, frame, bboxes, alert_type, confidence):
        """
        Draw bounding boxes on frame with enhanced styling
        
        Args:
            frame: OpenCV frame
            bboxes: List of bounding box dictionaries
            alert_type: Type of detection for color coding
            confidence: Overall confidence score
            
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
        
        # Draw title at top of frame
        title_text = f"{alert_type.replace('_', ' ').title()} Detection"
        cv2.putText(frame, title_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Draw overall confidence
        conf_text = f"Confidence: {confidence:.2f}"
        cv2.putText(frame, conf_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw bounding boxes
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            bbox_confidence = bbox['confidence']
            class_name = bbox['class_name']
            
            # Draw rectangle with thicker border
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw inner rectangle for better visibility
            cv2.rectangle(frame, (x1+1, y1+1), (x2-1, y2-1), (255, 255, 255), 1)
            
            # Draw label with background
            label = f"{class_name}: {bbox_confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background rectangle
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 15), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _create_thumbnail_from_video(self, video_path, thumbnail_path, bboxes, alert_type, confidence):
        """
        Create thumbnail from first frame of video
        
        Args:
            video_path: Path to video file
            thumbnail_path: Path to save thumbnail
            bboxes: Bounding boxes to draw
            alert_type: Type of detection
            confidence: Detection confidence
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    annotated_frame = self._draw_bounding_boxes(frame, bboxes, alert_type, confidence)
                    cv2.imwrite(thumbnail_path, annotated_frame)
                cap.release()
        except Exception as e:
            logger.error(f"Error creating thumbnail from video: {str(e)}")
    
    def _determine_test_severity(self, alert_type, confidence):
        """
        Determine alert severity for test detections
        
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
    
    def process_detection_with_video_and_bbox(self, camera, alert_type, confidence, detection_results, frame=None):
        """
        Process detection and create video with bounding box highlighting for camera streams
        
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
                    camera, video_path, bboxes, alert_type, confidence, duration=self.video_clip_duration
                )
                
                if recording_success:
                    # Create thumbnail from the current frame if available
                    if frame is not None:
                        annotated_frame = self._draw_bounding_boxes(frame.copy(), bboxes, alert_type, confidence)
                        cv2.imwrite(thumbnail_path, annotated_frame)
                    else:
                        # Create thumbnail from first frame of output video
                        self._create_thumbnail_from_video(video_path, thumbnail_path, bboxes, alert_type, confidence)
                    
                    # Determine severity
                    severity = self._determine_test_severity(alert_type, confidence)
                    
                    # Get relative paths for database storage
                    video_relative_path = os.path.relpath(video_path, settings.MEDIA_ROOT)
                    thumbnail_relative_path = os.path.relpath(thumbnail_path, settings.MEDIA_ROOT)
                    
                    # Create alert
                    alert = Alert.objects.create(
                        title=f"{alert_type.replace('_', ' ').title()} Detection",
                        description=f"Detected {alert_type.replace('_', ' ')} with {confidence:.2f} confidence on camera {camera.name}. Detection highlighted with bounding boxes.",
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
    
    def _record_detection_video_with_bbox(self, camera, output_path, bboxes, alert_type, confidence, duration=10):
        """
        Record video with bounding box overlay from camera stream
        
        Args:
            camera: Camera model instance
            output_path: Path to save the video
            bboxes: Bounding boxes to draw
            alert_type: Type of detection
            confidence: Detection confidence
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
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or self.fps
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Calculate total frames to record
            total_frames = fps * duration
            frames_recorded = 0
            
            logger.info(f"Recording detection video with bounding boxes: {output_path}")
            
            while frames_recorded < total_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame during recording")
                    break
                
                # Draw bounding boxes on frame
                annotated_frame = self._draw_bounding_boxes(frame.copy(), bboxes, alert_type, confidence)
                
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