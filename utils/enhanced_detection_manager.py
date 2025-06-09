# utils/enhanced_detection_manager.py - Enhanced detection manager with video file priority

import cv2
import numpy as np
import threading
import time
import os
import uuid
import logging
import queue
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from django.conf import settings
from django.utils import timezone
from django.db import transaction
import torch

from cameras.models import Camera
from alerts.models import Alert
from detectors import FireSmokeDetector, FallDetector, ViolenceDetector, ChokingDetector
from utils.model_manager import ModelManager
from utils.enhanced_video_processor import EnhancedVideoProcessor

logger = logging.getLogger('security_ai')

class EnhancedDetectionManager:
    """
    Enhanced detection manager that prioritizes video files over camera streams
    """
    
    def __init__(self):
        self.active_cameras = {}  # camera_id -> CameraProcessor
        self.active_video_processors = {}  # video_file -> VideoFileProcessor
        self.model_manager = ModelManager()
        self.video_processor = EnhancedVideoProcessor(self.model_manager)
        self.is_running = False
        self.main_thread = None
        self.stop_event = threading.Event()
        
        # Test video directory
        self.test_video_dir = os.path.join(settings.MEDIA_ROOT, 'testvideo')
        os.makedirs(self.test_video_dir, exist_ok=True)
        
        # Load all detectors
        self.detectors = {
            'fire_smoke': FireSmokeDetector(),
            'fall': FallDetector(),
            'violence': ViolenceDetector(),
            'choking': ChokingDetector()
        }
        
        # GPU optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Detection settings
        self.frame_skip = 5  # Process every 5th frame for performance
        self.detection_cooldown = 30  # Seconds between detections for same source
        self.video_clip_duration = 10  # Seconds for video clips
        
        # Alert tracking to prevent spam
        self.last_alerts = {}  # source_id -> {alert_type: timestamp}
        
    def start(self):
        """Start the enhanced detection manager"""
        if self.is_running:
            logger.warning("Enhanced detection manager is already running")
            return
            
        logger.info("Starting Enhanced Detection Manager with Video File Priority")
        self.is_running = True
        self.stop_event.clear()
        
        # Start main processing thread
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        
    def stop(self):
        """Stop the enhanced detection manager"""
        if not self.is_running:
            return
            
        logger.info("Stopping Enhanced Detection Manager")
        self.is_running = False
        self.stop_event.set()
        
        # Stop all processors
        for processor in self.active_cameras.values():
            processor.stop()
            
        for processor in self.active_video_processors.values():
            processor.stop()
            
        # Wait for main thread to finish
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=10)
            
        self.active_cameras.clear()
        self.active_video_processors.clear()
        
    def _main_loop(self):
        """Main processing loop with video file priority"""
        while self.is_running and not self.stop_event.is_set():
            try:
                # Step 1: Check for video files in test directory
                video_files = self._get_test_video_files()
                
                if video_files:
                    logger.info(f"Found {len(video_files)} test video files, prioritizing video processing")
                    self._process_video_files(video_files)
                    # Stop camera processors when processing video files
                    self._stop_all_camera_processors()
                else:
                    logger.info("No test video files found, switching to camera detection")
                    # Stop video processors
                    self._stop_all_video_processors()
                    # Process camera streams
                    self._process_camera_streams()
                
                # Clean up old alerts from tracking
                self._cleanup_old_alerts()
                
            except Exception as e:
                logger.error(f"Error in main enhanced detection loop: {str(e)}")
            
            # Sleep for a bit before next iteration
            time.sleep(5)
            
    def _get_test_video_files(self):
        """Get list of video files in test directory"""
        try:
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
            video_files = []
            
            for extension in video_extensions:
                pattern = os.path.join(self.test_video_dir, extension)
                video_files.extend(glob.glob(pattern))
            
            # Sort by modification time (newest first)
            video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            return video_files
            
        except Exception as e:
            logger.error(f"Error getting test video files: {str(e)}")
            return []
            
    def _process_video_files(self, video_files):
        """Process video files for detection"""
        current_processors = set(self.active_video_processors.keys())
        new_video_files = set(video_files)
        
        # Stop processors for videos that no longer exist
        videos_to_remove = current_processors - new_video_files
        for video_file in videos_to_remove:
            self._stop_video_processor(video_file)
        
        # Start processors for new video files
        videos_to_add = new_video_files - current_processors
        for video_file in videos_to_add:
            if os.path.exists(video_file):
                self._start_video_processor(video_file)
                
    def _process_camera_streams(self):
        """Process camera streams when no video files are present"""
        # Get online cameras from database
        online_cameras = Camera.objects.filter(
            status='online',
            detection_enabled=True
        ).select_related('user')
        
        current_camera_ids = set(self.active_cameras.keys())
        new_camera_ids = set(str(cam.id) for cam in online_cameras)
        
        # Stop processors for cameras that are no longer online
        cameras_to_remove = current_camera_ids - new_camera_ids
        for camera_id in cameras_to_remove:
            self._stop_camera_processor(camera_id)
        
        # Start processors for new cameras
        cameras_to_add = new_camera_ids - current_camera_ids
        for camera in online_cameras:
            if str(camera.id) in cameras_to_add:
                self._start_camera_processor(camera)
        
        # Update existing processors with new camera data
        for camera in online_cameras:
            camera_id = str(camera.id)
            if camera_id in self.active_cameras:
                self.active_cameras[camera_id].update_camera_data(camera)
                
    def _start_video_processor(self, video_file):
        """Start a processor for a video file"""
        try:
            logger.info(f"Starting video processor for file: {video_file}")
            
            processor = VideoFileProcessor(
                video_file, self.detectors, self.device, self, self.video_processor)
            processor.start()
            
            self.active_video_processors[video_file] = processor
            
        except Exception as e:
            logger.error(f"Error starting video processor for {video_file}: {str(e)}")
            
    def _stop_video_processor(self, video_file):
        """Stop a processor for a video file"""
        try:
            if video_file in self.active_video_processors:
                logger.info(f"Stopping video processor for file: {video_file}")
                self.active_video_processors[video_file].stop()
                del self.active_video_processors[video_file]
                
        except Exception as e:
            logger.error(f"Error stopping video processor for {video_file}: {str(e)}")
            
    def _start_camera_processor(self, camera):
        """Start a processor for a single camera"""
        try:
            camera_id = str(camera.id)
            logger.info(f"Starting processor for camera {camera_id} - {camera.name}")
            
            processor = EnhancedCameraProcessor(
                camera, self.detectors, self.device, self, self.video_processor)
            processor.start()
            
            self.active_cameras[camera_id] = processor
            
        except Exception as e:
            logger.error(f"Error starting processor for camera {camera.id}: {str(e)}")
            
    def _stop_camera_processor(self, camera_id):
        """Stop a processor for a single camera"""
        try:
            if camera_id in self.active_cameras:
                logger.info(f"Stopping processor for camera {camera_id}")
                self.active_cameras[camera_id].stop()
                del self.active_cameras[camera_id]
                
        except Exception as e:
            logger.error(f"Error stopping processor for camera {camera_id}: {str(e)}")
            
    def _stop_all_camera_processors(self):
        """Stop all camera processors"""
        for camera_id in list(self.active_cameras.keys()):
            self._stop_camera_processor(camera_id)
            
    def _stop_all_video_processors(self):
        """Stop all video processors"""
        for video_file in list(self.active_video_processors.keys()):
            self._stop_video_processor(video_file)
            
    def _cleanup_old_alerts(self):
        """Clean up old alert timestamps"""
        current_time = time.time()
        cutoff_time = current_time - self.detection_cooldown
        
        for source_id in list(self.last_alerts.keys()):
            source_alerts = self.last_alerts[source_id]
            for alert_type in list(source_alerts.keys()):
                if source_alerts[alert_type] < cutoff_time:
                    del source_alerts[alert_type]
            
            # Remove source entry if no alerts
            if not source_alerts:
                del self.last_alerts[source_id]
                
    def should_create_alert(self, source_id, alert_type):
        """Check if we should create a new alert (to prevent spam)"""
        current_time = time.time()
        
        if source_id not in self.last_alerts:
            self.last_alerts[source_id] = {}
            
        last_alert_time = self.last_alerts[source_id].get(alert_type, 0)
        
        if current_time - last_alert_time >= self.detection_cooldown:
            self.last_alerts[source_id][alert_type] = current_time
            return True
            
        return False
        
    def create_test_alert(self, source_name, alert_type, confidence, frame, detection_results):
        """Create alert for test video detection"""
        try:
            # Create a mock camera for test videos
            from django.contrib.auth import get_user_model
            User = get_user_model()
            
            # Get or create admin user for test videos
            admin_user = User.objects.filter(role='admin', is_active=True).first()
            if not admin_user:
                logger.warning("No admin user found for test video alerts")
                return None
            
            # Create or get test camera
            test_camera, created = Camera.objects.get_or_create(
                name=f"Test Video Camera - {source_name}",
                user=admin_user,
                defaults={
                    'stream_url': f"file://{source_name}",
                    'status': 'online',
                    'detection_enabled': True,
                    'fire_smoke_detection': True,
                    'fall_detection': True,
                    'violence_detection': True,
                    'choking_detection': True
                }
            )
            
            # Create pending alert
            alert = self.video_processor.create_test_detection_alert(
                test_camera, alert_type, confidence, detection_results, source_name
            )
            
            if alert:
                logger.info(f"Created test alert {alert.id} for {alert_type} detection in {source_name}")
                return alert
            else:
                logger.error(f"Failed to create test alert for {alert_type} detection")
                return None
                
        except Exception as e:
            logger.error(f"Error creating test alert: {str(e)}")
            return None


class VideoFileProcessor:
    """
    Processor for video files with detection capabilities
    """
    
    def __init__(self, video_file, detectors, device, manager, video_processor):
        self.video_file = video_file
        self.video_name = os.path.basename(video_file)
        self.detectors = detectors
        self.device = device
        self.manager = manager
        self.video_processor = video_processor
        
        self.is_running = False
        self.thread = None
        self.cap = None
        self.frame_count = 0
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45
        self.image_size = 640
        
    def start(self):
        """Start processing this video file"""
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop processing this video file"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            
    def _process_loop(self):
        """Main processing loop for this video file"""
        try:
            # Open video file
            self.cap = cv2.VideoCapture(self.video_file)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file: {self.video_file}")
                return
                
            logger.info(f"Started processing video file: {self.video_name}")
            
            # Get video properties
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video properties - Total frames: {total_frames}, FPS: {fps}, Duration: {duration:.2f}s")
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info(f"Finished processing video file: {self.video_name}")
                    break
                    
                self.frame_count += 1
                
                # Skip frames for performance
                if self.frame_count % self.manager.frame_skip != 0:
                    continue
                    
                # Process frame with all detectors
                self._process_frame(frame)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in video file {self.video_name} processing loop: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
                
    def _process_frame(self, frame):
        """Process a single frame with all detectors"""
        try:
            # Check each detector
            detectors_to_run = ['fire_smoke', 'fall', 'violence', 'choking']
            
            # Run object detection models
            for detector_type in detectors_to_run:
                try:
                    detector = self.detectors[detector_type]
                    
                    # Get detector-specific configuration
                    config = self.manager.model_manager.get_detector_config(detector_type)
                    conf_threshold = config['conf_threshold']
                    iou_threshold = config['iou_threshold']
                    image_size = config['image_size']
                    print("detector_type:", detector_type, " conf_threshold: ", conf_threshold, " iou_threshold: ", iou_threshold)
                    
                    # Run detection
                    annotated_frame, results = detector.predict_video_frame(
                        frame, 
                        conf_threshold,
                        iou_threshold,
                        image_size
                    )
                    
                    # Check for detections
                    has_detection, max_confidence = self._check_detection_results(results, conf_threshold)
                    
                    if has_detection and max_confidence >= conf_threshold:
                        # Check if we should create an alert
                        source_id = f"video_{self.video_name}"
                        if self.manager.should_create_alert(source_id, detector_type):
                            # Create test alert
                            self.manager.create_test_alert(
                                self.video_name,
                                detector_type,
                                max_confidence,
                                frame,
                                results
                            )
                            
                except Exception as e:
                    logger.error(f"Error running {detector_type} detector on video {self.video_name}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error processing frame for video {self.video_name}: {str(e)}")
            
    def _check_detection_results(self, results, conf_threshold):
        """Check detection results for valid detections"""
        has_detection = False
        max_confidence = 0.0
        
        try:
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    confidences = r.boxes.conf.tolist()
                    if confidences:
                        max_conf = max(confidences)
                        if max_conf >= conf_threshold:
                            has_detection = True
                            max_confidence = max(max_confidence, max_conf)
                            
        except Exception as e:
            logger.error(f"Error checking detection results: {str(e)}")
            
        return has_detection, max_confidence


class EnhancedCameraProcessor:
    """
    Enhanced processor for camera streams (identical to original but with enhanced naming)
    """
    
    def __init__(self, camera, detectors, device, manager, video_processor):
        self.camera = camera
        self.detectors = detectors
        self.device = device
        self.manager = manager
        self.video_processor = video_processor
        
        self.is_running = False
        self.thread = None
        self.cap = None
        self.frame_count = 0
        
        # Get detection settings per camera
        config = manager.model_manager.get_detector_config('fire_smoke')
        self.confidence_threshold = camera.confidence_threshold or config['conf_threshold']
        self.iou_threshold = camera.iou_threshold or config['iou_threshold']
        self.image_size = camera.image_size or config['image_size']
        
    def start(self):
        """Start processing this camera"""
        if self.is_running:
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop processing this camera"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            
    def update_camera_data(self, camera):
        """Update camera data"""
        self.camera = camera
        config = self.manager.model_manager.get_detector_config('fire_smoke')
        self.confidence_threshold = camera.confidence_threshold or config['conf_threshold']
        self.iou_threshold = camera.iou_threshold or config['iou_threshold']
        self.image_size = camera.image_size or config['image_size']
        
    def _process_loop(self):
        """Main processing loop for this camera"""
        try:
            # Open camera stream
            self.cap = cv2.VideoCapture(self.camera.stream_url)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera.id}: {self.camera.stream_url}")
                self._update_camera_status('offline')
                return
                
            self._update_camera_status('online')
            logger.info(f"Started processing camera {self.camera.id} - {self.camera.name}")
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from camera {self.camera.id}")
                    time.sleep(1)
                    continue
                    
                self.frame_count += 1
                
                # Skip frames for performance
                if self.frame_count % self.manager.frame_skip != 0:
                    continue
                    
                # Process frame with all enabled detectors
                self._process_frame(frame)
                
        except Exception as e:
            logger.error(f"Error in camera {self.camera.id} processing loop: {str(e)}")
            self._update_camera_status('error')
        finally:
            if self.cap:
                self.cap.release()

    def _process_frame(self, frame):
        """Process a single frame with conditional detection logic for video files"""
        try:
            # First, check for people using person detector
            person_detector = self.detectors.get('person')
            if person_detector is None:
                # Add person detector if not available
                from detectors import PersonDetector
                person_detector = PersonDetector()
                self.detectors['person'] = person_detector
            
            # Get person detection configuration
            person_config = self.manager.model_manager.get_detector_config('person')
            person_conf_threshold = person_config['conf_threshold']
            person_iou_threshold = person_config['iou_threshold']
            person_image_size = person_config['image_size']
            
            # Run person detection
            person_annotated_frame, person_results = person_detector.predict_video_frame(
                frame, 
                person_conf_threshold,
                person_iou_threshold,
                person_image_size
            )
            
            # Check if person is detected
            has_person, person_confidence = self._check_detection_results(person_results, person_conf_threshold)
            
            detectors_to_run = []
            
            if has_person:
                # Person detected - check for fall, choking, violence
                detectors_to_run.extend(['fall', 'violence', 'choking'])
            else:
                # No person detected - check for fire/smoke
                detectors_to_run.append('fire_smoke')
            
            # Run appropriate detectors based on person detection
            for detector_type in detectors_to_run:
                try:
                    detector = self.detectors[detector_type]
                    
                    # Get detector-specific configuration
                    config = self.manager.model_manager.get_detector_config(detector_type)
                    conf_threshold = config['conf_threshold']
                    iou_threshold = config['iou_threshold']
                    image_size = config['image_size']
                    
                    # Run detection
                    annotated_frame, results = detector.predict_video_frame(
                        frame, 
                        conf_threshold,
                        iou_threshold,
                        image_size
                    )
                    
                    # Check for detections
                    has_detection, max_confidence = self._check_detection_results(results, conf_threshold)
                    
                    if has_detection and max_confidence >= conf_threshold:
                        # Check if we should create an alert
                        source_id = f"video_{self.video_name}"
                        if self.manager.should_create_alert(source_id, detector_type):
                            # Create test alert with bounding box
                            self._create_test_alert_with_bbox(
                                detector_type,
                                max_confidence,
                                frame,
                                results
                            )
                            
                except Exception as e:
                    logger.error(f"Error running {detector_type} detector on video {self.video_name}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing frame for video {self.video_name}: {str(e)}")
    
    def create_test_alert_with_bbox(self, source_name, alert_type, confidence, frame, detection_results):
        """Create alert for test video detection with bounding box"""
        try:
            # Create a mock camera for test videos
            from django.contrib.auth import get_user_model
            User = get_user_model()
            
            # Get or create admin user for test videos
            admin_user = User.objects.filter(role='admin', is_active=True).first()
            if not admin_user:
                logger.warning("No admin user found for test video alerts")
                return None
            
            # Create or get test camera
            test_camera, created = Camera.objects.get_or_create(
                name=f"Test Video Camera - {source_name}",
                user=admin_user,
                defaults={
                    'stream_url': f"file://{source_name}",
                    'status': 'online',
                    'detection_enabled': True,
                    'fire_smoke_detection': True,
                    'fall_detection': True,
                    'violence_detection': True,
                    'choking_detection': True
                }
            )
            
            # Create pending alert with bounding box video
            alert = self.video_processor.create_test_detection_alert_with_bbox(
                test_camera, alert_type, confidence, detection_results, source_name, frame
            )
            
            if alert:
                logger.info(f"Created test alert {alert.id} with bounding box for {alert_type} detection in {source_name}")
                return alert
            else:
                logger.error(f"Failed to create test alert with bounding box for {alert_type} detection")
                return None
                
        except Exception as e:
            logger.error(f"Error creating test alert with bounding box: {str(e)}")
            return None

    def _create_test_alert_with_bbox(self, detector_type, confidence, frame, detection_results):
        """Create test alert with bounding box highlighting"""
        try:
            # Create test alert with bounding box video
            self.manager.create_test_alert_with_bbox(
                self.video_name,
                detector_type,
                confidence,
                frame,
                detection_results
            )
            
        except Exception as e:
            logger.error(f"Error creating test alert with bounding box: {str(e)}")

    def _check_detection_results(self, results, conf_threshold):
        """Check detection results for valid detections"""
        has_detection = False
        max_confidence = 0.0
        
        try:
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    confidences = r.boxes.conf.tolist()
                    if confidences:
                        max_conf = max(confidences)
                        if max_conf >= conf_threshold:
                            has_detection = True
                            max_confidence = max(max_confidence, max_conf)
                            
        except Exception as e:
            logger.error(f"Error checking detection results: {str(e)}")
            
        return has_detection, max_confidence
        
    def _update_camera_status(self, status):
        """Update camera status in database"""
        try:
            Camera.objects.filter(id=self.camera.id).update(
                status=status,
                last_online=timezone.now() if status == 'online' else None,
                updated_at=timezone.now()
            )
        except Exception as e:
            logger.error(f"Error updating camera status: {str(e)}")


# Global instance - replace the original detection manager
enhanced_detection_manager = EnhancedDetectionManager()