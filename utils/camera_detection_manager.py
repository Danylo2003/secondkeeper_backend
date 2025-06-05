# utils/camera_detection_manager.py

import cv2
import numpy as np
import threading
import time
import os
import uuid
import logging
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from django.conf import settings
from django.utils import timezone
from django.db import transaction
from django.core.files.base import ContentFile
import torch

from cameras.models import Camera
from alerts.models import Alert
from detectors import FireSmokeDetector, FallDetector, ViolenceDetector, ChokingDetector
from detectors.face_detector import FaceDetector
from utils.model_manager import ModelManager
from notifications.models import NotificationSetting, NotificationLog
from django.core.mail import send_mail

logger = logging.getLogger('security_ai')

class CameraDetectionManager:
    """
    Main manager for handling automatic detection across all cameras
    """
    
    def __init__(self):
        self.active_cameras = {}  # camera_id -> CameraProcessor
        self.model_manager = ModelManager()
        self.is_running = False
        self.main_thread = None
        self.stop_event = threading.Event()
        
        # Load all detectors
        self.detectors = {
            'fire_smoke': FireSmokeDetector(),
            'fall': FallDetector(),
            'violence': ViolenceDetector(),
            'choking': ChokingDetector()
        }
        
        # Face detector (separate handling)
        self.face_detector = FaceDetector()
        
        # GPU optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Detection settings
        self.frame_skip = 5  # Process every 5th frame for performance
        self.detection_cooldown = 30  # Seconds between notifications for same camera
        self.video_clip_duration = 10  # Seconds for video clips
        
        # Alert tracking to prevent spam
        self.last_alerts = {}  # camera_id -> {alert_type: timestamp}
        
    def start(self):
        """Start the detection manager"""
        if self.is_running:
            logger.warning("Detection manager is already running")
            return
            
        logger.info("Starting Camera Detection Manager")
        self.is_running = True
        self.stop_event.clear()
        
        # Start main processing thread
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        
    def stop(self):
        """Stop the detection manager"""
        if not self.is_running:
            return
            
        logger.info("Stopping Camera Detection Manager")
        self.is_running = False
        self.stop_event.set()
        
        # Stop all camera processors
        for processor in self.active_cameras.values():
            processor.stop()
            
        # Wait for main thread to finish
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=10)
            
        self.active_cameras.clear()
        
    def _main_loop(self):
        """Main processing loop"""
        while self.is_running and not self.stop_event.is_set():
            try:
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
                
                # Update existing processors
                for camera in online_cameras:
                    camera_id = str(camera.id)
                    if camera_id in self.active_cameras:
                        self.active_cameras[camera_id].update_camera_data(camera)
                
                # Clean up old alerts from tracking
                self._cleanup_old_alerts()
                
            except Exception as e:
                logger.error(f"Error in main detection loop: {str(e)}")
            
            # Sleep for a bit before next iteration
            time.sleep(10)
            
    def _start_camera_processor(self, camera):
        """Start a processor for a single camera"""
        try:
            camera_id = str(camera.id)
            logger.info(f"Starting processor for camera {camera_id} - {camera.name}")
            
            processor = CameraProcessor(camera, self.detectors, self.face_detector, self.device, self)
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
            
    def _cleanup_old_alerts(self):
        """Clean up old alert timestamps"""
        current_time = time.time()
        cutoff_time = current_time - self.detection_cooldown
        
        for camera_id in list(self.last_alerts.keys()):
            camera_alerts = self.last_alerts[camera_id]
            for alert_type in list(camera_alerts.keys()):
                if camera_alerts[alert_type] < cutoff_time:
                    del camera_alerts[alert_type]
            
            # Remove camera entry if no alerts
            if not camera_alerts:
                del self.last_alerts[camera_id]
                
    def should_create_alert(self, camera_id, alert_type):
        """Check if we should create a new alert (to prevent spam)"""
        current_time = time.time()
        
        if camera_id not in self.last_alerts:
            self.last_alerts[camera_id] = {}
            
        last_alert_time = self.last_alerts[camera_id].get(alert_type, 0)
        
        if current_time - last_alert_time >= self.detection_cooldown:
            self.last_alerts[camera_id][alert_type] = current_time
            return True
            
        return False
        
    def create_alert_and_notify(self, camera, alert_type, confidence, frame, detection_results):
        """Create alert and send notifications"""
        try:
            with transaction.atomic():
                # Create video clip filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"{alert_type}_{camera.id}_{timestamp}.mp4"
                
                # Determine severity based on confidence
                if confidence >= 0.9:
                    severity = 'critical'
                elif confidence >= 0.7:
                    severity = 'high'
                elif confidence >= 0.5:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                # Create alert
                alert = Alert.objects.create(
                    title=f"{alert_type.replace('_', ' ').title()} Detected",
                    description=f"Automatic detection of {alert_type.replace('_', ' ')} from camera {camera.name} with {confidence:.2f} confidence.",
                    alert_type=alert_type,
                    severity=severity,
                    confidence=confidence,
                    camera=camera,
                    location=camera.name,
                    status='new'
                )
                
                # Save thumbnail
                self._save_thumbnail(alert, frame)
                
                # Start video recording in background
                threading.Thread(
                    target=self._record_video_clip,
                    args=(camera, alert, video_filename),
                    daemon=True
                ).start()
                
                # Send notifications
                self._send_notifications(camera.user, alert)
                
                logger.info(f"Created alert {alert.id} for {alert_type} detection on camera {camera.id}")
                
        except Exception as e:
            logger.error(f"Error creating alert: {str(e)}")
            
    def _save_thumbnail(self, alert, frame):
        """Save thumbnail from detection frame"""
        try:
            # Create thumbnail directory if it doesn't exist
            thumbnail_dir = os.path.join(settings.MEDIA_ROOT, 'alerts', 'thumbnails')
            os.makedirs(thumbnail_dir, exist_ok=True)
            
            # Resize frame for thumbnail
            height, width = frame.shape[:2]
            max_dim = 400
            if height > width:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            else:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
                
            thumbnail = cv2.resize(frame, (new_width, new_height))
            
            # Save thumbnail
            thumbnail_filename = f"thumb_{alert.id}_{uuid.uuid4().hex[:8]}.jpg"
            thumbnail_path = os.path.join(thumbnail_dir, thumbnail_filename)
            
            cv2.imwrite(thumbnail_path, thumbnail)
            
            # Update alert with thumbnail path
            alert.thumbnail = f"alerts/thumbnails/{thumbnail_filename}"
            alert.save(update_fields=['thumbnail'])
            
        except Exception as e:
            logger.error(f"Error saving thumbnail: {str(e)}")
            
    def _record_video_clip(self, camera, alert, video_filename):
        """Record video clip after detection"""
        try:
            video_dir = os.path.join(settings.MEDIA_ROOT, 'alerts', 'videos')
            os.makedirs(video_dir, exist_ok=True)
            
            video_path = os.path.join(video_dir, video_filename)
            
            # Open camera stream
            cap = cv2.VideoCapture(camera.stream_url)
            if not cap.isOpened():
                logger.error(f"Failed to open camera stream for video recording: {camera.stream_url}")
                return
                
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Record for specified duration
            start_time = time.time()
            frame_count = 0
            target_frames = fps * self.video_clip_duration
            
            while frame_count < target_frames and time.time() - start_time < self.video_clip_duration + 5:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                out.write(frame)
                frame_count += 1
                
            cap.release()
            out.release()
            
            # Update alert with video file
            if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                alert.video_file = f"alerts/videos/{video_filename}"
                alert.save(update_fields=['video_file'])
                logger.info(f"Video clip saved: {video_path}")
            else:
                logger.error(f"Failed to create video clip: {video_path}")
                
        except Exception as e:
            logger.error(f"Error recording video clip: {str(e)}")
            
    def _send_notifications(self, user, alert):
        """Send notifications to the user"""
        try:
            # Get user's notification settings
            try:
                notification_settings = NotificationSetting.objects.get(user=user)
            except NotificationSetting.DoesNotExist:
                # Create default settings
                notification_settings = NotificationSetting.objects.create(user=user)
                
            # Check if notifications should be sent based on settings
            should_send_email = self._should_send_notification(
                notification_settings, alert, 'email'
            )
            
            if should_send_email:
                self._send_email_notification(user, alert, notification_settings)
                
        except Exception as e:
            logger.error(f"Error sending notifications: {str(e)}")
            
    def _should_send_notification(self, settings, alert, notification_type):
        """Check if notification should be sent based on user settings"""
        try:
            # Check if notification type is enabled
            if notification_type == 'email' and not settings.email_enabled:
                return False
            elif notification_type == 'sms' and not settings.sms_enabled:
                return False
            elif notification_type == 'push' and not settings.push_enabled:
                return False
                
            # Check alert type settings
            alert_type_enabled = getattr(
                settings, 
                f"{notification_type}_for_{alert.alert_type}", 
                True
            )
            if not alert_type_enabled:
                return False
                
            # Check severity threshold
            severity_rank = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            alert_severity_rank = severity_rank.get(alert.severity, 2)
            
            min_severity_field = f"min_severity_{notification_type}"
            min_severity = getattr(settings, min_severity_field, 'medium')
            min_severity_rank = severity_rank.get(min_severity, 2)
            
            if alert_severity_rank < min_severity_rank:
                return False
                
            # Check quiet hours
            if settings.quiet_hours_enabled:
                current_time = timezone.now().time()
                start_time = settings.quiet_hours_start
                end_time = settings.quiet_hours_end
                
                # Check if in quiet hours
                in_quiet_hours = False
                if start_time <= end_time:
                    in_quiet_hours = start_time <= current_time <= end_time
                else:
                    in_quiet_hours = current_time >= start_time or current_time <= end_time
                
                if in_quiet_hours and alert.severity != 'critical':
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking notification settings: {str(e)}")
            return False
            
    def _send_email_notification(self, user, alert, settings):
        """Send email notification"""
        try:
            title = f"ALERT: {alert.get_alert_type_display()} Detected"
            message = f"""
            {alert.get_alert_type_display()} detected at {alert.camera.name} with {alert.confidence:.2f} confidence.
            
            Details:
            - Camera: {alert.camera.name}
            - Time: {alert.detection_time.strftime('%Y-%m-%d %H:%M:%S')}
            - Severity: {alert.get_severity_display()}
            - Confidence: {alert.confidence:.2f}
            
            Please check your security system for more details.
            """
            
            # Create notification log
            notification_log = NotificationLog.objects.create(
                user=user,
                title=title,
                message=message,
                notification_type='email',
                alert=alert,
                status='pending'
            )
            
            # Send email
            send_mail(
                subject=title,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=False,
            )
            
            # Update notification log
            notification_log.status = 'sent'
            notification_log.sent_at = timezone.now()
            notification_log.save()
            
            logger.info(f"Email notification sent to {user.email} for alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {str(e)}")
            if 'notification_log' in locals():
                notification_log.status = 'failed'
                notification_log.error_message = str(e)
                notification_log.save()


class CameraProcessor:
    """
    Processor for a single camera
    """
    
    def __init__(self, camera, detectors, face_detector, device, manager):
        self.camera = camera
        self.detectors = detectors
        self.face_detector = face_detector
        self.device = device
        self.manager = manager
        
        self.is_running = False
        self.thread = None
        self.cap = None
        self.frame_count = 0
        
        # Detection settings per camera
        self.confidence_threshold = camera.confidence_threshold
        self.iou_threshold = camera.iou_threshold
        self.image_size = camera.image_size
        
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
        self.confidence_threshold = camera.confidence_threshold
        self.iou_threshold = camera.iou_threshold
        self.image_size = camera.image_size
        
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
        """Process a single frame with all detectors"""
        try:
            # Check each detector if enabled for this camera
            detectors_to_run = []
            
            if self.camera.fire_smoke_detection:
                detectors_to_run.append('fire_smoke')
            if self.camera.fall_detection:
                detectors_to_run.append('fall')
            if self.camera.violence_detection:
                detectors_to_run.append('violence')
            if self.camera.choking_detection:
                detectors_to_run.append('choking')
                
            # Run object detection models
            for detector_type in detectors_to_run:
                try:
                    detector = self.detectors[detector_type]
                    
                    # Run detection
                    annotated_frame, results = detector.predict_video_frame(
                        frame, 
                        self.confidence_threshold,
                        self.iou_threshold,
                        self.image_size
                    )
                    
                    # Check for detections
                    has_detection, max_confidence = self._check_detection_results(results)
                    
                    if has_detection and max_confidence >= self.confidence_threshold:
                        # Check if we should create an alert
                        if self.manager.should_create_alert(str(self.camera.id), detector_type):
                            self.manager.create_alert_and_notify(
                                self.camera,
                                detector_type,
                                max_confidence,
                                frame,
                                results
                            )
                            
                except Exception as e:
                    logger.error(f"Error running {detector_type} detector on camera {self.camera.id}: {str(e)}")
            
            # Run face recognition if enabled
            if self.camera.face_recognition:
                try:
                    has_unauthorized, face_confidence, face_annotated_frame, face_results = self.face_detector.detect_faces_in_frame(
                        frame, self.camera, self.confidence_threshold
                    )
                    
                    if has_unauthorized and face_confidence >= self.confidence_threshold:
                        # Check if we should create an alert for unauthorized face
                        if self.manager.should_create_alert(str(self.camera.id), 'unauthorized_face'):
                            self.manager.create_alert_and_notify(
                                self.camera,
                                'unauthorized_face',
                                face_confidence,
                                frame,
                                face_results
                            )
                            
                except Exception as e:
                    logger.error(f"Error running face recognition on camera {self.camera.id}: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error processing frame for camera {self.camera.id}: {str(e)}")
            
    def _check_detection_results(self, results):
        """Check detection results for valid detections"""
        has_detection = False
        max_confidence = 0.0
        
        try:
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    confidences = r.boxes.conf.tolist()
                    if confidences:
                        max_conf = max(confidences)
                        if max_conf >= self.confidence_threshold:
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


# Global instance
detection_manager = CameraDetectionManager()