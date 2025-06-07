# utils/enhanced_video_processor.py

import os
import time
import cv2
import numpy as np
import logging
import subprocess
import uuid
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
    """Enhanced video processor with web-compatible output and organized storage"""
    
    def __init__(self, model_manager=None):
        self.model_manager = model_manager or ModelManager()
        self.base_output_dir = os.path.join(settings.MEDIA_ROOT, 'detected_videos')
        
        # Create base output directory if it doesn't exist
        os.makedirs(self.base_output_dir, exist_ok=True)
    
    def get_user_detection_dir(self, user_id, detection_type):
        """
        Create and return the directory path for storing videos by user and detection type
        
        Args:
            user_id: User ID
            detection_type: Type of detection (fire_smoke, fall, violence, choking)
            
        Returns:
            str: Path to the user's detection type directory
        """
        user_dir = os.path.join(self.base_output_dir, f"user_{user_id}")
        detection_dir = os.path.join(user_dir, detection_type)
        
        # Create directories if they don't exist
        os.makedirs(detection_dir, exist_ok=True)
        
        return detection_dir
    
    def convert_to_web_compatible(self, video_path):
        """
        Converts a video to a web-compatible format using FFmpeg.
        
        Args:
            video_path: Path to the original video
            
        Returns:
            str: Path to the converted video or original if conversion failed
        """
        try:
            # Check if ffmpeg is available
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=10
            )
            ffmpeg_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            ffmpeg_available = False
            logger.warning("FFmpeg not found, skipping web-compatible conversion")
            return video_path
        
        if not ffmpeg_available:
            return video_path
        
        logger.info(f"Converting {video_path} to web-compatible format using FFmpeg")
        
        # Create new filename for the web-compatible video
        base_dir = os.path.dirname(video_path)
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        web_path = os.path.join(base_dir, f"{name}_web.mp4")
        
        try:
            # Use FFmpeg to convert the video to H.264 in MP4 container (web compatible)
            command = [
                'ffmpeg',
                '-i', video_path,                # Input file
                '-c:v', 'libx264',               # H.264 codec
                '-preset', 'fast',               # Encoding speed/compression tradeoff
                '-crf', '23',                    # Quality (lower = better)
                '-pix_fmt', 'yuv420p',           # Pixel format for compatibility
                '-movflags', '+faststart',       # Enable streaming
                '-y',                           # Overwrite output file if it exists
                web_path                        # Output file
            ]
            
            result = subprocess.run(
                command, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0 and os.path.exists(web_path):
                logger.info(f"Successfully converted video to web format: {web_path}")
                # Remove original file to save space
                try:
                    os.remove(video_path)
                    logger.info(f"Removed original file: {video_path}")
                except OSError as e:
                    logger.warning(f"Could not remove original file: {e}")
                return web_path
            else:
                logger.error(f"FFmpeg conversion failed: {result.stderr.decode()}")
                return video_path
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg conversion timed out")
            return video_path
        except Exception as e:
            logger.error(f"Error during FFmpeg conversion: {str(e)}")
            return video_path
    
    def create_detection_video(self, camera, alert_type, detection_frames, confidence, duration=10):
        """
        Create a video clip from camera stream when detection occurs
        
        Args:
            camera: Camera model instance
            alert_type: Type of detection
            detection_frames: List of frames with detections
            confidence: Detection confidence
            duration: Duration in seconds to record
            
        Returns:
            tuple: (video_path, thumbnail_path) or (None, None) if failed
        """
        try:
            # Get detection-specific configuration
            config = self.model_manager.get_detector_config(alert_type)
            
            # Create output directory for this user and detection type
            output_dir = self.get_user_detection_dir(camera.user.id, alert_type)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = uuid.uuid4().hex[:8]
            video_filename = f"{alert_type}_{camera.id}_{timestamp}_{unique_id}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            
            # Open camera stream
            cap = cv2.VideoCapture(camera.stream_url)
            if not cap.isOpened():
                logger.error(f"Failed to open camera stream: {camera.stream_url}")
                return None, None
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Failed to create video writer for: {video_path}")
                cap.release()
                return None, None
            
            # Record video for specified duration
            start_time = time.time()
            frame_count = 0
            target_frames = fps * duration
            thumbnail_frame = None
            
            logger.info(f"Recording {duration}s video for {alert_type} detection on camera {camera.id}")
            
            while frame_count < target_frames and time.time() - start_time < duration + 5:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                # Save first frame as thumbnail
                if thumbnail_frame is None:
                    thumbnail_frame = frame.copy()
                
                out.write(frame)
                frame_count += 1
                
                # Small delay to match FPS
                time.sleep(1.0 / fps)
            
            cap.release()
            out.release()
            
            # Check if video was created successfully
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                logger.error(f"Failed to create video file: {video_path}")
                return None, None
            
            logger.info(f"Video recorded successfully: {video_path}")
            
            # Convert to web-compatible format
            web_compatible_path = self.convert_to_web_compatible(video_path)
            
            # Create thumbnail
            thumbnail_path = None
            if thumbnail_frame is not None:
                thumbnail_path = self._create_thumbnail(thumbnail_frame, output_dir, f"{alert_type}_{camera.id}_{timestamp}_{unique_id}")
            
            return web_compatible_path, thumbnail_path
            
        except Exception as e:
            logger.error(f"Error creating detection video: {str(e)}")
            return None, None
    
    def _create_thumbnail(self, frame, output_dir, base_filename):
        """
        Create a thumbnail from a frame
        
        Args:
            frame: OpenCV frame
            output_dir: Directory to save thumbnail
            base_filename: Base filename for thumbnail
            
        Returns:
            str: Path to thumbnail or None if failed
        """
        try:
            thumbnail_filename = f"{base_filename}_thumb.jpg"
            thumbnail_path = os.path.join(output_dir, thumbnail_filename)
            
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
            success = cv2.imwrite(thumbnail_path, thumbnail)
            
            if success:
                logger.info(f"Thumbnail created: {thumbnail_path}")
                return thumbnail_path
            else:
                logger.error(f"Failed to save thumbnail: {thumbnail_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating thumbnail: {str(e)}")
            return None
    
    def process_detection_with_video(self, camera, alert_type, confidence, detection_results):
        """
        Process detection and create video with pending review status
        
        Args:
            camera: Camera model instance
            alert_type: Type of detection
            confidence: Detection confidence
            detection_results: Detection results from model
            
        Returns:
            Alert: Created alert instance or None if failed
        """
        try:
            with transaction.atomic():
                # Create video clip
                video_path, thumbnail_path = self.create_detection_video(
                    camera, alert_type, [], confidence, duration=10
                )
                
                if not video_path:
                    logger.error(f"Failed to create video for {alert_type} detection")
                    return None
                
                # Determine severity based on confidence and detection type
                severity = self._determine_severity(alert_type, confidence)
                
                # Get relative paths for database storage
                video_relative_path = os.path.relpath(video_path, settings.MEDIA_ROOT)
                thumbnail_relative_path = None
                if thumbnail_path:
                    thumbnail_relative_path = os.path.relpath(thumbnail_path, settings.MEDIA_ROOT)
                
                # Create alert with pending_review status
                alert = Alert.objects.create(
                    title=f"{alert_type.replace('_', ' ').title()} Detection - Pending Review",
                    description=f"Automatic detection of {alert_type.replace('_', ' ')} from camera {camera.name} with {confidence:.2f} confidence. Awaiting reviewer confirmation.",
                    alert_type=alert_type,
                    severity=severity,
                    confidence=confidence,
                    camera=camera,
                    location=camera.name,
                    video_file=video_relative_path,
                    thumbnail=thumbnail_relative_path,
                    status='pending_review'  # New status for reviewer workflow
                )
                
                logger.info(f"Created pending alert {alert.id} for {alert_type} detection on camera {camera.id}")
                
                # Send notification to reviewers instead of end user
                self._notify_reviewers(alert)
                
                return alert
                
        except Exception as e:
            logger.error(f"Error processing detection with video: {str(e)}")
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
    
    def _notify_reviewers(self, alert):
        """
        Send notification to reviewers about pending alert
        
        Args:
            alert: Alert instance
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
                logger.warning("No active reviewers found for alert notification")
                return
            
            title = f"Alert Pending Review: {alert.get_alert_type_display()}"
            message = f"""
            A new {alert.get_alert_type_display()} alert requires review:
            
            Camera: {alert.camera.name}
            Confidence: {alert.confidence:.2f}
            Time: {alert.detection_time.strftime('%Y-%m-%d %H:%M:%S')}
            Severity: {alert.get_severity_display()}
            
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
            
            logger.info(f"Notified {reviewers.count()} reviewers about alert {alert.id}")
            
        except Exception as e:
            logger.error(f"Error notifying reviewers: {str(e)}")
    
    def get_detection_statistics(self, user_id=None):
        """
        Get statistics about stored detection videos
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            dict: Statistics about detections
        """
        try:
            stats = {
                'total_videos': 0,
                'by_type': {},
                'by_user': {},
                'total_size_mb': 0
            }
            
            base_dir = self.base_output_dir
            if user_id:
                base_dir = os.path.join(self.base_output_dir, f"user_{user_id}")
            
            if not os.path.exists(base_dir):
                return stats
            
            # Walk through directory structure
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    if file.endswith(('.mp4', '.avi')):
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        stats['total_size_mb'] += file_size / (1024 * 1024)
                        stats['total_videos'] += 1
                        
                        # Extract detection type from path
                        path_parts = root.split(os.sep)
                        if len(path_parts) >= 2:
                            detection_type = path_parts[-1]
                            if detection_type not in stats['by_type']:
                                stats['by_type'][detection_type] = 0
                            stats['by_type'][detection_type] += 1
                        
                        # Extract user from path
                        if 'user_' in root:
                            user_part = [part for part in path_parts if part.startswith('user_')]
                            if user_part:
                                user_id_str = user_part[0]
                                if user_id_str not in stats['by_user']:
                                    stats['by_user'][user_id_str] = 0
                                stats['by_user'][user_id_str] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting detection statistics: {str(e)}")
            return stats