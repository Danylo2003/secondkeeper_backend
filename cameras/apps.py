# cameras/apps.py - Updated to auto-start detection service

from django.apps import AppConfig
import threading
import time
import logging

logger = logging.getLogger('security_ai')

class CamerasConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'cameras'
    
    def ready(self):
        """Called when Django is ready"""
        # Only start detection service in the main process
        # and not during migrations or management commands
        import os
        import sys
        
        # Check if this is the main Django process
        if (os.environ.get('RUN_MAIN') == 'true' or 
            'runserver' not in sys.argv and 
            'migrate' not in sys.argv and
            'makemigrations' not in sys.argv and
            'collectstatic' not in sys.argv and
            'shell' not in sys.argv):
            
            # Start detection service after a short delay
            def start_detection_service():
                time.sleep(5)  # Wait for Django to fully initialize
                try:
                    from utils.camera_detection_manager import detection_manager
                    logger.info("Auto-starting Camera Detection Service...")
                    detection_manager.start()
                    logger.info("Camera Detection Service started automatically")
                except Exception as e:
                    logger.error(f"Failed to auto-start detection service: {str(e)}")
            
            # Start in a separate thread
            thread = threading.Thread(target=start_detection_service, daemon=True)
            thread.start()