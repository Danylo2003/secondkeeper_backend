# cameras/management/commands/start_detection.py

import signal
import sys
import time
import logging
from django.core.management.base import BaseCommand
from django.conf import settings
from utils.camera_detection_manager import detection_manager

logger = logging.getLogger('security_ai')

class Command(BaseCommand):
    help = 'Start the automatic camera detection service'
    
    def __init__(self):
        super().__init__()
        self.shutdown_requested = False
        
    def add_arguments(self, parser):
        parser.add_argument(
            '--no-daemon',
            action='store_true',
            help='Run in foreground instead of daemon mode',
        )
        
    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting Camera Detection Service...'))
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        try:
            # Start the detection manager
            detection_manager.start()
            
            self.stdout.write(
                self.style.SUCCESS(
                    'Camera Detection Service started successfully!\n'
                    'The service is now monitoring all online cameras for threats.\n'
                    'Press Ctrl+C to stop the service.'
                )
            )
            
            # Keep the service running
            while not self.shutdown_requested:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nReceived interrupt signal...'))
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'Error in detection service: {str(e)}')
            )
            logger.error(f"Detection service error: {str(e)}")
        finally:
            self.shutdown()
            
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.stdout.write(self.style.WARNING(f'\nReceived signal {signum}. Shutting down...'))
        self.shutdown_requested = True
        
    def shutdown(self):
        """Shutdown the detection service gracefully"""
        try:
            self.stdout.write(self.style.WARNING('Stopping Camera Detection Service...'))
            detection_manager.stop()
            self.stdout.write(self.style.SUCCESS('Camera Detection Service stopped successfully.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error during shutdown: {str(e)}'))
            logger.error(f"Shutdown error: {str(e)}")
            
        sys.exit(0)