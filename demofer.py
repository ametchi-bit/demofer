# IMPORTANT: Page config must be the first Streamlit command
import streamlit as st

# Configure page first before any other Streamlit operations
st.set_page_config(
    page_title=" Computer Vision Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core imports
import cv2
# Disable OpenCV's GUI functions that won't work in cloud
cv2.setUseOptimized(True)
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import glob
import json
import csv
import ast
import pickle
import base64
from datetime import datetime, timedelta
import torch
from scipy import stats
from collections import defaultdict, deque 
import logging
import traceback
import threading
import queue
import time
import urllib
import urllib.parse

# Import language and styling modules
from language import LanguageManager, get_text, _, lang_manager as language_manager
from styles import StyleManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File and path configuration
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

# Set environment variables for headless operation
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Configuration constants
# IMAGE = 'Image' # Commented out
VIDEO = 'Video'
CAMERA = 'Live Camera'
SOURCES_LIST = [VIDEO, CAMERA] # 'IMAGE' removed

# Directory configurations
IMAGES_DIR = ROOT / 'media/images'
DEFAULT_IMAGE = IMAGES_DIR / 'image_1.jpg'
VIDEO_DIR = ROOT / 'media/videos'
MODEL_DIR = ROOT / 'weight_models'

# Video dictionary
VIDEOS_DICT = {
    'Voie principale': VIDEO_DIR / 'area1.mp4',
    'Acceuil': VIDEO_DIR / 'area2.mp4', 
    'Escalier': VIDEO_DIR / 'area3.mp4',
    'Parking': VIDEO_DIR / 'carPark.mp4',
    'Voie publique(EU) Plaque': VIDEO_DIR / 'sample.mp4',
    'Voie publique(CI) Plaque': VIDEO_DIR / 'sample2.mp4',
    'Voie publique(CI2) Plaque': VIDEO_DIR / 'samplex.mp4',
}

# Import from mainCountingTest
from ShapeAlertCounting.mainCountingTest import AlertCountingSystem, COLORS as COUNTING_COLORS, load_shapes as load_counting_shapes

# Import from parking_detection  
from parkingManagement.parking_detection import ParkingDetectionConfig, ParkingSpotDetector

# Model configurations
MODEL_CONFIGS = {
    'Detection': MODEL_DIR / 'yolo11n.pt',
    'Segmentation': MODEL_DIR / 'yolo11n-seg.pt', 
    'Pose Estimation': MODEL_DIR / 'yolo11n-pose.pt',
    'License Plate': MODEL_DIR / 'license_plate.pt',
    'Parking': MODEL_DIR / 'yolo11n.pt',  # Uses YOLO for vehicle detection in parking spots
    'Counting': MODEL_DIR / 'yolo11n.pt',  # Uses YOLO for object detection in counting system
}


# Enhanced custom modules imports with error handling
try:
    from main import (
        camera_1, camera_2, camera_3, camera_4, 
        get_pytorch_device, RTSPCameraManager,
        EnhancedLicensePlateProcessor,
        initialize_enhanced_license_plate_system,
        process_license_plates_with_visualization,
        get_enhanced_processor_for_display,
        process_frame_for_display,
        LICENSE_PLATE_UTILS_AVAILABLE,
        SORT_AVAILABLE,
        process_video_file_with_enhanced_visualization,
        live_demo_with_enhanced_visualization,
    )
    MAIN_MODULE_AVAILABLE = True
    logger.info("✅ Main module imported successfully")
except ImportError as e:
    logger.error(f"❌ Error importing main modules: {e}")
    MAIN_MODULE_AVAILABLE = False
    LICENSE_PLATE_UTILS_AVAILABLE = False
    SORT_AVAILABLE = False

# Import post-processing modules
try:
    from add_missing_data import interpolate_bounding_boxes
    POST_PROCESSING_AVAILABLE = True
    logger.info("✅ Post-processing modules available")
except ImportError:
    POST_PROCESSING_AVAILABLE = False
    logger.warning("⚠️ Post-processing modules not available")

# Attempt to import cvzone for enhanced visualization
CVZONE_AVAILABLE = False
try:
    import cvzone
    CVZONE_AVAILABLE = True
    logger.info("✅ CVZone available for enhanced visualization")
except ImportError:
    logger.warning("⚠️ CVZone not available, using basic OpenCV text drawing")

# Camera configuration from main.py
CAMERA_MAP = {
    '1': "10.20.1.7",
    '2': "10.20.1.8",
    '3': "10.20.1.18",
    '4': "10.20.1.19"
}

# Camera connection functions mapping
CAMERA_FUNCTIONS = {
    'Camera 1': camera_1 if MAIN_MODULE_AVAILABLE else None,
    'Camera 2': camera_2 if MAIN_MODULE_AVAILABLE else None,
    'Camera 3': camera_3 if MAIN_MODULE_AVAILABLE else None,
    'Camera 4': camera_4 if MAIN_MODULE_AVAILABLE else None,
}

# Enhanced availability flags
ENHANCED_LICENSE_PLATE_AVAILABLE = (
    MAIN_MODULE_AVAILABLE and 
    LICENSE_PLATE_UTILS_AVAILABLE and 
    SORT_AVAILABLE and
    MODEL_CONFIGS['License Plate'].exists() and
    MODEL_CONFIGS['Detection'].exists()
)

PARKING_AVAILABLE = (
    'ParkingDetectionConfig' in globals() and
    'ParkingSpotDetector' in globals() and
    MODEL_CONFIGS['Parking'].exists()
)

COUNTING_AVAILABLE = (
    'AlertCountingSystem' in globals() and
    MODEL_CONFIGS['Counting'].exists()
)

# Enhanced configuration constants
PARKING_SPACE_WIDTH_DEFAULT = 107
PARKING_SPACE_HEIGHT_DEFAULT = 48
DEFAULT_PARKING_PIXEL_THRESHOLD = 900
VEHICLE_CLASSES_FOR_PARKING = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
PARKING_YOLO_TARGET_CLASSES = ['car', 'truck', 'bus', 'motorcycle']


# Zone counting shapes
ZONE_SHAPE = ROOT / 'ShapeAlertCounting/mainRoadZone/CountingZonePos copy'

# Hall counting shapes
HALL_SHAPE = ROOT / 'ShapeAlertCounting/hallZone/CountingHallPos copy 2'

# Stair counting shapes
STAIR_SHAPE = ROOT / 'ShapeAlertCounting/StairsZone/CountingZonePos copy'

# Parking positions file
PARKING_POSITIONS = ROOT / 'parkingManagement/CarParkPos'

# Initialize global managers
style_manager = StyleManager()

logger.info(f"🔧 System Status - ELP: {ENHANCED_LICENSE_PLATE_AVAILABLE}, Parking: {PARKING_AVAILABLE}, Counting: {COUNTING_AVAILABLE}")

# Initialize session state variables
def initialize_session_state():
    """Initialize all session state variables for the vision platform"""
    
    # Core session states
    if 'camera_managers' not in st.session_state:
        st.session_state.camera_managers = {}
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False
    if 'current_camera' not in st.session_state:
        st.session_state.current_camera = None
    if 'session_start' not in st.session_state:
        st.session_state.session_start = time.time()
    
    # Enhanced processors
    if 'license_processor' not in st.session_state:
        st.session_state.license_processor = None
    if 'parking_detector' not in st.session_state:
        st.session_state.parking_detector = None
    if 'counting_system' not in st.session_state:
        st.session_state.counting_system = None
    
    # Processing statistics
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = {
            'frames_processed': 0,
            'detections_count': 0,
            'processing_fps': 0.0,
            'start_time': None
        }
    
    # Detection analytics - ENSURE THIS IS ALWAYS INITIALIZED
    if 'detection_analytics' not in st.session_state:
        st.session_state.detection_analytics = {
            'total_detections': 0,
            'unique_plates': set(),
            'detection_timeline': [],
            'confidence_scores': []
        }
    
    # Parking-related session states
    if 'posList' not in st.session_state:
        st.session_state.posList = []
    if 'parking_config' not in st.session_state and PARKING_AVAILABLE:
        try:
            st.session_state.parking_config = ParkingDetectionConfig()
        except:
            st.session_state.parking_config = None
    if 'parking_spot_detector' not in st.session_state and PARKING_AVAILABLE:
        try:
            st.session_state.parking_spot_detector = ParkingSpotDetector(
                config=st.session_state.parking_config if 'parking_config' in st.session_state else None
            )
        except:
            st.session_state.parking_spot_detector = None
    if 'parking_spot_states' not in st.session_state:
        st.session_state.parking_spot_states = {}
    if 'parking_events_log' not in st.session_state:
        st.session_state.parking_events_log = []
    
    # Counting system states
    if 'counting_shapes_list' not in st.session_state:
        st.session_state.counting_shapes_list = []
    if 'counting_events_history' not in st.session_state:
        st.session_state.counting_events_history = []
    if 'counting_alert_system_instance' not in st.session_state:
        st.session_state.counting_alert_system_instance = None
    
    # Enhanced processing flags
    if 'is_parking_detection_active' not in st.session_state:
        st.session_state.is_parking_detection_active = False
    if 'is_enhanced_license_plate_active' not in st.session_state:
        st.session_state.is_enhanced_license_plate_active = False
    if 'is_counting_alert_system_active' not in st.session_state:
        st.session_state.is_counting_alert_system_active = False

    # Camera specific settings
    if 'camera_settings' not in st.session_state:
        st.session_state.camera_settings = {}

    # For auto-analysis of latest counting log
    if 'last_auto_analyzed_counting_log_path' not in st.session_state:
        st.session_state.last_auto_analyzed_counting_log_path = None

    logger.info("✅ Session state initialized successfully")

# Handle missing display gracefully
try:
    import os
    if not os.environ.get('DISPLAY'):
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
except:
    pass
class LiveVideoProcessor:
    """Separate thread for live video processing using OpenCV display"""
    
    def __init__(self, camera_id: int, model, task_type: str, confidence: float, 
                 vision_platform_instance, specialized_processor=None, shapes_list=None): # Added shapes_list
        
        self.camera_id = camera_id
        self.model = model # For standard tasks if specialized_processor is not for this task_type
        self.task_type = task_type
        self.confidence = confidence
        self.vision_platform = vision_platform_instance # Renamed for clarity
        self.specialized_processor = specialized_processor # Store the passed-in processor
        self.shapes_list = shapes_list if shapes_list is not None else [] # Store shapes_list
        self.is_cloud_environment = self._detect_cloud_environment()

        self.running = False
        self.thread = None
        self.cap = None
        self.stats_queue = queue.Queue(maxsize=1)  # For sending stats back to Streamlit
        
        # Processing statistics
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = None
        
    def connect_camera(self):
        """Connect to camera using the same logic as mainCountingTest.py"""
        camera_ip = CAMERA_MAP[str(self.camera_id)]
        encoded_password = urllib.parse.quote("d@t@rium2023")
        username = "admin"
        port = "554"
        
        # RTSP URLs to try (same as mainCountingTest.py)
        rtsp_urls = [
            f"rtsp://{username}:{encoded_password}@{camera_ip}:{port}/h264/ch{self.camera_id}/main/av_stream?rtsp_transport=tcp",
            f"rtsp://{username}:{encoded_password}@{camera_ip}:{port}/h264/ch{self.camera_id}/sub/av_stream?rtsp_transport=tcp",
            f"rtsp://{username}:{encoded_password}@{camera_ip}:{port}/Streaming/Channels/{self.camera_id}01?rtsp_transport=tcp",
            f"rtsp://{username}:{encoded_password}@{camera_ip}:{port}/Streaming/Channels/{self.camera_id}02?rtsp_transport=tcp",
        ]
        
        print(f"Attempting to connect to camera {self.camera_id} at {camera_ip}...")
        
        for i, url in enumerate(rtsp_urls):
            print(f"Trying URL {i+1}: {url}")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Configure capture properties (same as mainCountingTest.py)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    print(f"Successfully connected using URL {i+1}")
                    self.cap = cap
                    return True
                else:
                    cap.release()
            
        print(f"Failed to connect to camera {self.camera_id}")
        return False
    
    def start_processing(self):
        """Start the video processing thread"""
        if not self.connect_camera():
            return False
        
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        return True
    
    def _detect_cloud_environment(self):
        """Detect if running in cloud environment"""
        return (
            os.environ.get('STREAMLIT_SHARING_MODE') == '1' or
            'streamlit.app' in os.environ.get('HOSTNAME', '') or
            not os.environ.get('DISPLAY')
        )
    
    def stop_processing(self):
        """Stop the video processing thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info(f"Camera {self.camera_id} processing stopped and OpenCV windows closed.")

        # Save counting log if applicable
        if self.task_type == "Counting & Alert System" and \
           self.specialized_processor and \
           hasattr(self.specialized_processor, 'save_log_to_csv') and \
           hasattr(self.specialized_processor, 'crossing_log') and \
           self.specialized_processor.crossing_log: # Only save if there's something to save

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # The identifier should lead to a filename like "live_counting_camera_1_20230101_120000_crossings.csv"
            # The AlertCountingSystem.save_log_to_csv will append "_crossings.csv"
            identifier = f"live_counting_camera_{self.camera_id}_{timestamp}"
            try:
                self.specialized_processor.save_log_to_csv(identifier)
                logger.info(f"Counting log for camera {self.camera_id} saved with base identifier: {identifier}")
            except Exception as e:
                logger.error(f"Error saving counting log from LiveVideoProcessor for camera {self.camera_id}: {e}")
    
    
    def _processing_loop(self):
        """Modified processing loop for cloud compatibility"""
        if self.is_cloud_environment:
            # In cloud environment, don't create OpenCV windows
            self._processing_loop_headless()
        else:
            # Local environment with GUI
            self._processing_loop_with_gui()
    
    def _processing_loop_headless(self):
        """Processing loop without OpenCV GUI (for cloud)"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                
                self.frame_count += 1
                
                # Process frame
                processed_frame, results = self._process_frame(frame)
                
                # Update analytics without displaying
                if results and not results.get('error'):
                    self.vision_platform.update_detection_analytics(results)
                    self.detection_count += results.get('detections_count', 0)
                
                # Send stats to Streamlit
                self._update_stats_queue()
                
                # Small delay to prevent overwhelming
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)
                
    def _processing_loop_with_gui(self):
        """Original processing loop with OpenCV windows (for local)"""
        window_name = f"Camera {self.camera_id} - {self.task_type}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.running:
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Camera {self.camera_id}: Too many consecutive errors")
                        break
                    time.sleep(0.1)
                    continue
                
                consecutive_errors = 0
                self.frame_count += 1
                
                # Process frame based on task type
                processed_frame, results = self._process_frame(frame)
                
                # Add overlay information
                self._add_overlay_info(processed_frame, results)
                
                # Display frame
                cv2.imshow(window_name, processed_frame)
                
                # Update analytics in vision platform
                if results and not results.get('error'):
                    self.vision_platform.update_detection_analytics(results)
                    self.detection_count += results.get('detections_count', 0)
                
                # Send stats to Streamlit (non-blocking)
                self._update_stats_queue()
                
                # Handle keyboard input (same as mainCountingTest.py)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._print_stats()
                elif key == ord('r'):
                    self.vision_platform.reset_tracking()
                    print("Tracking data reset")
                
            except Exception as e:
                print(f"Processing error: {e}")
                consecutive_errors += 1
                time.sleep(0.1)
        
        # Cleanup
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print(f"Camera {self.camera_id} processing stopped")
    
    def _process_frame(self, frame):
        """Process frame based on task type, using the specialized_processor if available."""
        try:
            if self.task_type == "Enhanced License Plate" and self.specialized_processor:
                # Assuming specialized_processor is the license_processor instance
                # process_frame_for_display already accepts the processor instance
                return self.vision_platform.process_frame_for_display(frame, self.specialized_processor)
            
            elif self.task_type == "Parking Detection" and self.specialized_processor:
                # Assuming specialized_processor is the parking_detector instance
                # VisionPlatform.process_parking_frame needs to accept this instance.
                return self.vision_platform.process_parking_frame(frame, parking_detector_instance=self.specialized_processor)
            
            elif self.task_type == "Counting & Alert System" and self.specialized_processor:
                # Assuming specialized_processor is the counting_system instance
                # VisionPlatform.process_counting_frame needs to accept this instance.
                return self.vision_platform.process_counting_frame(frame, counting_system_instance=self.specialized_processor, shapes_list_param=self.shapes_list)
            
            elif self.model: # Fallback to standard model (passed as self.model)
                return self.vision_platform.process_standard_frame(frame, self.model, self.confidence)
            else: 
                # No suitable processor found for the task
                error_msg = f"No processor for {self.task_type}"
                logger.error(f"LiveVideoProcessor: {error_msg}")
                cv2.putText(frame, error_msg, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                return frame, {'error': error_msg}
                
        except Exception as e:
            # Log the full traceback for better debugging
            detailed_error = traceback.format_exc()
            logger.error(f"Frame processing error in LiveVideoProcessor for task {self.task_type}: {e}\n{detailed_error}")
            cv2.putText(frame, "Processing Error", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return frame, {'error': str(e)}
    
    def _add_overlay_info(self, frame, results):
        """Add overlay information to frame"""
        try:
            # Performance metrics
            elapsed = time.time() - self.start_time if self.start_time else 1
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # Basic overlay information
            overlay_texts = [
                f"Camera {self.camera_id} - {self.task_type}",
                f"FPS: {fps:.1f}",
                f"Frames: {self.frame_count}",
                f"Detections: {self.detection_count}",
            ]
            
            # Task-specific overlay
            if results.get('model_type') == 'Enhanced License Plate':
                plates_read = results.get('plates_read', 0)
                if plates_read > 0:
                    overlay_texts.append(f"Plates: {plates_read}")
                    
            elif results.get('model_type') == 'Parking Detection':
                free_spaces = results.get('free_spaces', 0)
                total_spaces = results.get('total_spaces', 0)
                overlay_texts.append(f"Parking: {free_spaces}/{total_spaces}")
                
            elif results.get('model_type') == 'Counting & Alert System':
                alerts = results.get('active_alerts', 0)
                crossings = results.get('total_crossings', 0)
                overlay_texts.append(f"Alerts: {alerts}")
                overlay_texts.append(f"Crossings: {crossings}")
            
            # Draw overlay with better visibility
            y_offset = 30
            for text in overlay_texts:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (5, y_offset - 25), (15 + text_size[0], y_offset + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 30
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            timestamp_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (frame.shape[1] - timestamp_size[0] - 15, 5), 
                         (frame.shape[1] - 5, 35), (0, 0, 0), -1)
            cv2.putText(frame, timestamp, (frame.shape[1] - timestamp_size[0] - 10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add controls info
            controls_text = "Controls: Q=Quit, S=Stats, R=Reset"
            cv2.putText(frame, controls_text, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        except Exception as e:
            print(f"Error adding overlay: {e}")
    
    def _update_stats_queue(self):
        """Update statistics for Streamlit (non-blocking)"""
        try:
            elapsed = time.time() - self.start_time if self.start_time else 1
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            stats = {
                'frame_count': self.frame_count,
                'detection_count': self.detection_count,
                'fps': fps,
                'elapsed_time': elapsed
            }
            
            # Non-blocking update
            try:
                self.stats_queue.put_nowait(stats)
            except queue.Full:
                # Remove old stats and add new
                try:
                    self.stats_queue.get_nowait()
                    self.stats_queue.put_nowait(stats)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            print(f"Error updating stats: {e}")
    
    def get_stats(self):
        """Get latest statistics (for Streamlit)"""
        try:
            return self.stats_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _print_stats(self):
        """Print detailed statistics to console"""
        elapsed = time.time() - self.start_time if self.start_time else 1
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n📊 STATISTICS (Camera {self.camera_id}):")
        print(f"Frames processed: {self.frame_count}")
        print(f"Total detections: {self.detection_count}")
        print(f"Processing FPS: {fps:.2f}")
        print(f"Elapsed time: {elapsed:.1f}s")
        print("-" * 30)

class VisionPlatform:
    """Main Vision Platform Class with Enhanced Features"""
    
    def __init__(self):
        """Initialize the vision platform with all components"""
        try:
            # Initialize language manager first
            language_manager.initialize()
            
            # Initialize session state BEFORE anything else
            initialize_session_state()
            
            # Initialize live processors tracking
            self.live_processors = {}
            
            # Initialize other components
            self.initialize_components()
            
            logger.info("🚀 VisionPlatform initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing VisionPlatform: {e}")
            # Ensure basic session state exists even if there's an error
            if 'detection_analytics' not in st.session_state:
                st.session_state.detection_analytics = {
                    'total_detections': 0,
                    'unique_plates': set(),
                    'detection_timeline': [],
                    'confidence_scores': []
                }
            raise

    def initialize_components(self):
        """Initialize all platform components"""
        try:
            # Apply global styling
            style_manager.apply_global_styles()
            
            # Double-check session state is initialized
            if 'detection_analytics' not in st.session_state:
                initialize_session_state()
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            # Ensure critical session state exists
            if 'detection_analytics' not in st.session_state:
                st.session_state.detection_analytics = {
                    'total_detections': 0,
                    'unique_plates': set(),
                    'detection_timeline': [],
                    'confidence_scores': []
                }
    
    def render_main_header(self):
        """Render the main application header with language support"""
        
        # Create columns for layout
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            # Try to display logo
            try:
                if Path("media/fer_logo.png").exists():
                    st.image("media/fer_logo.png", width=120)
                else:
                    st.markdown("### 🤖")
            except:
                st.markdown("### 🤖")
        
        with col2:
            # Main title
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem 0;">
                <h1 style="color: #667eea; margin: 0; font-size: 2.5rem; font-weight: 700;">
                    {_('app_title')}
                </h1>
                <p style="color: #666; margin: 0.5rem 0; font-size: 1.2rem;">
                    {_('app_subtitle')}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.empty()
        
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the enhanced sidebar with all controls"""
        with st.sidebar:
            try:
                if os.path.exists("media/fer_logo.png"):
                    st.logo("media/fer_logo.png")
                else:
                    st.markdown("### 🤖 Vision Platform")
            except Exception:
                st.markdown("### 🤖 Vision Platform")
            
            # Language selector at the top - this will handle state changes
            language_manager.language_selector()
            st.markdown("---")
            
            # Model Configuration Section
            st.markdown(f"### {_('model_config')}")
            task_type, confidence = self.render_model_selection()
            
            st.markdown("---")
            
            # Source Selection
            st.markdown(f"### {_('source_selection')}")
            source_type = self.render_source_selection()
            
            st.markdown("---")
            
            # Task-specific controls
            if source_type == CAMERA:
                self.render_camera_controls()
            
            st.markdown("---")
            
            # Processing controls
            self.render_processing_controls()
            
            return task_type, confidence, source_type
    
    def render_model_selection(self):
        """Render model selection interface"""
        # Available models based on system capabilities
        
        # 1. Define the mapping of original task types to their translation keys
        model_key_map = {
            "Detection": "model_option_detection",
            "Segmentation": "model_option_segmentation",
            "Pose Estimation": "model_option_pose_estimation",
            "Enhanced License Plate": "model_option_elp",
            "Parking Detection": "model_option_parking",
            "Counting & Alert System": "model_option_counting"
        }

        # 2. Create the list of available English model names based on system capabilities
        available_models_english_names = ["Detection", "Segmentation", "Pose Estimation"]
        
        if ENHANCED_LICENSE_PLATE_AVAILABLE:
            available_models_english_names.append("Enhanced License Plate")
        if PARKING_AVAILABLE:
            available_models_english_names.append("Parking Detection")
        if COUNTING_AVAILABLE:
            available_models_english_names.append("Counting & Alert System")

        # 3. Create a list of translation keys for the selectbox options
        #    Only include keys for models that are actually available and in the map
        selectable_model_keys = [
            model_key_map[model_name] 
            for model_name in available_models_english_names 
            if model_name in model_key_map
        ]
        
        # 4. Task selection using keys
        selected_key = st.radio(
            _('task_selection'),
            selectable_model_keys, # Use the list of translation keys
            format_func=lambda key: _(key), # Translate keys for display
            key="task_type_selection_key" # Widget key
        )
        
        # 5. Map the selected key back to the original English task_type string
        task_type = ""
        for original_name, t_key in model_key_map.items():
            if t_key == selected_key:
                task_type = original_name
                break
        
        # Enhanced model information display
        self.display_model_info(task_type)
        
        # Confidence slider (not applicable for all models)
        if task_type not in ["Parking Detection"]:
            confidence = st.slider(
                _('confidence'),
                min_value=0.1,
                max_value=1.0,
                value=0.4,
                step=0.05,
                key="confidence_slider"
            )
        else:
            confidence = 0.4  # Default for parking detection
        
        return task_type, confidence
    
    def display_model_info(self, task_type):
        """Display enhanced model information"""
        if task_type == "Enhanced License Plate":
            # st.markdown(f"""
            # <div class="enhanced-license-info">
            #     <h4>🚗 {_('enhanced_license_plate')}</h4>
            #     <p>✅ {_('professional_tracking')}</p>
            #     <p>✅ {_('advanced_ocr')}</p>
            #     <p>✅ {_('real_time_visualization')}</p>
            #     <p>✅ {_('comprehensive_analytics')}</p>
            # </div>
            # """, unsafe_allow_html=True)
            pass
        
        elif task_type == "Parking Detection":
            # st.markdown(f"""
            # <div class="model-info">
            #     <h4>🅿️ {_('parking_detection')}</h4>
            #     <p><strong>{_('task')}:</strong> {_('parking_occupancy')}</p>
            #     <p><strong>{_('features')}:</strong> {_('parking_features')}</p>
            # </div>
            # """, unsafe_allow_html=True)
            pass
            
            # Parking-specific controls
            if PARKING_AVAILABLE and 'parking_config' in st.session_state:
                config = st.session_state.parking_config
                
                # PIXEL THRESHOLD CONTROL
                new_pixel_threshold = st.slider(
                    _('pixel_threshold'),
                    100, 5000, config.PIXEL_THRESHOLD, 50,
                    help=_('pixel_threshold_help')
                )
                if new_pixel_threshold != config.PIXEL_THRESHOLD:
                    config.PIXEL_THRESHOLD = new_pixel_threshold
                
                # NEW: YOLO VISUALIZATION TOGGLE
                st.markdown("---")
                st.markdown("### 🎯 Visualization Options")
                
                show_yolo = st.checkbox(
                    _('show_yolo_detections'),
                    value=config.SHOW_YOLO_DETECTIONS,
                    help=_('yolo_visualization_help')
                )
                
                if show_yolo != config.SHOW_YOLO_DETECTIONS:
                    config.SHOW_YOLO_DETECTIONS = show_yolo
                    if st.session_state.parking_detector:
                        st.session_state.parking_detector.toggle_yolo_visualization(show_yolo)
                
                # Color customization
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{_('yolo_box_color')}:**")
                    st.color_picker(_('yolo_box_color'), "#FF00FF", disabled=True, help="Magenta/Pink boxes for YOLO detections")
                
                with col2:
                    st.markdown(f"**{_('parking_spot_colors')}:**")
                    st.write(f"🟢 {_('free_spots_color')}")
                    st.write(f"🔴 {_('occupied_spots_color')}")
        
        elif task_type == "Counting & Alert System":
            # st.markdown(f"""
            # <div class="model-info">
            #     <h4>📊 {_('counting_alert_system')}</h4>
            #     <p><strong>{_('task')}:</strong> {_('motion_detection')}</p>
            #     <p><strong>{_('features')}:</strong> {_('counting_features')}</p>
            # </div>
            # """, unsafe_allow_html=True)
            pass
            
            # Shape configuration selection
            self.render_shape_configuration()
        
        else:
            # st.markdown(f"""
            # <div class="model-info">
            #     <h4>📋 {_('current_model')}: {task_type}</h4>
            #     <p><strong>{_('real_time')}:</strong> ✅ {_('enabled')}</p>
            #     <p><strong>{_('analytics')}:</strong> ✅ {_('available')}</p>
            # </div>
            # """, unsafe_allow_html=True)
            pass
    
    def render_shape_configuration(self):
        """Render shape configuration for counting system"""
        if not COUNTING_AVAILABLE:
            return
        
        # Shape file options
        shape_file_options = {
            _('zone_shapes'): ZONE_SHAPE,
            _('hall_shapes'): HALL_SHAPE,
            _('stair_shapes'): STAIR_SHAPE,
        }
        
        selected_shape_config = st.selectbox(
            _('select_shape_config'),
            list(shape_file_options.keys()),
            key="shape_config_selector",
            help=_('shape_config_help')
        )
        
        selected_shape_path = shape_file_options[selected_shape_config]
        
        # Load shapes
        if st.button(_('load_shapes'), key="load_shapes_btn"):
            shapes_loaded = self.load_counting_shapes(selected_shape_path)
            if shapes_loaded:
                st.success(f"✅ {_('loaded')} {len(shapes_loaded)} {_('shapes')}")
                st.session_state.counting_shapes_list = shapes_loaded
            else:
                st.error(f"❌ {_('failed_to_load_shapes')}")
    
    def load_counting_shapes(self, shape_path):
        """Load counting shapes from file"""
        try:
            if not shape_path.exists():
                logger.error(f"Shape file not found: {shape_path}")
                return []
            
            if shape_path.suffix.lower() == '.json':
                with open(shape_path, 'r') as f:
                    return json.load(f)
            else:
                with open(shape_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Error loading shapes: {e}")
            return []
    
    def render_source_selection(self):
        """Render source selection interface"""
        source_type = st.radio(
            _('choose_source'),
            SOURCES_LIST,
            key="source_selection_radio",
            help=_('source_selection_help')
        )
        return source_type
    
    def render_camera_controls(self):
        """Render enhanced camera control section"""
        st.markdown(f"### {_('camera_settings')}")
        
        # Camera selection with enhanced display
        camera_options = list(CAMERA_FUNCTIONS.keys())
        selected_camera = st.selectbox(
            _('select_camera'),
            camera_options,
            key="camera_selection",
            help=_('camera_selection_help')
        )
        
        # Extract camera number and display info
        camera_number = int(selected_camera.split()[-1])
        camera_ip = CAMERA_MAP.get(str(camera_number), "Unknown")
        
        st.info(f"📍 {_('selected')}: {selected_camera} (IP: {camera_ip})")
        
        # Enhanced connection controls
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button(_('connect_camera'), key="connect_btn", use_container_width=True):
                self.connect_camera(camera_number)
        
        with col2:
            if st.button(_('disconnect_camera'), key="disconnect_btn", use_container_width=True):
                self.disconnect_camera(camera_number)
        
        # Display camera status
        self.display_camera_status(camera_number)
        
        # Enhanced camera settings
        with st.expander(_('advanced_camera_settings'), expanded=False):
            self.render_advanced_camera_settings(camera_number)
    
    def render_advanced_camera_settings(self, camera_number):
        """Render advanced camera settings"""
        # Ensure settings for this camera exist in session state
        if camera_number not in st.session_state.camera_settings:
            st.session_state.camera_settings[camera_number] = {
                'connection_mode': "Auto",
                'stream_quality': "High",  # Default index 1 for ["Auto", "High", "Medium", "Low"]
                'buffer_size': 10,
                'auto_reconnect': True
            }

        current_settings = st.session_state.camera_settings[camera_number]

        col1, col2 = st.columns(2)

        with col1:
            # Get current index for connection_mode
            connection_mode_options = ["Auto", "TCP Only", "UDP Fallback", "Custom"]
            try:
                conn_mode_index = connection_mode_options.index(current_settings['connection_mode'])
            except ValueError:
                conn_mode_index = 0 # Default to Auto if not found

            current_settings['connection_mode'] = st.selectbox(
                _('connection_mode'),
                connection_mode_options,
                index=conn_mode_index,
                key=f"conn_mode_{camera_number}",
                help=_('connection_mode_help')
            )

            # Get current index for stream_quality
            stream_quality_options = ["Auto", "High", "Medium", "Low"]
            try:
                stream_quality_index = stream_quality_options.index(current_settings['stream_quality'])
            except ValueError:
                stream_quality_index = 1 # Default to High if not found

            current_settings['stream_quality'] = st.selectbox(
                _('stream_quality'),
                stream_quality_options,
                index=stream_quality_index,
                key=f"stream_quality_{camera_number}",
                help=_('stream_quality_help')
            )

        with col2:
            current_settings['buffer_size'] = st.slider(
                _('buffer_size'),
                1, 20, current_settings['buffer_size'],
                key=f"buffer_size_{camera_number}",
                help=_('buffer_size_help')
            )

            current_settings['auto_reconnect'] = st.checkbox(
                _('auto_reconnect'),
                value=current_settings['auto_reconnect'],
                key=f"auto_reconnect_{camera_number}",
                help=_('auto_reconnect_help')
            )
    
    def render_processing_controls(self):
        """Render processing control buttons"""
       # st.markdown(f"### {_('processing_options')}")
        
        # col1, col2 = st.columns(2)
        
        # with col1:
        #     if st.button(
        #         _('start_processing'), 
        #         key="start_btn", 
        #         disabled=st.session_state.processing_active,
        #         use_container_width=True,
        #         type="primary"
        #     ):
        #         st.session_state.processing_active = True
        #         st.session_state.processing_stats['start_time'] = time.time()
        #         st.success(f"🚀 {_('processing_started')}")
        #         st.rerun()
        
        # with col2:
        #     if st.button(
        #         _('stop_processing'), 
        #         key="stop_btn", 
        #         disabled=not st.session_state.processing_active,
        #         use_container_width=True
        #     ):
        #         st.session_state.processing_active = False
        #         st.info(f"⏹️ {_('processing_stopped')}")
        #         st.rerun()
        
        # # Additional controls
        # col3, col4 = st.columns(2)
        
        # with col3:
        #     if st.button(_('reset_tracking'), key="reset_btn", use_container_width=True):
        #         self.reset_tracking()
        
        # with col4:
        #     if st.button(_('save_results'), key="save_btn", use_container_width=True):
        #         self.save_results()
        pass
    
    def connect_camera(self, camera_id: int):
        """Connect to specified camera with enhanced error handling"""
        try:
            with st.spinner(_('connecting')):
                if MAIN_MODULE_AVAILABLE:
                    # Use RTSPCameraManager from main.py
                    camera_manager = RTSPCameraManager(camera_id)
                    
                    # Apply enhanced settings if available
                    if hasattr(camera_manager, 'set_connection_params'):
                        camera_manager.set_connection_params(
                            timeout=15,
                            retry_attempts=3,
                            buffer_size=10
                        )
                    
                    if camera_manager.connect():
                        camera_manager.start_capture_thread()
                        st.session_state.camera_managers[camera_id] = camera_manager
                        st.session_state.current_camera = camera_id
                        st.success(f"✅ {_('camera_connected')}: {camera_id}")
                        logger.info(f"Camera {camera_id} connected successfully")
                    else:
                        st.error(f"❌ {_('camera_connection_failed')}: {camera_id}")
                        logger.error(f"Failed to connect to camera {camera_id}")
                else:
                    # Fallback connection method
                    camera_function = CAMERA_FUNCTIONS.get(f'Camera {camera_id}')
                    if camera_function:
                        cap = camera_function()
                        if cap and cap.isOpened():
                            st.session_state.camera_managers[camera_id] = cap
                            st.session_state.current_camera = camera_id
                            st.success(f"✅ {_('camera_connected')}: {camera_id}")
                        else:
                            st.error(f"❌ {_('camera_connection_failed')}: {camera_id}")
                    else:
                        st.error(f"❌ {_('camera_not_available')}: {camera_id}")
                        
        except Exception as e:
            st.error(f"❌ {_('connection_error')}: {str(e)}")
            logger.error(f"Camera connection error: {e}")
    
    def disconnect_camera(self, camera_id: int):
        """Disconnect specified camera"""
        try:
            if camera_id in st.session_state.camera_managers:
                camera_manager = st.session_state.camera_managers[camera_id]
                
                # Enhanced cleanup for RTSPCameraManager
                if hasattr(camera_manager, 'stop'):
                    camera_manager.stop()
                elif hasattr(camera_manager, 'release'):
                    camera_manager.release()
                
                del st.session_state.camera_managers[camera_id]
                
                if st.session_state.current_camera == camera_id:
                    st.session_state.current_camera = None
                
                st.success(f"✅ {_('camera_disconnected')}: {camera_id}")
                logger.info(f"Camera {camera_id} disconnected successfully")
            else:
                st.warning(f"⚠️ {_('camera_not_connected')}: {camera_id}")
                
        except Exception as e:
            st.error(f"❌ {_('disconnection_error')}: {str(e)}")
            logger.error(f"Camera disconnection error: {e}")
    
    def display_camera_status(self, camera_id: int):
        """Display enhanced camera connection status"""
        if camera_id in st.session_state.camera_managers:
            camera_manager = st.session_state.camera_managers[camera_id]
            
            # Check if it's an enhanced camera manager
            if hasattr(camera_manager, 'connected') and camera_manager.connected:
                # Enhanced status display
                status_html = style_manager.create_status_indicator(
                    "connected", _('camera_connected')
                )
                st.markdown(status_html, unsafe_allow_html=True)
                
                # Display additional metrics for enhanced manager
                if hasattr(camera_manager, 'fps'):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(_('fps'), f"{camera_manager.fps:.1f}")
                    with col2:
                        st.metric(_('frames_captured'), 
                                getattr(camera_manager, 'frame_count', 0))
                        
            elif hasattr(camera_manager, 'isOpened') and camera_manager.isOpened():
                # Basic OpenCV VideoCapture
                status_html = style_manager.create_status_indicator(
                    "connected", _('camera_connected')
                )
                st.markdown(status_html, unsafe_allow_html=True)
            else:
                # Connection issues
                status_html = style_manager.create_status_indicator(
                    "error", _('camera_error')
                )
                st.markdown(status_html, unsafe_allow_html=True)
        else:
            # Not connected
            status_html = style_manager.create_status_indicator(
                "disconnected", _('camera_disconnected')
            )
            st.markdown(status_html, unsafe_allow_html=True)
    
    def reset_tracking(self):
        """Reset all tracking data across all systems"""
        try:
            # Reset license plate processor
            if st.session_state.license_processor:
                if hasattr(st.session_state.license_processor, 'license_plates'):
                    st.session_state.license_processor.license_plates.clear()
                if hasattr(st.session_state.license_processor, 'vehicle_history'):
                    st.session_state.license_processor.vehicle_history.clear()
            
            # Reset parking detector
            if st.session_state.parking_detector:
                for spot_id in st.session_state.parking_detector.parking_spot_states:
                    st.session_state.parking_detector.parking_spot_states[spot_id] = {
                        'object_class': None, 'entry_time': None
                    }
                st.session_state.parking_events_log.clear()
            
            # Reset counting system
            if st.session_state.counting_system:
                if hasattr(st.session_state.counting_system, 'zone_counters'):
                    st.session_state.counting_system.zone_counters.clear()
                if hasattr(st.session_state.counting_system, 'line_crossings'):
                    st.session_state.counting_system.line_crossings.clear()
                st.session_state.counting_events_history.clear()
            
            # Reset analytics
            st.session_state.detection_analytics = {
                'total_detections': 0,
                'unique_plates': set(),
                'detection_timeline': [],
                'confidence_scores': []
            }
            
            # Reset processing stats
            st.session_state.processing_stats = {
                'frames_processed': 0,
                'detections_count': 0,
                'processing_fps': 0.0,
                'start_time': time.time()
            }
            
            st.success(f"✅ {_('tracking_reset')}")
            logger.info("All tracking data reset successfully")
            
        except Exception as e:
            st.error(f"❌ {_('reset_error')}: {str(e)}")
            logger.error(f"Error resetting tracking: {e}")
    
    def save_results(self):
        """Save processing results from all active systems"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []
            
            # Save license plate results
            if (st.session_state.license_processor and 
                hasattr(st.session_state.license_processor, 'results') and
                st.session_state.license_processor.results):
                
                csv_file = st.session_state.license_processor.save_results(
                    f"license_plate_results_{timestamp}.csv"
                )
                if csv_file:
                    saved_files.append(f"{_('license_plates')}: {csv_file}")
            
            # Save parking report
            if st.session_state.parking_detector and st.session_state.parking_events_log:
                parking_file = st.session_state.parking_detector.save_parking_report(
                    f"parking_report_{timestamp}.csv"
                )
                if parking_file:
                    saved_files.append(f"{_('parking_report')}: {parking_file}")
            
            # Save counting results
            if (st.session_state.counting_system and 
                hasattr(st.session_state.counting_system, 'save_log_to_csv')):
                st.session_state.counting_system.save_log_to_csv(f"counting_log_{timestamp}")
                saved_files.append(f"{_('counting_results')}: counting_log_{timestamp}_crossings.csv")
            
            if saved_files:
                st.success(f"✅ {_('results_saved')}:")
                for file_info in saved_files:
                    st.write(f"• {file_info}")
                logger.info(f"Results saved: {saved_files}")
            else:
                st.info(f"ℹ️ {_('no_results_to_save')}")
        
        except Exception as e:
            st.error(f"❌ {_('save_error')}: {str(e)}")
            logger.error(f"Error saving results: {e}")
            
    def initialize_processors(self, task_type: str, confidence: float):
        """Initialize processing components based on task type"""
        try:
            model = None
            
            # Get device information
            device = get_pytorch_device() if MAIN_MODULE_AVAILABLE else torch.device('cpu')
            
            if task_type == "Enhanced License Plate":
                model = self.initialize_license_plate_processor(confidence, device)
                
            elif task_type == "Parking Detection":
                model = self.initialize_parking_processor(confidence, device)
                
            elif task_type == "Counting & Alert System":
                model = self.initialize_counting_processor(confidence, device)
                
            else:
                # Standard YOLO models (Detection, Segmentation, Pose)
                model = self.initialize_standard_model(task_type, confidence, device)
            
            if model is not None or task_type in ["Parking Detection", "Counting & Alert System"]:
                logger.info(f"✅ {task_type} processor initialized successfully")
                return model
            else:
                logger.error(f"❌ Failed to initialize {task_type} processor")
                return None
                
        except Exception as e:
            st.error(f"❌ {_('model_loading_error')}: {str(e)}")
            logger.error(f"Model initialization error: {e}")
            return None
    
    def initialize_license_plate_processor(self, confidence: float, device):
        """Initialize enhanced license plate processor"""
        if not ENHANCED_LICENSE_PLATE_AVAILABLE:
            st.error(f"❌ {_('enhanced_license_plate_not_available')}")
            return None
        
        try:
            if st.session_state.license_processor is None:
                with st.spinner(f"🚀 {_('loading_enhanced_license_plate')}..."):
                    st.session_state.license_processor = get_enhanced_processor_for_display(
                        str(MODEL_CONFIGS['Detection']), 
                        str(MODEL_CONFIGS['License Plate']), 
                        confidence
                    )
                    
                    # Set enhanced features flags
                    st.session_state.is_enhanced_license_plate_active = True
                    st.session_state.is_parking_detection_active = False
                    st.session_state.is_counting_alert_system_active = False
            
            if st.session_state.license_processor is None:
                st.error(f"❌ {_('failed_to_initialize_enhanced_system')}")
                return None
            
            st.sidebar.success(f"✅ {_('enhanced_system_loaded')}")
            return st.session_state.license_processor
            
        except Exception as e:
            st.error(f"❌ {_('license_plate_init_error')}: {str(e)}")
            logger.error(f"License plate processor initialization error: {e}")
            return None
    
    def initialize_parking_processor(self, confidence: float, device):
        """Initialize enhanced parking detection processor"""
        if not PARKING_AVAILABLE:
            st.error(f"❌ {_('parking_detection_not_available')}")
            return None
        
        try:
            # Initialize parking detector if not already done
            if st.session_state.parking_detector is None and 'parking_config' in st.session_state:
                st.session_state.parking_detector = ParkingSpotDetector(
                    config=st.session_state.parking_config
                )
            
            # Update confidence in config
            if 'parking_config' in st.session_state:
                st.session_state.parking_config.YOLO_CONFIDENCE = confidence
            
            # Load parking positions if available
            if not st.session_state.posList and PARKING_POSITIONS.exists():
                try:
                    with open(PARKING_POSITIONS, 'rb') as f:
                        st.session_state.posList = pickle.load(f)
                    logger.info(f"Loaded {len(st.session_state.posList)} parking positions")
                except Exception as e:
                    logger.warning(f"Could not load default parking positions: {e}")
            
            # Set flags
            st.session_state.is_parking_detection_active = True
            st.session_state.is_enhanced_license_plate_active = False
            st.session_state.is_counting_alert_system_active = False
            
            # Initialize detector's parking spots
            if st.session_state.parking_detector and st.session_state.posList:
                st.session_state.parking_detector.parking_spots = st.session_state.posList
                # Initialize spot states
                for i in range(len(st.session_state.posList)):
                    if i not in st.session_state.parking_detector.parking_spot_states:
                        st.session_state.parking_detector.parking_spot_states[i] = {
                            'object_class': None, 'entry_time': None
                        }
            
            # Enhanced: Show detailed status
            if st.session_state.posList:
                spots_count = len(st.session_state.posList)
                yolo_status = "with YOLO" if st.session_state.parking_detector.model else "motion-only"
                viz_status = "enabled" if st.session_state.parking_config.SHOW_YOLO_DETECTIONS else "disabled"
                
                st.sidebar.success(f"✅ {_('parking_system_loaded')}")
                st.sidebar.info(f"📍 {spots_count} {_('spaces')} | 🎯 YOLO {yolo_status} | 👁️ Visualization {viz_status}")
            else:
                st.sidebar.warning(f"⚠️ {_('no_parking_positions_loaded')}")
            
            return None  # Parking uses detector, not a model directly
            
        except Exception as e:
            st.error(f"❌ {_('parking_init_error')}: {str(e)}")
            logger.error(f"Parking processor initialization error: {e}")
            return None
    
    def initialize_counting_processor(self, confidence: float, device):
        """Initialize counting and alert system processor"""
        if not COUNTING_AVAILABLE:
            st.error(f"❌ {_('counting_system_not_available')}")
            return None
        
        try:
            # Initialize counting system if not already done
            if st.session_state.counting_system is None:
                yolo_model_path = str(MODEL_CONFIGS['Counting']) if MODEL_CONFIGS['Counting'].exists() else None
                
                st.session_state.counting_system = AlertCountingSystem(
                    yolo_model_path=yolo_model_path,
                    confidence_threshold=confidence,
                    target_object_classes=None,  # Detect all classes by default
                    device=str(device)
                )
                
                if yolo_model_path:
                    logger.info(f"Counting system initialized with YOLO: {MODEL_CONFIGS['Counting'].name}")
                else:
                    logger.info("Counting system initialized in motion-only mode")
            
            # Load shapes if available and not already loaded
            if not st.session_state.counting_shapes_list:
                # Try to load default shapes
                default_shapes = self.load_default_counting_shapes()
                if default_shapes:
                    st.session_state.counting_shapes_list = default_shapes
            
            # Set flags
            st.session_state.is_counting_alert_system_active = True
            st.session_state.is_parking_detection_active = False
            st.session_state.is_enhanced_license_plate_active = False
            
            shapes_count = len(st.session_state.counting_shapes_list)
            if shapes_count > 0:
                st.sidebar.success(f"✅ {_('counting_system_loaded')} ({shapes_count} {_('shapes')})")
            else:
                st.sidebar.warning(f"⚠️ {_('no_shapes_loaded')}")
            
            return None  # Counting uses alert system, not a model directly
            
        except Exception as e:
            st.error(f"❌ {_('counting_init_error')}: {str(e)}")
            logger.error(f"Counting processor initialization error: {e}")
            return None
    
    def initialize_standard_model(self, task_type: str, confidence: float, device):
        """Initialize standard YOLO models"""
        try:
            model_path = MODEL_CONFIGS.get(task_type)
            
            if not model_path or not model_path.exists():
                st.error(f"❌ {_('model_not_found')}: {model_path}")
                return None
            
            # Load YOLO model
            with st.spinner(f"🔄 {_('loading_model')} {task_type}..."):
                model = YOLO(str(model_path))
                
                # Move to appropriate device
                model.to(device)
                
                # Validate model
                if hasattr(model, 'names') and model.names:
                    logger.info(f"Model loaded successfully: {len(model.names)} classes")
                else:
                    logger.warning("Model loaded but class names not available")
            
            # Reset flags for standard models
            st.session_state.is_enhanced_license_plate_active = False
            st.session_state.is_parking_detection_active = False
            st.session_state.is_counting_alert_system_active = False
            
            st.sidebar.success(f"✅ {_('model_loaded')}: {task_type}")
            return model
            
        except Exception as e:
            st.error(f"❌ {_('standard_model_error')}: {str(e)}")
            logger.error(f"Standard model initialization error: {e}")
            return None
    
    def load_default_counting_shapes(self):
        """Load default counting shapes configuration"""
        try:
            # Try loading from JSON first
            if DEFAULT_COUNTING_SHAPES_JSON.exists():
                return self.load_counting_shapes(DEFAULT_COUNTING_SHAPES_JSON)
            
            # Try loading from zone shapes
            if ZONE_SHAPE.exists():
                return self.load_counting_shapes(ZONE_SHAPE)
            
            # Try loading from hall shapes
            if HALL_SHAPE.exists():
                return self.load_counting_shapes(HALL_SHAPE)
            
            logger.warning("No default counting shapes found")
            return []
            
        except Exception as e:
            logger.error(f"Error loading default counting shapes: {e}")
            return []
    
    def validate_model_requirements(self, task_type: str):
        """Validate that all requirements are met for the selected task"""
        requirements_met = True
        missing_components = []
        
        if task_type == "Enhanced License Plate":
            if not ENHANCED_LICENSE_PLATE_AVAILABLE:
                requirements_met = False
                if not LICENSE_PLATE_UTILS_AVAILABLE:
                    missing_components.append("EasyOCR and utils.py")
                if not SORT_AVAILABLE:
                    missing_components.append("SORT tracker")
                if not MODEL_CONFIGS['License Plate'].exists():
                    missing_components.append("License plate model")
                if not MODEL_CONFIGS['Detection'].exists():
                    missing_components.append("Vehicle detection model")
        
        elif task_type == "Parking Detection":
            if not PARKING_AVAILABLE:
                requirements_met = False
                missing_components.append("Parking detection modules")
            if not MODEL_CONFIGS['Parking'].exists():
                missing_components.append("YOLO model for parking")
        
        elif task_type == "Counting & Alert System":
            if not COUNTING_AVAILABLE:
                requirements_met = False
                missing_components.append("Counting system modules")
            if not MODEL_CONFIGS['Counting'].exists():
                missing_components.append("YOLO model for counting")
        
        else:
            # Standard models
            model_path = MODEL_CONFIGS.get(task_type)
            if not model_path or not model_path.exists():
                requirements_met = False
                missing_components.append(f"{task_type} model file")
        
        if not requirements_met:
            st.sidebar.error(f"❌ {_('missing_requirements')}")
            st.sidebar.write(f"**{_('missing_components')}:**")
            for component in missing_components:
                st.sidebar.write(f"• {component}")
        
        return requirements_met, missing_components
    
    def get_device_info(self):
        """Get and display device information"""
        try:
            if MAIN_MODULE_AVAILABLE:
                device = get_pytorch_device()
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Display device info in sidebar
            device_info = f"🔥 {_('processing_device')}: {device}"
            if torch.cuda.is_available() and device.type == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                device_info += f" ({gpu_name})"
            
            st.sidebar.info(device_info)
            return device
            
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return torch.device('cpu')
    
    def optimize_model_performance(self, model, device):
        """Optimize model performance based on device and settings"""
        try:
            if model is None:
                return
            
            # Device-specific optimizations
            if device.type == 'cuda':
                # GPU optimizations
                if hasattr(model, 'half'):
                    model.half()  # Use FP16 for better performance
                
                # Enable optimized attention if available
                torch.backends.cudnn.benchmark = True
                
            elif device.type == 'cpu':
                # CPU optimizations
                if hasattr(model, 'float'):
                    model.float()  # Ensure FP32 for CPU
                
                # Set number of threads for CPU processing
                torch.set_num_threads(4)
            
            logger.info(f"Model optimized for {device}")
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}")
            
    def process_single_image(self, image: Image.Image, model, task_type: str, confidence: float):
        """Process a single image with the specified model and task"""
        try:
            # Convert PIL to OpenCV format
            image_array = np.array(image)
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Process based on task type
            processed_frame, results = self.process_single_frame(
                image_array, model, task_type, confidence
            )
            
            # Convert back to RGB for display
            if processed_frame is not None and len(processed_frame.shape) == 3:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            return processed_frame, results
            
        except Exception as e:
            st.error(f"❌ {_('image_processing_error')}: {str(e)}")
            logger.error(f"Image processing error: {e}")
            return None, {'error': str(e)}
    
    def process_single_frame(self, frame: np.ndarray, model, task_type: str, confidence: float) -> Tuple[np.ndarray, Dict]:
        """Process a single frame based on task type"""
        try:
            results = {}
            processed_frame = frame.copy()
            
            if task_type == "Enhanced License Plate" and st.session_state.license_processor:
                # Enhanced license plate processing
                processed_frame, results = process_frame_for_display(
                    frame, st.session_state.license_processor
                )
                
                # Store frame data for final results generation
                if not hasattr(st.session_state, 'processed_frames_data'):
                    st.session_state.processed_frames_data = {}
                
                if results.get('frame_number'):
                    st.session_state.processed_frames_data[results['frame_number']] = frame.copy()
                    
                    # Keep only last 100 frames to manage memory
                    if len(st.session_state.processed_frames_data) > 100:
                        oldest_frame = min(st.session_state.processed_frames_data.keys())
                        del st.session_state.processed_frames_data[oldest_frame]
                
                # Update analytics
                self.update_detection_analytics(results)
                
            elif task_type == "Parking Detection" and st.session_state.parking_detector:
                # Parking detection processing
                processed_frame, results = self.process_parking_frame(frame)
                
            elif task_type == "Counting & Alert System" and st.session_state.counting_system:
                # Counting and alert system processing
                processed_frame, results = self.process_counting_frame(frame)
                
            else:
                # Standard YOLO processing
                processed_frame, results = self.process_standard_frame(frame, model, confidence)
            
            # Update processing statistics
            self.update_processing_stats(results)
            
            return processed_frame, results
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, {'error': str(e)}
    
    def process_parking_frame(self, frame: np.ndarray, parking_detector_instance=None) -> Tuple[np.ndarray, Dict]:
        """Process frame for parking detection with enhanced YOLO visualization, using provided instance."""
        try:
            detector = parking_detector_instance
            if not detector:
                # Fallback or error if not provided, though in LiveVideoProcessor context it should be.
                if 'parking_detector' in st.session_state and st.session_state.parking_detector:
                    logger.warning("process_parking_frame called without instance, falling back to session_state.parking_detector")
                    detector = st.session_state.parking_detector
                else:
                    error_msg = "Parking detector instance not provided and not in session_state"
                    logger.error(error_msg)
                    cv2.putText(frame, error_msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    return frame.copy(), {'error': error_msg}

            if not detector.parking_spots:
                # No parking spots loaded
                processed_frame = frame.copy()
                cv2.putText(processed_frame, _('upload_parking_positions'), 
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return processed_frame, {'error': 'no_parking_spots'}
            
            # Preprocess image for parking detection
            processed_img = detector.preprocess_image(frame)
            if processed_img is None:
                processed_frame = frame.copy()
                cv2.putText(processed_frame, _('preprocessing_error'), 
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return processed_frame, {'error': 'preprocessing_failed'}
            
            # Detect objects with YOLO (this now returns ALL detections, not just vehicles)
            yolo_detections = detector.detect_objects_yolo(frame)
            
            # Check parking spaces (this now draws both YOLO detections AND parking analysis)
            processed_frame, parking_stats = detector.check_parking_spaces(
                frame, processed_img, yolo_detections
            )
            
            # Add enhanced summary overlay
            if CVZONE_AVAILABLE:
                try:
                    # Main parking summary
                    summary_text = f"Free: {parking_stats.get('free_spaces', 0)}/{parking_stats.get('total_spaces', 0)}"
                    cvzone.putTextRect(processed_frame, summary_text, 
                                    (10, processed_frame.shape[0] - 60), 
                                    scale=1, thickness=2, colorR=(0, 255, 0), offset=10)
                    
                    # YOLO detections summary (if enabled)
                    if detector.config.SHOW_YOLO_DETECTIONS and yolo_detections:
                        yolo_summary = f"YOLO: {len(yolo_detections)} detections"
                        cvzone.putTextRect(processed_frame, yolo_summary, 
                                        (10, processed_frame.shape[0] - 100), 
                                        scale=0.8, thickness=2, colorR=(255, 0, 255), offset=8)
                except:
                    # Fallback to regular OpenCV text
                    summary_text = f"Free: {parking_stats.get('free_spaces', 0)}/{parking_stats.get('total_spaces', 0)}"
                    cv2.putText(processed_frame, summary_text, 
                            (10, processed_frame.shape[0] - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    if detector.config.SHOW_YOLO_DETECTIONS and yolo_detections:
                        yolo_summary = f"YOLO: {len(yolo_detections)} detections"
                        cv2.putText(processed_frame, yolo_summary, 
                                (10, processed_frame.shape[0] - 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Enhanced results dictionary
            results = {
                'parking_stats': parking_stats,
                'model_type': 'Parking Detection',
                'free_spaces': parking_stats.get('free_spaces', 0),
                'occupied_spaces': parking_stats.get('occupied_spaces', 0),
                'total_spaces': parking_stats.get('total_spaces', 0),
                'yolo_detections_count': len(yolo_detections),
                'yolo_visualization_enabled': detector.config.SHOW_YOLO_DETECTIONS,
                'detection_details': [
                    {
                        'class': det['class_name'],
                        'confidence': det['confidence'],
                        'bbox': det['bbox']
                    } for det in yolo_detections
                ] if yolo_detections else []
            }
            
            return processed_frame, results
            
        except Exception as e:
            logger.error(f"Parking frame processing error: {e}")
            processed_frame = frame.copy()
            cv2.putText(processed_frame, f"{_('parking_error')}: {str(e)[:50]}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return processed_frame, {'error': str(e)}
    
    def process_counting_frame(self, frame: np.ndarray, counting_system_instance=None, shapes_list_param=None) -> Tuple[np.ndarray, Dict]:
        """Process frame for counting and alert system, using provided instance and shapes list."""
        try:
            active_counting_system = counting_system_instance
            if not active_counting_system:
                if 'counting_system' in st.session_state and st.session_state.counting_system:
                    logger.warning("process_counting_frame called without instance, falling back to session_state.counting_system")
                    active_counting_system = st.session_state.counting_system
                else:
                    error_msg = "Counting System instance not provided and not in session_state"
                    logger.error(error_msg)
                    cv2.putText(frame, error_msg, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                    return frame.copy(), {'error': error_msg}

            # Prioritize shapes_list_param, then fallback to session_state, then error
            active_shapes_list = shapes_list_param
            if not active_shapes_list: # Check if the passed list is None or empty
                logger.debug("shapes_list_param not provided or empty for counting, trying st.session_state.counting_shapes_list")
                active_shapes_list = st.session_state.get("counting_shapes_list") # Safely get from session_state

            if not active_shapes_list: # If still no shapes after checking param and session_state
                processed_frame = frame.copy()
                cv2.putText(processed_frame, _('upload_shapes_file_or_ensure_initialized'), 
                           (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                logger.warning("No shapes available for counting system (neither passed nor in session state).")
                return processed_frame, {'error': 'no_shapes_available'}
            
            # Preprocess frame for motion detection
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
            imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 25, 16)
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
            
            # Process zones and lines
            processed_frame = frame.copy()
            active_counting_system.process_zones_and_lines(imgDilate, processed_frame, active_shapes_list) # Use active_shapes_list
            
            # Get summary statistics
            counting_summary = active_counting_system.get_summary_stats()
            
            # Add summary overlay
            if CVZONE_AVAILABLE:
                try:
                    summary_text = f"{_('alerts')}: {counting_summary.get('active_alerts', 0)} | {_('crossings')}: {counting_summary.get('total_crossings', 0)}"
                    cvzone.putTextRect(processed_frame, summary_text, 
                                     (10, processed_frame.shape[0] - 60), 
                                     scale=0.8, thickness=2, colorR=(0, 200, 0), offset=10)
                except:
                    # Fallback to regular OpenCV text
                    cv2.putText(processed_frame, summary_text, 
                               (10, processed_frame.shape[0] - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            
            results = {
                'counting_summary': counting_summary,
                'model_type': 'Counting & Alert System',
                'active_alerts': counting_summary.get('active_alerts', 0),
                'total_crossings': counting_summary.get('total_crossings', 0),
                'total_zones': counting_summary.get('total_zones', 0)
            }
            
            return processed_frame, results
            
        except Exception as e:
            logger.error(f"Counting frame processing error: {e}")
            processed_frame = frame.copy()
            cv2.putText(processed_frame, f"{_('counting_error')}: {str(e)[:50]}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return processed_frame, {'error': str(e)}
    
    def process_standard_frame(self, frame: np.ndarray, model, confidence: float) -> Tuple[np.ndarray, Dict]:
        """Process frame with standard YOLO models"""
        try:
            if model is None:
                processed_frame = frame.copy()
                cv2.putText(processed_frame, _('no_model_loaded'), 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                return processed_frame, {'error': 'no_model'}
            
            # Run YOLO prediction
            results = model.predict(frame, conf=confidence, verbose=False)
            
            # Plot results
            if results and len(results) > 0 and results[0].boxes is not None:
                processed_frame = results[0].plot()
                detection_count = len(results[0].boxes)
                
                # Extract class information
                classes = []
                confidences = []
                if hasattr(results[0].boxes, 'cls') and hasattr(results[0].boxes, 'conf'):
                    for cls_id, conf in zip(results[0].boxes.cls, results[0].boxes.conf):
                        class_name = model.names.get(int(cls_id), f"Class_{int(cls_id)}")
                        classes.append(class_name)
                        confidences.append(float(conf))
                
                results_data = {
                    'detections_count': detection_count,
                    'classes': classes,
                    'confidences': confidences,
                    'model_type': 'Standard Detection'
                }
            else:
                processed_frame = frame.copy()
                results_data = {
                    'detections_count': 0,
                    'classes': [],
                    'confidences': [],
                    'model_type': 'Standard Detection'
                }
            
            return processed_frame, results_data
            
        except Exception as e:
            logger.error(f"Standard frame processing error: {e}")
            processed_frame = frame.copy()
            cv2.putText(processed_frame, f"{_('processing_error')}: {str(e)[:50]}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return processed_frame, {'error': str(e)}
    
    def process_video_file(self, video_path: Path, model, task_type: str, confidence: float, save_output_video: bool = False):
        """Process entire video file with progress tracking and optional saving."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                st.error(f"❌ {_('video_open_error')}: {video_path}")
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps == 0: # Handle cases where FPS might not be read correctly
                logger.warning(f"FPS for video {video_path} reported as 0. Defaulting to 25 FPS for VideoWriter.")
                fps = 25.0
            
            st.info(f"📹 {_('processing_video')}: {video_path.name} ({total_frames} {_('frames')}, {fps:.1f} FPS, {width}x{height})")
            
            # Create progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_display = st.empty()
            
            # Processing loop
            frame_count = 0
            start_time = time.time()
            
            video_writer = None
            output_video_path_str = ""

            if save_output_video:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitize task_type for filename
                safe_task_type = "".join(c if c.isalnum() else "_" for c in task_type)
                output_filename = f"processed_{video_path.stem}_{safe_task_type}_{timestamp}.mp4"
                
                # Ensure VIDEO_DIR exists or use a local path
                save_directory = VIDEO_DIR
                if not save_directory.exists():
                    logger.warning(f"VIDEO_DIR {save_directory} does not exist. Saving to current directory.")
                    save_directory = Path(".") # Save to current working directory as fallback
                
                output_video_path = save_directory / output_filename
                output_video_path_str = str(output_video_path)
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Common and widely supported
                try:
                    video_writer = cv2.VideoWriter(output_video_path_str, fourcc, fps, (width, height))
                    if not video_writer.isOpened():
                        st.error(f"❌ {_('video_writer_error_check_codec')} {output_video_path_str}")
                        logger.error(f"Failed to open VideoWriter for {output_video_path_str}. Check codec (mp4v), FPS ({fps}), dimensions ({width}x{height}).")
                        video_writer = None # Ensure it's None if opening failed
                    else:
                        logger.info(f"✅ VideoWriter initialized for {output_video_path_str}")
                except Exception as e_vw:
                    st.error(f"❌ {_('video_writer_exception')}: {str(e_vw)}")
                    logger.error(f"Exception initializing VideoWriter: {e_vw}")
                    video_writer = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, results = self.process_single_frame(
                    frame, model, task_type, confidence
                )
                
                # Write to video if applicable and writer is open
                if video_writer is not None and video_writer.isOpened() and processed_frame is not None:
                    try:
                        video_writer.write(processed_frame)
                    except Exception as e_write:
                        logger.error(f"Error writing frame to video: {e_write}")
                        # Optionally stop trying to write if errors persist
                        # video_writer.release()
                        # video_writer = None
                        # st.warning("Video writing failed, stopping video save for this session.")
                
                # Display frame (every Nth frame to avoid overwhelming Streamlit)
                display_interval = max(1, int(fps / 5)) # Aim for around 5 FPS display in Streamlit
                if frame_count % display_interval == 0 and processed_frame is not None:
                    try:
                        display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_display.image(display_frame, use_container_width=True)
                    except Exception as e_display:
                        logger.error(f"Error converting/displaying frame in Streamlit: {e_display}")

                
                # Update progress
                frame_count += 1
                if total_frames > 0:
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                
                # Calculate processing speed
                elapsed_time = time.time() - start_time
                processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                eta_seconds = (total_frames - frame_count) / processing_fps if processing_fps > 0 and total_frames > 0 else 0
                
                status_text.text(
                    f"{_('processing')}: {frame_count}/{total_frames if total_frames > 0 else '?'} "
                    f"({progress*100:.1f}% if total_frames > 0 else '') - {_('speed')}: {processing_fps:.1f} FPS - "
                    f"ETA: {timedelta(seconds=int(eta_seconds))}"
                )
                
                # Update session analytics
                if results and not results.get('error'):
                    self.update_detection_analytics(results)
            
            # Cleanup
            cap.release()
            if video_writer is not None and video_writer.isOpened():
                video_writer.release()
                st.success(f"✅ {_('video_saved')}: {output_video_path_str}")
                logger.info(f"Processed video saved to {output_video_path_str}")
            elif save_output_video and video_writer is None: # If saving was intended but failed
                 st.warning(f"⚠️ {_('video_save_failed_check_logs')}")


            status_text.text(f"✅ {_('video_processing_complete')}")
            
            # Display final statistics
            self.display_video_processing_results(frame_count, elapsed_time, task_type)
            
        except Exception as e:
            st.error(f"❌ {_('video_processing_error')}: {str(e)}")
            logger.error(f"Video processing error for {video_path}: {traceback.format_exc()}")
            if 'video_writer' in locals() and video_writer is not None and video_writer.isOpened():
                video_writer.release()
    
    def display_video_processing_results(self, frame_count: int, elapsed_time: float, task_type: str):
        """Display video processing results summary"""
        try:
            avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(_('frames_processed'), f"{frame_count:,}")
                st.metric(_('processing_time'), f"{elapsed_time:.1f}s")
            
            with col2:
                st.metric(_('avg_processing_fps'), f"{avg_fps:.1f}")
                st.metric(_('total_detections'), st.session_state.detection_analytics['total_detections'])
            
            with col3:
                if task_type == "Enhanced License Plate":
                    unique_plates = len(st.session_state.detection_analytics['unique_plates'])
                    st.metric(_('unique_plates'), unique_plates)
                elif task_type == "Parking Detection":
                    parking_events = len(st.session_state.parking_events_log)
                    st.metric(_('parking_events'), parking_events)
                elif task_type == "Counting & Alert System":
                    counting_events = len(st.session_state.counting_events_history)
                    st.metric(_('counting_events'), counting_events)
            
        except Exception as e:
            logger.error(f"Error displaying video results: {e}")
    
    def update_detection_analytics(self, results: Dict):
        """Update detection analytics based on processing results"""
        try:
            # Ensure detection_analytics exists
            if 'detection_analytics' not in st.session_state:
                st.session_state.detection_analytics = {
                    'total_detections': 0,
                    'unique_plates': set(),
                    'detection_timeline': [],
                    'confidence_scores': []
                }
            
            current_time = time.time()
            
            # Enhanced License Plate analytics
            if results.get('model_type') == 'Enhanced License Plate' and results.get('results'):
                for vehicle_id, vehicle_data in results['results'].items():
                    if 'license_plate' in vehicle_data:
                        plate_text = vehicle_data['license_plate'].get('text', '')
                        confidence = vehicle_data['license_plate'].get('text_score', 0)
                        
                        if plate_text and confidence > 0:
                            st.session_state.detection_analytics['unique_plates'].add(plate_text)
                            st.session_state.detection_analytics['confidence_scores'].append(confidence)
                            st.session_state.detection_analytics['detection_timeline'].append({
                                'timestamp': current_time,
                                'plate': plate_text,
                                'confidence': confidence,
                                'vehicle_id': vehicle_id
                            })
                            
                            st.session_state.detection_analytics['total_detections'] += 1
            
            # Standard detection analytics
            elif results.get('detections_count', 0) > 0:
                st.session_state.detection_analytics['total_detections'] += results['detections_count']
                
                if results.get('confidences'):
                    st.session_state.detection_analytics['confidence_scores'].extend(results['confidences'])
            
            # Keep analytics manageable (last 1000 entries)
            if len(st.session_state.detection_analytics['detection_timeline']) > 1000:
                st.session_state.detection_analytics['detection_timeline'] = \
                    st.session_state.detection_analytics['detection_timeline'][-500:]
            
            if len(st.session_state.detection_analytics['confidence_scores']) > 1000:
                st.session_state.detection_analytics['confidence_scores'] = \
                    st.session_state.detection_analytics['confidence_scores'][-500:]
                    
        except Exception as e:
            logger.error(f"Error updating analytics: {e}")
            # Reinitialize analytics if there's an error
            st.session_state.detection_analytics = {
                'total_detections': 0,
                'unique_plates': set(),
                'detection_timeline': [],
                'confidence_scores': []
            }
    
    def update_processing_stats(self, results: Dict):
        """Update processing statistics"""
        try:
            stats = st.session_state.processing_stats
            stats['frames_processed'] += 1
            
            if results and not results.get('error'):
                if 'detections_count' in results:
                    stats['detections_count'] += results['detections_count']
                elif results.get('model_type') == 'Enhanced License Plate' and results.get('plates_read'):
                    stats['detections_count'] += results['plates_read']
            
            # Calculate FPS
            if stats['start_time']:
                elapsed = time.time() - stats['start_time']
                stats['processing_fps'] = stats['frames_processed'] / elapsed if elapsed > 0 else 0
                
        except Exception as e:
            logger.error(f"Error updating processing stats: {e}")
            
    def create_main_interface(self, task_type: str, confidence: float, source_type: str):
        """Create the main interface with tab system"""
        
        # Initialize processors
        model = self.initialize_processors(task_type, confidence)
        
        if model is None and task_type not in ["Parking Detection", "Counting & Alert System"]:
            st.error(f"❌ {_('model_initialization_failed')}")
            return
        
        # Create enhanced tab system based on task type
        if task_type == "Enhanced License Plate":
            tabs = st.tabs([
                f"🚗 {_('enhanced_live_detection')}",
                f"📊 {_('real_time_analytics')}", 
                f"📋 {_('license_plate_results')}",
                f"📈 {_('reports_export')}",
                f"⚙️ {_('system_management')}"
            ])
        else:
            tabs = st.tabs([
                f"📹 {_('live_detection')}",
                f"📊 {_('analytics_dashboard')}", 
                f"📋 {_('detection_results')}",
                f"📈 {_('reports_export')}"
            ])
        
        # Tab 1: Main Detection Interface
        with tabs[0]:
            self.render_detection_tab(source_type, model, task_type, confidence)
        
        # Tab 2: Analytics Dashboard
        with tabs[1]:
            self.render_analytics_tab(task_type)
        
        # Tab 3: Results Management
        with tabs[2]:
            self.render_results_tab(task_type)
        
        # Tab 4: Reports and Export
        with tabs[3]:
            self.render_reports_tab(task_type)
        
        # Tab 5: System Management (for enhanced license plate)
        if len(tabs) > 4:
            with tabs[4]:
                self.render_system_management_tab()
    
    def render_detection_tab(self, source_type: str, model, task_type: str, confidence: float):
        """Render the main detection tab"""
        st.header(f" {_('detection_interface')} - {task_type}")
        
        # if source_type == IMAGE: # Commented out image interface rendering
        #     self.render_image_interface(model, task_type, confidence)
        if source_type == VIDEO:
            self.render_video_interface(model, task_type, confidence)
        elif source_type == CAMERA:
            self.render_camera_interface(model, task_type, confidence)
    
    # def render_image_interface(self, model, task_type: str, confidence: float): # Commented out entire method
    #     """Render image processing interface"""
    #     st.subheader(f"🖼️ {_('image_analysis')}")
    #     
    #     # File uploader
    #     uploaded_file = st.file_uploader(
    #         _('upload_image'),
    #         type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
    #         help=_('upload_image_help')
    #     )
    #     
    #     col1, col2 = st.columns(2)
    #     
    #     with col1:
    #         st.markdown(f"### {_('original_image')}")
    #         if uploaded_file is not None:
    #             image = Image.open(uploaded_file)
    #             st.image(image, use_container_width=True)
    #         else:
    #             if DEFAULT_IMAGE.exists():
    #                 st.image(str(DEFAULT_IMAGE), caption=_('default_image'), use_container_width=True)
    #             else:
    #                 st.info(f"📁 {_('upload_image_prompt')}")
    #     
    #     with col2:
    #         st.markdown(f"### {_('processed_result')}")
    #         if uploaded_file is not None:
    #             if st.button(f"🚀 {_('analyze_image')}", type="primary", use_container_width=True):
    #                 with st.spinner(f"🔍 {_('processing_with')} {task_type}..."):
    #                     processed_image, results = self.process_single_image(
    #                         image, model, task_type, confidence
    #                     )
    #                     
    #                     if processed_image is not None:
    #                         st.image(processed_image, use_container_width=True)
    #                         
    #                         # Display task-specific results
    #                         self.display_image_results(results, task_type)
    #                     else:
    #                         st.error(f"❌ {_('image_processing_failed')}")
    #         else:
    #             st.info(f"📁 {_('upload_image_to_analyze')}")
    
    def render_video_interface(self, model, task_type: str, confidence: float):
        """Render video processing interface"""
        st.subheader(f"🎬 {_('video_analysis')}")
        
        # Video selection
        selected_video = st.selectbox(
            _('choose_video'),
            list(VIDEOS_DICT.keys()),
            help=_('choose_video_help')
        )
        
        video_path = VIDEOS_DICT[selected_video]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Video preview
            if video_path.exists():
                with open(video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.error(f"❌ {_('video_not_found')}: {video_path}")
        
        with col2:
            #st.markdown(f"### 🎛️ {_('processing_options')}")
            
            # Consistent Save Video Checkbox for all video tasks
            save_video_checkbox_state = True # Default to True, user can uncheck
            if task_type == "Enhanced License Plate":
                st.markdown(f"""
                <div class="enhanced-license-info">
                    <h4>🚗 {_('enhanced_license_plate')}</h4>
                    <p>✅ {_('professional_visualization')}</p>
                    <p>✅ {_('real_time_ocr')}</p>
                    <p>✅ {_('advanced_tracking')}</p>
                </div>
                """, unsafe_allow_html=True)
            elif task_type == "Parking Detection":
                st.info(f"🅿️ {_('parking_detection_mode')}")
                if not st.session_state.posList:
                    st.warning(f"⚠️ {_('no_parking_positions')}")
            elif task_type == "Counting & Alert System":
                st.info(f"📊 {_('counting_alert_mode')}")
                if not st.session_state.counting_shapes_list:
                    st.warning(f"⚠️ {_('no_shapes_loaded')}")
            else: # Standard detection, segmentation, pose
                st.info(f"🎯 {_(task_type.lower().replace(' ', '_'))} {_('mode')}") # Generic task type display

            save_video = st.checkbox(f"💾 {_('save_processed_video')}", value=save_video_checkbox_state, key=f"save_video_{task_type}")

        # Processing button
        if st.button(f"🚀 {_('process_video')}", type="primary", use_container_width=True):
            if video_path.exists():
                self.process_video_file(video_path, model, task_type, confidence, save_output_video=save_video)
            else:
                st.error(f"❌ {_('video_not_found_cannot_process')}: {video_path}")
    
    def render_camera_interface(self, model, task_type: str, confidence: float):
        """Render live camera processing interface - HYBRID APPROACH"""
        
        # Check if we're in cloud environment
        if self.is_cloud_deployment():
            # Use cloud-friendly interface
            self.render_camera_interface_cloud_friendly(model, task_type, confidence)
            return
        
        st.subheader(f"📡 {_('live_camera_analysis')}")
        
        # Check camera connection
        if not st.session_state.camera_managers:
            st.warning(f"⚠️ {_('no_camera_connected')}")
            st.info(f"💡 {_('connect_camera_sidebar')}")
            
            # Show available cameras
            st.markdown(f"### 📹 {_('available_cameras')}")
            for i, (camera_name, camera_ip) in enumerate(CAMERA_MAP.items(), 1):
                st.write(f"• **{_('camera')} {camera_name}:** {camera_ip}")
            return
        
        camera_id = st.session_state.current_camera
        if not camera_id or camera_id not in st.session_state.camera_managers:
            st.info(f"📹 {_('select_camera_to_start')}")
            return
        
        # Main interface layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### 🎬 OpenCV Live Processing Window")
            
            # Control buttons
            button_col1, button_col2, button_col3 = st.columns(3)
            
            with button_col1:
                start_btn = st.button(
                    f"▶️ Start Live Processing", 
                    key=f"start_opencv_{camera_id}",
                    disabled=camera_id in self.live_processors,
                    use_container_width=True,
                    type="primary"
                )
            
            with button_col2:
                stop_btn = st.button(
                    f"⏹️ Stop Processing", 
                    key=f"stop_opencv_{camera_id}",
                    disabled=camera_id not in self.live_processors,
                    use_container_width=True
                )
            
            with button_col3:
                reset_btn = st.button(
                    f"🔄 Reset Tracking", 
                    key=f"reset_opencv_{camera_id}",
                    use_container_width=True
                )
            
            # Handle button actions
            if start_btn:
                # Retrieve the appropriate specialized processor from session state
                # This assumes initialize_processors has already run and populated these.
                active_general_model = model # The model from sidebar (e.g., yolo11n.pt)
                specialized_processor_instance = None

                if task_type == "Enhanced License Plate":
                    specialized_processor_instance = st.session_state.get('license_processor')
                    active_general_model = None # ELP uses its own internal models via the processor
                elif task_type == "Parking Detection":
                    specialized_processor_instance = st.session_state.get('parking_detector')
                    active_general_model = None # Parking uses its own YOLO via the detector
                elif task_type == "Counting & Alert System":
                    specialized_processor_instance = st.session_state.get('counting_system')
                    # Counting might use a general model if not configured with its own,
                    # or if specialized_processor_instance.yolo_model is None.
                    # The LiveVideoProcessor's _process_frame will handle this logic.
                    # If counting_system_instance has its own YOLO, active_general_model could be None.
                    if specialized_processor_instance and hasattr(specialized_processor_instance, 'yolo_model') and specialized_processor_instance.yolo_model:
                        active_general_model = None


                # Check if the required specialized processor is available
                if task_type in ["Enhanced License Plate", "Parking Detection", "Counting & Alert System"] and not specialized_processor_instance:
                    st.error(f"Critical Error: {task_type} processor not found in session state. It might not have been initialized. Please re-select the task or check logs.")
                    logger.error(f"Attempted to start LiveVideoProcessor for {task_type}, but specialized_processor_instance is None.")
                    return # Stop if specialized processor is required but missing

                # Create and start the LiveVideoProcessor thread
                processor_thread = LiveVideoProcessor(
                    camera_id, 
                    active_general_model, # Pass the general model
                    task_type, 
                    confidence, 
                    self,  # Pass the VisionPlatform instance
                    specialized_processor_instance, # Pass the specific processor instance
                    st.session_state.get('counting_shapes_list', []) # Pass shapes_list
                )
                
                if processor_thread.start_processing():
                    self.live_processors[camera_id] = processor_thread
                    st.success(f"🚀 Live processing started in OpenCV window for {task_type}!")
                    st.info("💡 Check the OpenCV window for live video. Use Q to quit, S for stats, R to reset.")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to start processing for camera {camera_id}")
            
            if stop_btn and camera_id in self.live_processors:
                self.live_processors[camera_id].stop_processing()
                del self.live_processors[camera_id]
                st.info(f"⏹️ Live processing stopped")
                
                # Auto-save results
                if task_type == "Enhanced License Plate" and st.session_state.license_processor:
                    self.auto_save_live_results()
                
                st.rerun()
            
            if reset_btn:
                self.reset_tracking()
                st.success(f"✅ Tracking data reset")
            
            # Status display
            if camera_id in self.live_processors:
                processor = self.live_processors[camera_id]
                stats = processor.get_stats()
                
                if stats:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4>🟢 Live Processing Active</h4>
                        <p><strong>Frames Processed:</strong> {stats['frame_count']:,}</p>
                        <p><strong>Processing FPS:</strong> {stats['fps']:.1f}</p>
                        <p><strong>Total Detections:</strong> {stats['detection_count']:,}</p>
                        <p><strong>Elapsed Time:</strong> {stats['elapsed_time']:.1f}s</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("🟡 Processing starting...")
            else:
                st.info("🔴 No active processing. Click 'Start Live Processing' to begin.")
                
                # Show instructions
                st.markdown(f"""
                ### 📋 Instructions:
                1. Click **"Start Live Processing"** to open OpenCV window
                2. The video will appear in a separate window
                3. Use keyboard controls in the OpenCV window:
                - **Q**: Quit processing
                - **S**: Show statistics in console  
                - **R**: Reset tracking data
                4. Return to this Streamlit interface for analytics and export
                """)
        
        with col2:
            # Task-specific information and controls
            st.markdown(f"### 🎛️ {_('live_controls')}")
            
            camera_ip = CAMERA_MAP.get(str(camera_id), "Unknown")
            st.info(f"📍 Active: Camera {camera_id} ({camera_ip})")
            st.info(f"🎯 Task: {task_type}")
            
            # Task-specific static info
            if task_type == "Enhanced License Plate":
                self.render_license_plate_info_static()
            elif task_type == "Parking Detection":
                self.render_parking_info_static()
            elif task_type == "Counting & Alert System":
                self.render_counting_info_static()
            
            # Session analytics (updated when page refreshes)
            if st.session_state.processing_stats['frames_processed'] > 0:
                st.markdown(f"### 📊 Session Analytics")
                
                session_stats = st.session_state.processing_stats
                
                st.metric("Session Frames", f"{session_stats['frames_processed']:,}")
                st.metric("Session Detections", f"{session_stats['detections_count']:,}")
                
                if task_type == "Enhanced License Plate":
                    unique_plates = len(st.session_state.detection_analytics['unique_plates'])
                    st.metric("Unique Plates", unique_plates)
            
            # Refresh button for analytics
            if st.button("🔄 Refresh Analytics", use_container_width=True):
                st.rerun()
    
    def render_live_camera_feed(self, camera_id: int, model, task_type: str, confidence: float):
        """Render live camera feed with processing - FIXED VERSION"""
        camera_manager = st.session_state.camera_managers[camera_id]
        
        # Create display containers
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Create control buttons OUTSIDE the refresh loop
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_btn = st.button(
                f"▶️ {_('start_live')}", 
                key=f"start_live_{camera_id}",
                disabled=st.session_state.processing_active,
                use_container_width=True,
                type="primary"
            )
        
        with col2:
            stop_btn = st.button(
                f"⏹️ {_('stop_live')}", 
                key=f"stop_live_{camera_id}",
                disabled=not st.session_state.processing_active,
                use_container_width=True
            )
        
        with col3:
            reset_btn = st.button(
                f"🔄 {_('reset_tracking')}", 
                key=f"reset_tracking_{camera_id}",
                use_container_width=True
            )
        
        # Handle button clicks
        if start_btn:
            st.session_state.processing_active = True
            st.session_state.processing_stats['start_time'] = time.time()
            st.success(f"🚀 {_('live_processing_started')}")
            # Don't rerun here, let the processing loop handle it
        
        if stop_btn:
            st.session_state.processing_active = False
            # Auto-save for enhanced license plate
            if task_type == "Enhanced License Plate" and st.session_state.license_processor:
                self.auto_save_live_results()
            st.info(f"⏹️ {_('live_processing_stopped')}")
            # Don't rerun here
        
        if reset_btn:
            self.reset_tracking()
            st.success(f"✅ {_('tracking_reset')}")
        
        # Processing display logic - ONLY refresh content, not the whole page
        if st.session_state.processing_active:
            # Initialize frame counter if not exists
            if f'frame_counter_{camera_id}' not in st.session_state:
                st.session_state[f'frame_counter_{camera_id}'] = 0
            
            try:
                # Get frame from camera
                if hasattr(camera_manager, 'get_frame'):
                    frame = camera_manager.get_frame(timeout=1.0)
                elif hasattr(camera_manager, 'read'):
                    ret, frame = camera_manager.read()
                    frame = frame if ret else None
                else:
                    frame = None
                
                if frame is not None:
                    st.session_state.processing_stats['last_frame_time'] = time.time() # Ensure last_frame_time is updated
                    # Process frame
                    start_time = time.time()
                    processed_frame, results = self.process_single_frame(
                        frame, model, task_type, confidence
                    )
                    processing_time = time.time() - start_time
                    
                    if processed_frame is not None:
                        # Add live overlay information
                        self.add_live_overlay(processed_frame, results, processing_time)
                        
                        # Convert to RGB for display
                        if len(processed_frame.shape) == 3:
                            display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        else:
                            display_frame = processed_frame
                        
                        # Update ONLY the video container
                        with video_placeholder.container():
                            st.image(
                                display_frame,
                                caption=f"📹 {_('live_feed')} - {_('camera')} {camera_id}",
                                use_container_width=True
                            )
                        
                        # Update ONLY the status
                        self.update_live_status_display(status_placeholder, results, processing_time, camera_id)
                        
                        # Update analytics
                        self.update_detection_analytics(results)
                        
                        # Increment frame counter
                        st.session_state[f'frame_counter_{camera_id}'] += 1
                        
                        # CONTROLLED refresh - only every 10 frames or 3 seconds
                        frame_count = st.session_state[f'frame_counter_{camera_id}']
                        if frame_count % 10 == 0 or (time.time() - st.session_state.processing_stats.get('last_refresh', 0)) > 3:
                            st.session_state.processing_stats['last_refresh'] = time.time()
                            # Use a much longer delay and conditional rerun
                            time.sleep(0.5)  # Half second delay instead of 33ms
                            st.rerun()
                    
                else:
                    status_placeholder.warning(f"⚠️ {_('no_frame_available')}")
                    # Only rerun if no frame for extended period
                    if time.time() - st.session_state.processing_stats.get('last_frame_time', time.time()) > 5:
                        time.sleep(2)
                        st.rerun()
            
            except Exception as e:
                status_placeholder.error(f"❌ {_('live_processing_error')}: {str(e)}")
                logger.error(f"Live processing error: {e}")
        
        else:
            # Show static frame when not processing
            try:
                if hasattr(camera_manager, 'get_frame'):
                    frame = camera_manager.get_frame(timeout=1.0)
                elif hasattr(camera_manager, 'read'):
                    ret, frame = camera_manager.read()
                    frame = frame if ret else None
                else:
                    frame = None
                
                if frame is not None:
                    # Add "Ready" overlay
                    frame_with_overlay = frame.copy()
                    cv2.putText(frame_with_overlay, f"READY - {_('click_start_processing')}", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    display_frame = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(
                        display_frame,
                        caption=f"📹 {_('camera')} {camera_id} - {_('ready')}",
                        use_container_width=True
                    )
                else:
                    # Show placeholder
                    placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder_img, f"{_('camera')} {camera_id} - {_('connected')}", 
                            (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    video_placeholder.image(placeholder_img, channels="BGR", use_container_width=True)
                
                status_placeholder.info(f"🟡 {_('connected_ready_for_processing')}")
                
            except Exception as e:
                status_placeholder.error(f"❌ {_('camera_error')}: {str(e)}")
    
    def render_live_controls_static(self, model, task_type: str, confidence: float):
        """Render live processing controls without refresh issues"""
        st.markdown(f"### 🎛️ {_('live_controls')}")
        
        # Current camera info
        if st.session_state.current_camera:
            camera_id = st.session_state.current_camera
            camera_ip = CAMERA_MAP.get(str(camera_id), "Unknown")
            st.info(f"📍 Active: Camera {camera_id} ({camera_ip})")
        
        # Task-specific information (static)
        if task_type == "Enhanced License Plate":
            self.render_license_plate_info_static()
        elif task_type == "Parking Detection":
            self.render_parking_info_static()
        elif task_type == "Counting & Alert System":
            self.render_counting_info_static()
        
        # Session statistics (updated periodically, not constantly)
        if st.session_state.processing_active or st.session_state.processing_stats['frames_processed'] > 0:
            st.markdown(f"### 📊 {_('live_statistics')}")
            
            stats = st.session_state.processing_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(_('frames_processed'), f"{stats['frames_processed']:,}")
                st.metric(_('processing_fps'), f"{stats['processing_fps']:.1f}")
            
            with col2:
                st.metric(_('detections'), f"{stats['detections_count']:,}")
                
                if task_type == "Enhanced License Plate":
                    unique_plates = len(st.session_state.detection_analytics['unique_plates'])
                    st.metric(_('unique_plates'), unique_plates)
                    
    def render_license_plate_info_static(self):
        """Static license plate info display"""
        st.markdown(f"### 🚗 {_('live_license_plate_info')}")
        
        if st.session_state.license_processor and hasattr(st.session_state.license_processor, 'license_plates'):
            plates = st.session_state.license_processor.license_plates
            if plates:
                st.write(f"**{_('tracked_vehicles')}:** {len(plates)}")
                
                # Show recent detections in expander (static)
                with st.expander(f"🔍 {_('recent_detections')}", expanded=False):
                    for vehicle_id, plate_info in list(plates.items())[:5]:
                        st.write(f"• **{_('vehicle')} {vehicle_id}:** {plate_info.get('text', 'Unknown')} ({plate_info.get('text_score', 0):.3f})")
            else:
                st.info(f"ℹ️ {_('no_vehicles_currently_tracked')}")
                
    def render_parking_info_static(self):
        """Static parking info display"""
        st.markdown(f"### 🅿️ {_('live_parking_info')}")
        
        if st.session_state.parking_detector:
            summary = st.session_state.parking_detector.get_parking_summary()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(_('free_spaces'), summary['free_spots'])
            with col2:
                st.metric(_('total_spaces'), summary['total_spots'])

    def render_counting_info_static(self):
        """Static counting info display"""
        st.markdown(f"### 📊 {_('live_counting_info')}")
        
        if st.session_state.counting_system:
            stats = st.session_state.counting_system.get_summary_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(_('total_zones'), stats['total_zones'])
            with col2:
                st.metric(_('active_alerts'), stats['active_alerts'])
    
    def add_live_overlay(self, frame: np.ndarray, results: Dict, processing_time: float):
        """Add live information overlay to frame"""
        try:
            # Performance metrics
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Basic overlay information
            overlay_text = [
                f"FPS: {fps:.1f}",
                f"{_('detections')}: {results.get('detections_count', 0)}",
            ]
            
            # Task-specific overlay
            if results.get('model_type') == 'Enhanced License Plate':
                plates_read = results.get('plates_read', 0)
                if plates_read > 0:
                    overlay_text.append(f"{_('plates')}: {plates_read}")
                    
            elif results.get('model_type') == 'Parking Detection':
                free_spaces = results.get('free_spaces', 0)
                total_spaces = results.get('total_spaces', 0)
                overlay_text.append(f"{_('parking')}: {free_spaces}/{total_spaces}")
                
            elif results.get('model_type') == 'Counting & Alert System':
                alerts = results.get('active_alerts', 0)
                crossings = results.get('total_crossings', 0)
                overlay_text.append(f"{_('alerts')}: {alerts}")
                overlay_text.append(f"{_('crossings')}: {crossings}")
            
            # Draw overlay
            y_offset = 30
            for text in overlay_text:
                cv2.putText(frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 25
            
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (frame.shape[1] - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            logger.error(f"Error adding live overlay: {e}")
    
    def update_live_status_display(self, placeholder, results: Dict, processing_time: float, camera_id: int):
        """Update live processing status without full page refresh"""
        try:
            fps = 1.0 / processing_time if processing_time > 0 else 0
            current_time = datetime.now().strftime("%H:%M:%S")
            
            status_text = f"🟢 **{_('live_processing_active')}** | "
            status_text += f"{_('fps')}: {fps:.1f} | "
            status_text += f"{_('time')}: {current_time}"
            
            # Add detection info
            if results and not results.get('error'):
                if results.get('model_type') == 'Enhanced License Plate' and results.get('plates_read', 0) > 0:
                    status_text += f" | 🎯 **{results['plates_read']} {_('plates_detected')}!**"
                elif results.get('detections_count', 0) > 0:
                    status_text += f" | 🎯 **{results['detections_count']} {_('objects_detected')}!**"
            
            placeholder.markdown(status_text)
            
        except Exception as e:
            logger.error(f"Error updating live status: {e}")
            
    def add_live_overlay(self, frame: np.ndarray, results: Dict, processing_time: float):
        """Add live information overlay to frame - IMPROVED VERSION"""
        try:
            # Performance metrics
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Basic overlay information with better positioning
            overlay_texts = [
                f"FPS: {fps:.1f}",
                f"{_('detections')}: {results.get('detections_count', 0)}",
            ]
            
            # Task-specific overlay
            if results.get('model_type') == 'Enhanced License Plate':
                plates_read = results.get('plates_read', 0)
                if plates_read > 0:
                    overlay_texts.append(f"{_('plates')}: {plates_read}")
                    
            elif results.get('model_type') == 'Parking Detection':
                free_spaces = results.get('free_spaces', 0)
                total_spaces = results.get('total_spaces', 0)
                overlay_texts.append(f"{_('parking')}: {free_spaces}/{total_spaces}")
                
            elif results.get('model_type') == 'Counting & Alert System':
                alerts = results.get('active_alerts', 0)
                crossings = results.get('total_crossings', 0)
                overlay_texts.append(f"{_('alerts')}: {alerts}")
                overlay_texts.append(f"{_('crossings')}: {crossings}")
            
            # Draw overlay with better visibility
            y_offset = 30
            for text in overlay_texts:
                # Add black background for better readability
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(frame, (5, y_offset - 25), (15 + text_size[0], y_offset + 5), (0, 0, 0), -1)
                cv2.putText(frame, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                y_offset += 30
            
            # Add timestamp with background
            timestamp = datetime.now().strftime("%H:%M:%S")
            timestamp_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (frame.shape[1] - timestamp_size[0] - 15, 5), 
                        (frame.shape[1] - 5, 35), (0, 0, 0), -1)
            cv2.putText(frame, timestamp, (frame.shape[1] - timestamp_size[0] - 10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            logger.error(f"Error adding live overlay: {e}")
    
    def display_image_results(self, results: Dict, task_type: str):
        """Display image processing results with enhanced parking info"""
        if results.get('error'):
            st.error(f"❌ {_('processing_error')}: {results['error']}")
            return
        
        if task_type == "Enhanced License Plate" and results.get('results'):
            st.markdown(f"""
            <div class="success-box">
                <h4>🎉 {_('detection_results')}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for car_id, result in results['results'].items():
                if 'license_plate' in result:
                    plate_info = result['license_plate']
                    plate_text = plate_info.get('text', _('unknown'))
                    text_score = plate_info.get('text_score', 0)
                    
                    st.markdown(f"""
                    <div class="plate-detection">
                        <h4>🚙 {_('vehicle_id')}: {car_id}</h4>
                        <p><strong>📋 {_('license_plate')}:</strong> 
                        <span style="font-size: 1.5em; font-weight: bold; color: #007bff;">{plate_text}</span></p>
                        <p><strong>📊 {_('confidence')}:</strong> {text_score:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.success(f"✅ {_('detected')} {results['plates_read']} {_('license_plates')}")
        
        elif task_type == "Parking Detection" and results.get('parking_stats'):
            parking_stats = results['parking_stats']
            
            # Main parking status
            st.markdown(f"""
            <div class="success-box">
                <h4>🅿️ {_('parking_status')}</h4>
                <p><strong>{_('free_spaces')}:</strong> {parking_stats.get('free_spaces', 0)}</p>
                <p><strong>{_('occupied_spaces')}:</strong> {parking_stats.get('occupied_spaces', 0)}</p>
                <p><strong>{_('total_spaces')}:</strong> {parking_stats.get('total_spaces', 0)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced: Show YOLO detection info
            yolo_count = results.get('yolo_detections_count', 0)
            yolo_enabled = results.get('yolo_visualization_enabled', False)
            
            if yolo_count > 0:
                st.markdown(f"""
                <div class="model-info">
                    <h4>🎯 {_('yolo_detections')} Results</h4>
                    <p><strong>Total Objects Detected:</strong> {yolo_count}</p>
                    <p><strong>{_('detection_visualization')}:</strong> {'✅ Enabled' if yolo_enabled else '❌ Disabled'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show detection details if available
                detection_details = results.get('detection_details', [])
                if detection_details:
                    with st.expander(f"🔍 Detection Details ({len(detection_details)} objects)", expanded=False):
                        for i, det in enumerate(detection_details):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.write(f"**Object {i+1}:** {det['class']}")
                            with col2:
                                st.write(f"**Confidence:** {det['confidence']:.3f}")
                            with col3:
                                bbox = det['bbox']
                                st.write(f"**Location:** ({bbox[0]}, {bbox[1]})")
        
        elif task_type == "Counting & Alert System" and results.get('counting_summary'):
            counting_summary = results['counting_summary']
            st.markdown(f"""
            <div class="success-box">
                <h4>📊 {_('counting_alert_status')}</h4>
                <p><strong>{_('active_alerts')}:</strong> {counting_summary['active_alerts']}</p>
                <p><strong>{_('total_crossings')}:</strong> {counting_summary['total_crossings']}</p>
                <p><strong>{_('total_zones')}:</strong> {counting_summary['total_zones']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            # Standard detection results
            detections = results.get('detections_count', 0)
            if detections > 0:
                st.success(f"✅ {_('detected')} {detections} {_('objects')}")
                
                if results.get('classes'):
                    with st.expander(f"📋 {_('detection_details')}"):
                        for i, (cls, conf) in enumerate(zip(results['classes'], results.get('confidences', []))):
                            st.write(f"• {cls}: {conf:.3f}")
            else:
                st.info(f"ℹ️ {_('no_objects_detected')}")
    
    def auto_save_live_results(self):
        """Auto-save live processing results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if (st.session_state.license_processor and 
                hasattr(st.session_state.license_processor, 'results') and
                st.session_state.license_processor.results):
                
                csv_file = st.session_state.license_processor.save_results(
                    f"live_session_{timestamp}.csv"
                )
                if csv_file:
                    st.success(f"💾 {_('session_auto_saved')}: {csv_file}")
                    
        except Exception as e:
            logger.error(f"Auto-save error: {e}")

    def render_analytics_tab(self, task_type: str):
        """Render the analytics dashboard tab"""
        st.header(f"📊 {_('analytics_dashboard')} - {task_type}")
        
        if task_type == "Enhanced License Plate":
            self.render_license_plate_analytics()
        elif task_type == "Parking Detection":
            self.render_parking_analytics()
        elif task_type == "Counting & Alert System":
            self.render_counting_analytics()
        else:
            self.render_standard_analytics()
    
    def render_license_plate_analytics(self):
        """Render enhanced license plate analytics"""
        if not st.session_state.detection_analytics['detection_timeline']:
            st.info(f"📊 {_('start_processing_for_analytics')}")
            return
        
        # Real-time metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        timeline = st.session_state.detection_analytics['detection_timeline']
        unique_plates = st.session_state.detection_analytics['unique_plates']
        confidence_scores = st.session_state.detection_analytics['confidence_scores']
        
        with col1:
            st.metric(_('total_detections'), len(timeline))
        with col2:
            st.metric(_('unique_plates'), len(unique_plates))
        with col3:
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            st.metric(_('avg_confidence'), f"{avg_confidence:.3f}")
        with col4:
            session_duration = time.time() - st.session_state.session_start
            detection_rate = len(timeline) / (session_duration / 60) if session_duration > 0 else 0
            st.metric(_('detections_per_minute'), f"{detection_rate:.1f}")
        
        # Analytics tabs
        analytics_tabs = st.tabs([
            f"📈 {_('detection_timeline')}",
            f"📊 {_('confidence_analysis')}",
            f"🏷️ {_('plate_discovery')}",
            f"⚡ {_('performance_trends')}"
        ])
        
        with analytics_tabs[0]:
            self.render_detection_timeline_chart(timeline)
        
        with analytics_tabs[1]:
            self.render_confidence_analysis_chart(confidence_scores)
        
        with analytics_tabs[2]:
            self.render_plate_discovery_chart(timeline, unique_plates)
        
        with analytics_tabs[3]:
            self.render_performance_trends_chart(timeline)
            
    def render_parking_configuration(self):
        """Render parking configuration interface"""
        if not PARKING_AVAILABLE:
            return
        
        st.subheader(f"🅿️ {_('parking_configuration')}")
        
        # File upload for parking positions
        uploaded_parking_file = st.file_uploader(
            _('upload_parking_positions_file'),
            type=['pkl', 'pickle'],
            help=_('upload_parking_positions_help')
        )
        
        if uploaded_parking_file:
            try:
                # Save uploaded file
                with open('CarParkPos', 'wb') as f:
                    f.write(uploaded_parking_file.read())
                
                # Reload parking positions
                with open('CarParkPos', 'rb') as f:
                    st.session_state.posList = pickle.load(f)
                
                st.success(f"✅ {_('parking_positions_loaded')}: {len(st.session_state.posList)} {_('spots')}")
                
                # Reinitialize detector with new positions
                if st.session_state.parking_detector:
                    st.session_state.parking_detector.parking_spots = st.session_state.posList
                    
            except Exception as e:
                st.error(f"❌ {_('parking_file_error')}: {str(e)}")
    
    def render_parking_analytics(self):
        """Render parking detection analytics"""
        if not st.session_state.parking_detector:
            st.info(f"🅿️ {_('parking_system_not_initialized')}")
            return
        
        # Current parking status
        summary = st.session_state.parking_detector.get_parking_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(_('total_spaces'), summary['total_spots'])
        with col2:
            st.metric(_('free_spaces'), summary['free_spots'])
        with col3:
            st.metric(_('occupied_spaces'), summary['occupied_spots'])
        with col4:
            occupancy_rate = summary['occupancy_rate']
            st.metric(_('occupancy_rate'), f"{occupancy_rate:.1f}%")
            
        # Spot-by-spot status
        if st.session_state.parking_detector.parking_spot_states:
            st.subheader(f"📍 {_('individual_spot_status')}")
            
            spots_data = []
            for spot_id, state in st.session_state.parking_detector.parking_spot_states.items():
                spots_data.append({
                    'Spot ID': spot_id,
                    'Status': _('occupied') if state['object_class'] else _('free'),
                    'Vehicle Type': state['object_class'] or _('none'),
                    'Duration': str(datetime.now() - state['entry_time']) if state['entry_time'] else _('n_a')
                })
            
            df_spots = pd.DataFrame(spots_data)
            st.dataframe(df_spots, use_container_width=True)
        
        # Parking events analysis
        if st.session_state.parking_events_log:
            st.subheader(f"📈 {_('parking_events_analysis')}")
            
            events_df = pd.DataFrame(st.session_state.parking_events_log)
            
            if not events_df.empty and 'Entry_time' in events_df.columns:
                events_df['Entry_time'] = pd.to_datetime(events_df['Entry_time'])
                events_df['Hour'] = events_df['Entry_time'].dt.hour
                
                # Hourly parking activity
                hourly_activity = events_df.groupby('Hour').size()
                
                fig_parking = px.bar(
                    x=hourly_activity.index,
                    y=hourly_activity.values,
                    title=_('parking_entries_by_hour'),
                    labels={'x': _('hour_of_day'), 'y': _('number_of_entries')}
                )
                st.plotly_chart(fig_parking, use_container_width=True)
                
                # Parking duration analysis
                if 'Total_duration' in events_df.columns:
                    events_df['Duration_minutes'] = events_df['Total_duration'].dt.total_seconds() / 60
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        avg_duration = events_df['Duration_minutes'].mean()
                        st.metric(_('avg_parking_duration'), f"{avg_duration:.1f} min")
                    
                    with col2:
                        max_duration = events_df['Duration_minutes'].max()
                        st.metric(_('max_parking_duration'), f"{max_duration:.1f} min")
        else:
            st.info(f"📊 {_('no_parking_events_recorded')}")
    
    def render_counting_analytics(self):
        """Render counting system analytics, including from live and saved logs."""
        if not st.session_state.counting_system and not glob.glob("*counting*crossings.csv"): # Check if system or files exist
            st.info(f"📊 {_('counting_system_not_initialized_nor_logs_found')}")
            return

        # Current counting status from live system (if active)
        if st.session_state.counting_system:
            st.subheader(f"🔴 {_('live_counting_status')}")
            stats = st.session_state.counting_system.get_summary_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(_('total_zones'), stats['total_zones'])
            with col2:
                st.metric(_('active_alerts'), stats['active_alerts'])
            with col3:
                st.metric(_('total_crossings'), stats['total_crossings'])
            
            if stats['zone_details']:
                zone_df_live = pd.DataFrame(list(stats['zone_details'].items()), 
                                        columns=[_('zone_name'), _('crossings')])
                if not zone_df_live.empty:
                    fig_zones_live = px.bar(
                        zone_df_live,
                        x=_('zone_name'),
                        y=_('crossings'),
                        title=_('live_crossings_per_zone'),
                        color=_('zone_name')
                    )
                    st.plotly_chart(fig_zones_live, use_container_width=True)
        else:
            st.info(f"ℹ️ {_('live_counting_system_not_active')}")


        st.markdown("---")
        st.subheader(f"💾 {_('historical_counting_logs_analysis')}")

        # Discover counting log files
        counting_log_files = []
        all_discovered_files = self.discover_result_files() 
        for f_info in all_discovered_files:
            if "counting" in f_info['name'].lower() and f_info['name'].endswith("_crossings.csv"):
                counting_log_files.append(f_info)
        
        if not counting_log_files:
            st.info(_('no_counting_log_files_found_pattern'))
            st.caption(f"Searches for files ending with `_crossings.csv` and containing `counting`.")
        else:
            file_options = [f"{f['name']} ({f['size_mb']:.2f}MB, {f['modified']:%Y-%m-%d %H:%M})" for f in counting_log_files]
            
            selected_log_idx = st.selectbox(
                _('select_counting_log_file'),
                range(len(file_options)),
                format_func=lambda x: file_options[x],
                key="counting_log_selector_analytics"
            )

            selected_log_file_info = counting_log_files[selected_log_idx]

            if st.button(_('load_and_analyze_counting_log'), key="analyze_counting_log_btn_analytics"):
                try:
                    df_log = pd.read_csv(selected_log_file_info['path'])
                    
                    if df_log.empty:
                        st.warning(f"{_('selected_log_file_is_empty')}: {selected_log_file_info['name']}")
                        return

                    st.markdown(f"#### {_('analysis_of_selected_log')}: {selected_log_file_info['name']}")
                    st.dataframe(df_log.head(50), use_container_width=True)

                    log_total_crossings = df_log.shape[0]
                    st.metric(f"{_('total_crossings_in_log')}", log_total_crossings)

                    if 'Counter Shape Name' in df_log.columns:
                        zone_counts_from_log = df_log['Counter Shape Name'].value_counts().reset_index()
                        zone_counts_from_log.columns = [_('zone_name'), _('crossings')]

                        fig_zones_log = px.bar(
                            zone_counts_from_log,
                            x=_('zone_name'),
                            y=_('crossings'),
                            title=f"{_('crossings_per_zone')} ({_('from_log')})",
                            color=_('zone_name')
                        )
                        st.plotly_chart(fig_zones_log, use_container_width=True)

                    if 'Timestamp' in df_log.columns:
                        try:
                            df_log['time'] = pd.to_datetime(df_log['Timestamp'])
                            
                            if 'Object' in df_log.columns:
                                events_by_type_log = df_log.groupby([pd.Grouper(key='time', freq='H'), 'Object']).size().reset_index(name='count')
                                if not events_by_type_log.empty:
                                    fig_timeline_log = px.line(
                                        events_by_type_log,
                                        x='time',
                                        y='count',
                                        color='Object',
                                        title=f"{_('events_timeline_by_object')} ({_('from_log')}, {_('hourly')})"
                                    )
                                    st.plotly_chart(fig_timeline_log, use_container_width=True)
                            else:
                                hourly_activity_log = df_log.groupby(df_log['time'].dt.hour).size().reset_index(name='count')
                                hourly_activity_log.columns = [_('hour_of_day'), _('number_of_events')]
                                fig_hourly_log = px.bar(
                                    hourly_activity_log,
                                    x=_('hour_of_day'),
                                    y=_('number_of_events'),
                                    title=f"{_('events_by_hour')} ({_('from_log')})"
                                )
                                st.plotly_chart(fig_hourly_log, use_container_width=True)
                        except Exception as e_time:
                            st.warning(f"{_('could_not_parse_timestamps_for_timeline_chart')}: {e_time}")
                    
                    csv_string_log = df_log.to_csv(index=False)
                    st.download_button(
                        label=f"📥 {_('download_analyzed_log_csv')}",
                        data=csv_string_log,
                        file_name=f"analyzed_{selected_log_file_info['name']}",
                        mime="text/csv",
                        key="download_analyzed_counting_log_analytics"
                    )

                except FileNotFoundError:
                    st.error(f"{_('file_not_found')}: {selected_log_file_info['path']}")
                except pd.errors.EmptyDataError:
                    st.warning(f"{_('selected_log_file_is_empty')}: {selected_log_file_info['name']}")
                except Exception as e:
                    st.error(f"{_('error_analyzing_log_file')}: {str(e)}")
                    logger.error(f"Error analyzing counting log file {selected_log_file_info['path']}: {traceback.format_exc()}")
    
    def render_standard_analytics(self):
        """Render analytics for standard detection models"""
        stats = st.session_state.processing_stats
        
        if stats['frames_processed'] == 0:
            st.info(f"📊 {_('start_processing_for_analytics')}")
            return
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(_('frames_processed'), f"{stats['frames_processed']:,}")
        with col2:
            st.metric(_('total_detections'), f"{stats['detections_count']:,}")
        with col3:
            st.metric(_('processing_fps'), f"{stats['processing_fps']:.1f}")
        with col4:
            session_duration = time.time() - stats['start_time'] if stats['start_time'] else 0
            st.metric(_('session_duration'), f"{session_duration/60:.1f} min")
        
        # Simple performance chart
        if len(st.session_state.detection_analytics['confidence_scores']) > 0:
            st.subheader(f"📊 {_('detection_confidence_distribution')}")
            
            fig_hist = px.histogram(
                x=st.session_state.detection_analytics['confidence_scores'],
                title=_('confidence_distribution'),
                labels={'x': _('confidence_score'), 'y': _('frequency')},
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    def render_detection_timeline_chart(self, timeline):
        """Render detection timeline chart"""
        if len(timeline) < 2:
            st.info(f"📈 {_('need_more_detections_for_timeline')}")
            return
        
        df_timeline = pd.DataFrame(timeline[-100:])  # Last 100 detections
        df_timeline['time'] = pd.to_datetime(df_timeline['timestamp'], unit='s')
        df_timeline['quality'] = df_timeline['confidence'].apply(
            lambda x: _('excellent') if x > 0.8 else _('good') if x > 0.6 else _('fair')
        )
        
        fig_timeline = px.scatter(
            df_timeline,
            x='time',
            y='confidence',
            color='quality',
            hover_data=['plate', 'vehicle_id'] if 'vehicle_id' in df_timeline.columns else ['plate'],
            title=_('detection_timeline_chart'),
            labels={'confidence': _('confidence_score'), 'time': _('detection_time')}
        )
        
        # Add threshold lines
        fig_timeline.add_hline(y=0.8, line_dash="dash", line_color="green", 
                              annotation_text=_('excellent_threshold'))
        fig_timeline.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                              annotation_text=_('good_threshold'))
        
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    def render_confidence_analysis_chart(self, confidence_scores):
        """Render confidence analysis charts"""
        if not confidence_scores:
            st.info(f"📊 {_('no_confidence_data_available')}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution
            fig_hist = px.histogram(
                x=confidence_scores,
                title=_('confidence_distribution'),
                labels={'x': _('confidence_score'), 'y': _('frequency')},
                nbins=20
            )
            fig_hist.add_vline(x=0.8, line_dash="dash", line_color="green")
            fig_hist.add_vline(x=0.6, line_dash="dash", line_color="orange")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Quality breakdown pie chart
            excellent = sum(1 for score in confidence_scores if score > 0.8)
            good = sum(1 for score in confidence_scores if 0.6 <= score <= 0.8)
            fair = sum(1 for score in confidence_scores if score < 0.6)
            
            fig_pie = px.pie(
                values=[excellent, good, fair],
                names=[_('excellent'), _('good'), _('fair')],
                title=_('quality_distribution')
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def render_plate_discovery_chart(self, timeline, unique_plates):
        """Render plate discovery analysis"""
        if not timeline:
            st.info(f"🏷️ {_('no_plate_discovery_data')}")
            return
        
        # Unique plates over time
        unique_timeline = []
        seen_plates = set()
        
        for entry in timeline:
            seen_plates.add(entry['plate'])
            unique_timeline.append({
                'time': pd.to_datetime(entry['timestamp'], unit='s'),
                'unique_count': len(seen_plates)
            })
        
        df_unique = pd.DataFrame(unique_timeline)
        
        fig_discovery = px.line(
            df_unique,
            x='time',
            y='unique_count',
            title=_('unique_plates_discovery'),
            labels={'unique_count': _('cumulative_unique_plates'), 'time': _('time')}
        )
        st.plotly_chart(fig_discovery, use_container_width=True)
        
        # Top detected plates
        if len(unique_plates) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"🏷️ {_('discovered_plates')}")
                for i, plate in enumerate(list(unique_plates)[:10], 1):
                    st.write(f"{i}. **{plate}**")
            
            with col2:
                st.subheader(f"📈 {_('discovery_metrics')}")
                discovery_rate = len(unique_plates) / len(timeline) if timeline else 0
                st.metric(_('discovery_rate'), f"{discovery_rate:.3f}")
                st.metric(_('total_unique'), len(unique_plates))
    
    def render_performance_trends_chart(self, timeline):
        """Render performance trends analysis"""
        if len(timeline) < 10:
            st.info(f"⚡ {_('need_more_data_for_trends')}")
            return
        
        df_perf = pd.DataFrame(timeline)
        df_perf['time'] = pd.to_datetime(df_perf['timestamp'], unit='s')
        df_perf = df_perf.sort_values('time')
        
        # Rolling average confidence
        window_size = min(10, len(df_perf))
        df_perf['rolling_confidence'] = df_perf['confidence'].rolling(
            window=window_size, min_periods=1
        ).mean()
        
        fig_trend = go.Figure()
        
        # Individual detections
        fig_trend.add_trace(go.Scatter(
            x=df_perf['time'],
            y=df_perf['confidence'],
            mode='markers',
            name=_('individual_detections'),
            marker=dict(color='lightblue', size=6),
            opacity=0.6
        ))
        
        # Rolling average
        fig_trend.add_trace(go.Scatter(
            x=df_perf['time'],
            y=df_perf['rolling_confidence'],
            mode='lines',
            name=f"{_('rolling_average')} ({window_size})",
            line=dict(color='blue', width=3)
        ))
        
        fig_trend.update_layout(
            title=_('performance_trend_analysis'),
            xaxis_title=_('time'),
            yaxis_title=_('confidence_score')
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    def render_results_tab(self, task_type: str):
        """Render results management tab"""
        st.header(f"📋 {_('results_management')} - {task_type}")
        
        # File discovery and management
        result_files = self.discover_result_files()
        
        if result_files:
            # File selection interface
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📁 {_('available_result_files')}")
                
                # Create file selection
                file_options = []
                for file_info in result_files:
                    option_text = f"{file_info['name']} | {file_info['size_mb']:.1f}MB | {file_info['modified'].strftime('%Y-%m-%d %H:%M')}"
                    file_options.append(option_text)
                
                selected_idx = st.selectbox(
                    _('select_result_file'),
                    range(len(file_options)),
                    format_func=lambda x: file_options[x]
                )
                
                selected_file = result_files[selected_idx]
            
            with col2:
                st.subheader(f"📊 {_('file_info')}")
                
                st.markdown(f"""
                **{_('filename')}:** {selected_file['name']}  
                **{_('size')}:** {selected_file['size_mb']:.2f} MB  
                **{_('modified')}:** {selected_file['modified'].strftime('%Y-%m-%d %H:%M:%S')}  
                **{_('estimated_rows')}:** {selected_file['estimated_rows']:,}  
                """)
            
            # Load and analyze selected file
            if st.button(f"📊 {_('analyze_selected_file')}", type="primary"):
                self.analyze_result_file(selected_file['path'])
        
        else:
            st.info(f"📁 {_('no_result_files_found')}")
            
            # Quick actions for file creation
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"🎬 {_('process_video')}", use_container_width=True):
                    st.info(f"💡 {_('go_to_detection_tab_video')}")
            
            with col2:
                if st.button(f"📹 {_('start_live_session')}", use_container_width=True):
                    st.info(f"💡 {_('go_to_detection_tab_camera')}")
            
            with col3:
                uploaded_file = st.file_uploader(f"📁 {_('upload_results_file')}", type=['csv'])
                if uploaded_file:
                    self.handle_uploaded_result_file(uploaded_file)
    
    def render_reports_tab(self, task_type: str):
        """Render reports and export tab"""
        st.header(f"📈 {_('reports_export')} - {task_type}")
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"### 📊 {_('session_export')}")
            
            if st.button(f"📥 {_('export_session_data')}", use_container_width=True):
                self.export_session_data(task_type)
        
        with col2:
            st.markdown(f"### 📈 {_('analytics_export')}")
            
            if st.button(f"📥 {_('export_analytics')}", use_container_width=True):
                self.export_analytics_data(task_type)
        
        with col3:
            st.markdown(f"### 📋 {_('generate_report')}")
            
            if st.button(f"📥 {_('generate_summary_report')}", use_container_width=True):
                self.generate_summary_report(task_type)
        
        # Task-specific export options
        if task_type == "Enhanced License Plate" and st.session_state.license_processor:
            st.markdown("---")
            st.subheader(f"🚗 {_('license_plate_specific_exports')}")
            
            col1, col2, col3 = st.columns(3)
        
            with col1:
                if st.button(f"📊 {_('export_detection_results')}", use_container_width=True):
                    self.export_license_plate_results()
            
            with col2:
                if st.button(f"📈 {_('export_performance_report')}", use_container_width=True):
                    self.export_performance_report()
            
            # NEW: Final results generation
            with col3:
                if st.button("🎯 Generate Final Results", use_container_width=True):
                    self.generate_final_license_plate_results(task_type)
        
        elif task_type == "Parking Detection" and st.session_state.parking_detector:
            st.markdown("---")
            st.subheader(f"🅿️ {_('parking_specific_exports')}")
            
            if st.button(f"📊 {_('export_parking_report')}", use_container_width=True):
                self.export_parking_report()
        
        elif task_type == "Counting & Alert System" and st.session_state.counting_system:
            st.markdown("---")
            st.subheader(f"📊 {_('counting_specific_exports')}")
            
            if st.button(f"📊 {_('export_counting_log')}", use_container_width=True):
                self.export_counting_log()
    
    def render_system_management_tab(self):
        """Render system management tab (for enhanced license plate)"""
        st.header(f"⚙️ {_('system_management')}")
        
        # System status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader(f"🔧 {_('system_status')}")
            
            status_items = [
                (f"License Plate Utils", LICENSE_PLATE_UTILS_AVAILABLE),
                (f"SORT Tracker", SORT_AVAILABLE),
                (f"Post Processing", POST_PROCESSING_AVAILABLE),
                (f"CVZone", CVZONE_AVAILABLE)
            ]
            
            for item, available in status_items:
                status = "✅" if available else "❌"
                st.write(f"{status} {item}")
        
        with col2:
            st.subheader(f"📊 {_('memory_usage')}")
            
            # Session state size estimation
            session_size = len(st.session_state.keys())
            st.metric(_('session_variables'), session_size)
            
            # Analytics data size
            timeline_size = len(st.session_state.detection_analytics['detection_timeline'])
            st.metric(_('analytics_entries'), timeline_size)
        
        with col3:
            st.subheader(f"🔧 {_('maintenance')}")
            
            if st.button(f"🧹 {_('clear_analytics_data')}", use_container_width=True):
                self.clear_analytics_data()
            
            if st.button(f"🔄 {_('reset_all_systems')}", use_container_width=True):
                self.reset_all_systems()
        
        # Advanced debug information
        with st.expander(f"🔍 {_('debug_information')}", expanded=False):
            debug_info = {
                'session_start': st.session_state.session_start,
                'processing_active': st.session_state.processing_active,
                'current_camera': st.session_state.current_camera,
                'detection_analytics_size': {
                    'timeline': len(st.session_state.detection_analytics['detection_timeline']),
                    'unique_plates': len(st.session_state.detection_analytics['unique_plates']),
                    'confidence_scores': len(st.session_state.detection_analytics['confidence_scores'])
                }
            }
            st.json(debug_info)
    
    def discover_result_files(self):
        """Discover available result files"""
        try:
            patterns = [
                ('./*license*.csv', 'License Plate'),
                ('./*parking*.csv', 'Parking'),
                ('./*counting*.csv', 'Counting'),
                ('./*results*.csv', 'General Results'),
                ('./*_crossings.csv', 'Crossing Log')
            ]
            
            files = []
            for pattern, category in patterns:
                for file_path in glob.glob(pattern):
                    try:
                        stat = os.stat(file_path)
                        size_mb = stat.st_size / (1024 * 1024)
                        modified = datetime.fromtimestamp(stat.st_mtime)
                        
                        # Estimate rows
                        try:
                            with open(file_path, 'r') as f:
                                estimated_rows = sum(1 for _ in f) - 1  # Subtract header
                        except:
                            estimated_rows = 0
                        
                        files.append({
                            'path': file_path,
                            'name': os.path.basename(file_path),
                            'category': category,
                            'size_mb': size_mb,
                            'modified': modified,
                            'estimated_rows': estimated_rows
                        })
                    except Exception as e:
                        logger.error(f"Error analyzing file {file_path}: {e}")
            
            return sorted(files, key=lambda x: x['modified'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error discovering files: {e}")
            return []
    
    def analyze_result_file(self, file_path: str):
        """Analyze selected result file"""
        try:
            df = pd.read_csv(file_path)
            
            st.subheader(f"📊 {_('file_analysis')}: {os.path.basename(file_path)}")
            
            # Basic statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(_('total_records'), len(df))
                st.metric(_('columns'), len(df.columns))
            
            with col2:
                if 'license_number' in df.columns:
                    license_data = df[df['license_number'].notna() & (df['license_number'] != '0')]
                    st.metric(_('license_plates'), len(license_data))
                    st.metric(_('unique_plates'), license_data['license_number'].nunique())
            
            with col3:
                if 'license_number_score' in df.columns:
                    scores = df['license_number_score']
                    scores = scores[scores > 0]
                    if len(scores) > 0:
                        st.metric(_('avg_confidence'), f"{scores.mean():.3f}")
                        st.metric(_('max_confidence'), f"{scores.max():.3f}")
            
            # Data preview
            st.subheader(f"🔍 {_('data_preview')}")
            st.dataframe(df.head(20), use_container_width=True)
            
            # Download processed data
            csv_string = df.to_csv(index=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.download_button(
                label=f"📥 {_('download_csv')}",
                data=csv_string,
                file_name=f"analyzed_{timestamp}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ {_('file_analysis_error')}: {str(e)}")
    
    def export_session_data(self, task_type: str):
        """Export current session data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            session_data = {
                'timestamp': timestamp,
                'task_type': task_type,
                'session_duration': time.time() - st.session_state.session_start,
                'processing_stats': st.session_state.processing_stats,
                'detection_analytics': {
                    'total_detections': len(st.session_state.detection_analytics['detection_timeline']),
                    'unique_plates': list(st.session_state.detection_analytics['unique_plates']),
                    'confidence_scores': st.session_state.detection_analytics['confidence_scores'][-100:]
                }
            }
            
            json_string = json.dumps(session_data, indent=2, default=str)
            
            st.download_button(
                label=f"📥 {_('download_session_json')}",
                data=json_string,
                file_name=f"session_export_{timestamp}.json",
                mime="application/json"
            )
            
            st.success(f"✅ {_('session_data_exported')}")
            
        except Exception as e:
            st.error(f"❌ {_('export_error')}: {str(e)}")
    
    def is_cloud_deployment(self):
        """Check if running in cloud"""
        return (
            os.environ.get('STREAMLIT_SHARING_MODE') == '1' or
            'streamlit.app' in os.environ.get('HOSTNAME', '') or
            not os.environ.get('DISPLAY')
        )
        
    def render_camera_interface_cloud_friendly(self, model, task_type: str, confidence: float):
        """Cloud-friendly camera interface that doesn't rely on OpenCV windows"""
        st.subheader(f"📡 {_('live_camera_analysis')}")
        
        # Check camera connection
        if not st.session_state.camera_managers:
            st.warning(f"⚠️ {_('no_camera_connected')}")
            st.info(f"💡 Note: In cloud deployment, live camera processing is limited.")
            return
        
        # Show cloud deployment notice
        st.info("🌐 **Cloud Deployment Mode**: Live camera processing will run in background. Results will be displayed in Streamlit interface.")
        
        camera_id = st.session_state.current_camera
        if not camera_id or camera_id not in st.session_state.camera_managers:
            st.info(f"📹 {_('select_camera_to_start')}")
            return
        
        # Control buttons for cloud environment
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start Background Processing", key="start_cloud_processing"):
                # Start headless processing
                self.start_cloud_processing(camera_id, model, task_type, confidence)
        
        with col2:
            if st.button("⏹️ Stop Processing", key="stop_cloud_processing"):
                self.stop_cloud_processing(camera_id)
        
        # Display results in Streamlit instead of OpenCV windows
        self.display_cloud_results(camera_id, task_type)
        
    def start_cloud_processing(self, camera_id, model, task_type, confidence):
        """Start cloud-friendly processing"""
        try:
            processor_thread = LiveVideoProcessor(
                camera_id, model, task_type, confidence, self
            )
            
            if processor_thread.start_processing():
                self.live_processors[camera_id] = processor_thread
                st.success("🚀 Background processing started!")
                st.info("📊 Results will appear below. Refresh page to see updates.")
            else:
                st.error("❌ Failed to start processing")
        except Exception as e:
            st.error(f"❌ Processing error: {e}")
    
    def stop_cloud_processing(self, camera_id):
        """Stop cloud processing"""
        try:
            if camera_id in self.live_processors:
                self.live_processors[camera_id].stop_processing()
                del self.live_processors[camera_id]
                st.success("⏹️ Processing stopped")
        except Exception as e:
            st.error(f"❌ Stop processing error: {e}")
    
    def display_cloud_results(self, camera_id, task_type):
        """Display processing results in Streamlit interface"""
        if camera_id in self.live_processors:
            processor = self.live_processors[camera_id]
            stats = processor.get_stats()
            
            if stats:
                # Display stats in Streamlit
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Frames Processed", f"{stats['frame_count']:,}")
                with col2:
                    st.metric("Processing FPS", f"{stats['fps']:.1f}")
                with col3:
                    st.metric("Detections", f"{stats['detection_count']:,}")
                
                # Auto-refresh every 5 seconds
                if st.button("🔄 Refresh Results"):
                    st.rerun()
        else:
            st.info("No active processing. Start background processing to see results.")
            
    def render_demo_interface(self):
        """Demo-friendly interface for cloud deployment"""
        if self.is_cloud_deployment():
            st.info("🌐 **Cloud Demo Mode** - Optimized for online presentations")
            
            # Focus on file-based processing
            demo_tabs = st.tabs([
                "🎬 Video Processing Demo",
                "📸 Image Analysis Demo", 
                "📊 Sample Results",
                "🎯 Live Simulation"
            ])
            
            with demo_tabs[0]:
                self.render_video_demo()
            
            with demo_tabs[1]:
                self.render_image_demo()
            
            with demo_tabs[2]:
                self.render_sample_results()
            
            with demo_tabs[3]:
                self.render_simulated_live_demo()
        else:
            # Full local interface
            self.render_full_interface()
            
    def render_video_demo(self):
        """Video processing demo that works well in cloud"""
        st.subheader("🎬 Video Analysis Demo")
        
        # Pre-loaded sample videos work great in cloud
        sample_videos = {
            "License Plate Detection": "sample_license_plate.mp4",
            "Parking Detection": "sample_parking.mp4", 
            "Counting & Alerts": "sample_counting.mp4"
        }
        
        selected_demo = st.selectbox("Choose Demo Video", list(sample_videos.keys()))
        
        if st.button("🚀 Process Demo Video"):
            # Process the video and show results
            self.process_demo_video(sample_videos[selected_demo])

    def render_simulated_live_demo(self):
        """Simulate live processing for demo purposes"""
        st.subheader("🎯 Simulated Live Processing")
        st.info("This simulates real-time processing using pre-recorded data")
        
        if st.button("▶️ Start Simulation"):
            # Create a simulation that updates in real-time
            self.run_simulation_demo()
            
    def export_analytics_data(self, task_type: str):
        """Export analytics data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            analytics_data = {
                'export_timestamp': timestamp,
                'task_type': task_type,
                'detection_timeline': st.session_state.detection_analytics['detection_timeline'],
                'confidence_scores': st.session_state.detection_analytics['confidence_scores'],
                'unique_plates': list(st.session_state.detection_analytics['unique_plates'])
            }
            
            json_string = json.dumps(analytics_data, indent=2, default=str)
            
            st.download_button(
                label=f"📥 {_('download_analytics_json')}",
                data=json_string,
                file_name=f"analytics_export_{timestamp}.json",
                mime="application/json"
            )
            
            st.success(f"✅ {_('analytics_data_exported')}")
            
        except Exception as e:
            st.error(f"❌ {_('export_error')}: {str(e)}")
    
    def generate_summary_report(self, task_type: str):
        """Generate summary report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_duration = time.time() - st.session_state.session_start
            
            report = f"""
{_('vision_platform_report')}
{'='*50}

{_('session_information')}:
- {_('task_type')}: {task_type}
- {_('session_duration')}: {session_duration/60:.1f} {_('minutes')}
- {_('generated')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{_('processing_statistics')}:
- {_('frames_processed')}: {st.session_state.processing_stats['frames_processed']:,}
- {_('total_detections')}: {st.session_state.processing_stats['detections_count']:,}
- {_('processing_fps')}: {st.session_state.processing_stats['processing_fps']:.1f}

{_('detection_analytics')}:
- {_('timeline_entries')}: {len(st.session_state.detection_analytics['detection_timeline']):,}
- {_('unique_plates')}: {len(st.session_state.detection_analytics['unique_plates'])}
- {_('confidence_scores')}: {len(st.session_state.detection_analytics['confidence_scores'])}

{_('system_status')}:
- {_('license_plate_utils')}: {'✅' if LICENSE_PLATE_UTILS_AVAILABLE else '❌'}
- {_('sort_tracker')}: {'✅' if SORT_AVAILABLE else '❌'}
- {_('post_processing')}: {'✅' if POST_PROCESSING_AVAILABLE else '❌'}

{_('detected_license_plates')}:
"""
            
            # Add unique plates to report
            unique_plates = list(st.session_state.detection_analytics['unique_plates'])
            for i, plate in enumerate(unique_plates[:20], 1):  # Limit to first 20
                report += f"  {i}. {plate}\n"
            
            if len(unique_plates) > 20:
                report += f"  ... {_('and')} {len(unique_plates) - 20} {_('more')}\n"
            
            report += f"\n{_('end_of_report')}"
            
            st.download_button(
                label=f"📥 {_('download_report')}",
                data=report,
                file_name=f"vision_report_{timestamp}.txt",
                mime="text/plain"
            )
            
            st.success(f"✅ {_('report_generated')}")
            
        except Exception as e:
            st.error(f"❌ {_('report_generation_error')}: {str(e)}")
    
    def export_license_plate_results(self):
        """Export license plate detection results"""
        try:
            if not st.session_state.license_processor:
                st.warning(f"⚠️ {_('license_processor_not_available')}")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = st.session_state.license_processor.save_results(
                f"license_plate_export_{timestamp}.csv"
            )
            
            if csv_file:
                # Provide download option
                try:
                    with open(csv_file, 'r') as f:
                        csv_content = f.read()
                    
                    st.download_button(
                        label=f"📥 {_('download_license_plate_csv')}",
                        data=csv_content,
                        file_name=csv_file,
                        mime="text/csv"
                    )
                    
                    st.success(f"✅ {_('license_plate_results_exported')}: {csv_file}")
                    
                except Exception as e:
                    st.error(f"❌ {_('download_preparation_failed')}: {e}")
            else:
                st.warning(f"⚠️ {_('no_results_to_export')}")
                
        except Exception as e:
            st.error(f"❌ {_('export_error')}: {str(e)}")
    
    def export_performance_report(self):
        """Export performance analysis report"""
        try:
            if not st.session_state.detection_analytics['confidence_scores']:
                st.warning(f"⚠️ {_('no_performance_data')}")
                return
            
            confidence_scores = st.session_state.detection_analytics['confidence_scores']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            performance_data = {
                'export_timestamp': timestamp,
                'performance_metrics': {
                    'total_detections': len(confidence_scores),
                    'average_confidence': float(np.mean(confidence_scores)),
                    'median_confidence': float(np.median(confidence_scores)),
                    'std_confidence': float(np.std(confidence_scores)),
                    'min_confidence': float(np.min(confidence_scores)),
                    'max_confidence': float(np.max(confidence_scores)),
                    'excellent_count': int(sum(1 for score in confidence_scores if score > 0.8)),
                    'good_count': int(sum(1 for score in confidence_scores if 0.6 <= score <= 0.8)),
                    'fair_count': int(sum(1 for score in confidence_scores if score < 0.6))
                },
                'confidence_scores': confidence_scores
            }
            
            json_string = json.dumps(performance_data, indent=2, default=str)
            
            st.download_button(
                label=f"📥 {_('download_performance_json')}",
                data=json_string,
                file_name=f"performance_report_{timestamp}.json",
                mime="application/json"
            )
            
            st.success(f"✅ {_('performance_report_exported')}")
            
        except Exception as e:
            st.error(f"❌ {_('export_error')}: {str(e)}")
    
    def export_parking_report(self):
        """Export parking detection report"""
        try:
            if not st.session_state.parking_detector:
                st.warning(f"⚠️ {_('parking_detector_not_available')}")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = st.session_state.parking_detector.save_parking_report(
                f"parking_export_{timestamp}.csv"
            )
            
            if report_file:
                try:
                    with open(report_file, 'r') as f:
                        csv_content = f.read()
                    
                    st.download_button(
                        label=f"📥 {_('download_parking_csv')}",
                        data=csv_content,
                        file_name=report_file,
                        mime="text/csv"
                    )
                    
                    st.success(f"✅ {_('parking_report_exported')}: {report_file}")
                    
                except Exception as e:
                    st.error(f"❌ {_('download_preparation_failed')}: {e}")
            else:
                st.warning(f"⚠️ {_('no_parking_events_to_export')}")
                
        except Exception as e:
            st.error(f"❌ {_('export_error')}: {str(e)}")
    
    def export_counting_log(self):
        """Export counting system log"""
        try:
            if not st.session_state.counting_system:
                st.warning(f"⚠️ {_('counting_system_not_available')}")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_identifier = f"counting_export_{timestamp}"
            
            # Save log using the counting system's method
            st.session_state.counting_system.save_log_to_csv(log_identifier)
            
            # The actual filename will have "_crossings.csv" appended
            actual_filename = f"{log_identifier}_crossings.csv"
            
            if os.path.exists(actual_filename):
                try:
                    with open(actual_filename, 'r') as f:
                        csv_content = f.read()
                    
                    st.download_button(
                        label=f"📥 {_('download_counting_csv')}",
                        data=csv_content,
                        file_name=actual_filename,
                        mime="text/csv"
                    )
                    
                    st.success(f"✅ {_('counting_log_exported')}: {actual_filename}")
                    
                except Exception as e:
                    st.error(f"❌ {_('download_preparation_failed')}: {e}")
            else:
                st.warning(f"⚠️ {_('no_counting_events_to_export')}")
                
        except Exception as e:
            st.error(f"❌ {_('export_error')}: {str(e)}")
    
    def handle_uploaded_result_file(self, uploaded_file):
        """Handle uploaded result file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_path = f"uploaded_results_{timestamp}_{uploaded_file.name}"
            
            with open(saved_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            st.success(f"✅ {_('file_uploaded_and_saved')}: {saved_path}")
            st.info(f"💡 {_('refresh_page_to_see_file')}")
            
        except Exception as e:
            st.error(f"❌ {_('file_upload_error')}: {str(e)}")
    
    def clear_analytics_data(self):
        """Clear analytics data"""
        try:
            st.session_state.detection_analytics = {
                'total_detections': 0,
                'unique_plates': set(),
                'detection_timeline': [],
                'confidence_scores': []
            }
            
            st.session_state.counting_events_history.clear()
            st.session_state.parking_events_log.clear()
            
            st.success(f"✅ {_('analytics_data_cleared')}")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ {_('clear_data_error')}: {str(e)}")
    
    def reset_all_systems(self):
        """Reset all systems to initial state"""
        try:
            # Reset processors
            st.session_state.license_processor = None
            st.session_state.parking_detector = None
            st.session_state.counting_system = None
            
            # Reset analytics
            st.session_state.detection_analytics = {
                'total_detections': 0,
                'unique_plates': set(),
                'detection_timeline': [],
                'confidence_scores': []
            }
            
            # Reset processing stats
            st.session_state.processing_stats = {
                'frames_processed': 0,
                'detections_count': 0,
                'processing_fps': 0.0,
                'start_time': None
            }
            
            # Reset processing state
            st.session_state.processing_active = False
            
            # Clear events
            st.session_state.counting_events_history.clear()
            st.session_state.parking_events_log.clear()
            
            # Disconnect cameras
            for camera_id in list(st.session_state.camera_managers.keys()):
                self.disconnect_camera(camera_id)
            
            st.success(f"✅ {_('all_systems_reset')}")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ {_('reset_error')}: {str(e)}")
    
    def render_license_plate_live_info(self):
        """Render live info specific to license plate detection"""
        if st.session_state.license_processor and hasattr(st.session_state.license_processor, 'license_plates'):
            st.markdown(f"### 🚗 {_('live_license_plate_info')}")
            
            plates = st.session_state.license_processor.license_plates
            if plates:
                st.write(f"**{_('tracked_vehicles')}:** {len(plates)}")
                
                # Show recent detections
                with st.expander(f"🔍 {_('recent_detections')}", expanded=False):
                    for vehicle_id, plate_info in list(plates.items())[:5]:
                        st.write(f"• **{_('vehicle')} {vehicle_id}:** {plate_info.get('text', 'Unknown')} ({plate_info.get('text_score', 0):.3f})")
            else:
                st.info(f"ℹ️ {_('no_vehicles_currently_tracked')}")
    
    def render_parking_live_info(self):
        """Render enhanced live info specific to parking detection"""
        if st.session_state.parking_detector:
            st.markdown(f"### 🅿️ {_('live_parking_info')}")
            
            summary = st.session_state.parking_detector.get_parking_summary()
            
            # Live parking metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(_('total_spots'), summary['total_spots'])
                st.metric(_('free_spots'), summary['free_spots'])
            with col2:
                st.metric(_('occupied_spots'), summary['occupied_spots'])
                occupancy_rate = summary['occupancy_rate']
                st.metric(_('occupancy_rate'), f"{occupancy_rate:.1f}%")
            
            # Enhanced: YOLO detection status
            if hasattr(st.session_state.parking_detector, 'config'):
                config = st.session_state.parking_detector.config
                
                st.markdown("---")
                st.markdown(f"### 🎯 {_('detection_settings')}")
                
                col1, col2 = st.columns(2)
                with col1:
                    yolo_status = "🟢 Enabled" if config.SHOW_YOLO_DETECTIONS else "🔴 Disabled"
                    st.write(f"**{_('yolo_visualization')}:** {yolo_status}")
                    st.write(f"**{_('pixel_threshold')}:** {config.PIXEL_THRESHOLD}")
                
                with col2:
                    st.write(f"**{_('target_classes')}:** {', '.join(config.YOLO_TARGET_CLASSES_NAMES)}")
                    st.write(f"**{_('detection_confidence')}:** {config.YOLO_CONFIDENCE}")
            
            # Recent parking events
            if st.session_state.parking_detector.parking_events_log:
                recent_events = st.session_state.parking_detector.parking_events_log[-3:]  # Last 3 events
                
                with st.expander(f"📝 {_('recent_events')} ({len(recent_events)})", expanded=False):
                    for event in recent_events:
                        event_time = event['Entry_time'].strftime("%H:%M:%S") if event.get('Entry_time') else "Unknown"
                        st.write(f"• **Spot {event['Parking_ID']}:** {event['Object_detected']} at {event_time}")
    
    def render_counting_live_info(self):
        """Render live info specific to counting system"""
        if st.session_state.counting_system:
            st.markdown(f"### 📊 {_('live_counting_info')}")
            
            stats = st.session_state.counting_system.get_summary_stats()
            
            # Live counting metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(_('total_zones'), stats['total_zones'])
                st.metric(_('active_alerts'), stats['active_alerts'])
            with col2:
                st.metric(_('total_crossings'), stats['total_crossings'])
            
            # Show zone details if available
            if stats['zone_details']:
                with st.expander(f"📋 {_('zone_details')}", expanded=False):
                    for zone_name, count in stats['zone_details'].items():
                        st.write(f"• **{zone_name}:** {count} {_('crossings')}")
                        
    def run(self):
        """Main application run method"""
        try:
            # Initialize language manager first
            language_manager.initialize()
            
            # Render main header
            self.render_main_header()
            
            # Render sidebar and get configurations
            task_type, confidence, source_type = self.render_sidebar()
            
            # Validate requirements before proceeding
            requirements_met, missing_components = self.validate_model_requirements(task_type)
            
            if not requirements_met:
                st.error(f"❌ {_('requirements_not_met')}")
                st.info(f"💡 {_('install_missing_components_to_continue')}")
                self.render_system_status_footer()
                return
            
            # Create main interface
            self.create_main_interface(task_type, confidence, source_type)
            
            # Render footer
            self.render_system_status_footer()
            
        except Exception as e:
            self.handle_application_error(e)
    
    def render_system_status_footer(self):
        # """Render enhanced system status footer"""
        # st.markdown("---")
        # st.markdown(f"""
        # <div class="system-footer">
        #     <h4>{_('vision_platform_title')}</h4>
        #     <p style="margin: 5px 0 0 0;">{_('professional_computer_vision_system')}</p>
        # </div>
        # """, unsafe_allow_html=True)
        pass
    
    
    def handle_application_error(self, error: Exception):
        """Handle application-level errors gracefully"""
        logger.error(f"Application error: {str(error)}")
        logger.error(traceback.format_exc())
        
        st.error(f"❌ {_('application_error')}: {str(error)}")
        
        # Error details for debugging
        with st.expander(f"🔍 {_('error_details')}", expanded=False):
            st.code(traceback.format_exc())
        
        # Recovery suggestions
        st.markdown(f"""
        ### 🔧 {_('recovery_suggestions')}:
        - {_('refresh_page_to_restart')}
        - {_('check_system_requirements')}
        - {_('verify_model_files_exist')}
        - {_('ensure_dependencies_installed')}
        """)
        
        # Emergency reset button
        if st.button(f"🆘 {_('emergency_reset')}", type="secondary"):
            self.emergency_reset()
    
    def emergency_reset(self):
        """Emergency system reset"""
        try:
            logger.info("Performing emergency reset")
            
            # Clear all session state except essential items
            essential_keys = ['session_start']
            keys_to_remove = [k for k in st.session_state.keys() if k not in essential_keys]
            
            for key in keys_to_remove:
                try:
                    del st.session_state[key]
                except:
                    pass
            
            # Reinitialize essential components
            initialize_session_state()
            
            st.success(f"✅ {_('emergency_reset_complete')}")
            st.info(f"💡 {_('page_will_refresh_automatically')}")
            
            time.sleep(2)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ {_('emergency_reset_failed')}: {str(e)}")
    
    def refresh_system(self):
        """Refresh system components"""
        try:
            # Clear analytics data
            st.session_state.detection_analytics = {
                'total_detections': 0,
                'unique_plates': set(),
                'detection_timeline': [],
                'confidence_scores': []
            }
            
            # Reset processing stats
            st.session_state.processing_stats = {
                'frames_processed': 0,
                'detections_count': 0,
                'processing_fps': 0.0,
                'start_time': time.time()
            }
            
            # Clear event histories
            st.session_state.counting_events_history.clear()
            st.session_state.parking_events_log.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            st.success(f"✅ {_('system_refreshed')}")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ {_('refresh_failed')}: {str(e)}")
    
    def quick_export_session(self):
        """Quick export of current session"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_duration = time.time() - st.session_state.session_start
            
            quick_export = {
                'export_type': 'quick_session_export',
                'timestamp': timestamp,
                'session_duration_minutes': session_duration / 60,
                'stats': st.session_state.processing_stats,
                'detections_summary': {
                    'total_timeline_entries': len(st.session_state.detection_analytics['detection_timeline']),
                    'unique_plates_count': len(st.session_state.detection_analytics['unique_plates']),
                    'unique_plates_list': list(st.session_state.detection_analytics['unique_plates'])
                },
                'system_info': {
                    'enhanced_license_plate_available': ENHANCED_LICENSE_PLATE_AVAILABLE,
                    'parking_available': PARKING_AVAILABLE,
                    'counting_available': COUNTING_AVAILABLE,
                    'main_module_available': MAIN_MODULE_AVAILABLE
                }
            }
            
            json_string = json.dumps(quick_export, indent=2, default=str)
            
            st.download_button(
                label=f"📥 {_('download_quick_export')}",
                data=json_string,
                file_name=f"quick_session_{timestamp}.json",
                mime="application/json",
                key="footer_quick_export"
            )
            
        except Exception as e:
            st.error(f"❌ {_('quick_export_failed')}: {str(e)}")
    
    def cleanup_on_exit(self):
        """Cleanup function to run when app is closing"""
        try:
            logger.info("Performing application cleanup")
            
            # Stop all live processors
            if hasattr(self, 'live_processors'): # Check if live_processors exists
                for camera_id in list(self.live_processors.keys()): # Iterate over a copy of keys
                    processor = self.live_processors.pop(camera_id, None) # Remove and get
                    if processor:
                        try:
                            logger.info(f"Stopping live processor for camera {camera_id} on exit.")
                            processor.stop_processing()
                        except Exception as e:
                            logger.error(f"Error stopping processor for camera {camera_id} on exit: {e}")
            
            # Close camera connections (managed by VisionPlatform's main camera_managers)
            if 'camera_managers' in st.session_state:
                for camera_id in list(st.session_state.camera_managers.keys()):
                    try:
                        camera_manager = st.session_state.camera_managers[camera_id]
                        if hasattr(camera_manager, 'stop'):
                            camera_manager.stop()
                        elif hasattr(camera_manager, 'release'):
                            camera_manager.release()
                    except:
                        pass
            
            # Finalize processors
            if st.session_state.license_processor:
                try:
                    if hasattr(st.session_state.license_processor, 'finalize_session'):
                        st.session_state.license_processor.finalize_session()
                except:
                    pass
            
            if st.session_state.parking_detector:
                try:
                    if hasattr(st.session_state.parking_detector, 'finalize_session'):
                        st.session_state.parking_detector.finalize_session()
                except:
                    pass
            
            # Clear GPU memory
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def debug_language_state(self):
        """Debug method to check language state (optional)"""
        if st.sidebar.button("🔍 Debug Language"):
            st.sidebar.write("**Language Debug Info:**")
            st.sidebar.write(f"Session Language: {st.session_state.get('language', 'NOT SET')}")
            st.sidebar.write(f"Manager Current: {language_manager.current_language}")
            
            # Test translation
            test_key = 'app_title'
            st.sidebar.write(f"Test Translation ({test_key}): {_(test_key)}")
            
            # Query params
            try:
                query_params = st.experimental_get_query_params()
                st.sidebar.write(f"Query Params: {query_params}")
            except:
                st.sidebar.write("Query Params: ERROR")
                
    def generate_final_license_plate_results(self, task_type: str):
        """Generate final license plate results with unique plates per car"""
        try:
            if task_type != "Enhanced License Plate" or not st.session_state.license_processor:
                st.warning("Final results generation only available for Enhanced License Plate mode")
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"final_results_{timestamp}.csv"
            
            # Generate final results (frames_data would need to be collected during processing)
            final_csv = st.session_state.license_processor.generate_final_results(
                frames_data=getattr(st.session_state, 'processed_frames_data', {}),
                output_path=output_path
            )
            
            if final_csv:
                st.success(f"✅ Final results generated: {final_csv}")
                
                # Display summary
                try:
                    import pandas as pd
                    df = pd.read_csv(final_csv)
                    
                    st.subheader("📊 Final Results Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Unique Cars", len(df))
                    with col2:
                        st.metric("Avg Confidence", f"{df['confidence'].mean():.3f}")
                    with col3:
                        color_counts = df['car_color'].value_counts()
                        most_common_color = color_counts.index[0] if len(color_counts) > 0 else "Unknown"
                        st.metric("Most Common Color", most_common_color)
                    
                    # Display data preview
                    st.dataframe(df[['Car_id', 'license_plate_number', 'confidence', 'car_color']], use_container_width=True)
                    
                    # Download button
                    csv_string = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Final Results CSV",
                        data=csv_string,
                        file_name=final_csv,
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error displaying final results: {e}")
            
            return final_csv
            
        except Exception as e:
            st.error(f"Error generating final results: {e}")
            return None
                
    def check_streamlit_version(self):
        """Check Streamlit version for debugging"""
        try:
            import streamlit
            version = streamlit.__version__
            st.sidebar.write(f"Streamlit Version: {version}")
            
            # Test query params API
            try:
                query_params = st.query_params
                st.sidebar.write("✅ New query params API available")
            except:
                st.sidebar.write("❌ Old query params API")
                
        except Exception as e:
            st.sidebar.write(f"Version check error: {e}")
            
    def __del__(self):
        """Destructor to ensure cleanup, particularly for LiveVideoProcessor threads"""
        logger.info("VisionPlatform destructor called.")
        if hasattr(self, 'live_processors'):
            for cam_id in list(self.live_processors.keys()):
                processor = self.live_processors.pop(cam_id, None)
                if processor:
                    try:
                        logger.info(f"Stopping live processor for camera {cam_id} in destructor.")
                        processor.stop_processing()
                    except Exception as e:
                        logger.error(f"Error stopping processor for camera {cam_id} in destructor: {e}")
        self.cleanup_on_exit() # Call the main cleanup
    
    def ensure_analytics_initialized(self):
        """Ensure analytics are properly initialized"""
        if 'detection_analytics' not in st.session_state:
            st.session_state.detection_analytics = {
                'total_detections': 0,
                'unique_plates': set(),
                'detection_timeline': [],
                'confidence_scores': []
            }
            logger.info("Analytics reinitialized")

    def safe_update_analytics(self, results: Dict):
        """Safely update analytics with error handling"""
        try:
            self.ensure_analytics_initialized()
            self.update_detection_analytics(results)
        except Exception as e:
            logger.error(f"Error in safe analytics update: {e}")
            self.ensure_analytics_initialized()



# Main application initialization and startup
def main():
    """Main application entry point"""
    try:
        # Initialize session state FIRST
        initialize_session_state()
        
        # Initialize language manager
        language_manager.initialize()
        
        # Create and run the vision platform
        app = VisionPlatform()
        app.run()
        
        
    except Exception as e:
        logger.error(f"Application startup error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Ensure basic session state exists for error handling
        if 'detection_analytics' not in st.session_state:
            st.session_state.detection_analytics = {
                'total_detections': 0,
                'unique_plates': set(),
                'detection_timeline': [],
                'confidence_scores': []
            }
        
        st.error(f"❌ {_('application_startup_error')}: {str(e)}")
        
        # Emergency fallback interface
        st.markdown(f"""
        ## 🆘 {_('emergency_mode')}
        
        {_('application_encountered_startup_error')}
        
        ### 🔧 {_('troubleshooting_steps')}:
        1. {_('refresh_browser_page')}
        2. {_('check_all_dependencies_installed')}
        3. {_('verify_model_files_in_weight_models_directory')}
        4. {_('ensure_required_modules_available')}
        """)
        
        
        # Basic system check in emergency mode
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Python Version:** {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
            st.write(f"**Streamlit Version:** {st.__version__}")
            st.write(f"**PyTorch Available:** {'✅' if 'torch' in sys.modules else '❌'}")
        
        with col2:
            st.write(f"**OpenCV Available:** {'✅' if 'cv2' in sys.modules else '❌'}")
            st.write(f"**NumPy Available:** {'✅' if 'numpy' in sys.modules else '❌'}")
            st.write(f"**Pandas Available:** {'✅' if 'pandas' in sys.modules else '❌'}")
        
        # Show error details
        with st.expander(f"🔍 {_('error_details')}", expanded=False):
            st.code(traceback.format_exc())


# Application entry point check
if __name__ == "__main__":
    # Register cleanup function
    import atexit
    
    def cleanup_handler():
        try:
            # Try to access the app instance if it exists
            if 'app' in locals():
                app.cleanup_on_exit()
        except:
            pass
    
    atexit.register(cleanup_handler)
    
    # Run the main application
    main()


# Export important functions and classes for external use
__all__ = [
    'VisionPlatform',
    'initialize_session_state',
    'main',
    'ENHANCED_LICENSE_PLATE_AVAILABLE',
    'PARKING_AVAILABLE',
    'COUNTING_AVAILABLE',
    'MODEL_CONFIGS',
    'CAMERA_MAP'
]


# Version information
__version__ = "1.0.0"
__author__ = "Vision Platform Team"
__description__ = "Enhanced YOLO11 Vision Platform with Multi-language Support"


# Final system validation
try:
    # Verify critical imports
    assert 'streamlit' in sys.modules, "Streamlit not available"
    assert 'cv2' in sys.modules, "OpenCV not available"
    assert 'torch' in sys.modules, "PyTorch not available"
    
    # Verify directory structure
    assert MODEL_DIR.exists(), f"Model directory not found: {MODEL_DIR}"
    assert IMAGES_DIR.exists() or True, f"Images directory not found: {IMAGES_DIR}"  # Optional
    assert VIDEO_DIR.exists() or True, f"Video directory not found: {VIDEO_DIR}"    # Optional
    
    logger.info("✅ All critical components validated successfully")
    
except AssertionError as e:
    logger.error(f"❌ Critical component validation failed: {e}")
except Exception as e:
    logger.error(f"❌ Unexpected validation error: {e}")


# Application ready message
logger.info("🚀 Vision Platform ready for initialization")
logger.info(f"📊 Available features: ELP={ENHANCED_LICENSE_PLATE_AVAILABLE}, Parking={PARKING_AVAILABLE}, Counting={COUNTING_AVAILABLE}")