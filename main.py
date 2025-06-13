# main.py - Enhanced with Real-time Visualization and Interpolation

import cv2
import urllib.parse
import time
import numpy as np
import torch
import logging
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import threading
import queue
from pathlib import Path
import ast
from scipy.interpolate import interp1d
from collections import defaultdict
import pandas as pd
import base64
from sort.sort import Sort

# YOLO and tracking imports
from ultralytics import YOLO
# try:
#     from sort.sort import Sort
#     SORT_AVAILABLE = True
# except ImportError:
#     print("Warning: SORT tracker not available. Vehicle tracking disabled.")
#     SORT_AVAILABLE = False

# License plate processing imports
try:
    from utils import (
        read_license_plate, 
        get_car, 
        write_csv,
        license_complies_format,
        generate_final_results_csv
    )
    import easyocr
    LICENSE_PLATE_UTILS_AVAILABLE = True
except ImportError:
    print("Warning: License plate utilities not available. OCR processing disabled.")
    LICENSE_PLATE_UTILS_AVAILABLE = False

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NVR and camera configuration
username = "admin"
password = "d@t@rium2023"
port = "554"
camera_map = {
    '1': "10.20.1.7",
    '2': "10.20.1.8", 
    '3': "10.20.1.18",
    '4': "10.20.1.19"
}

class_config = {
    0: {"name": "Person", "color": (0, 255, 0)},
    1: {"name": "Bicycle", "color": (255, 165, 0)},
    2: {"name": "Car", "color": (255, 0, 0)},
    3: {"name": "Motorcycle", "color": (255, 128, 0)},
    5: {"name": "Bus", "color": (128, 0, 255)},
    7: {"name": "Truck", "color": (255, 0, 255)},
    15: {"name": "Cat", "color": (0, 255, 255)},
}

# Vehicle classes for license plate detection
VEHICLE_CLASSES = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck

# PyTorch device selection
def get_pytorch_device():
    """Selects CUDA device if available, otherwise CPU, and logs the choice."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)  # Use the first GPU
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA is not available. Using CPU.")
    return device

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    """
    Draw custom corner borders around vehicles (from visualize.py)
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Ensure coordinates are within image bounds
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    
    # Adjust line lengths to fit within bounding box
    line_length_x = min(line_length_x, (x2 - x1) // 3)
    line_length_y = min(line_length_y, (y2 - y1) // 3)

    try:
        # Top-left corner
        cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
        cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

        # Bottom-left corner
        cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
        cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

        # Top-right corner
        cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

        # Bottom-right corner
        cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
        cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    except Exception as e:
        logger.warning(f"Error drawing border: {e}")

    return img

def interpolate_bbox(prev_bbox, curr_bbox, alpha):
    """Interpolate between two bounding boxes"""
    if prev_bbox is None or curr_bbox is None:
        return curr_bbox
    
    x1 = prev_bbox[0] + alpha * (curr_bbox[0] - prev_bbox[0])
    y1 = prev_bbox[1] + alpha * (curr_bbox[1] - prev_bbox[1])
    x2 = prev_bbox[2] + alpha * (curr_bbox[2] - prev_bbox[2])
    y2 = prev_bbox[3] + alpha * (curr_bbox[3] - prev_bbox[3])
    
    return [x1, y1, x2, y2]

class RTSPCameraManager:
    """Simplified camera manager focused on reliability"""
    
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.camera_ip = camera_map[str(camera_id)]
        self.channel = camera_id
        self.encoded_password = urllib.parse.quote(password)
        self.cap = None
        self.connected = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = None
        self.stop_event = threading.Event()
        self.frame_count = 0
        self.fps = 0
        self.last_frame_time = time.time()
        self.working_url = None # Added to store the working URL
        
        # RTSP URLs to try - Prioritized based on mainVideo.py logic
        self.rtsp_urls = [
            f"rtsp://{username}:{self.encoded_password}@{self.camera_ip}:{port}/h264/ch{self.channel}/main/av_stream?rtsp_transport=tcp",
            f"rtsp://{username}:{self.encoded_password}@{self.camera_ip}:{port}/h264/ch{self.channel}/sub/av_stream?rtsp_transport=tcp",
            f"rtsp://{username}:{self.encoded_password}@{self.camera_ip}:{port}/Streaming/Channels/{self.channel}01?rtsp_transport=tcp", # Alternative main
            f"rtsp://{username}:{self.encoded_password}@{self.camera_ip}:{port}/Streaming/Channels/{self.channel}02?rtsp_transport=tcp", # Alternative sub
        ]
    
    def connect(self) -> bool:
        """Try to connect to the camera"""
        for url_index, url in enumerate(self.rtsp_urls): # Added url_index for logging
            logger.info(f"Trying to connect to camera {self.camera_id} with URL ({url_index+1}/{len(self.rtsp_urls)}): {url}")
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            # Configure capture properties
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
            
            if cap.isOpened():
                # Test frame reading
                success_count = 0
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        success_count += 1
                    time.sleep(0.1)
                
                if success_count >= 3:  # At least 3 successful reads
                    self.cap = cap
                    self.connected = True
                    self.working_url = url
                    logger.info(f"Camera {self.camera_id} connected successfully")
                    return True
                else:
                    cap.release()
            else:
                cap.release()
        
        logger.error(f"Failed to connect to camera {self.camera_id}")
        return False
    
    def start_capture_thread(self):
        """Start background frame capture thread"""
        if self.connected:
            self.stop_event.clear()
            self.capture_thread = threading.Thread(target=self._capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            logger.info(f"Started capture thread for camera {self.camera_id}")
    
    def _capture_frames(self):
        """Background thread for continuous frame capture"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while not self.stop_event.is_set() and self.connected:
            try:
                ret, frame = self.cap.read()
                current_time = time.time()
                
                if not ret or frame is None or frame.size == 0:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.warning(f"Camera {self.camera_id}: Too many consecutive errors, attempting reconnection")
                        self.connected = False
                        self.cap.release()
                        if self.connect():
                            consecutive_errors = 0
                            continue
                        else:
                            break
                    time.sleep(0.1)
                    continue
                
                # Calculate FPS
                if self.last_frame_time > 0:
                    time_diff = current_time - self.last_frame_time
                    if time_diff > 0:
                        self.fps = 0.8 * self.fps + 0.2 * (1.0 / time_diff)
                
                self.last_frame_time = current_time
                consecutive_errors = 0
                self.frame_count += 1
                
                # Add frame to queue (non-blocking)
                try:
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    # Remove oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, block=False)
                    except queue.Empty:
                        pass
                        
            except Exception as e:
                logger.error(f"Camera {self.camera_id} capture error: {str(e)}")
                consecutive_errors += 1
                time.sleep(0.1)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get latest frame from queue"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop capture thread and release resources"""
        self.stop_event.set()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
        if self.cap:
            self.cap.release()
        self.connected = False
        logger.info(f"Camera {self.camera_id} stopped")

class EnhancedLicensePlateProcessor:
    """Enhanced License Plate Processor with Real-time Visualization and Interpolation"""
    
    def __init__(self, vehicle_model_path: str, plate_model_path: str, confidence: float = 0.4):
        """
        Initialize enhanced license plate processing system
        """
        self.confidence = confidence
        self.results = {}
        self.frame_number = 0
        
        # Real-time tracking data
        self.vehicle_history = defaultdict(lambda: deque(maxlen=30))  # 30 frame history
        self.license_plates = {}  # Best license plate for each vehicle
        self.last_detection_time = defaultdict(float)
        
        # Check if required components are available
        if not LICENSE_PLATE_UTILS_AVAILABLE:
            raise ImportError("License plate utilities not available.")
        
        if not SORT_AVAILABLE:
            raise ImportError("SORT tracker not available.")
        
        # Initialize models
        try:
            self.vehicle_model = YOLO(vehicle_model_path)
            self.plate_model = YOLO(plate_model_path)
            logger.info("License plate models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load license plate models: {e}")
            raise
        
        # Initialize tracker and OCR
        self.mot_tracker = Sort()
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("OCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR reader: {e}")
            raise
    
    def process_frame_with_visualization(self, frame: np.ndarray, enable_interpolation: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Process frame with real-time visualization and interpolation
        """
        if frame is None:
            return frame, {}
        
        self.frame_number += 1
        frame_results = {}
        
        try:
            # Step 1: Detect vehicles
            vehicle_detections = self.vehicle_model(frame)[0]
            vehicle_boxes = []
            
            for detection in vehicle_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in VEHICLE_CLASSES and score > self.confidence:
                    vehicle_boxes.append([x1, y1, x2, y2, score])
            
            # Step 2: Track vehicles
            if len(vehicle_boxes) > 0:
                track_ids = self.mot_tracker.update(np.asarray(vehicle_boxes))
            else:
                track_ids = []
            
            # Step 3: Update vehicle history for interpolation
            current_time = time.time()
            for track in track_ids:
                xcar1, ycar1, xcar2, ycar2, car_id = track
                car_id = int(car_id)
                
                # Store vehicle history
                self.vehicle_history[car_id].append({
                    'frame': self.frame_number,
                    'bbox': [xcar1, ycar1, xcar2, ycar2],
                    'timestamp': current_time
                })
                self.last_detection_time[car_id] = current_time
            
            # Step 4: Detect license plates
            plate_detections = self.plate_model(frame)[0]
            
            # Step 5: Process license plates
            for plate_detection in plate_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = plate_detection
                
                if score > self.confidence:
                    # Associate license plate with vehicle
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(plate_detection, track_ids)
                    
                    if car_id != -1:
                        car_id = int(car_id)
                        
                        # Crop and process license plate
                        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                        
                        if license_plate_crop.size > 0:
                            license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                            _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)
                            
                            # Read license plate text
                            license_plate_text, text_score = read_license_plate(license_plate_thresh)
                            
                            if license_plate_text is not None:
                                # Store results
                                frame_results[car_id] = {
                                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                    'license_plate': {
                                        'bbox': [x1, y1, x2, y2],
                                        'text': license_plate_text,
                                        'bbox_score': score,
                                        'text_score': text_score
                                    }
                                }
                                
                                # Update best license plate for this vehicle
                                if (car_id not in self.license_plates or 
                                    text_score > self.license_plates[car_id].get('text_score', 0)):
                                    
                                    # Resize license plate crop for overlay
                                    h, w = license_plate_crop.shape[:2]
                                    if h > 0 and w > 0:
                                        aspect_ratio = w / h
                                        new_height = 100
                                        new_width = int(new_height * aspect_ratio)
                                        license_crop_resized = cv2.resize(license_plate_crop, (new_width, new_height))
                                        
                                        self.license_plates[car_id] = {
                                            'text': license_plate_text,
                                            'text_score': text_score,
                                            'bbox_score': score,
                                            'crop': license_crop_resized,
                                            'frame': self.frame_number
                                        }
                                
                                logger.info(f"License plate detected: {license_plate_text} (Car ID: {car_id})")
            
            # Store frame results
            if frame_results:
                self.results[self.frame_number] = frame_results
            
            # Step 6: Create enhanced visualization
            annotated_frame = self._create_enhanced_visualization(
                frame, track_ids, frame_results, enable_interpolation
            )
            
            return annotated_frame, {
                'frame_number': self.frame_number,
                'vehicles_detected': len(track_ids),
                'plates_detected': len([p for p in plate_detections.boxes.data.tolist() if p[4] > self.confidence]),
                'plates_read': len(frame_results),
                'results': frame_results,
                'tracked_vehicles': len(self.license_plates)
            }
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, {'error': str(e)}
    
    def _create_enhanced_visualization(self, frame: np.ndarray, track_ids: List, 
                                     frame_results: Dict, enable_interpolation: bool) -> np.ndarray:
        """Create enhanced visualization with professional styling"""
        
        annotated_frame = frame.copy()
        current_time = time.time()
        
        # Draw vehicles with custom borders and interpolation
        active_vehicles = set()
        
        for track in track_ids:
            xcar1, ycar1, xcar2, ycar2, car_id = track
            car_id = int(car_id)
            active_vehicles.add(car_id)
            
            # Draw custom vehicle border (green corner lines)
            draw_border(annotated_frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), 
                       (0, 255, 0), 8, line_length_x=60, line_length_y=60)
            
            # Draw vehicle ID
            cv2.putText(annotated_frame, f"ID: {car_id}", (int(xcar1), int(ycar1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw license plate detections and overlays
        for car_id, result in frame_results.items():
            plate_bbox = result['license_plate']['bbox']
            x1, y1, x2, y2 = map(int, plate_bbox)
            
            # Draw license plate bounding box (red)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Draw persistent license plate information for all tracked vehicles
        for car_id, plate_info in self.license_plates.items():
            # Only show if vehicle was recently detected (within last 5 seconds)
            if current_time - self.last_detection_time[car_id] < 5.0:
                
                # Get latest vehicle position
                if car_id in self.vehicle_history and self.vehicle_history[car_id]:
                    latest_vehicle = self.vehicle_history[car_id][-1]
                    car_bbox = latest_vehicle['bbox']
                    xcar1, ycar1, xcar2, ycar2 = map(int, car_bbox)
                    
                    # Calculate position for license plate overlay
                    license_crop = plate_info['crop']
                    H, W = license_crop.shape[:2]
                    
                    # Position above the vehicle
                    overlay_y1 = max(0, ycar1 - H - 20)
                    overlay_y2 = overlay_y1 + H
                    overlay_x1 = max(0, int((xcar1 + xcar2 - W) / 2))
                    overlay_x2 = min(annotated_frame.shape[1], overlay_x1 + W)
                    
                    # Adjust if overlay goes outside frame
                    if overlay_x2 > annotated_frame.shape[1]:
                        overlay_x1 = annotated_frame.shape[1] - W
                        overlay_x2 = annotated_frame.shape[1]
                    
                    try:
                        # Create white background for license plate
                        cv2.rectangle(annotated_frame, (overlay_x1 - 5, overlay_y1 - 40), 
                                     (overlay_x2 + 5, overlay_y2 + 5), (255, 255, 255), -1)
                        cv2.rectangle(annotated_frame, (overlay_x1 - 5, overlay_y1 - 40), 
                                     (overlay_x2 + 5, overlay_y2 + 5), (0, 0, 0), 2)
                        
                        # Overlay license plate crop
                        if (overlay_y2 <= annotated_frame.shape[0] and overlay_x2 <= annotated_frame.shape[1] and
                            overlay_y1 >= 0 and overlay_x1 >= 0):
                            annotated_frame[overlay_y1:overlay_y2, overlay_x1:overlay_x2] = license_crop
                        
                        # Add license plate text
                        text = plate_info['text']
                        confidence = plate_info['text_score']
                        
                        # Calculate text position
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                        text_x = overlay_x1 + (W - text_width) // 2
                        text_y = overlay_y1 - 10
                        
                        cv2.putText(annotated_frame, text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                        
                        # Add confidence score
                        conf_text = f"Conf: {confidence:.2f}"
                        cv2.putText(annotated_frame, conf_text, (overlay_x1, overlay_y1 - 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                        
                    except Exception as e:
                        logger.warning(f"Error drawing license plate overlay: {e}")
        
        # Add frame info
        cv2.putText(annotated_frame, f"Frame: {self.frame_number}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Tracked Vehicles: {len(active_vehicles)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Known Plates: {len(self.license_plates)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return annotated_frame
    
    def extract_license_plate_crop(self, frame, bbox):
        """Extract and encode license plate crop"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                # Resize for consistency
                crop_resized = cv2.resize(crop, (200, 50))
                
                # Convert to base64 string
                _, buffer = cv2.imencode('.jpg', crop_resized)
                crop_base64 = base64.b64encode(buffer).decode('utf-8')
                return crop_base64
            
        except Exception as e:
            logger.error(f"Error extracting license plate crop: {e}")
        
        return ""

    def extract_car_crop_and_color(self, frame, bbox):
        """Extract car crop and determine color"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            car_crop = frame[y1:y2, x1:x2]
            
            if car_crop.size > 0:
                from utils import get_car_color
                car_color = get_car_color(car_crop)
                return car_color
            
        except Exception as e:
            logger.error(f"Error extracting car color: {e}")
        
        return "Unknown"

    def generate_final_results(self, frames_data=None, output_path="final_results.csv"):
        """
        Generate final results CSV with unique license plates per car
        
        Args:
            frames_data: Dictionary mapping frame numbers to actual frame data
            output_path: Output CSV file path
            
        Returns:
            str: Path to generated CSV file
        """
        try:
            
            
            if not self.results:
                logger.warning("No results to process for final CSV")
                return None
            
            final_data = []
            car_license_map = defaultdict(list)
            
            # Group detections by car_id
            for frame_num, frame_data in self.results.items():
                for car_id, detection in frame_data.items():
                    if 'license_plate' in detection and 'text' in detection['license_plate']:
                        car_license_map[car_id].append({
                            'frame': frame_num,
                            'detection': detection,
                            'confidence': detection['license_plate']['text_score']
                        })
            
            # Process each car
            for car_id, detections in car_license_map.items():
                # Get best detection (highest confidence)
                best_detection_entry = max(detections, key=lambda x: x['confidence'])
                best_detection = best_detection_entry['detection']
                best_frame = best_detection_entry['frame']
                
                # Extract license plate crop if frame data available
                license_crop_base64 = ""
                car_color = "Unknown"
                
                if frames_data and best_frame in frames_data:
                    frame = frames_data[best_frame]
                    
                    # Extract license plate crop
                    if 'license_plate' in best_detection and 'bbox' in best_detection['license_plate']:
                        license_crop_base64 = self.extract_license_plate_crop(
                            frame, best_detection['license_plate']['bbox']
                        )
                    
                    # Extract car color
                    if 'car' in best_detection and 'bbox' in best_detection['car']:
                        car_color = self.extract_car_crop_and_color(
                            frame, best_detection['car']['bbox']
                        )
                
                final_data.append({
                    'Car_id': car_id,
                    'license_plate_number': best_detection['license_plate']['text'],
                    'confidence': best_detection['license_plate']['text_score'],
                    'car_color': car_color,
                    'license_plate_crop': license_crop_base64
                })
            
            # Create DataFrame and save
            df = pd.DataFrame(final_data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Final results saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating final results: {e}")
            return None
    def save_results(self, output_path: str = None) -> str:
        """Save detection results to CSV file"""
        if not self.results:
            logger.warning("No results to save")
            return None
        
        if output_path is None:
            output_path = f"license_plate_results_{int(time.time())}.csv"
        
        try:
            write_csv(self.results, output_path)
            logger.info(f"Results saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None
    
    def get_summary(self) -> Dict:
        """Get comprehensive summary of detection results"""
        total_frames = len(self.results)
        total_detections = sum(len(frame_data) for frame_data in self.results.values())
        
        # Get unique license plates with their best scores
        unique_plates = {}
        for frame_data in self.results.values():
            for car_data in frame_data.values():
                if 'license_plate' in car_data and 'text' in car_data['license_plate']:
                    plate_text = car_data['license_plate']['text']
                    score = car_data['license_plate']['text_score']
                    if plate_text not in unique_plates or score > unique_plates[plate_text]:
                        unique_plates[plate_text] = score
        
        return {
            'total_frames_processed': self.frame_number,
            'frames_with_detections': total_frames,
            'total_license_plates_detected': total_detections,
            'unique_license_plates': len(unique_plates),
            'unique_plate_texts': list(unique_plates.keys()),
            'plate_scores': unique_plates,
            'tracked_vehicles': len(self.license_plates)
        }

def initialize_enhanced_license_plate_system(vehicle_model_path: str, plate_model_path: str, 
                                           confidence: float = 0.4) -> EnhancedLicensePlateProcessor:
    """Initialize enhanced license plate detection system"""
    try:
        processor = EnhancedLicensePlateProcessor(vehicle_model_path, plate_model_path, confidence)
        logger.info("Enhanced license plate system initialized successfully")
        return processor
    except Exception as e:
        logger.error(f"Failed to initialize enhanced license plate system: {e}")
        raise

def process_license_plates_with_visualization(frame: np.ndarray, processor: EnhancedLicensePlateProcessor, 
                                            enable_interpolation: bool = True) -> Tuple[np.ndarray, Dict]:
    """Process frame with enhanced visualization"""
    return processor.process_frame_with_visualization(frame, enable_interpolation)

# Camera connection functions (for backward compatibility)
def get_camera_connection(camera_id):
    """Get camera connection (legacy function)"""
    camera_manager = RTSPCameraManager(camera_id)
    if camera_manager.connect():
        return camera_manager.cap
    return None

def camera_1(): return get_camera_connection(1)
def camera_2(): return get_camera_connection(2)
def camera_3(): return get_camera_connection(3)
def camera_4(): return get_camera_connection(4)

def connect_all_cameras_enhanced():
    """Connect all cameras with enhanced management"""
    camera_managers = {}
    
    for i in range(1, 5):
        camera_manager = RTSPCameraManager(i)
        if camera_manager.connect():
            camera_manager.start_capture_thread()
            camera_managers[f"camera_{i}"] = camera_manager
            logger.info(f"Camera {i} connected and started")
        else:
            logger.warning(f"Failed to connect camera {i}")
            camera_managers[f"camera_{i}"] = None
    
    return camera_managers

def connect_all_cameras():
    """Legacy function for backward compatibility"""
    camera_vars = {}
    for i in range(1, 5):
        cap = get_camera_connection(i)
        camera_vars[f"camera_{i}"] = cap
        if cap:
            logger.info(f"Camera {i} connected")
        else:
            logger.warning(f"Camera {i} failed to connect")
    return camera_vars

# Example usage and main function
if __name__ == '__main__':
    # Test enhanced license plate system
    if LICENSE_PLATE_UTILS_AVAILABLE and SORT_AVAILABLE:
        print("Testing enhanced license plate detection system...")
        
        try:
            # Initialize enhanced processor
            processor = initialize_enhanced_license_plate_system("yolo11n.pt", "./models/license_plate_detector.pt")
            print("Enhanced license plate system initialized successfully")
            
            # Test with camera if available
            camera_managers = connect_all_cameras_enhanced()
            
            # Test first available camera
            for i in range(1, 5):
                camera_manager = camera_managers[f"camera_{i}"]
                if camera_manager is not None and camera_manager.connected:
                    print(f"Testing enhanced license plate detection with Camera {i}")
                    
                    # Process frames with real-time visualization
                    for frame_count in range(10):
                        frame = camera_manager.get_frame(timeout=2.0)
                        if frame is not None:
                            processed_frame, results = process_license_plates_with_visualization(frame, processor)
                            print(f"Frame {frame_count}: {results.get('plates_read', 0)} plates read, "
                                  f"{results.get('tracked_vehicles', 0)} vehicles tracked")
                            
                            # Display frame (comment out for headless operation)
                            # cv2.imshow(f"Enhanced License Plate Detection - Camera {i}", processed_frame)
                            # if cv2.waitKey(1) & 0xFF == ord('q'):
                            #     break
                        time.sleep(0.5)
                    
                    # Get summary and save results
                    summary = processor.get_summary()
                    print("Enhanced Detection Summary:", summary)
                    
                    # Save results if any
                    if processor.results:
                        output_file = processor.save_results()
                        print(f"Results saved to: {output_file}")
                    
                    # Clean up
                    cv2.destroyAllWindows()
                    break
            
            # Cleanup all cameras
            for i in range(1, 5):
                camera_manager = camera_managers[f"camera_{i}"]
                if camera_manager is not None:
                    camera_manager.stop()
        
        except Exception as e:
            print(f"Error testing enhanced license plate system: {e}")
    
    else:
        print("Enhanced license plate detection not available - missing dependencies")
        
        # Test basic camera connections
        camera_managers = connect_all_cameras_enhanced()
        
        # Test each camera
        for i in range(1, 5):
            camera_manager = camera_managers[f"camera_{i}"]
            if camera_manager is not None and camera_manager.connected:
                print(f"Camera {i}: Connected and working")
                
                # Test frame capture
                frame = camera_manager.get_frame(timeout=2.0)
                if frame is not None:
                    print(f"Camera {i}: Frame captured successfully ({frame.shape})")
                else:
                    print(f"Camera {i}: Failed to capture frame")
            else:
                print(f"Camera {i}: Not connected")
        
        # Cleanup
        for i in range(1, 5):
            camera_manager = camera_managers[f"camera_{i}"]
            if camera_manager is not None:
                camera_manager.stop()
    
    logger.info("Enhanced test completed")

# Additional utility functions for integration

def create_video_writer(output_path: str, fps: float, width: int, height: int):
    """Create video writer for saving processed video"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def process_video_file_with_enhanced_visualization(video_path: str, output_path: str, 
                                                 vehicle_model_path: str, plate_model_path: str,
                                                 confidence: float = 0.4) -> Dict:
    """
    Process entire video file with enhanced license plate detection and visualization
    Similar to the original workflow but with real-time visualization
    """
    if not LICENSE_PLATE_UTILS_AVAILABLE or not SORT_AVAILABLE:
        raise ImportError("License plate utilities not available")
    
    # Initialize processor
    processor = initialize_enhanced_license_plate_system(vehicle_model_path, plate_model_path, confidence)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    out = create_video_writer(output_path, fps, width, height)
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with enhanced visualization
            processed_frame, results = process_license_plates_with_visualization(frame, processor)
            
            # Write to output video
            out.write(processed_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                progress = (frame_count / total_frames) * 100
                fps_processing = frame_count / elapsed
                eta = (total_frames - frame_count) / fps_processing if fps_processing > 0 else 0
                
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                      f"Processing FPS: {fps_processing:.1f} - ETA: {eta:.0f}s")
                
                if results.get('plates_read', 0) > 0:
                    print(f"  → License plates read in this batch: {results['plates_read']}")
    
    finally:
        # Cleanup
        cap.release()
        out.release()
    
    # Save CSV results
    csv_output = output_path.replace('.mp4', '_results.csv')
    processor.save_results(csv_output)
    
    # Get final summary
    summary = processor.get_summary()
    
    processing_time = time.time() - start_time
    print(f"\nVideo processing completed!")
    print(f"Output video: {output_path}")
    print(f"CSV results: {csv_output}")
    print(f"Processing time: {processing_time:.1f}s")
    print(f"Summary: {summary}")
    
    return {
        'output_video': output_path,
        'csv_results': csv_output,
        'processing_time': processing_time,
        'summary': summary
    }

def batch_process_videos(video_list: List[str], output_dir: str, 
                        vehicle_model_path: str, plate_model_path: str,
                        confidence: float = 0.4) -> List[Dict]:
    """
    Batch process multiple videos with enhanced license plate detection
    """
    results = []
    
    for i, video_path in enumerate(video_list):
        print(f"\n{'='*50}")
        print(f"Processing video {i+1}/{len(video_list)}: {video_path}")
        print(f"{'='*50}")
        
        try:
            # Generate output paths
            video_name = Path(video_path).stem
            output_video = Path(output_dir) / f"{video_name}_processed.mp4"
            
            # Process video
            result = process_video_file_with_enhanced_visualization(
                video_path, str(output_video), vehicle_model_path, plate_model_path, confidence
            )
            
            result['input_video'] = video_path
            result['success'] = True
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            results.append({
                'input_video': video_path,
                'success': False,
                'error': str(e)
            })
    
    return results

def live_demo_with_enhanced_visualization(camera_id: int = 1, 
                                        vehicle_model_path: str = "yolo11n.pt",
                                        plate_model_path: str = "./models/license_plate_detector.pt",
                                        confidence: float = 0.4,
                                        save_video: bool = False,
                                        output_path: str = None):
    """
    Live demo with enhanced license plate detection and visualization
    """
    if not LICENSE_PLATE_UTILS_AVAILABLE or not SORT_AVAILABLE:
        print("License plate utilities not available for live demo")
        return
    
    try:
        # Initialize enhanced processor
        processor = initialize_enhanced_license_plate_system(vehicle_model_path, plate_model_path, confidence)
        print("Enhanced license plate system initialized")
        
        # Connect to camera
        camera_manager = RTSPCameraManager(camera_id)
        if not camera_manager.connect():
            print(f"Failed to connect to camera {camera_id}")
            return
        
        camera_manager.start_capture_thread()
        print(f"Connected to camera {camera_id}")
        
        # Setup video writer if saving
        video_writer = None
        if save_video:
            if output_path is None:
                output_path = f"live_demo_camera_{camera_id}_{int(time.time())}.mp4"
            
            # Get frame to determine dimensions
            test_frame = camera_manager.get_frame(timeout=5.0)
            if test_frame is not None:
                h, w = test_frame.shape[:2]
                video_writer = create_video_writer(output_path, 20.0, w, h)
                print(f"Saving video to: {output_path}")
        
        print("\nStarting live demo...")
        print("Press 'q' to quit, 's' to save current results, 'r' to reset tracking")
        
        # Create window
        cv2.namedWindow(f'Enhanced License Plate Detection - Camera {camera_id}', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f'Enhanced License Plate Detection - Camera {camera_id}', 1280, 720)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            frame = camera_manager.get_frame(timeout=1.0)
            if frame is not None:
                # Process with enhanced visualization
                processed_frame, results = process_license_plates_with_visualization(frame, processor)
                
                # Add performance info
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                cv2.putText(processed_frame, f"Live FPS: {fps:.1f}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Display frame
                cv2.imshow(f'Enhanced License Plate Detection - Camera {camera_id}', processed_frame)
                
                # Save frame if recording
                if video_writer is not None:
                    video_writer.write(processed_frame)
                
                # Print detection info
                if results.get('plates_read', 0) > 0:
                    print(f"Frame {frame_count}: Detected plates - {[r['license_plate']['text'] for r in results['results'].values()]}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current results
                csv_file = processor.save_results(f"live_demo_results_{int(time.time())}.csv")
                print(f"Results saved to: {csv_file}")
                summary = processor.get_summary()
                print(f"Current summary: {summary}")
            elif key == ord('r'):
                # Reset tracking
                processor.license_plates.clear()
                processor.vehicle_history.clear()
                processor.last_detection_time.clear()
                print("Tracking data reset")
        
        # Cleanup
        cv2.destroyAllWindows()
        camera_manager.stop()
        
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {output_path}")
        
        # Final summary
        final_summary = processor.get_summary()
        print(f"\nLive demo completed!")
        print(f"Final summary: {final_summary}")
        
        if processor.results:
            final_csv = processor.save_results(f"live_demo_final_{int(time.time())}.csv")
            print(f"Final results saved to: {final_csv}")
    
    except Exception as e:
        print(f"Error in live demo: {e}")
    finally:
        cv2.destroyAllWindows()

# Integration helper functions for display.py

def get_enhanced_processor_for_display(vehicle_model_path: str, plate_model_path: str, 
                                     confidence: float = 0.4) -> Optional[EnhancedLicensePlateProcessor]:
    """
    Get enhanced processor instance for use in display.py
    Returns None if dependencies are not available
    """
    if not LICENSE_PLATE_UTILS_AVAILABLE or not SORT_AVAILABLE:
        return None
    
    try:
        return initialize_enhanced_license_plate_system(vehicle_model_path, plate_model_path, confidence)
    except Exception as e:
        logger.error(f"Failed to create enhanced processor: {e}")
        return None

def process_frame_for_display(frame: np.ndarray, processor: EnhancedLicensePlateProcessor) -> Tuple[np.ndarray, Dict]:
    """
    Process frame for display.py integration
    """
    if processor is None or frame is None:
        return frame, {}
    
    try:
        return process_license_plates_with_visualization(frame, processor, enable_interpolation=True)
    except Exception as e:
        logger.error(f"Error processing frame for display: {e}")
        return frame, {'error': str(e)}

# Export functions for display.py
__all__ = [
    'get_pytorch_device',
    'RTSPCameraManager', 
    'EnhancedLicensePlateProcessor',
    'initialize_enhanced_license_plate_system',
    'process_license_plates_with_visualization',
    'get_enhanced_processor_for_display',
    'process_frame_for_display',
    'camera_1', 'camera_2', 'camera_3', 'camera_4',
    'connect_all_cameras_enhanced',
    'LICENSE_PLATE_UTILS_AVAILABLE',
    'SORT_AVAILABLE',
    'live_demo_with_enhanced_visualization',
    'process_video_file_with_enhanced_visualization',
    'batch_process_videos'
]