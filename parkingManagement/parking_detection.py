# parking_detection.py - Modular Version with Enhanced YOLO Visualization
import cv2
import pickle
import cvzone
import numpy as np
import urllib.parse
from ultralytics import YOLO
import time
from datetime import datetime, timedelta
import csv
import os
from typing import List, Dict, Tuple, Optional, Any

class ParkingDetectionConfig:
    """Configuration class for parking detection system"""
    
    def __init__(self):
        # YOLO Configuration
        self.YOLO_MODEL_PATH = "yolo11n.pt"
        self.YOLO_CONFIDENCE = 0.4
        self.YOLO_TARGET_CLASSES_NAMES = ['car', 'truck', 'bus', 'motorcycle']
        self.YOLO_DEVICE = 'cpu'
        
        # Parking Detection Configuration
        self.PIXEL_THRESHOLD = 900
        self.DEFAULT_SPOT_WIDTH = 107
        self.DEFAULT_SPOT_HEIGHT = 48
        self.PARKING_POSITIONS_FILE = 'CarParkPos'
        
        # Visualization Configuration
        self.SHOW_YOLO_DETECTIONS = True  # NEW: Control YOLO visualization
        self.YOLO_BOX_COLOR = (255, 0, 255)  # Magenta/Pink for YOLO boxes
        self.YOLO_TEXT_COLOR = (255, 0, 255)  # Magenta/Pink for YOLO text
        self.PARKING_FREE_COLOR = (0, 255, 0)  # Green for free spots
        self.PARKING_OCCUPIED_COLOR = (0, 0, 255)  # Red for occupied spots

class ParkingSpotDetector:
    """Main class for parking spot detection and tracking"""
    
    def __init__(self, config: ParkingDetectionConfig = None):
        self.config = config or ParkingDetectionConfig()
        self.model = None
        self.parking_spots = []
        self.parking_spot_states = {}
        self.parking_events_log = []
        
        # Load YOLO model
        self._load_yolo_model()
        
        # Load parking positions
        self._load_parking_positions()
    
    def _load_yolo_model(self):
        """Load YOLO model with error handling"""
        try:
            self.model = YOLO(self.config.YOLO_MODEL_PATH)
            
            # Convert to FP32 for CPU compatibility
            if self.config.YOLO_DEVICE == 'cpu':
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'float'):
                    print(f"Converting model '{self.config.YOLO_MODEL_PATH}' parameters to FP32 for CPU execution.")
                    self.model.model.float()
            
            self.model.to(self.config.YOLO_DEVICE)
            print(f"YOLO model '{self.config.YOLO_MODEL_PATH}' loaded successfully on device '{self.config.YOLO_DEVICE}'.")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None
    
    def _load_parking_positions(self):
        """Load parking positions from pickle file"""
        try:
            if os.path.exists(self.config.PARKING_POSITIONS_FILE):
                with open(self.config.PARKING_POSITIONS_FILE, 'rb') as f:
                    self.parking_spots = pickle.load(f)
                
                # Initialize parking spot states
                for i in range(len(self.parking_spots)):
                    self.parking_spot_states[i] = {'object_class': None, 'entry_time': None}
                    
                print(f"Loaded {len(self.parking_spots)} parking spots from {self.config.PARKING_POSITIONS_FILE}")
            else:
                print(f"Warning: Parking positions file '{self.config.PARKING_POSITIONS_FILE}' not found")
                self.parking_spots = []
                
        except Exception as e:
            print(f"Error loading parking positions: {e}")
            self.parking_spots = []
    
    def detect_objects_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO model"""
        detections = []
        
        if not self.model:
            return detections
        
        try:
            results = self.model.predict(image, conf=self.config.YOLO_CONFIDENCE, 
                                       device=self.config.YOLO_DEVICE, verbose=False)
            
            if results and results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': class_name
                    })
                    
        except Exception as e:
            print(f"Error during YOLO detection: {e}")
        
        return detections
    
    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess image for parking spot analysis"""
        try:
            imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
            
            if imgBlur is None or imgBlur.size == 0:
                return None
            
            if imgBlur.dtype != np.uint8:
                imgBlur = np.uint8(imgBlur)
            
            # Check dimensions for adaptive threshold
            adaptive_block_size = 25
            if imgBlur.shape[0] < adaptive_block_size or imgBlur.shape[1] < adaptive_block_size:
                return None
            
            imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, adaptive_block_size, 16)
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
            
            return imgDilate
            
        except Exception as e:
            print(f"Error during image preprocessing: {e}")
            return None
    
    def check_parking_spaces(self, original_image: np.ndarray, processed_image: np.ndarray, 
                           yolo_detections: List[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Check parking spaces and draw both YOLO detections and parking analysis
        """
        
        if not self.parking_spots:
            return original_image, {
                'free_spaces': 0, 
                'occupied_spaces': 0,
                'total_spaces': 0,
                'yolo_detections': len(yolo_detections) if yolo_detections else 0
            }
        
        annotated_image = original_image.copy()
        space_counter = 0 # Counts free spaces
        
        # STEP 1: Draw all YOLO detections first (so parking spots are drawn on top)
        if self.config.SHOW_YOLO_DETECTIONS and yolo_detections:
            annotated_image = self._draw_yolo_detections(annotated_image, yolo_detections)
        
        # STEP 2: Process parking spots
        for spot_id, pos_entry in enumerate(self.parking_spots):
            # Parse parking spot configuration
            spot_info = self._parse_parking_spot(pos_entry)
            if not spot_info:
                continue
            
            shape_type, coords, poly_points = spot_info
            
            # Check YOLO detections in this spot
            yolo_occupied, yolo_class = self._check_yolo_occupation(coords, poly_points, 
                                                                   shape_type, yolo_detections)
            
            # Calculate pixel count
            pixel_count = self._calculate_pixel_count(processed_image, coords, poly_points, shape_type)
            
            # Determine current occupant
            current_occupant = self._determine_occupant(yolo_occupied, yolo_class, pixel_count)
            
            # Update state and log events
            self._update_spot_state(spot_id, current_occupant)
            
            # Determine display properties
            is_free = self.parking_spot_states[spot_id]['object_class'] is None
            if is_free:
                space_counter += 1
            
            # Draw parking spot on image
            annotated_image = self._draw_parking_spot(annotated_image, spot_info, spot_id, 
                                                    pixel_count, is_free)
        
        total_spots = len(self.parking_spots)
        occupied_spots = total_spots - space_counter
        
        parking_stats = {
            'free_spaces': space_counter,
            'occupied_spaces': occupied_spots,
            'total_spaces': total_spots,
            'yolo_detections': len(yolo_detections) if yolo_detections else 0
        }
        return annotated_image, parking_stats
    
    def _draw_yolo_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw YOLO detections on image - ALL detections, not just target classes"""
        annotated_image = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box for ALL detections (like the standalone version)
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), self.config.YOLO_BOX_COLOR, 2)
            
            # Add confidence score and class name
            label_text = f'{class_name} {conf:.2f}'
            
            # Use cvzone if available, otherwise use OpenCV
            try:
                cvzone.putTextRect(annotated_image, label_text, 
                                 (x1, y1 - 10), scale=0.5, thickness=1, 
                                 colorR=self.config.YOLO_TEXT_COLOR, offset=3)
            except:
                # Fallback to OpenCV text
                cv2.putText(annotated_image, label_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.YOLO_TEXT_COLOR, 1)
        
        return annotated_image
    
    def _parse_parking_spot(self, pos_entry) -> Optional[Tuple]:
        """Parse parking spot configuration"""
        try:
            if isinstance(pos_entry[-1], str) and pos_entry[-1] == 'polygon':
                # Polygon format
                points_flat = pos_entry[:-1]
                if len(points_flat) < 6 or len(points_flat) % 2 != 0:
                    return None
                
                poly_points_coords = [(points_flat[i], points_flat[i+1]) 
                                    for i in range(0, len(points_flat), 2)]
                np_poly_points = np.array(poly_points_coords)
                poly_points = np_poly_points.astype(np.int32)
                
                x_coord = int(np_poly_points[:, 0].min())
                y_coord = int(np_poly_points[:, 1].min())
                width = int(np_poly_points[:, 0].max() - x_coord)
                height = int(np_poly_points[:, 1].max() - y_coord)
                
                coords = (x_coord, y_coord, width, height)
                return 'polygon', coords, poly_points
                
            elif len(pos_entry) == 5:
                # Rectangle with dimensions
                x, y, w, h, _ = pos_entry
                coords = (x, y, w, h)
                return 'rectangle', coords, None
                
            elif len(pos_entry) == 2:
                # Rectangle with default dimensions
                x, y = pos_entry
                coords = (x, y, self.config.DEFAULT_SPOT_WIDTH, self.config.DEFAULT_SPOT_HEIGHT)
                return 'rectangle', coords, None
                
        except Exception as e:
            print(f"Error parsing parking spot: {e}")
        
        return None
    
    def _check_yolo_occupation(self, coords, poly_points, shape_type, yolo_detections) -> Tuple[bool, str]:
        """Check if YOLO detected a target object in the parking spot"""
        if not self.model or not yolo_detections:
            return False, None
        
        x_coord, y_coord, width, height = coords
        
        for det in yolo_detections:
            obj_bbox = det['bbox']
            obj_class_name = det['class_name']
            
            # Only check target vehicle classes for parking occupation
            if obj_class_name not in self.config.YOLO_TARGET_CLASSES_NAMES:
                continue
            
            obj_center_x = (obj_bbox[0] + obj_bbox[2]) // 2
            obj_center_y = (obj_bbox[1] + obj_bbox[3]) // 2
            
            if shape_type == 'polygon':
                if cv2.pointPolygonTest(poly_points, (obj_center_x, obj_center_y), False) >= 0:
                    return True, obj_class_name
            elif shape_type == 'rectangle':
                if (x_coord <= obj_center_x < x_coord + width and 
                    y_coord <= obj_center_y < y_coord + height):
                    return True, obj_class_name
        
        return False, None
    
    def _calculate_pixel_count(self, processed_image, coords, poly_points, shape_type) -> int:
        """Calculate pixel count in parking spot"""
        x_coord, y_coord, width, height = coords
        
        try:
            if shape_type == 'polygon':
                if width <= 0 or height <= 0:
                    return 999
                
                imgCrop = processed_image[y_coord:y_coord + height, x_coord:x_coord + width]
                if imgCrop.size == 0:
                    return 999
                
                mask = np.zeros(imgCrop.shape[:2], dtype=np.uint8)
                relative_poly_points = poly_points - [x_coord, y_coord]
                cv2.fillPoly(mask, [relative_poly_points.astype(np.int32)], 255)
                masked_img_crop = cv2.bitwise_and(imgCrop, imgCrop, mask=mask)
                return cv2.countNonZero(masked_img_crop)
                
            elif shape_type == 'rectangle':
                imgCrop = processed_image[y_coord:y_coord + height, x_coord:x_coord + width]
                return cv2.countNonZero(imgCrop) if imgCrop.size > 0 else 999
                
        except Exception as e:
            print(f"Error calculating pixel count: {e}")
        
        return 999
    
    def _determine_occupant(self, yolo_occupied, yolo_class, pixel_count) -> Optional[str]:
        """Determine current occupant of parking spot"""
        if yolo_occupied:
            return yolo_class
        elif pixel_count >= self.config.PIXEL_THRESHOLD:
            return "UnknownVehicle_Pixel"
        return None
    
    def _update_spot_state(self, spot_id: int, current_occupant: Optional[str]):
        """Update parking spot state and log events"""
        prev_state = self.parking_spot_states[spot_id]
        prev_occupant = prev_state['object_class']
        prev_entry_time = prev_state['entry_time']
        
        if prev_occupant != current_occupant:
            now = datetime.now()
            
            # Log exit event
            if prev_occupant is not None and prev_entry_time is not None:
                duration = now - prev_entry_time
                self.parking_events_log.append({
                    'Object_detected': prev_occupant,
                    'Entry_time': prev_entry_time,
                    'Exit_time': now,
                    'Total_duration': duration,
                    'Parking_ID': spot_id
                })
            
            # Update state
            if current_occupant is not None:
                self.parking_spot_states[spot_id]['object_class'] = current_occupant
                self.parking_spot_states[spot_id]['entry_time'] = now
            else:
                self.parking_spot_states[spot_id]['object_class'] = None
                self.parking_spot_states[spot_id]['entry_time'] = None
    
    def _draw_parking_spot(self, image, spot_info, spot_id, pixel_count, is_free) -> np.ndarray:
        """Draw parking spot on image"""
        shape_type, coords, poly_points = spot_info
        x_coord, y_coord, width, height = coords
        
        # Determine display properties
        if is_free:
            color = self.config.PARKING_FREE_COLOR  # Green for free
            thickness = 5
            text = str(pixel_count)
        else:
            color = self.config.PARKING_OCCUPIED_COLOR  # Red for occupied
            thickness = 2
            occupant = self.parking_spot_states[spot_id]['object_class']
            if occupant == "UnknownVehicle_Pixel":
                text = f"P:{pixel_count}"
            else:
                text = str(occupant)[:10]
        
        # Draw shape
        if shape_type == 'polygon' and poly_points is not None:
            cv2.polylines(image, [poly_points], isClosed=True, color=color, thickness=thickness)
        elif shape_type == 'rectangle':
            cv2.rectangle(image, (x_coord, y_coord), 
                         (x_coord + width, y_coord + height), color, thickness)
        
        # Draw text
        try:
            cvzone.putTextRect(image, text, (x_coord, y_coord + height - 3), 
                              scale=1, thickness=2, offset=0, colorR=color)
        except:
            # Fallback to OpenCV text
            cv2.putText(image, text, (x_coord, y_coord + height - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return image
    
    def draw_yolo_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw YOLO detections on image (kept for compatibility)"""
        return self._draw_yolo_detections(image, detections)
    
    def save_parking_report(self, filename: str = None) -> str:
        """Save parking events to CSV file"""
        if not self.parking_events_log:
            print("No parking events to report.")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"parking_report_{timestamp}.csv"
        
        try:
            fieldnames = ['Object_detected', 'Entry_time', 'Exit_time', 'Total_duration', 'Parking_ID']
            
            with open(filename, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for event in self.parking_events_log:
                    formatted_event = {
                        'Object_detected': event['Object_detected'],
                        'Entry_time': event['Entry_time'].strftime("%Y-%m-%d %H:%M:%S") if event['Entry_time'] else '',
                        'Exit_time': event['Exit_time'].strftime("%Y-%m-%d %H:%M:%S") if event['Exit_time'] else '',
                        'Total_duration': str(event['Total_duration']) if event['Total_duration'] else '',
                        'Parking_ID': event['Parking_ID']
                    }
                    writer.writerow(formatted_event)
            
            print(f"Parking report saved to {filename}")
            return filename
            
        except Exception as e:
            print(f"Error saving parking report: {e}")
            return None
    
    def finalize_session(self):
        """Finalize parking session and log remaining occupants"""
        current_time = datetime.now()
        
        for spot_id, state in self.parking_spot_states.items():
            if state['object_class'] is not None and state['entry_time'] is not None:
                duration = current_time - state['entry_time']
                self.parking_events_log.append({
                    'Object_detected': state['object_class'],
                    'Entry_time': state['entry_time'],
                    'Exit_time': current_time,
                    'Total_duration': duration,
                    'Parking_ID': spot_id
                })
    
    def get_parking_summary(self) -> Dict:
        """Get current parking summary"""
        total_spots = len(self.parking_spots)
        occupied_spots = sum(1 for state in self.parking_spot_states.values() 
                           if state['object_class'] is not None)
        free_spots = total_spots - occupied_spots
        
        return {
            'total_spots': total_spots,
            'occupied_spots': occupied_spots,
            'free_spots': free_spots,
            'occupancy_rate': (occupied_spots / total_spots * 100) if total_spots > 0 else 0,
            'events_logged': len(self.parking_events_log)
        }
    
    def toggle_yolo_visualization(self, show: bool = None):
        """Toggle YOLO detection visualization on/off"""
        if show is None:
            self.config.SHOW_YOLO_DETECTIONS = not self.config.SHOW_YOLO_DETECTIONS
        else:
            self.config.SHOW_YOLO_DETECTIONS = show
        
        print(f"YOLO visualization: {'ON' if self.config.SHOW_YOLO_DETECTIONS else 'OFF'}")
        return self.config.SHOW_YOLO_DETECTIONS