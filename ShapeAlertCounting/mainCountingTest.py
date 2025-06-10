import cv2
import pickle
import cvzone
import numpy as np
import urllib.parse
import json
from datetime import datetime
import threading
import time
from collections import defaultdict, deque
import pandas as pd
import os
from ultralytics import YOLO

# --- RTSP Camera Configuration ---
NVR_USERNAME = "admin"
NVR_PASSWORD = "d@t@rium2023"
NVR_PORT = "554"
CAMERA_MAP = {
    '1': "10.20.1.7",
    '2': "10.20.1.8",
    '3': "10.20.1.18",
    '4': "10.20.1.19"
}
ENCODED_NVR_PASSWORD = urllib.parse.quote(NVR_PASSWORD)

# --- Available Colors ---
COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'orange': (0, 165, 255),
    'purple': (128, 0, 128),
    'white': (255, 255, 255),
    'black': (0, 0, 0)
}

# Detection_model = "yolo11n.pt" # Will be moved to main()

# --- Alert and Counting System ---
class AlertCountingSystem:
    def __init__(self, yolo_model_path=None, confidence_threshold=0.4, target_object_classes=None, device='cpu'):
        self.zone_counters = defaultdict(int)
        self.line_crossings = defaultdict(int)
        self.alert_zones = {}
        self.counting_lines = {}
        self.objects_in_zone_history = defaultdict(lambda: deque(maxlen=10))
        self.alert_triggered = defaultdict(bool)
        self.last_alert_time = defaultdict(float)
        self.alert_cooldown = 3.0  # 3 seconds cooldown between alerts
        
        self.crossing_log = []
        self.line_last_log_time = defaultdict(float)
        self.line_log_cooldown = 2.0
        
        self.yolo_model = None
        if yolo_model_path:
            try:
                self.yolo_model = YOLO(yolo_model_path)
                self.yolo_model.to(device)
                print(f"YOLO model '{yolo_model_path}' loaded successfully on device '{device}'.")
            except Exception as e:
                print(f"Error loading YOLO model from '{yolo_model_path}': {e}")
                self.yolo_model = None
        
        self.confidence_threshold = confidence_threshold
        self.target_object_classes = target_object_classes
        self.device = device

    def _run_yolo_detection(self, frame):
        """Runs YOLO detection on a given frame."""
        if not self.yolo_model:
            return []
        
        results = self.yolo_model.predict(frame, conf=self.confidence_threshold, verbose=False, device=self.device)
        detections = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.yolo_model.names[cls]
                
                if self.target_object_classes is None or cls in self.target_object_classes:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': class_name
                    })
        return detections

    def process_zones_and_lines(self, processed_img, original_img, shapes_list):
        """Process all zones and lines for counting and alerts using YOLO if available"""
        current_time = time.time()
        
        yolo_detections = []
        if self.yolo_model:
            yolo_detections = self._run_yolo_detection(original_img)

        for i, shape_entry in enumerate(shapes_list):
            if not isinstance(shape_entry, dict):
                continue
                
            shape_type = shape_entry.get('type')
            shape_name = shape_entry.get('name', f'Unknown_{i}')
            shape_color = COLORS.get(shape_entry.get('color', 'white'), (255, 255, 255))
            
            if shape_type == 'polygon':
                self._process_polygon_zone(processed_img, original_img, shape_entry, shape_name, shape_color, current_time, yolo_detections)
            elif shape_type == 'rectangle':
                self._process_rectangle_zone(processed_img, original_img, shape_entry, shape_name, shape_color, current_time, yolo_detections)
            elif shape_type == 'line':
                self._process_counting_line(processed_img, original_img, shape_entry, shape_name, shape_color, current_time, yolo_detections)
    
    def _process_polygon_zone(self, processed_img, original_img, shape_entry, name, color, current_time, yolo_detections):
        """Process polygon zones for object detection and alerts."""
        points_list = shape_entry.get('points')
        if not points_list or len(points_list) < 3:
            return
            
        try:
            np_points = np.array(points_list, dtype=np.int32)
        except ValueError:
            print(f"Error: Invalid points for polygon zone {name}.")
            return
            
        objects_in_zone_count = 0
        detected_object_details = []

        if self.yolo_model and yolo_detections:
            for det in yolo_detections:
                x1, y1, x2, y2 = det['bbox']
                # Check if the center of the detected object is inside the polygon
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2
                
                if cv2.pointPolygonTest(np_points, (obj_center_x, obj_center_y), False) >= 0:
                    objects_in_zone_count += 1
                    detected_object_details.append(f"{det['class_name']}({det['confidence']:.2f})")
                    # Optionally draw bounding box of detected object in zone
                    cv2.rectangle(original_img, (x1,y1), (x2,y2), (255,165,0), 2) # Orange for detected objects in zone

            self.objects_in_zone_history[name].append(objects_in_zone_count)
            # alert_threshold = shape_entry.get('object_alert_threshold', 1) # e.g., alert if 1 or more objects
            is_alert = objects_in_zone_count > 0 # Simple: alert if any target object is found
        else:
            # Fallback to motion detection if YOLO is not used or no detections
            x_min_poly, y_min_poly = np_points.min(axis=0)
            x_max_poly, y_max_poly = np_points.max(axis=0)
            img_h, img_w = processed_img.shape[:2]
            x_min_poly, y_min_poly = max(0, x_min_poly), max(0, y_min_poly)
            x_max_poly, y_max_poly = min(img_w, x_max_poly), min(img_h, y_max_poly)

            if x_min_poly >= x_max_poly or y_min_poly >= y_max_poly: return
            
            crop_motion = processed_img[y_min_poly:y_max_poly, x_min_poly:x_max_poly]
            if crop_motion.size == 0: return

            mask_motion = np.zeros(crop_motion.shape[:2], dtype=np.uint8)
            relative_points_motion = np_points - [x_min_poly, y_min_poly]
            cv2.fillPoly(mask_motion, [relative_points_motion.astype(np.int32)], 255)
            
            masked_crop_motion = cv2.bitwise_and(crop_motion, crop_motion, mask=mask_motion)
            motion_count = cv2.countNonZero(masked_crop_motion)
            
            self.objects_in_zone_history[name].append(motion_count) # Still use objects_in_zone_history for consistency
            
            zone_area = cv2.contourArea(np_points)
            motion_alert_threshold = max(50, zone_area * 0.03) # Lower sensitivity for motion if objects are primary
            is_alert = motion_count > motion_alert_threshold
            objects_in_zone_count = motion_count # For display purposes if using motion
            detected_object_details.append(f"Motion: {motion_count}")

        alert_status = ""
        if is_alert:
            if not self.alert_triggered[name] or (current_time - self.last_alert_time[name]) > self.alert_cooldown:
                self.alert_triggered[name] = True
                self.last_alert_time[name] = current_time
                alert_status = " ⚠️ ALERT!"
                if self.yolo_model and yolo_detections:
                    print(f"🚨 ALERT: {objects_in_zone_count} object(s) detected in {name}: {', '.join(detected_object_details)}")
                else:
                    print(f"🚨 ALERT: Motion detected in {name} - Level: {objects_in_zone_count}")
        else:
            self.alert_triggered[name] = False
        
        display_color = (0, 0, 255) if is_alert else color
        thickness = 4 if is_alert else 2
        
        cv2.polylines(original_img, [np_points], isClosed=True, color=display_color, thickness=thickness)
        
        center_x = int(np.mean(np_points[:, 0]))
        center_y = int(np.mean(np_points[:, 1]))
        
        info_text = f"Objects: {objects_in_zone_count}" if self.yolo_model else f"Motion: {objects_in_zone_count}"
        cvzone.putTextRect(original_img, f"{name}{alert_status}", 
                          (center_x - 50, center_y - 20), scale=0.8, thickness=2, 
                          colorR=display_color, offset=5)
        cvzone.putTextRect(original_img, info_text, 
                          (center_x - 30, center_y + 10), scale=0.6, thickness=1, 
                          colorR=display_color, offset=3)
        if detected_object_details and self.yolo_model:
             details_to_show = ", ".join(detected_object_details[:2]) # Show first 2 detected object types
             if len(detected_object_details) > 2: details_to_show += "..."
             cvzone.putTextRect(original_img, details_to_show,
                               (center_x - 50, center_y + 35), scale=0.5, thickness=1,
                               colorR=display_color, offset=3)

    def _process_rectangle_zone(self, processed_img, original_img, shape_entry, name, color, current_time, yolo_detections):
        """Process rectangle zones for object detection and alerts."""
        x = shape_entry.get('x', 0)
        y = shape_entry.get('y', 0)
        width = shape_entry.get('width', 100)
        height = shape_entry.get('height', 100)
        
        img_h, img_w = original_img.shape[:2] # Use original_img for boundary checks with shape coords
        rect_x_start, rect_y_start = max(0, x), max(0, y)
        rect_x_end = min(img_w, x + width)
        rect_y_end = min(img_h, y + height)
        
        if rect_x_start >= rect_x_end or rect_y_start >= rect_y_end:
            return

        objects_in_zone_count = 0
        detected_object_details = []

        if self.yolo_model and yolo_detections:
            for det in yolo_detections:
                x1, y1, x2, y2 = det['bbox']
                obj_center_x = (x1 + x2) // 2
                obj_center_y = (y1 + y2) // 2
                
                # Check if the center of the detected object is inside the rectangle zone
                if rect_x_start <= obj_center_x < rect_x_end and \
                   rect_y_start <= obj_center_y < rect_y_end:
                    objects_in_zone_count += 1
                    detected_object_details.append(f"{det['class_name']}({det['confidence']:.2f})")
                    cv2.rectangle(original_img, (x1,y1), (x2,y2), (255,165,0), 2)


            self.objects_in_zone_history[name].append(objects_in_zone_count)
            # alert_threshold = shape_entry.get('object_alert_threshold', 1)
            is_alert = objects_in_zone_count > 0 
        else:
            # Fallback to motion detection
            # Ensure processed_img bounds are correct for cropping
            crop_x_start, crop_y_start = max(0, x), max(0, y)
            img_proc_h, img_proc_w = processed_img.shape[:2]
            crop_x_end = min(img_proc_w, x + width)
            crop_y_end = min(img_proc_h, y + height)

            if crop_x_start >= crop_x_end or crop_y_start >= crop_y_end: return

            crop_motion = processed_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
            if crop_motion.size == 0: return
            
            motion_count = cv2.countNonZero(crop_motion)
            self.objects_in_zone_history[name].append(motion_count)
            
            zone_area = width * height
            motion_alert_threshold = max(50, zone_area * 0.03)
            is_alert = motion_count > motion_alert_threshold
            objects_in_zone_count = motion_count # For display
            detected_object_details.append(f"Motion: {motion_count}")

        alert_status = ""
        if is_alert:
            if not self.alert_triggered[name] or (current_time - self.last_alert_time[name]) > self.alert_cooldown:
                self.alert_triggered[name] = True
                self.last_alert_time[name] = current_time
                alert_status = " ⚠️ ALERT!"
                if self.yolo_model and yolo_detections:
                    print(f"🚨 ALERT: {objects_in_zone_count} object(s) detected in {name}: {', '.join(detected_object_details)}")
                else:
                    print(f"🚨 ALERT: Motion detected in {name} - Level: {objects_in_zone_count}")
        else:
            self.alert_triggered[name] = False
        
        display_color = (0, 0, 255) if is_alert else color
        thickness = 4 if is_alert else 2
        
        cv2.rectangle(original_img, (rect_x_start, rect_y_start), (rect_x_end, rect_y_end), display_color, thickness)
        
        info_text = f"Objects: {objects_in_zone_count}" if self.yolo_model else f"Motion: {objects_in_zone_count}"
        cvzone.putTextRect(original_img, f"{name}{alert_status}", 
                          (rect_x_start + 5, rect_y_start + 25), scale=0.7, thickness=2, 
                          colorR=display_color, offset=5)
        cvzone.putTextRect(original_img, info_text, 
                          (rect_x_start + 5, rect_y_start + height - 10 if height > 10 else rect_y_start + 5), scale=0.5, thickness=1,
                          colorR=display_color, offset=3)
        if detected_object_details and self.yolo_model:
             details_to_show = ", ".join(detected_object_details[:2])
             if len(detected_object_details) > 2: details_to_show += "..."
             cvzone.putTextRect(original_img, details_to_show,
                               (rect_x_start + 5, rect_y_start + 50), scale=0.4, thickness=1,
                               colorR=display_color, offset=3)

    def _process_counting_line(self, processed_img, original_img, shape_entry, name, color, current_time, yolo_detections):
        """Process counting lines for object crossing detection, using YOLO if available."""
        start_point = shape_entry.get('start')
        end_point = shape_entry.get('end')
        
        if not start_point or not end_point:
            return
        
        # Ensure points are tuples of integers for OpenCV functions
        p1 = tuple(map(int, start_point))
        p2 = tuple(map(int, end_point))
            
        thickness = 3
        cv2.line(original_img, p1, p2, color, thickness)
        
        # Check for object crossing if YOLO is active
        object_crossed_this_frame = False
        crossed_object_name = "Motion Detected" # Default

        if self.yolo_model and yolo_detections:
            for det in yolo_detections:
                x1, y1, x2, y2 = det['bbox']
                # For line crossing, it's often better to check if any part of the bbox intersects the line.
                # A simple check can be if the line segment (p1,p2) intersects the rectangle (x1,y1,x2,y2).
                # For simplicity here, let's check if the center of the bbox crosses a slightly thickened line.
                # More robust: check intersection of line segment with bbox rectangle.
                
                # Create a temporary mask for the current detection's bounding box
                obj_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
                cv2.rectangle(obj_mask, (x1,y1), (x2,y2), 255, -1) # Filled rectangle for the object

                # Create a mask for the counting line (can be slightly thicker for better intersection detection)
                line_check_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
                cv2.line(line_check_mask, p1, p2, 255, thickness=max(5, shape_entry.get('line_thickness_check', 10)))

                # Check for intersection
                intersection = cv2.bitwise_and(obj_mask, line_check_mask)
                if cv2.countNonZero(intersection) > 0:
                    object_crossed_this_frame = True
                    crossed_object_name = det['class_name']
                    # Highlight the object that crossed
                    cv2.rectangle(original_img, (x1,y1), (x2,y2), (0,255,255), 2) # Yellow for crossing object
                    cvzone.putTextRect(original_img, f"Crossing: {crossed_object_name}", (x1, y1 - 5), scale=0.5, thickness=1, colorR=(0,255,255) )
                    break # Count first detected object crossing this line in this frame
        
        # If YOLO didn't detect a crossing, or YOLO is not active, fall back to motion
        if not object_crossed_this_frame and not (self.yolo_model and yolo_detections):
            line_mask_motion = np.zeros(processed_img.shape[:2], dtype=np.uint8)
            cv2.line(line_mask_motion, p1, p2, 255, thickness=max(5, shape_entry.get('line_thickness_check', 10)))
            line_motion = cv2.bitwise_and(processed_img, processed_img, mask=line_mask_motion)
            motion_on_line = cv2.countNonZero(line_motion)
            
            crossing_threshold = shape_entry.get('sensitivity', 100) # Motion sensitivity
            if motion_on_line > crossing_threshold:
                object_crossed_this_frame = True
                # crossed_object_name remains "Motion Detected"
        
        if object_crossed_this_frame:
            if current_time - self.line_last_log_time[name] > self.line_log_cooldown:
                self.line_crossings[name] += 1
                self.line_last_log_time[name] = current_time
                
                timestamp_obj = datetime.now()
                log_entry = {
                    'Timestamp': timestamp_obj.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    'Object': crossed_object_name, # Log the detected object class or "Motion Detected"
                    'Counter Shape Name': name,
                    'Count': self.line_crossings[name]
                }
                self.crossing_log.append(log_entry)
                
                print(f"📊 LOGGED CROSSING: '{crossed_object_name}' on line '{name}'. Total: {self.line_crossings[name]}")
        
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        
        cvzone.putTextRect(original_img, name, 
                          (mid_x - 30, mid_y - 20), scale=0.7, thickness=2, 
                          colorR=color, offset=5)
        cvzone.putTextRect(original_img, f"Count: {self.line_crossings[name]}", 
                          (mid_x - 25, mid_y + 10), scale=0.6, thickness=1, 
                          colorR=color, offset=3)
        if crossed_object_name != "Motion Detected" and object_crossed_this_frame :
             cvzone.putTextRect(original_img, f"Last: {crossed_object_name}",
                               (mid_x - 40, mid_y + 35), scale=0.5, thickness=1,
                               colorR=color, offset=3)

    def get_summary_stats(self):
        """Get summary statistics for display"""
        active_alerts = sum(1 for alert in self.alert_triggered.values() if alert)
        total_crossings = sum(self.line_crossings.values())
        total_zones = len(self.objects_in_zone_history)
        
        return {
            'active_alerts': active_alerts,
            'total_crossings': total_crossings,
            'total_zones': total_zones,
            'zone_details': dict(self.line_crossings)
        }

    def save_log_to_csv(self, camera_name_identifier):
        """Save the crossing log to a CSV file, appending if it exists."""
        if not self.crossing_log:
            print("No new crossing events to save.")
            return

        try:
            df = pd.DataFrame(self.crossing_log)
            # camera_name_identifier will be something like "counting_log_camera_1" from display2.py
            # or "video_area1" / "camera_1" if mainCountingTest.py is run standalone.
            csv_filename = f"{camera_name_identifier}_crossings.csv"

            file_exists = os.path.exists(csv_filename)
            # Check if file is empty only if it exists, otherwise consider it empty for initial write.
            is_empty = os.path.getsize(csv_filename) == 0 if file_exists else True

            if file_exists and not is_empty:
                # Append without header
                df.to_csv(csv_filename, mode='a', header=False, index=False)
                print(f"Successfully appended {len(self.crossing_log)} new events to '{csv_filename}'")
            else:
                # Write new file with header (or overwrite if file exists but was empty)
                df.to_csv(csv_filename, mode='w', header=True, index=False)
                if file_exists and is_empty:
                    print(f"Successfully wrote {len(self.crossing_log)} events to existing empty file '{csv_filename}' with header.")
                else:
                    print(f"Successfully created and saved {len(self.crossing_log)} events to '{csv_filename}' with header.")

            # Clear the log after successful save to prevent re-saving same entries
            # from this instance in subsequent calls.
            self.crossing_log.clear()
            print(f"Internal crossing log (in AlertCountingSystem instance) cleared after saving to CSV.")

        except Exception as e:
            print(f"Error saving/appending crossing log to CSV for '{csv_filename}': {e}")

def load_shapes():
    """Load shapes from both pickle and JSON files"""
    shapes_list = []
    
    # Try to load from pickle file first (backward compatibility)
    try:
        with open('mainRoadZone/CountingZonePos copy', 'rb') as f:
       # with open('hallZone/CountingHallPos copy 2', 'rb') as f:
        #with open('StairsZone/CountingZonePos copy', 'rb') as f:
            shapes_list = pickle.load(f)
        print(f"Loaded {len(shapes_list)} shapes from pickle file")
    except FileNotFoundError:
        print("Pickle file not found, trying JSON...")
    except Exception as e:
        print(f"Error loading pickle file: {e}")
    
    # Try to load from JSON file if pickle failed or is empty
    if not shapes_list:
        try:
            with open('shapes_config.json', 'r') as f:
                shapes_list = json.load(f)
            print(f"Loaded {len(shapes_list)} shapes from JSON file")
        except FileNotFoundError:
            print("No shape configuration files found!")
            return []
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return []
    
    return shapes_list

def setup_video_source():
    """Setup video source (file or RTSP camera)"""
    cap = None
    is_video_file = False
    camera_name = None
    
    print("Select video source:")
    print("1: Video File (e.g., area1.mp4, area2.mp4)")
    print("2: RTSP Camera Stream")
    source_choice = input("Enter choice (1 or 2): ")
    
    if source_choice == '1':
        video_file_path = '..\\media\\videos\\area1.mp4'
        cap = cv2.VideoCapture(video_file_path)
        is_video_file = True
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_file_path}'")
            return None, False, None
        base_name = video_file_path.split('/')[-1].split('\\')[-1]
        camera_name = f"video_{base_name.split('.')[0]}"
        print(f"Using video file '{video_file_path}' as source. Camera name: {camera_name}")
        
    elif source_choice == '2':
        print("\nAvailable cameras:")
        for cam_id, cam_ip in CAMERA_MAP.items():
            print(f"  Camera {cam_id}: {cam_ip}")
        
        camera_id_input = input("Enter camera number: ").strip()
        if camera_id_input in CAMERA_MAP:
            camera_ip = CAMERA_MAP[camera_id_input]
            
            rtsp_urls = [
                f"rtsp://{NVR_USERNAME}:{ENCODED_NVR_PASSWORD}@{camera_ip}:{NVR_PORT}/h264/ch{camera_id_input}/main/av_stream?rtsp_transport=tcp",
                f"rtsp://{NVR_USERNAME}:{ENCODED_NVR_PASSWORD}@{camera_ip}:{NVR_PORT}/h264/ch{camera_id_input}/sub/av_stream?rtsp_transport=tcp",
                f"rtsp://{NVR_USERNAME}:{ENCODED_NVR_PASSWORD}@{camera_ip}:{NVR_PORT}/Streaming/Channels/{camera_id_input}01?rtsp_transport=tcp",
                f"rtsp://{NVR_USERNAME}:{ENCODED_NVR_PASSWORD}@{camera_ip}:{NVR_PORT}/Streaming/Channels/{camera_id_input}02?rtsp_transport=tcp",
                f"rtsp://{NVR_USERNAME}:{ENCODED_NVR_PASSWORD}@{camera_ip}:{NVR_PORT}/cam/realmonitor?channel={camera_id_input}&subtype=0&unicast=true&proto=Onvif",
                f"rtsp://{NVR_USERNAME}:{ENCODED_NVR_PASSWORD}@{camera_ip}/stream1"
            ]
            
            print(f"Attempting to connect to camera {camera_id_input} at {camera_ip}...")
            
            for i, url in enumerate(rtsp_urls):
                print(f"Trying URL {i+1}: {url}")
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"Successfully connected using URL {i+1}")
                        is_video_file = False
                        camera_name = f"camera_{camera_id_input}"
                        break
                    else:
                        print(f"URL {i+1} opened but failed to read frame. Releasing.")
                        cap.release()
                        cap = None
                else:
                    cap = None
            
            if cap is None:
                print(f"Error: Could not connect to camera {camera_id_input} using any RTSP URL.")
                return None, False, None
        else:
            print("Error: Invalid camera number selected.")
            return None, False, None
    else:
        print("Error: Invalid source choice.")
        return None, False, None
    
    return cap, is_video_file, camera_name

def main():
    """Main function"""
    # --- Configuration for YOLO (if used) ---
    YOLO_MODEL_PATH = "yolo11n.pt"  # Your specified detection model
    YOLO_CONFIDENCE = 0.3          # Confidence threshold for YOLO detections
    YOLO_TARGET_CLASSES = None     # Example: [0] for 'person' if using COCO. None for all classes.
    YOLO_DEVICE = 'cpu'            # 'cpu' or 'cuda' if available and PyTorch with CUDA is installed

    # Load shapes configuration
    shapes_list = load_shapes()
    if not shapes_list:
        print("No shapes found! Please create shapes using the enhanced shape drawer first.")
        return
    
    # Setup video source
    cap, is_video_file, camera_name = setup_video_source()
    if cap is None:
        return
    
    # Initialize alert and counting system with YOLO configuration
    alert_system = AlertCountingSystem(
        yolo_model_path=YOLO_MODEL_PATH,
        confidence_threshold=YOLO_CONFIDENCE,
        target_object_classes=YOLO_TARGET_CLASSES,
        device=YOLO_DEVICE
    )
    
    if alert_system.yolo_model:
        print(f"\n🎯 Monitoring System Started with YOLO ({YOLO_MODEL_PATH})!")
    else:
        print(f"\n🎯 Monitoring System Started (Motion Detection Only - YOLO model '{YOLO_MODEL_PATH}' not loaded or error occurred).")
        
    print(f"📊 Loaded {len(shapes_list)} monitoring zones/lines")
    print("🎮 Controls: 'q' to quit, 's' to show statistics")
    print("=" * 50)
    
    frame_count = 0
    
    try:
        while True:
            # Handle video file looping
            if is_video_file and cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if total_frames > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    print("Video file ended and has 0 frames or unknown frame count. Stopping.")
                    break
            
            success, img = cap.read()
            if not success:
                print("Failed to read frame, attempting to reconnect...")
                if not is_video_file:
                    # Try to reconnect for RTSP streams
                    time.sleep(1)
                    continue
                else:
                    break
            
            frame_count += 1
            
            # Image preprocessing for motion detection
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
            imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY_INV, 25, 16)
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
            
            # Process all zones and lines
            alert_system.process_zones_and_lines(imgDilate, img, shapes_list)
            
            # Get and display summary statistics
            stats = alert_system.get_summary_stats()
            
            # Display summary information
            summary_text = f"Zones: {stats['total_zones']} | Alerts: {stats['active_alerts']} | Crossings: {stats['total_crossings']}"
            cvzone.putTextRect(img, summary_text, (10, 30), scale=1, thickness=2, 
                             colorR=(0, 200, 0), offset=10)
            
            # Display timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cvzone.putTextRect(img, f"Time: {timestamp}", (10, img.shape[0] - 30), 
                             scale=0.7, thickness=1, colorR=(255, 255, 255), offset=5)
            
            # Show processed frame
            cv2.imshow("Enhanced Counting & Alert System", img)
            
            # Optional: Show processed image for debugging
            # cv2.imshow("Motion Detection", imgDilate)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Show detailed statistics
                print(f"\n📊 DETAILED STATISTICS (Frame {frame_count}):")
                print(f"Active Alerts: {stats['active_alerts']}")
                print(f"Total Crossings: {stats['total_crossings']}")
                print(f"Zone Details:")
                for zone_name, count in stats['zone_details'].items():
                    print(f"  - {zone_name}: {count} crossings")
                print("-" * 30)
    
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Cleanup
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        # Save crossing log to CSV
        if alert_system and camera_name:
            alert_system.save_log_to_csv(camera_name)
        
        # Final statistics
        final_stats = alert_system.get_summary_stats()
        print(f"\n🏁 FINAL STATISTICS:")
        print(f"Total Zones Monitored: {final_stats['total_zones']}")
        print(f"Total Crossings Detected: {final_stats['total_crossings']}")
        print("Thank you for using the Enhanced Counting & Alert System!")

if __name__ == "__main__":
    main()
    
    
    
    
    