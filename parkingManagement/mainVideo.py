# parking_detection Stand-Alone version.py
import cv2
import pickle
import cvzone
import numpy as np
import urllib.parse # Added for RTSP URL encoding
from ultralytics import YOLO
import time # Added for handling frame read errors
from datetime import datetime, timedelta # Added for tracking and reporting
import csv # Added for CSV report generation

# --- YOLO Configuration ---
YOLO_MODEL_PATH = "yolo11n.pt"  # Or your specific model path like "yolov8n.pt"
YOLO_CONFIDENCE = 0.4
# Define target classes for parking (e.g., from COCO dataset: car=2, motorcycle=3, bus=5, truck=7)
# You might need to get the class names/IDs from your specific yolo11n.pt model
YOLO_TARGET_CLASSES_NAMES = ['car', 'truck', 'bus', 'motorcycle'] # Adjust as per your model's classes
YOLO_DEVICE = 'cpu'  # Use 'cuda' if GPU is available

# Load YOLO model
try:
    model = YOLO(YOLO_MODEL_PATH)
    # If using CPU and the model might be FP16, convert to FP32 for wider compatibility.
    # The error "Input type (torch.uint8) and bias type (torch.float16) should be the same"
    # suggests the model weights are FP16. Forcing to FP32 on CPU can resolve this.
    if YOLO_DEVICE == 'cpu':
        # Access the underlying PyTorch model and convert its parameters to float32
        if hasattr(model, 'model') and hasattr(model.model, 'float'):
            print(f"Converting model '{YOLO_MODEL_PATH}' parameters to FP32 for CPU execution.")
            model.model.float()
        else:
            print(f"Warning: Could not access model.model.float() to convert to FP32. Model might not be a standard PyTorch module.")

    model.to(YOLO_DEVICE)
    print(f"YOLO model '{YOLO_MODEL_PATH}' loaded successfully on device '{YOLO_DEVICE}'.")
    # If your model uses specific class IDs, you might need to map names to IDs
    # Example: coco_model_names = model.names
    # YOLO_TARGET_CLASSES_IDS = [k for k, v in coco_model_names.items() if v in YOLO_TARGET_CLASSES_NAMES]
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None
# --- End YOLO Configuration ---

# --- RTSP Camera Configuration ---
NVR_USERNAME = "admin"  # Replace with your NVR username
NVR_PASSWORD = "d@t@rium2023"  # Replace with your NVR password
NVR_PORT = "554"
CAMERA_MAP = {
    '1': "10.20.1.7",   # Example IP, replace with your camera IPs
    '2': "10.20.1.8",
    '3': "10.20.1.18",
    '4': "10.20.1.19"
    # Add more cameras as needed
}
ENCODED_NVR_PASSWORD = urllib.parse.quote(NVR_PASSWORD)
# --- End RTSP Configuration ---

# --- Parking Tracking Globals ---
parking_spot_states = {} 
parking_events_log = []
# --- End Parking Tracking Globals ---

cap = None
is_video_file = False

print("Select video source:")
print("1: Video File (carPark.mp4)")
print("2: RTSP Camera Stream")
source_choice = input("Enter choice (1 or 2): ")

if source_choice == '1':
    cap = cv2.VideoCapture('carPark.mp4')
    is_video_file = True
    if not cap.isOpened():
        print("Error: Could not open video file 'carPark.mp4'")
        exit()
    print("Using video file 'carPark.mp4' as source.")
elif source_choice == '2':
    print("\nAvailable cameras:")
    for cam_id, cam_ip in CAMERA_MAP.items():
        print(f"  Camera {cam_id}: {cam_ip}")
    
    camera_id_input = input("Enter camera number: ").strip()
    if camera_id_input in CAMERA_MAP:
        camera_ip = CAMERA_MAP[camera_id_input]
        # Try a couple of common RTSP URL patterns
        # Pattern 1: Main stream (often higher quality)
        rtsp_url_main = f"rtsp://{NVR_USERNAME}:{ENCODED_NVR_PASSWORD}@{camera_ip}:{NVR_PORT}/h264/ch{camera_id_input}/main/av_stream?rtsp_transport=tcp"
        # Pattern 2: Sub stream (often lower quality, but more stable if main fails)
        rtsp_url_sub = f"rtsp://{NVR_USERNAME}:{ENCODED_NVR_PASSWORD}@{camera_ip}:{NVR_PORT}/h264/ch{camera_id_input}/sub/av_stream?rtsp_transport=tcp"
        # Alternative patterns if the above don't work for your camera
        # rtsp_url_alt_main = f"rtsp://{NVR_USERNAME}:{ENCODED_NVR_PASSWORD}@{camera_ip}:{NVR_PORT}/Streaming/Channels/{camera_id_input}01?rtsp_transport=tcp"
        # rtsp_url_alt_sub = f"rtsp://{NVR_USERNAME}:{ENCODED_NVR_PASSWORD}@{camera_ip}:{NVR_PORT}/Streaming/Channels/{camera_id_input}02?rtsp_transport=tcp"

        print(f"Attempting to connect to camera {camera_id_input} at {camera_ip}...")
        print(f"Trying main stream: {rtsp_url_main}")
        cap = cv2.VideoCapture(rtsp_url_main, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Failed to open main stream. Trying sub stream: {rtsp_url_sub}")
            cap = cv2.VideoCapture(rtsp_url_sub, cv2.CAP_FFMPEG)
            if cap.isOpened():
                print("Connected to sub stream.")
            else:
                # print(f"Failed to open sub stream. Trying alternative main stream: {rtsp_url_alt_main}")
                # cap = cv2.VideoCapture(rtsp_url_alt_main, cv2.CAP_FFMPEG)
                # if not cap.isOpened():
                #     print(f"Failed to open alternative main stream. Trying alternative sub stream: {rtsp_url_alt_sub}")
                #     cap = cv2.VideoCapture(rtsp_url_alt_sub, cv2.CAP_FFMPEG)
                #     if cap.isOpened():
                #         print("Connected to alternative sub stream.")
                #     else:
                print(f"Error: Could not connect to camera {camera_id_input} using common RTSP URLs.")
                exit()
        else:
            print("Connected to main stream.")
        is_video_file = False
    else:
        print("Error: Invalid camera number selected.")
        exit()
else:
    print("Error: Invalid source choice.")
    exit()

with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

# Initialize parking_spot_states based on the number of parking spots
if posList:
    for i in range(len(posList)):
        parking_spot_states[i] = {'object_class': None, 'entry_time': None}

width, height = 107, 48


def checkParkingSpace(imgPro, yolo_detections=None):
    global parking_spot_states, parking_events_log # Use global variables for state and log
    spaceCounter = 0

    for spot_id, pos_entry in enumerate(posList):
        # Default to global width and height, used for legacy or as fallback
        current_rect_width, current_rect_height = width, height 
        
        shape_type = None
        poly_points_for_drawing = None # For drawing the outline
        x_coord, y_coord = -1, -1 # Initialize coordinates for text and checks

        # Determine shape type and extract relevant data (same as original)
        if isinstance(pos_entry[-1], str) and pos_entry[-1] == 'polygon':
            shape_type = 'polygon'
            points_flat = pos_entry[:-1]
            if len(points_flat) < 6 or len(points_flat) % 2 != 0:
                # print(f"Skipping malformed polygon entry: {pos_entry}") # Optional: reduce console noise
                continue
            poly_points_coords = [(points_flat[i], points_flat[i+1]) for i in range(0, len(points_flat), 2)]
            np_poly_points = np.array(poly_points_coords)
            poly_points_for_drawing = np_poly_points.astype(np.int32)
            x_coord = int(np_poly_points[:, 0].min())
            y_coord = int(np_poly_points[:, 1].min())
            current_rect_width = int(np_poly_points[:, 0].max() - x_coord)
            current_rect_height = int(np_poly_points[:, 1].max() - y_coord)
        elif len(pos_entry) == 5:
            shape_type = 'rectangle'
            x_coord, y_coord, current_rect_width, current_rect_height, _ = pos_entry
        elif len(pos_entry) == 2:
            shape_type = 'rectangle'
            x_coord, y_coord = pos_entry
        else:
            # print(f"Skipping unknown pos format: {pos_entry}") # Optional: reduce console noise
            continue

        # 1. Check if YOLO detected a target object in this specific spot
        is_yolo_occupied_here = False
        yolo_class_name_here = None
        if model and yolo_detections:
            for det in yolo_detections:
                obj_bbox = det['bbox']
                obj_class_name = det['class_name']
                obj_center_x = (obj_bbox[0] + obj_bbox[2]) // 2
                obj_center_y = (obj_bbox[1] + obj_bbox[3]) // 2
                if obj_class_name in YOLO_TARGET_CLASSES_NAMES:
                    if shape_type == 'polygon':
                        if cv2.pointPolygonTest(np_poly_points, (obj_center_x, obj_center_y), False) >= 0:
                            is_yolo_occupied_here = True
                            yolo_class_name_here = obj_class_name
                            break
                    elif shape_type == 'rectangle':
                        if x_coord <= obj_center_x < x_coord + current_rect_width and \
                           y_coord <= obj_center_y < y_coord + current_rect_height:
                            is_yolo_occupied_here = True
                            yolo_class_name_here = obj_class_name
                            break
        
        # 2. Calculate pixel count for this spot using imgPro (the processed image)
        spot_pixel_count = 0
        if shape_type == 'polygon':
            if current_rect_width <= 0 or current_rect_height <= 0:
                spot_pixel_count = 999 # Mark as occupied if issue
            else:
                imgCrop = imgPro[y_coord:y_coord + current_rect_height, x_coord:x_coord + current_rect_width]
                if imgCrop.size == 0:
                    spot_pixel_count = 999 
                else:
                    mask = np.zeros(imgCrop.shape[:2], dtype=np.uint8)
                    relative_poly_points = np_poly_points - [x_coord, y_coord]
                    cv2.fillPoly(mask, [relative_poly_points.astype(np.int32)], 255)
                    # Ensure imgCrop is single channel for bitwise_and with mask if it's multi-channel
                    # imgPro should already be single channel (imgDilate)
                    masked_img_crop = cv2.bitwise_and(imgCrop, imgCrop, mask=mask)
                    spot_pixel_count = cv2.countNonZero(masked_img_crop)
        elif shape_type == 'rectangle':
            imgCrop = imgPro[y_coord:y_coord + current_rect_height, x_coord:x_coord + current_rect_width]
            spot_pixel_count = cv2.countNonZero(imgCrop) if imgCrop.size > 0 else 999
        else: # Should not be reached due to earlier continue
            spot_pixel_count = 999

        # 3. Determine current frame's actual occupant for tracking
        current_actual_occupant = None
        if is_yolo_occupied_here:
            current_actual_occupant = yolo_class_name_here
        elif spot_pixel_count >= 900: # Pixel threshold for non-YOLO occupancy
            current_actual_occupant = "UnknownVehicle_Pixel"
        # Else, current_actual_occupant remains None (spot is free)

        # 4. State Transition Logic
        prev_state = parking_spot_states[spot_id]
        prev_occupant_class_in_state = prev_state['object_class']
        prev_entry_time_in_state = prev_state['entry_time']

        if prev_occupant_class_in_state != current_actual_occupant:
            now = datetime.now()
            if prev_occupant_class_in_state is not None and prev_entry_time_in_state is not None:
                # Previous occupant is leaving or changing
                duration = now - prev_entry_time_in_state
                parking_events_log.append({
                    'Object_detected': prev_occupant_class_in_state,
                    'Entry_time': prev_entry_time_in_state,
                    'Exit_time': now,
                    'Total_duration': duration,
                    'Parking_ID': spot_id
                })
            
            if current_actual_occupant is not None:
                # New occupant is entering
                parking_spot_states[spot_id]['object_class'] = current_actual_occupant
                parking_spot_states[spot_id]['entry_time'] = now
            else:
                # Spot became empty
                parking_spot_states[spot_id]['object_class'] = None
                parking_spot_states[spot_id]['entry_time'] = None
        
        # 5. Setup display properties based on the *updated* (current frame's) tracking state
        final_tracked_state = parking_spot_states[spot_id] # Get the latest state for display
        text_for_display_spot = ""
        color_for_display_spot = (0,0,0) # Default color
        thickness_for_display_spot = 1 # Default thickness

        if final_tracked_state['object_class'] is not None: # Spot is tracked as occupied
            color_for_display_spot = (0, 0, 255)  # Red for occupied
            thickness_for_display_spot = 2
            if final_tracked_state['object_class'] == "UnknownVehicle_Pixel":
                text_for_display_spot = f"P:{spot_pixel_count}" # Show pixel count that triggered this
            else: # YOLO detected class
                text_for_display_spot = str(final_tracked_state['object_class'])[:10] # Limit length for display
        else: # Spot is tracked as free
            spaceCounter += 1
            color_for_display_spot = (0, 255, 0)  # Green for free
            thickness_for_display_spot = 5
            text_for_display_spot = str(spot_pixel_count) # Show current pixel count for free spots
        
        # 6. Drawing the space outline and text
        text_pos_y_offset = current_rect_height # Default for text below shape
        if shape_type == 'polygon' and poly_points_for_drawing is not None:
            cv2.polylines(img, [poly_points_for_drawing], isClosed=True, color=color_for_display_spot, thickness=thickness_for_display_spot)
            text_anchor_x, text_anchor_y = x_coord, y_coord 
        elif shape_type == 'rectangle':
            cv2.rectangle(img, (x_coord, y_coord), (x_coord + current_rect_width, y_coord + current_rect_height), color_for_display_spot, thickness_for_display_spot)
            text_anchor_x, text_anchor_y = x_coord, y_coord
        else: 
            continue # Should not be reached

        # Adjust text display logic (same as original, but using determined text_for_display_spot)
        cvzone.putTextRect(img, text_for_display_spot, (text_anchor_x, text_anchor_y + text_pos_y_offset - 3), scale=1,
                           thickness=2, offset=0, colorR=color_for_display_spot)

    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                           thickness=5, offset=20, colorR=(0,200,0))

def save_parking_report(log, filename="parking_report.csv"):
    if not log:
        print("No parking events to report.")
        return

    print(f"Saving parking report to {filename}...")
    fieldnames = ['Object_detected', 'Entry_time', 'Exit_time', 'Total_duration', 'Parking_ID']
    
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for event in log:
            # Format datetime and timedelta objects for CSV
            formatted_event = {
                'Object_detected': event['Object_detected'],
                'Entry_time': event['Entry_time'].strftime("%Y-%m-%d %H:%M:%S") if event['Entry_time'] else '',
                'Exit_time': event['Exit_time'].strftime("%Y-%m-%d %H:%M:%S") if event['Exit_time'] else '',
                'Total_duration': str(event['Total_duration']) if event['Total_duration'] else '', # timedelta to string
                'Parking_ID': event['Parking_ID']
            }
            writer.writerow(formatted_event)
    print(f"Parking report saved to {filename}.")


try:
    while True:
        # Video file looping logic
        if is_video_file:
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if total_frames > 0 and current_pos >= total_frames: # Check if total_frames is valid
                # print("End of video file reached. Resetting.") # Less verbose
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, img = cap.read() # Read immediately
            else:
                success, img = cap.read()
        else: # RTSP or other stream
            success, img = cap.read()

        if not success:
            # print("Failed to read frame (success=False).") # Less verbose
            if is_video_file:
                # print("Assuming end of video or read error. Resetting video.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01) # Brief pause
                continue 
            else: 
                # print("RTSP stream read failure. Waiting and trying to continue...")
                time.sleep(0.5) 
                continue 

        if img is None or img.size == 0:
            # print("Read an empty frame (img is None or size 0).") # Less verbose
            if is_video_file:
                # print("Empty frame from video file. Resetting video.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue
            else: 
                # print("Empty frame from RTSP stream. Waiting and trying to continue...")
                time.sleep(0.5)
                continue
        
        # Image processing
        try:
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)

            if imgBlur is None or imgBlur.size == 0:
                # print("Error: imgBlur is None or empty after GaussianBlur. Skipping frame.")
                continue
            
            if imgBlur.dtype != np.uint8:
                # print(f"Warning: imgBlur dtype is {imgBlur.dtype}, forcing to uint8.")
                imgBlur = np.uint8(imgBlur)

            # Check if image dimensions are sufficient for the adaptiveThreshold blockSize
            adaptive_block_size = 25
            if imgBlur.shape[0] < adaptive_block_size or imgBlur.shape[1] < adaptive_block_size:
                # print(f"Warning: imgBlur dimensions ({imgBlur.shape[0]}x{imgBlur.shape[1]}) are too small for adaptive_block_size {adaptive_block_size}. Skipping frame.")
                continue

            imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY_INV, adaptive_block_size, 16)
            imgMedian = cv2.medianBlur(imgThreshold, 5)
            kernel = np.ones((3, 3), np.uint8)
            imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

        except cv2.error as e:
            # print(f"OpenCV error during image processing: {e}. Skipping frame.")
            continue
        except Exception as e:
            # print(f"An unexpected error occurred during image processing: {e}. Skipping frame.")
            continue
        
        # --- YOLO Detection (on the original 'img') ---
        yolo_detections_processed = []
        if model and success: 
            results = model.predict(img, conf=YOLO_CONFIDENCE, device=YOLO_DEVICE, verbose=False)
            if results and results[0].boxes:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    
                    yolo_detections_processed.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class_id': cls_id,
                        'class_name': class_name
                    })
                    
                    # Draw all TARGET YOLO detections on the main image
                    if class_name in YOLO_TARGET_CLASSES_NAMES:
                         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2) 
                         cvzone.putTextRect(img, f'{class_name} {conf:.2f}', (x1, y1 - 10), 
                                           scale=0.5, thickness=1, colorR=(255,0,255), offset=3)
        # --- End YOLO Detection ---

        checkParkingSpace(imgDilate, yolo_detections_processed) # Pass imgDilate for pixel counting
        cv2.imshow("Image", img)
        # cv2.imshow("ImageBlur", imgBlur)
        # cv2.imshow("ImageThres", imgMedian) # imgMedian is imgDilate before dilation

        key = cv2.waitKey(10) & 0xFF # Reduced wait time for smoother video, was 10
        if key == ord('q'):
            print("Quitting application...")
            break
finally:
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    
    # Log exit for any objects still in spots when script ends
    current_time_on_exit = datetime.now()
    if parking_spot_states: # Ensure it was initialized
        for spot_id, state in parking_spot_states.items():
            if state['object_class'] is not None and state['entry_time'] is not None:
                duration = current_time_on_exit - state['entry_time']
                parking_events_log.append({
                    'Object_detected': state['object_class'],
                    'Entry_time': state['entry_time'],
                    'Exit_time': current_time_on_exit, 
                    'Total_duration': duration,
                    'Parking_ID': spot_id
                })
    
    save_parking_report(parking_events_log)