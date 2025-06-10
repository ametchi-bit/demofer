 #utils.py
import easyocr
import string
import pickle
from skimage.transform import resize
import numpy as np
import cv2
import csv
import pandas as pd
from collections import defaultdict


# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Mapping dictionaries for character conversion
dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}


# Write to CSV file
def write_csv(results, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score', 'car_color'])
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                if 'car' in results[frame_nmr][car_id].keys() and 'license_plate' in results[frame_nmr][car_id].keys() and 'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    # Check if car color exists, else set to None
                    car_color = results[frame_nmr][car_id]['car'].get('color', 'Unknown')
                    
                    writer.writerow([
                        frame_nmr,
                        car_id,
                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['car']['bbox']),
                        '[{} {} {} {}]'.format(*results[frame_nmr][car_id]['license_plate']['bbox']),
                        results[frame_nmr][car_id]['license_plate']['bbox_score'],
                        results[frame_nmr][car_id]['license_plate']['text'],
                        results[frame_nmr][car_id]['license_plate']['text_score'],
                        car_color
                    ])
                    
# Function to check license plate format
def license_complies_format(text):
    """
    Check if license plate text complies with any of the supported formats.
    
    Args:
        text: License plate text to check
        
    Returns:
        tuple: (is_valid, format_type, formatted_text)
    """
    # Format 1: 7-character format (AA-NNN-AA)
    if len(text) == 7:
        if ((text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and
            (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and
            (text[2] in string.digits or text[2] in dict_char_to_int.keys()) and
            (text[3] in string.digits or text[3] in dict_char_to_int.keys()) and
            (text[4] in string.digits or text[4] in dict_char_to_int.keys()) and
            (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and
            (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys())):
            
            # Format the license plate
            license_plate_ = ''
            mapping = {
                0: dict_int_to_char, 1: dict_int_to_char,
                2: dict_char_to_int, 3: dict_char_to_int, 4: dict_char_to_int,
                5: dict_int_to_char, 6: dict_int_to_char
            }
            for j in range(7):
                license_plate_ += mapping[j].get(text[j], text[j])
            
            return True, '7-char', license_plate_
    
    # Format 2: 8-character Ivorian format (NNNN-AA-NN)
    elif len(text) == 8:
        if ((text[0] in string.digits or text[0] in dict_char_to_int.keys()) and
            (text[1] in string.digits or text[1] in dict_char_to_int.keys()) and
            (text[2] in string.digits or text[2] in dict_char_to_int.keys()) and
            (text[3] in string.digits or text[3] in dict_char_to_int.keys()) and
            (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and
            (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and
            (text[6] in string.digits or text[6] in dict_char_to_int.keys()) and
            (text[7] in string.digits or text[7] in dict_char_to_int.keys())):
            
            # Format the license plate
            license_plate_ = ''
            mapping = {
                0: dict_char_to_int, 1: dict_char_to_int, 2: dict_char_to_int, 3: dict_char_to_int,
                4: dict_int_to_char, 5: dict_int_to_char,
                6: dict_char_to_int, 7: dict_char_to_int
            }
            for j in range(8):
                license_plate_ += mapping[j].get(text[j], text[j])
            
            return True, '8-char', license_plate_
    
    # No valid format found
    return False, None, text

# Function to read license plate text
def read_license_plate(license_plate_crop):
    """
    Read and validate license plate text from an image.
    
    Args:
        license_plate_crop: Cropped image of the license plate
        
    Returns:
        tuple: (formatted_text, confidence_score) or (None, None) if invalid
    """
    detections = reader.readtext(license_plate_crop)
    
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        
        is_valid, format_type, formatted_text = license_complies_format(text)
        
        if is_valid:
            print(f"Valid license plate detected: {formatted_text} (Format: {format_type})")
            return formatted_text, score
    
    return None, None

# Function to get car details
def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return vehicle_track_ids[j]
    return -1, -1, -1, -1, -1



# ========================================
# Get car color and final license plate csv
# ========================================

def get_car_color(car_crop):
    """
    Detect the dominant color of a car crop
    
    Args:
        car_crop: Cropped image of the car (numpy array)
        
    Returns:
        str: Detected car color name
    """
    import cv2
    import numpy as np
    from collections import Counter
    
    # Resize for faster processing
    car_crop = cv2.resize(car_crop, (100, 100))
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(car_crop, cv2.COLOR_BGR2HSV)
    
    # Define color ranges in HSV
    color_ranges = {
        'White': [(0, 0, 200), (180, 30, 255)],
        'Black': [(0, 0, 0), (180, 255, 50)],
        'Gray': [(0, 0, 50), (180, 30, 200)],
        'Red': [(0, 120, 70), (10, 255, 255)],
        'Red2': [(170, 120, 70), (180, 255, 255)],  # Red wraps around
        'Blue': [(100, 150, 0), (130, 255, 255)],
        'Green': [(40, 40, 40), (80, 255, 255)],
        'Yellow': [(20, 100, 100), (30, 255, 255)],
        'Silver': [(0, 0, 180), (180, 30, 220)],
        'Brown': [(10, 50, 20), (20, 255, 200)]
    }
    
    color_counts = {}
    
    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        color_counts[color_name] = cv2.countNonZero(mask)
    
    # Handle red wrap-around
    if 'Red2' in color_counts:
        color_counts['Red'] += color_counts['Red2']
        del color_counts['Red2']
    
    # Get dominant color
    dominant_color = max(color_counts, key=color_counts.get)
    
    # Return 'Unknown' if no significant color detected
    if color_counts[dominant_color] < 100:
        return 'Unknown'
    
    return dominant_color

def generate_final_results_csv(license_plate_results, interpolated_results, output_path="final_results.csv"):
    """
    Generate final CSV with unique license plates per car including crops
    
    Args:
        license_plate_results: Results from license plate processor
        interpolated_results: Results from interpolation
        output_path: Path for final CSV file
        
    Returns:
        str: Path to generated CSV file
    """
    
    final_data = []
    car_license_map = defaultdict(list)
    
    # Group detections by car_id and find best detection for each car
    for frame_num, frame_data in license_plate_results.items():
        for car_id, detection in frame_data.items():
            if 'license_plate' in detection and 'text' in detection['license_plate']:
                car_license_map[car_id].append({
                    'frame': frame_num,
                    'text': detection['license_plate']['text'],
                    'confidence': detection['license_plate']['text_score'],
                    'bbox': detection['license_plate']['bbox'],
                    'car_bbox': detection['car']['bbox']
                })
    
    # Process each car to get unique license plate
    for car_id, detections in car_license_map.items():
        # Get detection with highest confidence
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        # Extract license plate crop (you'll need the original frame)
        # This is a placeholder - you'll need to implement actual crop extraction
        license_crop_base64 = ""  # Convert crop to base64 string
        
        # Determine car color (placeholder - you'll need car crop)
        car_color = "Unknown"  # Use get_car_color() with actual car crop
        
        final_data.append({
            'Car_id': car_id,
            'license_plate_number': best_detection['text'],
            'confidence': best_detection['confidence'],
            'car_color': car_color,
            'license_plate_crop': license_crop_base64
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(final_data)
    df.to_csv(output_path, index=False)
    
    return output_path