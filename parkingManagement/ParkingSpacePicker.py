# ParkingSpacePicker.py drawing multiple shapes horizontal and vertical, polygon, rectangle
import cv2
import pickle
import numpy as np


# Default dimensions for rectangular parking spaces
default_horizontal_width, default_horizontal_height = 107, 48
default_vertical_width, default_vertical_height = 48, 107

# Current modes
current_orientation = 'horizontal'  # 'horizontal' or 'vertical'
current_shape_mode = 'rectangle'    # 'rectangle' or 'polygon'

# State variables
is_resizing = False
resize_index = -1
resize_corner = None
resize_start_pos = None

# Polygon creation state
is_creating_polygon = False
current_polygon_points = []
polygon_preview_point = None

try:
    with open('CarParkPos', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []


def get_default_dimensions():
    """Get default width and height based on orientation"""
    if current_orientation == 'horizontal':
        return default_horizontal_width, default_horizontal_height
    else:
        return default_vertical_width, default_vertical_height


def point_in_polygon(point, polygon_points):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon_points)
    inside = False
    
    p1x, p1y = polygon_points[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def get_space_info(pos):
    """Get information about a parking space"""
    if len(pos) >= 6 and pos[-1] == 'polygon':
        # Polygon format: [point1_x, point1_y, point2_x, point2_y, ..., 'polygon']
        points = []
        for i in range(0, len(pos) - 1, 2):
            points.append((pos[i], pos[i + 1]))
        return 'polygon', points, None, None
    elif len(pos) == 5:  # Rectangle format: (x, y, width, height, orientation)
        x, y, width, height, orientation = pos
        return 'rectangle', [(x, y)], width, height
    elif len(pos) == 3:  # Format: (x, y, orientation)
        x, y, orientation = pos
        if orientation == 'horizontal':
            width, height = default_horizontal_width, default_horizontal_height
        else:
            width, height = default_vertical_width, default_vertical_height
        return 'rectangle', [(x, y)], width, height
    else:  # Legacy format: (x, y)
        x, y = pos
        width, height = default_horizontal_width, default_horizontal_height
        return 'rectangle', [(x, y)], width, height


def find_space_at_position(x, y):
    """Find parking space at given position"""
    for i, pos in enumerate(posList):
        shape_type, points, width, height = get_space_info(pos)
        
        if shape_type == 'polygon':
            if point_in_polygon((x, y), points):
                # Find closest point for editing
                min_dist = float('inf')
                closest_point = -1
                for j, point in enumerate(points):
                    dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                    if dist < min_dist and dist < 15:  # 15 pixel threshold
                        min_dist = dist
                        closest_point = j
                return i, 'polygon_point', closest_point
        else:  # Rectangle
            px, py = points[0]
            if px <= x <= px + width and py <= y <= py + height:
                # Determine resize handle
                corner_size = 15
                if px + width - corner_size <= x <= px + width and py + height - corner_size <= y <= py + height:
                    return i, 'bottom-right', None
                elif px <= x <= px + corner_size and py <= y <= py + corner_size:
                    return i, 'top-left', None
                elif px + width - corner_size <= x <= px + width and py <= y <= py + corner_size:
                    return i, 'top-right', None
                elif px <= x <= px + corner_size and py + height - corner_size <= y <= py + height:
                    return i, 'bottom-left', None
                elif px + width - corner_size <= x <= px + width:
                    return i, 'right', None
                elif px <= x <= px + corner_size:
                    return i, 'left', None
                elif py + height - corner_size <= y <= py + height:
                    return i, 'bottom', None
                elif py <= y <= py + corner_size:
                    return i, 'top', None
                else:
                    return i, 'move', None
    return -1, None, None


def mouseClick(events, x, y, flags, params):
    global current_orientation, current_shape_mode, is_resizing, resize_index, resize_corner, resize_start_pos
    global is_creating_polygon, current_polygon_points, polygon_preview_point
    
    if events == cv2.EVENT_LBUTTONDOWN:
        if current_shape_mode == 'polygon':
            if not is_creating_polygon:
                # Start creating a new polygon
                is_creating_polygon = True
                current_polygon_points = [(x, y)]
            else:
                # Add point to current polygon
                # Check if we're closing the polygon (click near first point)
                first_point = current_polygon_points[0]
                dist_to_first = np.sqrt((x - first_point[0])**2 + (y - first_point[1])**2)
                
                if len(current_polygon_points) >= 3 and dist_to_first < 15:
                    # Close polygon and save it
                    polygon_data = []
                    for point in current_polygon_points:
                        polygon_data.extend([point[0], point[1]])
                    polygon_data.append('polygon')
                    posList.append(tuple(polygon_data))
                    
                    # Reset polygon creation
                    is_creating_polygon = False
                    current_polygon_points = []
                else:
                    # Add new point
                    current_polygon_points.append((x, y))
        else:
            # Rectangle mode
            space_index, corner, point_index = find_space_at_position(x, y)
            
            if space_index >= 0 and corner and corner != 'move':
                if corner == 'polygon_point':
                    # Start editing polygon point
                    is_resizing = True
                    resize_index = space_index
                    resize_corner = 'polygon_point'
                    resize_start_pos = point_index
                else:
                    # Start resizing rectangle
                    is_resizing = True
                    resize_index = space_index
                    resize_corner = corner
                    resize_start_pos = (x, y)
            elif space_index >= 0 and corner == 'move':
                print(f"Clicked inside space {space_index}")
            else:
                # Add new rectangular parking space
                width, height = get_default_dimensions()
                posList.append((x, y, width, height, current_orientation))
                
    elif events == cv2.EVENT_RBUTTONDOWN:
        if is_creating_polygon:
            # Cancel polygon creation
            is_creating_polygon = False
            current_polygon_points = []
        else:
            # Remove parking space
            space_index, _, _ = find_space_at_position(x, y)
            if space_index >= 0:
                posList.pop(space_index)
                
    elif events == cv2.EVENT_MOUSEMOVE:
        if current_shape_mode == 'polygon' and is_creating_polygon:
            # Update polygon preview
            polygon_preview_point = (x, y)
        elif is_resizing:
            # Handle resizing
            if resize_index >= 0 and resize_index < len(posList):
                pos = posList[resize_index]
                shape_type, points, width, height = get_space_info(pos)
                
                if resize_corner == 'polygon_point':
                    # Move polygon point
                    polygon_data = list(pos[:-1])  # Remove 'polygon' marker
                    point_idx = resize_start_pos * 2
                    polygon_data[point_idx] = x
                    polygon_data[point_idx + 1] = y
                    polygon_data.append('polygon')
                    posList[resize_index] = tuple(polygon_data)
                    
                elif shape_type == 'rectangle':
                    # Resize rectangle (same as before)
                    px, py, old_width, old_height = pos[0], pos[1], pos[2], pos[3]
                    orientation = pos[4] if len(pos) >= 5 else 'horizontal'
                    
                    if resize_corner == 'bottom-right':
                        new_width = max(20, x - px)
                        new_height = max(20, y - py)
                    elif resize_corner == 'top-left':
                        new_width = max(20, px + old_width - x)
                        new_height = max(20, py + old_height - y)
                        px, py = x, y
                    elif resize_corner == 'top-right':
                        new_width = max(20, x - px)
                        new_height = max(20, py + old_height - y)
                        py = y
                    elif resize_corner == 'bottom-left':
                        new_width = max(20, px + old_width - x)
                        new_height = max(20, y - py)
                        px = x
                    elif resize_corner == 'right':
                        new_width = max(20, x - px)
                        new_height = old_height
                    elif resize_corner == 'left':
                        new_width = max(20, px + old_width - x)
                        new_height = old_height
                        px = x
                    elif resize_corner == 'bottom':
                        new_width = old_width
                        new_height = max(20, y - py)
                    elif resize_corner == 'top':
                        new_width = old_width
                        new_height = max(20, py + old_height - y)
                        py = y
                    else:
                        new_width, new_height = old_width, old_height
                    
                    posList[resize_index] = (px, py, new_width, new_height, orientation)
                
    elif events == cv2.EVENT_LBUTTONUP:
        # Stop resizing
        if is_resizing:
            is_resizing = False
            resize_index = -1
            resize_corner = None
            resize_start_pos = None

    # Save updated list
    with open('CarParkPos', 'wb') as f:
        pickle.dump(posList, f)


def draw_instructions(img):
    """Draw instructions on the image"""
    instructions = [
        f"Mode: {current_shape_mode.upper()} | Orient: {current_orientation.upper()}",
        "Shape Mode:",
        "R - Rectangle mode",
        "P - Polygon mode",
        "Orientation (Rectangle):",
        "H - Horizontal | V - Vertical",
        "Actions:",
        "Left Click - Add/Edit space",
        "Right Click - Remove/Cancel",
        "Polygon: Click points, close near start",
        "Q - Quit"
    ]
    
    # Draw background for instructions
    bg_height = len(instructions) * 25 + 40
    cv2.rectangle(img, (10, 10), (420, bg_height), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (420, bg_height), (255, 255, 255), 2)
    
    # Draw text
    for i, instruction in enumerate(instructions):
        if i == 0:
            color = (0, 255, 0)  # Green for current mode
        elif ":" in instruction and not instruction.startswith(" "):
            color = (255, 255, 0)  # Yellow for headers
        else:
            color = (255, 255, 255)  # White for regular text
            
        cv2.putText(img, instruction, (20, 35 + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_polygon_handles(img, points):
    """Draw handles on polygon points"""
    handle_size = 8
    handle_color = (255, 255, 0)  # Yellow handles
    
    for point in points:
        cv2.circle(img, point, handle_size, handle_color, -1)
        cv2.circle(img, point, handle_size + 2, (0, 0, 0), 2)


def draw_rectangle_handles(img, x, y, width, height):
    """Draw resize handles on corners and edges of rectangle"""
    handle_size = 8
    handle_color = (255, 255, 0)  # Yellow handles
    
    # Corner handles
    corners = [(x, y), (x + width, y), (x, y + height), (x + width, y + height)]
    for corner in corners:
        cv2.rectangle(img, 
                     (corner[0] - handle_size//2, corner[1] - handle_size//2),
                     (corner[0] + handle_size//2, corner[1] + handle_size//2),
                     handle_color, -1)
    
    # Edge handles
    edges = [(x + width//2, y), (x + width//2, y + height), (x, y + height//2), (x + width, y + height//2)]
    for edge in edges:
        cv2.rectangle(img,
                     (edge[0] - handle_size//2, edge[1] - handle_size//2),
                     (edge[0] + handle_size//2, edge[1] + handle_size//2),
                     handle_color, -1)


while True:
    img = cv2.imread('carParkImg.png')
    
    if img is None:
        print("Error: Could not load image 'carParkImg.png'")
        break
    
    # Draw all parking spaces
    for i, pos in enumerate(posList):
        shape_type, points, width, height = get_space_info(pos)
        
        if shape_type == 'polygon':
            # Draw polygon
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)  # Cyan for polygons
            cv2.fillPoly(img, [pts], (0, 255, 255, 50))  # Semi-transparent fill
            
            # Draw point count
            center_x = sum(p[0] for p in points) // len(points)
            center_y = sum(p[1] for p in points) // len(points)
            cv2.putText(img, f"POLY {len(points)}pts", (center_x - 40, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw handles if being edited
            if is_resizing and resize_index == i:
                draw_polygon_handles(img, points)
                
        else:  # Rectangle
            x, y = points[0]
            # Convert old formats to new format
            if len(pos) == 3:
                orientation = pos[2]
                if orientation == 'horizontal':
                    width, height = default_horizontal_width, default_horizontal_height
                else:
                    width, height = default_vertical_width, default_vertical_height
                posList[i] = (x, y, width, height, orientation)
            elif len(pos) == 2:
                width, height = default_horizontal_width, default_horizontal_height
                orientation = 'horizontal'
                posList[i] = (x, y, width, height, orientation)
            else:
                orientation = pos[4] if len(pos) >= 5 else 'horizontal'
            
            # Draw rectangle
            color = (255, 0, 255) if orientation == 'horizontal' else (0, 255, 255)
            cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
            
            # Draw size info
            text = f"{orientation[0].upper()} {width}x{height}"
            cv2.putText(img, text, (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw handles if being resized
            if is_resizing and resize_index == i:
                draw_rectangle_handles(img, x, y, width, height)
    
    # Draw polygon creation preview
    if is_creating_polygon and current_polygon_points:
        # Draw current polygon points and lines
        for i, point in enumerate(current_polygon_points):
            cv2.circle(img, point, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(img, current_polygon_points[i-1], point, (0, 255, 0), 2)
        
        # Draw preview line to mouse
        if polygon_preview_point:
            cv2.line(img, current_polygon_points[-1], polygon_preview_point, (128, 255, 128), 1)
            
        # Draw closing line preview if near start
        if len(current_polygon_points) >= 3 and polygon_preview_point:
            dist_to_first = np.sqrt((polygon_preview_point[0] - current_polygon_points[0][0])**2 + 
                                  (polygon_preview_point[1] - current_polygon_points[0][1])**2)
            if dist_to_first < 15:
                cv2.line(img, current_polygon_points[-1], current_polygon_points[0], (0, 255, 0), 2)
                cv2.putText(img, "Click to close polygon", (polygon_preview_point[0] + 10, polygon_preview_point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw instructions
    draw_instructions(img)
    
    cv2.imshow("Parking Space Picker", img)
    cv2.setMouseCallback("Parking Space Picker", mouseClick)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('h') or key == ord('H'):
        current_orientation = 'horizontal'
        print("Switched to HORIZONTAL mode")
    elif key == ord('v') or key == ord('V'):
        current_orientation = 'vertical'
        print("Switched to VERTICAL mode")
    elif key == ord('r') or key == ord('R'):
        current_shape_mode = 'rectangle'
        is_creating_polygon = False
        current_polygon_points = []
        print("Switched to RECTANGLE mode")
    elif key == ord('p') or key == ord('P'):
        current_shape_mode = 'polygon'
        print("Switched to POLYGON mode")

cv2.destroyAllWindows()
print(f"Saved {len(posList)} parking spaces")
print("Rectangle spaces: Magenta (horizontal) / Cyan (vertical)")
print("Polygon spaces: Cyan with point count")
print("Data formats: Rectangle (x,y,w,h,orientation) | Polygon (x1,y1,x2,y2,...,'polygon')")