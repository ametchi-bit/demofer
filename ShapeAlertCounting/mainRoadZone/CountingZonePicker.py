# Enhanced Shape Drawer with Lines, Naming, and Color Assignment
import cv2
import pickle
import numpy as np
import json

# Default dimensions for rectangular parking spaces
default_horizontal_width, default_horizontal_height = 107, 48
default_vertical_width, default_vertical_height = 48, 107

# Available colors for shapes
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

# Current modes
current_orientation = 'horizontal'  # 'horizontal' or 'vertical'
current_shape_mode = 'rectangle'    # 'rectangle', 'polygon', or 'line'

# State variables
is_resizing = False
resize_index = -1
resize_corner = None
resize_start_pos = None

# Polygon creation state
is_creating_polygon = False
current_polygon_points = []
polygon_preview_point = None

# Line creation state
is_creating_line = False
line_start_point = None
line_preview_point = None

# Shape naming and editing
show_naming_dialog = False
selected_shape_index = -1
current_input_text = ""
input_mode = "name"  # "name" or "color"

try:
    with open('CountingZonePos copy', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

# Convert old format to new format with metadata
def upgrade_shape_format():
    """Upgrade old shape format to include metadata"""
    global posList
    new_posList = []
    
    for i, pos in enumerate(posList):
        if isinstance(pos, dict):
            # Already in new format
            new_posList.append(pos)
        else:
            # Convert old format to new format
            shape_type, points, width, height = get_shape_info(pos)
            
            if shape_type == 'polygon':
                new_shape = {
                    'type': 'polygon',
                    'points': points,
                    'name': f'Polygon_{i+1}',
                    'color': 'cyan'
                }
            elif shape_type == 'rectangle':
                orientation = pos[4] if len(pos) >= 5 else 'horizontal'
                new_shape = {
                    'type': 'rectangle',
                    'x': points[0][0],
                    'y': points[0][1],
                    'width': width,
                    'height': height,
                    'orientation': orientation,
                    'name': f'Parking_{i+1}',
                    'color': 'magenta' if orientation == 'horizontal' else 'cyan'
                }
            new_posList.append(new_shape)
    
    posList = new_posList

def get_default_dimensions():
    """Get default width and height based on orientation"""
    if current_orientation == 'horizontal':
        return default_horizontal_width, default_horizontal_height
    else:
        return default_vertical_width, default_vertical_height

def get_shape_info(shape):
    """Get information about a shape (backward compatibility)"""
    if isinstance(shape, dict):
        # New format
        if shape['type'] == 'polygon':
            return 'polygon', shape['points'], None, None
        elif shape['type'] == 'rectangle':
            return 'rectangle', [(shape['x'], shape['y'])], shape['width'], shape['height']
        elif shape['type'] == 'line':
            return 'line', [shape['start'], shape['end']], None, None
    else:
        # Old format (backward compatibility)
        if len(shape) >= 6 and shape[-1] == 'polygon':
            points = []
            for i in range(0, len(shape) - 1, 2):
                points.append((shape[i], shape[i + 1]))
            return 'polygon', points, None, None
        elif len(shape) == 5:
            x, y, width, height, orientation = shape
            return 'rectangle', [(x, y)], width, height
        elif len(shape) == 3:
            x, y, orientation = shape
            if orientation == 'horizontal':
                width, height = default_horizontal_width, default_horizontal_height
            else:
                width, height = default_vertical_width, default_vertical_height
            return 'rectangle', [(x, y)], width, height
        else:
            x, y = shape
            width, height = default_horizontal_width, default_horizontal_height
            return 'rectangle', [(x, y)], width, height

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

def point_to_line_distance(point, line_start, line_end):
    """Calculate the distance from a point to a line segment"""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Calculate the distance from point to line segment
    A = x0 - x1
    B = y0 - y1
    C = x2 - x1
    D = y2 - y1
    
    dot = A * C + B * D
    len_sq = C * C + D * D
    
    if len_sq == 0:
        return np.sqrt(A * A + B * B)
    
    param = dot / len_sq
    
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D
    
    dx = x0 - xx
    dy = y0 - yy
    return np.sqrt(dx * dx + dy * dy)

def find_shape_at_position(x, y):
    """Find shape at given position"""
    for i, shape in enumerate(posList):
        if not isinstance(shape, dict):
            continue
            
        if shape['type'] == 'polygon':
            if point_in_polygon((x, y), shape['points']):
                # Find closest point for editing
                min_dist = float('inf')
                closest_point = -1
                for j, point in enumerate(shape['points']):
                    dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
                    if dist < min_dist and dist < 15:
                        min_dist = dist
                        closest_point = j
                return i, 'polygon_point', closest_point
                
        elif shape['type'] == 'rectangle':
            px, py = shape['x'], shape['y']
            width, height = shape['width'], shape['height']
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
                    
        elif shape['type'] == 'line':
            dist = point_to_line_distance((x, y), shape['start'], shape['end'])
            if dist < 10:  # 10 pixel threshold for line selection
                # Check if near start or end point
                start_dist = np.sqrt((x - shape['start'][0])**2 + (y - shape['start'][1])**2)
                end_dist = np.sqrt((x - shape['end'][0])**2 + (y - shape['end'][1])**2)
                
                if start_dist < 15:
                    return i, 'line_start', None
                elif end_dist < 15:
                    return i, 'line_end', None
                else:
                    return i, 'line_move', None
    
    return -1, None, None

def show_naming_interface(img, shape_index):
    """Show naming interface for a shape"""
    global current_input_text, input_mode
    
    shape = posList[shape_index]
    
    # Draw dialog background
    dialog_width, dialog_height = 400, 200
    start_x = (img.shape[1] - dialog_width) // 2
    start_y = (img.shape[0] - dialog_height) // 2
    
    # Background
    cv2.rectangle(img, (start_x, start_y), (start_x + dialog_width, start_y + dialog_height), (50, 50, 50), -1)
    cv2.rectangle(img, (start_x, start_y), (start_x + dialog_width, start_y + dialog_height), (255, 255, 255), 2)
    
    # Title
    cv2.putText(img, f"Edit {shape['type'].title()}", (start_x + 20, start_y + 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Current name
    cv2.putText(img, f"Name: {shape.get('name', 'Unnamed')}", (start_x + 20, start_y + 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Current color
    color_bgr = COLORS.get(shape.get('color', 'white'), (255, 255, 255))
    cv2.rectangle(img, (start_x + 20, start_y + 75), (start_x + 60, start_y + 95), color_bgr, -1)
    cv2.putText(img, f"Color: {shape.get('color', 'white')}", (start_x + 70, start_y + 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Input field
    input_y = start_y + 120
    cv2.rectangle(img, (start_x + 20, input_y - 5), (start_x + 380, input_y + 25), (100, 100, 100), -1)
    cv2.rectangle(img, (start_x + 20, input_y - 5), (start_x + 380, input_y + 25), (255, 255, 255), 1)
    
    if input_mode == "name":
        prompt = f"New name: {current_input_text}_"
    else:
        prompt = f"Color ({'/'.join(COLORS.keys())}): {current_input_text}_"
    
    cv2.putText(img, prompt, (start_x + 25, input_y + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Instructions
    instructions = [
        "Enter = Confirm | ESC = Cancel",
        "Tab = Switch Name/Color | Delete = Remove Shape"
    ]
    
    for i, instruction in enumerate(instructions):
        cv2.putText(img, instruction, (start_x + 20, start_y + 160 + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

def mouseClick(events, x, y, flags, params):
    global current_orientation, current_shape_mode, is_resizing, resize_index, resize_corner, resize_start_pos
    global is_creating_polygon, current_polygon_points, polygon_preview_point
    global is_creating_line, line_start_point, line_preview_point
    global show_naming_dialog, selected_shape_index
    
    if show_naming_dialog:
        return  # Don't process mouse clicks during naming dialog
    
    if events == cv2.EVENT_LBUTTONDOWN:
        if current_shape_mode == 'polygon':
            if not is_creating_polygon:
                is_creating_polygon = True
                current_polygon_points = [(x, y)]
            else:
                first_point = current_polygon_points[0]
                dist_to_first = np.sqrt((x - first_point[0])**2 + (y - first_point[1])**2)
                
                if len(current_polygon_points) >= 3 and dist_to_first < 15:
                    # Close polygon and create new shape
                    new_shape = {
                        'type': 'polygon',
                        'points': current_polygon_points.copy(),
                        'name': f'Polygon_{len(posList) + 1}',
                        'color': 'cyan'
                    }
                    posList.append(new_shape)
                    
                    is_creating_polygon = False
                    current_polygon_points = []
                    
                    # Show naming dialog for the new shape
                    selected_shape_index = len(posList) - 1
                    show_naming_dialog = True
                else:
                    current_polygon_points.append((x, y))
                    
        elif current_shape_mode == 'line':
            if not is_creating_line:
                is_creating_line = True
                line_start_point = (x, y)
            else:
                # Complete line creation
                new_shape = {
                    'type': 'line',
                    'start': line_start_point,
                    'end': (x, y),
                    'name': f'Line_{len(posList) + 1}',
                    'color': 'green'
                }
                posList.append(new_shape)
                
                is_creating_line = False
                line_start_point = None
                
                # Show naming dialog for the new shape
                selected_shape_index = len(posList) - 1
                show_naming_dialog = True
                
        else:  # Rectangle mode
            shape_index, corner, point_index = find_shape_at_position(x, y)
            
            if shape_index >= 0 and corner and corner not in ['move', 'line_move']:
                if corner == 'polygon_point':
                    is_resizing = True
                    resize_index = shape_index
                    resize_corner = 'polygon_point'
                    resize_start_pos = point_index
                elif corner in ['line_start', 'line_end']:
                    is_resizing = True
                    resize_index = shape_index
                    resize_corner = corner
                    resize_start_pos = (x, y)
                else:
                    is_resizing = True
                    resize_index = shape_index
                    resize_corner = corner
                    resize_start_pos = (x, y)
            elif shape_index >= 0 and corner in ['move', 'line_move']:
                # Show naming dialog for existing shape
                selected_shape_index = shape_index
                show_naming_dialog = True
            else:
                # Add new rectangular shape
                width, height = get_default_dimensions()
                new_shape = {
                    'type': 'rectangle',
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'orientation': current_orientation,
                    'name': f'Parking_{len(posList) + 1}',
                    'color': 'magenta' if current_orientation == 'horizontal' else 'cyan'
                }
                posList.append(new_shape)
                
                # Show naming dialog for the new shape
                selected_shape_index = len(posList) - 1
                show_naming_dialog = True
                
    elif events == cv2.EVENT_RBUTTONDOWN:
        if is_creating_polygon:
            is_creating_polygon = False
            current_polygon_points = []
        elif is_creating_line:
            is_creating_line = False
            line_start_point = None
        else:
            shape_index, _, _ = find_shape_at_position(x, y)
            if shape_index >= 0:
                posList.pop(shape_index)
                
    elif events == cv2.EVENT_MOUSEMOVE:
        if current_shape_mode == 'polygon' and is_creating_polygon:
            polygon_preview_point = (x, y)
        elif current_shape_mode == 'line' and is_creating_line:
            line_preview_point = (x, y)
        elif is_resizing and resize_index >= 0 and resize_index < len(posList):
            shape = posList[resize_index]
            
            if resize_corner == 'polygon_point':
                # Move polygon point
                shape['points'][resize_start_pos] = (x, y)
            elif resize_corner in ['line_start', 'line_end']:
                # Move line endpoint
                shape[resize_corner.replace('line_', '')] = (x, y)
            elif shape['type'] == 'rectangle':
                # Resize rectangle
                old_x, old_y = shape['x'], shape['y']
                old_width, old_height = shape['width'], shape['height']
                
                if resize_corner == 'bottom-right':
                    shape['width'] = max(20, x - old_x)
                    shape['height'] = max(20, y - old_y)
                elif resize_corner == 'top-left':
                    shape['width'] = max(20, old_x + old_width - x)
                    shape['height'] = max(20, old_y + old_height - y)
                    shape['x'], shape['y'] = x, y
                # Add other resize corners as needed...
                
    elif events == cv2.EVENT_LBUTTONUP:
        if is_resizing:
            is_resizing = False
            resize_index = -1
            resize_corner = None
            resize_start_pos = None

    # Save updated list
    save_shapes()

def save_shapes():
    """Save shapes to file"""
    with open('CountingZonePos copy', 'wb') as f:
        pickle.dump(posList, f)
    
    # Also save as JSON for easy reading
    json_data = []
    for shape in posList:
        if isinstance(shape, dict):
            json_data.append(shape)
    
    with open('shapes_config.json', 'w') as f:
        json.dump(json_data, f, indent=2)

def draw_instructions(img):
    """Draw instructions on the image"""
    instructions = [
        f"Mode: {current_shape_mode.upper()} | Orient: {current_orientation.upper()}",
        "Shape Modes:",
        "R - Rectangle | P - Polygon | L - Line",
        "Orientation (Rectangle):",
        "H - Horizontal | V - Vertical",
        "Actions:",
        "Left Click - Add/Edit shape",
        "Right Click - Remove/Cancel",
        "Polygon: Click points, close near start",
        "Line: Click start, then end point",
        "Click shape to name/color it",
        "Q - Quit"
    ]
    
    bg_height = len(instructions) * 25 + 40
    cv2.rectangle(img, (10, 10), (450, bg_height), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 10), (450, bg_height), (255, 255, 255), 2)
    
    for i, instruction in enumerate(instructions):
        if i == 0:
            color = (0, 255, 0)
        elif ":" in instruction and not instruction.startswith(" "):
            color = (255, 255, 0)
        else:
            color = (255, 255, 255)
            
        cv2.putText(img, instruction, (20, 35 + i * 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_shape_handles(img, shape):
    """Draw handles for shape editing"""
    handle_size = 8
    handle_color = (255, 255, 0)
    
    if shape['type'] == 'polygon':
        for point in shape['points']:
            cv2.circle(img, point, handle_size, handle_color, -1)
            cv2.circle(img, point, handle_size + 2, (0, 0, 0), 2)
    elif shape['type'] == 'line':
        for point in [shape['start'], shape['end']]:
            cv2.circle(img, point, handle_size, handle_color, -1)
            cv2.circle(img, point, handle_size + 2, (0, 0, 0), 2)
    elif shape['type'] == 'rectangle':
        x, y, width, height = shape['x'], shape['y'], shape['width'], shape['height']
        corners = [(x, y), (x + width, y), (x, y + height), (x + width, y + height)]
        for corner in corners:
            cv2.rectangle(img, 
                         (corner[0] - handle_size//2, corner[1] - handle_size//2),
                         (corner[0] + handle_size//2, corner[1] + handle_size//2),
                         handle_color, -1)

def handle_keyboard_input(key):
    """Handle keyboard input for naming dialog"""
    global current_input_text, input_mode, show_naming_dialog, selected_shape_index
    
    if not show_naming_dialog:
        return False
    
    if key == 27:  # ESC
        show_naming_dialog = False
        current_input_text = ""
        selected_shape_index = -1
        return True
    elif key == 13:  # Enter
        if current_input_text.strip():
            shape = posList[selected_shape_index]
            if input_mode == "name":
                shape['name'] = current_input_text.strip()
            elif input_mode == "color" and current_input_text.lower() in COLORS:
                shape['color'] = current_input_text.lower()
            save_shapes()
        
        show_naming_dialog = False
        current_input_text = ""
        selected_shape_index = -1
        return True
    elif key == 9:  # Tab
        input_mode = "color" if input_mode == "name" else "name"
        current_input_text = ""
        return True
    elif key == 8:  # Backspace
        current_input_text = current_input_text[:-1]
        return True
    elif key == 127:  # Delete key
        if selected_shape_index >= 0:
            posList.pop(selected_shape_index)
            save_shapes()
        show_naming_dialog = False
        current_input_text = ""
        selected_shape_index = -1
        return True
    elif 32 <= key <= 126:  # Printable characters
        current_input_text += chr(key)
        return True
    
    return False

# Upgrade old format on startup
upgrade_shape_format()

# Main loop
while True:
    img = cv2.imread('CountingZone.png')
    
    if img is None:
        print("Error: Could not load image 'CountingZone.png'")
        break
    
    # Draw all shapes
    for i, shape in enumerate(posList):
        if not isinstance(shape, dict):
            continue
            
        color = COLORS.get(shape.get('color', 'white'), (255, 255, 255))
        name = shape.get('name', 'Unnamed')
        
        if shape['type'] == 'polygon':
            pts = np.array(shape['points'], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, color, 2)
            
            # Draw name
            center_x = sum(p[0] for p in shape['points']) // len(shape['points'])
            center_y = sum(p[1] for p in shape['points']) // len(shape['points'])
            cv2.putText(img, name, (center_x - 30, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if is_resizing and resize_index == i:
                draw_shape_handles(img, shape)
                
        elif shape['type'] == 'line':
            cv2.line(img, shape['start'], shape['end'], color, 3)
            
            # Draw name at midpoint
            mid_x = (shape['start'][0] + shape['end'][0]) // 2
            mid_y = (shape['start'][1] + shape['end'][1]) // 2
            cv2.putText(img, name, (mid_x - 30, mid_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if is_resizing and resize_index == i:
                draw_shape_handles(img, shape)
                
        elif shape['type'] == 'rectangle':
            x, y = shape['x'], shape['y']
            width, height = shape['width'], shape['height']
            
            cv2.rectangle(img, (x, y), (x + width, y + height), color, 2)
            
            # Draw name and info
            cv2.putText(img, name, (x + 5, y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img, f"{width}x{height}", (x + 5, y + height - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            if is_resizing and resize_index == i:
                draw_shape_handles(img, shape)
    
    # Draw creation previews
    if is_creating_polygon and current_polygon_points:
        for i, point in enumerate(current_polygon_points):
            cv2.circle(img, point, 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(img, current_polygon_points[i-1], point, (0, 255, 0), 2)
        
        if polygon_preview_point:
            cv2.line(img, current_polygon_points[-1], polygon_preview_point, (128, 255, 128), 1)
            
            if len(current_polygon_points) >= 3:
                dist_to_first = np.sqrt((polygon_preview_point[0] - current_polygon_points[0][0])**2 + 
                                      (polygon_preview_point[1] - current_polygon_points[0][1])**2)
                if dist_to_first < 15:
                    cv2.line(img, current_polygon_points[-1], current_polygon_points[0], (0, 255, 0), 2)
                    cv2.putText(img, "Click to close", (polygon_preview_point[0] + 10, polygon_preview_point[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if is_creating_line and line_start_point:
        cv2.circle(img, line_start_point, 5, (0, 255, 0), -1)
        if line_preview_point:
            cv2.line(img, line_start_point, line_preview_point, (0, 255, 0), 2)
            cv2.putText(img, "Click to finish line", (line_preview_point[0] + 10, line_preview_point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show naming dialog if active
    if show_naming_dialog and selected_shape_index >= 0:
        show_naming_interface(img, selected_shape_index)
    else:
        draw_instructions(img)
    
    cv2.imshow("Enhanced Shape Drawer", img)
    cv2.setMouseCallback("Enhanced Shape Drawer", mouseClick)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    # Handle naming dialog input first
    if handle_keyboard_input(key):
        continue
    
    # Regular keyboard shortcuts
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
        is_creating_line = False
        current_polygon_points = []
        line_start_point = None
        print("Switched to RECTANGLE mode")
    elif key == ord('p') or key == ord('P'):
        current_shape_mode = 'polygon'
        is_creating_line = False
        line_start_point = None
        print("Switched to POLYGON mode")
    elif key == ord('l') or key == ord('L'):
        current_shape_mode = 'line'
        is_creating_polygon = False
        current_polygon_points = []
        print("Switched to LINE mode")

cv2.destroyAllWindows()
print(f"Saved {len(posList)} shapes")
print("Shapes saved to 'CountingZonePos copy' and 'shapes_config.json'")

# Print summary of shapes
shape_summary = {}
for shape in posList:
    if isinstance(shape, dict):
        shape_type = shape['type']
        if shape_type not in shape_summary:
            shape_summary[shape_type] = 0
        shape_summary[shape_type] += 1

print("\nShape Summary:")
for shape_type, count in shape_summary.items():
    print(f"- {shape_type.title()}s: {count}")

print("\nShape naming and color assignment completed!")
print("Available colors:", ', '.join(COLORS.keys()))