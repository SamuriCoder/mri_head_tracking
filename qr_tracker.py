import cv2
import numpy as np
import math
import time
from collections import deque

TARGET_WIDTH_INCHES = 1.5

KNOWN_QR_CODE_WIDTH_CM = TARGET_WIDTH_INCHES * 2.54
MOVEMENT_THRESHOLD_MM = 2.5
STILL_TIME_THRESHOLD = 1.0

# QR Code Data
TARGET_QR_DATA = "MRI_HEAD_MOTION_TRACKER_V1.0"
CENTER_QR_DATA = "MRI_CENTER_LOC"

center_pixel = None
is_center_set = False
pixels_per_mm = None
measurement_unit = 'mm'  # Default to millimeters

position_history = deque(maxlen=100)  # Store recent positions with timestamps
last_movement_time = None

def calculate_displacement(current_center_pixel):
    if not is_center_set or pixels_per_mm is None or pixels_per_mm == 0:
        return 0, 0, 0

    delta_x_pix = current_center_pixel[0] - center_pixel[0]
    delta_y_pix = current_center_pixel[1] - center_pixel[1]
    pixel_distance = math.sqrt(delta_x_pix**2 + delta_y_pix**2)

    mm_distance = pixel_distance / pixels_per_mm
    delta_x_mm = delta_x_pix / pixels_per_mm
    delta_y_mm = delta_y_pix / pixels_per_mm

    return mm_distance, delta_x_mm, delta_y_mm

def update_position_history(current_center_pixel, current_time):
    """Update position history for tracking stillness"""
    global last_movement_time
    
    if len(position_history) == 0:
        position_history.append((current_center_pixel, current_time))
        last_movement_time = current_time
        return
    
    recent_positions = [pos for pos, timestamp in position_history if current_time - timestamp <= 0.5]  # Last 0.5 seconds
    
    if recent_positions:
        avg_x = sum(pos[0] for pos in recent_positions) / len(recent_positions)
        avg_y = sum(pos[1] for pos in recent_positions) / len(recent_positions)
        avg_position = (avg_x, avg_y)
        
        distance_pixels = math.sqrt((current_center_pixel[0] - avg_position[0])**2 + 
                                  (current_center_pixel[1] - avg_position[1])**2)
        
        if pixels_per_mm and pixels_per_mm > 0:
            distance_mm = distance_pixels / pixels_per_mm
            
            if distance_mm > MOVEMENT_THRESHOLD_MM:
                last_movement_time = current_time
    
    position_history.append((current_center_pixel, current_time))
    
    while position_history and current_time - position_history[0][1] > 3.0:
        position_history.popleft()

def is_qr_still(current_time):
    if last_movement_time is None:
        return False
    return (current_time - last_movement_time) >= STILL_TIME_THRESHOLD

def is_qr_centered(total_distance_mm):
    return total_distance_mm <= MOVEMENT_THRESHOLD_MM

def get_status_text(is_still, is_centered):
    movement_status = "STILL" if is_still else "MOVING"
    centering_status = "Centered" if is_centered else "Not Centered"
    return f"{movement_status}; {centering_status}"

def get_status_color(is_still, is_centered):
    if is_still and is_centered:
        return (0, 255, 0)  # Green - "STILL; Centered"
    if is_still and not is_centered:
        return (0, 255, 255)  # Yellow - "STILL; Not Centered"
    else:
        return (0, 0, 255)  # Red - "MOVING; Not Centered"

def convert_to_display_units(mm_value):
    if measurement_unit == 'cm':
        return mm_value / 10.0
    return mm_value

def get_unit_label():
    return measurement_unit

def detect_qr_codes(frame, qr_detector):
    """Detect multiple QR codes in the frame and return their data and positions"""
    qr_codes = []
    
    # Use detectAndDecodeMulti to find multiple QR codes
    success, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)
    
    if success and decoded_info:
        for i, (data, bbox) in enumerate(zip(decoded_info, points)):
            if data and len(bbox) >= 4:  # Valid QR code with proper bounding box
                center = tuple(np.mean(bbox, axis=0).astype(int))
                pixel_width = np.linalg.norm(bbox[0] - bbox[1])
                qr_codes.append({
                    'data': data,
                    'center': center,
                    'bbox': bbox,
                    'pixel_width': pixel_width
                })
    
    return qr_codes

def main():
    global center_pixel, is_center_set, pixels_per_mm, measurement_unit, last_movement_time

    while True:  # Get user preference for measurement unit
        unit_choice = input("Enter measurement unit ('mm' for millimeters or 'cm' for centimeters): ").lower().strip()
        if unit_choice in ['mm', 'cm']:
            measurement_unit = unit_choice
            break
        else:
            print("Invalid choice. Please enter 'mm' or 'cm'.")

    print(f"\nUsing {measurement_unit} for measurements.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    qr_detector = cv2.QRCodeDetector()

    print('\n--- Dual QR Code Motion Tracker Initialized ---')
    print('Instructions:')
    print('  - Point the camera at both QR codes.')
    print('  - The center QR code will automatically set the reference position.')
    print('  - The target QR code movement will be tracked relative to the center.')
    print('  - Press "r" to reset the center (clear current center setting).')
    print('  - Press "q" to quit the application.')
    print(f'  - QR code is considered "Still" after {STILL_TIME_THRESHOLD} seconds within {MOVEMENT_THRESHOLD_MM}mm')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break

        current_time = time.time()
        found_target_qr = False
        found_center_qr = False
        current_center_pixel = None
        
        # Detect all QR codes in the frame
        qr_codes = detect_qr_codes(frame, qr_detector)

        for qr_code in qr_codes:
            data = qr_code['data']
            center = qr_code['center']
            bbox = qr_code['bbox']
            pixel_width = qr_code['pixel_width']

            # Handle center QR code
            if data == CENTER_QR_DATA:
                found_center_qr = True
                
                if not is_center_set or (is_center_set and 
                    math.sqrt((center[0] - center_pixel[0])**2 + (center[1] - center_pixel[1])**2) > 10):
                    center_pixel = center
                    is_center_set = True
                    last_movement_time = time.time()
                    position_history.clear()
                    print(f"Center automatically set at pixel coordinates: {center_pixel}")
                
                cv2.polylines(frame, [bbox.astype(int)], isClosed=True, color=(255, 0, 0), thickness=3)
                cv2.circle(frame, center, 8, (255, 0, 0), -1)
                cv2.putText(frame, "CENTER", (center[0] - 30, center[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 180), 2)

            # Handle target QR code
            elif data == TARGET_QR_DATA:
                found_target_qr = True
                current_center_pixel = center
                
                if pixel_width > 0:
                    pixels_per_mm = pixel_width / (KNOWN_QR_CODE_WIDTH_CM * 10)

                update_position_history(current_center_pixel, current_time)

                cv2.polylines(frame, [bbox.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.circle(frame, current_center_pixel, 5, (0, 0, 255), -1)
                cv2.putText(frame, "TARGET", (current_center_pixel[0] - 30, current_center_pixel[1] - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0), 2)

                if is_center_set:
                    total_dist_mm, dist_x_mm, dist_y_mm = calculate_displacement(current_center_pixel)
                    cv2.line(frame, center_pixel, current_center_pixel, (255, 0, 255), 2)

                    if total_dist_mm > 0:
                        dx = center_pixel[0] - current_center_pixel[0]
                        dy = center_pixel[1] - current_center_pixel[1]
                        
                        magnitude = math.sqrt(dx**2 + dy**2)
                        udx, udy = dx / magnitude, dy / magnitude

                        arrow_length = 40
                        offset_distance = 25
                        perp_dx, perp_dy = -udy, udx

                        start_point = (int(current_center_pixel[0] + perp_dx * offset_distance), 
                                        int(current_center_pixel[1] + perp_dy * offset_distance))
                        end_point = (int(start_point[0] + udx * arrow_length), 
                                        int(start_point[1] + udy * arrow_length))

                        cv2.arrowedLine(frame, start_point, end_point, (255, 255, 255), 2, tipLength=0.4)

                    dist_x_display = convert_to_display_units(dist_x_mm)
                    dist_y_display = convert_to_display_units(dist_y_mm)
                    total_dist_display = convert_to_display_units(total_dist_mm)
                    unit_label = get_unit_label()
                    
                    info_text = f"X: {dist_x_display:.2f}{unit_label} | Y: {dist_y_display:.2f}{unit_label}"
                    total_dist_text = f"Total Disp: {total_dist_display:.2f}{unit_label}"
                    cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 0), 2)
                    cv2.putText(frame, total_dist_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 0), 2)

                    # Determine status based on stillness and centering
                    is_still = is_qr_still(current_time)
                    is_centered = is_qr_centered(total_dist_mm)
                    
                    status_text = get_status_text(is_still, is_centered)
                    status_color = get_status_color(is_still, is_centered)
                    
                    cv2.putText(frame, f"Status: {status_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                    
                    time_since_movement = current_time - last_movement_time if last_movement_time else 0
                    debug_text = f"Still Time: {time_since_movement:.1f}s"
                    cv2.putText(frame, debug_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Display status messages
        if not is_center_set:
            if found_center_qr:
                cv2.putText(frame, "Center QR detected - setting reference...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 180), 2)
            else:
                cv2.putText(frame, "Show CENTER QR code to set reference", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Center Set. Tracking...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)
            cv2.circle(frame, center_pixel, 7, (0, 255, 0), -1)
            cv2.drawMarker(frame, center_pixel, (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

        # Display QR code detection status
        qr_status = []
        if found_center_qr:
            qr_status.append("CENTER")
        if found_target_qr:
            qr_status.append("TARGET")
        
        if qr_status:
            status_msg = f"QR Codes: {', '.join(qr_status)}"
            cv2.putText(frame, status_msg, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

        cv2.imshow('MRI Head Motion Tracker - Dual QR', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("'q' pressed. Exiting.")
            break
        elif key == ord('r'):
            center_pixel = None
            is_center_set = False
            position_history.clear()
            last_movement_time = None
            print("Center reset. Show CENTER QR code to set new reference.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()