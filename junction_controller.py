
import time
import cv2
from ultralytics import YOLO
from datetime import datetime

VIDEO_PATHS = [
    "videos/lane1.mp4",
    "videos/lane2.mp4",
    "videos/lane3.mp4",
    "videos/lane4.mp4",
]

# Density thresholds 
T1 = 3    # Less than 3 cars = Low
T2 = 11   # 3-10 cars = Medium, 11+ cars = High

# Signal timing
TOTAL_CYCLE_TIME = 120  # 2 minutes total cycle
MIN_GREEN = 10          # Minimum green time
MAX_GREEN = 60          # Maximum green time

# Other settings
SAMPLE_TIME = 5         # When 5 seconds left, check other lanes
FPS = 20                # Frame rate

# Vehicle types to count (from COCO dataset)
VEHICLE_CLASSES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

# === SIMPLE DENSITY CLASSIFIER ===
class SimpleDensityClassifier:
    """Simple traffic density classifier"""
    
    def __init__(self, low_threshold=3, high_threshold=11):
        self.T1 = low_threshold      # Low to Medium
        self.T2 = high_threshold     # Medium to High
    
    def get_density(self, vehicle_count):
        """
        Returns density level and label
        1 = Low (0-2 cars)
        2 = Medium (3-10 cars) 
        3 = High (11+ cars)
        """
        if vehicle_count < self.T1:
            return 1, "Low"
        elif vehicle_count < self.T2:
            return 2, "Medium"
        else:
            return 3, "High"

# === SIMPLE TIMING CALCULATOR ===
class SimpleTimingCalculator:
    """Calculates green times based on density"""
    
    def __init__(self, total_cycle=120, min_time=10, max_time=60):
        self.total_cycle = total_cycle
        self.min_time = min_time
        self.max_time = max_time
    
    def calculate_times(self, density_levels):
        """
        Calculate green times: (density/total_density) * total_cycle
        Ensures minimum and maximum times
        """
        num_lanes = len(density_levels)
        
        # Sum of all density levels
        total_density = sum(density_levels)
        
        # If no traffic, give equal time
        if total_density == 0:
            equal_time = max(self.min_time, self.total_cycle / num_lanes)
            times = [min(equal_time, self.max_time) for _ in range(num_lanes)]
            return times
        
        # Calculate proportional times
        green_times = []
        for density in density_levels:
            time = (density / total_density) * self.total_cycle
            # Apply min and max limits
            if time < self.min_time:
                time = self.min_time
            if time > self.max_time:
                time = self.max_time
            green_times.append(round(time, 1))
        
        return green_times


def open_videos(video_paths):
    """Open all video files"""
    caps = []
    for path in video_paths:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            caps.append(cap)
        else:
            print(f"Warning: Cannot open {path}")
            caps.append(None)
    return caps

def get_next_frame(cap):
    """Get next frame from video, restart if ended"""
    if cap is None:
        return None
    
    ret, frame = cap.read()
    if not ret:  
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
        ret, frame = cap.read()
    
    return frame if ret else None

def count_vehicles(frame, model):
    """Count vehicles in frame using YOLO"""
    if frame is None:
        return 0, frame
    
    results = model(frame, verbose=False)
    
    count = 0
    annotated_frame = frame.copy()
    
    try:
        # Get detection results
        boxes = results[0].boxes
        if boxes is not None:
            # Get class IDs (what objects were detected)
            class_ids = boxes.cls.cpu().numpy().astype(int)
            box_coords = boxes.xyxy.cpu().numpy()
            
            for i, class_id in enumerate(class_ids):
                if class_id in VEHICLE_CLASSES:  # It's a vehicle
                    count += 1
                    
                    x1, y1, x2, y2 = map(int, box_coords[i])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    except:
        pass  
    
    return count, annotated_frame

def show_lane_info(frame, lane_num, status, time_left, vehicles, density):
    """Add information text to video frame"""
    if frame is None:
        return frame
    
    display_frame = frame.copy()
    height, width = display_frame.shape[:2]
    
    overlay = display_frame.copy()
    if status == "GREEN":
        color = (0, 200, 0)  # Green
    else:
        color = (0, 0, 200)  # Red
    cv2.rectangle(overlay, (0, 0), (width, 70), color, -1)
    
    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
    
    cv2.putText(display_frame, f"LANE {lane_num}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(display_frame, f"STATUS: {status}", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(display_frame, f"VEHICLES: {vehicles}", (width-200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"DENSITY: {density}", (width-200, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # If green, show timer
    if status == "GREEN" and time_left > 0:
        cv2.putText(display_frame, f"TIME LEFT: {time_left:.1f}s", 
                    (width//2 - 100, height-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return display_frame

def print_simple_dashboard(counts, densities, green_times, current_lane, time_left, cycle):
    print("\n" + "=" * 60)
    print(f"TRAFFIC CONTROL SYSTEM - CYCLE {cycle}")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Total Cycle Time: {TOTAL_CYCLE_TIME}s")
    print("-" * 60)
    
    for i in range(len(counts)):
        lane_name = f"Lane {i+1}"
        
        if i == current_lane:
            status = "GREEN"
            remaining = f"{time_left:.1f}s"
        else:
            status = "RED"
            remaining = "---"
        
        print(f"{lane_name}:")
        print(f"  Status: {status}")
        print(f"  Time left: {remaining}")
        print(f"  Vehicles: {counts[i]}")
        print(f"  Density: {densities[i]}")
        print(f"  Green time: {green_times[i]}s")
        print("-" * 60)
    

    total_vehicles = sum(counts)
    total_green = sum(green_times)
    efficiency = (total_green / TOTAL_CYCLE_TIME) * 100
    
    print(f"TOTAL VEHICLES: {total_vehicles}")
    print(f"GREEN TIME USED: {total_green:.1f}s")
    print(f"EFFICIENCY: {efficiency:.1f}%")
    print("=" * 60)

# === MAIN FUNCTION ===
def main():
    print("=" * 60)
    print("SIMPLE TRAFFIC LIGHT CONTROLLER")
    print("=" * 60)
    print("Density thresholds: Low (<3), Medium (3-10), High (11+)")
    print("=" * 60)
    
    # Load AI model
    print("\nLoading YOLO model for vehicle detection...")
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully!")
    
    # Initialize our tools
    density_checker = SimpleDensityClassifier(low_threshold=T1, high_threshold=T2)
    timing_calculator = SimpleTimingCalculator(
        total_cycle=TOTAL_CYCLE_TIME,
        min_time=MIN_GREEN,
        max_time=MAX_GREEN
    )
    
    # Open video files
    print("\nOpening video files...")
    caps = open_videos(VIDEO_PATHS)
    num_lanes = len(caps)
    print(f"Found {num_lanes} lanes")
    
    # Create display windows
    print("Creating display windows...")
    for i in range(num_lanes):
        window_name = f"Lane {i+1}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        row = i // 2 
        col = i % 2   
        cv2.moveWindow(window_name, col * 650, row * 500)
    
    # Get initial vehicle counts
    print("\nGetting initial vehicle counts...")
    vehicle_counts = [0] * num_lanes
    
    for i, cap in enumerate(caps):
        if cap is not None:
            frame = get_next_frame(cap)
            if frame is not None:
                count, _ = count_vehicles(frame, model)
                vehicle_counts[i] = count
    
    print(f"Initial counts: {vehicle_counts}")
    
    # Calculate initial densities and green times
    density_levels = []
    density_labels = []
    
    for count in vehicle_counts:
        level, label = density_checker.get_density(count)
        density_levels.append(level)
        density_labels.append(label)
    
    print(f"Density levels: {density_labels}")
    
    green_times = timing_calculator.calculate_times(density_levels)
    print(f"Initial green times: {green_times}")
    
    
    for i in range(num_lanes):
        frame = get_next_frame(caps[i])
        if frame is not None:
            count = vehicle_counts[i]
            label = density_labels[i]
            status = "GREEN" if i == 0 else "RED"

            init_frame = show_lane_info(
                frame,
                i + 1,
                status,
                green_times[0] if i == 0 else 0,
                count,
                label
            )
            cv2.imshow(f"Lane {i + 1}", init_frame)

    cv2.waitKey(1)

    
    # Main control loop
    current_lane = 0  # Start with lane 1
    cycle_number = 1
    running = True
    
    print("\n" + "=" * 60)
    print("STARTING TRAFFIC CONTROL")
    print("Press 'q' on any window to quit")
    print("=" * 60)
    
    try:
        while running:
            current_green_time = green_times[current_lane]
            
            # Show dashboard
            print_simple_dashboard(
                vehicle_counts, density_labels, green_times,
                current_lane, current_green_time, cycle_number
            )
            
            print(f"\n>>> Lane {current_lane+1} is now GREEN for {current_green_time} seconds")
            print(f"    Vehicles waiting: {vehicle_counts[current_lane]}")
            print(f"    Traffic density: {density_labels[current_lane]}")
            
            # Green light phase
            start_time = time.time()
            checked_other_lanes = False
            
            while time.time() - start_time < current_green_time:
                time_elapsed = time.time() - start_time
                time_left = current_green_time - time_elapsed
                
                frame = get_next_frame(caps[current_lane])
                if frame is not None:
                  
                    count, detected_frame = count_vehicles(frame, model)
                    vehicle_counts[current_lane] = count
                    
                    level, label = density_checker.get_density(count)
                    density_levels[current_lane] = level
                    density_labels[current_lane] = label
                    
                    display_frame = show_lane_info(
                        detected_frame, 
                        current_lane + 1, 
                        "GREEN", 
                        time_left,
                        count,
                        label
                    )
                    cv2.imshow(f"Lane {current_lane + 1}", display_frame)
                
                # Check other lanes when 5 seconds left
                if time_left <= SAMPLE_TIME and not checked_other_lanes:
                    print(f"\n    Checking other lanes ({time_left:.1f}s left)...")
                    
                    for i in range(num_lanes):
                        if i == current_lane:
                            continue  
                        
                        frame = get_next_frame(caps[i])
                        if frame is not None:
                            count, _ = count_vehicles(frame, model)
                            vehicle_counts[i] = count
                            
                           
                            level, label = density_checker.get_density(count)
                            density_levels[i] = level
                            density_labels[i] = label
                            
                            
                            other_frame = show_lane_info(
                                frame, i + 1, "RED", 0, count, label
                            )
                            cv2.imshow(f"Lane {i + 1}", other_frame)
                    
                    checked_other_lanes = True
                    print(f"    Updated counts: {vehicle_counts}")
                
                # Check if user pressed 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n>>> User pressed 'q' - stopping...")
                    running = False
                    break
                
                # Small delay to control speed
                time.sleep(0.05)
            
            
            print(f"\n>>> Lane {current_lane+1} GREEN phase ended")
            
            
            for i in range(num_lanes):
                frame = get_next_frame(caps[i])
                if frame is not None:
                    count = vehicle_counts[i]
                    label = density_labels[i]
                    status_frame = show_lane_info(frame, i + 1, "RED", 0, count, label)
                    cv2.imshow(f"Lane {i + 1}", status_frame)
            
            # If we completed a full cycle (all lanes had green), recalculate times
            if current_lane == num_lanes - 1:
                print(f"\n>>> Completed Cycle {cycle_number}")
                print("    Recalculating green times for next cycle...")
                
                
                green_times = timing_calculator.calculate_times(density_levels)
                print(f"    New green times: {green_times}")
                
                cycle_number += 1
            
            # Move to next lane
            current_lane += 1
            if current_lane >= num_lanes:
                current_lane = 0  
            
            print("\n>>> All-red pause for 1 second...")
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n>>> Stopped by user (Ctrl+C)")
    
    finally:
    
        print("\n" + "=" * 60)
        print("CLEANING UP...")
        
        
        for cap in caps:
            if cap is not None:
                cap.release()
        
       
        cv2.destroyAllWindows()
        
        # Final report
        print("\nFINAL REPORT:")
        print("-" * 60)
        print(f"Total cycles completed: {cycle_number - 1}")
        print(f"Final vehicle counts: {vehicle_counts}")
        print(f"Final traffic densities: {density_labels}")
        print(f"Final green times: {green_times}")
        
        if cycle_number > 1:
            avg_green = sum(green_times) / len(green_times)
            print(f"Average green time: {avg_green:.1f}s")
        
        print("=" * 60)
        print("Program ended successfully!")
        print("=" * 60)

# Run the program
if __name__ == "__main__":
    main()




