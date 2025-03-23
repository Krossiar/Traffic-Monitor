import cv2
import numpy as np
import time

# Load YOLOv3
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class names
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define vehicle classes
vehicle_classes = ["motorbike", "car", "bus", "truck"]

# Initialize video capture (use 0 for webcam or provide video file path)
cap = cv2.VideoCapture("abc.mp4")  # Replace with 0 for webcam

# Define a scaling factor
scaling_factor = 0.5  # Adjust based on your display preferences

# Define lanes (for simplicity, dividing the frame horizontally)
def define_lanes(frame_width, frame_height, num_lanes=4):
    lane_height = frame_height // num_lanes
    lanes = []
    for i in range(num_lanes):
        lanes.append((i * lane_height, (i + 1) * lane_height))
    return lanes

# Initialize lane counts
lane_counts = [0] * 4  # Assuming 4 lanes

# Define function to classify vehicle type
def classify_vehicle(class_id, confidence):
    if classes[class_id] == "motorbike":
        return "2-wheeler"
    elif classes[class_id] in ["car", "bus", "truck"]:
        return "4-wheeler"
    else:
        return "other"

# Optional: Define perspective transform (adjust src_points based on your camera)
apply_perspective = False  # Set to True if you want to apply perspective transform

if apply_perspective:
    # Example source points (adjust these to match your camera view)
    src_points = np.float32([[100, 200], [1200, 200], [1200, 800], [100, 800]])
    # Destination points for a top-down view
    dst_points = np.float32([[0, 0], [1280, 0], [1280, 720], [0, 720]])
    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Create a resizable window
cv2.namedWindow("Traffic Management", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Traffic Management", 1280, 720)  # Adjust as needed

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: Apply perspective transform
    if apply_perspective:
        frame = cv2.warpPerspective(frame, matrix, (1280, 720))

    # Resize the frame
    frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    frame_height, frame_width, _ = frame.shape
    lanes = define_lanes(frame_width, frame_height, num_lanes=4)

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outs = net.forward(output_layers)
    end = time.time()

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Iterate over each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if classes[class_id] in vehicle_classes and confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                w = int(detection[2] * frame_width)
                h = int(detection[3] * frame_height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Max Suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Reset lane counts
    lane_counts = [0] * 4

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            vehicle_type = classify_vehicle(class_ids[i], confidence)

            # Determine which lane the vehicle is in based on its center y-coordinate
            center_y = y + h // 2
            for idx, (lane_start, lane_end) in enumerate(lanes):
                if lane_start <= center_y < lane_end:
                    if vehicle_type == "2-wheeler":
                        lane_counts[idx] += 1  # 1 point
                    elif vehicle_type == "4-wheeler":
                        lane_counts[idx] += 2  # 2 points for higher priority
                    break

            # Draw bounding box and label
            color = (0, 255, 0) if vehicle_type == "4-wheeler" else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{vehicle_type} {confidence:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display lane lines
    for idx, (lane_start, lane_end) in enumerate(lanes):
        cv2.line(frame, (0, lane_start), (frame_width, lane_start), (255, 255, 255), 2)

    # Determine lane densities
    densities = lane_counts.copy()
    max_density = max(densities) if densities else 0
    max_lane = densities.index(max_density) + 1 if densities else 1  # 1-based indexing

    # Simulate traffic light control
    # For simplicity, we'll just display which lane has the green light
    traffic_light_text = f"Green Light: Lane {max_lane}"
    cv2.putText(frame, traffic_light_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display densities
    for idx, count in enumerate(densities):
        cv2.putText(frame, f"Lane {idx+1}: {count}",
                    (50, 100 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Traffic Management", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()



