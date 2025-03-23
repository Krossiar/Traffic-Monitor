from flask import Flask, render_template, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

# Load YOLOv3
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class names
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define vehicle classes
vehicle_classes = ["motorbike", "car", "bus", "truck"]

# Load video file (change this to 0 for webcam)
VIDEO_SOURCE = "traffic.mp4"
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Define lanes (horizontally split)
def define_lanes(frame_width, frame_height, num_lanes=4):
    lane_height = frame_height // num_lanes
    return [(i * lane_height, (i + 1) * lane_height) for i in range(num_lanes)]

def classify_vehicle(class_id):
    return "2-wheeler" if classes[class_id] == "motorbike" else "4-wheeler"

def detect_objects():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width, _ = frame.shape
        lanes = define_lanes(frame_width, frame_height, num_lanes=4)

        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        lane_counts = [0] * 4  # 4 lanes

        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if classes[class_id] in vehicle_classes and confidence > 0.5:
                    center_x, center_y, w, h = (detection[:4] * [frame_width, frame_height, frame_width, frame_height]).astype(int)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                confidence = confidences[i]
                vehicle_type = classify_vehicle(class_ids[i])

                # Determine lane
                center_y = y + h // 2
                for idx, (lane_start, lane_end) in enumerate(lanes):
                    if lane_start <= center_y < lane_end:
                        lane_counts[idx] += 1 if vehicle_type == "2-wheeler" else 2
                        break

                # Draw bounding box
                color = (0, 255, 0) if vehicle_type == "4-wheeler" else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{vehicle_type} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw lane lines
        for lane_start, lane_end in lanes:
            cv2.line(frame, (0, lane_start), (frame_width, lane_start), (255, 255, 255), 2)

        # Determine lane with highest traffic
        max_lane = lane_counts.index(max(lane_counts)) + 1 if max(lane_counts) > 0 else 1
        cv2.putText(frame, f"Green Light: Lane {max_lane}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display lane densities
        for idx, count in enumerate(lane_counts):
            cv2.putText(frame, f"Lane {idx+1}: {count}", (50, 100 + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)



