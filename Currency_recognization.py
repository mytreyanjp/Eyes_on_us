import cv2
from ultralytics import YOLO
import supervision as sv
import torch
import pyttsx3
import requests
import time


url1 = "https://llamalab.com/automate/cloud/message"
headers = {
    'Content-Type': 'application/json'
}

def send_amount_to_automate(amount):
    data = {
    "secret": "2.8CC7HPIjnrdoYJi-QETJ-bzZO1XpFMiyD1gO1KjLkg4",
    "to": "mytreyan197@gmail.com",
    "device": None,  # Use None for null value in JSON
    "priority": "normal",
    "payload": f"Total amount detected is {amount} Rupees"
    }
    response = requests.post(url1, json=data, headers=headers)

def wait_for_server(url, max_retries=30, wait_time=5):
    """Wait for the server to start."""
    retries = 0
    while retries < max_retries:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            cap.release()
            print("Server is up and running.")
            return True
        else:
            print(f"Server not available. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    print("Failed to connect to the server after multiple retries.")
    return False

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO("best_d.pt").to(device)

    video_url = "http://192.168.1.4:8080/video"
    
    if not wait_for_server(video_url):
        print("Exiting due to server connection issues.")
        return

    cap = cv2.VideoCapture(video_url)
    box_annotator = sv.BoxAnnotator()

    class_values = {
        0: 10,
        1: 100,
        2: 20,
        3: 200,
        4: 2000,
        5: 50,
        6: 500
    }

    engine = pyttsx3.init()
    resize_width = 640
    resize_height = 480

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        result = model(frame)[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        detections = sv.Detections(
            xyxy=boxes,
            confidence=scores,
            class_id=classes
        )

        frame = box_annotator.annotate(scene=frame, detections=detections)

        total_amount = 0

        for box, class_id in zip(boxes, classes):
            class_value = class_values.get(class_id, 0)
            total_amount += class_value

            class_name = str(class_value)
            x1, y1, x2, y2 = box.astype(int)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        resized_frame = cv2.resize(frame, (resize_width, resize_height))

        if total_amount > 0:
            # Send the total amount to Automate
            send_amount_to_automate(total_amount)

            

            # Display the detected frame
            cv2.imshow('Detected Frame', resized_frame)
            cv2.waitKey(0)  # Wait until a key is pressed

            cv2.destroyAllWindows()

            # Wait for 10 seconds before resuming detection
            time.sleep(10)

            # Reset the detection loop
            continue

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
