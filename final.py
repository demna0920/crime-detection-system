import datetime
import threading
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing



last_email_time = None
def send_email_async(subject, body, image_path, receiver_email):
   
    threading.Thread(target=send_email, args=(subject, body, image_path, receiver_email)).start()

def send_email(subject, body, image_path, receiver_email):
    global last_email_time
    receiver_email= "51305130ss@gmail.com"
    sender_email = "s51300@naver.com"
    password = "elfama250"

    current_time = datetime.datetime.now()
    if last_email_time is not None and (current_time - last_email_time).total_seconds() < 30:
        print("Email not sent to avoid spamming. Wait for 30 seconds.")
        return
    last_email_time = current_time

    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg.attach(MIMEText(body, 'plain'))

    with open(image_path, 'rb') as file:
        img = MIMEImage(file.read())
        img.add_header('Content-Disposition', 'attachment; filename="scene.jpg"')
        msg.attach(img)

    server = smtplib.SMTP('smtp.naver.com', 587)
    server.starttls()
    server.login(sender_email, password)
    server.send_message(msg)
    server.quit()
    print("Email sent.")

def capture_and_notify(frame, message, email):
    image_path = "/Users/jeongjanghun/Desktop/FacialRecognitionProject apr 13/robbery capture/scene.jpg"
    cv2.imwrite(image_path, frame)
 
    send_email_async("Alert: Important Event Detected!", message, image_path, email)


# Load YOLO configuration and weights
modelConfiguration = "yolov4-tiny-custom.cfg"
modelWeights = "yolov4-tiny-custom_final.weights"
net = cv2.dnn.readNet(modelWeights, modelConfiguration)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Class labels for YOLO
classes = ["gun", "hammer", "helmet", "knife", "mask", "person", "rifle"]

# Load video

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def detect_gun_aiming_pose(landmarks):
    
    right_elbow_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    )
    left_elbow_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    )
    right_shoulder_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    )
    left_shoulder_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    )
    right_torso_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    )
    left_torso_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    )
   
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        
        

  
    condition1 = ((165 < right_elbow_angle < 185 and 75 < right_shoulder_angle < 95 and 80 < right_torso_angle < 100 and right_wrist.y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) or 
    (165 < left_elbow_angle < 185 and 75 < left_shoulder_angle < 95 and 80 < left_torso_angle < 100 and left_wrist.y < landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y))

    
    condition2 = (right_elbow_angle < 160 and
                left_elbow_angle < 60 and
                right_shoulder_angle < 60 and
                left_shoulder_angle < 60)

    if (right_wrist.y > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y or 
        left_wrist.y > landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y or
        right_wrist.y < landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y or
        left_wrist.y < landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y):
            return False

    return condition1 or condition2

# This function should be called where landmarks are available, for example within the main loop processing video frames.

def is_hand_above_shoulder(landmarks):
    wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    return wrist.y < shoulder.y  # Check if the wrist is above the shoulder

def is_gun_aiming(landmarks):
   
    if not landmarks:
        return False

    
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    shoulder_to_elbow_right = np.array([right_shoulder.x - right_elbow.x, right_shoulder.y - right_elbow.y])
    elbow_to_wrist_right = np.array([right_elbow.x - right_wrist.x, right_elbow.y - right_wrist.y])
    arm_angle_right = np.degrees(np.arccos(np.clip(np.dot(shoulder_to_elbow_right, elbow_to_wrist_right) /
                                             (np.linalg.norm(shoulder_to_elbow_right) * np.linalg.norm(elbow_to_wrist_right)), -1.0, 1.0)))

   
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    shoulder_to_elbow_left = np.array([left_shoulder.x - left_elbow.x, left_shoulder.y - left_elbow.y])
    elbow_to_wrist_left = np.array([left_elbow.x - left_wrist.x, left_elbow.y - left_wrist.y])
    arm_angle_left = np.degrees(np.arccos(np.clip(np.dot(shoulder_to_elbow_left, elbow_to_wrist_left) /
                                             (np.linalg.norm(shoulder_to_elbow_left) * np.linalg.norm(elbow_to_wrist_left)), -1.0, 1.0)))

    
    return arm_angle_right < 30 or arm_angle_left < 30

hammer_up_detected = False
hammer_down_detected = False

def detect_hammering(landmarks, frame):
    global hammer_up_detected, hammer_down_detected  # 글로벌 변수 사용 선언
    
    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1],
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * frame.shape[1],
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * frame.shape[0]]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1],
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]]
    angle = calculate_angle(shoulder, elbow, wrist)

    
    cv2.putText(frame, str(int(angle)), (int(wrist[0]), int(wrist[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if angle > 160 and not hammer_up_detected:
        hammer_up_detected = True
    elif angle < 45 and not hammer_down_detected:
        hammer_down_detected = True

    
    if hammer_up_detected and hammer_down_detected:
        cv2.putText(frame, "Hammer Action Detection Occurrence of dangerous situations", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imwrite('/Users/jeongjanghun/Desktop/FacialRecognitionProject apr 13/robbery capture/hammer capture/hammer_capture.jpg', frame)
        hammer_up_detected = False
        hammer_down_detected = False


    

    
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    
   
   

    # YOLO Detection setup
    blob = cv2.dnn.blobFromImage(frame,0.00392, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    outs = net.forward(output_layers)

    # Post-processing the YOLO detections
    detected_objects = {cls: False for cls in classes} 
    class_ids, confidences, boxes = [], [], []
    detected_objects = {'gun': False, 'hammer': False, 'knife': False, 'person': False, 'mask': False, 'helmet': False, 'rifle': False}


    human_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >0.7:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = center_x - w // 2
                y = center_y - h // 2
                label = classes[class_id]
                
                if label in ["person"]:
                    human_boxes.append((x, y, w, h))
                if label in ["gun", "hammer","knife","helmet", "mask", "rifle"]:
                    detected_objects[label] = True
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255,0), 2)
        cv2.putText(frame, classes[class_ids[i]], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,255,0), 2)

    # MediaPipe Pose Detection within human bounding boxes
    # MediaPipe Pose Detection within human bounding boxes
    for (x, y, w, h) in human_boxes:
        human_frame = frame[y:y+h, x:x+w]
        if human_frame.size ==0:
            continue
        rgb_frame = cv2.cvtColor(human_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            if detected_objects['hammer']:
                detect_hammering(landmarks, frame)
                
                
            
            if detected_objects['gun']:
                if is_hand_above_shoulder(landmarks):
                    gun_aiming_by_angle = is_gun_aiming(landmarks)
                    gun_aiming_by_pose = detect_gun_aiming_pose(landmarks)
                    if gun_aiming_by_angle or gun_aiming_by_pose:
                        cv2.putText(frame, "Gun aiming detected! Occurrence of dangerous situations", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
                        capture_and_notify(frame, "Suspected robbery detected in the video feed.", "51305130ss@gmail.com")
            if ((detected_objects['knife'] and detected_objects['helmet']) or 
                (detected_objects['knife'] and detected_objects['mask']) or
                (detected_objects['gun'] and detected_objects['helmet']) or
                (detected_objects['gun'] and detected_objects['mask']) or
                (detected_objects['hammer'] and detected_objects['mask']) or
                (detected_objects['hammer'] and detected_objects['helmet']) or
                (detected_objects['rifle'] and detected_objects['helmet']) or
                (detected_objects['rifle'] and detected_objects['mask'])):
                cv2.putText(frame, "Suspected robbery", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255),2)
                capture_and_notify(frame, "Suspected robbery detected in the video feed.", "51305130ss@gmail.com")
                
                
                 
             


    cv2.imshow("Combined YOLO and MediaPipe Pose Detection", frame)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
