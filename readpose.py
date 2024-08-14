import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle 


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)


image_path = '/Users/jeongjanghun/Desktop/FacialRecognitionProject apr 13/test data/hammer data/image copy 3.png'
image = cv2.imread(image_path)


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgb)


if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
   
    landmark_points = [(lm.x, lm.y) for lm in landmarks]

   
    angles = {
        'Right_elbow': calculate_angle(landmark_points[12], landmark_points[14], landmark_points[16]),
        'Left_elbow': calculate_angle(landmark_points[11], landmark_points[13], landmark_points[15]),
        'Right_shoulder': calculate_angle(landmark_points[24], landmark_points[12], landmark_points[14]),
        'Left_shoulder': calculate_angle(landmark_points[23], landmark_points[11], landmark_points[13]),
        'Right_hip': calculate_angle(landmark_points[24], landmark_points[26], landmark_points[28]),
        'Left_hip': calculate_angle(landmark_points[23], landmark_points[25], landmark_points[27]),
        'Right_knee': calculate_angle(landmark_points[24], landmark_points[26], landmark_points[28]),
        'Left_knee': calculate_angle(landmark_points[23], landmark_points[25], landmark_points[27]),
        'Right_ankle': calculate_angle(landmark_points[26], landmark_points[28], landmark_points[32]),
        'Left_ankle': calculate_angle(landmark_points[25], landmark_points[27], landmark_points[31]),
    }

    
    for joint, angle in angles.items():
        print(f'{joint}: {angle:.2f} degrees')
        
    
    mp_drawing = mp.solutions.drawing_utils
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    
    cv2.imshow('Pose Landmarks and Angles', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


