import cv2
import os

def video_to_frames(video_path, out_folder):
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    while True:

        ret, frame = cap.read()

        if not ret:
            break  
        
        frame_filename = f"frame{count}.jpg"
        cv2.imwrite(os.path.join(out_folder, frame_filename), frame)
        count += 1

    cap.release()
    print(f"done: {count}")

video_path = '/Users/jeongjanghun/Desktop/FacialRecognitionProject apr 13/test data/Video shows robbery at Wichita Falls convenience store Sunday night (online-video-cutter.com).mp4'
out_folder = '/Users/jeongjanghun/Desktop/FacialRecognitionProject apr 13/data output/output_frames'
video_to_frames(video_path, out_folder)
