import subprocess
import cv2
import gui


video_path = gui.select_video()


if video_path:
    
    cap = cv2.VideoCapture(video_path)

    
    subprocess.call(["python", "final.py", video_path])
