import tkinter as tk
from tkinter import filedialog
import os
import subprocess

def select_video():
    
    video_path = filedialog.askopenfilename(title="Select a video", filetypes=[("Video files", "*.mp4")])

    
    if video_path:
        
        video_label.config(text="Selected video: {}".format(os.path.basename(video_path)))

      
        subprocess.call(["python", "final.py", video_path])


root = tk.Tk()
root.title("Cecurity Camera")


root.config(bg="white")

video_label = tk.Label(root, text="No video selected , if you want to stop video press 'q'", bg="white")

video_label.pack()


select_video_button = tk.Button(root, text="Select video", command=select_video, bg="white")
select_video_button.pack()

root.geometry("400x150")


root.mainloop()
