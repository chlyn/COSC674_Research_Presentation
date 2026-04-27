import subprocess
import tkinter as tk

current_process = None

def stop_current():
    global current_process
    if current_process is not None:
        current_process.terminate()
        current_process = None

def run_yolo():
    global current_process
    stop_current()
    current_process = subprocess.Popen(["python3", "yolo.py"])

def run_faster_rcnn():
    global current_process
    stop_current()
    current_process = subprocess.Popen(["python3", "faster_rcnn.py"])

def quit_app():
    stop_current()
    root.destroy()

root = tk.Tk()
root.title("Object Detection Demo Launcher")
root.geometry("350x220")

title = tk.Label(root, text="YOLO vs Faster R-CNN Demo", font=("Arial", 16))
title.pack(pady=15)

yolo_btn = tk.Button(root, text="Run YOLO", command=run_yolo, font=("Arial", 14), width=20)
yolo_btn.pack(pady=8)

frcnn_btn = tk.Button(root, text="Run Faster R-CNN", command=run_faster_rcnn, font=("Arial", 14), width=20)
frcnn_btn.pack(pady=8)

quit_btn = tk.Button(root, text="Quit", command=quit_app, font=("Arial", 14), width=20)
quit_btn.pack(pady=8)

root.protocol("WM_DELETE_WINDOW", quit_app)
root.mainloop()