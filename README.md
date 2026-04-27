# COSC674: YOLO vs Faster R-CNN Object Detection Demo

This project demonstrates a live comparison between YOLO and Faster R-CNN using webcam-based object detection. The goal is to show how YOLO achieves real-time speed while older region-based detectors such as Faster R-CNN are slower..

### Course Information
**By:** Chenilyn Joy Espineda & Dayana Ferrufino <br>
**Course:** COSC 674-101<br>
**Instructor:** Dr. Appolo Tankeh

<br>

---

## 💻 Tech Stack
* Python
* OpenCV
* PyTorch
* Torchvision
* Ultralytics YOLOv8
* Webcam Object Detection

<br>

---

## 🛠️ Setup Steps

### Step 1: Clone this Repository
Run the following to download the project from GitHub to your local machine:
```bash
git clone https://github.com/chlyn/COSC674_Research_Presentation.git
cd COSC474_Research_Presentation
```

<br>

### Step 2: Verify Python Installation
Make sure Python 3 is installed on your system:
```bash
python3 --version
```
If Python is not installed, download it from: [https://www.python.org/downloads/](https://www.python.org/downloads/)

<br>

### Step 3: Install Python Dependencies
Run the following command to install the required packages:
```bash
python3 -m pip install ultralytics opencv-python torch torchvision
```

<br>

### Step 4: Run Launcher
The launcher allows switching between both demos, YOLO and Faster R-CNN, quickly.
```bash
python3 launcher.py
```

<br>

---

## 📊 Expected Results

### YOLO
* Smooth real-time detection
* Fast response to movement
* Better for live applications such as surveillance, robotics, and self-driving systems

<br>

### Faster R-CNN
* More traditional two-stage detector
* Slower frame updates, shows noticeable lag
* Useful for understanding older object detection pipelines

<br>

---