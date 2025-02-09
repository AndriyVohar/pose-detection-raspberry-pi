# Pose detection on Raspberry Pi


## Prerequisites

Ensure you have the following installed:  
- Python (version 3.x or higher)

## Installation

Follow these steps to set up the project:

### 1. Clone the repository
```bash
git clone https://github.com/AndriyVohar/pose-detection-raspberry-pi.git
cd pose-detection-raspberry-pi
```

### 2. Set up a virtual environment
Create and activate a virtual environment:  

- **On Linux/macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- **On Windows:**
  ```bash
  python -m venv .venv
  .venv/Scripts/activate
  ```

### 3. Install dependencies
Install required Python packages:  
```bash
pip install -r requirements.txt
```

## Running the Project

To run the pose detection project, follow these steps:

1. **Run the program**:

   You need to specify which pose detection method you want to use by passing the `-m` argument. For example:

   - To run with **MediaPipe** pose detection:
     ```bash
     python3 src/main.py -m mediapipe
     ```
     
   - To run with **YOLO** pose detection (if implemented):
     ```bash
     python3 src/main.py -m yolo
     ```
     
   - To run with **TensorFlow Single** pose detection:
     ```bash
     python3 src/main.py -m tensorflow_single
     ```
     
   - To run with **TensorFlow Multi** pose detection:
     ```bash
     python3 src/main.py -m tensorflow_multi
     ```
     
   - To run with **OpenPose** pose detection:
     ```bash
     python3 src/main.py -m openpose
     ```

2. **Interact with the program**:

   - The application will open a webcam feed and begin performing pose detection using the selected method.
   - Press `q` to quit the application.

That's it! You can switch between different pose detection models by changing the `-m` argument when running the program.


## Todo
- [X] OpenPose: An open-source library for real-time multi-person keypoint detection.
- [X] MediaPipe: A cross-platform framework for building multimodal applied machine learning pipelines, including pose estimation.
- [X] YOLO (You Only Look Once): A real-time object detection system that can be adapted for pose estimation.
- [X] TensorFlow: An open-source machine learning framework that includes models for pose estimation.
- [ ] PyTorch: Another popular machine learning framework with various pose estimation models.
- [ ] Detectron2: A Facebook AI Research library that provides state-of-the-art detection and segmentation algorithms.
- [ ] AlphaPose: An accurate multi-person pose estimator.
- [ ] HRNet: High-Resolution Network for accurate human pose estimation.
- [ ] DeepLabCut: A toolbox for markerless pose estimation of animals.
- [ ] PoseNet: A vision model that can estimate the pose of a person in real-time.

## Contributing
This project does not accept contributions.