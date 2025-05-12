# Pose Detection on Raspberry Pi

This project implements various pose detection models optimized for Raspberry Pi and similar edge devices.

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

### Running Individual Pose Estimators

To run a specific pose detection model:

```bash
python3 src/main.py -m <model_name>
```

Available models include:

- `mediapipe`: MediaPipe pose detection
- `yolo`: YOLO pose detection
- `tensorflow_single`: TensorFlow single-person pose detection
- `tensorflow_multi`: TensorFlow multi-person pose detection
- `openpose`: OpenPose detection
- `detectron2`: Detectron2 pose detection
- `blazepose`: BlazePose detection

Example:
```bash
python3 src/main.py -m yolo
```

Press `q` to quit when running a model.

### Running Benchmarks

The project includes a benchmarking utility to compare performance across different pose detection models:

```bash
python3 src/benchmark.py --video <path_to_video> --output <output_directory>
```

Optional arguments:
- `--models`: Comma-separated list of models to benchmark (default: all)
- `--frames`: Number of frames to process (default: all)
- `--visualization`: Enable real-time visualization (default: disabled)

Example:
```bash
python3 src/benchmark.py --video data/sample.mp4 --models yolo,blazepose --frames 100 --visualization
```

### Real-time Demonstration

To run a real-time demonstration of all models:

```bash
python3 src/demo.py --video <path_to_video>
```

Optional arguments:
- `--model`: Specific model to demonstrate (default: all models)
- `--duration`: Duration in seconds for each model (default: 30)

Example:
```bash
python3 src/demo.py --video data/sample.mp4 --model yolo --duration 60
```

## Project Architecture

This project is organized with a modular architecture:

- **Base Estimator**: All pose estimators inherit from a common base class (`BasePoseEstimator`) that standardizes the interface
- **Model Implementations**: Each pose detection model is implemented in its own module
- **Benchmark Utilities**: Tools for performance measurement and comparison
- **Visualization**: Real-time display of pose detection results

## Implemented Models

The following pose detection models have been successfully implemented:

- [x] **OpenPose**: An open-source library for real-time multi-person keypoint detection.
- [x] **MediaPipe**: A cross-platform framework for building multimodal applied machine learning pipelines.
- [x] **YOLO (You Only Look Once)**: A real-time object detection system adapted for pose estimation.
- [x] **TensorFlow Single-Pose**: Single-person pose detection using TensorFlow.
- [x] **TensorFlow Multi-Pose**: Multi-person pose detection using TensorFlow.
- [x] **Detectron2**: Facebook AI Research's detection and segmentation library.
- [x] **BlazePose**: A lightweight pose detection model designed for mobile devices.
- [ ] **AlphaPose**: An accurate multi-person pose estimator.
- [ ] **PoseNet**: A vision model for real-time pose estimation.

## Benchmark Results

The benchmark utility generates several outputs:

1. **Performance Metrics**: CSV files with detailed metrics for each model
2. **Comparison Charts**: Visual comparison of models' performance
3. **Processed Videos**: Visualizations of each model's pose detection results

Key metrics include:
- Frames per second (FPS)
- Processing time per frame
- Memory usage
- Accuracy (if ground truth data available)

## Contributing

This project does not accept contributions.
