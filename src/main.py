import argparse
from src.pose_estimation.mediapipe_estimator import MediaPipePoseDetector
from src.pose_estimation.yolo_estimator import YoloPoseEstimator
from src.pose_estimation.tensorflow_multi_estimator import TensorFlowMultiPoseEstimator
from src.pose_estimation.tensorflow_single_estimator import TensorFlowSinglePoseEstimator


def main():
    # Create an argument parser for selecting the method
    parser = argparse.ArgumentParser(description="Select a pose detection method")
    parser.add_argument(
        "-m", "--method",
        choices=["mediapipe", "yolo", "tensorflow_multi", "tensorflow_single"],
        default="mediapipe",
        help="Select the method: 'mediapipe' or 'yolo'."
    )
    args = parser.parse_args()

    # Initialize the selected pose detection method
    if args.method == "mediapipe":
        print("Starting MediaPipe...")
        detector = MediaPipePoseDetector()
    elif args.method == "yolo":
        print("Starting YOLO...")
        detector = YoloPoseEstimator()
    elif args.method == "tensorflow_multi":
        print("Starting TensorFlow Multi...")
        detector = TensorFlowMultiPoseEstimator()
    elif args.method == "tensorflow_single":
        print("Starting TensorFlow Single...")
        detector = TensorFlowSinglePoseEstimator()
    else:
        print("Unsupported method.")
        return

    # Run the pose detection
    detector.run()

if __name__ == "__main__":
    main()
