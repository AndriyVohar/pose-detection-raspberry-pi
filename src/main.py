import argparse
from pose_estimation.mediapipe_estimator import MediaPipePoseDetector
from pose_estimation.yolo_estimator import YoloPoseEstimator
from pose_estimation.tensorflow_multi_estimator import TensorFlowMultiPoseEstimator
from pose_estimation.tensorflow_single_estimator import TensorFlowSinglePoseEstimator
from pose_estimation.openpose_estimator import OpenPosePoseEstimator


def main():
    # Create an argument parser for selecting the method
    parser = argparse.ArgumentParser(description="Select a pose detection method")
    parser.add_argument(
        "-m", "--method",
        choices=[
            "mediapipe",
            "yolo",
            "tensorflow_multi",
            "tensorflow_single",
            "openpose",
        ],
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
    elif args.method == "openpose":
        print("Starting OpenPose Model...")
        detector = OpenPosePoseEstimator()
    else:
        print("Unsupported method.")
        return

    # Run the pose detection
    detector.run()

if __name__ == "__main__":
    main()
