import argparse
from mediapipe_pose import MediaPipePoseDetector
from yolo_pose import YoloPoseEstimator


def main():
    # Create an argument parser for selecting the method
    parser = argparse.ArgumentParser(description="Select a pose detection method")
    parser.add_argument(
        "-m", "--method",
        choices=["mediapipe", "yolo"],
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
    else:
        print("Unsupported method.")
        return

    # Run the pose detection
    detector.run()

if __name__ == "__main__":
    main()
