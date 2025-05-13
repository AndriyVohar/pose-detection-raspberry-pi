import argparse
from pose_estimation.mediapipe_estimator import MediaPipePoseEstimator
from pose_estimation.yolo_estimator import YoloPoseEstimator
from pose_estimation.tensorflow_multi_estimator import TensorFlowMultiPoseEstimator
from pose_estimation.tensorflow_single_estimator import TensorFlowSinglePoseEstimator
from pose_estimation.openpose_estimator import OpenPosePoseEstimator
from pose_estimation.blazepose_estimator import BlazePoseEstimator
from pose_estimation.detectron2_estimator import Detectron2PoseEstimator

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
            "blazepose",
            "detectron2",
        ],
        default="mediapipe",
        help="Select the method: 'mediapipe', 'yolo', etc."
    )
    args = parser.parse_args()

    # Initialize the selected pose detection method
    if args.method == "mediapipe":
        print("Starting MediaPipe...")
        detector = MediaPipePoseEstimator()
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
    elif args.method == "blazepose":
        print("Starting BlazePose...")
        detector = BlazePoseEstimator()
    elif args.method == "detectron2":
        print("Starting Detectron2...")
        detector = Detectron2PoseEstimator()
    else:
        print("Unsupported method.")
        return

    # Run the pose detection
    detector.run()

if __name__ == "__main__":
    main()
