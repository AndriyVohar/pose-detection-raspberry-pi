from ultralytics import YOLO
import cv2
from base_estimator import BasePoseEstimator

class YoloPoseEstimator(BasePoseEstimator):
    def __init__(self, yolo: str = 'yolo11n-pose.pt', camera_id=0, width=640, height=480) -> None:
        """
        Initialize the YOLO11 Pose Estimator.
        """
        super().__init__(camera_id, width, height)
        self.model = YOLO(yolo)
        self.classNames = ["person"]

    def _process_frame_impl(self, frame):
        """
        Process a frame with YOLO pose estimation.
        """
        # Perform pose estimation
        results = self.model(frame, stream=True)

        for r in results:
            keypoints = r.keypoints
            boxes = r.boxes

            for box, kpts in zip(boxes, keypoints):
                # Draw the bounding box for person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Draw keypoints (pose landmarks)
                for kp in kpts.xy[0]:  # Each keypoint [x, y]
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    
        return frame


# Run the Pose Estimator
if __name__ == "__main__":
    pose_estimator = YoloPoseEstimator()
    pose_estimator.run()
