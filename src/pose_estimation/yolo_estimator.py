from ultralytics import YOLO
import cv2

class YoloPoseEstimator:
    def __init__(self, yolo: str = 'yolo11n-pose.pt') -> None:
        """
        Initialize the YOLO11 Pose Estimator.
        """
        self.model = YOLO(yolo)
        self.classNames = ["person"]

    def run(self) -> None:
        """
        Start the pose estimation process using the webcam.
        """
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)
        cap.set(4, 480)

        if not cap.isOpened():
            print("Failed to open the camera.")
            return

        print("Press 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Failed to read a frame.")
                break

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

            # Display the frame
            cv2.imshow("YOLO Pose Estimation", frame)

            # Quit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


# Run the Pose Estimator
if __name__ == "__main__":
    pose_estimator = YoloPoseEstimator()
    pose_estimator.run()
