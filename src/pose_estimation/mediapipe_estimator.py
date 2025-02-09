import cv2
import mediapipe as mp
import time

class MediaPipePoseDetector:
    def __init__(self) -> None:
        """
        Initialize the MediaPipe Pose detector.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Use dynamic mode for videos
            model_complexity=0,  # Use a simpler model
            enable_segmentation=False,  # Disable segmentation if not needed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )  # Pose detector
        self.mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

    def process_frame(self, frame: cv2.Mat):
        """
        Process a single frame to detect poses.
        :param frame: The input frame (BGR format).
        :return: Pose detection results.
        """
        # Convert the frame to RGB as required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)  # Pose detection results

    def draw_landmarks(self, frame: cv2.Mat, results) -> None:
        """
        Draw pose landmarks on the frame.
        :param frame: The input frame.
        :param results: The pose detection results.
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS, None)

    def run(self) -> None:
        """
        Start the pose detection process using the webcam.
        """
        cap = cv2.VideoCapture(0)  # Open webcam

        if not cap.isOpened():
            print("Failed to open the camera.")
            return

        print("Press 'q' to quit.")

        prev_time = 0
        target_fps = 30  # Limit to 30 FPS
        frame_interval = 1.0 / target_fps

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Failed to read a frame.")
                break

            # Limit frame rate
            current_time = time.time()
            if current_time - prev_time < frame_interval:
                continue
            prev_time = current_time

            # Process the frame for pose detection
            results = self.process_frame(frame)

            # Draw pose landmarks on the frame
            if results:
                self.draw_landmarks(frame, results)

            # Display the frame
            cv2.imshow("MediaPipe Pose Detection", frame)

            # Quit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()


# Run the detector
if __name__ == "__main__":
    detector = MediaPipePoseDetector()
    detector.run()