import cv2
import mediapipe as mp

class BlazePoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
        self.mp_drawing = mp.solutions.drawing_utils

    def run(self):
        cap = cv2.VideoCapture(0)  # Open webcam
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Press 'q' to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with BlazePose
            results = self.pose.process(rgb_frame)

            # Draw pose landmarks on the frame
            if results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )

            # Display the frame
            cv2.imshow("BlazePose", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the detector
if __name__ == "__main__":
    detector = BlazePoseEstimator()
    detector.run()