import cv2
import mediapipe as mp
from base_estimator import BasePoseEstimator

class BlazePoseEstimator(BasePoseEstimator):
    def __init__(self, camera_id=0, width=640, height=480, model_complexity=1):
        super().__init__(camera_id, width, height)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            model_complexity=model_complexity, 
            enable_segmentation=False
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def _process_frame_impl(self, frame):
        """
        Process a frame with BlazePose estimation.
        """
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with BlazePose
        results = self.pose.process(rgb_frame)

        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
            )
            
        return frame

# Run the detector
if __name__ == "__main__":
    detector = BlazePoseEstimator()
    detector.run()
