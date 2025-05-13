import cv2
import mediapipe as mp
import time
from base_estimator import BasePoseEstimator

class MediaPipePoseEstimator(BasePoseEstimator):
    def __init__(self, camera_id=0, width=640, height=480, model_complexity=0) -> None:
        """
        Initialize the MediaPipe Pose detector.
        
        Args:
            camera_id: Camera device ID
            width: Camera frame width
            height: Camera frame height
            model_complexity: Model complexity (0=Lite, 1=Full, 2=Heavy)
        """
        super().__init__(camera_id, width, height)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Use dynamic mode for videos
            model_complexity=model_complexity,  # Use a simpler model
            enable_segmentation=False,  # Disable segmentation if not needed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )  # Pose detector
        self.mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks
        self.prev_time = 0
        self.target_fps = 30  # Limit to 30 FPS
        self.frame_interval = 1.0 / self.target_fps

    def _process_frame_impl(self, frame):
        """
        Implementation of frame processing for MediaPipe pose estimation.
        
        Args:
            frame: Input frame to process
            
        Returns:
            The processed frame with pose visualization
        """
        # Limit frame rate
        current_time = time.time()
        if current_time - self.prev_time < self.frame_interval:
            return frame
        self.prev_time = current_time

        # Convert the frame to RGB as required by MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        # Draw pose landmarks on the frame
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            
        return frame


# Run the detector
if __name__ == "__main__":
    detector = MediaPipePoseEstimator(model_complexity=0)  # Use lite model
    detector.run()
