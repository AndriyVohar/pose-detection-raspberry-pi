import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from base_estimator import BasePoseEstimator

class TensorFlowSinglePoseEstimator(BasePoseEstimator):
    """
    A class to perform pose estimation using TensorFlow and TensorFlow Hub.
    """

    def __init__(self, camera_id=0, width=640, height=480, confidence_threshold=0.1):
        """
        Initialize the pose estimator by loading the model.
        
        Args:
            camera_id: Camera device ID
            width: Camera frame width
            height: Camera frame height
            confidence_threshold: Minimum confidence score for keypoints
        """
        super().__init__(camera_id, width, height)
        self.model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
        self.movenet = self.model.signatures['serving_default']
        self.confidence_threshold = confidence_threshold

    def _process_frame_impl(self, frame):
        """
        Process a frame with TensorFlow Single-Pose estimation.
        
        Args:
            frame: Input frame to process
            
        Returns:
            The processed frame with pose visualization
        """
        # Resize image to the expected input shape
        img = frame.copy()
        img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
        input_img = tf.cast(img, dtype=tf.int32)

        # Detection section
        results = self.movenet(input_img)
        keypoints_with_scores = results['output_0'].numpy().reshape((1, 17, 3))

        # Render keypoints
        self._draw_keypoints_and_edges(frame, keypoints_with_scores[0], self.confidence_threshold)
        
        return frame
        
    def _draw_keypoints_and_edges(self, frame, keypoints, confidence_threshold):
        """
        Draw keypoints and edges on the frame if confidence score is above threshold.
        """
        y, x, _ = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        
        # Draw keypoints
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)
        
        # Draw edges
        for edge, color in EDGES.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]

            if (c1 > confidence_threshold) & (c2 > confidence_threshold):
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

if __name__ == "__main__":
    estimator = TensorFlowSinglePoseEstimator()
    estimator.run()
