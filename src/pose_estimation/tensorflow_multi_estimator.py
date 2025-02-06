import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

class TensorFlowMultiPoseEstimator:
    """
    A class to perform pose estimation using TensorFlow and TensorFlow Hub.
    """

    def __init__(self):
        """
        Initialize the pose estimator by loading the model and setting up the video capture.
        """
        self.model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
        self.movenet = self.model.signatures['serving_default']
        self.cap = cv2.VideoCapture(0)

    def run(self):
        """
        Run the pose estimation on video frames captured from the webcam.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # Resize image to a smaller size for faster processing
            img = frame.copy()
            img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 256, 256)
            input_img = tf.cast(img, dtype=tf.int32)

            # Detection section
            results = self.movenet(input_img)
            keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))

            # Render keypoints
            self.loop_through_people(frame, keypoints_with_scores, EDGES, 0.1)

            cv2.imshow('Movenet Multipose', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def loop_through_people(self, frame, keypoints_with_scores, edges, confidence_threshold):
        """
        Loop through detected people and draw keypoints and connections on the frame.
        """
        for person in keypoints_with_scores:
            self.draw_connections(frame, person, edges, confidence_threshold)
            self.draw_keypoints(frame, person, confidence_threshold)

    def draw_keypoints(self, frame, keypoints, confidence_threshold):
        """
        Draw keypoints on the frame if their confidence score is above the threshold.
        """
        y, x, _ = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)

    def draw_connections(self, frame, keypoints, edges, confidence_threshold):
        """
        Draw connections between keypoints on the frame if their confidence scores are above the threshold.
        """
        y, x, _ = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

        for edge, color in edges.items():
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
    estimator = TensorFlowMultiPoseEstimator()
    estimator.run()