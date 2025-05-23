import cv2
from base_estimator import BasePoseEstimator

class OpenPosePoseEstimator(BasePoseEstimator):
    def __init__(self, camera_id=0, width=640, height=480):
        """
        Initialize the OpenPose estimator.
        """
        super().__init__(camera_id, width, height)
        
        # Download the model configuration file from the following URL and place it in the 'models' directory:
        # https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/models/pose/coco/pose_deploy_linevec.prototxt
        self.protoFile = "../../models/pose_deploy_linevec.prototxt"

        # Download the model weights file from the following URL and place it in the 'models' directory:
        # https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe-models/blob/master/caffe_models/openpose/caffe_model/pose_iter_440000.caffemodel
        self.weightsFile = "../../models/pose_iter_440000.caffemodel"

        self.nPoints = 18
        self.POSE_PAIRS = [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
            [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
            [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
        ]
        self.net = cv2.dnn.readNetFromCaffe(self.protoFile, self.weightsFile)
        # Use CUDA if available, otherwise fall back to CPU
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except:
            print("CUDA not available, using CPU for OpenPose.")

    def _process_frame_impl(self, frame):
        """
        Process a frame with OpenPose pose estimation.
        
        Args:
            frame: Input frame to process
            
        Returns:
            The processed frame with pose visualization
        """
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inWidth = 368
        inHeight = 368
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inpBlob)
        output = self.net.forward()

        H = output.shape[2]
        W = output.shape[3]

        points = []
        for i in range(self.nPoints):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > 0.1:
                points.append((int(x), int(y)))
            else:
                points.append(None)

        for pair in self.POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
                cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        return frame


if __name__ == "__main__":
    estimator = OpenPosePoseEstimator()
    estimator.run()
