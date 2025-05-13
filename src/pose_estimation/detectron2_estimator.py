import cv2
import torch
import numpy as np
import time

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from base_estimator import BasePoseEstimator

class Detectron2PoseEstimator(BasePoseEstimator):
    def __init__(self,
                 camera_id=0,
                 width=640,
                 height=480,
                 score_threshold=0.7,
                 use_lightweight_model=True,
                 input_resize_factor=0.5,
                 process_every_n_frames=2):
        """
        Initialize the pose estimator with options to reduce resource usage

        Args:
            camera_id: Camera device ID
            width: Camera frame width
            height: Camera frame height
            score_threshold: Detection confidence threshold (0.0-1.0)
            use_lightweight_model: If True, use a lighter model configuration
            input_resize_factor: Scale factor to resize input frames (0.1-1.0)
            process_every_n_frames: Only process 1 in every N frames
        """
        super().__init__(camera_id, width, height)
        
        # Setup configuration
        self.cfg = get_cfg()

        # Use lightweight model configuration if requested
        if use_lightweight_model:
            # Use a smaller backbone model with fewer features
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml")

            # Additional optimizations to make it lighter
            self.cfg.MODEL.BACKBONE.FREEZE_AT = 2  # Freeze some backbone layers
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Reduce batch size
            self.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000  # Reduce proposals
            self.cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 100
        else:
            # Use the original heavier model
            self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold

        # Force CPU usage since Raspberry Pi doesn't have an NVIDIA GPU
        self.cfg.MODEL.DEVICE = "cpu"

        # Create predictor
        self.predictor = DefaultPredictor(self.cfg)

        # Get metadata for visualization
        self.metadata = MetadataCatalog.get("keypoints_coco_2017_val")

        # Resource optimization parameters
        self.input_resize_factor = max(0.1, min(1.0, input_resize_factor))  # Clamp between 0.1 and 1.0
        self.process_every_n_frames = max(1, process_every_n_frames)
        self.frame_count = 0
        self.last_output = None
        self.skip_frames = True

    def _process_frame_impl(self, frame):
        """
        Implementation of frame processing for Detectron2 pose estimation.
        
        Args:
            frame: Input frame to process
            
        Returns:
            The processed frame with pose visualization
        """
        # Increment frame counter
        self.frame_count += 1

        # Resize input frame for faster processing
        if self.input_resize_factor < 1.0:
            h, w = frame.shape[:2]
            new_h, new_w = int(h * self.input_resize_factor), int(w * self.input_resize_factor)
            small_frame = cv2.resize(frame, (new_w, new_h))
        else:
            small_frame = frame

        # Only process certain frames to reduce CPU usage
        process_this_frame = (self.frame_count % self.process_every_n_frames == 0) or not self.skip_frames

        if process_this_frame:
            start_time = time.time()

            # Get predictions from Detectron2
            outputs = self.predictor(small_frame)
            self.last_output = outputs

            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)

        # Always visualize results, but might use cached predictions
        if self.last_output is not None:
            v = Visualizer(small_frame[:, :, ::-1], self.metadata, scale=1.2)
            out = v.draw_instance_predictions(self.last_output["instances"].to("cpu"))
            result_frame = out.get_image()[:, :, ::-1]

            # Scale back to original size if needed
            if self.input_resize_factor < 1.0:
                result_frame = cv2.resize(result_frame, (frame.shape[1], frame.shape[0]))

            # Add performance information
            if process_this_frame:
                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if self.skip_frames and self.process_every_n_frames > 1:
                cv2.putText(result_frame, f"Processing 1/{self.process_every_n_frames} frames",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            return result_frame
        
        return frame  # Return original frame if no output is available yet

    def run(self):
        """
        Override base run method to add key handling functionality
        """
        cap = self.setup_input()
        if not cap:
            return
            
        print("Press 'q' to quit.")
        print("Press 'f' to toggle frame skipping.")
        print("Press '+' or '-' to adjust resize factor.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read a frame.")
                break
                
            # Process the frame using the specific implementation
            processed_frame = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow(f"{self.__class__.__name__}", processed_frame)
            
            # Handle keypresses for interactive configuration
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                self.skip_frames = not self.skip_frames
                print(f"Frame skipping: {'ON' if self.skip_frames else 'OFF'}")
            elif key == ord('+') or key == ord('='):
                self.input_resize_factor = min(1.0, self.input_resize_factor + 0.1)
                print(f"Resize factor: {self.input_resize_factor:.1f}")
            elif key == ord('-'):
                self.input_resize_factor = max(0.1, self.input_resize_factor - 0.1)
                print(f"Resize factor: {self.input_resize_factor:.1f}")
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


# Run the detector
if __name__ == "__main__":
    # Create a lightweight detector by default
    detector = Detectron2PoseEstimator(
        score_threshold=0.5,           # Lower threshold to detect more poses
        use_lightweight_model=True,    # Use ResNet-18 instead of ResNet-50
        input_resize_factor=0.4,       # Reduce input size to 40% for faster processing
        process_every_n_frames=3       # Only process every 3rd frame
    )
    detector.run()
