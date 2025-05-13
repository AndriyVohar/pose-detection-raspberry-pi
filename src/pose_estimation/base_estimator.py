import cv2
from abc import ABC, abstractmethod

class BasePoseEstimator(ABC):
    """
    Base class for pose estimation implementations.
    """
    def __init__(self, input_source=0, width=640, height=480):
        """
        Args:
            input_source: Camera ID or video file path
            width: Frame width
            height: Frame height
        """
        self.input_source = input_source
        self.width = width
        self.height = height

    def setup_input(self):
        cap = cv2.VideoCapture(self.input_source)

        # Only set dimensions for camera, not for video files
        if isinstance(self.input_source, int):
            cap.set(3, self.width)
            cap.set(4, self.height)

        if not cap.isOpened():
            print(f"Failed to open input source: {self.input_source}")
            return None

        return cap
    
    def process_frame(self, frame):
        """
        Process a single frame for pose estimation.
        
        Args:
            frame: Input frame to process
            
        Returns:
            The processed frame with pose visualization
        """
        # This method should be implemented by subclasses
        return self._process_frame_impl(frame)
    
    @abstractmethod
    def _process_frame_impl(self, frame):
        """
        Implementation of frame processing specific to each pose estimator.
        
        Args:
            frame: Input frame to process
            
        Returns:
            The processed frame with pose visualization
        """
        pass
        
    def run(self):
        """
        Start the pose estimation process using the camera.
        """
        cap = self.setup_input()
        if not cap:
            return
            
        print("Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read a frame.")
                break
                
            # Process the frame using the specific implementation
            processed_frame = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow(f"{self.__class__.__name__}", processed_frame)
            
            # Quit when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
