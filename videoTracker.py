import cv2
from copy import copy
import time
from tracker import TrackingManager
from image_processing import prepare_image, filter_contours, draw_tracks
from utils import save_yolo_annotations

class VideoTracker:
    """A video processing pipeline that tracks objects across frames."""
    
    def __init__(self, input_video_path, output_video_path):
        """Initialize the video tracking pipeline.
        
        Args:
            input_video_path: Path to the input video file
            output_video_path: Path where the processed video will be saved
        """
        # Video file paths
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path

        # Initialize components
        self.tracking_manager = TrackingManager()
        
        # Parameters
        self.area_threshold1 = 1500
        self.area_threshold2 = 5000
        self.cannyEdgeTreshold1 = 40
        self.cannyEdgeTreshold2 = 200

        # Timing info
        self.frame_times = []

        # Initialize video capture and writer
        self.cap = cv2.VideoCapture(self.input_video_path)
        self.out = self.set_video_parameters()

    def set_video_parameters(self):
        """Set up video writer with same properties as input."""
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            exit()
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps,
                              (frame_width, frame_height), isColor=False)
        return out

    def process(self, max_frames, measure_time=False):
        """Process the video and track objects.
        
        Args:
            max_frames: Maximum number of frames to process
            measure_time: If True, measure processing time for each frame
        """
        i = 0
        while i < max_frames:
            ret, frame = self.cap.read()
            if not ret:
                break

            # start timer for this frame
            t0 = time.perf_counter() if measure_time else None

            # Image processing
            originalFrame = copy(frame)
            processed_frame = prepare_image(
                frame, 
                canny_threshold1=self.cannyEdgeTreshold1,
                canny_threshold2=self.cannyEdgeTreshold2
            )
            
            # Contour detection and filtering
            contours, _ = cv2.findContours(
                processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = filter_contours(
                contours, 
                area_min=self.area_threshold1,
                area_max=self.area_threshold2
            )
            
            # Draw contours on processed frame
            cv2.drawContours(processed_frame, contours, -1,
                           (122, 255, 122), thickness=cv2.FILLED)

            # Update tracking
            tracked_objects = self.tracking_manager.process_frame(originalFrame, contours)
            
            # Visualize results
            result_img = draw_tracks(originalFrame, tracked_objects)

            # Save YOLO annotations
            annotations_dir = 'E:/yolo_annotations'
            save_yolo_annotations(i, tracked_objects, frame.shape, annotations_dir)

            # Optional: save or display result
            self.out.write(result_img)

            # stop timer and record
            if measure_time:
                t1 = time.perf_counter()
                dur = (t1 - t0) if t0 is not None else 0.0
                self.frame_times.append(dur)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            i += 1

        # Release resources
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

