from pitchBackgroundRemover import BackgroundRemoverConfig
from pitchBackgroundRemover import PitchBackgroundRemover
from videoTracker import VideoTracker

def remove_background_segment(input_video_path):
    """Remove pitch background from a video segment."""

    # Times in seconds
    start_time = 0 * 60 + 0  # 0:00
    end_time = 3 * 60 + 0    # 3:00

    config = BackgroundRemoverConfig(num_samples=10, epsilon_factor=0.005)
    remover = PitchBackgroundRemover(input_video_path, start_time, end_time, config)
    remover.process()

def extract_contour_information():
    """Main entry point for the video tracking application."""
    # Example usage
    # Ez a rész a háttér eltávolítása UTÁN fut, a már feldolgozott videón
    tracker = VideoTracker(
        input_video_path='E:/Videos/no_background_dynamic.mp4',
        output_video_path='E:/Videos/edge_detected.mp4'
    )
    tracker.process(max_frames=1000, measure_time=True)

def main():
    # Add meg itt a bemeneti videó elérési útját
    video_to_process = 'E:\\MainCamera\\test_videos\\raw_video-Scene-246.mp4'
    remove_background_segment(video_to_process)
    #extract_contour_information()

if __name__ == "__main__":
    main()