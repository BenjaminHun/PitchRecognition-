from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg

VIDEO_PATH = 'E:/Downloads/raw_video.mkv'
OUTPUT_VIDEO_PATH = 'E:/Downloads/'

# Detect scenes in the video.
scene_list = detect(VIDEO_PATH, AdaptiveDetector(), show_progress=True)

# Split video into clips. The `arg_override` parameter forces ffmpeg to re-encode
# the video, which allows for precise cuts at scene changes. This is slower
# than the default but avoids including frames from the previous scene.
split_video_ffmpeg(VIDEO_PATH, scene_list, output_dir=OUTPUT_VIDEO_PATH, arg_override='-c:v libx264 -preset medium -crf 22 -c:a aac')