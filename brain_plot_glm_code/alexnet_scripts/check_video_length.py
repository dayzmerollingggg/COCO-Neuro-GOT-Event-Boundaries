# Quick script for checking video length, num_frames, and fps
# For both Rebecca and Daisy's clips, video length is ~4 seconds
# Num frames is ~96
# FPS is ~24 

import os
import cv2


PROJ_DIR = '/mnt/labdata/got_project/ian'


def check_video_length(video_path):
    """
    Extracts the first frame from a video and saves it as an image.

    Args:
        video_path (str): The path to the video file.
        output_path (str): The path to save the first frame as an image.
    """
    video_capture = cv2.VideoCapture(video_path)
    
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    length_seconds = frame_count / fps
    print(frame_count, fps, length_seconds)
    
    video_capture.release()

if __name__ == "__main__":
    # video_dir = os.path.join(PROJ_DIR, 'data/clips/clips_Rebecca_renamed')
    video_dir = os.path.join(PROJ_DIR, 'data/clips/clips_Daisy')
    
    videos = sorted(os.listdir(video_dir))
    for v in videos:
        video_path = os.path.join(video_dir, v)
        check_video_length(video_path)
        