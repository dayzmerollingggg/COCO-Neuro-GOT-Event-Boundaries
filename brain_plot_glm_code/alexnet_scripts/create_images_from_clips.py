# Script for creating images from the first frame of each video clip
# Images, after a bit of preprocessing, will be fed into AlexNet
# This script is for manually examining outputs, as well as preparing for
# resizing. (Technically all of this could be automated into a pipeline starting
# from the video clip to feeding acquired frame into AlexNet)

import os
import cv2


PROJ_DIR = '/mnt/labdata/got_project/ian'


def get_first_frame(video_path, output_path):
    video_capture = cv2.VideoCapture(video_path)
    _, image = video_capture.read()
    cv2.imwrite(output_path, image)

    video_capture.release()


def clips_Rebecca():
    video_dir = os.path.join(PROJ_DIR, 'data/clips/clips_Rebecca_renamed')
    output_dir = os.path.join(PROJ_DIR, 'data/images/images_Rebecca')
    os.makedirs(output_dir, exist_ok=True)
    
    videos = sorted(os.listdir(video_dir))
    
    for v in videos:
        out_fn = v.replace('.mp4', '.jpg')
        video_path = os.path.join(video_dir, v)
        output_path = os.path.join(output_dir, out_fn)
        get_first_frame(video_path, output_path)


def clips_Daisy():
    video_dir = os.path.join(PROJ_DIR, 'data/clips/clips_Daisy')
    output_dir = os.path.join(PROJ_DIR, 'data/images/images_Daisy')
    os.makedirs(output_dir, exist_ok=True)
    
    videos = sorted(os.listdir(video_dir))
    # print(videos)
    for v in videos:
        out_fn = v.replace('.mov', '.jpg')
        video_path = os.path.join(video_dir, v)
        output_path = os.path.join(output_dir, out_fn)
        get_first_frame(video_path, output_path)


if __name__ == "__main__":
    clips_Rebecca()
    clips_Daisy()
    