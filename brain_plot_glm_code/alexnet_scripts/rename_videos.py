# Script for renaming Rebecca's clips
# Adds leading zeros to clip numbers
# Unifies sorting and naming conventions for downstream analyses

import os
import shutil

PROJ_DIR = '/mnt/labdata/got_project/ian'


if __name__ == "__main__":
    video_dir = os.path.join(PROJ_DIR, 'data/clips/clips_Rebecca')
    videos = sorted(os.listdir(video_dir))

    output_dir = os.path.join(PROJ_DIR, 'data/clips/clips_Rebecca_renamed')
    os.makedirs(output_dir, exist_ok=True)

    # print(videos)
    for v in videos:
        clip_str, num_end = v.split('_')
        # print(clip_str, num_end)
        num = num_end.replace('.mp4', '')
        new_num = num.zfill(3)
        # print(new_num)
        new_fn = f'clip_{new_num}.mp4'
        print(new_fn)

        old_path = os.path.join(video_dir, v)
        new_path = os.path.join(output_dir, new_fn)
        shutil.copy2(old_path, new_path)