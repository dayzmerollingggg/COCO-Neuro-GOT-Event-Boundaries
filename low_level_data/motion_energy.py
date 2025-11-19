# import cv2
# import numpy as np
# import os
# import json
# import moten


# def get_motion_energy(video_path):
#     luminance_images = moten.io.video2luminance(video_path, nimages=100)

#     # Create a pyramid of spatio-temporal gabor filters
#     nimages, vdim, hdim = luminance_images.shape
#     pyramid = moten.get_default_pyramid(vhsize=(vdim, hdim), fps=24)

#     # Compute motion energy features
#     moten_features = pyramid.project_stimulus(luminance_images)
#     return moten_features


# def process_videos(directory):
#     video_data = {}
#     for file in os.listdir(directory):
#         if file.endswith(".mov"):
#             video_path = os.path.join(directory, file)
#             motion_energy = get_motion_energy(video_path)
#             if motion_energy is not None:
#                 video_data[file] = {
#                     "frame_motion_energy": motion_energy
#                 }
#     return video_data

# def save_to_json(data, output_file):
#     with open(output_file, "w") as json_file:
#         json.dump(data, json_file, indent=4)

# if __name__ == "__main__":
#     directory = os.path.join(os.getcwd(), "movie_test")
#     #directory = os.path.join(os.getcwd(), "movie_clips", "movie_clips")  # Change this to your directory
#     output_file = "motion_energy_output.json"  # Name of the output JSON file
    
#     result = process_videos(directory)
#     save_to_json(result, output_file)
    
#     print(f"Results saved to {output_file}")
