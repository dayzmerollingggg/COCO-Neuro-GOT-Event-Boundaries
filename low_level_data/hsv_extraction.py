import cv2
import numpy as np
import os
import json

def get_average_hsv(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None, []
    
    total_hsv = np.zeros(3, dtype=np.float64)
    frame_count = 0
    frame_hsv_values = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_hsv = np.mean(hsv_frame.reshape(-1, 3), axis=0).tolist()
        total_hsv += avg_hsv
        frame_hsv_values.append(avg_hsv)
        frame_count += 1
    
    cap.release()
    
    if frame_count == 0:
        return None, []
    
    return (total_hsv / frame_count).tolist(), frame_hsv_values

def process_videos(directory):
    video_data = {}
    for file in os.listdir(directory):
        if file.endswith(".mov"):
            video_path = os.path.join(directory, file)
            avg_hsv, frame_hsv_values = get_average_hsv(video_path)
            if avg_hsv:
                video_data[file] = {
                    "movie_clip": file,
                    "average_hsv": avg_hsv
                    #"frame_hsv_values": frame_hsv_values
                }
    return video_data

def save_to_json(data, output_file):
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    directory = os.path.join(os.getcwd(), "movie_clips", "movie_clips")  # Change this to your directory
    output_file = "hsv_output.json"  # Name of the output JSON file
    
    result = process_videos(directory)
    save_to_json(result, output_file)
    
    print(f"Results saved to {output_file}")
