import os
import json
import cv2
import numpy as np

def calculate_motion_energy(prev_frame, current_frame):
    """Calculates the motion energy between two grayscale frames."""
    if prev_frame is None:
        return 0
    frame_diff = cv2.absdiff(prev_frame, current_frame)
    _, thresholded = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed
    motion_energy = np.sum(thresholded) / thresholded.size
    return motion_energy

def process_video(video_path):
    """Processes a single video file to extract the average motion energy."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None

        prev_frame_gray = None
        total_motion_energy = 0
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_energy = calculate_motion_energy(prev_frame_gray, frame_gray)
            total_motion_energy += motion_energy
            frame_count += 1
            prev_frame_gray = frame_gray

        cap.release()

        if frame_count > 0:
            average_motion_energy = total_motion_energy / frame_count
            #print(f"Processed {os.path.basename(video_path)} - Average Motion Energy: {average_motion_energy:.4f}")
            return average_motion_energy
        else:
            print(f"Warning: No frames found in {os.path.basename(video_path)}")
            return None

    except Exception as e:
        print(f"Error processing {os.path.basename(video_path)}: {e}")
        return None

def process_videos(directory):
    """Processes all MOV video files in the given directory."""
    results = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(".mov"):
            video_path = os.path.join(directory, filename)
            avg_motion_energy = process_video(video_path)
            if avg_motion_energy is not None:
                results[filename] = avg_motion_energy
    return results

def save_to_json(data, output_file):
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    directory = os.path.join(os.getcwd(), "movie_clips", "movie_clips")  # Change this to your directory

    output_file = "motion_energy_output.json"  # Name of the output JSON file

    results = process_videos(directory)
    save_to_json(results, output_file)

    print(f"Results saved to {output_file}")