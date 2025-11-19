import numpy as np
import os
import json
import librosa
import librosa.display
import soundfile as sf
import moviepy.editor as mp


def extract_audio(video_path, audio_path):
    """Extracts audio from a video file."""
    try:
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        video.close()
        return True
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False
def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None)  # Use original sample rate
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Calculate average amplitude using RMS <-- better captures signal peaks
    average_amplitude = np.mean(librosa.feature.rms(y=y))

    # Extract the strongest pitch per frame <--less susceptible to higher harmonics dominating the average pitch calculation
    pitch_values = []
    for t in range(pitches.shape[1]):  # Iterate over time frames
        index = magnitudes[:, t].argmax()  # Find max magnitude index
        pitch = pitches[index, t]  # Extract corresponding pitch
        if pitch > 0: # Exclude zero-pitch frames <-- average pitch of a clip should be from perceptible audio
            pitch_values.append(pitch) 
    average_pitch = np.mean(pitch_values)

    return average_amplitude, average_pitch



def process_videos(directory):
    """Processes all .mov files in the given directory."""
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith(".mov"):
            video_path = os.path.join(directory, filename)
            audio_filename = os.path.splitext(filename)[0] + ".wav"
            audio_path = os.path.join(directory, audio_filename)

            if extract_audio(video_path, audio_path):
                avg_amplitude, avg_pitch = analyze_audio(audio_path)
                if avg_amplitude is not None and avg_pitch is not None:
                    results[filename] = {
                        "average_amplitude": float(avg_amplitude),
                        "average_pitch_hz": float(avg_pitch)
                    }
                # Clean up the extracted audio file
                try:
                    os.remove(audio_path)
                except OSError as e:
                    print(f"Error deleting audio file {audio_path}: {e}")
    return results

def save_to_json(data, output_file):
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    directory = os.path.join(os.getcwd(), "movie_clips", "movie_clips")  # Adjust path as needed
    output_file = "audio_output.json"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}. Please place your .mov files inside.")
    else:
        result = process_videos(directory)
        save_to_json(result, output_file)

        print(f"Results saved to {output_file}")