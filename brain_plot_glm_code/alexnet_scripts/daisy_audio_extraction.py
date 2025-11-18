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
    """Analyzes the audio file to get average amplitude and pitch."""
    try:
        y, sr = librosa.load(audio_path, sr=None)  # Use original sample rate
        if len(y) == 0:
            print("Warning: Audio signal is empty!")
            return None, None

        # Calculate average amplitude
        amplitude = np.abs(y)
        average_amplitude = np.mean(amplitude)

        # Calculate average pitch (using librosa's pitch tracking)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C7'), sr=sr)
        
        # Remove NaNs from f0 and filter out non-pitched regions
        f0_non_zero = f0[np.isfinite(f0)]  # Consider only finite (non-NaN) pitch values
        average_pitch = np.mean(f0_non_zero) if f0_non_zero.size > 0 else 0.0

        return average_amplitude, average_pitch
    except Exception as e:
        print(f"Error analyzing audio {audio_path}: {e}")
        return None, None


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
    directory = 'C:\Users\Admin\Documents\CoCo-Lab\got_dataset\data\clips_Daisy\clips_Daisy\Game of Thrones'
    # directory = os.path.join(os.getcwd(), "movie_clips", "movie_clips")  # Adjust path as needed
    output_file = "audio_output.json"

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}. Please place your .mov files inside.")
    else:
        result = process_videos(directory)
        save_to_json(result, output_file)

        print(f"Results saved to {output_file}")