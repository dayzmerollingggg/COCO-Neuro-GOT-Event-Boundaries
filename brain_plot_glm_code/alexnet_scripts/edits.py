def analyze_audio_modified(audio_path):
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

    
def compute_motion_energy(video_file):
    # Copied as is from Rebecca's script for reference
    cap = cv2.VideoCapture(video_file)
    prev_gray = None
    motion_energy = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            motion_energy.append(np.mean(flow**2))

        prev_gray = gray

    cap.release()
    return np.mean(motion_energy) if motion_energy else 0
