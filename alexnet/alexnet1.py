import os
import cv2
import torch
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torchvision.transforms as transforms
import torchvision.models as models

# ---------- CONFIG ----------
VIDEO_DIR = '.'  # <<< SET THIS
SAVE_FEATURES_PATH = 'visual_features1.npy'
SAVE_PCA_PATH = 'visual_features_pca1.npy'

# ---------- Preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_frame(frame):
    """Preprocess a single frame for AlexNet."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    return tensor

# ---------- Load AlexNet and Register Hook ----------
alexnet = models.alexnet(pretrained=True)
alexnet.eval()

features_list = []

def hook_fn(module, input, output):
    """Hook to capture conv5 output (before ReLU & MaxPool)."""
    features_list.append(output.clone().detach())

# conv5 output is at features[10]
alexnet.features[10].register_forward_hook(hook_fn)

def extract_features_from_frame(frame):
    """Run a frame through AlexNet and return conv5 features (flattened)."""
    features_list.clear()
    input_tensor = preprocess_frame(frame)
    with torch.no_grad():
        _ = alexnet(input_tensor)
    if features_list:
        return features_list[0].flatten().cpu().numpy()
    else:
        return None

# ---------- Extract Features from Videos ----------
def extract_first_frame(video_path):
    """Extract the first frame from a video file."""
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    return frame if success else None

def extract_features_from_video_dir(video_dir):
    """Extract features from the first frame of each video."""
    all_features = []
    video_names = []
    fnames = sorted(os.listdir(video_dir))    
    for fname in fnames:
        if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            full_path = os.path.join(video_dir, fname)
            frame = extract_first_frame(full_path)
            if frame is not None:
                features = extract_features_from_frame(frame)
                if features is not None:
                    all_features.append(features)
                    video_names.append(fname)
                    print(f"Processed {fname}")
                else:
                    print(f"Warning: No features for {fname}")
            else:
                print(f"Warning: Couldn't read frame from {fname}")
    
    return np.array(all_features), video_names

# ---------- PCA Dimensionality Reduction ----------
def reduce_features_with_pca(features, variance_ratio=0.7):
    """Apply PCA to reduce features to the number of components that explain given variance."""
    pca = PCA(n_components=variance_ratio)
    reduced = pca.fit_transform(features)
    print(f"PCA reduced to {reduced.shape[1]} components (70% variance)")
    return reduced, pca

# ---------- MAIN ----------
if __name__ == "__main__":
    print("Extracting features...")
    all_features, video_names = extract_features_from_video_dir(VIDEO_DIR)

    print("Running PCA...")
    reduced_features, pca_model = reduce_features_with_pca(all_features)

    print(f"Saving features to {SAVE_FEATURES_PATH} and {SAVE_PCA_PATH}")
    np.save(SAVE_FEATURES_PATH, all_features)
    np.save(SAVE_PCA_PATH, reduced_features)




