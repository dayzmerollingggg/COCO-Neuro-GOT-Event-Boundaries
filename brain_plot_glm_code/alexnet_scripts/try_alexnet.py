# Test script for setting up AlexNet
# First pass for an AlexNet pipeline

import os
import torch
import numpy as np
from torchvision import models
from torchvision import transforms
from PIL import Image


PROJ_DIR = '/mnt/labdata/got_project/ian'


def pipeline(alexnet, image_path, output_path=None):
    # Preprocess images
    transform = transforms.Compose([
        transforms.CenterCrop(224), # Crop from 398x224 → 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path)
    img_tensor = transform(img)

    with torch.no_grad():
        # Add batch dimension: (1, 3, 224, 224)
        img_batch = img_tensor.unsqueeze(0)

        # # Get activations from the 5th layer (conv5) pre-ReLU and pre-pooling
        features = alexnet.features[:11](img_batch)  # Output: (1, 256, 13, 13)
        
        # # Forward through first 10 layers (up to conv4)
        # x = alexnet.features[:11](img_batch)

        # # Run only layer 10 (conv5, pre-ReLU, pre-pool)
        # features = alexnet.features[11](x)


    flat_features = features.view(-1).numpy()
    if output_path is not None:
        np.save(output_path, flat_features)
    return flat_features


def try_Rebecca():
    alexnet = models.alexnet(pretrained=True).eval()
    
    image_dir = os.path.join(PROJ_DIR, 'data/images/images_resized_Rebecca')
    output_dir = os.path.join(PROJ_DIR, 'data/alexnet')
    os.makedirs(output_dir, exist_ok=True)

    res = []
    for i in sorted(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, i)
        # out_fn = i.replace('.jpg', '.npy')
        # output_path = os.path.join(output_dir, out_fn)
        flat_features = pipeline(alexnet, image_path)#, output_path)
    
        res.append(flat_features)
    full_out_fn = os.path.join(output_dir, 'Rebecca_all_clips.npy')
    np.save(full_out_fn, np.array(res))


def try_Daisy():
    alexnet = models.alexnet(pretrained=True).eval()
    
    image_dir = os.path.join(PROJ_DIR, 'data/images/images_resized_Daisy')
    output_dir = os.path.join(PROJ_DIR, 'data/alexnet')
    os.makedirs(output_dir, exist_ok=True)

    res = []
    for i in sorted(os.listdir(image_dir)):
        print(i)
        image_path = os.path.join(image_dir, i)
        # out_fn = i.replace('.jpg', '.npy')
        # output_path = os.path.join(output_dir, out_fn)
        flat_features = pipeline(alexnet, image_path)#, output_path)

        res.append(flat_features)
    full_out_fn = os.path.join(output_dir, 'Daisy_all_clips.npy')
    np.save(full_out_fn, np.array(res))


def _test_alexnet():
    # Preprocess images
    transform = transforms.Compose([
        transforms.CenterCrop(224), # Crop from 398x224 → 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Test preprocessing
    image_dir = os.path.join(PROJ_DIR, 'data/images/images_resized_Daisy')
    test_file = os.listdir(image_dir)[0]
    test_fn = os.path.join(image_dir, test_file)
    test_img = Image.open(test_fn)

    img_tensor = transform(test_img)
    print(img_tensor.shape) # Shape of (3, 224, 224)

    # Try pretrained AlexNet
    alexnet = models.alexnet(pretrained=True).eval()
        
    with torch.no_grad():
        # Add batch dimension: (1, 3, 224, 224)
        img_batch = img_tensor.unsqueeze(0)

        # # Get activations from the 5th layer (conv5) pre-ReLU and pre-pooling
        # features = alexnet.features[:10](img_batch)  # Output: (1, 256, 13, 13)

        # Forward through first 10 layers (up to conv4)
        x = alexnet.features[:10](img_batch)

        # Run only layer 10 (conv5, pre-ReLU, pre-pool)
        features = alexnet.features[10](x)

    flat_features = features.view(-1)  
    print(type(flat_features), flat_features.shape) # Shape of (43,264,)
    numpy_features = flat_features.numpy()
    print(type(numpy_features), numpy_features.shape)


if __name__ == "__main__":
    _test_alexnet()
    try_Rebecca()
    try_Daisy()