# Script for resizing images from (1280 x 720) down to (398 x 224)
# Keeping aspect ratio, resizing down to 224 pixel height for AlexNet
# After resizing down to correct input height for Alexnet, 
# torchvision.transforms.CenterCrop(224) should crop a (224 x 224) box in the
# center of the resized image. 

# The alternative to this is to feed non-resized images into AlexNet
# In that case, transforms.CenterCrop(224) will crop a box in the center of the 
# original (1280 x 720) image. Hence, only a small box in the center of each 
# scene would be sampled. This may have an effect on the extracted features.
# See "images" and "images_resized" directories for comparison.

import os
from PIL import Image


PROJ_DIR = '/mnt/labdata/got_project/ian'


def resize_image(image_path, output_path=None, new_height=224):
    image = Image.open(image_path)

    width, height = image.size
    new_width = int(new_height * width / height)

    resized_image = image.resize((new_width, new_height)) # should be 398x224
    
    if output_path is not None:
        resized_image.save(output_path)
        
    return resized_image


def resize_frame(frame, output_path=None, new_height=224):

    width, height = frame.size
    new_width = int(new_height * width / height)

    resized_image = frame.resize((new_width, new_height)) # should be 398x224
    
    if output_path is not None:
        resized_image.save(output_path)
        
    return resized_image


def try_Rebecca():
    image_dir = os.path.join(PROJ_DIR, 'data/images/images_Rebecca')
    output_dir = os.path.join(PROJ_DIR, 'data/images/images_resized_Rebecca')
    os.makedirs(output_dir, exist_ok=True)

    for i in os.listdir(image_dir):
        image_path = os.path.join(image_dir, i)
        output_path = os.path.join(output_dir, i)

        resize_image(image_path, output_path)


def try_Daisy():
    image_dir = os.path.join(PROJ_DIR, 'data/images/images_Daisy')
    output_dir = os.path.join(PROJ_DIR, 'data/images/images_resized_Daisy')
    os.makedirs(output_dir, exist_ok=True)

    for i in os.listdir(image_dir):
        image_path = os.path.join(image_dir, i)
        output_path = os.path.join(output_dir, i)

        resize_image(image_path, output_path)
        

if __name__ == "__main__":
    try_Rebecca()
    try_Daisy()