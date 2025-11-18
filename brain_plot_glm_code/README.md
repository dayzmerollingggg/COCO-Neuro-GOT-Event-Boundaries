### Ian's scripts/work for the GoT project 

#### Setting up the python environment
Assuming that you're using anaconda/miniconda
After conda is set up on your account or personal machine, in a terminal, run

conda create -n got_env -f environment.yml 

(terminal must be in this directory)
This should install the required packages for all scripts (for now, at least).

The environment can be activated with 
conda activate got_env

Update (Oct. 31, 2025):
nilearn and brainplotlib had to be pip installed, not available through conda install

#### AlexNet pipeline (first pass)
Current script pipeline is
- create_images_from_clips.py
- resize_images.py
- try_alexnet.py

/data/alexnet/
should have a outputs from the fifth layer of AlexNet for each clip
Specifically, from each first frame for each clip, after the first frame was 
resized to have height of 224 pixels

Outputs are a flattened array of shape (43264,), from (256, 13, 13)

Correlations between Rebecca and Daisy's first pass outputs are in
alexnet_correlations.csv

#### Extra notes
This project can be handled through VSCode using the .code-workspace file, 
though this is optional