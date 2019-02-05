#!/usr/bin/env bash

# Ron the short-list GPU queue
#SBATCH -p slurm_sbel_cmg
#SBATCH --account=cmg --qos=cmg_owner

## Request one CPU core from the scheduler
#SBATCH -c 1

## Request a GPU from the scheduler, we don't care what kind
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH -t 4-1:00 # time (D-HH:MM)

## Create a unique output file for the job
#SBATCH -o cuda_Training-%j.log

## Load custimized CUDA
module load usermods
module load user/cuda
#module load cuda/8.0
# activate virtual environment
source activate resnet 

# install tensorflow and other libraries for machine learning
# All software installnation should be outside the scripts
#conda install --name resnet tensorflow-gpu
#conda install --name resnet keras-gpu
pip install tensorflow-gpu
pip install keras

# /srv/home/shenmr/anaconda3/envs/resnet/bin/pip git+https://www.github.com/keras-team/keras-contrib.git

conda install --name resnet matplotlib 
conda install --name resnet -c anaconda scikit-learn  
conda install --name resnet numpy 
conda install --name resnet scipy 
conda install --name resnet pillow
conda install --name resnet scikit-image
#conda install --name resnet scikit-image
#conda install --name resnet keras-contrib

# run the training scripts
python Focus152.py 
