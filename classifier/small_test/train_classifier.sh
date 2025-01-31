#!/bin/bash

# If needed, source the correct environment on the cluster
# Example: (change accordingly)

source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate cogtorch
#conda activate neural_network_sur_gpu

export CUDA_DIR=/usr/local/cuda-11.2
export CUDA_PATH=/usr/local/cuda-11.2
export LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH
export PATH=$PATH:/usr/local/cuda-11.2/bin
#CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
#export CUDNN_PATH=/home/lucy.thomas/.conda/envs/neural_network_sur_gpu/lib/python3.9/site-packages/nvidia/cudnn/lib
export CUDNN_PATH=/home/lucy.thomas/.conda/envs/cogtorch/lib/python3.12
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TF_CPP_MIN_LOG_LEVEL=2

#echo "LD_LIBRARY_PATH:"
#echo $LD_LIBRARY_PATH

#export KERAS_BACKEND="tensorflow"
#export CUDA_VISIBLE_DEVICES="2"

# The commmand structur isfunction is:
# ABSOLUTE_PATH_TO_PYTHON ABSOLUTE_PATH_TO_SCRIPT
# In our case this means pointing to the python installed in the
# environment (do not change this) and then the path to your python script
# that you wish to run (change this)


# Main:
# exectuable script
# Change accordingly
/home/lucy.thomas/.conda/envs/cogtorch/bin/python /home/lucy.thomas/projects/cog/cogwheel-machine/classifier/train_classifier.py
