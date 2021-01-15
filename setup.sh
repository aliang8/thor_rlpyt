# Installation steps

# 1. Install rlpyt
cd rlpyt
conda env create -f linux_cuda9.yml
source activate rlpyt
pip install -e .

# Additional packages - make this into a requirements thing
gym
ai2thor
ipdb
tensorboard