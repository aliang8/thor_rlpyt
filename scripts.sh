# Test environment
python3 -m ipdb -c continue envs/ThorEnv.py --tasks turnon_microwave --InitialRandomLocation 1

# Serial sampler
python3 -m ipdb -c continue src/train.py --tasks turnon_microwave --InitialRandomLocation 1 --sampler 0

# GPU sampler
python3 -m ipdb -c continue src/train.py --tasks turnon_microwave --InitialRandomLocation 1 --num-envs 16 --sampler 1 --name full --mid-batch-reset True