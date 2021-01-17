# from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
# from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector, GpuWaitResetCollector)

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.serial.sampler import SerialSampler
# from rlpyt.samplers.serial.collectors import (GpuResetCollector, GpuWaitResetCollector)

from envs.ThorEnv import ThorEnv, ThorTrajInfo
from src.agent import ThorCNNAgent, ThorParameterizedCNNAgent
from src.algorithms.ppo import PPO_Custom
from src.collectors.cpu_collector import CpuResetCollector_Custom
from src.cnn_architectures import *

from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context

def build_and_train(config):
  affinity = make_affinity(
      run_slot=0,
      n_cpu_core=24,  # Use 16 cores across all experiments.
      n_gpu=4,  # Use 8 gpus across all experiments.
      # hyperthread_offset=24,  # If machine has 24 cores.
      # n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
      # gpu_per_run=2,  # How many GPUs to parallelize one run across.
      # cpu_per_run=1,
  )
  # Collector = GpuResetCollector if config['mid_batch_reset'] else GpuWaitResetCollector
  # print(f"To satisfy mid_batch_reset == {config['mid_batch_reset']}, using {Collector}.")

  sampler = SerialSampler(
      EnvCls=ThorEnv,
      TrajInfoCls=ThorTrajInfo,
      env_kwargs={'config': config},  # Learn on individual frames.
      CollectorCls=CpuResetCollector_Custom,
      batch_T=8,  # Longer sampling/optimization horizon for recurrence.
      batch_B=1,  # 16 parallel environments.
      max_decorrelation_steps=400,
  )
  optim_kwargs = dict()
  algo = PPO_Custom(
    optim_kwargs=optim_kwargs
  )  # Run with defaults.
  agent = ThorParameterizedCNNAgent(model_kwargs=small)
  runner = MinibatchRl(
      algo=algo,
      agent=agent,
      sampler=sampler,
      n_steps=50e6,
      log_interval_steps=1e5,
      affinity=affinity,
  )
  with logger_context(config['log_dir'], config['run_id'], config['name'], config, use_summary_writer=True):
      runner.train()


if __name__ == "__main__":
  import argparse
  from envs.config import default_config
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--cuda-idx', help='gpu to use', type=int, default=None)
  parser.add_argument('--run_id', help='run identifier (logging)', type=int, default=0)
  parser.add_argument('--mid-batch-reset', help='whether environment resets during itr', type=bool, default=False)
  parser.add_argument('--n-parallel', help='number of sampler workers', type=int, default=2)
  parser.add_argument('--log-dir', help='folder to store experiment', type=str, default="test")
  parser.add_argument('--remove-prefix', help='parameter for logging context', type=bool, default=False)
  parser.add_argument('--name', help='experiment name', type=str, default="debug")

  # Add thor environment configs
  args = default_config(parser)
  build_and_train(args)