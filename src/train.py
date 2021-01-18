from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector, GpuWaitResetCollector)

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.serial.sampler import SerialSampler

from envs.ThorEnv import ThorEnv, ThorTrajInfo
from src.agent import ThorCNNAgent, ThorParameterizedCNNAgent
from src.algorithms.ppo import PPO_Custom
from src.collectors import CpuResetCollector_Custom, GpuResetCollector_Custom
from src.cnn_architectures import *

from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.samplers.async_.gpu_sampler import AsyncGpuSampler
from rlpyt.runners.async_rl import AsyncRl, AsyncRlEval
from rlpyt.runners.sync_rl import SyncRl


def build_and_train(config):
  if config['sampler'] == 0:
    SamplerCls = SerialSampler
  elif config['sampler'] == 1:
    SamplerCls = GpuSampler
  elif config['sampler'] == 2:
    SamplerCls = AsyncGpuSampler

  if SamplerCls == GpuSampler:
    Collector = GpuResetCollector_Custom if config['mid_batch_reset'] else GpuWaitResetCollector
    print(f"To satisfy mid_batch_reset == {config['mid_batch_reset']}, using {Collector}.")
  else:
    Collector = CpuResetCollector_Custom

  # affinity = make_affinity(
  #     run_slot=0,
  #     n_cpu_core=8,  # Use 16 cores across all experiments.
  #     n_gpu=2,  # Use 8 gpus across all experiments.
  #     gpu_per_run=1,
  #     sample_gpu_per_run=1,
  #     async_sample=True,
  #     optim_sample_share_gpu=False,
  #     # hyperthread_offset=24,  # If machine has 24 cores.
  #     n_socket=1,  # Presume CPU socket affinity to lower/upper half GPUs.
  #     # gpu_per_run=2,  # How many GPUs to parallelize one run across.
  #     # cpu_per_run=1,
  # )
  affinity = make_affinity(
    run_slot=0,
    n_cpu_core=16,  # Use 16 cores across all experiments.
    n_gpu=4,  # Use 8 gpus across all experiments.
    hyperthread_offset=24,  # If machine has 24 cores.
    n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
    gpu_per_run=2,  # How many GPUs to parallelize one run across.
    # cpu_per_run=1,
  )

  sampler = SamplerCls(
      EnvCls=ThorEnv,
      TrajInfoCls=ThorTrajInfo,
      env_kwargs={'config': config},  # Learn on individual frames.
      CollectorCls=Collector,
      batch_T=32,  # Longer sampling/optimization horizon for recurrence.
      batch_B=config['num_envs'],  # 16 parallel environments.
      max_decorrelation_steps=400,
      # eval_env_kwargs={'config': config},
      # eval_n_envs=2,
      # eval_max_steps=int(10e3),
      # eval_max_trajectories=4,
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
      n_steps=5e6,
      log_interval_steps=1e3,
      affinity=affinity,
  )
  with logger_context(config['log_dir'], config['run_id'], config['name'], config, use_summary_writer=True, snapshot_mode='gap'):
      runner.train()


if __name__ == "__main__":
  import argparse
  from envs.config import default_config
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--cuda-idx', help='gpu to use', type=int, default=None)
  parser.add_argument('--run_id', help='run identifier (logging)', type=int, default=0)
  parser.add_argument('--mid-batch-reset', help='whether environment resets during itr', type=bool, default=True)
  parser.add_argument('--n-parallel', help='number of sampler workers', type=int, default=2)
  parser.add_argument('--log-dir', help='folder to store experiment', type=str, default="test")
  parser.add_argument('--remove-prefix', help='parameter for logging context', type=bool, default=False)
  parser.add_argument('--name', help='experiment name', type=str, default="debug")
  parser.add_argument('--num-envs', help='number of training envs', type=int, default=1)
  parser.add_argument('--sampler', help='gpu or serial sampler', type=int, default=0)


  # Add thor environment configs
  args = default_config(parser)
  build_and_train(args)