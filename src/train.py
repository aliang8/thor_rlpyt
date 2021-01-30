from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector, GpuWaitResetCollector)

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.serial.sampler import SerialSampler

from envs.ThorEnv import ThorEnv, ThorEnvFlatObjectCategories, ThorTrajInfo
from src.agent import (ThorCNNAgent, 
                       ThorParameterizedCNNAgent, 
                       ThorParameterizedCNNLSTMAgent,
                       ThorObjectSelectionR2D1Agent)
from src.algorithms.ppo import PPO_Custom
from src.algorithms.r2d1 import R2D1_Custom
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
    # hyperthread_offset=24,  # If machine has 24 cores.
    # n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
    # gpu_per_run=1,  # How many GPUs to parallelize one run across.
    # cpu_per_run=1,
  )
  # affinity = dict(cuda_idx=args['cuda_idx'])

  if config['env'] == 0:
    EnvCls = ThorEnv
  elif config['env'] == 1:
    EnvCls = ThorEnvFlatObjectCategories

  sampler = SamplerCls(
      EnvCls=EnvCls,
      TrajInfoCls=ThorTrajInfo,
      env_kwargs={'config': config},  # Learn on individual frames.
      CollectorCls=Collector,
      batch_T=64,  # Longer sampling/optimization horizon for recurrence.
      batch_B=config['num_envs'],  # 16 parallel environments.
      max_decorrelation_steps=0,
      # eval_env_kwargs={'config': config},
      # eval_n_envs=2,
      # eval_max_steps=int(10e3),
      # eval_max_trajectories=4,
  )

  # algo = PPO_Custom(**algo_kwargs) 
  # agent = ThorParameterizedCNNLSTMAgent(model_kwargs=small)
  if config['algo'] == 'ppo':
    algo_kwargs = dict(
      discount=0.99,
      learning_rate=0.001,
      value_loss_coeff=1.,
      entropy_loss_coeff=0.01,
      # OptimCls=torch.optim.Adam,
      optim_kwargs=None,
      clip_grad_norm=1.,
      initial_optim_state_dict=None,
      gae_lambda=1,
      minibatches=1,
      epochs=4,
      ratio_clip=0.1,
      linear_lr_schedule=True,
      normalize_advantage=False,
    )
    algo = PPO_Custom(**algo_kwargs)
    agent = ThorParameterizedCNNLSTMAgent(model_kwargs=small)
  else:
    algo_kwargs = dict(
      prioritized_replay=False,
      input_priorities=False
    )
    algo = R2D1_Custom(**algo_kwargs)

  runner = MinibatchRl(
      algo=algo,
      agent=agent,
      sampler=sampler,
      n_steps=5e6,
      log_interval_steps=1e4,
      affinity=affinity,
  )
  with logger_context(config['log_dir'], config['run_id'], config['name'], config, use_summary_writer=True, snapshot_mode='gap'):
      runner.train()


if __name__ == "__main__":
  import argparse
  from envs.config import default_config
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--cuda-idx', help='gpu to use', type=int, default=0)
  parser.add_argument('--run_id', help='run identifier (logging)', type=int, default=0)
  parser.add_argument('--mid-batch-reset', help='whether environment resets during itr', type=bool, default=True)
  parser.add_argument('--n-parallel', help='number of sampler workers', type=int, default=2)
  parser.add_argument('--log-dir', help='folder to store experiment', type=str, default="test")
  parser.add_argument('--remove-prefix', help='parameter for logging context', type=bool, default=False)
  parser.add_argument('--name', help='experiment name', type=str, default="debug")
  parser.add_argument('--num-envs', help='number of training envs', type=int, default=1)
  parser.add_argument('--sampler', help='gpu or serial sampler', type=int, default=0)
  parser.add_argument('--env', help='which thor env to use', type=int, default=0)
  parser.add_argument('--algo', help='which algo to use', type=str, default="ppo")


  # Add thor environment configs
  args = default_config(parser)
  build_and_train(args)