import numpy as np
from rlpyt.utils.logging import logger
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.utils.buffer import (torchify_buffer, numpify_buffer, buffer_from_example, buffer_method)
from rlpyt.agents.base import AgentInputs

class CpuResetCollector_Custom(CpuResetCollector):
  def start_envs(self, max_decorrelation_steps=0):
    """Calls ``reset()`` on every environment instance, then steps each
    one through a random number of random actions, and returns the
    resulting agent_inputs buffer (`observation`, `prev_action`,
    `prev_reward`)."""
    traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
    observations = list()
    for env in self.envs:
        observations.append(env.reset())
    observation = buffer_from_example(observations[0], len(self.envs))
    for b, obs in enumerate(observations):
        observation[b] = obs  # numpy array or namedarraytuple
    act = env.action_space.null_value().base_action
    point = env.action_space.null_value().pointer
    null_value = np.concatenate((np.expand_dims(act,0), point))
    prev_action = np.stack([null_value for env in self.envs])
    prev_reward = np.zeros(len(self.envs), dtype="float32")
    if self.rank == 0:
        logger.log("Sampler decorrelating envs, max steps: "
            f"{max_decorrelation_steps}")
    if max_decorrelation_steps != 0:
        for b, env in enumerate(self.envs):
            n_steps = 1 + int(np.random.rand() * max_decorrelation_steps)
            for _ in range(n_steps):
                a = env.action_space.sample()
                act, point = a.base_action, a.pointer
                a = np.concatenate((np.expand_dims(act,0), point))
                o, r, d, info = env.step(a)
                traj_infos[b].step(o, a, r, d, None, info)
                if getattr(info, "traj_done", d):
                    o = env.reset()
                    traj_infos[b] = self.TrajInfoCls()
                if d:
                    a = env.action_space.null_value()
                    r = 0
            observation[b] = o
            prev_action[b] = a
            prev_reward[b] = r
    # For action-server samplers.
    if hasattr(self, "step_buffer_np") and self.step_buffer_np is not None:
        self.step_buffer_np.observation[:] = observation
        self.step_buffer_np.action[:] = prev_action
        self.step_buffer_np.reward[:] = prev_reward
    return AgentInputs(observation, prev_action, prev_reward), traj_infos