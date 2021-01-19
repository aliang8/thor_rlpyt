import numpy as np
from rlpyt.utils.logging import logger
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.parallel.gpu.collectors import GpuResetCollector
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
        act = env.action_space.null_value().base
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
                    act, point = a.base, a.pointer
                    a = np.concatenate((np.expand_dims(act,0), point))
                    o, r, d, info = env.step(a)
                    traj_infos[b].step(o, a, r, d, None, info)

                    if getattr(info, "traj_done", d):
                        o = env.reset()
                        traj_infos[b] = self.TrajInfoCls()
                    if d:
                        a = env.action_space.null_value()
                        act, point = a.base, a.pointer
                        a = np.concatenate((np.expand_dims(act,0), point))
                        r = 0
                observation[b] = o
                prev_action[b] = a
                prev_reward[b] = r
        # For action-server samplers.
        if hasattr(self, "step_buffer_np") and self.step_buffer_np is not None:
            self.step_buffer_np.observation[:] = observation
            self.step_buffer_np.action[:] = prev_action
            self.step_buffer_np.reward[:] = prev_reward

        self.maximum_episode_lengths = np.array([0 for i in range(len(self.envs))])

        return AgentInputs(observation, prev_action, prev_reward), traj_infos

    def collect_batch(self, agent_inputs, traj_infos, itr):

        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        self.agent.sample_mode(itr)
        for t in range(self.batch_T):
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d, agent_info[b],
                    env_info)
                self.maximum_episode_lengths += 1
                if getattr(env_info, "traj_done", d) or self.maximum_episode_lengths[b] % 500 == 0:
                    print(f'here: {d}')
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    self.maximum_episode_lengths[b] = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
                env_buf.done[t, b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return AgentInputs(observation, action, reward), traj_infos, completed_infos

class GpuResetCollector_Custom(GpuResetCollector):
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
        act = env.action_space.null_value().base
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
                    act, point = a.base, a.pointer
                    a = np.concatenate((np.expand_dims(act,0), point))
                    o, r, d, info = env.step(a)
                    traj_infos[b].step(o, a, r, d, None, info)
                    if getattr(info, "traj_done", d):
                        o = env.reset()
                        traj_infos[b] = self.TrajInfoCls()
                    if d:
                        a = env.action_space.null_value()
                        act, point = a.base, a.pointer
                        a = np.concatenate((np.expand_dims(act,0), point))
                        r = 0
                observation[b] = o
                prev_action[b] = a
                prev_reward[b] = r
        # For action-server samplers.
        if hasattr(self, "step_buffer_np") and self.step_buffer_np is not None:
            self.step_buffer_np.observation[:] = observation
            self.step_buffer_np.action[:] = prev_action
            self.step_buffer_np.reward[:] = prev_reward

        self.maximum_episode_lengths = np.array([0 for i in range(len(self.envs))])

        return AgentInputs(observation, prev_action, prev_reward), traj_infos

    def collect_batch(self, agent_inputs, traj_infos, itr):
        """Params agent_inputs and itr unused."""
        act_ready, obs_ready = self.sync.act_ready, self.sync.obs_ready
        step = self.step_buffer_np
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        agent_buf.prev_action[0] = step.action
        env_buf.prev_reward[0] = step.reward
        obs_ready.release()  # Previous obs already written, ready for new.
        completed_infos = list()
        for t in range(self.batch_T):
            env_buf.observation[t] = step.observation
            act_ready.acquire()  # Need sampled actions from server.
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(step.action[b])
                traj_infos[b].step(step.observation[b], step.action[b], r, d,
                    step.agent_info[b], env_info)
                self.maximum_episode_lengths += 1
                if getattr(env_info, "traj_done", d) or self.maximum_episode_lengths[b] % 500 == 0:
                    print(f'here: {d}')
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                step.observation[b] = o
                step.reward[b] = r
                step.done[b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = step.action  # OPTIONAL BY SERVER
            env_buf.reward[t] = step.reward
            env_buf.done[t] = step.done
            if step.agent_info:
                agent_buf.agent_info[t] = step.agent_info  # OPTIONAL BY SERVER
            obs_ready.release()  # Ready for server to use/write step buffer.

        return None, traj_infos, completed_infos