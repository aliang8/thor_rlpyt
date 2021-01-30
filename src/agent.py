import torch
import torch.nn.functional as F

from rlpyt.utils.collections import namedarraytuple

from rlpyt.agents.base import AgentStep, BaseAgent, RecurrentAgentMixin
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from src.beta import Beta, DistInfoBeta
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method

from rlpyt.agents.dqn.dqn_agent import DqnAgent
from models import (ThorCNNModel, 
                    ParameterizedActionModel, 
                    ParameterizedRecurrentActionModel,
                    ThorR2D1Model)


DistInfoTotal = namedarraytuple("DistInfoTotal", ["pi", "mean", "log_std"])
AgentInfo = namedarraytuple("AgentInfo", ["dist_info_b", "dist_info_o", "value"])
AgentInfoRNN = namedarraytuple("AgentInfoRNN", ["dist_info_b", "dist_info_o", "value", "prev_rnn_state"])
AgentInfoObjectSelection = namedarraytuple("AgentInfoObjectSelection", ["base_q", "object_selection_q", "prev_rnn_state"])

class CNNAgent(BaseAgent):

  def __call__(self, observation, prev_action, prev_reward):
    """Performs forward pass on training data, for algorithm."""
    base_policy_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
    pi, img_encoding = self.model.compute_base_action(*base_policy_inputs)

    pointer_policy_inputs = buffer_to((img_encoding, pi, prev_action, prev_reward), device=self.device)
    mu, log_std, value = self.model.compute_pointer_action(*pointer_policy_inputs)
    return buffer_to((DistInfo(prob=pi), DistInfoStd(mean=mu, log_std=log_std), value), device="cpu")

  def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
    super().initialize(env_spaces, share_memory)
    self.categorical_dist = Categorical(dim=env_spaces.action.spaces[0].n)
    self.gaussian_dist = Gaussian(
        dim=env_spaces.action.spaces[1].shape[0],
        clip=env_spaces.action.spaces[1].high[0]/2,
    )

  @torch.no_grad()
  def step(self, observation, prev_action, prev_reward):
    '''Compute policy action distribution from inputs and sample an action.
    '''
    # -------------------------
    # Get base action
    # -------------------------
    base_policy_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
    pi, img_encoding = self.model.compute_base_action(*base_policy_inputs)
    dist_info_b = DistInfo(prob=pi)
    base_action = self.categorical_dist.sample(dist_info_b)

    # -------------------------
    # Get pointer action
    # -------------------------
    pointer_policy_inputs = buffer_to((img_encoding, pi, prev_action, prev_reward), device=self.device)
    mu, log_std, value = self.model.compute_pointer_action(*pointer_policy_inputs)

    dist_info_p = DistInfoStd(mean=mu, log_std=log_std)
    # clipped action [-0.5,0.5] + 0.5 -> [0, 1]
    pointer_action = self.gaussian_dist.sample(dist_info_p) + 0.5

    agent_info = AgentInfo(dist_info_b=dist_info_b, dist_info_p=dist_info_p, value=value)

    base_action, pointer_action, agent_info = buffer_to((base_action, pointer_action, agent_info), device="cpu")
    # base_action = base_action.view(1,1)
    # action = torch.cat([base_action, pointer_action], dim=-1)
    action = torch.cat([base_action.unsqueeze(0).float(), pointer_action], dim=-1)
    return AgentStep(action=action, agent_info=agent_info)

  @torch.no_grad()
  def value(self, observation, prev_action, prev_reward):
    """
    Compute the value estimate for the environment state, e.g. for the
    bootstrap value, V(s_{T+1}), in the sampler.  (no grad)
    """
    base_policy_inputs = buffer_to((observation, prev_action, prev_reward),
        device=self.device)
    pi, img_encoding = self.model.compute_base_action(*base_policy_inputs)
    pointer_policy_inputs = buffer_to((img_encoding, pi, prev_action, prev_reward))
    _mu, _log_std, value = self.model.compute_pointer_action(*pointer_policy_inputs)
    return value.to("cpu")

class ParameterizedCNNAgent(BaseAgent):

  def __call__(self, observation, prev_action, prev_reward):
    """Performs forward pass on training data, for algorithm."""
    policy_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
    # pi, mu, log_std, value = self.model(*policy_inputs)
    pi, alpha, beta, value = self.model(*policy_inputs)
    # return buffer_to((DistInfo(prob=pi), DistInfoStd(mean=mu, log_std=log_std), value), device="cpu")
    return buffer_to((DistInfo(prob=pi), DistInfoBeta(alpha=alpha, beta=beta), value), device="cpu")

  def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
    super().initialize(env_spaces, share_memory)
    self.categorical_dist = Categorical(dim=env_spaces.action.spaces[0].n)
    # self.gaussian_dist = Gaussian(
    #     dim=env_spaces.action.spaces[1].shape[0],
    #     clip=env_spaces.action.spaces[1].high[0],
    # )
    self.beta_dist = Beta(
      dim=env_spaces.action.spaces[1].shape[0]
    )

  @torch.no_grad()
  def step(self, observation, prev_action, prev_reward):
    '''Compute policy action distribution from inputs and sample an action.
    '''
    policy_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
    # pi, mu, log_std, value = self.model(*policy_inputs)
    pi, alpha, beta, value = self.model(*policy_inputs)
    dist_info_b = DistInfo(prob=pi)
    base_action = self.categorical_dist.sample(dist_info_b)

    dist_info_p = DistInfoBeta(alpha=alpha, beta=beta)
    pointer_action = self.beta_dist.sample(dist_info_p)
    # dist_info_p = DistInfoStd(mean=mu, log_std=log_std)
    # pointer_action = self.gaussian_dist.sample(dist_info_p)
    # clip action space

    # pointer_action = torch.clip(pointer_action, self.env_spaces.action.spaces[1].low[0], self.env_spaces.action.spaces[1].high[0])
    # pointer_action = (pointer_action + 1) / 2

    agent_info = AgentInfo(dist_info_b=dist_info_b, dist_info_p=dist_info_p, value=value)

    base_action, pointer_action, agent_info = buffer_to((base_action, pointer_action, agent_info), device="cpu")
    action = torch.hstack([base_action.unsqueeze(-1).float(), pointer_action])
    return AgentStep(action=action, agent_info=agent_info)

  @torch.no_grad()
  def value(self, observation, prev_action, prev_reward):
    """
    Compute the value estimate for the environment state, e.g. for the
    bootstrap value, V(s_{T+1}), in the sampler.  (no grad)
    """
    policy_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
    _, _, _, value = self.model(*policy_inputs)
    return value.to("cpu")


class ParameterizedCNNLSTMAgentBase(BaseAgent):

  def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
    """Performs forward pass on training data, for algorithm."""
    policy_inputs = buffer_to((observation, prev_action, prev_reward, init_rnn_state), device=self.device)
    if self.model.action_sizes[0][1] == 'categorical':
        pi, other_pi, value, next_rnn_state = self.model(*policy_inputs)
      # return buffer_to((DistInfo(prob=pi), DistInfoStd(mean=mu, log_std=log_std), value), device="cpu")
        dist_info_b, dist_info_o, value = buffer_to((DistInfo(prob=pi), DistInfo(prob=other_pi), value), device="cpu")
    else:
      # pi, mu, log_std, value = self.model(*policy_inputs)
      pi, alpha, beta, value, next_rnn_state = self.model(*policy_inputs)
    # return buffer_to((DistInfo(prob=pi), DistInfoStd(mean=mu, log_std=log_std), value), device="cpu")
      dist_info_b, dist_info_o, value = buffer_to((DistInfo(prob=pi), DistInfoBeta(alpha=alpha, beta=beta), value), device="cpu")
    return dist_info_b, dist_info_o, value, next_rnn_state

  def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
    super().initialize(env_spaces, share_memory)

    if self.model.action_sizes[0][1] == 'categorical':
      # Base action
      self.base_action_dist = Categorical(dim=env_spaces.action.spaces[0].n)

    if self.model.action_sizes[1][1] == 'categorical':
      self.other_action_dist = Categorical(dim=env_spaces.action.spaces[1].n)
    else:
      # self.gaussian_dist = Gaussian(
      #     dim=env_spaces.action.spaces[1].shape[0],
      #     clip=env_spaces.action.spaces[1].high[0],
      # )
      self.beta_dist = Beta(
        dim=env_spaces.action.spaces[1].shape[0]
      )

  @torch.no_grad()
  def step(self, observation, prev_action, prev_reward):
    '''Compute policy action distribution from inputs and sample an action.
    '''
    if not torch.is_tensor(prev_action):
      if 'object' in prev_action._fields:
        prev_action = torch.hstack([prev_action.base, prev_action.object])        
      else:
        prev_action = torch.hstack([prev_action.base, prev_action.pointer])

    policy_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
    # pi, mu, log_std, value = self.model(*policy_inputs)

    if self.model.action_sizes[1][1] == 'categorical':
      base_pi, other_pi, value, rnn_state = self.model(*policy_inputs, self.prev_rnn_state)
    else:
      pi, alpha, beta, value, rnn_state = self.model(*policy_inputs, self.prev_rnn_state)

    dist_info_b = DistInfo(prob=base_pi)
    base_action = self.base_action_dist.sample(dist_info_b)

    if self.model.action_sizes[1][1] == 'categorical':
      dist_info_o = DistInfo(prob=other_pi)
      other_action = self.other_action_dist.sample(dist_info_o)
    else:    
      dist_info_p = DistInfoBeta(alpha=alpha, beta=beta)
      other_action = self.beta_dist.sample(dist_info_p)
    # dist_info_p = DistInfoStd(mean=mu, log_std=log_std)
    # pointer_action = self.gaussian_dist.sample(dist_info_p)
    # clip action space

    # pointer_action = torch.clip(pointer_action, self.env_spaces.action.spaces[1].low[0], self.env_spaces.action.spaces[1].high[0])
    # pointer_action = (pointer_action + 1) / 2

    prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
    # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
    prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)

    agent_info = AgentInfoRNN(dist_info_b=dist_info_b, dist_info_o=dist_info_o, value=value, prev_rnn_state=prev_rnn_state)

    base_action, other_action, agent_info = buffer_to((base_action, other_action, agent_info), device="cpu")

    if self.model.action_sizes[1][1] == 'categorical':
      action = torch.hstack([base_action.float(), other_action]).unsqueeze(0)
    else:
      action = torch.hstack([base_action.unsqueeze(-1).float(), other_action])

    self.advance_rnn_state(rnn_state)
    return AgentStep(action=action, agent_info=agent_info)

  @torch.no_grad()
  def value(self, observation, prev_action, prev_reward):
    """
    Compute the value estimate for the environment state, e.g. for the
    bootstrap value, V(s_{T+1}), in the sampler.  (no grad)
    """
    policy_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
    if self.model.action_sizes[1][1] == 'categorical':
      _, _, value, _ = self.model(*policy_inputs, self.prev_rnn_state)
    else:
      _, _, _, value, _ = self.model(*policy_inputs, self.prev_rnn_state)

    return value.to("cpu")

from rlpyt.agents.dqn.epsilon_greedy import EpsilonGreedyAgentMixin
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy
from rlpyt.utils.logging import logger

class CompositeDQNAgent(EpsilonGreedyAgentMixin, BaseAgent):
    def initialize(self, env_spaces, share_memory=False,
        global_B=1, env_ranks=None):
        """Along with standard initialization, creates vector-valued epsilon
        for exploration, if applicable, with a different epsilon for each
        environment instance."""
        _initial_model_state_dict = self.initial_model_state_dict
        self.initial_model_state_dict = None  # don't let base agent try to initialize model
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.target_model = self.ModelCls(**self.env_model_kwargs,
            **self.model_kwargs)
        if _initial_model_state_dict is not None:
            self.model.load_state_dict(_initial_model_state_dict['model'])
            self.target_model.load_state_dict(_initial_model_state_dict['model'])
            env_spaces.action.spaces[0].n,
        self.base_distribution = EpsilonGreedy(dim=env_spaces.action.spaces[0].n)
        self.object_selection_distribution = EpsilonGreedy(dim=env_spaces.action.spaces[1].n)

        if env_ranks is not None:
            self.make_vec_eps(global_B, env_ranks)

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.target_model.to(self.device)

    def state_dict(self):
        return dict(model=self.model.state_dict(),
            target=self.target_model.state_dict())

    def update_target(self, tau=1):
        """Copies the model parameters into the target model."""
        update_state_dict(self.target_model, self.model.state_dict(), tau)

    # Eps greedy stuff
    def set_sample_epsilon_greedy(self, epsilon):
        self.base_distribution.set_epsilon(epsilon)
        self.object_selection_distribution.set_epsilon(epsilon)

    def sample_mode(self, itr):
        """Extend method to set epsilon for sampling (including annealing)."""
        BaseAgent.sample_mode(self, itr)
        itr_min = self._eps_itr_min_max[0]  # Shared memory for CpuSampler
        itr_max = self._eps_itr_min_max[1]
        if itr <= itr_max:
            prog = min(1, max(0, itr - itr_min) / (itr_max - itr_min))
            self.eps_sample = prog * self.eps_final + (1 - prog) * self.eps_init
            if itr % (itr_max // 10) == 0 or itr == itr_max:
                logger.log(f"Agent at itr {itr}, sample eps {self.eps_sample}"
                    f" (min itr: {itr_min}, max_itr: {itr_max})")
        self.base_distribution.set_epsilon(self.eps_sample)
        self.object_selection_distribution.set_epsilon(self.eps_sample)

    def eval_mode(self, itr):
        """Extend method to set epsilon for evaluation, using 1 for
        pre-training eval."""
        BaseAgent.sample_mode(self, itr)
        logger.log(f"Agent at itr {itr}, eval eps "
            f"{self.eps_eval if itr > 0 else 1.}")
        self.base_distribution.set_epsilon(self.eps_eval if itr > 0 else 1.)
        self.object_selection_distribution.set_epsilon(self.eps_eval if itr > 0 else 1.)


class ThorObjectSelectionR2D1AgentBase(CompositeDQNAgent):

  def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
    # Assume init_rnn_state already shaped: [N,B,H]
    prev_base_action = self.base_distribution.to_onehot(prev_action)
    prev_object_selection_action = self.object_selection_distribution.to_onehot(prev_action)
    model_inputs = buffer_to((observation, prev_action, prev_reward,
        init_rnn_state), device=self.device)
    base_q, object_selection_q, rnn_state = self.model(*model_inputs)
    return base_q.cpu(), object_selection_q.cpu(), rnn_state  # Leave rnn state on device.

  @torch.no_grad()
  def step(self, observation, prev_action, prev_reward):
    """Computes Q-values for states/observations and selects actions by
    epsilon-greedy (no grad).  Advances RNN state."""
    if not torch.is_tensor(prev_action):
      prev_base_action = prev_action.base
      prev_object_selection_action = prev_action.object
    else:
      # T x 1
      prev_base_action, prev_object_selection_action = torch.split(prev_action, 1, dim=-1)
      # TODO: why squeeze?
      prev_base_action = prev_base_action.squeeze(0)
      prev_object_selection_action = prev_object_selection_action.squeeze(0)

    # print(prev_base_action.shape)
    prev_base_action = self.base_distribution.to_onehot(prev_base_action)
    # print(prev_object_selection_action.shape)
    prev_object_selection_action = self.object_selection_distribution.to_onehot(prev_object_selection_action)
    prev_action = torch.hstack([prev_base_action, prev_object_selection_action])
    agent_inputs = buffer_to((observation, prev_action, prev_reward),
        device=self.device)
    base_q, object_selection_q, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)  # Model handles None.
    base_q = base_q.cpu()
    object_selection_q = object_selection_q.cpu()
    base_action = self.base_distribution.sample(base_q)
    object_selection_action = self.object_selection_distribution.sample(object_selection_q)

    prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
    # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
    # (Special case, model should always leave B dimension in.)
    prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
    prev_rnn_state = buffer_to(prev_rnn_state, device="cpu")
    agent_info = AgentInfoObjectSelection(base_q=base_q, object_selection_q=object_selection_q, prev_rnn_state=prev_rnn_state)
    self.advance_rnn_state(rnn_state)  # Keep on device.

    action = torch.hstack([base_action.float(), object_selection_action]).unsqueeze(0)
    return AgentStep(action=action, agent_info=agent_info)
  
  def target(self, observation, prev_action, prev_reward, init_rnn_state):
    # Assume init_rnn_state already shaped: [N,B,H]
    prev_base_action = self.base_distribution.to_onehot(prev_action)
    prev_object_selection_action = self.object_selection_distribution.to_onehot(prev_action)
    model_inputs = buffer_to((observation, prev_action, prev_reward, init_rnn_state),
        device=self.device)
    target_base_q, target_object_selection_q, rnn_state = self.target_model(*model_inputs)
    return target_q.cpu(), target_object_selection_q.cpu(), rnn_state  # Leave rnn state on device.

class ThorAgentMixin:
  """
  Mixin class defining which environment interface properties
  are given to the model.
  """
  def make_env_to_model_kwargs(self, env_spaces):
    """Extract image shape and action size."""
    if 'object' in env_spaces[1].names:
        return dict(
            image_shape=env_spaces.observation.shape.image,
            action_sizes=[(env_spaces.action.spaces[0].n, 'categorical'),
                          (env_spaces.action.spaces[1].n, 'categorical')]
        )
    else:
        return dict(
          image_shape=env_spaces.observation.shape.image,
          action_sizes=[(env_spaces.action.spaces[0].n, 'categorical'),
                        (env_spaces.action.spaces[1].shape[0], 'gaussian')]
        )

class ThorCNNAgent(ThorAgentMixin, CNNAgent):
  def __init__(self, ModelCls=ThorCNNModel, **kwargs):
    super().__init__(ModelCls=ModelCls, **kwargs)

class ThorParameterizedCNNAgent(ThorAgentMixin, ParameterizedCNNAgent):
  def __init__(self, ModelCls=ParameterizedActionModel, **kwargs):
    super().__init__(ModelCls=ModelCls, **kwargs)

class ThorParameterizedCNNLSTMAgent(ThorAgentMixin, RecurrentAgentMixin, ParameterizedCNNLSTMAgentBase):
  def __init__(self, ModelCls=ParameterizedRecurrentActionModel, **kwargs):
    super().__init__(ModelCls=ModelCls, **kwargs)

class ThorObjectSelectionR2D1Agent(ThorAgentMixin, RecurrentAgentMixin, ThorObjectSelectionR2D1AgentBase):

    def __init__(self, ModelCls=ThorR2D1Model, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)