import torch
import torch.nn.functional as F

from rlpyt.utils.collections import namedarraytuple

from rlpyt.agents.base import AgentStep, BaseAgent, RecurrentAgentMixin
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from src.beta import Beta, DistInfoBeta
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method

from models import ThorCNNModel, ParameterizedActionModel, ParameterizedRecurrentActionModel

DistInfoTotal = namedarraytuple("DistInfoTotal", ["pi", "mean", "log_std"])
AgentInfo = namedarraytuple("AgentInfo", ["dist_info_b", "dist_info_p", "value"])
AgentInfoRNN = namedarraytuple("AgentInfoRNN", ["dist_info_b", "dist_info_p", "value", "prev_rnn_state"])

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
    # pi, mu, log_std, value = self.model(*policy_inputs)
    pi, alpha, beta, value, next_rnn_state = self.model(*policy_inputs)
    # return buffer_to((DistInfo(prob=pi), DistInfoStd(mean=mu, log_std=log_std), value), device="cpu")
    dist_info_b, dist_info_p, value = buffer_to((DistInfo(prob=pi), DistInfoBeta(alpha=alpha, beta=beta), value), device="cpu")
    return dist_info_b, dist_info_p, value, next_rnn_state

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
    if not torch.is_tensor(prev_action):
      prev_action = torch.hstack([prev_action.base, prev_action.pointer])

    policy_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
    # pi, mu, log_std, value = self.model(*policy_inputs)
    pi, alpha, beta, value, rnn_state = self.model(*policy_inputs, self.prev_rnn_state)
    dist_info_b = DistInfo(prob=pi)
    base_action = self.categorical_dist.sample(dist_info_b)

    dist_info_p = DistInfoBeta(alpha=alpha, beta=beta)
    pointer_action = self.beta_dist.sample(dist_info_p)
    # dist_info_p = DistInfoStd(mean=mu, log_std=log_std)
    # pointer_action = self.gaussian_dist.sample(dist_info_p)
    # clip action space

    # pointer_action = torch.clip(pointer_action, self.env_spaces.action.spaces[1].low[0], self.env_spaces.action.spaces[1].high[0])
    # pointer_action = (pointer_action + 1) / 2

    prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
    # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
    prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)

    agent_info = AgentInfoRNN(dist_info_b=dist_info_b, dist_info_p=dist_info_p, value=value, prev_rnn_state=prev_rnn_state)

    base_action, pointer_action, agent_info = buffer_to((base_action, pointer_action, agent_info), device="cpu")
    action = torch.hstack([base_action.unsqueeze(-1).float(), pointer_action])

    self.advance_rnn_state(rnn_state)
    return AgentStep(action=action, agent_info=agent_info)

  @torch.no_grad()
  def value(self, observation, prev_action, prev_reward):
    """
    Compute the value estimate for the environment state, e.g. for the
    bootstrap value, V(s_{T+1}), in the sampler.  (no grad)
    """
    policy_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
    _, _, _, value, _ = self.model(*policy_inputs, self.prev_rnn_state)
    return value.to("cpu")


class ThorAgentMixin:
  """
  Mixin class defining which environment interface properties
  are given to the model.
  """
  def make_env_to_model_kwargs(self, env_spaces):
    """Extract image shape and action size."""
    return dict(
      image_shape=env_spaces.observation.shape.image,
      base_action_size=env_spaces.action.spaces[0].n,
      pointer_action_size=env_spaces.action.spaces[1].shape[0]
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