import torch
import torch.nn.functional as F

from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.mlp import MlpModel

from rlpyt.agents.base import (AgentStep, BaseAgent, RecurrentAgentMixin,
    AlternatingRecurrentAgentMixin)

from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method

RnnState = namedarraytuple("RnnState", ["h", "c"])  # For downstream namedarraytuples to work

DistInfoTotal = namedarraytuple("DistInfoTotal", ["pi", "mean", "log_std"])
AgentInfo = namedarraytuple("AgentInfo", ["dist_info_base", "dist_info_pointer", "value"])

class BaseModel(torch.nn.Module):
  """Recurrent model: a convolutional network into an FC layer into an LSTM which outputs action probabilities and state-value estimate.
  """
  def __init__(
    self,
    image_shape,
    output_size,
    fc_sizes=512,  # Between conv and lstm.
    lstm_size=512,
    use_maxpool=False,
    channels=None,  # None uses default.
    kernel_sizes=None,
    strides=None,
    paddings=None,
  ):
    """Instantiate neural net module according to inputs."""
    super().__init__()
    self.conv = Conv2dHeadModel(
      image_shape=image_shape,
      channels=channels or [16, 32],
      kernel_sizes=kernel_sizes or [8, 4],
      strides=strides or [4, 2],
      paddings=paddings or [0, 1],
      use_maxpool=use_maxpool,
      hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
    )
    # self.lstm = torch.nn.LSTM(self.conv.output_size + output_size + 1, lstm_size)
    # self.pi = torch.nn.Linear(lstm_size, output_size)
    # self.value = torch.nn.Linear(lstm_size, 1)
    self.pi = torch.nn.Linear(self.conv.output_size, output_size)
    # self.value = torch.nn.Linear(self.conv.output_size, 1)

  def forward(self, image, prev_action, prev_reward):
    """
    Compute action probabilities and value estimate from input state.
    Infers leading dimensions of input: can be [T,B], [B], or []; provides
    returns with same leading dims.  Convolution layers process as [T*B,
    *image_shape], with T=1,B=1 when not given.  Expects uint8 images in
    [0,255] and converts them to float32 in [0,1] (to minimize image data
    storage and transfer).  Recurrent layers processed as [T,B,H]. Used in
    both sampler and in algorithm (both via the agent).  Also returns the
    next RNN state.
    """
    img = image.type(torch.float)  # Expect torch.uint8 inputs
    img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

    # Infer (presence of) leading dimensions: [T,B], [B], or [].
    lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

    fc_out = self.conv(img.view(T * B, *img_shape))
    # lstm_input = torch.cat([
    #     fc_out.view(T, B, -1),
    #     prev_action.view(T, B, -1),  # Assumed onehot.
    #     prev_reward.view(T, B, 1),
    #     ], dim=2)
    # init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
    # lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
    # pi = F.softmax(self.pi(lstm_out.view(T * B, -1)), dim=-1)
    # v = self.value(lstm_out.view(T * B, -1)).squeeze(-1)
    pi = F.softmax(self.pi(fc_out.view(T*B, -1)), dim=-1)
    # v = self.value(fc_out.view(T*B,-1)).squeeze(-1)

    # Restore leading dimensions: [T,B], [B], or [], as input.
    pi = restore_leading_dims((pi), lead_dim, T, B)
    # Model should always leave B-dimension in rnn state: [N,B,H].
    # next_rnn_state = RnnState(h=hn, c=cn)

    # return pi, v, next_rnn_state, fc_out
    return pi, fc_out

class PointerModel(torch.nn.Module):
  def __init__(
    self,
    base_action_size,
    pointer_action_size,
    input_dim=512,
    fc_sizes=512,  # Between conv and lstm.
    lstm_size=512,
    use_maxpool=False,
    channels=None,  # None uses default.
    kernel_sizes=None,
    strides=None,
    paddings=None,
  ):
    super().__init__()
    self.pointer_action_size = pointer_action_size
    self.action_head = torch.nn.Linear(base_action_size, fc_sizes)

    # mu and std, and one for value
    self.pi = MlpModel(input_dim*2, hidden_sizes=fc_sizes, output_size=pointer_action_size*2 + 1)

  def forward(self, obs_encoding, base_action, prev_action, prev_reward):
    # Compute base action encoding
    lead_dim, T, B, D = infer_leading_dims(base_action, 1)
    base_action_encoding = self.action_head(base_action.view(T*B,-1))

    lead_dim, T, B, D = infer_leading_dims(obs_encoding, 1)

    # Combine image encoding with base action encoding
    policy_inputs = torch.cat([obs_encoding, base_action_encoding], dim=-1)
    out = self.pi(policy_inputs.view(T*B,-1))
    mu = out[:, :self.pointer_action_size]
    log_std = out[:, self.pointer_action_size:-1]
    v = out[:, -1].unsqueeze(-1)

    # Restore leading dimensions: [T,B], [B], or [], as input.`
    mu, log_std, v = restore_leading_dims((mu, log_std, v), lead_dim, T, B)
    return mu, log_std, v

class ThorLSTMModel(torch.nn.Module):
  def __init__(
    self,
    image_shape,
    base_action_size,
    pointer_action_size
  ):
    super().__init__()
    self.base_model = BaseModel(image_shape, base_action_size)
    self.pointer_model = PointerModel(base_action_size, pointer_action_size)

  def compute_base_action(self, observation, prev_action, prev_reward):
    return self.base_model(observation.image, prev_action, prev_reward)

  def compute_pointer_action(self, obs_encoding, base_action, prev_action, prev_reward):
    return self.pointer_model(obs_encoding, base_action, prev_action, prev_reward)

class Agent(BaseAgent):

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
        dim=env_spaces.action.spaces[1].shape[0], # TODO; fix
        # min_std=MIN_STD,
        clip=env_spaces.action.spaces[1].high[0] / 2,  # Probably +1?
    )

  @torch.no_grad()
  def step(self, observation, prev_action, prev_reward):
    '''Compute policy action distribution from inputs and sample an action.
      Call model to first produce base action using categorical distribution.
    '''
    base_policy_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)

    # -------------------------
    # Get base action
    # -------------------------
    # pi, value, rnn_state, img_encoding = self.model.compute_base_action(*base_policy_inputs, self.prev_rnn_state)
    pi, img_encoding = self.model.compute_base_action(*base_policy_inputs)
    dist_info_base = DistInfo(prob=pi)
    base_action = self.categorical_dist.sample(dist_info_base)
    base_action = base_action.view(1,1).type(torch.float32)

    # -------------------------
    # Get pointer action
    # -------------------------
    # prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
    # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
    # (Special case: model should always leave B dimension in.)
    # prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
    pointer_policy_inputs = buffer_to((img_encoding, pi, prev_action, prev_reward), device=self.device)
    # mu, log_std, value, rnn_state = self.model.compute_pointer_action(*pointer_policy_inputs, self.prev_rnn_state)
    mu, log_std, value = self.model.compute_pointer_action(*pointer_policy_inputs)
    dist_info_pointer = DistInfoStd(mean=mu, log_std=log_std)
    # squash action [-0.5,0.5] + 1 -> [0, 1]
    pointer_action = self.gaussian_dist.sample(dist_info_pointer) + 1
    # prev_rnn_state = self.prev_rnn_state or buffer_func(rnn_state, torch.zeros_like)
    # prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)

    agent_info = AgentInfo(dist_info_base=dist_info_base, dist_info_pointer=dist_info_pointer, value=value)
        # prev_rnn_state=prev_rnn_state)

    base_action, pointer_action, agent_info = buffer_to((base_action, pointer_action, agent_info), device="cpu")
    # self.advance_rnn_state(rnn_state)
    action = torch.cat([base_action, pointer_action], dim=-1)
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

class ThorLSTMAgent(ThorAgentMixin, Agent):
  def __init__(self, ModelCls=ThorLSTMModel, **kwargs):
    super().__init__(ModelCls=ModelCls, **kwargs)
