import torch
import torch.nn.functional as F

from rlpyt.models.mlp import MlpModel
from rlpyt.models.conv2d import Conv2dHeadModel

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple

RnnState = namedarraytuple("RnnState", ["h", "c"])

class BaseActionModel(torch.nn.Module):
  """A convolutional network into an FC layer which outputs action probabilities and state-value estimate for the base action
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
    self.pi = torch.nn.Linear(self.conv.output_size, output_size)

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
    pi = F.softmax(self.pi(fc_out.view(T*B, -1)), dim=-1)

    # Restore leading dimensions: [T,B], [B], or [], as input.
    pi = restore_leading_dims((pi), lead_dim, T, B)
    return pi, fc_out

class PointerActionModel(torch.nn.Module):
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

    # Encodes base action probabilities
    self.action_head = torch.nn.Linear(base_action_size, fc_sizes)

    # Outputs mu and std for each action, and one for value
    self.pi = MlpModel(input_dim*2, hidden_sizes=fc_sizes, output_size=pointer_action_size*2 + 1)

  def forward(self, obs_encoding, base_action, prev_action, prev_reward):
    # # Compute base action encoding
    lead_dim, T, B, D = infer_leading_dims(base_action, 1)
    base_action_encoding = self.action_head(base_action.view(T*B,-1))

    # Combine image encoding with base action encoding
    _, T, B, D = infer_leading_dims(obs_encoding, 1)
    policy_input = torch.cat([obs_encoding, base_action_encoding], dim=-1)
    out = self.pi(policy_input.view(T*B,-1))

    mu = out[:, :self.pointer_action_size]
    log_std = out[:, self.pointer_action_size:-1]
    v = out[:, -1].unsqueeze(-1)

    # Restore leading dimensions: [T,B], [B], or [], as input.`
    mu, log_std, v = restore_leading_dims((mu, log_std, v), lead_dim, T, B)
    return mu, log_std, v

class ThorCNNModel(torch.nn.Module):
  def __init__(
    self,
    image_shape,
    base_action_size,
    pointer_action_size
  ):
    super().__init__()
    self.base_model = BaseActionModel(image_shape, base_action_size)
    self.pointer_model = PointerActionModel(base_action_size, pointer_action_size)

  def compute_base_action(self, observation, prev_action, prev_reward):
    return self.base_model(observation.image, prev_action, prev_reward)

  def compute_pointer_action(self, obs_encoding, base_action, prev_action, prev_reward):
    return self.pointer_model(obs_encoding, base_action, prev_action, prev_reward)


class ParameterizedActionModel(torch.nn.Module):
  def __init__(
    self,
    image_shape,
    base_action_size,
    pointer_action_size,
    fc_sizes=512,  # Between conv and lstm.
    lstm_size=512,
    use_maxpool=False,
    channels=None,  # None uses default.
    kernel_sizes=None,
    strides=None,
    paddings=None,
  ):
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

    self.base_action_size = base_action_size
    self.pointer_action_size = pointer_action_size
    self.base_pi = torch.nn.Linear(self.conv.output_size, base_action_size)
    self.pointer_pi = torch.nn.Linear(self.conv.output_size, pointer_action_size*2)
    self.value = torch.nn.Linear(self.conv.output_size, 1)

  def forward(self, observation, prev_action, prev_reward):
    img = observation.image.type(torch.float)
    img = img.mul_(1. / 255)

    # Infer (presence of) leading dimensions: [T,B], [B], or [].
    lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

    fc_out = self.conv(img.view(T*B, *img_shape))

    pi = F.softmax(self.base_pi(fc_out.view(T*B, -1)), dim=-1)
    pointer_out = torch.sigmoid(self.pointer_pi(fc_out.view(T*B,-1)))

    # mu = pointer_out[:, :self.pointer_action_size]
    # log_std = pointer_out[:, self.pointer_action_size:-1]
    # v = pointer_out[:, -1].unsqueeze(-1)

    mu = pointer_out[:, :self.pointer_action_size]
    log_std = pointer_out[:, self.pointer_action_size:]
    # v = pointer_out[:, -1].unsqueeze(-1)

    v = self.value(fc_out.view(T*B,-1)).squeeze(-1)

    # Restore leading dimensions: [T,B], [B], or [], as input.
    pi, mu, log_std, v = restore_leading_dims((pi, mu, log_std, v), lead_dim, T, B)

    return pi, mu, log_std, v

class ParameterizedRecurrentActionModel(torch.nn.Module):
  def __init__(
    self,
    image_shape,
    action_sizes, # [(size, distr)]
    fc_sizes=512,  # Between conv and lstm.
    lstm_size=512,
    use_maxpool=False,
    channels=None,  # None uses default.
    kernel_sizes=None,
    strides=None,
    paddings=None,
  ):
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

    self.action_sizes = action_sizes
    self.other_action_size = 0
    if len(action_sizes) > 1:
      self.other_action_size = sum([1 if x[1] == 'categorical' else x[0] for x in action_sizes[1:]])
    base_output_size = 1
    self.lstm = torch.nn.LSTM(self.conv.output_size + base_output_size + self.other_action_size + 1, lstm_size)
    self.base_pi = torch.nn.Linear(lstm_size, action_sizes[0][0])
    if len(action_sizes) > 1:
      self.other_pis = torch.nn.ModuleList()
      for size in action_sizes[1:]:
        if size[1] == 'gaussian':
          self.other_pis.append(torch.nn.Linear(lstm_size, size[0]*2))
        else:
          self.other_pis.append(torch.nn.Linear(lstm_size, size[0]))

    self.value = torch.nn.Linear(lstm_size, 1)

  def forward(self, observation, prev_action, prev_reward, init_rnn_state):
    img = observation.image.type(torch.float)
    img = img.mul_(1. / 255)

    # Infer (presence of) leading dimensions: [T,B], [B], or [].
    lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

    fc_out = self.conv(img.view(T*B, *img_shape))

    lstm_input = torch.cat([
      fc_out.view(T, B, -1),
      prev_action.view(T, B, -1),  # Assumed onehot.
      prev_reward.view(T, B, 1),
      ], dim=2)
    init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
    lstm_out, (hn, cn) = self.lstm(lstm_input.float(), init_rnn_state)

    pi = F.softmax(self.base_pi(lstm_out.view(T*B, -1)), dim=-1)
    v = self.value(lstm_out.view(T*B,-1)).squeeze(-1)

    for i, size in enumerate(self.action_sizes[1:]):
      if size[1] == 'gaussian':
        out = torch.sigmoid(self.other_pis[i](lstm_out.view(T*B,-1)))

        # mu = out[:, :self.pointer_action_size]
        # log_std = out[:, self.pointer_action_size:-1]
        # v = out[:, -1].unsqueeze(-1)

        mu = out[:, :self.pointer_action_size]
        log_std = out[:, self.pointer_action_size:]
        # v = out[:, -1].unsqueeze(-1)
      else:
        out = F.softmax(self.other_pis[i](lstm_out.view(T*B, -1)).cuda(), dim=-1)

    # Restore leading dimensions: [T,B], [B], or [], as input.
    pi, out, v = restore_leading_dims((pi, out, v), lead_dim, T, B)
    next_rnn_state = RnnState(h=hn, c=cn)

    return pi, out, v, next_rnn_state

class ThorR2D1Model(torch.nn.Module):
  def __init__(
      self,
      image_shape,
      base_action_size,
      object_selection_size,
      fc_size=512,  # Between conv and lstm.
      lstm_size=512,
      head_size=512,
      dueling=False,
      use_maxpool=False,
      channels=None,  # None uses default.
      kernel_sizes=None,
      strides=None,
      paddings=None,
      ):
      """Instantiates the neural network according to arguments; network defaults
      stored within this method."""
      super().__init__()
      self.dueling = dueling
      self.conv = Conv2dHeadModel(
        image_shape=image_shape,
        channels=channels or [32, 64, 64],
        kernel_sizes=kernel_sizes or [8, 4, 3],
        strides=strides or [4, 2, 1],
        paddings=paddings or [0, 1, 1],
        use_maxpool=use_maxpool,
        hidden_sizes=fc_size,  # ReLU applied here (Steven).
      )
      self.lstm = torch.nn.LSTM(self.conv.output_size + base_action_size + object_selection_size + 1, lstm_size)
      if dueling:
        self.base_head = DuelingHeadModel(lstm_size, head_size, base_action_size)
        self.object_selection_head = DuelingHeadModel(lstm_size, head_size, object_selection_size)
      else:
        self.base_head = MlpModel(lstm_size, head_size, output_size=base_action_size)
        self.object_selection_head = MlpModel(lstm_size, head_size, output_size=object_selection_size)

  def forward(self, observation, prev_action, prev_reward, init_rnn_state):
    """Feedforward layers process as [T*B,H]. Return same leading dims as
    input, can be [T,B], [B], or []."""
    img = observation.image.type(torch.float)  # Expect torch.uint8 inputs
    img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

    # Infer (presence of) leading dimensions: [T,B], [B], or [].
    lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

    conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.

    lstm_input = torch.cat([
      conv_out.view(T, B, -1),
      prev_action.view(T, B, -1),  # Assumed onehot.
      prev_reward.view(T, B, 1),
      ], dim=2)
    init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
    lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)

    base_q = self.base_head(lstm_out.view(T * B, -1))
    object_selection_q = self.object_selection_head(lstm_out.view(T * B, -1))

    # Restore leading dimensions: [T,B], [B], or [], as input.
    base_q = restore_leading_dims(base_q, lead_dim, T, B)
    object_selection_q = restore_leading_dims(object_selection_q, lead_dim, T, B)

    # Model should always leave B-dimension in rnn state: [N,B,H].
    next_rnn_state = RnnState(h=hn, c=cn)

    return base_q, object_selection_q, next_rnn_state