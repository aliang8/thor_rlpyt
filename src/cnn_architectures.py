import numpy as np

extra_small=dict(
  kernel_sizes=[4, 4],
  channels=[32, 64],
  paddings=list(np.ones(2, dtype=np.int64)),
  strides=[2, 2],
  )

extra_small2=dict(
  kernel_sizes=[4, 4],
  channels=[64, 64],
  paddings=list(np.ones(2, dtype=np.int64)),
  strides=[2, 2],
  )


small=dict(
  kernel_sizes=[4, 4, 4],
  channels=[32, 64, 64],
  paddings=list(np.ones(3, dtype=np.int64)),
  strides=[2, 2, 2],
  )

dqn=dict(
  kernel_sizes=[8, 4, 3],
  channels=[32, 64, 64],
  paddings=list(np.zeros(3, dtype=np.int64)),
  strides=[4, 2, 1],
  )

medium=dict(
  kernel_sizes=[4, 4, 4, 4],
  channels=[32, 32, 64, 64],
  paddings=list(np.ones(4, dtype=np.int64)),
  strides=[2, 2, 2, 2],
  )