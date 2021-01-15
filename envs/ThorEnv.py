import ai2thor.controller

import cv2
import numpy as np

import gym
from rlpyt.envs.base import Env, EnvStep
from rlpyt.envs.gym import GymEnvWrapper, EnvSpaces
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from rlpyt.spaces.composite import Composite
from rlpyt.utils.collections import namedarraytuple, namedtuple

from rlpyt.samplers.collections import TrajInfo

from config import default_config
from utils import ACTIONS
from tasks.task import Task


EnvInfo = namedtuple("EnvInfo", ["done"])
Action = namedarraytuple("Action", ["action", "pointer"])
ImageObservation = namedarraytuple("ImageObservation", ["image"])

class ThorTrajInfo(TrajInfo):
  """TrajInfo class for use with Thor Env"""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def step(self, observation, action, reward, done, agent_info, env_info):
    super().step(observation, action, reward, done, agent_info, env_info)

class ThorEnv(Env):
  def __init__(
    self,
    config,
    env_kwargs={}
  ):
    super(ThorEnv, self).__init__()

    self.config = config
    self.num_channels = 1 if self.config['grayscale'] else 3

    # --------------
    # Initialize thor instance
    # --------------
    if ai2thor.__version__ < '2.3.2':
      raise NotImplementedError

    self.headless = False

    self.controller = ai2thor.controller.Controller(
      quality=self.config['thor_quality'] if not self.headless else 'Very Low',
      headless=self.headless,
      width=self.config['screen_size'],
      height=self.config['screen_size'],
      gridSize=self.config['gridSize'],
      rotateStepDegrees=self.config['rotateStepDegrees'],
      renderClassImage=self.config['renderClassImage'],
      renderObjectImage=self.config['renderObjectImage'],
      visibilityDistance=self.config['visibility_distance']
    )

    # --------------
    # Set floorplan
    # --------------
    floorplan = self.config['floorplans']
    if isinstance(floorplan, list):
      if len(floorplan) > 1:
        raise RuntimeError(
          "Only support setting up with single floorplan. you can change the floorplan later with the step function")
      self.floorplan = "FloorPlan%d" % floorplan[0]
    elif isinstance(floorplan, int):
      self.floorplan = "FloorPlan%d" % floorplan
    else:
      raise RuntimeError("Must set floorplan")

    # --------------
    # Get list of initial locations for reseting agent
    # --------------
    event = self.controller.step(dict(action='GetReachablePositions'))
    reachable_positions = event.metadata['reachablePositions']
    num_random_choices = min(self.config['InitialRandomLocation'], len(reachable_positions))
    self.random_initial_location_options = np.random.choice(reachable_positions, num_random_choices)

    # --------------
    # Load tasks
    # --------------
    if len(self.config['tasks']) > 1:
      raise NotImplementedError()
    else:
      task = self.config['tasks'][0]
      self.tasks = [Task(f'{task}.json')]

    # TODO: include options for observation

  def process_scene_image(self, image):
    # --------------
    # grayscale
    # --------------

    if self.config['grayscale'] and image.shape[-1] != 1:
      image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # --------------
    # resize input
    # --------------
    if self.config['input_size'] != self.config['screen_size']:
      new_shape = (self.config['input_size'], self.config['input_size'])
      image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)


    if self.config['grayscale'] and image.shape[0] != 1:
      image = np.expand_dims(image, 0)

    return image

  def current_observation(self):
    processed_obs = self.process_scene_image(self.event.frame)
    return ImageObservation(image=processed_obs)

  def step(self, action_dict):
    ''' Take an environment step
    Args:
      action (dict): 'action' and 'pointer'
    '''
    if type(action_dict) == np.ndarray:
      action = self.config['policy_actions'][int(action_dict[0])]
    else:
      action = action_dict.action

    # ----------------------
    # Handle movement actions
    # ----------------------
    if action in ACTIONS['movement_actions'] or action in ACTIONS['view_actions']:
      kwargs={}
      if action in ['LookUp', 'LookDown']:
        kwargs['degrees'] = self.config['rotateHorizonDegrees']
      if action in ['RotateLeft', 'RotateRight']:
        kwargs['degrees'] = self.config['rotateStepDegrees']
      self._event = self.controller.step(dict(action=action, **kwargs))

    # ----------------------
    # Handle interaction actions
    # ----------------------
    elif action in ACTIONS['interact_actions']:
      if type(action_dict) == dict and not 'pointer' in action_dict:
        raise RuntimeError('Missing pointer in interact action.')

      if type(action_dict) == np.ndarray:
        pointer = action_dict[-2:]
      else:
        pointer = action_dict.pointer
      self._event = self.controller.step(dict(action=action, x=pointer[0], y=pointer[1]))

    elif action in ACTIONS['no_op']:
      reward = 0

    # ----------------------
    # Task is done
    # ----------------------
    done = self.tasks[0].check_task_conditions(self, conditions='goal')
    reward = 1 if done else 0

    obs = self.current_observation()
    info = {'done': done}
    info = EnvInfo(**info)
    return EnvStep(obs, reward, done, info)

  # -------------------------------
  # Helper functions for reset
  # -------------------------------
  def spawn_agent_helper(self, fixed_location=None, point=None):
    if fixed_location:
      self._event = controller.step(dict(action='TeleportFull', **fixed_location))

    elif self.config['InitialRandomLocation'] and not fixed_location:
      coord = np.random.randint(len(self.random_initial_location_options))
      rand_coord = spawn_positions[coord]

      horizon_options = np.arange(-30, 60.1, self.config['rotateHorizonDegrees'])
      rotation_options = np.arange(0, 360.1, self.config['rotateStepDegrees'])

      horizon = np.random.choice(horizon_options)
      rotation = np.random.choice(rotation_options)

      self._event = controller.step(action='TeleportFull', **rand_coord, rotation=dict(y=rotation), horizon=horizon)

      if point: raise RuntimeError("Don't yet support both random initial location and random point")

    elif point:
      self._event = controller.step(dict(action='TeleportFull', **point))

    return self._event

  def reset(self):
    # -----------------------------------------
    # Steps for reset (ordering is important)
    # 1. Reset controller
    # 2. Spawn agent at location
    # 3. Load task specifications
    # 4. Randomize object location
    # 5. Randomize object states
    # 6. Check task preconditions
    # 7. Remove objects that aren't distractors
    # 8. Remove non-distractors
    # 9. Check that task conditions are met
    # -----------------------------------------

    # 1
    self._event = self.controller.reset(self.floorplan)

    # 2
    self.spawn_agent_helper(fixed_location=None, point=None)

    # 3

    # 4
    # if self.config['InitialRandomSpawn']:
    #   randomize_object_locations()

    # 5
    # if self.config['RandomToggleStateOfAllObjects']:
    #   randomize_object_states()

    # 6
    self.tasks[0].check_task_conditions(self)

    # 7
    # self.compute_types_interact_and_remove()

    # 8
    # self.remove_non_distractors()

    # 9
    self.tasks[0].check_task_objects_exist(self)

    return self.current_observation()

  @property
  def action_space(self):
    action = IntBox(low=0, high=len(self.config['policy_actions']))
    pointer = FloatBox(low=0, high=1, shape=(2,))
    return Composite([action, pointer], Action)

  @property
  def observation_space(self):
    image = FloatBox(low=0, high=255, shape=(self.num_channels, self.config['input_size'], self.config['input_size']))
    return Composite([image], ImageObservation)

  @property
  def spaces(self):
    return EnvSpaces(
      observation=self.observation_space,
      action=self.action_space,
    )

  def close(self):
    pass

  @property
  def event(self): return self._event

  @property
  def objects(self):
    return {o['objectId'] : o for o in self.event.metadata['objects']}

  @property
  def objects_by_type_dict(self):
    return {o['objectType'] : o for o in self.event.metadata['objects']}

  @property
  def metadata(self):
    if self.event is None:
      return {}
    return self.event.metadata

  # -------------------------------
  # Helper functions
  # -------------------------------
  def objects_of_type(self, object_type):
    return {o: info for o, info in self.objects.items() if info['objectType'] == object_type}


if __name__ == '__main__':
  args = default_config()
  env = ThorEnv(args)
  print(env.action_space)
  print(env.observation_space)
  env.reset()
  a = env.action_space.sample()
  obs, reward, done, info = env.step(a)
  import ipdb; ipdb.set_trace()
  print('done')