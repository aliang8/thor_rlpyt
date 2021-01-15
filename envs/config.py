import argparse

def add_thor_args(parser):
  thor = parser.add_argument_group("thor")
  thor.add_argument('-f', '--dummy-f', type=str, default=None, help="dummy for jupyter")
  thor.add_argument('-fps', '--floorplans', nargs='+', type=int, default=[24], help="floorplans to use for training object model")
  thor.add_argument('--policy-actions', type=str, nargs='+', default=[
      'MoveAhead',
      'MoveBack',
      'MoveRight',
      'MoveLeft',
      'LookUp',
      'LookDown',
      'RotateRight',
      'RotateLeft',
      "PickupObject",
      "PutObject",
      "OpenObject",
      "CloseObject",
      "ToggleObjectOn",
      "ToggleObjectOff",
      'SliceObject',
    ], help="actions that agent is allowed to take")
  thor.add_argument('--training-actions', type=str, nargs='+', default=["Close"], help="actions for the environment which the agent is not allowed to take. this is a convenience in order to use the baselines vec env environment ")
  thor.add_argument('--num-channels', type=int, default=3, help="number of channels for image (1 if grayscale)")
  thor.add_argument('--grayscale', type=int, default=1, help="convert scene image to grayscale")
  thor.add_argument('--n-stacks', type=int, default=1, help="number of frames as inputs")
  thor.add_argument('--screen-size', type=int, default=300, help="width/height of controller window")
  thor.add_argument('--input-size', type=int, default=128, help="size reshaping of thor window for neural network")
  thor.add_argument('--visibility-distance', type=float, default=1.5, help="how far things can be and still be visible")
  thor.add_argument('--conservative-interaction-distance', type=float, default=1, help="how far things can be and still be visible")

  thor.add_argument('--force-visible', type=int, default=1, help="Whether to force objects to be visible or not")
  thor.add_argument('--InitialRandomSpawn', type=int, default=0, choices=[0,1], help="Whether to use this function to intialize scenes. Main utility is setting seed for object generation. however, as of 1.0.2, this seed doesn't give consistent object ids.")
  thor.add_argument('--RandomToggleStateOfAllObjects', type=int, default=0, choices=[0,1], help="Whether to randomize initial states of objects")

  thor.add_argument('--InitialRandomLocation', type=int, default=0, help="Whether to initialize agent in random locations. if yes, the integer specifies how many to use")

  thor.add_argument('--renderClassImage', type=int, default=1, choices=[0,1])
  thor.add_argument('--renderObjectImage', type=int, default=1, choices=[0,1])
  thor.add_argument('--maxNumRepeats', type=int, default=5, help="Max number of repeat initializations to spawn all objects visibly. A thor argument.")
  thor.add_argument('--rotateStepDegrees', type=float, default=90, help="degrees to use for rotation")
  thor.add_argument('--rotateHorizonDegrees', type=float, default=30, help="degrees to use for rotation")

  # TASK STUFF - clean this up later
  thor.add_argument('--tasks', nargs='+', type=str, default=[], help="tasks to load thor environment with")
  thor.add_argument('--reward-density', type=int, default=[], nargs='+', help="0 - single terminal reward, 1 - reward after series of object interactions, 2 - reward after each object interaction")
  thor.add_argument('--reward-start-value', type=float, default=0.05, help="starting reward value for task")
  thor.add_argument('--reward-increment-value', type=float, default=0.05, help="how much to increase reward by for each task step")
  thor.add_argument('--reward-terminal-value', type=float, default=1, help="final reward value for task, if not set defaults to final value of incrementing")

  thor.add_argument('--predefined-task', type=str, default=None, help="load predefined task in thor/tasks/directory.py")
  thor.add_argument('--predefined-multitask', type=str, default=None, help="load predefined task in thor/multitasks/directory.yaml")
  thor.add_argument('--thor-quality', type=str, default='Very High',
    choices = ['DONOTUSE', 'Very Low', 'Low', 'Medium', 'MediumCloseFitShadows', 'High', 'Very High', 'Ultra', 'High WebGL'],
    help="Quality of images. The lower, the faster the fps.")
  thor.add_argument('-oa', '--object-actions', nargs='+', type=str, default=['PreviousObject', 'NextObject'], help="actions for shifting between objects. Used for ThorObjectEnv")
  thor.add_argument('-pa', '--point-actions', nargs='+', type=str, default=['PreviousPoint', 'NextPoint'], help="actions for shifting between points in map. Used for ThorFinderEnv")
  thor.add_argument('--bandit-reward-source', type=str, default="random", choices=['none', 'random', 'affordances'], help="actions for shifting between points in map. Used for ThorFinderEnv")
  thor.add_argument('--goal-wrapper', type=int, default=0, help="what type of goal wrapper to add for environment. 1: object image + scene images")
  thor.add_argument('--observe-nouns', type=str, default='none', choices=['all', 'none', 'task'], help="whether to display nouns of objects in environment when returning environment observation")

  thor.add_argument('--num-distractor-categories', type=int, default=1000, help="how many nontask object categories to keep")
  thor.add_argument('--num-object-configurations', type=int, default=1000, help="how many object configurations to choose from")
  thor.add_argument('--parents-are-distractors', type=int, default=0, help="whether to treat parents as distractors")
  thor.add_argument('--types-to-remove', type=str, default=[], nargs="+", help="how nontask object categories to definitely remove. any from task will not be removed")
  thor.add_argument('--types-to-disable', type=str, default=[], nargs="+", help="object categories to disable from scene (i.e. for not interactable)")
  thor.add_argument('--types-to-keep-interact', type=str, default=[], nargs="+", help="which nontask object categories to keep")
  thor.add_argument('--types-to-keep-nointeract', type=str, default=[], nargs="+", help="which nontask object categories to keep")
  thor.add_argument('--predefined-spawning-positions-file', type=str, default=None, help="file with predefined spawning positions")
  thor.add_argument('--excluded-target-spawning', type=str, nargs="+", default=[], help="where target objects are not allowed to spawn")
  thor.add_argument('--max-target-distance', type=float, default=2.5, help="max distance of spawning target object")
  thor.add_argument('--gridSize', type=float, default=0.25, help="degrees to use for rotation")

  thor.add_argument('--remove-excess-pickupable', type=int, default=0, choices=[0,1], help="whether to remove excess objects")
  thor.add_argument('--keep-k-task-neighbors', type=int, default=2, help="number of non-task objects to not remove")

  return parser

def default_config(parser=None):
  if not parser: parser = argparse.ArgumentParser()
  parser = add_thor_args(parser)
  args, unknown = parser.parse_known_args()
  args = vars(args)
  return args