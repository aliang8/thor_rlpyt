import json
from utils import condition2statetype

class Task(object):
  '''Task object
  '''
  def __init__(self, task_json):
    self.task_json = json.load(open(task_json,'r'))
    self.task_completed = False

  def __str__(self):
    return self.task_json[0]['taskname']

  def abstract_task_type(self):
    pass

  @property
  def preconditions(self):
    return self.task_json[0]['preconditions']

  @property
  def objects(self):
    if 'objects' in self.task_json[0]:
      return self.task_json[0]['objects']
    return []

  @property
  def actions(self):
    if 'actions' in self.task_json[0]:
      return self.task_json[0]['actions']
    return []

  @property
  def clear(self):
    if 'clear' in self.task_json[0]:
      return self.task_json[0]['clear']
    return []

  @property
  def reward(self):
    if 'reward' in self.task_json[0]:
      return self.task_json[0]['reward']
    return 1

  @property
  def goal_conditions(self):
    return self.task_json[0]['goal']

  def display(self):
    return ''

  def objects_to_clear(self, objects):
    '''Remove objects contained in receptacle objects to be cleared
    '''
    matching_objects = [o for o, info in objects.items() if info['objectType'].lower() in self.clear]

    types_to_remove = []
    for obj in matching_objects:
      if obj['receptacleObjectIds']:
        types_to_remove.extend([objects[o]['objectType'] for o in obj['receptacleObjectIds']])

    return types_to_remove

  def check_task_conditions(self, env, conditions="pre", count=0):
    '''Check that conditions of the task has been satisfied
    '''
    conditions = self.preconditions if conditions == 'pre' else self.goal_conditions

    done = [False] * len(conditions)

    for i, cond in enumerate(conditions):
      if len(cond) == 3:
        property_, object_type, val = cond
        obj_infos = env.objects_of_type(object_type)

        # -----------------------------------------------------
        # Update object settings that don't match preconditions
        # -----------------------------------------------------
        condition_satisfied = any(info for o, info in obj_infos.items() if info[property_] == val)

        if not condition_satisfied:
          if conditions == 'pre':
            if count == 0:
              object_states = {
                'objectType': object_type,
                'stateChange': condition2statetype(property_),
                property_: val
              }

              env._event = env.controller.step(
                action='SetObjectStates',
                SetObjectStates=object_states)

              if not env.event.metadata['lastActionSuccess']:
                raise RuntimeError(env.event.metadata['errorMessage'])
            else:
              raise RuntimeError(f'Failed precondition: {cond}')
        else:
          done[i] = True

      elif len(cond) == 4:
        # Check that o1 cond o2 is true
        cond, o1, o2, val = cond

        o1_objs = env.objects_of_type(o1)
        o2_objs = env.objects_of_type(o2)

        if cond == 'on':
          condition_satisfied = any(info for o, info in o1_objs.items() if len(set(o2_objs.keys()).intersection(set(info['parentReceptacles']))) > 0)

          if condition_satisfied:
            done[i] = True

    # ----------------------------------------------------
    # Check preconditions again after objects are updated
    # ----------------------------------------------------
    if count == 1: return
    if conditions == 'pre': # check it one more time
      self.check_task_preconditions(env, 1)
    else:
      return all(done)

  def check_task_objects_exist(self, env):
    """Check that all objects necessary for task are present
    """
    intersection = set(env.objects_by_type_dict.keys()).intersection(set(self.objects))

    if len(intersection) != len(self.objects):
      print(f"Invalid Task: {self.__str__()}")

    return False

if __name__ == "__main__":
  task = Task('place_apple_on_diningtable.json')