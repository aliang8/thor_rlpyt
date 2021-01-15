ACTIONS = {
    "interact_actions": [
      'ToggleObjectOn',
      'ToggleObjectOff',
      'PickupObject',
      'PutObject',
      'CloseObject',
      'OpenObject',
      'SliceObject',
      'FillObjectWithLiquid',
      'CleanObject',
      'CookObject',
      'BreakObject',
      'DirtyObject',
      'EmptyLiquidFromObject',
      'UseUpObject'
    ],
    "movement_actions": [
        'MoveAhead',
        'MoveBack',
        'MoveRight',
        'MoveLeft',
      ],
    "view_actions": [
        'LookUp',
        'LookDown',
        'RotateRight',
        'RotateLeft',
      ],
    "object_shift_actions": ["PreviousObject", "NextObject"],
    "point_shift_actions": ["PreviousPoint", "NextPoint", "RandomPoint"],
    'goto_action': ['GotoPoint'],
    'eat_action': ["EatPoint"],
    "no_op" : ['NoOp'],
    "close" : ['Close'],
}

def condition2statetype(condition):
  if condition in ['isPickedUp']:
    raise RuntimeError(f"Can't initialize with {condition}")
  return dict(
    isSliced="sliceable",
    isToggled="toggleable",
    isBroken="breakable",
    isFilledWithLiquid="canFillWithLiquid",
    isUsedUp="canBeUsed",
    isCooked="cookable",
    isOpen="openable",
    isDirty="dirtyable",
    )[condition]