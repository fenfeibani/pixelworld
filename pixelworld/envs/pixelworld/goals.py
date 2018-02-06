'''
    basic set of Goals for PixelWorld
'''
import core
from utils import is_iterable


class AnythingGoal(core.Goal):
    """everybody wins!"""
    
    def _is_achieved(self):
        return True
    
    def _achieve(self):
        return True


class ActionGoal(core.Goal):
    """achieved when the agent performs one of the specified actions."""
    
    #a list of actions that achieve the goal
    _actions = None
    
    def __init__(self, world, actions=None, **kwargs):
        if actions is not None: self.actions = actions
        
        super(ActionGoal, self).__init__(world, **kwargs)
    
    @property
    def actions(self):
        return self._actions
    
    @actions.setter
    def actions(self, actions):
        if not is_iterable(actions):
            raise TypeError('actions must be list-like')
        
        self._actions = list(actions)
    
    @property
    def action(self):
        if len(self.actions) != 1: raise RuntimeError('a single action is not defined')
        return self.actions[0]
    
    @action.setter
    def action(self, action):
        self.actions = [action]
    
    def _is_achieved(self):
        """True if any of the specified actions was the last to be performed"""
        for item in self.world.history[::-1]:
            if item['type'] == 'action' and len(item['action']) > 0:
                return any(action in self.actions for action in item['action'])
        
        return False
    
    def _achieve(self):
        """perform one of the actions"""
        if len(self.actions) == 0:
            return 'no actions are defined'
        else:
            self.world.step(self.rng.choice(self.actions))
            return True
