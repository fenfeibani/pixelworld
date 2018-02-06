'''
    basic set of Agents for PixelWorld
'''
from copy import copy

import core
from copy import deepcopy
from utils import KeyboardController


class RandomAgent(core.Agent):
    """random policy Agent"""
    _name = 'random'
    
    def _get_action(self, obs, tools):
        return tools['rng'].choice(tools['actions'])


class HumanAgent(core.Agent):
    """Agent controlled by a human via a keyboard interface. may or may not
    learn."""
    _name = 'human'
    
    _controller = None
    
    #the rate at which the world should advance, in Hz
    _rate = 0
    
    #will store the previous action set, so we can tell when they change
    _previous_actions = None
    
    #new attributes to include in States
    _state_attributes = ['_rate']
    
    def __init__(self, world, rate=None, **kwargs):
        """
        Parameters
        ----------
        world : PixelWorld
            the host world
        rate : Number
            the number of times to step() the world each second. if this is
            specified and the human doesn't select an action, then the default
            action (from core.Agent) is selected.
        """
        if rate is not None and rate is not False:
            self._rate = rate
        
        super(HumanAgent, self).__init__(world, **kwargs)
    
    @property
    def controller(self):
        """the keyboard controller"""
        if self._controller is None:
            self._set_controller()
        
        return self._controller
    
    def _set_controller(self):
        """set up the keyboard controller"""
        #make sure the window exists
        self.world.render(capture=False)
        window = self._world._viewer.window
        
        #set up the keyboard interface
        actions = self._world.actions
        self._controller = KeyboardController(window=window, actions=actions)
        
        #add a QUIT action
        self._controller.register_key('ESCAPE', 'QUIT')
        self._controller.register_action('QUIT')
    
    def _get_action(self, obs, tools):
        """overrides core.Agent.  get an action from they keyboard."""
        actions = self.world.actions
        
        #make sure we have all the actions mapped
        self.controller.register_actions(actions)
        
        #display the key legend
        if actions != self._previous_actions:
            self.controller.print_legend()
        
        self._previous_actions = actions
        
        #get the current action
        timeout = float('inf') if self._rate == 0 else 1./self._rate
        action = self.controller.get_action(timeout=timeout,
                    default=self._default_action)
        
        #return it (or None if the user wants to quit)
        return None if action == 'QUIT' else action
    
    def __deepcopy__(self, memo):
        """deepcopy everything except the controller, since it involves
        ctypes"""
        #make a new HumanAgent
        agent = self.__class__.__new__(self.__class__)
        memo[id(self)] = agent
        
        #deepcopy everything except the controller
        for attr,value in self.__dict__.iteritems():
            if attr != '_controller':
                value_id = id(value)
                new_value = memo[value_id] if value_id in memo else deepcopy(value, memo)
                agent.__dict__[attr] = new_value
        
        return agent

    def __getstate__(self):
        """Custom pickling behavior: omit agent._controller, since it is not
        picklable."""
        d = copy(self.__dict__)
        try:
            del d['_controller']
        except KeyError:
            pass
        return d
