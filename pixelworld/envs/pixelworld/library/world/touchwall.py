'''
    TOUCHWALL
    
    the agent must touch the frame
    
    Reward
    ------
    step: -1
    agent is touching the frame: 1000
    
    Success
    -------
    agent is touching the frame
    
    Parameters
    ----------
    colors : bool
        True to color code the objects (default: True)
    sides : list
        a list of frame sides to include (from ['left', 'top', 'right', 'bottom'])
'''
import numpy as np

from pixelworld.envs import pixelworld as px
from pixelworld.envs.pixelworld.utils import roundup


class TouchingObjectAttribute(  px.core.BooleanObjectAttribute,
                                px.core.DerivedObjectAttribute):
    """indicates whether the agent is touching a wall"""
    _depends_on = ['position']
    
    #so we can detect position changes
    _step_before = 'position'
    
    frame = None
    
    def prepare(self):
        self.frame = self.world.objects['frame']
    
    def _get_data_object(self, obj):
        p_wall = self._other_attr['position'].get(self.frame.children)
        
        p = obj.position
        
        p_diff = np.reshape(roundup(np.abs(p_wall - p)), (-1, 2))
        
        return np.any(np.sum(p_diff, axis=1) == 1)


class TouchSelfObject(px.objects.SelfObject):
    _attributes = ['touching']


class TouchingGoal(px.core.Goal):
    """True when any touching attribute is True"""
    touching = None
    
    reward = 1000
    
    def prepare(self):
        self.touching = self.world.object_attributes['touching']
    
    def _is_achieved(self):
        return np.any(self.touching())


class TouchWallWorld(px.core.PixelWorld):
    
    def __init__(self, frame=True, colors=True, sides=None, objects=None,
                    goals=None, **kwargs):
        """
        Parameters
        ----------
        frame : bool, optional
            True to include a frame object that contains the agent
        colors : bool, optional
            True to color code the objects
        sides : list, optional
            a list of frame sides to include (from ['left', 'top', 'right',
            'bottom'])
        objects : list, optional
            see PixelWorld
        goals : list, optional
            see PixelWorld
        **kwargs
            extra arguments for PixelWorld
        """
        if objects is None:
            objects = []
            
            #frame
            objects += [['frame', {'sides': sides}]]
            
            #moveable agent
            agent_color = 2 if colors else 1
            objects += [['touch_self', {'color': agent_color}]]
        
        if goals is None:
            goals = ['touching']
        
        super(TouchWallWorld, self).__init__(objects=objects, goals=goals, **kwargs)

world = TouchWallWorld
randomizer = 'random_positions'
