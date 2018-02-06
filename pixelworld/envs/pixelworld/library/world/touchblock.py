'''
    TOUCHBLOCK
    
    the agent must touch the block
    
    Reward
    ------
    step: -1
    agent moves closer to block: 2, if warmercolder==True
    block is adjacent to agent: 1000
    
    Success
    -------
    block is adjacent to agent
    
    Parameters
    ----------
    frame : bool
        True to include a frame object that contains the agent (default: True)
    colors : bool
        True to color code the objects (default: True)
    warmercolder : bool
        True if the reward should indicate whether the agent is moving toward or
        away from the block (default: False)
    num_blocks : int
        the number of blocks to include (default: 1)
'''
import numpy as np

from pixelworld.envs import pixelworld as px
from pixelworld.envs.pixelworld.utils import roundup, PointMath as pmath

class DistanceObjectAttribute(  px.core.FloatObjectAttribute,
                                px.core.ChangeTrackingObjectAttribute,
                                px.core.DerivedObjectAttribute):
    """distance between the block and the agent"""
    _depends_on = ['position']
    
    _step_before = ['position']
    
    agent = None
    
    def prepare(self):
        self.agent = self.world.objects['agent']
    
    def _get_data(self, idx):
        p_agent = self.agent.position
        
        p = self._other_attr['position'].get(idx)
        
        return pmath.magnitude(p - p_agent)


class TouchingObjectAttribute(  px.core.BooleanObjectAttribute,
                                px.core.DerivedObjectAttribute):
    """indicates whether a block is touching the agent"""
    _depends_on = ['position']
    
    agent = None

    _step_after = ['pushes', 'position']
    
    def prepare(self):
        self.agent = self.world.objects['agent']
    
    def _get_data(self, idx):
        p_agent = self.agent.position
        
        p = self._other_attr['position'].get(idx)
        
        p_diff = np.reshape(roundup(np.abs(p_agent - p)), (-1, 2))
        
        return np.sum(p_diff, axis=1) <= 1


class WarmerObjectAttribute(    px.core.BooleanObjectAttribute,
                                px.core.DerivedObjectAttribute):
    """indicates whether the agent moved closer to a block"""
    _depends_on = ['distance']
    
    def _get_data(self, idx):
        return np.any(self._other_attr['distance'].change(step=True) < 0)


class BlockObject(px.objects.BasicObject):
    _attributes = ['distance', 'touching']


class AgentObject(px.objects.SelfObject):
    _attributes = ['warmer']


class TouchingGoal(px.core.Goal):
    """True when any touching attribute is True"""
    touching = None
    
    reward = 1000
    
    def prepare(self):
        self.touching = self.world.get_object_attribute('touching')
    
    def _is_achieved(self):
        return np.any(self.touching())


class TouchBlockJudge(px.core.Judge):
    """a judge that rewards touching a block, penalizes inefficiency, and
    optionally rewards moving closer to a block"""
    agent = None
    
    warmercolder = None
    
    def __init__(self, world, warmercolder=False, **kwargs):
        self.warmercolder = warmercolder
        
        super(TouchBlockJudge, self).__init__(world, **kwargs)
    
    def prepare(self):
        self.agent = self.world.objects['agent']
        
    def _calculate_reward(self, goals, events):
        reward = 0
        
        #for moving closer to a block
        if self.warmercolder and self.agent.warmer:
            reward += 2
        
        return reward


class TouchBlockWorld(px.core.PixelWorld):
    
    def __init__(self, frame=True, colors=True, warmercolder=False,
                    num_blocks=1, objects=None, goals=None, judge=None, **kwargs):
        """
        Parameters
        ----------
        frame : bool, optional
            True to include a frame object that contains the agent
        colors : bool, optional
            True to color code the objects
        warmercolder : bool, optional
            True if the Judge should reward moving closer to a block
        num_blocks : int, optional
            the number of blocks to include
        objects : list, optional
            see PixelWorld
        goals : list, optional
            see PixelWorld
        judge : Judge, optional
            see PixelWorld
        **kwargs
            extra arguments for PixelWorld
        """
        if objects is None:
            objects = []
            
            #frame
            if frame:
                objects += ['frame']
            
            #blocks
            block_color = 3 if colors else 1
            objects += num_blocks * [['block', {'color': block_color}]]
            
            #moveable agent
            agent_color = 2 if colors else 1
            objects += [['agent', {'color': agent_color}]]

        if goals is None:
            goals = ['touching']
        
        if judge is None:
            judge = ['touch_block', {'warmercolder': warmercolder}]
        
        super(TouchBlockWorld, self).__init__(objects=objects, goals=goals,
                judge=judge, **kwargs)

world = TouchBlockWorld
randomizer = 'random_positions'
