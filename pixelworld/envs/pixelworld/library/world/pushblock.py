'''
    PUSHBLOCK
    
    the agent must push the block onto the goal square
    
    Reward
    ------
    step: -1
    block moves closer to goal: 2, if warmercolder==True
    block is on top of goal: 1000
    
    Success
    -------
    block is on top of goal
    
    Parameters
    ----------
    frame : bool
        True to include a frame object that contains the agent (default: True)
    colors : bool
        True to color code the objects (default: True)
    warmercolder : bool
        True if the reward should indicate whether the block is moving toward
        or away from the goal (default: False)
    num_blocks : int
        the number of blocks to include (default: 1)
'''
import numpy as np

from pixelworld.envs import pixelworld as px
from pixelworld.envs.pixelworld.utils import roundup, PointMath as pmath

class DistanceObjectAttribute(  px.core.FloatObjectAttribute,
                                px.core.ChangeTrackingObjectAttribute,
                                px.core.DerivedObjectAttribute):
    """distance between the block and the goal"""
    _depends_on = ['position']
    
    _step_before = ['position']
    
    agent = None
    
    def prepare(self):
        self.goal = self.world.objects['goal']
    
    def _get_data(self, idx):
        p_goal = self.goal.position
        
        p = self._other_attr['position'].get(idx)
        
        rv = pmath.magnitude(p - p_goal)
        return rv


class WarmerObjectAttribute(    px.core.BooleanObjectAttribute,
                                px.core.DerivedObjectAttribute):
    """indicates whether the block moved closer to the goal"""
    _depends_on = ['distance']
    
    def _get_data(self, idx):
        return np.any(self._other_attr['distance'].change(step=True) < 0)


class BlockObject(px.objects.BasicObject):
    _attributes = ['distance']


class GoalObject(px.objects.BasicObject):
    _defaults = {'mass': 0, 'zorder': -1}


class AgentObject(px.objects.SelfObject):
    _attributes = ['warmer']


class PushBlockGoal(px.core.Goal):
    """True when some block is on top of the goal"""
    touching = None
    
    reward = 1000
    
    def prepare(self):
        self.distance = self.world.object_attributes['distance']
    
    def _is_achieved(self):
        return np.any(self.distance() <= 0.5)


class PushBlockJudge(px.core.Judge):
    """a judge that rewards the block being on the goal, penalizes inefficiency,
    and optionally rewards moving a block closer to the goal"""
    agent = None
    
    warmercolder = None
    
    def __init__(self, world, warmercolder=False, **kwargs):
        self.warmercolder = warmercolder
        
        super(PushBlockJudge, self).__init__(world, **kwargs)
    
    def prepare(self):
        self.agent = self.world.objects['agent']
        
    def _calculate_reward(self, goals, events):
        reward = 0
        
        #for moving closer to a block
        if self.warmercolder and self.agent.warmer:
            reward += 2
        
        return reward


class PushBlockWorld(px.core.PixelWorld):
    
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

            #goal
            goal_color = 4 if colors else 1
            objects += [['goal', {'color': goal_color}]]
            
            #moveable agent
            agent_color = 2 if colors else 1
            objects += [['agent', {'color': agent_color}]]

        if goals is None:
            goals = ['push_block']
        
        if judge is None:
            judge = ['push_block', {'warmercolder': warmercolder}]
        
        super(PushBlockWorld, self).__init__(objects=objects, goals=goals,
                                             judge=judge, **kwargs)

world = PushBlockWorld
randomizer = 'random_positions'
