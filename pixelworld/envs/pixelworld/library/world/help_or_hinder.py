import numpy as np

from pixelworld.envs import pixelworld as px
from pixelworld.envs.pixelworld.utils import PointMath as pmath, roundup


class GotFruitEvent(px.core.Event):
    _parameters = ['idx', 'type']
    
    _rewards = {
        'good': 100,
        'bad': -100,
    }
    
    def __init__(self, *args, **kwargs):
        super(GotFruitEvent, self).__init__(*args, **kwargs)
        
        self._reward = self._rewards[self.type]
    
    def _get_description(self):
        obj = self._world._objects[self.idx]
        return 'got %s fruit (%s)' % (self.type, obj.unique_name)


class GatherableObjectAttribute(    px.core.BooleanObjectAttribute,
                                    px.core.SteppingObjectAttribute):
    _default_value = True
    
    _step_after = ['pushes']
    
    agent = None
    
    def prepare(self):
        self.agent = self.world.objects.get(name='self')
    
    def _step_object(self, obj, t, dt, agent_id, action):
        if self.get(obj) and np.all(obj.state_index == self.agent.state_index):
            evt = GotFruitEvent(self.world, idx=obj.id, type=obj._fruit_type)
            obj.remove()


class FollowsObjectAttribute(   px.core.BooleanObjectAttribute,
                                px.core.SteppingObjectAttribute):
    _default_value = True
    
    _step_after = ['pushes']
    
    agent = None
    
    def prepare(self):
        self.agent = self.world.objects.get(name='self')
    
    def _step_object(self, obj, t, dt, agent_id, action):
        if self.get(obj):
            position = self.world.object_attributes['position']
            
            p_obj = obj.position
            p_agent = self.agent.position
            
            target_fruit = 'good_fruit' if obj.agent_type == 0 else 'bad_fruit'
            fruit = self.world.objects.find(name=target_fruit)
            
            if len(fruit) == 0:
                return
            
            p_fruit = position.get(fruit)
            
            d = pmath.magnitude2(p_agent - p_fruit)
            
            idx_closest = np.where(d == np.min(d))[0][0]
            
            p_fruit = p_fruit[idx_closest]
            p_target = roundup(p_fruit + pmath.direction(p_agent - p_fruit))
            
            p_diff = p_obj - p_target
            mag = pmath.magnitude(p_diff)
            p_diff /= max(mag/2, 2)
            
            if pmath.magnitude(p_diff) < 1:
                p_next = p_target
            else:
                p_next = obj.position - roundup(p_diff)
            
            idx_next = position._position_to_index(p_next, to_tuple=False)
            
            if not np.all(idx_next == self.agent.state_index):
                obj.position = p_next


class AgentTypeObjectAttribute(px.core.IntegerObjectAttribute):
    
    _depends_on = ['color']
    
    def _default_value(self, obj):
        return self.rng.randint(2)
    
    def _set_data(self, idx, value):
        super(AgentTypeObjectAttribute, self)._set_data(idx, value)
        
        color = self._other_attr['color']
        set_color_idx = np.where(color.get(idx) != 1)[0]
        if len(set_color_idx) > 0:
            color.set(idx[set_color_idx], value[set_color_idx] + 5)


class FruitObject(px.objects.BasicObject):
    _attributes = ['gatherable']
    
    _defaults = {'zorder': -np.inf}
    
    def prepare(self):
        self.mass = 0


class GoodFruitObject(FruitObject):
    _fruit_type = 'good'
    _defaults = {'color': 3}


class BadFruitObject(FruitObject):
    _fruit_type = 'bad'
    _defaults = {'color': 4}


class OtherAgentObject(px.objects.UnpushableObject):
    _attributes = ['agent_type', 'follows']


class FruitsGatheredGoal(px.core.Goal):
    """True when all fruits are gathered"""
    
    @property
    def num_good_fruits(self):
        return len(self.world.objects.find(name='good_fruit'))
    
    def _is_achieved(self):
        return self.num_good_fruits == 0


class HelpOrHinderWorld(px.core.PixelWorld):
    
    def __init__(self, stage=0, colors=True, num_fruit=5, objects=None, goals=None,
                    **kwargs):
        """
        Parameters
        ----------
        stage : int, optional
            the curriculum stage
        colors : bool, optional
            True to color code the objects
        num_blocks : int, optional
            the number of fruit of each type to include
        objects : list, optional
            see PixelWorld
        goals : list, optional
            see PixelWorld
        **kwargs
            extra arguments for PixelWorld
        """
        assert 0 <= stage <= 1, 'invalid stage'
        
        if objects is None:
            objects = []
            
            #blocks
            fruit_color = None# if colors else 1
            objects += num_fruit * [['good_fruit', {'color': fruit_color}]]
            objects += num_fruit * [['bad_fruit', {'color': fruit_color}]]
            
            #moveable agent
            agent_color = 2 if colors else 1
            objects += [['self', {'color': agent_color}]]
            
            #helper or hinderer
            if stage == 1:
                other_color = 5 if colors else 1
                objects += [['other_agent', {'color': other_color}]]
        
        if goals is None:
            goals = ['fruits_gathered']
        
        super(HelpOrHinderWorld, self).__init__(objects=objects, goals=goals, **kwargs)


world = HelpOrHinderWorld
