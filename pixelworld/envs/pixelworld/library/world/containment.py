'''
    CONTAINMENT
    
    the agent must place all of the blocks in the box.
    
    Reward
    ------
    step: -1
    block moved into box: 100
    block moved out of box: -100
    
    Success
    -------
    all blocks in the box
    
    Parameters
    ----------
    colors : bool
        True to color code the objects
    box_type : string
        one of the following:
            'agent': box is the agent
            'moveable': box can be moved
            'immoveable': box position is fixed
    box_has_lid : bool
        True if the box has an openable/closeable lid
    num_blocks : int
        the number of blocks to include
'''
from copy import deepcopy

import numpy as np

from pixelworld.envs import pixelworld as px
from pixelworld.envs.pixelworld import randomizers
from pixelworld.envs.pixelworld.utils import switch, to_iterable
from ..helpers import h, L, LC

def show_goal_states(n=10, goals=['all_contained', 'almost_contained']):
    for _ in xrange(n):
        world = ContainmentWorld()
        
        for name in goals:
            goal_world = deepcopy(world)
            
            for goal in goal_world.goals:
                if goal.name == name: break
            if goal.name != name: raise ValueError('no goal named "%s"' % (name))
            
            goal.achieve()
            goal_world.render()


class ContainedObjectAttribute( px.core.BooleanObjectAttribute,
                                px.core.DerivedObjectAttribute,
                                px.core.ChangeTrackingObjectAttribute):
    """indicates whether a block is contained within the box"""
    _depends_on = ['position']
    
    #so we can detect position changes
    _step_before = 'position'
    
    @property
    def box(self):
        return self.world.objects.get(name='box')
    
    def _get_data(self, idx):
        box_extent = self.box.extent
        
        p = self._other_attr['position'].get(idx)
        
        tl = p >= box_extent[:2] + 1
        br = p <= box_extent[2:] - 1
        
        return np.all(np.hstack((tl, br)), axis=-1)


class BlockObject(px.objects.BasicObject):
    _attributes = ['contained']
    
    _defaults = {
        'color': 3,
    }


class LiddedObjectAttribute(px.core.AbilityObjectAttribute):
    _actions = ['OPEN', 'CLOSE']
    
    def _execute_action(self, obj, t, dt, agent_id, action):
        if action == 'OPEN':
            obj.lid_state = False
        elif action == 'CLOSE':
            obj.lid_state = True


class LidStateObjectAttribute(  px.core.BooleanObjectAttribute,
                                px.core.DerivedObjectAttribute):
    """ lid_state == False => open
        lid_state == True => closed
    """
    def _get_data_object(self, obj):
        return bool(obj.sprite)
    
    def _set_data_object(self, obj, value):
        obj.sprite = int(value)


class ContainmentSelfObject(px.objects.SelfObject):
    _attributes = ['grips']


class ContainmentGoal(px.core.Goal):
    """base class for containment-related goals"""
    contained = None
    
    #maximum number of time to try something before failing
    _max_tries = 100
    
    @property
    def box(self):
        return self.world.objects.get(name='box')
    
    @property
    def blocks(self):
        return self.contained.objects
    
    @property
    def agent(self):
        return self.world.objects.get(name='containment_self') or self.box
    
    def prepare(self):
        self.contained = self.world.get_object_attribute('contained')
    
    def _get_random_position(self, extent=None, perimeter=False):
        """get a random position within the visible world
        
        Parameters
        ----------
        extent : ndarray, optional
            the bounds (exclusive) of the random position
        perimeter : bool, optional
            True to only return values on the perimeter of the specified extent
        
        Returns
        -------
        y : int
            the y-position
        x : int
            the x-position
        """
        if extent is None:
            extent = np.array([0, 0, self.world.height - 1, self.world.width - 1])
        
        if perimeter:
            side = self.rng.randint(4)
            
            y_random = x_random = False
            if side == 0:  # left
                y_random = True
                x = extent[1]
            elif side == 1: # top
                y = extent[0]
                x_random = True
            elif side == 2:  # right
                y_random = True
                x = extent[3]
            elif side == 3:  # bottom
                y = extent[2]
                x_random = True
            else:
                raise ValueError('invalid side')
            
            if y_random: y = self.rng.randint(extent[0], extent[2] + 1)
            if x_random: x = self.rng.randint(extent[1], extent[3] + 1)
        else:
            y = self.rng.randint(extent[0] + 1, extent[2])
            x = self.rng.randint(extent[1] + 1, extent[3])
        
        return y, x
    
    def _get_next_inside_position(self, box_extent):
        """find the next position that is unoccupied and inside the box
        
        Parameters
        ----------
        box_extent : ndarray
            the box extent
        
        Returns
        -------
        y : Number
            the new y-position
        x : Number
            the new x-position
        success : bool
            True if the position is valid
        """
        num_tries = 0
        
        success = False
        while not success:
            num_tries += 1
            
            if num_tries > self._max_tries: return None, None, False
            
            y, x = self._get_random_position(extent=box_extent)
            
            success = self._is_in_box(y, x, box_extent) and \
                      not self._is_occupied(y, x)
        
        return y, x, True
    
    def _get_next_outside_position(self, box_extent):
        """find the next position that is unoccupied and outside the box
        
        Parameters
        ----------
        box_extent : ndarray
            the box extent
        
        Returns
        -------
        y : Number
            the new y-position
        x : Number
            the new x-position
        success : bool
            True if the position is valid
        """
        num_tries = 0
        
        success = False
        while not success:
            num_tries += 1
            
            if num_tries > self._max_tries: return None, None, False
            
            y, x = self._get_random_position()
            
            success = not self._is_in_box(y, x, box_extent) and \
                      not self._is_occupied(y, x)
        
        return y, x, True
    
    def _get_next_adjacent_position(self, box_extent):
        """find the next position that is adjacent to the box
        
        Parameters
        ----------
        box_extent : ndarray
            the box extent
        
        Returns
        -------
        y : Number
            the new y-position
        x : Number
            the new x-position
        success : bool
            True if the position is valid
        """
        #determine the box perimeter
        perimeter_extent = box_extent
        perimeter_extent[0] -= 1
        perimeter_extent[1] -= 1
        perimeter_extent[2] += 1
        perimeter_extent[3] += 1
        if not self.box.lidded or not self.box.lid_state:  # box lid positions are also an option
            perimeter_extent[0] += 1
        
        num_tries = 0
        
        success = False
        while not success:
            num_tries += 1
            
            if num_tries > self._max_tries: return None, None, False
            
            y, x = self._get_random_position(extent=perimeter_extent, perimeter=True)
            
            success = not self._is_occupied(y, x)
        
        return y, x, True
    
    def _is_in_box(self, y, x, box_extent):
        """test whether a position is within the box"""
        return  (box_extent[0] < y < box_extent[2]) and \
                (box_extent[1] < x < box_extent[3])
    
    def _is_occupied(self, y, x):
        """test whether a position is occupied"""
        return len(self.world.objects.find(object_type='SIMPLE', position=(y, x))) > 0


class AllContainedGoal(ContainmentGoal):
    """True if all blocks are contained in the box"""
    _exclusive = True
    
    def _is_achieved(self):
        all_contained = np.all(self.contained())
        
        return all_contained and (not self.box.lidded or self.box.lid_state)
    
    def _achieve(self):
        """move all the blocks into the box. assumes a rectangular box."""
        box_extent = self.box.extent
        
        #make sure the non-box agent is out of the way
        if self.agent.name == 'containment_self':
            y_agent, x_agent, success = self._get_next_outside_position(box_extent)
            
            if not success: return False
            
            self.agent.position = (y_agent, x_agent)
        
        #box each block. this should really be done via step(), but that's too
        #difficult for now.
        for block in self.blocks:
            if not block.contained:
                y, x, success = self._get_next_inside_position(box_extent)
                
                if not success: return False
                
                block.position = (y, x)
        
        #close the lid
        if self.box.lidded and not self.box.lid_state:
            self.world.step('CLOSE')
        
        return True


class NoneContainedGoal(ContainmentGoal):
    """True if no blocks are contained in the box"""
    _exclusive = True
    
    _active = False
    
    def _is_achieved(self):
        any_contained = np.any(self.contained())
        
        return not any_contained or (self.box.lidded and not self.box.lid_state)
    
    def _achieve(self):
        """move all the blocks out of the box. assumes a rectangular box."""
        box_extent = self.box.extent
        
        #move the blocks out of the box. this should really be done via step(),
        #but that's too difficult for now.
        for block in self.blocks:
            if block.contained:
                y, x, success = self._get_next_outside_position(box_extent)
                
                if not success: return False
                
                block.position = (y, x)
        
        #open the lid
        if self.box.lidded and self.box.lid_state:
            self.world.step('OPEN')
        
        return True


class AlmostContainedGoal(AllContainedGoal):
    """True if all but one block is contained"""
    _exclusive = True
    
    _active = False
    
    def _is_achieved(self):
        num_contained = np.sum(self.contained())
        
        return num_contained == len(self.blocks) - 1
    
    def _achieve(self):
        """move all blocks into the block, then move one block adjacent to the
        box"""
        success = super(AlmostContainedGoal, self)._achieve()
        
        if success:
            box_extent = self.box.extent
            
            y, x, success = self._get_next_adjacent_position(box_extent)
            
            if not success: return False
            
            self.blocks[0].position = (y, x)
        
        #open the lid
        if self.box.lidded and self.box.lid_state:
            self.world.step('OPEN')
            
        return success


class BoxHeightWorldAttribute(px.core.IntegerWorldAttribute):
    _default_value = 5


class BoxWidthWorldAttribute(px.core.IntegerWorldAttribute):
    _default_value = 5


class BoxHasLidWorldAttribute(px.core.BooleanWorldAttribute):
    _default_value = False


class BoxTypeWorldAttribute(px.core.StringWorldAttribute):
    _default_value = 'immoveable'


class BoxHeightVariant(px.core.IntegerVariant, px.core.WorldAttributeVariant):
    _min_state = 3
    _max_state = 7


class BoxWidthVariant(px.core.IntegerVariant, px.core.WorldAttributeVariant):
    _min_state = 3
    _max_state = 7


class BoxHasLidVariant(px.core.BooleanVariant, px.core.WorldAttributeVariant):
    pass


class BoxTypeVariant(px.core.WorldAttributeVariant):
    """determines whether the box is the agent or a moveable or immoveable
    Object."""
    #agent is twice so we get an equal split between agent/object
    _states = ['agent', 'agent', 'moveable', 'immoveable']


class BlockCountVariant(px.core.ObjectCountVariant):
    """determines the number of blocks in the world"""
    
    _object_type = 'block'
    
    #max_state defaults to the number of blocks that can fit in the box
    _min_state = 1
    _max_state = None
    
    @property
    def max_state(self):
        """unless it has been overridden, max block count is the number of
        blocks that can fit in the box"""
        max_state = super(BlockCountVariant, self).max_state
        
        if max_state is None:
            max_state = self.get_max_state_given_box_dimensions(
                            self.world.box_height, self.world.box_width,
                            )
        
        return max_state
    
    @max_state.setter
    def max_state(self, max_state):
        self.states = None
        self._max_state = max_state
    
    def get_conditional_states(self, variant_states):
        """overridden since max_state might depend on the current values of
        box_height and box_width"""
        if 'box_height' in variant_states or 'box_width' in variant_states:
            min_state = self.min_state
            max_state = super(BlockCountVariant, self).max_state
            
            if max_state is None:
                max_state = self.get_max_state_given_box_dimensions(
                                variant_states.get('box_height', self.world.box_height),
                                variant_states.get('box_width', self.world.box_width),
                                )
            
            return self.get_states_given_range(min_state, max_state)
        else:
            return super(BlockCountVariant, self).get_conditional_states(variant_states)
    
    def get_max_state_given_box_dimensions(self, box_height, box_width):
        """calculate the default max_state value, given the dimensions of the
        box. this will be the maximum number of blocks that could fit in the
        given box."""
        return int(box_height * box_width - 2 * (box_height + box_width - 2))


class ContainmentRandomizer(randomizers.RandomPositionsRandomizer):
    
    def _get_object_specs(self):
        """generate the new agent Objects based on the new Variant states"""
        object_specs = super(ContainmentRandomizer, self)._get_object_specs()
        
        agent_color = 2
        
        #box specs
        box_type = self.world.box_type
        box_object = switch(box_type,
                        agent='self_sprite',
                        moveable='complex_sprite',
                        immoveable='immoveable_sprite',
                        )
        box_color = agent_color if box_type == 'agent' else 1
        box_sprites = self._get_box_sprites()
        object_specs += [[box_object, {
            'name': 'box',
            'sprites': box_sprites,
            'color': box_color,
            'lidded': self.world.box_has_lid,
            'lid_state': False,
            }]]
        
        #moveable agent
        if box_type != 'agent':
            object_specs += [['containment_self', {'color': agent_color}]]
        
        return object_specs
    
    def _get_box_sprites(self):
        height = self.world.box_height
        width = self.world.box_width
        
        return [
            self._get_box_sprite(height, width, True),
            self._get_box_sprite(height, width, False),
        ]
    
    def _get_box_sprite(self, height, width, opened):
        box_char = 'X'
        top_char = '0' if opened else box_char
        
        box = []
        
        #box top
        box.append(top_char * width)
        
        #box sides
        box.extend((height - 2) * [box_char + ' ' * (width - 2) + box_char])
        
        #box bottom
        box.append(box_char * width)
        
        return h.sprite.from_string('\n'.join(box))


class ContainmentJudge(px.core.Judge):
    """a judge that rewards contained blocks and penalizes inefficiency"""
    contained = None
    
    def prepare(self):
        self.contained = self.world.get_object_attribute('contained')
        
    def _calculate_reward(self, goals, events):
        return 100 * np.sum(self.contained.change(step=True))


class ContainmentWorld(px.core.PixelWorld):
    
    def __init__(self, colors=None, box_type=None, box_dim_range=None,
        box_has_lid=None, block_count_range=None, goals=None,
        judge='containment', variants=None, randomizer='containment', **kwargs):
        """
        Parameters
        ----------
        colors : bool
            True to color code the objects
        box_type : list[string]
            a list of the following:
                'agent': box is the agent
                'moveable': box can be moved
                'immoveable': box position is fixed
        box_dim_range : list[int, int, int, int]
            a list specifying the range of possible box dimensionas as:
            [min_height, min_width, max_height, max_width]
        box_has_lid : bool
            True if the box has an openable/closeable lid
        block_count_range : list[int, int]
            a list specifying the minimum and maximum number of blocks
        goals : list, optional
            see PixelWorld
        judge : Judge, optional
            see PixelWorld
        variants : list, optional
            see PixelWorld
        randomizer : Randomizer, optional
            see PixelWorld
        **kwargs
            extra arguments for PixelWorld
        """
        if box_dim_range is None: box_dim_range = [None] * 4
        if block_count_range is None: block_count_range = [1, None]
        if box_has_lid is not None: box_has_lid = to_iterable(box_has_lid)
        
        #goal specs
        if goals is None:
            goals = ['all_contained', 'none_contained', 'almost_contained']
        
        #variant specs
        if variants is None:
            show_colors = ['show_colors', {
                            'states': colors,
                            }]
            box_height = ['box_height', {
                            'min_state': box_dim_range[0],
                            'max_state': box_dim_range[2],
                            }]
            box_width = ['box_width', {
                            'min_state': box_dim_range[1],
                            'max_state': box_dim_range[3],
                            }]
            box_type = ['box_type', {
                            'states': box_type,
                            }]
            box_has_lid = ['box_has_lid', {
                            'states': box_has_lid,
                            }]
            block_count = ['block_count', {
                            'min_state': block_count_range[0],
                            'max_state': block_count_range[1],
                            }]
            
            variants = [show_colors, box_height, box_width, box_type,
                        box_has_lid, block_count]
        
        super(ContainmentWorld, self).__init__(goals=goals, judge=judge,
                variants=variants, randomizer=randomizer, **kwargs)


world = ContainmentWorld
