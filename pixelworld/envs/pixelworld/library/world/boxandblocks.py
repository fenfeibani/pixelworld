'''
    BOX AND BLOCKS
    
    an environment that uses the following concepts:
        moving (to)
        touching
        pushing (to)
        pulling (to)
        gathering
        containment
        
        (possibly later:
            *strength
            *arranging
            *occlusion)
    
    Reward
    ------
    step: -1
    block moved into box: 100
    block moved out of box: -100
    achieve the active goal: 1000
    
    Goals
    -----
    (* denotes goals that are included by default)
    *move_direction
        move the agent object in the world.goal_direction
    *move_to_corner
        move the agent object to the world.goal_corner
    *move_into_box
        move the simple self object into the box
    *touch_block
        touch any block
    *push_block_to_corner
        push any block to the world.goal_corner
    *push_two_blocks_together
        push the two blocks together (box is hidden)
    *gather_blocks
        gather all blocks into a single touching group
    *put_block_in_box
        put the only block into the box (no lid)
    *all_contained
        put all the blocks into the box and close the lid if the box has one
    none_contained
        move all the blocks out of the box and open the lid if the box has one
    almost_contained
        put all but one of the blocks in the box
    *not_all_contained
        make sure not all of the blocks are in the box
    *pull_out_of_box
        move all of the blocks out of the box (setting up the goal pre-condition
        makes the box skinny and puts the blocks in the box, so that they will
        have to be pulled out)
    
    Parameters
    ----------
    show_colors : bool
        color code the objects (default: random)
    has_frame : bool
        show a frame around the visible world (default: random)
    goal_corner : string
        the goal corner for move_to_corner and push_block_to_corner goals. must
        be one of 'top_left', 'top_right', 'bottom_right', or 'bottom_left'.
        (default: random)
    goal_action : string
        goal action for move_direction goal. must be one of 'LEFT', 'UP',
        'RIGHT', or 'DOWN'. (default: random)
    box_types : list
        allowable box types as a list of the following values:
            'agent': box is the agent (responds to movement actions)
            'moveable': box can be pushed/pulled
            'immoveable': box has fixed position
        (default: ['agent', 'moveable', 'immoveable'])
    min_box_height : int
        minimum box height. this shouldn't be less than 3. (default: 3)
    max_box_height : int
        maximum box height (default: 7)
    min_box_width : int
        minimum box width. this shouldn't be less than 3. (default: 3)
    max_box_width : int
        maximum box width (default: 7)
    box_type : string
        the box type (see box_types)
    box_has_lid : bool
        True if the box should have a lid (default: random)
    box_height : int
        the box height (default: random from [min_box_height, max_box_height])
    box_width : int
        the box width (default: random from [min_box_width, max_box_width])
    agent_color : int
        color of the agent (simple self or box, depending on box.box_type)
        (default: 2 [red])
    block_color : int
        color for blocks (default: 3 [blue])
    box_color : int
        color for the non-agent box (default: 1 [white])
    min_num_blocks : int
        minimum number of blocks to show (default: 1)
    max_num_blocks : int
        maximum number of blocks to show (default: maximum holdable by box)
    num_blocks : int
        number of blocks to show (default: random from
        [min_num_blocks, max_num_blocks])
'''
from copy import copy, deepcopy
from collections import OrderedDict

import numpy as np

from pixelworld.envs import pixelworld as px
from pixelworld.envs.pixelworld.utils import switch, to_iterable, to_vector, \
    cartesian_product, roundup, PointMath as pmath
from ..helpers import h


#---WorldAttributes------------------------------------------------------------#
class HasFrameWorldAttribute(px.core.BooleanWorldAttribute):
    """True if the world's frame should appear. changing this creates/shows or
    hides the frame."""
    
    def _get(self):
        return False if self.world._frame is None else self.world.frame.visible
    
    def _set(self, value):
        if value or self.world._frame is not None:
            self.world.frame.visible = value


class BoxTypesWorldAttribute(px.core.WorldAttribute):
    """allowable box types (see BoxTypeObjectAttribute)"""
    _default_value = ['agent', 'moveable', 'immoveable']


class MinBoxHeightWorldAttribute(px.core.PositiveIntegerWorldAttribute):
    """minimum height of the box"""
    _default_value = 3


class MaxBoxHeightWorldAttribute(px.core.PositiveIntegerWorldAttribute):
    """maximum height of the box"""
    _default_value = 7


class MinBoxWidthWorldAttribute(px.core.PositiveIntegerWorldAttribute):
    """minimum width of the box"""
    _default_value = 3


class MaxBoxWidthWorldAttribute(px.core.PositiveIntegerWorldAttribute):
    """maximum width of the box"""
    _default_value = 7


class BoxTypeWorldAttribute(px.core.BooleanWorldAttribute):
    """maps onto box.box_type"""
    _default_value = ''
    
    def get(self): return self.world.box.box_type
    def set(self, value): self.world.box.box_type = value


class BoxHasLidWorldAttribute(px.core.BooleanWorldAttribute):
    """maps onto box.lidded"""
    _default_value = False
    
    def get(self): return self.world.box.lidded
    def set(self, value): self.world.box.lidded = value


class BoxHeightWorldAttribute(px.core.NonNegativeIntegerWorldAttribute):
    """maps onto box.height"""
    _default_value = 3
    
    def get(self): return self.world.box.height
    def set(self, value): self.world.box.height = value


class BoxWidthWorldAttribute(px.core.NonNegativeIntegerWorldAttribute):
    """maps onto box.width"""
    _default_value = 3
    
    def get(self): return self.world.box.width
    def set(self, value): self.world.box.width = value


class MinNumBlocksWorldAttribute(px.core.NonNegativeIntegerWorldAttribute):
    """minimum number of blocks to include"""
    _default_value = 1


class MaxNumBlocksWorldAttribute(px.core.WorldAttribute):
    """maximum number of blocks to include"""
    
    _default_value = None
    
    def _get(self):
        """if the actual value is None, return the maximum value based on the
        current box size"""
        value = super(MaxNumBlocksWorldAttribute, self)._get()
        
        return self.get_value_from_box() if value is None else value
    
    def get_value_from_box(self):
        """calculate the current value based on the number of blocks that could
        fit in the current box"""
        height = self.world.box.height
        width = self.world.box.width
        
        #area - perimeter
        return int(height * width - 2 * (height + width - 2))


class NumBlocksWorldAttribute(px.core.NonNegativeIntegerWorldAttribute):
    """the current number of blocks in the world. setting this creates and/or
    changes the visibility of existing blocks."""
    
    def _default_value(self, world):
        return world.min_num_blocks
    
    def _get(self):
        return len(self.world.blocks)
    
    def _set(self, value):
        """create blocks and set their visibility as needed"""
        blocks = self.world.blocks
        num_blocks = len(blocks)
        
        if value > num_blocks:  # need to create more blocks
            self.world.create_objects(['block'] * (value - num_blocks))
            blocks = self.world.blocks
        
        #set block visibility
        visible = self.world.object_attributes['visible']
        block_visible = visible.get(blocks)
        num_visible = np.sum(block_visible)
        if value > num_visible:  # need to make some blocks visible
            idx = np.where(~block_visible)[0]
            visible.set(blocks[idx[:value - num_visible]], True)
        elif value < num_visible:  # need to make some blocks invisible
            idx = np.where(block_visible)[0]
            visible.set(blocks[idx[:num_visible - value]], False)


class AgentColorWorldAttribute(px.core.NonNegativeIntegerWorldAttribute):
    """the color for agent objects"""
    _default_value = 2


class BoxColorWorldAttribute(px.core.NonNegativeIntegerWorldAttribute):
    """the box color (when it isn't an agent)"""
    _default_value = 1


class BlockColorWorldAttribute(px.core.NonNegativeIntegerWorldAttribute):
    """the block color"""
    _default_value = 3


class GoalCornerWorldAttribute(px.core.SetWorldAttribute):
    """the corner to move to when MoveToCornerGoal is active"""
    _values = ['top_left', 'top_right', 'bottom_right', 'bottom_left']


class GoalActionWorldAttribute(px.core.SetWorldAttribute):
    """the target action when MoveDirectionGoal is active"""
    _values = ['LEFT', 'UP', 'RIGHT', 'DOWN']


#---Variants-------------------------------------------------------------------#
class HasFrameVariant(  px.core.BooleanVariant,
                        px.core.WorldAttributeVariant):
    """frame visibility"""
    pass


class GoalCornerVariant(px.core.SetWorldAttributeVariant): pass
class GoalActionVariant(px.core.SetWorldAttributeVariant): pass


class BoxTypeVariant(px.core.WorldAttributeVariant):
    """box type (see BoxTypeObjectAttribute)"""
    _states = 'box_types'


class BoxHasLidVariant( px.core.BooleanVariant,
                        px.core.WorldAttributeVariant):
    """whether the box has a lid"""


class BoxHeightVariant( px.core.IntegerVariant,
                        px.core.WorldAttributeVariant):
    """height of the box"""
    _min_state = 'min_box_height'
    _max_state = 'max_box_height'


class BoxWidthVariant(  px.core.IntegerVariant,
                        px.core.WorldAttributeVariant):
    """width of the box"""
    _min_state = 'min_box_width'
    _max_state = 'max_box_width'


class NumBlocksVariant( px.core.IntegerVariant,
                        px.core.WorldAttributeVariant):
    """number of blocks in the scene (see NumBlocksWorldAttribute)"""
    _min_state = 'min_num_blocks'
    _max_state = 'max_num_blocks'


#---Events---------------------------------------------------------------------#
class ContainmentEvent(px.core.Event):
    """occurs when an object moves into or out of the box"""
    #idx: id of object that moved
    #object_name: the name of the object
    #moved_in: True if object moved into the box
    _parameters = ['idx', 'object_name', 'moved_in']
    
    def _get_description(self):
        object_name = self.world.objects[self.idx].unique_name
        direction = 'into' if self.moved_in else 'out of'
        return '%s moved %s the box' % (object_name, direction)


#---ObjectAttributes-----------------------------------------------------------#
class ContainedObjectAttribute( px.core.BooleanObjectAttribute,
                                px.core.DerivedObjectAttribute,
                                px.core.ChangeTrackingObjectAttribute):
    """indicates whether an object is contained within the box"""
    _depends_on = ['position']
    
    _step_after = 'pushes'
    
    def _get_data(self, idx):
        box = self.world.box
        box_extent = box.extent_with_lid
        
        p = self._other_attr['position'].get(idx)
        
        tl = p >= box_extent[:2] + 1
        br = p <= box_extent[2:] - 1
        
        return np.all(np.hstack((tl, br)), axis=-1)
    
    def _step_object(self, obj, t, dt, agent_id, action):
        """test whether the object moved into or out of the box"""
        change = self.change(obj)
        
        #create a ContainmentEvent that the judge will see
        if change: event = ContainmentEvent(self.world,
                            idx=obj._id,
                            object_name=obj.name,
                            moved_in=change == 1,
                            )
        
        #set the current value so we register any changes on the next step
        self.set_previous(obj)


class TouchingObjectAttribute(  px.core.BooleanObjectAttribute,
                                px.core.DerivedObjectAttribute):
    """indicates whether a block is touching the agent"""
    _depends_on = ['position']
    
    def _get_data(self, idx):
        idx = to_vector(idx)
        
        position = self.world.object_attributes['position']
        
        #get the simple objects associated with the agent
        agent = self.world.agent_object
        idx_agent_simple = np.array([obj._id for obj in agent._simple_leaf_descendants])
        
        #keep only the visible ones
        visible = self.world.object_attributes['visible']
        idx_agent_visible = idx_agent_simple[visible.get(idx_agent_simple)]
        
        #get their positions
        p_agent = position.get(idx_agent_visible)
        
        #block positions to test
        p = position.get(idx)
        
        #test every block position against every agent position
        p_rep, p_agent_rep = cartesian_product(p.T, p_agent.T)
        p_rep = p_rep.reshape((2, p.shape[0], -1), order='F')
        p_agent_rep = p_agent_rep.reshape((2, p.shape[0], -1), order='F')
        
        #look for block objects that are adjacent to agent objects
        p_diff = roundup(np.abs(p_rep - p_agent_rep))
        adjacent = np.sum(p_diff, axis=0) == 1
        
        #touching if adjacent to any agent object
        return np.any(adjacent, axis=1)


class HeightObjectAttribute(px.core.NonNegativeIntegerObjectAttribute,
                            px.core.DerivedObjectAttribute
                            ):
    """the height of the box"""
    _read_only = False
    
    _null_value = 3
    
    _initialize_after = ['position']
    
    _depends_on = ['extent', 'lid_state']
    
    def _default_value(self, obj):
        return obj.world.min_box_height
    
    def _get_data_object(self, obj):
        extent = obj.extent_with_lid
        return int(extent[2] - extent[0] + 1)
    
    def _set_data_object(self, obj, value):
        width = getattr(obj, 'width', self.world.min_box_width)
        obj.set_sprite('%d_%d_%d' % (value, width, obj.lid_state))


class WidthObjectAttribute( px.core.NonNegativeIntegerObjectAttribute,
                            px.core.DerivedObjectAttribute
                            ):
    """the width of the box"""
    _read_only = False
    
    _null_value = 3
    
    _initialize_after = ['position']
    
    _depends_on = ['extent', 'lid_state']
    
    def _default_value(self, obj):
        return obj.world.min_box_width
    
    def _get_data_object(self, obj):
        extent = obj.extent
        return int(extent[3] - extent[1] + 1)
    
    def _set_data_object(self, obj, value):
        height = getattr(obj, 'height', self.world.min_box_height)
        obj.set_sprite('%d_%d_%d' % (height, value, obj.lid_state))


class BoxTypeObjectAttribute(   px.core.StringObjectAttribute,
                                px.core.DerivedObjectAttribute,
                                ):
    """get/set the box type. one of the following:
        agent: box responds to pushes
        moveable: box can be pushed/gripped
        immoveable: box is stationary
    """
    _read_only = False
    
    _initialize_after = ['position']
    
    def _default_value(self, obj):
        return obj.world.box_types[0]
    
    def _get_data_object(self, obj):
        if hasattr(obj, 'pushes'):
            return 'agent'
        elif hasattr(obj, 'mass'):
            return 'moveable'
        else:
            return 'immoveable'
    
    def _set_data_object(self, obj, value):
        """add and remove a set of attributes to make the box act like the
        indicated Object type"""
        if len(value) == 0:
            value = self.world.box_types[0]
        
        if value not in self.world.box_types:
            raise ValueError('"%s" is not an allowed box type' % (value))
        
        #get the attributes to add and remove
        show_self_object = True
        if value == 'agent':
            show_self_object = False
            to_add = ['pushes', 'orientation', 'orients', 'grips', 'velocity', 'acceleration']
            to_remove = ['kinetic_energy', 'momentum', 'mass']
        elif value == 'moveable':
            to_add = ['mass', 'velocity', 'acceleration', 'kinetic_energy', 'momentum']
            to_remove = ['pushes', 'orientation', 'orients', 'grips']
        elif value == 'immoveable':
            to_add = []
            to_remove = ['pushes', 'orientation', 'orients', 'grips', 'momentum', 'kinetic_energy', 'acceleration', 'velocity', 'mass']
        else:
            raise ValueError('"%s" is an unrecognized box type' % (value))
        
        #add new attributes
        for name in to_add:
            if not hasattr(obj, name):
                obj.add_attribute(name)
        
        #remove old attributes
        for name in to_remove:
            if hasattr(obj, name):
                obj.remove_attribute(name)
        
        #set the box color
        color = self.world.agent_color if value == 'agent' else self.world.box_color
        obj.color = color
        
        #set the sprite again since we screwed up the child colors
        obj.sprite = obj.sprite
        
        #show or hide the self object
        if show_self_object:
            self.world.self_object.visible = True
        elif self.world._self_object is not None:
            self.world.self_object.visible = False


class LiddedObjectAttribute(px.core.AbilityObjectAttribute):
    """determines whether actions are available to open/close the box lid"""
    _actions = ['OPEN', 'CLOSE']
    
    _initialize_after = ['lid_state']
    
    _depends_on = ['lid_state']
    
    def set(self, x, value, **kwargs):
        """make sure the lid is open if we are going to unlidded"""
        idx = self._get_index(x)
        
        super(LiddedObjectAttribute, self).set(idx, value, **kwargs)
        
        #unlidded boxes
        idx = to_vector(idx)
        value = to_vector(value)
        idx_unlidded = idx[~value]
        
        if idx_unlidded.size > 0:
            lid_state = self.world.object_attributes['lid_state']
            
            #closed boxes
            state = lid_state.get(idx_unlidded)
            idx_closed = idx_unlidded[state]
            
            if idx_closed.size > 0:
                lid_state.set(idx_closed, False)
    
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
    _read_only = False
    
    _initialize_after = ['position']
    
    def _get_data_object(self, obj):
        return bool(obj.sprite % 2)
    
    def _set_data_object(self, obj, value):
        height = getattr(obj, 'height', self.world.min_box_height)
        width = getattr(obj, 'width', self.world.min_box_width)
        obj.set_sprite('%d_%d_%d' % (height, width, value))


#---Objects--------------------------------------------------------------------#
class BlockObject(px.objects.BasicObject):
    """a block that can be pushed and pulled around"""
    _attributes = ['contained', 'touching']
    
    _defaults = {
        'color': 3,
    }


class BoxAndBlocksSelfObject(px.objects.SelfObject):
    """a simple self object that grips things"""
    _name = 'self'
    
    _attributes = ['grips', 'contained']
    
    _defaults = {
        'color': lambda obj: obj.world.agent_color,
    }


class BoxObject(px.objects.ComplexSpriteObject):
    """a box object that can change shape, be moveable or immoveable, and can
    become an agent.
    """
    _attributes = ['lid_state', 'lidded', 'height', 'width', 'box_type']
    
    _defaults = {'color': lambda obj: obj.world.box_color}
    
    #see show() and hide()
    _old_height = None
    _old_width = None
    _old_lid_state = None
    
    def __init__(self, world, sprites=None, **kwargs):
        if sprites is None:
            sprites = self.get_box_sprites
        
        super(BoxObject, self).__init__(world, sprites=sprites, **kwargs)
    
    @property
    def extent_with_lid(self):
        """get a version of the box extent that includes the lid, regardless of
        whether it is open or closed"""
        extent = self.extent
        
        if not self.lid_state:
            extent[0] -= 1
        
        return extent
    
    def get_box_sprites(self, obj):
        sprites = OrderedDict()
        
        for height in xrange(self.world.min_box_height, self.world.max_box_height + 1):
            for width in xrange(self.world.min_box_width, self.world.max_box_width + 1):
                for closed in [False, True]:
                    sprite_name = '%d_%d_%d' % (height, width, closed)
                    sprites[sprite_name] = self.get_box_sprite(height, width, closed)
        
        #add an invisible box
        sprites['hidden'] = self.get_box_sprite(self.world.max_box_height,
                                self.world.max_box_width, False, box_char='0')
        
        return sprites
    
    def get_box_sprite(self, height, width, closed, box_char='X'):
        top_char = box_char if closed else '0'
        
        box = []
        
        #box top
        box.append(top_char * width)
        
        #box sides
        box.extend((height - 2) * [box_char + '0' * (width - 2) + box_char])
        
        #box bottom
        box.append(box_char * width)
        
        #pad each row to the maximum box width
        max_width = self.world.max_box_width
        for idx,row in enumerate(box):
            box[idx] = row.center(max_width, '0')
        
        #pad to the maximum box height
        max_height = self.world.max_box_height
        row_pre = int((max_height - height) / 2)
        row_post = max_height - height - row_pre
        blank_row = '0' * max_width
        box = [blank_row] * row_pre + box + [blank_row] * row_post
        
        return h.sprite.from_string('\n'.join(box))
    
    def show(self):
        """this is a hack to mimic setting visible, but without the current bug
        that sets all invisible sprite pixels to visible when the box goes from
        invisible to visible"""
        if self._old_height is not None:
            height = self._old_height
            width = self._old_width
            lid_state = self._old_lid_state
            self.set_sprite('%d_%d_%d' % (height, width, lid_state))
            
            self._old_height = None
            self._old_width = None
    
    def hide(self):
        """see show()"""
        if self._old_height is None:
            self._old_height = self.height
            self._old_width = self.width
            self._old_lid_state = self.lid_state
            self.set_sprite('hidden')


#---Goals----------------------------------------------------------------------#
class BoxAndBlocksGoal(px.core.Goal):
    """base class for all box-and-box goals"""
    _exclusive = True
    _active = False
    
    #maximum number of time to try something before failing
    _max_tries = 100
    
    #whether the box should be visible. set to 'random' for random visibility
    #each time setup_preconditions is called.
    _visible_box = True
    
    def setup_preconditions(self, **kwargs):
        """set the box visibility"""
        show_box = self.rng.rand() < 0.5 if self._visible_box == 'random' else self._visible_box
        if show_box:
            self.world.box.show()
        else:
            self.world.box.hide()
        
        return super(BoxAndBlocksGoal, self).setup_preconditions(**kwargs)
    
    def setup_simple_self(self):
        """make sure the world includes a simple self object
        
        Returns
        -------
        success : bool | string
            True if the simple self was successfully set up, or a failure
            message
        """
        if self.world.box.box_type == 'agent':
            box_types = list(set(self.world.box_types) - set(['agent']))
            
            if len(box_types) == 0: return 'no allowed box types include a simple self object'
            
            self.world.box.box_type = self.rng.choice(box_types)
        
        return True
    
    def clear_position(self, y, x):
        """make sure nothing is at the specified position
        
        Parameters
        ----------
        y : Number
            the y-position
        x : Number
            the x-position
        
        Returns
        -------
        success : bool | string
            True if the position was cleared, or a failure description
        moved : bool
            True if an object was moved
        """
        moved = False
        
        num_tries = 0
        
        #move any blocking objects
        obj = self.is_occupied_head(y, x)
        while obj:
            moved = True
            
            num_tries += 1
            
            if num_tries > self._max_tries: return 'exceeded maximum number of tries to clear (%f, %f)' % (y, x), moved
            
            success = self.move_to_random_position(obj)
            if not self._test_success(success): return success, moved
            
            obj = self.is_occupied_head(y, x)
        
        return True, moved
    
    def clear_positions(self, positions):
        """clear a set of positions
        
        Parameters
        ----------
        positions : list-like
            a list of positions to clear
        
        Returns
        
        Returns
        -------
        success : bool | string
            True if the positions were cleared, or a failure description
        moved : bool
            True if any objects were moved
        """
        moved = False
        
        num_tries = 0
        success = False
        
        while not success:
            num_tries += 1
            
            if num_tries > self._max_tries: return 'could not clear positions %s' % (positions,), moved
            
            #clear each position
            success = True
            for p in positions:
                #clear the position
                clear_success, clear_moved = self.clear_position(*p)
                moved = moved or clear_moved
                
                if not self._test_success(clear_success): return clear_success, moved  # clearing failed
                
                #repeat the process if we moved something
                if clear_moved: success = False
        
        return True, moved
    
    def clear_region(self, extent):
        """clear a region of objects
        
        Parameters
        ----------
        extent : ndarray
            the (top, left, bottom, right) bounds of the region to clear
        
        Returns
        -------
        success : bool | string
            True if the region was cleared, or a failure description
        moved : bool
            True if any objects were moved
        """
        #make sure we have good range bounds
        extent[[0, 2]] = np.floor(extent[[0, 2]])
        extent[[1, 3]] = np.ceil(extent[[1, 3]])
        extent = extent.astype(int)
        
        positions = [(y, x) for y in xrange(extent[0], extent[2] + 1)
                        for x in xrange(extent[1], extent[3] + 1)]
        
        return self.clear_positions(positions)
    
    def move_to_position(self, obj, y, x):
        """move an object to a position
        
        Parameters
        ----------
        obj : Object
            the Object to move
        y : Number
            the y value of the target position
        x : Number
            the x value of the target position
        
        Returns
        -------
        success : bool | string
            True if successful, or a failure description
        """
        p = np.array([y, x])
        
        num_tries = 0
        
        success = False
        while not success:
            num_tries += 1
            
            if num_tries > self._max_tries: return 'could not move %s to (%f, %f)' % (obj.unique_name, y, x)
            
            obj.position = p
            
            success = np.array_equal(obj.position, p)
        
        return success
    
    def move_to_positions(self, objects, positions):
        """move a set of Objects to a set of positions
        
        Parameters
        ----------
        objects : list-like
            a list of Objects
        positions : list-like
            a corresponding list of positions
        
        Returns
        -------
        success : bool | string
            True if successful, or a failure description
        """
        position = self.world.object_attributes['position']
        
        #make sure we have an N x 2 array of positions
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        
        #first clear the positions
        success, moved = self.clear_positions(positions)
        if not self._test_success(success): return success
        
        #now move each Object into position
        num_tries = 0
        success = False
        while not success:
            num_tries += 1
            
            if num_tries > self._max_tries: return 'could not move objects into position in time'
            
            #move each Object
            for obj,p in zip(objects, positions):
                success = self.move_to_position(obj, *p)
                if not self._test_success(success): return success
            
            #did all the Objects and up in the right place?
            success = np.all(position.compare(position.get(objects), positions))
        
        return success
    
    def move_to_random_position(self, obj, extent=None, perimeter=False):
        """move an object to a random, unoccupied position.
        
        Parameters
        ----------
        obj : Object
            the Object to move
        extent : ndarray, optional
            the bounds (exclusive) of the random position
        perimeter : bool, optional
            True to move the object to the perimeter of the specified extent
        
        Returns
        -------
        success : bool | string
            True if successful, or a failure description
        """
        if extent is None:
            obj_extent = obj.extent
            height = obj_extent[2] - obj_extent[0] + 1
            width = obj_extent[3] - obj_extent[1] + 1
            
            h_offset = np.ceil(height / 2.)
            w_offset = np.ceil(width / 2.)
            
            extent = np.array([h_offset, w_offset, self.world.height - 1 - h_offset, self.world.width - 1 - w_offset])
        
        num_tries = 0
        
        success = False
        while not success:
            num_tries += 1
            
            if num_tries > self._max_tries: return 'could not move %s to a random position' % (obj.unique_name)
            
            y, x = self._get_random_position(extent=extent, perimeter=perimeter)
            if y is None: return 'could not find a random position for %s' % (obj.unique_name)
            
            obj.position = (y, x)
            y, x = obj.position
            
            if extent is not None:
                if perimeter:
                    success = ((y == extent[0] or y == extent[2]) and \
                              extent[1] <= x <= extent[3]) or \
                              ((x == extent[1] or x == extent[3]) and \
                              extent[0] <= y <= extent[2])
                else:
                    success = extent[0] < y < extent[2] and \
                              extent[1] < x < extent[3]
            else:
                success = True
        
        return success
    
    def move_next_to(self, obj_source, obj_target):
        """move one Object next to another
        
        Parameters
        ----------
        obj_source : Object
            the Object to move
        obj_target : Object
            the Object to move next to
        
        Returns
        -------
        success : bool | string
            True if successful, or a failure description
        """
        y, x = self._get_nearest_neighboring_position(obj_source, obj_target)
        
        return self.move_to_position(obj_source, y, x)
    
    def is_occupied(self, y, x):
        """test whether a position is occupied
        
        Returns
        -------
        obj : Object | False
            the Object at position (y, x), or False if there is no Object there
        """
        return self.world.objects.get(object_type='SIMPLE', position=(y, x), visible=True) or False
    
    def is_occupied_head(self, y, x):
        """test whether a position is occupied, and return the head of any
        occupying Object's family
        
        Returns
        -------
        obj : Object | False
            the head of the family of the Object at position (y, x), or False if
            there is no Object there
        """
        obj = self.is_occupied(y, x)
        
        return obj._head_of_family if obj else obj
    
    def _get_random_position(self, extent=None, perimeter=False):
        """get a random, unoccupied position within the visible world
        
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
            extent = np.array([1, 1, self.world.height - 2, self.world.width - 2])
        
        num_tries = 0
        success = False
        
        p_tried = []
        while not success:
            num_tries += 1
                
            if num_tries > self._max_tries: return None, None
            
            bounds = copy(extent)
            
            #determine which coordinates to randomize and the position bounds
            if perimeter:
                #choose the box side
                side = self.rng.randint(4)
                
                y_random = x_random = False
                if side == 0:  # left
                    y_random = True
                    x = bounds[1]
                elif side == 1:  # top
                    y = bounds[0]
                    x_random = True
                elif side == 2:  # right
                    y_random = True
                    x = bounds[3]
                elif side == 3:  # bottom
                    y = bounds[2]
                    x_random = True
                else:
                    raise ValueError('invalid side')
                
                if y_random: bounds[2] += 1
                if x_random: bounds[3] += 1
            else:
                y_random = x_random = True
                bounds[0] += 1
                bounds[1] += 1
            
            if y_random: y = self.rng.randint(bounds[0], bounds[2])
            if x_random: x = self.rng.randint(bounds[1], bounds[3])
            
            #don't keep trying the same positions
            p = (y, x)
            if p in p_tried:
                num_tries -= 1
                continue
            else:
                p_tried.append(p)
            
            success = not self.is_occupied(y, x)
        
        return y, x
    
    def _get_nearest_neighboring_position(self, source_obj, target_obj):
        """get the nearest position that neighbors an Object
        
        Parameters
        ----------
        source_obj : Object
            the Object that wants to move near target_obj
        target_obj : Object
            the Object whose neighboring position should be determined
        
        Returns
        -------
        y : int
            the y value of the neighbor position
        x : int
            the x value of the neighbor position
        """
        p_source = source_obj.position
        p_target = target_obj.position
        
        #direction from target to source
        direction = pmath.direction(p_target - p_source)
        p_rel = roundup(direction)
        
        #neighbor position is one step along the projection of this direction
        #onto the nearest axis
        p_neighbor = copy(p_target)
        coord_max = np.argmax(np.abs(direction))
        p_neighbor[coord_max] -= p_rel[coord_max]
        
        return p_neighbor
    
    def _get_nearest_block(self):
        """find the nearest block"""
        position = self.world.get_object_attribute('position')
        blocks = self.world.blocks
        
        p_agent = self.world.agent_object.position
        p_blocks = position.get(blocks)
        
        #get the position of the closest block
        d2 = pmath.magnitude2(p_agent - p_blocks)
        idx_min = np.argmin(d2)
        
        return blocks[idx_min]


class MoveDirectionGoal(px.goals.ActionGoal, BoxAndBlocksGoal):
    """base class for goals that entail moving in one of the four directions"""
    
    @property
    def actions(self):
        if self._actions is None:
            return [self.world.goal_action]
        else:
            return px.goals.ActionGoal.actions.fget(self)
    
    def match(self, other):
        """make sure both goals use the same action"""
        super(MoveDirectionGoal, self).match(other)
        
        self.world.goal_action = other.world.goal_action
    
    def move_opposite(self):
        """move the agent in the opposite direction of the goal action"""
        self.world.agent_object.position -= self._action_to_direction()
    
    def clear_adjacent_positions(self):
        """make sure nothing is adjacent to the agent in the direction it will
        move"""
        num_tries = 0
        success = False
        
        region = self._get_agent_adjacent_region()
        
        success, moved = self.clear_region(region)
        
        return success
    
    def will_be_in_inner_position(self):
        """True if the agent will be within the frame area after executing the
        action"""
        agent_extent = self.world.agent_object.extent
        
        return switch(self.actions[0],
                LEFT=agent_extent[1] >= 2,
                UP=agent_extent[0] >= 2,
                RIGHT=agent_extent[3] <= self.world.width - 3,
                DOWN=agent_extent[2] <= self.world.height - 3,
                )
    
    def _setup_preconditions(self):
        """steps:
        -   make sure the agent won't move out of the visible world
        -   make sure nothing is at the position to which the agent will move
        """
        #make sure we aren't on the edge of the world
        num_tries = 0
        while not self.will_be_in_inner_position():
            num_tries += 1
            
            if num_tries > self._max_tries: return 'could not move the agent away from the frame'
            
            self.move_opposite()
        
        #make sure nothing is in the destination position
        return self.clear_adjacent_positions()
    
    def _action_to_direction(self):
        """map the action string to the direction of movement"""
        return switch(self.actions[0],
                LEFT=(0, -1),
                UP=(-1, 0),
                RIGHT=(0, 1),
                DOWN=(1, 0),
                )
    
    def _get_agent_adjacent_region(self):
        """get the extent of the region that is adjacent to the agent in the
        direction it will move"""
        #get the whole agent region
        region = copy(self.world.agent_object.extent)
        
        #modify the region based on the movement direction
        action = self.actions[0]
        if action == 'LEFT':
            region[1] -= 1
            region[3] = region[1]
        elif action == 'UP':
            region[0] -= 1
            region[2] = region[0]
        elif action == 'RIGHT':
            region[3] += 1
            region[1] = region[3]
        elif action == 'DOWN':
            region[2] += 1
            region[0] = region[2]
        
        return region


class MoveToGoal(BoxAndBlocksGoal):
    """base class for goals that move the simple self object to a designated
    goal position, defined by self.goal_position"""
    _visible_box = 'random'
    
    _goal_position = None
    
    def __init__(self, world, goal_position=None, **kwargs):
        self.goal_position = goal_position or self._goal_position
        
        super(MoveToGoal, self).__init__(world, **kwargs)
    
    @property
    def goal_position(self):
        if self._goal_position is None:
            self.goal_position = self._get_random_position()
        
        return self._goal_position
    
    @goal_position.setter
    def goal_position(self, p):
        self._goal_position = p
    
    def _is_achieved(self):
        """is the self object at the goal position?"""
        return self.world._self_object is not None and np.all(self.world.self_object.position == self.goal_position)
    
    def _setup_preconditions(self):
        """steps:
        -   make sure the box type includes a simple self object
        -   make sure the self isn't at the goal position
        """
        #make sure we have a simple self object
        success = self.setup_simple_self()
        if not self._test_success(success): return success
        
        #make sure the self isn't at the goal position
        if np.all(self.world.self_object.position == self.goal_position):
            success, moved = self.clear_position(*self.goal_position)
            if not self._test_success(success): return success
        
        return True
    
    def _achieve(self):
        """steps:
        -   make sure the goal position is unoccupied
        -   move the self to the goal position
        """
        #make sure the goal position isn't occupied by a conflicting object
        success, moved = self.clear_position(*self.goal_position)
        if not self._test_success(success): return success
        
        #move the self object to the goal position
        obj = self.world.self_object
        success = self.move_to_position(obj, *self.goal_position)
        if not self._test_success(success): return success
        
        return True


class MoveToCornerGoal(MoveToGoal):
    """base class for goals the move the simple self object to a corner of the
    visible world. _goal_position is a string describing the corner, which gets
    translated to the actual position by goal_position."""
    
    @property
    def goal_position(self):
        top = left = 0
        bottom = self.world.height - 1
        right = self.world.width - 1
        
        if self.world.has_frame:
            top += 1
            left += 1
            bottom -= 1
            right -= 1
        
        goal_position = self.world.goal_corner if self._goal_position is None \
                        else self._goal_position
        
        return switch(goal_position,
            top_left=(top, left),
            top_right=(top, right),
            bottom_left=(bottom, left),
            bottom_right=(bottom, right),
            )
    
    @goal_position.setter
    def goal_position(self, p):
        if p is not None: assert isinstance(p, basestring)
        self._goal_position = p
    
    def match(self, other):
        """make sure both goals use the same corner"""
        super(MoveToCornerGoal, self).match(other)
        
        self.world.goal_corner = other.world.goal_corner


class TouchBlockGoal(MoveToGoal):
    """make the agent touch a block"""
    touching = None
    
    @property
    def goal_position(self):
        """goal position is a point next to the nearest block"""
        #get the nearest block
        block = self._get_nearest_block()
        
        agent = self.world.agent_object
        
        return self._get_nearest_neighboring_position(agent, block)
    
    @goal_position.setter
    def goal_position(self, p):
        if p is not None: raise AttributeError('cannot set goal_position')
    
    def prepare(self):
        self.touching = self.world.get_object_attribute('touching')
    
    def _is_achieved(self):
        """is the agent touching a block?"""
        return np.any(self.touching())
    
    def _setup_preconditions(self):
        """make sure the agent is not touching a block. steps:
        -   make sure the box type includes a simple self object
        -   alternate moving left, right, up, and down (and reversing the
            action) until we've pushed all the blocks out of the way
        """
        #make sure we have a simple self object
        success = self.setup_simple_self()
        
        #make sure we aren't touching anything
        num_tries = 0
        
        actions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        current_actions = []
        
        while self.achieved:
            num_tries += 1
            
            if num_tries > self._max_tries: return 'could not move the agent away from the blocks'
            
            #shuffle the movement actions
            if len(current_actions) == 0:
                current_actions = copy(actions)
                self.rng.shuffle(current_actions)
            
            #try moving out of the way
            action = current_actions.pop()
            self.world.step(action)
            if not self.achieved: break
            
            #try moving back (maybe we pushed the block out of the way)
            self.world.step(self._get_opposite_action(action))
        
        return True
    
    def _get_opposite_action(self, action):
        """get the action that undoes the specified action"""
        return switch(action,
                LEFT='RIGHT',
                RIGHT='LEFT',
                UP='DOWN',
                DOWN='UP',
                )

class PushBlockToCornerGoal(MoveToCornerGoal):
    """push a block to the corner"""
    
    def _is_achieved(self):
        """is a block in the goal position?"""
        obj = self.is_occupied(*self.goal_position)
        return obj and obj.name == 'block'
    
    def _setup_preconditions(self):
        """steps:
        -   setup MoveToCornerGoal preconditions
        -   make sure the corner is clear
        """
        #move to corner preconditions
        success = MoveToCornerGoal._setup_preconditions(self)
        if not self._test_success(success): return success
        
        #clear the corner
        success, moved = self.clear_position(*self.goal_position)
        if not self._test_success(success): return success
        
        return True
    
    def _achieve(self):
        """steps:
        -   move the nearest block to the goal corner
        -   move the self object adjacent to that block
        """
        #move the nearest block to the goal corner
        block = self._get_nearest_block()
        success = self.move_to_position(block, *self.goal_position)
        if not self._test_success(success): return success
        
        #move the agent to an adjacent position
        agent = self.world.agent_object
        success = self.move_next_to(agent, block)
        if not self._test_success(success): return success
        
        return True


class GatherBlocksGoal(MoveToGoal):
    """gather the blocks together"""
    
    @property
    def goal_position(self):
        """default goal position is position of first block"""
        if self._goal_position is None:
            self.goal_position = self.world.blocks[0].position
        
        return self._goal_position
    
    @goal_position.setter
    def goal_position(self, p):
        self._goal_position = p
    
    def _is_achieved(self):
        """all all the blocks touching each other?"""
        #get all block positions
        blocks = self.world.blocks
        positions = self.world.object_attributes['position'].get(blocks)
        positions = tuple(map(tuple, positions.astype(int)))
        
        connected = set()
        edge = [positions[0]]
        
        #find all blocks that are connect to the first one
        while not len(edge) == 0:
            #current position to check
            pos = edge.pop()
            
            #add it to the list of connected positions
            connected.add(pos)
            
            #find its neighbors
            y, x = pos
            np = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
            neighbors = [n for n in np if n in positions and not n in connected]
            
            edge.extend(neighbors)
        
        #are all blocks connected?
        return len(blocks) == len(connected)
    
    def _setup_preconditions(self):
        """steps:
        -   make sure there are at least two blocks
        -   MoveToGoal preconditions
        """
        #make sure there are at least two blocks
        if self.world.num_blocks < 2: self.world.num_blocks = 2
            
        return super(GatherBlocksGoal, self)._setup_preconditions()
    
    def _achieve(self):
        """gather all the blocks around the goal position"""
        blocks = self.world.blocks
        p_block = self._get_gathered_block_positions()
        
        return self.move_to_positions(blocks, p_block)
    
    def _get_gathered_block_positions(self):
        """get the positions the blocks should be in once gathered"""
        blocks = self.world.blocks
        
        #group center position
        p = np.array(self.goal_position)
        
        #variables to keep track of spiraling search
        step_directions = ( (0, -1), (-1, 0), (0, 1), (1, 0) )
        idx_direction = len(step_directions) - 1
        run_length = 0
        run_step = 0
        
        #get the positions to which to move the blocks
        num_blocks = len(blocks)
        block_positions = [None] * num_blocks
        for idx in xrange(num_blocks):
            #record the current block position
            block_positions[idx] = p
            
            #get the next block position
            num_tries = 0
            success = False
            while not success:
                #change step directions
                if run_step == run_length:
                    run_step = 0
                    
                    #change directions and potentially increase the run length
                    idx_direction = (idx_direction + 1) % len(step_directions)
                    if idx_direction in (0, 2): run_length += 1
                
                run_step += 1
                
                #get the next block position
                p = copy(p) + step_directions[idx_direction]
                
                #is it valid?
                success = (1 <= p[0] <= self.world.height - 2) and \
                          (1 <= p[1] <= self.world.width - 2)
        
        return block_positions


class PushTwoBlocksTogetherGoal(GatherBlocksGoal):
    """push two blocks together. like GatherBlocksGoal but only two blocks and
    we hide the box to make it simple."""
    _visible_box = False
    
    def _setup_preconditions(self):
        """steps:
        -   make sure there are at exactly two blocks
        -   GatherBlocks preconditions
        """
        #make sure there are at least two blocks
        if self.world.num_blocks != 2: self.world.num_blocks = 2
            
        return super(PushTwoBlocksTogetherGoal, self)._setup_preconditions()


class ContainmentGoal(BoxAndBlocksGoal):
    """base class for containment-related goals"""
    #reference to the contained attribute
    contained = None
    
    def prepare(self):
        self.contained = self.world.get_object_attribute('contained')
    
    def move_inside_box(self, obj, box_extent=None):
        """move an Object inside the box
        
        Parameters
        ----------
        obj : Object
            the object to move
        box_extent : ndarray, optional
            the box extent, including the lid
        
        Returns
        -------
        success : bool | string
            True if successful, or a failure description
        """
        if box_extent is None: box_extent = self.world.box.extent_with_lid
        
        y, x = self._get_next_inside_position(box_extent)
        
        if y is None: return 'could not find a position inside the box for %s' % (obj.unique_name)
        
        obj.position = (y, x)
        
        return True
    
    def move_outside_box(self, obj, box_extent=None):
        """move an Object outside the box
        
        Parameters
        ----------
        obj : Object
            the object to move
        box_extent : ndarray, optional
            the box extent, including the lid
        
        Returns
        -------
        success : bool | string
            True if successful, or a failure description
        """
        if box_extent is None: box_extent = self.world.box.extent_with_lid
        
        y, x = self._get_next_outside_position(box_extent)
        
        if y is None: return 'could not find a position outside the box for %s' % (obj.unique_name)
        
        obj.position = (y, x)
        
        return True
    
    def move_adjacent_to_box(self, obj, box_extent=None):
        """move an Object adjacent to the box
        
        Parameters
        ----------
        obj : Object
            the object to move
        box_extent : ndarray, optional
            the box extent, including the lid
        
        Returns
        -------
        success : bool | string
            True if successful, or a failure description
        """
        if box_extent is None: box_extent = self.world.box.extent_with_lid
        
        y, x = self._get_next_adjacent_position(box_extent)
        
        if y is None: return 'could not find a position adjacent to the box for %s' % (obj.unique_name)
        
        obj.position = (y, x)
        
        return True
    
    def close_box_lid(self, box_extent=None):
        """close the box lid after moving conflicting objects away (so the box
        doesn't go flying due to a collision)
        
        Parameters
        ----------
        box_extent : ndarray, optional
            the box extent, including the lid
        
        Returns
        -------
        success : bool | string
            True if successful, or a failure description
        """
        if box_extent is None: box_extent = self.world.box.extent_with_lid
        
        y_test = box_extent[0]
        x_test_min = box_extent[1]
        x_test_max = box_extent[3]
        
        num_tries = 0
        
        #check each position along the lid, moving objects that are in the way
        x_test = x_test_min
        while x_test <= x_test_max:
            if num_tries > self._max_tries: return 'exceeded maximum number of tries to clear the block lid'
            
            #clear the current lid position
            success, moved = self.clear_position(y_test, x_test)
            if not self._test_success(success): return success
            
            if moved:  # reset the x-position
                num_tries += 1
                x_test = x_test_min
            else:  # increment the x-position
                x_test += 1
        
        #close the lid
        self.world.step('CLOSE')
        
        return True
    
    def is_in_box(self, y, x, box_extent):
        """test whether a position is within the box"""
        return  (box_extent[0] < y < box_extent[2]) and \
                (box_extent[1] < x < box_extent[3])
    
    def is_achieved_all_contained(self):
        all_contained = np.all(self.contained(self.world.blocks))
        box = self.world.box
        
        return all_contained and (not box.lidded or box.lid_state)
    
    def achieve_preconditions_all_contained(self):
        return self.achieve_not_all_contained()
    
    def achieve_all_contained(self):
        """steps:
        -   move all blocks into the box
        -   close the lid if it exists
        """
        box = self.world.box
        box_extent = box.extent_with_lid
        
        #make sure the self object is out of the way
        self_object = self.world._self_object
        if self_object is not None:
            success = self.move_outside_box(self_object, box_extent=box_extent)
            if not self._test_success(success): return success

        #box each block. this should really be done via step(), but that's too
        #difficult for now.
        for block in self.world.blocks:
            if not block.contained:
                success = self.move_inside_box(block, box_extent=box_extent)
                if not self._test_success(success): return success
        
        #close the lid
        if box.lidded and not box.lid_state:
            self.close_box_lid()
        
        #we did it!
        return True
    
    def is_achieved_not_all_contained(self):
        return not self.is_achieved_all_contained()
    
    def achieve_preconditions_not_all_contained(self):
        return self.achieve_all_contained()
    
    def achieve_not_all_contained(self):
        """steps:
        -   move the blocks to random positions (that aren't all contained)
        -   set the lid to a random state
        """
        blocks = self.world.blocks
        box_extent = self.world.box.extent_with_lid
        
        num_tries = 0
        
        success = False
        while not success:
            num_tries += 1
            
            if num_tries > self._max_tries: return 'exceeded maximum number of tries to randomize the blocks into a not-all-contained state'
            
            #move the blocks to random positions
            for block in blocks:
                success = self.move_to_random_position(block)
                if not self._test_success(success): return success
            
            #wait to set the lid state until we get the blocks positioned
            success = self.is_achieved_not_all_contained()
            if not success: continue
            
            #set a random lid state
            if self.world.box.lidded:
                if self.rng.rand() < 0.5:  # close
                    self.close_box_lid(box_extent=box_extent)
                else:  # open
                    self.world.step('OPEN')
        
        return True
    
    def is_achieved_none_contained(self):
        return not np.any(self.contained(self.world.blocks)) and not self.world.box.lid_state
    
    def achieve_preconditions_none_contained(self):
        return self.achieve_all_contained()
    
    def achieve_none_contained(self):
        """steps:
        -   move all the blocks out of the box
        """
        box = self.world.box
        box_extent = box.extent_with_lid
        
        #move the blocks out of the box. this should really be done via step(),
        #but that's too difficult for now.
        for block in self.world.blocks:
            if block.contained:
                success = self.move_outside_box(block, box_extent=box_extent)
                
                if not success: return success
        
        #open the lid
        if box.lid_state:
            self.world.step('OPEN')
        
        return True
    
    def is_achieved_almost_contained(self):
        blocks = self.world.blocks
        return np.sum(self.contained(blocks)) == len(blocks) - 1
    
    def achieve_preconditions_almost_contained(self):
        return self.achieve_not_all_contained()
    
    def achieve_almost_contained(self):
        """steps:
        -   move all blocks into the block (close the lid because i'm lazy)
        -   open the lid
        -   move one block adjacent to the box"""
        blocks = self.world.blocks
        
        if len(blocks) == 0: return 'no blocks to contain'
        
        #move all blocks into the box
        success = self.achieve_all_contained()
        if not self._test_success(success): return success
        
        #open the lid
        if self.world.box.lid_state:
            self.world.step('OPEN')
        
        #now move one block out of the box
        success = self.move_adjacent_to_box(blocks[0])
        if not self._test_success(success): return success
        
        return True
    
    def _get_next_inside_position(self, box_extent):
        """find the next position that is unoccupied and inside the box
        
        Parameters
        ----------
        box_extent : ndarray
            the box extent, including the lid
        
        Returns
        -------
        y : Number
            the new y-position
        x : Number
            the new x-position
        """
        y, x = self._get_random_position(extent=box_extent)
        
        if y is not None:
            assert self.is_in_box(y, x, box_extent), 'did not get a position inside the box'
        
        return y, x
    
    def _get_next_outside_position(self, box_extent):
        """find the next position that is unoccupied and outside the box
        
        Parameters
        ----------
        box_extent : ndarray
            the box extent, including the lid
        
        Returns
        -------
        y : Number
            the new y-position
        x : Number
            the new x-position
        """
        num_tries = 0
        
        success = False
        while not success:
            num_tries += 1
            
            if num_tries > self._max_tries: return None, None
            
            y, x = self._get_random_position()
            
            success = not self.is_in_box(y, x, box_extent)
        
        return y, x
    
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
        if not self.world.box.lid_state:  # box lid positions are also an option
            perimeter_extent[0] += 1
        
        return self._get_random_position(extent=perimeter_extent, perimeter=True)


class MoveIntoBoxGoal(ContainmentGoal):
    """move the simple self object into the box"""
    
    def _is_achieved(self):
        """is the self object in the box?"""
        return getattr(self.world._self_object, 'contained', False)
    
    def _setup_preconditions(self):
        """steps:
        -   make sure the box type includes a simple self object
        -   make sure the self isn't in the box
        """
        #make sure we have a simple self object
        success = self.setup_simple_self()
        if not self._test_success(success): return success
        
        #make sure the self isn't in the box
        agent = self.world.self_object
        success = self.move_outside_box(agent)
        if not self._test_success(success): return success
        
        return True
    
    def _achieve(self):
        """steps:
        -   move the self into the box
        """
        agent = self.world.self_object
        success = self.move_inside_box(agent)
        if not self._test_success(success): return success
        
        return True


class AllContainedGoal(ContainmentGoal):
    """all blocks are contained in the box and the lid is closed if it exists"""
    def _is_achieved(self):
        return self.is_achieved_all_contained()
    
    def _setup_preconditions(self):
        return self.achieve_preconditions_all_contained()
    
    def _achieve(self):
        return self.achieve_all_contained()


class PutBlockInBoxGoal(AllContainedGoal):
    """put the only block into the box (that has no lid)"""
    def _setup_preconditions(self):
        #make sure we have one block and an unlidded box
        if self.world.num_blocks != 1: self.world.num_blocks = 1
        if self.world.box.lidded: self.world.box.lidded = False
        
        return super(PutBlockInBoxGoal, self)._setup_preconditions()


class NotAllContained(ContainmentGoal):
    """not all the blocks are contained"""
    def _is_achieved(self):
        return self.is_achieved_not_all_contained()
    
    def _setup_preconditions(self):
        return self.achieve_preconditions_not_all_contained()
    
    def _achieve(self):
        return self.achieve_not_all_contained()


class NoneContainedGoal(ContainmentGoal):
    """no blocks are contained in the box and the lid is optionally opened"""
    def _is_achieved(self):
        return self.is_achieved_none_contained()
    
    def _setup_preconditions(self):
        return self.achieve_preconditions_none_contained()
    
    def _achieve(self):
        return self.achieve_none_contained()


class AlmostContainedGoal(ContainmentGoal):
    """all but one block is contained"""
    def _is_achieved(self):
        return self.is_achieved_almost_contained()
    
    def _setup_preconditions(self):
        return self.achieve_preconditions_almost_contained()
    
    def _achieve(self):
        return self.achieve_almost_contained()


class PullOutOfBoxGoal(NoneContainedGoal):
    """pull all the blocks out of a skinny, immoveable box"""
    
    def _setup_preconditions(self):
        """steps:
        -   make sure we have a skinny, immoveable box
        -   contain all the blocks
        """
        box = self.world.box
        
        #box is immoveable
        if box.box_type != 'immoveable':
            box.box_type = 'immoveable'
        
        #box is skinny
        if box.width != 3:
            box.width = 3
        
        #box has no lid
        box.lidded = False
        
        #make sure we don't have too many blocks
        if self.world.num_blocks > self.world.max_num_blocks:
            if 'num_blocks' in self.world.variants:
                num_blocks = self.rng.choice(self.world.variants['num_blocks'].states)
            else:
                num_blocks = self.rng.randint(1, self.world.max_num_blocks + 1)
            
            self.world.num_blocks = num_blocks
        
        return super(PullOutOfBoxGoal, self)._setup_preconditions()


#---other Entities-------------------------------------------------------------#
class BoxAndBlocksRandomizer(px.randomizers.RandomPositionsRandomizer):
    """randomizes based on Variants and position"""
    pass


class BoxAndBlocksJudge(px.core.Judge):
    """a judge that rewards contained blocks and penalizes inefficiency"""
    _reward_events = [
        {
            'event': 'containment',
            'reward': lambda judge, event: (100 * (1 if event.moved_in else -1)) if event.object_name == 'block' else 0,
        },
    ]


#---PixelWorld-----------------------------------------------------------------#
class BoxAndBlocksWorld(px.core.PixelWorld):
    
    _frame = None
    _box = None
    _self_object = None
    
    def __init__(self, variants=None, goals=None, randomizer='box_and_blocks',
                    judge='box_and_blocks', **kwargs):
        """
        **kwargs
            extra arguments for PixelWorld
        """
        #variants
        if variants is None:
            variants = []
            
            #include variants if their values weren't fixed by the input
            #arguments
            for name in ['show_colors', 'goal_corner', 'goal_action',
                'has_frame', 'box_type', 'box_has_lid', 'box_height',
                'box_width', 'num_blocks']:
                if name not in kwargs:
                    variants += [name]
        
        #goals
        if goals is None: goals = [
            'move_direction',
            'move_to_corner',
            'move_into_box',
            'touch_block',
            'push_block_to_corner',
            'push_two_blocks_together',
            'gather_blocks',
            'put_block_in_box',
            'all_contained',
            #'none_contained',
            #'almost_contained',
            'not_all_contained',
            'pull_out_of_box',
            ]
        
        super(BoxAndBlocksWorld, self).__init__(variants=variants, goals=goals,
                randomizer=randomizer, judge=judge, **kwargs)
    
    @property
    def frame(self):
        """get the frame object"""
        if self._frame is None:  # need to create the frame
            self._frame = self.create_object('frame')
        
        return self._frame
    
    @property
    def box(self):
        """get the box object"""
        if self._box is None:  # need to create the box
            self._box = self.create_object('box')
        
        return self._box
    
    @property
    def blocks(self):
        """get a collection of block objects"""
        return self.objects.find(name='block', visible=True)
    
    @property
    def self_object(self):
        """get the self object, if it exists (or None)"""
        if self._self_object is None:  # need to create the self
            self._self_object = self.create_object('self')
        
        return self._self_object
    
    @property
    def agent_name(self):
        """name of the agent object"""
        return 'box' if self._self_object is None or not self.self_object.visible else 'self'
    
    @property
    def agent_object(self):
        """get the object that currently responds to the agent"""
        return self.box if self.agent_name == 'box' else self.self_object
    
    def test_goals(self, goals=None, **kwargs):
        """test that all goals are achievable
        
        Parameters
        ----------
        goals : list, optional
            a list of goal names to test
        **kwargs
            extra arguments to Goal.test_achieve()
        
        Returns
        -------
        f : dict
            a dict mapping goal names to test results
        """
        if goals is None: goals = self.goals.keys()
        
        num_tested = 0
        results = {}
        for name in goals:
            num_tested += 1
            
            print 'testing %s (%d/%d)' % (name, num_tested, len(goals))
            
            results[name] = self.goals[name].test_achieve(**kwargs)
        
        return results


world = BoxAndBlocksWorld
