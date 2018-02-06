''' 
    basic set of ObjectAttributes for PixelWorld
'''
import warnings
from copy import copy
from collections import OrderedDict
from math import pi, floor, sqrt, acos, ceil
from numbers import Number

import numpy as np
from numpy import ma

import core, events
from library.helpers import h

from utils import eps_threshold, fix_float_integer, roundup, is_iterable, \
                    to_vector, sub2ind, ind2sub, Debug, PointMath as pmath



class PositionConflictFactorObjectAttribute(core.ObjectAttribute):
    """an attribute whose value affects whether position conflicts occur. this
    checks position for conflicts every time the attribute value is set.
    """
    _name = 'position_conflict_factor_attribute'
    
    #initialize before position so that initial collisions can properly take
    #the attribute into account
    _initialize_before = ['position']
    
    def set(self, x, value, validate=True):
        """set attribute values, then check for and resolve position conflicts
        involving the Objects whose attribute values were set"""
        idx = self._get_index(x)
        
        super(PositionConflictFactorObjectAttribute, self).set(idx, value, validate=validate)
        
        #resolve position conflicts
        position = self._world._object_attributes.get('position', None)
        if position is not None:
            #get the indices that have position
            idx = to_vector(idx)
            idx = idx[~position._get_mask(idx)]
            
            position.resolve_conflicts(idx)


class VisibleObjectAttribute(   core.BooleanObjectAttribute,
                                core.CoupledFamilyObjectAttribute,
                                PositionConflictFactorObjectAttribute,
                                ):
    """boolean attribute that indicates whether an Object is visible or hidden.
    
    setting this attribute to False causes the Object to "disappear" from the
    world (i.e. it no longer affects the world.state, and will not interact
    with other Objects).
    
    TODO: define AnyChildObjectAttribute, which is a
    CoupledFamilyObjectAttribute that is True if any of a parent's children are
    True
    """
    pass


class VisiblyAffectedObjectAttribute(core.ObjectAttribute):
    """an attribute that returns a special value for Objects that are invisible.
    
    subclasses should define _invisible_value. see zorder, color, and mass for
    examples.
    """
    _name = 'visibly_affected_attribute'
    
    #define the value that the attribute should have for Objects with
    #visible == False
    _invisible_value = 0
    
    _depends_on = ['visible']
    
    _initialize_after = ['visible']
    
    _auto_state = False
    
    def _get_data(self, idx):
        """return the normal value if visible == True, or the invisible value if
        visible == False
        
        Parameters
        ----------
        idx : int | ndarray
            an index or array of indices in the attribute's data array
        
        Returns
        -------
        values : self.dtype | ndarray
            an array of values from the data array
        """
        values = super(VisiblyAffectedObjectAttribute, self)._get_data(idx)
        
        
        visible = self._other_attr['visible']
        if isinstance(idx, Number):
            if not visible.get(idx): values = self._invisible_value
        else:
            values[~visible.get(idx)] = self._invisible_value
        
        return values


class ZorderObjectAttribute(    core.FloatObjectAttribute,
                                VisiblyAffectedObjectAttribute
                                ):
    """ObjectAttribute to indicate the "z-order" of an Object. used by
    PositionObjectAttribute to decide which Object should appear in the world
    state when multiple Objects occupy the same location. Objects with greater
    zorder are essentially "in front" of Objects with lesser zorder.
    """
    _invisible_value = -np.inf


class ColorObjectAttribute( core.NonNegativeIntegerObjectAttribute,
                            core.ModeParentObjectAttribute,
                            core.RandomizingObjectAttribute,
                            VisiblyAffectedObjectAttribute,
                            ):
    """color which, along with position, defines how Objects determine the
    world.state.
    
    colors are non-negative integers. the "color" of a compound object is just
    the most-frequently occurring color among its children.
    """
    _default_value = 1
    _null_value = 0
    _invisible_value = 0

    def _get_random_values(self, indices):
        """return random colors between 1 and 10"""
        return self.rng.randint(1, 10, size=len(indices))


class MassObjectAttribute(  core.NonNegativeFloatObjectAttribute,
                            core.SumParentObjectAttribute,
                            core.RandomizingObjectAttribute,
                            VisiblyAffectedObjectAttribute,
                            PositionConflictFactorObjectAttribute,
                            ):
    """the unit-less mass of an Object. by convention, Objects without the mass
    Object are "immoveable", i.e. the act as if they have infinite mass. Objects
    with zero mass do not physically interact with other Objects.
    
    the mass of a CompoundObject is the sum of its children's masses.
    """
    _default_value = 1
    _null_value = np.inf
    _invisible_value = 0
    
    def _get_random_values(self, idx):
        """Get a random mass. Return a random float between 1 and 10.
        Parameters
        ----------
        idx : int | ndarray
            an index or array of indices of objects
        """
        if isinstance(idx, np.ndarray):
            return 1 + 9 * self.rng.rand(len(idx))
        else:
            return 1 + 9 * self.rng.rand()


class DepthObjectAttribute( core.IntegerObjectAttribute,
                            core.CoupledFamilyObjectAttribute,
                            PositionConflictFactorObjectAttribute,
                            ):
    """the Object's current plane of existence"""
    
    _coupled_siblings = True
    
    _null_value = np.iinfo(np.int64).min
    _default_value = _null_value

    def _coerce_value(self, value, num_values=None, **kwargs):
        """make sure the values are within the bounds of the world depth"""
        depth = getattr(self._world, 'depth', 0)
        
        if depth != 0:
            valid = np.all((value == self._null_value) | ((value >= 0) & (value < depth)))
        else:
            valid = True
        
        err = False if valid else TypeError('must be a valid depth (0 <= depth < %d)' % (depth))
        
        return value, err
    
    def _get_random_values(self, idx):
        """Get a random depth. Return a random int between 0 and world.depth.

        Parameters
        ----------
        idx : int | ndarray
            an index or array of indices of objects
        """
        max_depth = self.world.depth
        if max_depth == 0:
            max_depth = 10

        if isinstance(idx, np.ndarray):
            return self.rng.randint(max_depth, size=idx.size)
        else:
            return self.rng.randint(max_depth)


class AccelerationObjectAttribute(  core.PointObjectAttribute,
                                    core.CoupledFamilyObjectAttribute,
                                    core.SteppingObjectAttribute,
                                    core.RandomizingObjectAttribute,
                                    ):
    """acceleration increments velocity by itself on every step.
    
    acceleration is a 2D vector. all Objects in an object family have the same
    acceleration (i.e. CompoundObjects are rigid).
    """
    _depends_on = ['velocity']
    
    #so acceleration affects velocity before it changes positions
    _step_before = ['velocity']
    
    def _get_random_values(self, indices):
        """return random accelerations with components between -1 and 1"""
        return -1 + 2 * self.rng.rand(len(indices), 2)

    def _step(self, t, dt, agent_id, action):
        velocity = self._other_attr['velocity']
        idx = self.object_indices
        
        #current velocity
        v = velocity.get(idx)
        
        #change in velocity
        dv = self.get(idx) * dt
        
        #increment the velocity
        velocity.set(idx, v + dv)


class VelocityObjectAttribute(  core.PointObjectAttribute,
                                core.CoupledFamilyObjectAttribute,
                                core.SteppingObjectAttribute,
                                core.RandomizingObjectAttribute,
                                ):
    """velocity increments position by itself on every step.
    
    velocity is a 2D vector. all Objects in an object family have the same
    velocity (i.e. CompoundObjects are rigid).
    """
    _depends_on = ['position']
    
    #so positions previous value registers before this changes the position
    _step_after = ['position']

    def _get_random_values(self, indices):
        """return random velocities with components between -1 and 1"""
        return -1 + 2 * self.rng.rand(len(indices), 2)
    
    @property
    def _substep_factor(self):
        """velocity needs to substep so that collisions aren't missed by the
        discretization of position changes, but we want to keep this as low as
        possible since the velocity step function will probably usually be the
        most expensive aspect of the simulation. we only need to make the
        substep resolution fine enough so that the fastest moving object moves
        by no more than 0.5 unit in either direction (0.5 rather than 1 so that
        we don't have e.g. object end up colliding on the wrong side of the
        frame and get bounced off into infinity).
        """
        if len(self._data) > 0:
            return max(1, ceil(2*np.max(np.abs(self._data))))
        else:
            return 1
    
    def _step(self, t, dt, agent_id, action):
        position = self._other_attr['position']
        idx = self.object_indices
        
        #current position
        p = position.get(idx)
        
        #change in velocity
        dp = self.get(idx) * dt
        
        #increment the position
        position.set(idx, p + dp)


class StateIndexObjectAttribute(core.PointObjectAttribute, core.DerivedObjectAttribute):
    """the state array index associated with an Object's position"""
    _depends_on = ['position']
    _dtype = np.dtype(int)

    def _get_data(self, idx):
        """convert positions to their equivalent state array indices

        Parameters
        ----------
        idx : int or ndarray
            Object indices.

        Returns
        -------
        values : self.dtype | ndarray
            a state index or N x 2 array of state indices
        """
        position = self._other_attr['position'].get(idx)
        return roundup(position).astype(int)


class PositionXObjectAttribute(core.FloatObjectAttribute, core.DerivedObjectAttribute):
    """the x coordinate of an Object's position"""
    _depends_on = ['position']

    def _get_data(self, idx):
        """extract x from positions

        Parameters
        ----------
        idx : int or ndarray
            Object indices.

        Returns                                                                                                                                                                                                                           
        -------
            a float or length N array of floats
        """
        position = self._other_attr['position'].get(idx)
        if position.ndim == 2:
            return position[:, 1]
        else:
            return position[1]

    def _set_data(self, idx, value):
        """set x-values of positions

        Parameters
        ----------
        idx : int or ndarray
            Object indices.

        value : float or ndarray
            Value to set the x-positions to
        """
        position = self._other_attr['position'].get(idx)
        if position.ndim == 2:
            position[:, 1] = value
        else:
            position[1] = value
        self._other_attr['position'].set(idx, position)

    def _get_random_values(self, idx):
        position = self._other_attr['position']._get_random_values(idx)
        if position.ndim == 2:
            return position[:, 1]
        else:
            return position[1]        


class PositionYObjectAttribute(core.FloatObjectAttribute, core.DerivedObjectAttribute):
    """the y coordinate of an Object's position"""
    _depends_on = ['position']

    def _get_data(self, idx):
        """extract y from positions

        Parameters
        ----------
        idx : int or ndarray
            Object indices.

        Returns                                                                                                                                                                                                                           
        -------
            a float or length N array of floats
        """
        position = self._other_attr['position'].get(idx)
        if position.ndim == 2:
            return position[:, 0]
        else:
            return position[0]

    def _set_data(self, idx, value):
        """set y-values from positions

        Parameters
        ----------
        idx : int or ndarray
            Object indices.
        value : float or ndarray
            Value to set the x-positions to
        """
        position = self._other_attr['position'].get(idx)
        if position.ndim == 2:
            position[:, 0] = value
        else:
            position[0] = value
        self._other_attr['position'].set(idx, position)

    def _get_random_values(self, idx):
        position = self._other_attr['position']._get_random_values(idx)
        if position.ndim == 2:
            return position[:, 0]
        else:
            return position[0]


class TopLeftObjectAttribute(core.PointObjectAttribute, core.DerivedObjectAttribute):
    """the top-most and left-most coordinates of an Object's pixels"""
    _name = 'top_left'
    _depends_on = ['position', 'extent']
    _initialize_after = ['position', 'extent']
    _read_only = False

    def _get_data(self, idx):
        """extract top-left coordinates from extent

        Parameters
        ----------
        idx : int or ndarray
            Object indices.

        Returns
        -------
        value : ndarray
            a length-2 array of floats or an Nx2 array of floats
        """

        # extract top and left coordinates from extent
        if is_iterable(idx):
            extents = self._other_attr['extent'].get(idx)
            return extents[:, :2]
        else:
            t, l, b, r = self._other_attr['extent'].get(idx)
            return np.hstack((t, l))

    def _set_data(self, idx, value):
        """set coordinates of an Object's top-left-most pixel

        Parameters
        ----------
        idx : int or ndarray
            Object indices.
        value : float or ndarray
            Value to set the top-left coordinates to
        """
        # calculate offset based on current position and current top_left
        cur_position = self._other_attr['position'].get(idx)
        cur_top_left = self.get(idx)
        new_position = cur_position + value - cur_top_left

        if is_iterable(idx):
            # break up sets to enforce coupling of position
            for id, posn in zip(idx, new_position):
                self._other_attr['position'].set(id, posn)
        else:
            self._other_attr['position'].set(idx, new_position)


class PositionObjectAttribute(  core.PointObjectAttribute,
                                core.RigidFamilyObjectAttribute,
                                core.ChangeTrackingObjectAttribute,
                                core.RandomizingObjectAttribute,
                                core.ConflictObjectAttribute,
                                ):
    """position which, along with color, depth, and zorder, defines how Objects
    determine the world.state.
    
    no two non-zero mass Objects at the same depth can occupy the same position.
    if this happens, a collision occurs and is resolved.
    
    position is a 2D vector. a CompoundObject's position is the mean position of
    its children.
    """
    _depends_on = ['zorder']
    
    #moving a child moves its siblings
    _coupled_siblings = True

    #maximum distance between two Objects for a collision to occur
    _collision_threshold = 1 - eps_threshold
    
    _auto_state_exclude = ['_collision_threshold']
    
    #for keeping track of visible Object location indices
    _data_fields = {
        '_object_loc': {
            'field':        lambda (self): '%s_location' % (self._name),
            'dtype':        np.dtype(int),
            'null_value':   -1,
            'ndim':         1,
            },
    }
    
    def get_visible_indices(self, within_window=True):
        """get the indices of the objects that are visible, and the state
        indices associated with their positions
        
        Parameters
        ----------
        within_window : bool
            True to only return indices within the visible state window
        
        Returns
        -------
        idx : ndarray
            an array of visible Object indices
        loc : ndarray
            a tuple of the visible Objects' state indices, useful for indexing
            into the state array
        """
        if within_window:
            #indices of Objects within the visible window
            idx = np.where(self._object_loc.filled() != -1)[0]
        else:
            #all simple Objects
            idx = self._get_object_indices(object_type=core.Object.SIMPLE)
        
        #order by depth and then by zorder, so that Objects with lower depth and
        #higher zorder take precedence (depth takes precedence over zorder)
        zorder = self._other_attr['zorder'].get(idx)
        depth = -self._get_effective_depth(idx)
        idx_sort = np.lexsort((zorder, depth))
        idx = idx[idx_sort]
        
        #get the state indices of the visible Objects
        loc = ind2sub(self._world.shape, self._object_loc[idx])
        
        return idx, loc
    
    def _get_random_values(self, indices):
        """return random positions for all objects indicated. see
        _get_random_value_object for details."""
        objects = self._world.objects[indices]
        
        rv = np.zeros((len(objects), self._ndim))
        for idx,obj in enumerate(objects):
            rv[idx] = self._get_random_value_object(obj)
        
        return rv

    def _get_random_value_object(self, obj, shape=None):
        """Randomized value is a random point within the state window. For
        CompoundObjects, the random position is kept such that no part of the
        CompoundObject falls outside of the window.
        
        Parameters
        ----------
        obj : Object
            the Object for which to choose a position
        shape : list[(y, x)] | array_like
            a list/array of the relative positions of a ShapeObject's children
        """
        #get the shape extrema
        if shape is None:
            if hasattr(obj, 'shape') and len(obj.shape) > 0:
                shape_min = np.min(obj.shape, axis=0)
                shape_max = np.max(obj.shape, axis=0)
            else:
                shape_min = np.array([0, 0])
                shape_max = shape_min
        else:
            shape_min = np.min(shape, axis=0)
            shape_max = np.max(shape, axis=0)
        
        #random position that makes sure there is at least one empty space in
        #between the object and the frame
        return np.array([obj.rng.randint(2 - mn, d - 3 - mx) for d,mn,mx in \
                zip(self.world.shape, shape_min, shape_max)], dtype=int)

    def _default_value(self, obj, shape=None):
        """default value is a random point within the state window. for
        CompoundObjects, the random position is kept such that no part of the
        CompoundObject falls outside of the window.
        
        Parameters
        ----------
        obj : Object
            the Object for which to choose a position
        shape : list[(y, x)] | array_like
            a list/array of the relative positions of a ShapeObject's children
        """
        return self._get_random_value_object(obj, shape)
    
    def _get_effective_depth(self, x):
        """get the effective depths of a set of Objects
        
        Parameters
        ----------
        x : Object | int | ndarray
            the Objects/indices
        
        Returns
        -------
        d : int | ndarray
            the effective depths
        """
        depth = self._world._object_attributes.get('depth', None)
        
        if depth is None:
            d_column = DepthObjectAttribute._null_value
            if isinstance(x, np.ndarray):
                return d_column * np.ones(x.shape, dtype=DepthObjectAttribute._dtype)
            else:
                return d_column
        else:
            return depth.get(x)
    
    def _get_effective_mass(self, obj):
        """get the effective mass of an Object, for computing collisions.
        simple Objects take on the mass of their parents, Objects without mass
        effectively have infinite mass.
        
        Parameters
        ----------
        obj : Object
            an Object
        
        Returns
        -------
        m : float
            the Object's mass
        """
        idx = obj._head_of_family._id
        
        mass = self._world._object_attributes.get('mass', None)
        
        return MassObjectAttribute._null_value if mass is None else mass.get(idx)
    
    def _get_effective_masses(self, obj1, obj2):
        """get the effective masses of two Objects for the purpose of
        calculating a collision. this is like _get_effective_mass, except that
        it takes into account different classes of infinite mass Objects:
        Objects without mass and without velocity (immoveable) are treated as
        infinitely more massive than Objects without mass but with velocity
        (unpushable).
        
        Parameters
        ----------
        obj1 : Object
            the first Object
        obj2 : Object
            the second Object
        
        Returns
        -------
        m1 : float
            the first Object's effective mass
        m2 : float
            the second Object's effective mass
        fm1 : float
            the first Object's effective mass as a fraction of the total mass of
            the two Objects
        fm2 : float
            the second Object's effective mass as a fraction of the total mass
            of the two Objects
        """
        m1 = self._get_effective_mass(obj1)
        m2 = self._get_effective_mass(obj2)
        
        #mass proportions
        if np.isinf(m1):
            if np.isinf(m2):
                #immoveable objects beat everything, unpushable objects beat
                #objects with mass
                push1 = obj1._get('pushable')
                move1 = obj1._get('moveable')
                push2 = obj2._get('pushable')
                move2 = obj2._get('moveable')
                
                if move1:
                    if move2:
                        if push1:
                            if push2:
                                fm1 = 0.5
                            else:
                                fm1 = 0
                        elif push2:
                            fm1 = 1
                        else:
                            fm1 = 0.5
                    else:
                        fm1 = 0
                elif move2:
                    fm1 = 1
                else:
                    fm1 = 0.5
            else:
                fm1 = 1
        elif np.isinf(m2):
            fm1 = 0
        else:
            total_mass = m1 + m2
            
            if total_mass==0:
                fm1 = 0.5
            else:
                fm1 = m1/float(total_mass)
        
        fm2 = 1 - fm1
        
        return m1, m2, fm1, fm2
    
    def _get_effective_velocity(self, x):
        """get the effective velocities of a set of Objects. for Objects without
        a velocity attribute, velocity is calculated implicitly as the
        within-step change in position.
        
        Parameters
        ----------
        x : Object | int | ndarray
            the Objects/indices
        
        Returns
        -------
        v : int | ndarray
            the effective velocities
        """
        velocity = self._world._object_attributes.get('velocity', None)
        
        if velocity is None:
            return self.change(x, step=True)
        else:
            idx = to_vector(self._get_index(x))
            
            value = velocity.get(idx)
            
            no_velocity = velocity._get_mask(idx)
            if np.any(no_velocity):
                value[no_velocity] = self.change(idx[no_velocity], step=True)
            
            if isinstance(x, np.ndarray):
                return value
            else:
                return value[0]
    
    def _calculate_time_to_wall(self, p, v):
        """calculate the amount of time it takes for a point to move from
        somewhere in the interval (-1, 1) to the edge of that interval
        
        Parameters
        ----------
        p : Number
            the starting position of the point
        v : Number
            the velocity of the point
        
        Returns
        -------
        t : float | None
            the amount of time until the point hits the edge of the interval, or
            None if it is not moving
        """
        if v==0:
            return None
        elif v>0:
            return (1-p)/v
        else:
            return -(1+p)/v
    
    def _is_collision_object_at_state_index(self, state_index, d, p, v):
        """determine whether an Object exists at the specified state_index and
        is both at a collidable depth and moving toward the specified test
        Object at the specified point
        
        Parameters
        ----------
        state_index : ndarray
            the state index to check
        d : int
            the test depth
        p : ndarray
            the test position
        v : ndarray
            the test velocity
        
        Returns
        -------
        obj_at : bool
            True if there is colliding Object at the specified location
        """
        loc = sub2ind(self._world.shape, *tuple(state_index))
        
        idx_at = np.where(self._object_loc == loc)[0]
        
        if idx_at.size == 0:
            return False
        
        #keep only objects at the same depth
        d_column = DepthObjectAttribute._null_value
        if d != d_column:
            d_at = self._get_effective_depth(idx_at)
            idx_at = idx_at[(d_at == d) | (d_at == d_column)]
        
        if idx_at.size == 0:
            return False
        
        #keep only objects that are moving toward the reference object
        dp = self.get(idx_at) - p
        dv = self._get_effective_velocity(idx_at) - v
        
        moving_toward = np.any(dp * dv > 0, axis=-1)
        
        return np.any(moving_toward)
    
    def _calculate_effective_collision_location(self, idx1, p1, idx2, p2):
        """calculate the effective relative location of two Objects when they
        collided. collisions are constrained to having occurred at one of eight
        relative grid locations, i.e.:
            123
            4X5
            678
        
        Parameters
        ----------
        idx1 : int
            the index of the first Object
        p1 : ndarray
            the position of the first Object
        idx2 : int
            the index of the second Object
        p2 : ndarray
            the position of the second Object
        
        Returns
        -------
        dp : ndarray
            the effective relative location (i.e. p2 - p1) of the two Objects at
            the time of collision
        """
        world_shape = self._world.shape
        
        #current relative positions
        dp = p2 - p1
        
        #index locations
        p1_idx = self._position_to_index(p1, to_tuple=False)
        p2_idx = self._position_to_index(p2, to_tuple=False)
        
        #calculate the collision as if it happened at one of the eight
        #relative grid locations (i.e. with sides perfectly aligned, or
        #in which two corners collide)
        adp = np.abs(dp)
        old_dp = copy(dp)
        if adp[0] > adp[1]:  # collision along horizontal edges
            dp[0] = 1 if dp[0]>0 else -1
            dp[1] = 0
        elif adp[1] > adp[0]:  # collision along vertical edges
            dp[0] = 0
            dp[1] = 1 if dp[1]>0 else -1
        else:  # collision at corners
            #make sure there aren't neighboring objects that would make a
            #corner collision silly
            
            #get the locations to test for Objects
            idx_check1 = list(p1_idx)
            idx_check2 = list(p1_idx)
            p2_above = p2_idx[0] > p1_idx[0]
            p2_right = p2_idx[1] > p1_idx[1]
            check1_offset = 1 if p2_above else -1
            check2_offset = 1 if p2_right else -1
            idx_check1[0] += check1_offset
            idx_check2[1] += check2_offset
            
            #determine whether Objects exist at the test locations
            #TODO this is not correct, since it is testing for Objects based on
            #their current locations, while the p1/p2 passed in here are the
            #position at some time in the past. this whole thing is a mess, and
            #we should clean it up.
            d = self._get_effective_depth(idx1)
            v = self._get_effective_velocity(idx1)
            obj_at1 = self._is_collision_object_at_state_index(idx_check1, d, p1, v)
            obj_at2 = self._is_collision_object_at_state_index(idx_check2, d, p1, v)
            
            if obj_at1 and not obj_at2:  #collision along vertical edges
                dp[0] = 1 if dp[0]>0 else -1
                dp[1] = 0
            elif obj_at2:  #collision along horizontal edges
                dp[0] = 0
                dp[1] = 1 if dp[1]>0 else -1
            else:  #actual corner collision
                m = 1/sqrt(2)
                dp[0] = m if dp[0]>0 else -m
                dp[1] = m if dp[1]>0 else -m
        
        #if there is already an Object at the effective p1, then we have the
        #wrong collision location, so we'll assume the Objects collided along
        #the other sides (across the corner)
        p1_idx_eff = self._position_to_index(p2 - dp)
        loc_eff = sub2ind(world_shape, *p1_idx_eff)
        idx_at = np.where(self._object_loc == loc_eff)[0]
        if idx_at.size != 0 and np.any(idx_at != idx1):  # different Object exists
            if dp[0] != 0:  #vertical -> horizontal wall collision
                dp[0] = 0
                dp[1] = 1 if old_dp[1]>0 else -1
            else:  # horizontal -> vertical wall collision
                dp[0] = 1 if old_dp[0]>0 else -1
                dp[1] = 0
        
        return dp
    
    def _calculate_elastic_collision(self, idx1, fm1, p1, v1, idx2, fm2, p2, v2):
        """calculate the new velocities of two unit length, fixed-axis-aligned
        squares that undergo an elastic collision
        
        Parameters
        ----------
        idx1 : int
            index of the first Object
        fm1 : Number
            fractional mass of the first square (relative to the total mass)
        p1 : Number
            first square's position
        v1 : ndarray
            first square's velocity
        idx2 : int
            index of the second Object
        fm2 : Number
            fractional mass of the second square
        p2 : Number
            second square's position
        v2 : Number
            second square's velocity
        
        Returns:
        -------
        p1 : Number
            first square's position after the collision
        v1 : Number
            first square's velocity after the collision
        p2 : Number
            second square's position after the collision
        v2 : Number
            second square's velocity after the collision
        """
        p1 = p1.copy()
        p2 = p2.copy()
        v1 = v1.copy()
        v2 = v2.copy()
        
        #relative velocities
        dv = v2 - v1
        
        if any(dv): # collision only occurs if the squares are moving relative to each other
            #relative positions
            dp = p2 - p1
            
            #time to nearest wall
            ty = self._calculate_time_to_wall(dp[0], -dv[0])
            tx = self._calculate_time_to_wall(dp[1], -dv[1])
            if ty is None:
                tmin = tx
            elif tx is None:
                tmin = ty
            else:
                tmin = min(ty, tx)
            
            #run time backwards to where they were when they collided
            p1 -= v1*tmin
            p2 -= v2*tmin
            
            #relative point of collision
            dp = self._calculate_effective_collision_location(idx1, p1, idx2, p2)
            
            #direction of velocity change
            dv_dot_dp = np.sum(dv*dp)
            delta_v = dv_dot_dp * dp
            
            #update the velocities
            if fm2!=0:
                v1 += 2*fm2*delta_v
            if fm1!=0:
                v2 -= 2*fm1*delta_v
            
            #now run time forward again
            p1 += v1*tmin
            p2 += v2*tmin
        else:  # otherwise, just bounce the squares off of each other
            p1, p2 = self._calculate_bounce(idx1, fm1, p1, v1, idx2, fm2, p2, v2)
        
        return p1, v1, p2, v2
    
    def _calculate_bounce(self, idx1, fm1, p1, v1, idx2, fm2, p2, v2):
        """calculate the new positions of squares that "bounce" off of each
        other to non-overlapping locations
        
        Parameters
        ----------
        idx1 : int
            index of the first object
        fm1 : Number
            fractional mass of the first square (relative to the total mass)
        p1 : Number
            first square's position
        v1 : ndarray
            first square's velocity
        idx2 : int
            index of the second object
        fm2 : Number
            fractional mass of the second square
        p2 : Number
            second square's position
        v2 : Number
            second square's velocity
        
        Returns:
        -------
        p1 : Number
            the first object's position after the bounce
        p2 : Number
            the second object's position after the bounce
        """
        p1 = p1.copy()
        p2 = p2.copy()
        
        #relative positions
        dp = p2 - p1
        
        #no bounce is necessary if the squares aren't touching
        if any(dp >= 1):
            return p1, p2
        
        #if both velocities are zero, use position change instead
        if np.all(v1 == 0) and np.all(v2 == 0):
            v1 = self.change(idx1, step=True)
            v2 = self.change(idx2, step=True)
        else:
            v1 = v1.copy()
            v2 = v2.copy()
        
        #relative velocities
        dv = v2 - v1
        
        if any(dv):  # continue along the current trajectories
            #scale the velocities by the angle between them, so that as the
            #angle becomes smaller, the object that was already moving faster
            #does more of the bouncing. this prevents small differences between
            #the velocities from causing the bounce distances to be large
            a = pmath.angle_abs(v1, v2)
            a_factor = 1 - (pi - a)/pi
            if pmath.magnitude2(v2) > pmath.magnitude2(v1):
                v1 *= a_factor
            else:
                v2 *= a_factor
            
            #updated velocity difference
            dv = v2 - v1
            
            #time to separation
            ty = self._calculate_time_to_wall(dp[0], dv[0])
            tx = self._calculate_time_to_wall(dp[1], dv[1])
            if ty is None:
                tmin = tx
            elif tx is None:
                tmin = ty
            else:
                tmin = min(ty, tx)
            
            #bounce until separation
            p1 += v1*tmin
            p2 += v2*tmin
        elif any(dp):  # bounce along the separation line
            #find the minimum bounce distance that will separate at least one
            #of the coordinates by one unit
            adp = abs(dp)
            if adp[0] > adp[1]:
                bounce_factor = (1 - adp[0]) / adp[0]
            else:
                bounce_factor = (1 - adp[1]) / adp[1]
            
            bounce = dp * bounce_factor
            
            #bounce inversely proportional to mass
            p1 -= bounce*fm2
            p2 += bounce*fm1
        else:
            #objects are perfectly overlapped and stationary relative to each
            #other. this is getting pretty weird. just move the less massive
            #object to an unoccupied point.
            if fm1 > fm2:
                d2 = self._get_effective_depth(idx2)
                p2 = self._find_unoccupied_point(p2, d2)
            else:
                d1 = self._get_effective_depth(idx1)
                p1 = self._find_unoccupied_point(p1, d1)
        
        return p1, p2
    
    def _collide(self, idx1, idx2):
        """simulate an elastic collision between two Objects. if the objects
        have the same velocity, then just bounce them away from each other to
        eliminate position overlap.
        
        Parameters
        ----------
        idx1 : int
            the index of the first Object
        idx2 : int
            the index of the second Objects
        
        Returns
        -------
        success : bool
            always True, since the collision is always successful
        """
        print_debug = self._world.debug.test(Debug.INFO)
        
        obj1 = self._world.objects[idx1]
        obj2 = self._world.objects[idx2]
        
        #current positions
        op1 = self.get(obj1)
        op2 = self.get(obj2)
        
        #current velocities
        ov1 = self._get_effective_velocity(obj1)
        ov2 = self._get_effective_velocity(obj2)
        
        #masses
        m1, m2, fm1, fm2 = self._get_effective_masses(obj1, obj2)
        
        #get some additional starting values for debugging
        if print_debug:
            oke1 = getattr(obj1, 'kinetic_energy', 0)
            oke2 = getattr(obj2, 'kinetic_energy', 0)
            owe = self._world.energy
        
        #if the two objects are moving away from each other in each direction,
        #then something strange like a recoil involving a compound object
        #happened. in this case, just bounce the objects apart.
        dp = op2 - op1
        dv = ov2 - ov1
        if np.all(dp*dv >= 0) and not np.all(dp == 0):
            p1, p2 = self._calculate_bounce(idx1, fm1, op1, ov1, idx2, fm2, op2, ov2)
            
            v1 = ov1.copy()
            v2 = ov2.copy()
        else:  # calculate the elastic collision or bounce
            p1, v1, p2, v2 = self._calculate_elastic_collision(idx1, fm1, op1, ov1, idx2, fm2, op2, ov2)
        
        #apply the velocity changes
        if any(v1!=ov1) and obj1._get('moveable'):
            obj1._set('velocity', fix_float_integer(v1), validate=False)
        if any(v2!=ov2) and obj2._get('moveable'):
            obj2._set('velocity', fix_float_integer(v2), validate=False)
        
        #show debug info here in case setting the position causes more conflicts
        if print_debug:
            ke1 = getattr(obj1, 'kinetic_energy', 0)
            ke2 = getattr(obj2, 'kinetic_energy', 0)
            we = self._world.energy
            
            description = 'elastic collision between %s and %s' % (obj1.unique_name, obj2.unique_name)
            info = OrderedDict([
                ('before', OrderedDict([
                    ('position', [op1, op2]),
                    ('velocity', [ov1, ov2]),
                    ('mass', [m1, m2]),
                    ('energy', [oke1, oke2, (oke1+oke2,)]),
                    ('world energy', owe),
                ])),
                ('after', OrderedDict([
                    ('position', [p1, p2]),
                    ('velocity', [v1, v2]),
                    ('energy', [ke1, ke2, (ke1+ke2,)]),
                    ('world energy', we),
                ])),
            ])
            
            self._world.debug.log(description, info, level=Debug.INFO, hr=True)
        
        #apply the position changes
        resolve_after = self._defer_conflicts()
        
        if any(p1 != op1):
            self.set(obj1, p1, validate=False)
        
        if any(p2 != op2):
            self.set(obj2, p2, validate=False)
        
        if resolve_after:
            self._resolve_deferred_conflicts()
        
        return True
    
    def _find_unoccupied_point(self, p, d=DepthObjectAttribute._null_value):
        """find a point near p that is unoccupied at the indicated depth
        
        Parameters
        ----------
        p : ndarray
            a position
        d : int, optional
            a depth
        
        Returns
        -------
        p : ndarray
            an unoccupied position near p
        """
        world_shape = self._world.shape
        height, width = world_shape
        
        p = self._position_to_index(p, to_tuple=False)
        
        #info for determining search direction
        direction = 'down'
        num_steps = 0
        step = 0
        
        #start at the point and spiral outward, looking for unoccupied points
        while True:
            d_column = DepthObjectAttribute._null_value
            
            #determine whether the current point is occupied
            if p[0]>=0 and p[0]<height and p[1]>=0 and p[1]<width:  # quick way if the point is visible
                #state index code
                loc = sub2ind(world_shape, *tuple(p))
                
                #objects at the indicated position
                idx = np.where(self._object_loc == loc)[0]
                
                #keep only the visible ones
                if idx.size > 0:
                    idx = idx[self.world.object_attributes['visible'].get(idx)]
                
                if idx.size > 0:
                    if d == d_column:
                        point_occupied = True
                    else:
                        d_other = self._get_effective_depth(idx)
                        point_occupied = np.any(d_other == d) or np.any(d_other == d_column)
                else:
                    point_occupied = False
            else:  # slow way if the point is off-screen
                search = {
                    'position': p,
                    'object_type': 'SIMPLE',
                    'visible': True,
                    }
                
                if 'depth' in self._world._object_attributes:
                    point_occupied = self._world.objects.get(depth=d, **search) is not None or \
                        self._world.objects.get(depth=d_column, **search) is not None
                else:
                    point_occupied = self._world.objects.get(**search) is not None
            
            #we're done if it is unoccupied
            if not point_occupied:
                return p
            
            #check for a change in search path direction
            if step == num_steps:
                step = 0
                
                if direction == 'left':
                    direction = 'up'
                    dp = (-1, 0)
                elif direction == 'up':
                    num_steps += 1
                    direction = 'right'
                    dp = (0, 1)
                elif direction == 'right':
                    direction = 'down'
                    dp = (1, 0)
                elif direction == 'down':
                    num_steps += 1
                    direction = 'left'
                    dp = (0, -1)
                else:
                    raise ValueError('wtf?')
            
            #step to the next position
            p += dp
            
            #increment along the path
            step += 1
    
    def _compare(self, x, y):
        """determine whether two points share the same state index
        
        Parameters
        ----------
        x : ndarray
            a position or N x 2 array of positions
        y : ndarray
            another position or N x 2 array of positions
        
        Returns
        -------
        b : bool | ndarray
            a boolean or boolean array indicating whether the positions share
            the same index
        """
        x_index = self._position_to_index(x, to_tuple=False).T
        y_index = self._position_to_index(y, to_tuple=False).T
        
        return np.all(x_index == y_index, axis=1)
    
    def _conflicts(self, idx1, idx2):
        """determine whether two Objects are in conflicting positions. two
        Objects' positions conflict if neither Object has zero mass and their
        positions are less than the collision threshold apart. this overrides
        ConflictObjectAttribute's method."""
        #zero mass objects can occupy any position
        if 'mass' in self._world._object_attributes:
            mass = self._world._object_attributes['mass']
            m1 = mass.get(idx1)
            m2 = mass.get(idx2)
            
            conflicts = np.all((m1, m2), axis=0)
        else:
            conflicts = np.ones(idx1.shape, dtype=bool)
        
        #objects at different depths do not interact
        if 'depth' in self._world._object_attributes:
            depth = self._world._object_attributes['depth']
            d1 = depth.get(idx1[conflicts])
            d2 = depth.get(idx2[conflicts])
            
            #same depth or either is a column depth
            d_column = DepthObjectAttribute._null_value
            conflicts[conflicts] = (d1 == d2) | (d1 == d_column) | (d2 == d_column)
        
        #check for position overlap among the remaining objects
        p1 = self.get(idx1[conflicts])
        p2 = self.get(idx2[conflicts])
        conflicts[conflicts] = self._positions_overlap(p1, p2)
        
        return conflicts
    
    def _conflicts_with_value(self, idx, value):
        """determine whether an Objects' position conflicts with a point. an
        Object's position conflicts with a point if the Object does not have
        zero mass and its position is less than the collision threshold away
        from the point. this overrides ConflictObjectAttribute's method."""
        #zero mass objects can occupy any position
        if 'mass' in self._world._object_attributes:
            m = self._world._object_attributes['mass'].get(idx)
            
            conflicts = m != 0
        else:
            conflicts = np.ones(idx.shape, dtype=bool)
        
        #check for position overlap among the remaining objects
        p = self.get(idx[conflicts])
        conflicts[conflicts] = self._positions_overlap(p, value[conflicts])
        
        return conflicts
    
    def _position_to_index(self, p, to_tuple=True):
        """convert positions to their equivalent state array indices
        
        Parameters
        ----------
        p : ndarray
            a position or N x 2 array of positions
        to_tuple : bool, optional
            True to convert the index array to a tuple that can be used directly
            for indexing
        """
        idx = roundup(p).astype(int).T
        
        return tuple(idx) if to_tuple else idx
    
    def _positions_overlap(self, p1, p2):
        """determine whether two points overlap (i.e. their relative distance is
        less than the collision threshold)
        
        Parameters
        ----------
        p1 : ndarray
            a point or N x 2 array of points
        p2 : ndarray
            another point or N x 2 array of points
        
        Returns
        -------
        overlap : ndarray
            a boolean array indicating which pairs of points overlap
        """
        return np.all(np.abs(p1 - p2) < self._collision_threshold, axis=-1)
    
    def _position_is_visible(self, p, to_tuple=True):
        """determine whether positions are within the viewable world.state
        window, and return the state array indices associated with the visible
        positions
        
        Parameters
        ----------
        p : ndarray
            a position or N x 2 array of positions
        to_tuple : bool, optional
            True to convert the indices to a tuple for direct indexing into an
            array
        
        Returns
        -------
        visible : ndarray
            a boolean array indicating which positions are visible
        p_idx : tuple | ndarray | None
            an array if state array indices associated with the visible
            positions, or None if there were no visible positions. if to_tuple
            is True, the array is converted to a tuple for direct array
            indexing.
        """
        #get the position indices
        p_idx = self._position_to_index(p, to_tuple=False)
        y = p_idx[0]
        x = p_idx[1]
        
        #test whether they are in the world bounds
        visible = to_vector(y >= 0)
        np.logical_and(visible, y < self._world.height, out=visible)
        np.logical_and(visible, x >= 0, out=visible)
        np.logical_and(visible, x < self._world.width, out=visible)
        
        if p_idx.ndim == 1:  # single position
            if not visible:
                return visible, None
        else:  # array of positions
            if np.any(visible):
                p_idx = p_idx[:,visible]
            else:
                return visible, None
        
        if to_tuple:
            p_idx = tuple(p_idx)
        
        return visible, p_idx
    
    def _resolve_single_conflict(self, idx1, idx2):
        """resolve a collision between two Objects
        
        Parameters
        ----------
        idx1 : int
            the index of the first Object in the collision
        idx2 : int
            the index of the second Object in the collision
        
        Returns
        -------
        success : bool
            True if the collision succeeded (which it always will)s
        """
        #register a collision event
        event = events.CollisionEvent(self._world, indices=[idx1, idx2])
        
        return self._collide(idx1, idx2)
    
    def _set_data(self, idx, value):
        """set position data in the data array. we override the default method
        so we can additionally keep track of visible positions that are
        occupied.
        
        Parameters
        ----------
        idx : ndarray
            an array of Object indices
        value : ndarray
            an array of positions to set
        """
        #only keep track of simple objects
        simple = self._world._object_types[idx] == core.Object.SIMPLE
        
        if np.any(simple):
            simple_idx = idx[simple]
            simple_value = value[simple]
            
            #mark the new occupied positions
            visible, p_idx = self._position_is_visible(simple_value)
            if p_idx is not None:
                visible_idx = simple_idx[visible]
                self._object_loc[visible_idx] = sub2ind(self._world.shape, *p_idx)
            
            #unmark the invisible positions
            invisible_idx = simple_idx[~visible]
            if invisible_idx.size != 0:
                #create a leave event if any objects just left the screen
                just_left = self._object_loc[invisible_idx] != -1
                new_idx = invisible_idx[just_left]
                if new_idx.size != 0:
                    event = events.LeaveScreenEvent(self._world, indices=new_idx)
                
                self._object_loc[invisible_idx] = -1
        
        super(PositionObjectAttribute, self)._set_data(idx, value)


class MomentumObjectAttribute(  core.PointObjectAttribute,
                                core.DerivedObjectAttribute,
                                ):
    """momentum = mass * velocity"""
    _depends_on = ['mass', 'velocity']
    
    def _get_data(self, idx):
        mass = self._other_attr['mass'].get(idx)
        
        if is_iterable(idx):
            mass = np.tile(np.reshape(mass, (-1,1)), (1, 2))
        else:
            mass = np.tile(np.reshape(mass, (-1,1)), (1, 2))[0]
        
        velocity = self._other_attr['velocity'].get(idx)
        
        #so we don't get warnings
        if not isinstance(mass, Number):
            velocity[mass == np.inf] = 1
        elif mass == np.inf:
            velocity[:] = 1
        
        return mass * velocity


class KineticEnergyObjectAttribute( core.NonNegativeFloatObjectAttribute,
                                    core.DerivedObjectAttribute,
                                    ):
    """kinetic_energy = 1/2 * mass * velocity^2"""
    _depends_on = ['mass', 'velocity']
    
    def _get_data(self, idx):
        mass = self._other_attr['mass'].get(idx)
        
        #we don't want infinitely massive Objects screwing up energy calculations
        if not isinstance(mass, Number):
            mass[mass == np.inf] = 0
        elif mass == np.inf:
            mass = 0
        
        velocity = self._other_attr['velocity'].get(idx)
        
        return mass * pmath.magnitude2(velocity) / 2


class ExtentObjectAttribute(core.FloatVectorObjectAttribute,
                            core.DerivedObjectAttribute,
                            ):
    """the extent (T, L, B, R) of a ShapeObject"""
    _ndim = 4
    
    _depends_on = ['position', 'visible']
    
    def _get_data_object(self, obj):
        descendants = [o for o in obj._simple_leaf_descendants if getattr(o, 'visible', True)]
        
        if len(descendants) == 0:
            #this isn't the best, since we probably only got here because of a
            #totally invisible Object. but what else would we do here?
            lt = obj.position
            br = lt
        else:
            p = self._other_attr['position'].get(descendants)
            lt = np.min(p, axis=0)
            br = np.max(p, axis=0)
        
        return np.hstack((lt, br))



class MetaObjectAttribute(core.ObjectObjectAttribute):
    """ Existence of this class allows inclusion of 'meta' attribute """
    pass


class ShapeObjectAttribute( core.ObjectObjectAttribute,
                            core.DerivedObjectAttribute):
    """the shape of a ShapeObject is an N x 2 array of the relative positions
    of its children. setting an Object's shape changes the positions of its
    children accordingly."""
    _depends_on = ['position', 'children']
    
    _read_only = False
    
    _auto_state = False
    
    def _coerce_single_value(self, value, **kwargs):
        """make sure the shape is an N x 2 array of relative positions
        
        Parameters
        ----------
        value : string | array_like
            the shape. string inputs are converted to shapes using the
            library.world._helpers.shape function.
        
        Returns
        -------
        shape : ndarray
            an N x 2 array of relative child positions
        """
        if isinstance(value, basestring):
            value = h.world.shape(value)
        
        if value is not None:
            try:
                if not isinstance(value, np.ndarray):
                    value = np.array(value)
                if value.ndim != 2 or value.shape[1] != 2:
                    value = np.reshape(value, (-1, 2))
            except ValueError:
                return value, ValueError('is not properly formatted')
        
        return value, None
    
    def _get_data_object(self, obj):
        return self._other_attr['position'].get(obj._simple_leaf_descendants) - obj.position
    
    def _set_data_object(self, obj, value):
        if value is not None:
            self._other_attr['position'].set(obj._simple_leaf_descendants, obj.position + value)


class SpriteObjectAttribute(    core.IntegerObjectAttribute,
                                core.LinkObjectAttribute):
    """each SpriteObject stores a list of shapes that can be used to reconfigure
    its children's positions. this ObjectAttribute is linked to the
    SpriteObject's _sprite attribute and is used to get and set the index of the
    active shape.
    """
    _depends_on = ['position', 'color', 'visible', 'children']
    
    _initialize_after = ['position', 'color', 'children', 'visible']
    
    _auto_state = False
    
    def _set_data_object(self, obj, value):
        #sprite must be an index into the sprite shape list
        assert value >= 0 and value < len(obj._sprites), 'invalid sprite'
        
        super(SpriteObjectAttribute, self)._set_data_object(obj, value)
        
        sprite = obj._sprites[value]
        children = obj.children
        
        #set the sprite pixel visibility
        self._other_attr['visible'].set(children, sprite['visible'])
        
        #set the sprite shape
        obj.shape = sprite['shape']
        
        #set the sprite colors
        colors = obj._parse_sprite_colors(sprite['color'])
        self._other_attr['color'].set(children, colors)


class AnimatedObjectAttribute(  core.BooleanObjectAttribute,
                                core.SteppingObjectAttribute):
    """this ObjectAttribute enables animated sprites that cycle through their
    sprite shapes during world.steps"""
    _default_value = False
    
    _depends_on = ['sprite']
    
    #the new sprite shape should take effect before things move
    _step_before = ['velocity', 'pushes']
    
    _auto_state = False
    
    def _step_object(self, obj, t, dt, agent_id, action):
        """step to the next sprite shape in the animation cycle"""
        if self.get(obj._id):  # the Object is animated
            obj.sprite = (obj.sprite + 1) % len(obj._sprites)


class ControllerObjectAttribute(core.IntegerObjectAttribute):
    """Which agent controls the object. A value of -1 means that all agents control
    the object."""
    _null_value = -1


class StrengthObjectAttribute(core.NonNegativeFloatObjectAttribute):
    """Amount of mass the object is capable of pushing. Default is that the object
    can push any amount of mass.
    """
    _default_value = np.inf

    _null_value = np.inf

class PushesObjectAttribute(core.AbilityObjectAttribute):
    """enables pushing behavior, in which an Object moves LEFT, UP, RIGHT, or
    DOWN in response to an action of the same name, and additionally propagates
    that movement (i.e. "pushes") to any Objects that occupy the target
    position(s).

    If the pushing object has a strength attribute, then they can only push as
    much total mass as the strength.
    
    The value of the attribute is boolean, and setting it to false disables
    pushing behavior.
    """
    _default_value = True

    _depends_on = ['visible', 'depth', 'position', 'pushable', 'extent']

    #pushes acts before velocity
    _step_before = ['velocity']
    
    #so position's previous value registers before this changes the position
    _step_after = ['position']
    
    #mapping from action names to push directions
    _push_direction = {
            'LEFT':  np.array((0,-1)),
            'RIGHT': np.array((0,1)),
            'UP':    np.array((-1,0)),
            'DOWN':  np.array((1,0)),
        }
    
    #the actions responded to by the attribute
    _actions = _push_direction.keys()
    
    def _calculate_pushed_coordinate_distance(self, x, x_from, x_dir):
        """calculate the distance to push an Object for a single coordinate. an
        Object should only be pushed at most until it no longer overlaps with
        the pushing Object.
        
        Parameters
        ----------
        x : float
            the pushed Object's current coordinate position
        x_from : float
            the pushing Object's starting coordinate position
        x_dir : Number
            the magnitude of the push along the coordinate's axis
        
        Returns
        -------
        dx : float
            the distance the pushed Object should move along the axis
        """
        if x_dir > 0:  # pushing in the positive direction
            #distance between the two Objects
            x_diff = fix_float_integer(x - x_from)
            
            #make sure the pushing Object is behind the pushed Object
            #TODO do we need these assertions. i'm getting errors when a push
            #occurs after unresolved position conflicts.
            #assert x_diff >= 0, 'cannot push from ahead'
            
            #push magnitude required to resolve the Objects' overlap
            return max(0, x_dir - x_diff)
        elif x_dir < 0:  # pushing in the negative direction
            x_diff = fix_float_integer(x_from - x)
            #TODO see above
            #assert x_diff >= 0, 'cannot push from ahead'
            
            return min(0, x_dir + x_diff)
        else:  # no push in this direction
            return 0
    
    def _calculate_pushed_distance(self, p_obj, p_from, direction):
        """calculate the distance to push an Object that overlaps with the
        pushing Object
        
        Parameters
        ----------
        p_obj : ndarray
            the position of the pushed Object
        p_from : ndarray
            the position of the pushing Object
        direction : ndarray
            the push direction
        
        Returns
        -------
        dp : ndarray
            the change in position that the pushed Object should undergo due to
            the push
        """
        dy = self._calculate_pushed_coordinate_distance(p_obj[0], p_from[0], direction[0])
        dx = self._calculate_pushed_coordinate_distance(p_obj[1], p_from[1], direction[1])
        
        return np.array((dy, dx))
    
    def _find_objects_to_push(self, idx, p_from, direction, strength, initial_pusher, all_push_indices=None, 
                              all_push_to=None):
        """find Objects that should be pushed as a result of a push action,
        either directly due to the push, or indirectly due to changes in pushed
        Objects' positions
        
        Parameters
        ----------
        idx : int
            the index of the Object that is being pushed
        p_from : ndarray
            the position of the pushing Object
        direction : ndarray
            the push direction
        strength : float
            The strength of the pusher.
        initial_pusher : int
            the index of the Object that did the pushing initially
        all_push_indices : ndarray, optional
            an array of indices of all pushed Objects found so far (first call
            should omit this)
        all_push_to : ndarray, optional
            an N x 2 array of the end positions of all pushed Objects found so
            far (first call should omit this)
        
        Returns
        -------
        success : bool
            True if the push is possible, False if something is blocking the
            push from happening
        push_indices : ndarray
            an array of indices of all Objects to push
        push_to : ndarray
            an N x 2 array of the target positions of the pushed Objects
        """
        position = self._other_attr['position']
        
        obj = self._world.objects[idx]
        
        #calculate the push distance
        p_obj = position.get(idx)
        dp = self._calculate_pushed_distance(p_obj, p_from, direction)
        
        #all objects associated with the pushed object
        push_indices = obj._family_ids
        
        #process only the new objects
        if all_push_indices is None:  # initialize the output arrays
            all_push_indices = np.array([], dtype=int)
            all_push_to = np.empty(shape=(0, 2))
        else:
            push_indices = np.setdiff1d(push_indices, all_push_indices, assume_unique=True)
            
            if push_indices.size == 0:  # no new objects
                return True, all_push_indices, all_push_to
            
            #make sure all the new objects are pushable
            if not all(self._other_attr['pushable'].get(push_indices)):
                return False, None, None
        
        #get the target locations
        push_to = fix_float_integer(position.get(push_indices) + dp)
        
        #add them to the list
        all_push_indices = np.append(all_push_indices, push_indices)
        all_push_to = np.vstack((all_push_to, push_to))
        
        #propagate only simple, visible objects
        is_simple = self._world._object_types[push_indices] == obj.SIMPLE
        is_visible = self._other_attr['visible'].get(push_indices)
        to_propagate = np.logical_and(is_simple, is_visible)
        
        #propagate the push to conflicting objects
        depth = self._other_attr['depth']
        obj_depth = depth.get(idx)
        for p in push_to[to_propagate]:
            #find objects that conflict with the new position
            conflict_indices = position.find_conflicts(p)
            
            #process only the new ones
            conflict_indices = np.setdiff1d(conflict_indices, all_push_indices, assume_unique=True)
            
            #process only objects at the same depth as the pusher
            if obj_depth != DepthObjectAttribute._null_value:
                depths = depth.get(conflict_indices)
                is_same_depth = np.logical_or(
                    depths == DepthObjectAttribute._null_value,
                    depths == obj_depth)
                conflict_indices = conflict_indices[is_same_depth]
            
            for conflict_idx in conflict_indices:
                success, all_push_indices, all_push_to = self._find_objects_to_push(
                    conflict_idx, p, direction, strength, initial_pusher, all_push_indices=all_push_indices,
                    all_push_to=all_push_to)
                
                if not success:
                    return False, None, None

        #check whether the pusher is strong enough to push everything
        all_but_pusher = np.array([x for x in all_push_indices.tolist() if x != initial_pusher])
        mass = self._world.object_attributes.get('mass', None)
        if mass is None:
            return True, all_push_indices, all_push_to
        else:
            masses = mass.get(all_but_pusher)
            total_mass = masses.sum()
            if total_mass <= strength:
                return True, all_push_indices, all_push_to
            else:
                return False, None, None
    
    def _push(self, idx, direction):
        """push an Object
        
        Parameters
        ----------
        idx : int
            the index of the Object to push
        direction : ndarray
            the push direction
        """
        position = self._other_attr['position']
        
        # figure out strength of pusher
        strength = self._world.object_attributes.get('strength', None)
        if strength is None:
            strength = np.inf
        else:
            strength = strength.get(idx)
        
        #simulate an object pushing directly from behind
        p_from = position.get(idx)

        #propagate the push until we find all conflicts
        success, push_indices, push_to = self._find_objects_to_push(idx, p_from, direction, strength, idx)

        if success:  # push succeeded
            #print debug info
            if self._world.debug.test(Debug.INFO):
                obj = self._world.objects[idx]
                
                info = OrderedDict()
                for i,p in zip(push_indices, push_to):
                    o = self._world.objects[i]
                    
                    info[o.unique_name] = [tuple(position.get(i)), tuple(p)]
                
                self._world.debug.log('%s push %s propagated to the following objects:' % (obj.unique_name, tuple(direction)), info, level=Debug.INFO)
            
            #defer conflict resolution, even though none should occur here
            resolve_after = position._defer_conflicts()
            
            #set the new Object positions
            position.set(push_indices, push_to, validate=False)
            
            if resolve_after:  # resolve any conflicts that occurred due to the push
                position._resolve_deferred_conflicts()
        elif self._world.debug.test(Debug.INFO):  # something blocked the push
            obj = self._world.objects[idx]
            self._world.debug.log('%s push %s aborted because of blocked path' % (obj.unique_name, direction), level=Debug.INFO)
        
        return success

    def _execute_action(self, obj, t, dt, agent_id, action):
        """push the object along the direction indicated by the action"""
        
        #get the push direction
        direction = self._push_direction.get(action, None)
        
        if direction is not None:
            #in case of fractional steps
            if dt != 1:
                direction = direction.astype(float) * dt
            
            #push the Object
            success = self._push(obj.id, direction)
                    
            #register a push event
            event = events.PushEvent(self._world, idx=obj.id, success=success)


class KilledByObjectAttribute(core.StringObjectAttribute):
    """Cause of death for an object in the world"""
    pass


class AliveObjectAttribute(core.BooleanObjectAttribute):
    """If the object can be killed, give it this attribute. Setting it to False
    makes the object invisible and unanimated, and optionally deletes the
    object if the world attribute killing_deletes is True."""

    _default_value = True

    _depends_on = ['visible', 'killed_by']

    def _set_value(self, idx, value):
        if not is_iterable(idx):
            idx = [idx]
        if not is_iterable(value):
            value = [value] * len(idx)

        to_remove = []
        kill_list = []

        if self.world.killing_deletes:
            for id, val in zip(idx, value):
                if not val:
                    to_remove.append(self.world.objects[id])
                    kill_list.append(id)
        else:
            for id, val in zip(idx, value):
                if not val:
                    self.world.objects[id].visible = False
                    if hasattr(self.world.objects[id], 'animated'):
                        self.world.objects[id].animated = False
                    kill_list.append(id)

        for id in kill_list:
            event = events.KillEvent(self._world, victim=id,
                                     reason=getattr(self.world.objects[id], 'killed_by', 'misadventure'))

        super(AliveObjectAttribute, self)._set_value(idx, value)
        self.world.remove_objects(to_remove)


class BulletShooterObjectAttribute(core.ObjectObjectAttribute):
    """Identifies who shot a bullet. This attribute applies to the bullet.
    """
    pass


class DeadlyObjectAttribute(core.InteractsObjectAttribute):
    """Object attribute that makes things kill objects they intersect with.
    If this attribute is False, the object will not interact with anything and
    will not be deadly.

    Override _should_kill_self, _should_kill_self_reason, _should_kill_other,
    and _should_kill_other_reason in order to define custom behavior in
    subclasses.
    """
    _depends_on = ['position', 'visible']

    _step_after = ['position', 'velocity', 'acceleration']

    def _interact(self, obj, obj2):
        """Handle the interaction between the Deadly object and the other object.

        Parameters
        ----------
        obj : Object
            The first object (the one with the DeadlyObjectAttribute)
        obj2 : Object
            The second object
        """
        posn = obj.position
        obj2 = self.world.objects[obj2._head_of_family._id]

        # call these before we start killing things so that all objects still
        # exist
        should_kill_self = self._should_kill_self(obj, obj2)
        should_kill_other = self._should_kill_other(obj, obj2)

        # if we should kill obj, and obj is alive, kill it by giving obj a
        # killed_by attr and setting obj.alive=False
        # if _should_kill_self returned a string, that means we should kill
        if (should_kill_self or isinstance(should_kill_self, str)) and hasattr(obj, 'alive') and obj.alive:
            obj.add_attribute('killed_by')
            if isinstance(should_kill_self, str):
                obj.killed_by = should_kill_self
            else:
                obj.killed_by = self._should_kill_self_reason(obj, obj2) 
            obj.alive = False

        # if we should kill obj2, and obj2 is alive, kill it by giving obj2 a
        # killed_by attr and setting obj2.alive=False

        # if _should_kill_other returned a string, that means we should kill
        if (should_kill_other or isinstance(should_kill_other, str)) and hasattr(obj2, 'alive') and obj2.alive:
            obj2.add_attribute('killed_by')
            if isinstance(should_kill_other, str):
                obj2.killed_by = should_kill_other
            else:
                obj2.killed_by = self._should_kill_other_reason(obj, obj2) 
            obj2.alive = False

    def _should_kill_self(self, obj, obj2):
        """Should we kill the first object? Override to implement custom behavior in
        subclasses. Default behavior is to never kill the first object.

        Parameters
        ----------
        obj : Object
            The first object (the one with the DeadlyObjectAttribute)
        obj2 : Object
            The second object

        Returns
        -------
        reason : string or bool
            If bool, True means kill and False means don't kill. If string,
            always kill, and use the string as the reason instead of calling
            _should_kill_self_reason.
        """
        return False

    def _should_kill_self_reason(self, obj, obj2):
        """What is the reason for death? Won't get called if _should_kill_self returns
        a string. Override to implement custom behavior in subclasses. Default
        behavior is to to use the second object's name as the reason.
        Parameters
        ----------
        obj : Object
            The first object (the one with the DeadlyObjectAttribute)
        obj2 : Object
            The second object
        Returns
        -------
        reason : string
        """
        return obj2._name

    def _should_kill_other(self, obj, obj2):
        """Should we kill the second object? Override to implement custom behavior in
        subclasses. Default behavior is to always kill the second object.

        Parameters
        ----------
        obj : Object
            The first object (the one with the DeadlyObjectAttribute)
        obj2 : Object
            The second object

        Returns
        -------
        reason : string or bool
            If bool, True means kill and False means don't kill. If string,
            always kill, and use the string as the reason instead of calling
            _should_kill_other_reason.
        """
        return True

    def _should_kill_other_reason(self, obj, obj2):
        """What is the reason for death? Won't get called if _should_kill_other returns
        a string. Override to implement custom behavior in subclasses. Default
        behavior is to use the first object's name as the reason.
        Parameters
        ----------
        obj : Object
            The first object (the one with the DeadlyObjectAttribute)
        obj2 : Object
            The second object
        Returns
        -------
        reason : string
        """
        return obj._name


class BulletDeadlyObjectAttribute(DeadlyObjectAttribute):
    """Object attribute that makes bullets kill objects they intersect with.

    Checks to see if the bullet was shot by the object it's intersecting with,
    so it's impossible to shoot yourself.

    If this attribute is False, the bullet will not interact with anything and
    will not be deadly.
    """
    _depends_on = ['position', 'alive', 'visible', 'bullet_shooter']

    # so you can shoot deadly things and possibly kill them
    _step_before = ['deadly']

    def _should_kill_self(self, obj, obj2):
        """Should we kill the first object? Override to implement custom behavior in
        subclasses. Default behavior is to never kill the first object.

        Parameters
        ----------
        obj : Object
            The first object (the bullet)
        obj2 : Object
            The second object

        Returns
        -------
        reason : string or bool
            If bool, True means kill and False means don't kill. If string,
            always kill, and use the string as the reason instead of calling
            _should_kill_self_reason.
        """
        # bullets cannot kill or be killed by their shooter
        if obj2 is obj.bullet_shooter:
            return False
        # bullets cannot kill or be killed by other bullets
        elif hasattr(obj2, 'bullet_shooter'):
            return False
        else:
            obj.velocity = (0, 0)
            return 'sudden stop'

    def _should_kill_other(self, obj, obj2):
        """Should we kill the second object? Override to implement custom behavior in
        subclasses. Default behavior is to always kill the second object.

        Parameters
        ----------
        obj : Object
            The first object (the bullet)
        obj2 : Object
            The second object

        Returns
        -------
        reason : string or bool
            If bool, True means kill and False means don't kill. If string,
            always kill, and use the string as the reason instead of calling
            _should_kill_other_reason.
        """
        # bullets cannot kill or be killed by their shooter
        if obj2 is obj.bullet_shooter:
            return False
        # bullets cannot kill or be killed by other bullets
        elif hasattr(obj2, 'bullet_shooter'):
            return False
        else:
            return 'bullet'


class OrientationObjectAttribute(core.NonNegativeIntegerObjectAttribute):
    """Attribute for objects that can be facing in one of the four cardinal
    directions. This attribute does nothing to alter the appearance of the
    object.

    The code is 
        left  - 0
        up    - 1
        right - 2
        down  - 3
    """

    def _coerce_value(self, value, num_values=None):
        return value % 4, False


class OrientsObjectAttribute(core.AbilityObjectAttribute):
    """Attribute that causes the object's orientation to be set whenever any of the
    LEFT/UP/RIGHT/DOWN actions are used.
    """
    _depends_on = ['orientation']

    def _execute_action(self, obj, t, dt, agent_id, action):
        if action == 'LEFT':
            obj.orientation = 0
        elif action == 'UP':
            obj.orientation = 1
        elif action == 'RIGHT':
            obj.orientation = 2
        elif action == 'DOWN':
            obj.orientation = 3


class AmmoObjectAttribute(core.StringObjectAttribute):
    """Name of the object class that a shooter should shoot; this attribute applies
    to the shooter. This will be passed to Object.get_instance to create
    bullets or whatever other objects you want to be produced when an object
    with a ShootsObjectAttribute-derived shooting behavior executes its
    shooting action.
    """
    _default_value = 'bullet'

    _null_value = 'bullet'


class ShootsObjectAttribute(core.AbilityObjectAttribute):
    """Attribute that makes something able to shoot. This is an abstract class; you
    should use one of the classes that decides _when_ to shoot, like
    SelfShootsObjectAttribute or EnemyShootsObjectAttribute.

    Subclasses should implement _should_shoot() with a function that decides
    when to shoot.
    """

    #lookup table for what vector each direction represents
    _shoot_direction = [
        np.array((0,-1)), # left
        np.array((-1,0)), # up
        np.array((0,1)),  # right
        np.array((1,0)),  # down
        ]

    # speed of the bullets
    _speed = 1

    _depends_on = ['orientation']
    _steps_before = ['position']

    def _shoot(self, obj):
        """Emit a bullet (or whatever type of object is specified by obj.ammo, if obj
        has the attribute ammo).

        Parameters
        ----------
        obj : Object
            The object doing the shooting.
        """
        dir = self._other_attr['orientation'].get(obj)
        velocity = self._shoot_direction[dir] * self._speed
        posn = obj.position

        #get all SIMPLE leaf descendants
        objs = obj._simple_leaf_descendants
            
        #advance the bullet until it does not intersect with the shooter
        while any(np.linalg.norm(x.position - posn) < 1.0 for x in objs):
            posn += self._shoot_direction[dir]
        posn -= velocity

        bullet = self._world.create_object([getattr(obj, 'ammo', AmmoObjectAttribute._null_value), 
                                            dict(position=posn,
                                                 velocity=velocity,
                                                 bullet_shooter=obj,
                                                 )])
        event = events.BulletFireEvent(self._world, position=obj.position, dir=dir, bullet_id=bullet.id)

    def _execute_action(self, obj, t, dt, agent_id, action):
        """Execute the shooting action by figuring out whether we should shoot and then
        either doing it or not doing it.

        Parameters
        ----------
        obj : Object
            The object doing the shooting.
        t : number
            The simulation time.
        dt : number
            Time since last step.
        agent_id : number
            Id of currently stepping agent.
        action : string
            Last selected action
        """
        if self.should_shoot(obj, t, dt, agent_id, action):
            self._shoot(obj)

    def should_shoot(self, obj, t, dt, agent_id, action):
        """Public version of should_shoot, only allows the object to maybe shoot if the
        object is visible. Calls the private version, which should be
        implemented by subclasses.

        Parameters
        ----------
        obj : Object
            The object doing the shooting.
        t : number
            The simulation time.
        dt : number
            Time since last step.
        agent_id : number
            Id of currently stepping agent.
        action : string
            Last selected action
        """
        if not obj.visible:
            return False
        return self._should_shoot(obj, t, dt, agent_id, action)

    def _should_shoot(self, obj, t, dt, agent_id, action):
        """Subclasses should override this function.

        Parameters
        ----------
        obj : Object
            The object doing the shooting.
        t : number
            The simulation time.
        dt : number
            Time since last step.
        agent_id : number
            Id of currently stepping agent.
        action : string
            Last selected action
        """
        raise NotImplementedError
        

class SelfShootsObjectAttribute(ShootsObjectAttribute):
    """Attribute that causes object to shoot whenever the SHOOT action is
    performed.
    """

    _actions = ['SHOOT']

    def _should_shoot(self, obj, t, dt, agent_id, action):
        """Decide whether to shoot. Currently shoots whenever the SHOOT action is
        taken.
        
        Parameters
        ----------
        obj : Object
            The object doing the shooting
        t : number
            The simulation time
        dt : number
            Time since last step
        agent_id : number
            Id of currently stepping agent.
        action : string
            Last executed action
        """
        return (action == 'SHOOT')


class RandomlyShootsObjectAttribute(core.FractionalFloatObjectAttribute,
                                    ShootsObjectAttribute,
                                    ):
    """Attribute that causes object to shoot with some probability at each time
    step. The value of the attribute is the probability; set to zero to disable
    shooting.
    """

    def _should_shoot(self, obj, t, dt, agent_id, action):
        """Decide whether to shoot based on our probability.
        
        Parameters
        ----------
        obj : Object
            The object doing the shooting
        t : number
            The simulation time
        dt : number
            Time since last step
        agent_id : number
            Id of currently stepping agent.
        action : string
            Last executed action
        """
        return (self.rng.rand() < self.get(obj))


class InitialPositionObjectAttribute(core.PointObjectAttribute):
    """An attribute that remembers the initial position of the object."""

    _initialize_after = ['position']
    _depends_on = ['position']

    def prepare(self):
        self.set(None, self._other_attr['position'].get(self.objects))


class GripObjectAttribute(core.ObjectObjectAttribute):
    """See GripsObjectAttribute for more context.

    Attribute that marks an object as a grip object that exists to be the parent
    of both the gripper and the grippee, so that they move in perfect unison.

    The value of this object is the gripping object. None indicates that no
    gripping object exists, which should never happen when something has this
    attribute.
    """
    _default_value = None
    _null_value = None


class GrippedObjectAttribute(core.ObjectObjectAttribute):
    """Attribute to specify that an object is gripped and it should respond to
    pushes. While the object is gripped, it and the gripper are both the
    children of a grip object.

    The value of this object is the gripping object. None indicates that no one
    is gripping the object.
    """
    _default_value = None
    _null_value = None


class GripsObjectAttribute(core.AbilityObjectAttribute):
    """Attribute to specify that the object can grip things using GRIP/UNGRIP
    commands. When this happens, we assemble a grip object which has the
    gripper and grippee as children. This attribute listens for GRIP commands
    and assembles the grip object when appropriate. This attribute also listens
    for the UNGRIP command, upon which it disassembles the grip object.
    """
    _actions = ['GRIP', 'UNGRIP']
    _depends_on = ['position', 'orientation']
    _step_before = ['pushes', 'position']

    #lookup table for what vector each direction represents
    _grip_direction = [
        np.array((0,-1)), # left
        np.array((-1,0)), # up
        np.array((0,1)),  # right
        np.array((1,0)),  # down
        ]

    def _execute_action(self, obj, t, dt, agent_id, action):
        """Listen for GRIP/UNGRIP commands and assemble/disassemble grip objects as
        appropriate.

        Parameters
        ----------
        obj : Object
            The object that has the 'grips' attribute.
        t : number
            The simulation time.
        dt : number
            The time since the last step.
        agent_id : int
            The id of the agent that is currently stepping.
        action : string
            The most recent action executed.
        """
        assert hasattr(obj, 'orientation'), 'Only objects with orientation can have grips=True'

        # check that obj is either head of family or underneath a grip object
        assert obj._head_of_family._id == obj.id or hasattr(obj._parent, 'grip'), \
            'Only head-of-family objects can have attribute grips'

        if action == 'GRIP':
            # if any simple leaf descendant of obj is distance 1 away in the
            # direction of the gripping object's orientation, and the object
            # passes our later checks, we should grip the object (unless it's
            # part of the gripper, which we check for later)
            selector = np.zeros(len(self.world.objects), dtype=bool)
            state_index_values = self.world.object_attributes['state_index'].get(np.arange(len(self.world.objects)))
            for descendant in obj._simple_leaf_descendants:
                selector = np.logical_or(
                    selector,
                    np.all(np.abs(state_index_values -
                                  (descendant.state_index + self._grip_direction[obj.orientation])) < 1, axis=1))

            # can only pick up SIMPLE things (pick up compound objects by their children)
            close_simple_objects = np.array([x for x in self.world.objects[selector] if x._class_type == obj.SIMPLE])
                
            # can only pick up things at same depth as you or things with null depth
            depth_values = self.world.object_attributes['depth'].get(close_simple_objects)
            selector = np.logical_or(
                depth_values == obj.depth,
                depth_values == DepthObjectAttribute._null_value)
            close_simple_objects = close_simple_objects[selector]

            # can only pick up visible things
            selector = self.world.object_attributes['visible'].get(close_simple_objects)
            close_simple_objects = close_simple_objects[selector]

            # can only pick up things with a state_index
            close_simple_objects = [x for x in close_simple_objects if hasattr(x, 'state_index')]

            for obj2 in close_simple_objects:
                # can't grip self or already gripped objects
                if obj2._head_of_family._id == obj._head_of_family._id:
                    continue

                # ascend the hierarchy to get to the top object that isn't
                # a grip object
                while hasattr(obj2, 'parent') and not hasattr(obj2._parent, 'grip'):
                    obj2 = self.world.objects[obj2.parent]

                if hasattr(obj2, 'gripped'):
                    pass # we're already gripping it

                elif not obj2.pushable or not obj2.moveable or hasattr(obj2, 'pushes'):
                    #grip failed, you cannot grip an unpushable object, an immoveable object, or a pusher
                    event = events.GripEvent(self._world, gripper=obj.id, grippee=obj2.id, success=False)

                else:
                    # if our gripper has a parent, it's a grip object and we
                    # should use it for the new grippee also
                    if hasattr(obj, 'parent') and obj._parent is not None:
                        assert hasattr(obj._parent, 'grip'), \
                            'Only top-level objects should have the "grips" attribute'
                        grip = obj._parent
                    
                    # otherwise, create a new grip object and make the gripper a child
                    else:
                        grip = self.world.create_object('grip')
                        grip.grip = obj
                        grip.add_child(obj)

                    # make the grippee a child
                    grip.add_child(obj2)

                    # mark obj2 as gripped by obj
                    obj2.add_attribute('gripped', obj)

                    # add an event
                    event = events.GripEvent(self._world, gripper=obj.id, grippee=obj2.id, success=True)

        elif action == 'UNGRIP':
            # if gripped attribute doesn't even exist, create a failed
            # UngripEvent and do nothing
            if 'gripped' not in self.world.object_attributes:
                event = events.UngripEvent(self._world, gripper=obj.id, grippee=None, success=False)
                return

            # get the gripped objects
            gripped_objects = self.world.objects[self.world.object_attributes['gripped'].get(
                    np.arange(len(self.world.objects))) == obj]

            for obj2 in gripped_objects:
                # delete grip object if not already deleted. (prob because we
                # were gripping two objects)
                if obj2._parent is not None:
                    assert hasattr(obj2._parent, 'grip')
                    self.world.remove_objects(obj2._parent)

                # remove gripped from obj2
                obj2.remove_attribute('gripped')

                event = events.UngripEvent(self._world, gripper=obj.id, grippee=obj2.id, success=True)

            if len(gripped_objects) == 0:
                event = events.UngripEvent(self._world, gripper=obj.id, grippee=None, success=False)


