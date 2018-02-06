'''
    PixelWorld, a grid-based OpenAI gym environment that blends physics
    simulation and arcade environment with flexible object and attribute
    behaviors.
    
    This file defines the core class types that are used in a PixelWorld:
        Entity:
            the base class for all entities (see below) that live in a
            PixelWorld
        ObjectAttribute:
            defines an attribute possessed by an object. ObjectAttributes are
            where most behaviors should be implemented. they not only have
            values, but can also exhibit behaviors such as changing their or
            other ObjectAttribute's values when the world steps. some basic
            ObjectAttributes are defined here, with others defined in
            object_attributes.py.
        Object:
            an Object should usually do little more than define the object
            attributes it possesses. CompoundObjects can bind together simple
            Objects in order to couple their attributes' behaviors, e.g. so
            their positions become linked or their masses become pooled. only
            Object and CompoundObject are defined here, with some more defined
            in objects.py.
        Event:
            a flexible class for recording the occurrence of events. Judges can
            determine reward and episode termination conditions based on the
            occurrence of Events. only the base Event class is defined here,
            with some other Events used by some basic ObjectAttributes defined
            in events.py. Events should probably only be defined for events
            whose occurrence cannot easily be identified by inspecting the
            PixelWorld and its Entities.
        Goal:
            a class for defining episode goals. a Goal determines whether or not
            it has been achieved in the current world state and optionally
            implements a method for achieving the goal from any world state.
            Goals can also be active or not, indicate whether or not they should
            contribute to terminating the episode, and/or supply an intrinsic
            reward for achieving them.
        Judge:
            Judges evaluate the world based on Events, Goals, world timeouts,
            and potentially anything else. when the world steps, it passes the
            Events that occurred during the step to a Judge in order to
            determine the reward to give the agents and whether the episode
            should terminate. the base Judge defined here always returns -1
            reward per step and does not end the episode.
        Variant:
            a Variant defines and brings about a set of variants of a
            PixelWorld. this is a very general class that can be used to e.g.
            choose between variations of an Object, turn colors on or off, etc.
        Randomizer:
            when the world is reset, the randomizer takes care of randomizing the 
            starting conditions before the start of the next episode. The default 
            randomizer does nothing. some basic randomizers are defined in
            randomizers.py.
        Agent:
            although a PixelWorld can be used like any OpenAI gym environment,
            in which the agent exists independently and passes an action to the
            environment's step() function, you can also set up a PixelWorld such
            that the agent is embedded within the environment by creating a
            subclass of the base Agent class. in this case, the world step()
            method should be called with no arguments, and the world will then
            query the agent for an action, execute the step, and tell the agent
            what reward resulted from the action and whether the episode has
            terminated. if no agent is defined at world creation time, then the
            base Agent will still exist in the world as a placeholder. the base
            Agent always chooses the 'NOOP' action, but will never be queried if
            an action is always passed to the world's step() function. some
            basic agents are defined in agents.py.

            PixelWorld supports multiple agents, in which case you should pass
            in an iterable of actions of the correct length when you call
            step() with actions.
        WorldAttribute:
            a WorldAttribute stores and manipulates data associated with a
            PixelWorld (e.g. its height and width, current time, etc.).
        PixelWorld:
            a two-dimensional grid world in which a judge, an agent or agents,
            and a set of objects with object attributes live, and in which
            events occur.  each time the world advances by one step(), it
            optionally queries its agent(s) for an action, calls the step()
            method of each of its SteppingObjectAttributes (which in turn
            perform whatever behaviors they have defined), records events as
            they occur, queries its judge for the reward and episode
            termination based on the events that occurred during the step, and
            informs the agent(s) of the reward it received and whether the episode
            has terminated.
            
            the state of the world is a 2D integer numpy array. its value is
            determined by the position and color of the world's Objects. Object
            positions are continuous, but are rounded to the nearest integer
            when determining the state element they affect. note that Objects
            can have positions that lie outside the bounds of the state
            (i.e. < -0.5 or > world.<height|width> - 0.5), in which case they
            will no longer determine the state. the world's state can therefore
            be thought of as a window on an infinite plane. objects can be
            confined to the visible region of the world by including a
            FrameObject (defined in objects.py).
            
            an agent's observations of the world can be its state (the default)
            its object_state, which is a list of dicts of attribute values of
            all of the world's objects, or array/vector versions of its
            object_state. object_state-based observations can be filtered for
            particular objects or attributes.
            
            the world includes a random number generator (world.rng), that all
            pseudorandom behaviors should use, so that states and episodes are
            reproducible. each Entity has access to the world's rng via its own
            rng attribute (i.e. obj.rng is world.rng).
    
    see the *.py files in pixelworld/library/world/ for example environments.
    these files define the object attributes, objects, judge, and agent(s) for a
    PixelWorld simulation, and the Demo class in demo.py assembles them into a
    PixelWorld and runs the simulation. e.g.:
        from pixelworld.envs.pixelworld.demo import Demo
        d = Demo('blockpush')
    environments can also easily be created from these library worlds:
        from pixelworld.envs.pixelworld.universe import create_world
        env = create_world('blockpush')
'''
import os, sys
import inspect
import warnings
import tempfile, shutil
from datetime import datetime
from copy import copy, deepcopy
from numbers import Number
from collections import OrderedDict, defaultdict
from PIL import Image
import logging

import numpy as np
from numpy import ma
from scipy.misc import imresize, imsave
from scipy.misc.pilutil import toimage

import gym
from gym import error, spaces
from gym.utils import seeding

# put these imports inside try-catch blocks so that we can run on systems with
# no GUI
gui_exists = True
gui_exceptions = []
try:
    from gym.envs.classic_control import rendering
except Exception as e:
    gui_exists = False
    gui_exceptions.append(e)
try:
    import pyglet
except Exception as e:
    gui_exists = False
    gui_exceptions.append(e)
if not gui_exists:
    logging.warn('Warning: proceeding in non-GUI mode: ' + '; '.join(str(e) for e in gui_exceptions))


from utils import fix_float_integer, is_iterable, to_iterable, switch, mode, \
    plural, generate_color_palette, ind_to_rgb, imresize, to_vector, \
    cartesian_product, topological_sort, setdiff2d, get_centered_subarray, \
    append_items, Debug, CustomOrderedDict
from datatypes import Data, TypeFamily, StateObject, State


module_path = os.path.dirname(__file__)


def check_gui_exists():
    """Check that we successfully imported pyglet and rendering, and fail if we
    have not."""
    if not gui_exists:
        raise ImportError('a GUI function was called, but GUI-related imports failed: %s ' % 
                          ('; '.join(str(e) for e in gui_exceptions),))


class ObjectStateSpace(gym.Space):
    """gym class that defines the space for object-based observations"""
    
    def sample(self, seed=0):
        """hopefully we don't need this since this space will be used for
        observations rather than actions"""
        raise NotImplementedError

    def contains(self, x):
        """just test whether x satisfies the constraint of being a list of
        dicts."""
        return isinstance(x, list) and all([isinstance(e, dict) for e in x])


class ListSpace(gym.Space):
    """space defined by a list of things (e.g. action names). PixelWorld's
    action space is a list of action names, rather than a set of integers used
    in other environments.
    """
    _world = None
    _list = None
    
    def __init__(self, world, lst):
        """
        Parameters
        ----------
        world : PixelWorld
            the PixelWorld being simulated
        lst : list
            the list of items that comprise the space
        """
        assert isinstance(world, PixelWorld), 'world must be a PixelWorld'
        assert isinstance(lst, list), 'lst must be a list'
        
        self._world = world
        self._list = lst
    
    def sample(self):
        """sample an element of the space's list.
        
        Returns
        -------
        x : any
            a random element from the list
        """
        return self._world.rng.choice(self._list)
    
    def contains(self, x):
        """is x in the list space?"""
        return x in self._list
    
    def __repr__(self):
        return 'List(%s)' % (self._list)
    
    def __eq__(self, other):
        return self._list == other._list

    def __ne__(self, other):
        """Implements a != b.

        Parameters
        ----------
        self : ListSpace
        other : ListSpace

        Returns
        -------
        rv : bool
           False if the two spaces are equal, True otherwise.
        """
        return self._list != other._list

class MultiListSpace(ListSpace):
    """space defined by a list of things (e.g. action names). PixelWorld's
    action space is a list of action names, rather than a set of integers used
    in other environments.

    This differs from ListSpace in that elements of the space are tuples of
    actions, one for each agent. The actions all come from a single list of
    action names.
    """

    _num = 1

    def __init__(self, world, lst, num):
        """
        Parameters
        ----------
        world : PixelWorld
            the PixelWorld being simulated
        lst : list
            the list of items that comprise the space
        num : int
            the number of agents
        """
        assert isinstance(world, PixelWorld), 'world must be a PixelWorld'
        assert isinstance(lst, list), 'lst must be a list'
        
        self._world = world
        self._list = lst
        self._num = num

    def sample(self):
        """sample num elements of the space's list to get a tuple suitable for feeding
        into step()
        
        Returns
        -------
        x : tuple(any)
            num random element from the list
        """
        return tuple(self._world.rng.choice(self._list) for _ in xrange(self._num))
    
    def __repr__(self):
        return 'MultiList(%s, %d)' % (self._list, self._num)
    
    def __eq__(self, other):
        return self._list == other._list and self._num == other._num

    def __ne__(self, other):
        """Implements a != b.

        Parameters
        ----------
        self : MultiListSpace
        other : MultiListSpace

        Returns
        -------
        rv : bool
           False if the two spaces are equal, True otherwise.
        """
        return self._list != other._list or self._num != other._num

class Entity(StateObject):
    """all entities in a PixelWorld (e.g. ObjectAttribute, Object, Judge,
    Agent) subclass off of this class
    """
    _class_family = 'entity'
    
    #the host PixelWorld
    _world = None
    
    #new attributes to include in States
    _state_attributes = ['_name', '_world']
    
    @classmethod
    def get_class(cls, c):
        """return a class (in the class family), given something that references
        the class
        
        Parameters
        ----------
        c : string | TypeFamily | Entity | list[string | type, dict]
            one of the following:
                -   the name of a class in the same family as cls (e.g. 'self')
                -   a class in the family (e.g. SelfObject)
                -   an instance of the class, in which case the instance is
                    returned
                -   a one or two element list consisting of the class/name and
                    optionally a dict of keyword arguments to pass to the
                    constructor (e.g. ['self', {'color': 2}])
        
        Return
        ------
        c : TypeFamily
            the matching class
        """
        if isinstance(c, basestring):  # class name
            try:
                return cls._class_family[c]
            except KeyError:
                raise ValueError('no %s with name "%s" exists' % (cls._class_family_name, c))
        elif inspect.isclass(c) and c._class_family is cls._class_family:  # class
            return c
        elif isinstance(c, Entity):  # class instance
            return c.__class__
        elif isinstance(c, list) and len(c) > 0:  # list specification
            return cls.get_class(c[0])
        else:  # nope
            raise TypeError('%s is not a valid %s' % (c, cls._class_family_name))
    
    @classmethod
    def get_instance(cls, world, spec):
        """construct an instance of the class or another class in the same
        family
        
        Parameters
        ----------
        world : PixelWorld
            the PixelWorld to add the entity to
        spec : string | TypeFamily | Entity | list[string | type, dict]
            one of the following:
                -   the name of a class in the same family as cls (e.g. 'self')
                -   a class in the family (e.g. SelfObject)
                -   an instance of the class, in which case the instance is
                    returned
                -   a one or two element list consisting of the class/name and
                    optionally a dict of keyword arguments to pass to the
                    constructor (e.g. ['self', {'color': 2}])
        
        Returns
        -------
        inst : Entity
            an instance of the class specified by x
        """
        if isinstance(spec, list):
            if len(spec) == 2:
                c = spec[0]
                d = spec[1]
            elif len(spec) == 1:
                c = spec[0]
                d = {}
            else:
                raise TypeError('invalid instance parameters')
        else:
            c = spec
            d = {}
        
        if isinstance(c, cls):
            assert c.world is world, 'entities cannot move between worlds'
            
            #update the object's attributes
            for key,value in d.iteritems():
                setattr(c, key, value)
            
            return c
        else:
            c = cls.get_class(c)
            return c(world, **d)
    
    @classmethod
    def get_init_args(cls):
        """construct a aggregate list of the arguments that the class and its
        superclasses accept in the __init__() methods"""
        args = []
        
        #compile the super class arguments
        for base_cls in cls.__bases__:
            if hasattr(base_cls, 'get_init_args'):
                args.extend(base_cls.get_init_args())
        
        #add our own
        args.extend(inspect.getargspec(cls.__init__).args)
        
        #return the unique items
        return list(set(args))
    
    def __new__(cls, *args, **kwargs):
        """this is just here to catch arguments from subclasses"""
        return super(Entity, cls).__new__(cls)
    
    def __init__(self, world, name=None, prepare=True):
        """
        Parameters
        ----------
        world : PixelWorld
            the PixelWorld in which the Entity lives
        name : string
            optionally rename the entity at creation time
        prepare : bool
            True to call the object's prepare() method, but only if the world
            is populated.
        """
        #optional custom name
        if name is not None:
            self._name = name
        
        #add the entity to the world
        self._add_world(world)
        
        if prepare and self.world.populated:
            self.prepare()
    
    @property
    def specifications(self):
        """construct a specification list that can be passed as the spec
        argument to Entity.get_instance()"""
        spec_params = {}
        
        #get the __init__ parameters and their current values
        for arg in self.get_init_args():
            if arg not in ['self', 'world', 'prepare']:
                if hasattr(self, '_' + arg):
                    spec_params[arg] = getattr(self, '_' + arg)
                elif hasattr(self, arg):
                    spec_params[arg] = getattr(self, arg)
        
        return [self.__class__._name, spec_params]
    
    @property
    def name(self):
        """the Entity's name. an Entity's name can be used when creating new
        Entity instances (see Entity.get_instance)."""
        return self._name
    
    @property
    def world(self):
        """the entity's host PixelWorld"""
        return self._world
    
    @property
    def rng(self):
        """provides convenient access to the world's random number generator"""
        return self._world._rng
    
    @property
    def exists(self):
        """True if the Entity is associated with a PixelWorld"""
        return self._world is not None
    
    def prepare(self):
        """this method is called for each entity in the world, immediately after
        the world is fully populated. subclasses can override this if they need
        to do something at the beginning of the world, but after all entities
        exist. for entities that are created at a later time, this method is
        called during object initialization by default.
        """
        pass
    
    def remove(self):
        """remove the entity from the world and all other linked Entities.
        Entity subclasses will add their own custom removal processes here.
        """
        self._remove_world()
    
    def __dir__(self):
        """basically reimplement dir() so subclasses can override it"""
        #get class attributes
        d = set(dir(self.__class__))
        
        #add instance dict attributes
        d.update(self.__dict__.keys())
        
        return sorted(d)
        
    def _add_world(self, world):
        """integrate the entity into the world. subclasses should override this
        (remember to call super at the start of the method) to perform any
        class-specific integration steps.
        
        Parameters
        ----------
        world : PixelWorld
            the world to add the Entity to
        """
        self._world = world
    
    def _remove_world(self):
        """remove references to the Entity from the world. whatever happens here
        in subclasses should not assume that the references exist (i.e. the
        reference removal may have already happened somewhere else).
        """
        self._world = None
    
    def _encode_state(self, state):
        """use to perform custom state encoding, beyond that defined via the
        _state_attributes list above (see StateObject)
        
        Parameters
        ----------
        state : datatypes.State
            the encoded Entity state
        """
        pass
    
    def _decode_state(self, state):
        """use to perform custom state decoding, beyond that defined via that
        _state_attributes list above (see StateObject)
        
        Parameters
        ----------
        state : datatypes.State
            the encoded Entity state, with all values decoded
        """
        pass


#---ObjectAttribute------------------------------------------------------------#
class ObjectAttribute(Entity):
    """an ObjectAttribute stores and manipulates data associated with Objects.
    this class defines some core ObjectAttribute behaviors. below it, some
    generic subclasses define additional behaviors that some ObjectAttributes
    might want. below that, some core ObjectAttributes that all Objects should
    possess are defined. additional ObjectAttributes are defined in
    object_attributes.py.
    
    many ObjectAttribute methods have a corresponding _method. subclasses should
    probably override _method rather than method, since many generic
    ObjectAttribute classes add functionality to method, but leave _method
    alone. for this base class, there are the following method pairs:
        set -> _set_data or _set_value
        get -> _get_data or _get_value
        compare -> _compare
    the generic subclasses below adopt the same convention of public/private
    method pairs (this is inspired by the OpenAI gym Environment class)
    """
    class __metaclass__(type(Entity)):
        #this defines the partial orderings that will be collected by the metaclass
        #above. subclasses can add their own order types by defining _order_types (see the class definition below) as
        #a list of types to add. the class and its subclasses can then define
        #_<type>_before and _<type>_after attributes as lists of names of other
        #ObjectAttributes that should appear either before or after that
        #ObjectAttribute in the ordering. the _<type>_before and _<type>_after
        #attributes should also probably be defined as _class_tree_attributes so
        #that subclasses inherit the pairwise orders defined by their superclasses.
        _attribute_order = {}
        
        """this metaclass collects specifications for partial ordering defined
        collectively by the family of ObjectAttributes"""
        def __new__(meta, name, bases, dct):
            cls = type(Entity).__new__(meta, name, bases, dct)
            
            #construct partial sorting specifications for each order type
            for order_type in cls._order_types:
                if not order_type in cls._attribute_order:
                    order = []
                    cls._attribute_order[order_type] = []
                else:
                    order = cls._attribute_order[order_type]
                
                #add each defined order pair
                for name in getattr(cls, '_%s_before' % (order_type), []):
                    order.append([cls._name, name])
                for name in getattr(cls, '_%s_after' % (order_type), []):
                    order.append([name, cls._name])
            
            return cls
    
    #see note about class hierarchy snapshots in TypeFamily
    TypeFamily._class_hierarchy_snapshot_attributes.append([__metaclass__, '_attribute_order'])
    
    #define a new class family (see TypeFamily)
    _class_family = 'object_attribute'
    
    #the attribute's name. if subclasses do not assign this explicitly, the name
    #is derived sensibly from the class name, e.g. PositionObjectAttribute
    #becomes 'position' and KineticEnergyObjectAttribute becomes
    #'kinetic_energy'.
    _name = 'attribute'
    
    #this causes each ObjectAttribute to have the specified attributes, which
    #are lists that are assembled based on the ObjectAttribute's superclasses
    #(see TypeFamily and the attributes below for details)
    _class_tree_attributes = ['_order_types', '_data_fields', '_depends_on',
        '_initialize_before', '_initialize_after', '_coerce_value']
    
    #see the note about _attribute_order in the metaclass above
    _order_types = ['initialize']
    
    #if Objects should initialize the ObjectAttribute in a particular order
    #relative to other ObjectAttributes, define so here
    _initialize_before = []
    _initialize_after = []
    
    #the numpy dtype of the attribute's data array
    _dtype = np.dtype(int)
    
    #the dimensionality of each attribute value
    _ndim = 1
    
    #the null value of the attribute. this value is used when an attribute value
    #is needed but one doesn't exist, such as when the value of the attribute is
    #needed for an object that doesn't have that attribute, or when taking the
    #sum or mean of an empty set, or when a parent object derives its value from
    #its children, but has no children. NOTE that because of constraints on
    #MaskedArray fill values, this needs to be a scalar regardless of the
    #dimensionality of the attribute.
    _null_value = 0
    
    #the default value of an attribute, used when an Object has the attribute
    #but doesn't explicitly define what the value of the attribute should be.
    #this can also be a function that takes the Object as input and returns a
    #value.
    _default_value = 0
    
    #if an ObjectAttribute requires Objects with that attribute to also have
    #other attributes (e.g. momentum depends on mass and velocity), it should
    #define a list of the names of those ObjectAttributes here. note that at
    #class construction, TypeFamily replaces this with an aggregated list of all
    #attributes depended on by the ObjectAttribute and its superclasses.
    #ObjectAttributes should make sure that their dependencies are defined first
    #(i.e. appear first in an Object's list of attributes)
    _depends_on = []
    
    #if an ObjectAttribute defines attribute dependencies above, those
    #attributes will be available in _other_attr, which will be a dict organized
    #by attribute name (e.g. MassObjectAttribute is available to
    #MomentumObjectAttribute in _other_attr['mass'])
    _other_attr = None
    
    #if the ObjectAttribute supplies its own data in the world's master data
    #array, a view of the data will be available here
    _data = None
    
    #during initialization, this will be set to True if the ObjectAttribute has
    #a field of its own name
    _has_own_data = False
    
    #set this to True if the ObjectAttribute's values should not be set. NOTE
    #that read-only ObjectAttributes should also override set() to raise an
    #error. this is only here to provide a record that the ObjectAttribute
    #intends to be read-only.
    _read_only = False
    
    #this dict specifies fields that the ObjectAttribute should add to the
    #master data array, and the corresponding attributes that should become
    #views to those fields. dict keys are attribute names, and dict values
    #are themselves dicts with the following elements that specify information
    #about the field to add:
    #   'field':        the field name, or a function that takes self as an
    #                   input argument and returns the field name. if this is a
    #                   function, it is resolved to its computed value during
    #                   initialization.
    #   'dtype':        the field's numpy dtype. if omitted, uses self._dtype.
    #   'null_value':   the field's null value. if omitted, uses
    #                   self._null_value.
    #   'ndim':         the dimensionality of each value. if omitted, uses
    #                   self._ndim.
    #these fields are added/remove in _add_data and _remove_data, respectively,
    #and views to the fields are set in _set_view, which in turn is called by
    #PixelWorld._set_views, which in turn is registered as a callback with
    #PixelWorld._data and called whenever the data array changes.
    _data_fields = {'_data': {'field': lambda (self): self._name}}
    
    _auto_state = False
    
    def __new__(cls, *args, **kwargs):
        attr = super(ObjectAttribute, cls).__new__(cls, *args, **kwargs)
        
        #make bound methods out of the _coerce_value functions (see
        #_coerce_value for an explanation)
        attr._coerce_value = [f.__get__(attr, cls) for f in attr._coerce_value]
        
        return attr
    
    def __init__(self, world, prepare=True, **kwargs):
        #initialize some objects
        self._other_attr = {}
        
        super(ObjectAttribute, self).__init__(world, prepare=False, **kwargs)
        
        #resolve data fields
        self._data_fields = deepcopy(self._data_fields)
        for attr,info in self._data_fields.iteritems():
            #resolve lazy field names
            if callable(info['field']):
                info['field'] = info['field'](self)
            
            #test for a self data field
            if info['field'] == self._name:
                self._has_own_data = True
        
        #add the data
        self._add_data()
        
        if prepare and self.world.populated:
            self.prepare()
    
    @property
    def dtype(self):
        """the ObjectAttribute's data array dtype"""
        return self._dtype
    
    @property
    def ndim(self):
        """the dimensionality of each attribute value"""
        return self._ndim
    
    @property
    def null_value(self):
        """the attribute's null value"""
        return self.coerce_value(copy(self._null_value))
    
    @property
    def data(self):
        """a view of the ObjectAttribute's field in the master data array"""
        return self._data
    
    @property
    def objects(self):
        """an array of all Objects that have this ObjectAttribute"""
        return copy(self._get_objects())
    
    @property
    def object_indices(self):
        """an array of the indices of all Objects that have this ObjectAttribute
        """
        return self._get_object_indices()
    
    def coerce_value(self, value, error=True, **kwargs):
        """coerce values to comply with the requirements of the ObjectAttribute,
        or raise an error if it can't comply. this method passes values
        sequentially through all _coerce_value methods defined by the class and
        its superclasses, starting with the root ObjectAttribute class.
        
        Parameters
        ----------
        value : any
            the value (or iterable of values) to coerce
        error : bool
            True if an exception should be raised if the value does not comply
            with the ObjectAttribute's requirements. if this is False and the
            input does not comply, None will be returned.
        **kwargs
            extra keyword arguments defined by particular _coerce_value methods
        
        Returns
        -------
        value : self.dtype
            the coerced value(s)
        """
        for f in self._coerce_value:
            value, err = f(value, **kwargs)
            
            if err:
                if error:
                    raise type(err)('%s %s' % (self._name, err))
                else:
                    return None
        
        return value
    
    def remove(self):
        """remove the ObjectAttribute from the world. this destroys all record
        of the ObjectAttribute. it could also potentially change the structure
        of attribute-based observations, so beware."""
        #remove from all Objects
        for obj in self.objects:
            if obj.exists:  # make sure the Object still exists
                obj.remove_attribute(self._name)
        
        #remove our data
        self._remove_data()
        
        super(ObjectAttribute, self).remove()
    
    def get(self, x=None):
        """return the current attribute value(s) for a set of Objects
        
        Parameters
        ----------
        x : Object | int | iterable, optional
            an Object, object index, or array of object indices. if unspecified,
            returns all values
        
        Returns
        -------
        value : self.dtype | ndarray
            the Objects' attribute value(s)
        """
        idx = self._get_index(x)
        return self._get_value(idx)
    
    def set(self, x, value, validate=True):
        """set the attribute value(s) for a set of Objects
        
        Parameters
        ----------
        x : Object | int | iterable | None
            an Object, object index, or array of object indices, or None. sets
            all Object attributes if x is None.
        value : self.dtype
            the new value (or array of values) of the attribute for the
            specified Object(s)
        validate : bool, optional
            True to pass the value(s) through coerce_value(). if this is False,
            the method assumes value is valid for the ObjectAttribute's dtype.
        """
        idx = self._get_index(x)
        
        if validate:
            num_values = len(idx) if is_iterable(idx) else None
            value = self.coerce_value(value, num_values=num_values)
        
        self._set_value(idx, value)
    
    def compare(self, x, y, validate=True):
        """compare two sets of values
        
        Parameters
        ----------
        x : Object | self.dtype | ndarray
            an Object, value, or array of values
        y : Object | self.dtype | ndarray
            another Object, value, or array of values
        validate : bool, optional
            True to validate x and y (see set())
        
        Returns
        -------
        b : bool | ndarray
            the boolean comparison between x and y, or a boolean array of the
            comparison between corresponding elements of x and y. essentially,
            whether x and y are "equal".
        """
        if isinstance(x, Object):
            x = self.get(x)
        elif validate:
            num_values = len(x) if isinstance(x, np.ndarray) and (x.ndim == 2 or self.ndim == 1) else None
            x = self.coerce_value(x, num_values=num_values)
        
        if isinstance(y, Object):
            y = self.get(y)
        elif validate:
            num_values = len(y) if isinstance(y, np.ndarray) and (y.ndim == 2 or self.ndim == 1) else None
            y = self.coerce_value(y, num_values=num_values)
        
        #make sure the sizes are compatible
        if self._ndim > 1:
            if x.ndim == 2 and y.ndim != 2:
                y = self._to_array(y, num_values=len(x))
            elif y.ndim == 2:
                x = self._to_array(x, num_values=len(y))
        
        return self._compare(x, y)
    
    def __call__(self, *args, **kwargs):
        """calling the ObjectAttribute directly mimics calling get()"""
        return self.get(*args, **kwargs)

    def __deepcopy__(self, memo):
        """Deep-copy the attribute. A special method is needed because we need to reset
        all the views into the master data array.

        Parameters
        ----------
        memo : dict
            Memoization dict
        """
        # create a new entity of the right class
        rv = self.__class__.__new__(self.__class__)
        memo[id(self)] = rv        

        # copy all of the attributes
        for attr, value in self.__dict__.iteritems():
            value_id = id(value)
            new_value = memo[value_id] if value_id in memo else deepcopy(value, memo)
            rv.__dict__[attr] = new_value

        assert rv._world is not None
            
        if rv._world._data is not None:
            rv._set_view()

        return rv
    
    def _to_numpy(self, value, from_list=None):
        """make sure a value is a numpy array of attribute values
        
        Parameters
        ----------
        value : any
            a value that is or can be converted to a numpy array
        from_list : bool
            True if value is already a list, and each element of the list should
            become an element of the array. for all but object arrays, numpy can
            determine this automatically, and specifying this parameter is
            unnecessary.
        
        Returns
        -------
        value : ndarray
            value as a numpy array
        """
        if isinstance(value, np.ndarray):
            if value.dtype == self._dtype:  # already well-formatted
                return value
            elif self._dtype != object:  # convert the type
                return value.astype(self._dtype)
        
        if self._dtype == object:  # wrap the value in an object array
            if from_list:
                num_values = len(value)
                x = np.empty((num_values,), dtype=object)
                for idx in xrange(num_values):
                    x[idx] = value[idx]
            else:
                x = np.empty((1,), dtype=object)
                x[0] = value
            
            return x
        else:
            try:
                return np.array(value, dtype=self._dtype)
            except (TypeError, ValueError):
                raise TypeError('must be convertible to %s' % (self._dtype.name))
    
    def _to_vector(self, value):
        """make sure a value is a 1D numpy array of the same length as the
        dimensionality of the ObjectAttribute
        
        Parameters
        ----------
        value : any
            a value that is or can be converted to a 1D numpy array of length
            self.ndim. scalars are tiled.
        
        Returns
        -------
        value : ndarray
            value as a 1D numpy array of length self.ndim
        """
        #make sure we have a numpy array
        value = self._to_numpy(value)
        
        if value.ndim == 1 and len(value) == self._ndim:  # we have what we need
            return value
        elif value.ndim == 0:  # tile if we got a scalar
            return np.tile(value, (self._ndim,))
        elif value.shape == (1, self._ndim) or value.shape == (self._ndim, 1): # convert 1xN to N
            return np.reshape(value, (self._ndim,))
        
        raise ValueError('must be a %d-dimensional vector' % (self._ndim))
    
    def _to_array(self, value, num_values=None, from_list=None):
        """make sure we have a num_values-length array of attribute values. if
        self.ndim > 1, the result will be a num_values x self.ndim array
        
        Parameters
        ----------
        value : any
            a valid attribute value or array of attribute values
        num_values : int, optional
            the number of values that are needed. if this is specified and the
            input cannot be tiled to (num_values,) (or (num_values, self.ndim),
            an exception is raised.
        from_list : bool, optional
            True if value is already a list, and each element of the list should
            become an element of the array. for all but object arrays, numpy can
            determine this automatically, and specifying this parameter is
            unnecessary.
        
        Returns
        -------
        value : ndarray
            value as a (num_values,) or (num_values, self.ndim) numpy array
        """
        #make sure we have a numpy array
        value = self._to_numpy(value, from_list=from_list)
        
        if self._ndim > 1:  # want array of values
            if value.ndim != 2:  # convert to array
                if value.size == 1:  # got  a scalar
                    value = np.tile(value, (1, self._ndim))
                else:
                    try:
                        value = np.reshape(value, (-1, self._ndim))
                    except ValueError:
                        raise ValueError('must be %d-dimensional' % (self._ndim))
        else:  # want vector of values
            if value.ndim != 1: # convert to vector 
                value = np.reshape(value, (-1,))
        
        if num_values is None:
            num_values = len(value)
        
        if len(value) == num_values:  # we have what we need
            return value
        elif len(value) == 1:  # tile to num_values
            reps = (num_values,) if self._ndim == 1 else (num_values, 1)
            value = np.tile(value, reps)
            return value
        else:  # not good
            raise ValueError('needed %d value(s) but got %d' % (num_values, len(value)))
    
    def _coerce_value(self, value, num_values=None):
        """each ObjectAttribute can optionally define a _coerce_value method
        that checks and/or coerces values into values that are valid for the
        ObjectAttribute's dtype. when an ObjectAttribute's public coerce_value
        method is called, each private _coerce_value method in the chain of
        classes is called sequentially from the root member of the class family
        up until the ObjectAttribute itself. i.e. the output of one
        _coerce_value method becomes the input to the next. the method defined
        here performs some basic shape checking on values, and also attempts to
        coerce the value to be of type self.dtype.
        
        a _coerce_value method should not raise an exception in case of an
        invalid value. instead, it should return an Exception describing the
        issue as the second output argument (see below).
        
        Parameters
        ----------
        value : any
            a potential attribute value, or iterable of attribute values
        num_values : int, optional
            the number of values needed
        
        Returns
        -------
        value : self.dtype | ndarray | list
            if num_values was unspecified, then value as a valid attribute
            value. otherwise, a num_values x self.ndim numpy array of valid
            values.
        err : False | Exception
            if the value was successfully coerced into a valid attribute value,
            then should be False. if an exception occurred because the value
            could not be coerced into a valid attribute value, then this should
            be the exception that occurred. note that if coerce_value ends up
            raising an exception because of the failure, its message will be
            this Exception's message, prepended with the name of the
            ObjectAttribute (e.g. if the ObjectAttribute's name is 'mass' and
            the error is 'must be non-negative', then the exception message will
            be 'mass must be non-negative').
        """
        if num_values is None:  # we want a single value
            if self._ndim != 1:
                #make sure we have a vector
                try:
                    value = self._to_vector(value)
                except (TypeError, ValueError) as e:
                    return value, e
            elif not self._dtype == object and not isinstance(value, self._dtype.type):
                #make sure we have the right data type
                try:
                    value = self._dtype.type(value)
                except TypeError as e:
                    return value, TypeError('must be convertible to %s' % (self._dtype.type.__name__))
        else:  # we want multiple values
            try:
                value = self._to_array(value, num_values)
            except (TypeError, ValueError) as e:
                return value, e
        
        return value, False
    
    def _add_world(self, world):
        """see Entity._add_world()"""
        super(ObjectAttribute, self)._add_world(world)
        
        world._object_attributes[self._name] = self
    
    def _remove_world(self):
        """see Entity._remove_world()"""
        try:
            del self._world._object_attributes[self._name]
        except KeyError:  # must have already been removed
            pass
        
        super(ObjectAttribute, self)._remove_world()
    
    def _add_data(self):
        """add a field to the data array that will store the ObjectAttribute's
        data (and any auxillary data that subclasses may use), and additionally
        #store references to attributes we depend on. called during object
        initialization."""
        #add the specified data fields
        for attr,info in self._data_fields.iteritems():
            field = info['field']
            dtype = info.get('dtype', self._dtype)
            null_value = info.get('null_value', self._null_value)
            ndim = info.get('ndim', self._ndim)
            
            self._world._data.add_field(field, dtype, null_value, ndim)
        
        #add the attributes we depend on
        for name in self._depends_on:
            self._other_attr[name] = self._world.get_object_attribute(name)
    
    def _remove_data(self):
        """remove the data objects added by _add_data"""
        #remove the attributes we depend on
        self._other_attr.clear()
        
        #remove the specified data fields
        for attr,info in self._data_fields.iteritems():
            setattr(self, attr, None)
            self._world._data.remove_field(info['field'])
    
    def _set_view(self):
        """create a view to the ObjectAttribute's data field in the master
        array"""
        for attr,info in self._data_fields.iteritems():
            field = info['field']
            
            #try since we might not have actually added the field yet
            try:
                view = self._world._data.view_field(field)
                setattr(self, attr, view)
            except ValueError:
                pass
    
    def _get_mask(self, idx=None):
        """construct a boolean array indicating which elements of the
        ObjectAttribute's data field are masked, i.e. which Object's do not have
        this attribute. the master data array is a numpy MaskedArray, with True
        values in the mask indicating that the Object at that row of the array
        does not have the attribute represented by that field.
        
        Parameters
        ----------
        idx : int | ndarray, optional
            the Object index/indices for which to construct the mask array
        
        Returns
        -------
        mask : bool | ndarray
            the mask value for the specified Objects
        """
        if idx is None:
            mask = self._data.mask
        else:
            mask = self._data.mask[idx]
        
        if mask.ndim==2:
            #data and masks for multi-dimensional attribute dtypes are
            #themselves N x self.ndim arrays, but with our use all mask elements
            #in a row will have the same value
            mask = np.any(mask, axis=1)
        elif self._ndim > 1:
            #this occurs when a single Object index is passed in for a
            #multi-dimensional ObjectAttribute
            mask = np.any(mask)

        return mask
    
    def _get_index(self, x):
        """transform ambiguous input into an Object index or numpy array of
        Object indices
        
        Parameters
        ----------
        x : Object | int | iterable | None
            an Object, Object index, or iterable of Object indices. gets all
            Object indices if x is None.
        
        Returns
        -------
        idx : int | ndarray
            the Object's index or numpy array of indices
        """
        if isinstance(x, np.ndarray):
            #scalar array
            if x.ndim == 0:
                x = np.reshape(x, (1,))
            
            if x.size > 0 and isinstance(x[0], Object):  # array of Objects
                return np.array([obj._id for obj in x], dtype=int)
            elif x.dtype != int:
                return x.astype(int)
            else:
                return x
        elif isinstance(x, Number):  # single index
            return x
        elif isinstance(x, Object):  # single Object
            return x._id
        elif is_iterable(x):  # convert to array
            return self._get_index(np.array(x))
        elif x is None:  # get all indices
            return self.object_indices
        else:
            raise TypeError('%s is not a valid Object or Object index' % (x))
    
    def _get_effective_head_of_family(self, obj):
        """Get an object's effective head of family relative to this attribute, i.e.,
        the highest ancestor such that it and all intermediate objects have
        this attribute.

        Parameters
        ----------
        obj : Object
            The object whose effective head of family we want.

        Returns
        -------
        head : Object
            The object's effective head of family.
        """
        if self._name in obj._effective_head_of_family_cached:
            return obj._effective_head_of_family_cached[self._name]

        # keep going up the tree as long as our current candidate has a parent
        # and that parent has the attribute
        head = obj
        while head._parent is not None and not self._get_mask(head._parent._id):
            head = head._parent

        # cache the answer for all the nodes we examined, which is obj and its
        # ancestors between it and head
        while obj is not head:
            obj._effective_head_of_family_cached[self._name] = head
            obj = obj._parent
        head._effective_head_of_family_cached[self._name] = head

        return head
            

    def _get_simple_indices(self, x):
        """get an array of indices of simple Objects associated with an Object
        or set of Objects
        
        Parameters
        ----------
        x : Object | int | iterable | None
            an Object, Object index, or iterable of Object indices. gets all
            simple Object indices if x is None.
        
        Returns
        -------
        idx : ndarray
            a numpy array of the indices of simple Objects associated with the
            input Object(s)
        """
        #first get the indices of the input Objects
        idx = to_vector(self._get_index(x))
        
        if len(idx) == 0:
            return np.array([], dtype=int)
        
        #now convert to simple indices
        objects = self.world.objects[idx]

        # get unique heads of family
        heads = np.unique([self._get_effective_head_of_family(obj) for obj in objects])

        # get simple family of all heads of family
        idx = np.hstack([obj._simple_leaf_descendants_ids for obj in heads]).astype(int)
        
        # get rid of objects that don't have the attribute
        idx = idx[~self._get_mask(idx)]
        
        return idx
    
    def _get_object_indices(self, object_type=None):
        """get an array of indices of Objects that have this attribute
        
        Parameters
        ----------
        object_type : int, optional
            only return Object's with the specified _class_type (see
            TypeFamily). in practice, this will probably only be used to return
            Objects with class type Object.SIMPLE.
        
        Returns
        -------
        indices : ndarray
            a numpy array of matching Object indices
        """
        #which Objects have this attribute?
        match = ~self._get_mask()
        
        #keep only the specified Object types
        if object_type is not None:
            np.logical_and(match, self._world.object_types == object_type, out=match)
        
        return np.where(match)[0]
    
    def _get_objects(self, object_type=None):
        """get an array of Objects that have this ObjectAttribute
        
        Parameters
        ----------
        object_type : int, optional
            only return Object's with the specified _class_type (see
            TypeFamily). in practice, this will probably only be used to return
            Objects with class type Object.SIMPLE.
        
        Returns
        -------
        objects : ndarray
            a numpy array of matching Objects
        """
        return self._world.objects[self._get_object_indices(object_type=object_type)]

    def _get_leaf_descendant_indices(self, parent_id):
        """get the indices of the leaf descendants (i.e., all descendants who do not
        have children) of an Object that have this attribute
        
        Parameters
        ----------
        parent_id : int
            the index of the parent Object
        
        Returns
        -------
        descendant_indices : ndarray
            an array of indices of the parent's leaf descendants that have this
            attribute
        """
        assert not is_iterable(parent_id)

        parent = self._world.objects[parent_id]
        if not hasattr(parent, 'children') or len(parent.children) == 0:
            descendant_indices = np.array([parent._id], dtype=int)
        else:
            descendant_indices = np.hstack([self._get_leaf_descendant_indices(idx) for idx in parent.children])

        #just the ones that have this attribute
        descendant_mask = self._get_mask(descendant_indices)
        descendant_indices = descendant_indices[~descendant_mask]

        return descendant_indices
    
    def _get_child_indices(self, parent_id):
        """get the indices of the children of an Object that have this attribute
        
        Parameters
        ----------
        parent_id : int
            the index of the parent Object
        
        Returns
        -------
        children_id : ndarray
            an array of indices of the parent's children that have this
            attribute
        """
        assert not is_iterable(parent_id)

        #all child ids
        children_id = getattr(self._world.objects[parent_id], 'children', np.array([], dtype=int))

        #just the ones that have this attribute
        children_mask = self._get_mask(children_id)
        children_id = children_id[~children_mask]
        
        return children_id
    
    def _get_parent_index(self, child_id):
        """get the index of the parent of a child
        
        Parameters
        ----------
        child_id : int
            the index of the child
        
        Returns
        -------
        parent_id : int
            the index of the parent
        """
        return self._world.objects[child_id].parent
    
    def _get_value(self, idx):
        """get the attribute value corresponding to an Object index or array of
        indices. for indices corresponding to Object's without this attribute,
        return the attribute's null value. this method is overloaded by several
        of the ObjectAttribute subclasses below.
        
        Parameters
        ----------
        idx : int | ndarray
            an Object index or array of indices
        
        Returns
        -------
        value : self.dtype | ndarray
            the Object(s) attribute value(s)
        """
        return self._get_data(idx)
    
    def _set_value(self, idx, value):
        """set the attribute value corresponding to an Object index or array of
        indices. note that, because of the way MaskedArrays work, setting the
        value for an Object that doesn't have this attribute will make it appear
        to the data array as if the Object does actually have the attribute, but
        the Object itself won't know that it has the attribute (in other words,
        don't do this).
        
        Parameters
        ----------
        idx : int | ndarray
            an Object index or array of indices
        value : self.dtype | ndarray
            a properly formatted (see coerce_value) set of values to assign to
            as the value of the Object(s) attribute(s)
        """
        if not isinstance(idx, np.ndarray):
            idx = np.array([idx], dtype=int)

        if idx.size != 0:
            value = self._to_array(value, num_values=len(idx))
            self._set_data(idx, value)
    
    def _get_data(self, idx):
        """get values from the attribute's data array
        
        Parameters
        ----------
        idx : int | ndarray
            an index or array of indices in the attribute's data array
        
        Returns
        -------
        values : self.dtype | ndarray
            a value or array of values from the data array
        """
        if self._ndim == 1 and isinstance(idx, Number):
            if self._get_mask(idx):
                return self._data.fill_value
            else:
                return self._data[idx]
        else:
            return self._data[idx].filled()
    
    def _set_data(self, idx, value):
        """set values in the attribute's data array. this is separated out from
        _set_value mainly for PositionObjectAttribute (see
        object_attributes.py), which needs to keep track of all write activity
        to its data array, even if it doesn't happen through _set_value. for
        this reason, nothing should ever directly manipulate an
        ObjectAttribute's data array, but should instead go through this method.
        
        Parameters
        ----------
        idx : ndarray
            an array of indices in the attribute's data array. can be assumed to
            be non-empty.
        value : ndarray
            an array of values to set
        """
        self._data[idx] = value
    
    def _compare(self, x, y):
        """ObjectAttributes may want to override this method to define what a
        "comparison" between two values means. usually this will define what it
        means for two values to be equal. see PositionObjectAttribute for a
        slightly different example.
        
        Parameters
        ----------
        x : self.dtype | ndarray
            a value or array of values
        y : self.dtype | ndarray
            another value or array of values
        
        Returns
        -------
        b : bool | ndarray
            a boolean or boolean array indicating the result of the
            comparison(s)
        """
        if self._ndim > 1:
            return np.all(x == y, axis=-1)
        else:
            return x == y
    
    def _sum(self, x):
        """calculate the sum a set of attribute values. if there are no values,
        return the attribute's null value.
        
        Parameters
        ----------
        x : ndarray
            an array of attribute values (possibly empty)
        
        Returns
        -------
        s : self.dtype
            the sum of the values, or the attribute's null value if there were
            no values to sum
        """
        return self.null_value if x.size == 0 else np.sum(x, axis=0)
    
    def _mean(self, x):
        """calculate the mean of a set of attribute values. if there are no
        values, return the attribute's null value.
        
        Parameters
        ----------
        x : ndarray
            an array of attribute values (possibly empty)
        
        Returns
        -------
        m : self.dtype
            the mean of the values, or the attribute's null value if there were
            no values to mean
        """
        return self.null_value if x.size == 0 else fix_float_integer(np.mean(x, axis=0))
    
    def _mode(self, x):
        """calculate the mode of a set of attribute values. if there are no
        values, return the attribute's null value.
        
        Parameters
        ----------
        x : ndarray
            an array of attribute values (possibly empty)
        
        Returns
        -------
        m : self.dtype
            the mode of the values, or the attribute's null value if there were
            no values to take the mode of
        """
        return self.null_value if x.size == 0 else mode(x)


class CopyGetObjectAttribute(ObjectAttribute):
    """ObjectAttribute that returns copies of attribute values. this is useful
    for mutable attribute types (e.g. when self.ndim > 1, meaning attribute
    values are numpy arrays), to ensure that manipulating values after they are
    retrieved doesn't also manipulate the underlying data in the master data
    array. PointObjectAttributes (below) use this class.
    """
    _name = 'copy_get_attribute'
    
    _auto_state = False
    
    def _get_data(self, idx):
        return copy(super(CopyGetObjectAttribute, self)._get_data(idx))


class CoupledFamilyObjectAttribute(ObjectAttribute):
    """ObjectAttribute whose values are coupled in some way between parent,
    child, and/or sibling Objects. e.g. a parent's mass should be the sum of its
    children's, and changing the position of a child object should propagate to
    corresponding changes in the parent's and sibling's positions. this class
    defines some basic architecture for linking attribute values between family
    members, and the subclasses below then define specific coupling behaviors.
    
    unless altered by a subclass, the coupling rule is that all family members
    share a single attribute value. so changing the value of a child's attribute
    or a parent's attribute changes the value of all family members' attributes.
    """
    _name = 'coupled_family_attribute'
    
    #set this to True if a parent value can be set independently of its
    #children's values at parent Object initialization
    _uncoupled_at_initialization = False
    
    #does setting a child's value propagate up to the parent and then back down
    #to the child's siblings? e.g. changing a child's position should also
    #affect the positions of each of its siblings (_coupled_siblings==True), but
    #changing a child's mass should not (_coupled_siblings==False).
    _coupled_siblings = True
    
    #when adding a child to the family, does the child's value affect the
    #parent's value, or vice versa? e.g. a child's velocity should become the
    #velocity of the family (_child_beats_parent==False), but a child's position
    #should result in an updated parent position, since the parent position is
    #the mean of its children's positions (_child_beats_parent==True).
    _child_beats_parent = False
    
    _auto_state = False
    
    def _set_value(self, idx, value):
        """this overloads the default _set_value in order to provide different
        behaviors depending on whether a parent or child value is being
        set. (Or both, in the case of more-than-two-level hierarchies.)
        
        NOTE that attribute coupling is ignored if multiple values are being
        set at once. this is done both to speed up performance and to avoid
        ambiguous situations, e.g. when both a parent's and its child's values
        are being set at the same time. this does mean, though, that when you
        pass iterables of indices to the set() method, you need to make sure
        that the values you are setting respect the attribute's coupling rules.
        this should happen automatically for the most part, e.g. when
        incrementing all Objects' velocities by their accelerations, the
        coupling defined by the VelocityObjectAttribute will always be
        maintained.
        """
        if is_iterable(idx):
            #skip coupling. see note above.
            super(CoupledFamilyObjectAttribute, self)._set_value(idx, value)
        else:
            obj = self._world.objects[idx]

            old_value = self.get(idx)

            # if we have children, then set the values for the subtree rooted
            # at obj
            if obj.has_children:
                # if we have a parent and _coupled_siblings is True, then skip
                # _set_value_subtree, because the necessary work will be done
                # by _set_value_ancestors
                if not (obj.has_parent and self._coupled_siblings):
                    self._set_value_subtree(idx, value, old_value)

            # if we have a parent, then set all of our ancestors (possibly with
            # coupling)
            if obj.has_parent:
                self._set_value_ancestors(idx, value, old_value)

            # if we have no parent and no children, set it the old-fashioned
            # way
            if not obj.has_parent and not obj.has_children:
                super(CoupledFamilyObjectAttribute, self)._set_value(idx, value)
    
    def _set_value_ancestors(self, idx, value, old_value):
        """set the value of an Object that has ancestors. In the case where
        _coupled_siblings=True, find the head of family and set that to the
        value returned by _calculate_children_values([head_of_family], value,
        old_value).

        as defined here, this causes all family members' attributes to be set
        to the value.
        
        Parameters
        ----------
        idx : int
            the index of the child Object
        value : self.dtype
            the child's new attribute value
        old_value : self.dtype
            the child's old attribute value
        """
        assert not is_iterable(idx)

        if self._coupled_siblings:
            #make sure the children are unmasked
            if np.any(self._get_mask(idx)):
                #set the children's values in order to unmask them
                super(CoupledFamilyObjectAttribute, self)._set_value(idx, value)

            # go up the tree until we reach the head of family or until the
            # next node doesn't have the attribute
            root = self.world.objects[idx]
            while root._parent is not None and hasattr(root._parent, self._name):
                root = root._parent

            # apply the same change to the root to get new value for
            # root
            root_value = self._calculate_children_values([root], value, old_value)

            if root._id == idx:
                # set without coupling
                self.set([root], root_value[0], validate=False)
            else:
                # set with coupling
                self.set(root, root_value[0], validate=False)

        else:
            super(CoupledFamilyObjectAttribute, self)._set_value(idx, value)

            # find our ancestors, not including us. exclude anyone who doesn't
            # have the attribute.
            obj = self.world.objects[idx]
            ancestor_ids = [x._id for x in obj._ancestors if hasattr(x, self._name)]

            # recalculate values for each ancestor based on its leaf descendants
            for id in ancestor_ids:
                leaf_descendants_values = self.get(self.world.objects[id]._leaf_descendants)
                val = self._calculate_parent_value_given_leaf_descendants(id, leaf_descendants_values)
                super(CoupledFamilyObjectAttribute, self)._set_value(id, val)
    
    def _set_value_subtree(self, idx, value, old_value):
        """set the value of an Object, and recursively call _set_value_subtree on all
        the children of the object. Children's values are computed by
        self._calculate_children_values.

        as defined here, this causes all family members' attributes to be set
        to the value.
        """
        assert not is_iterable(idx)

        children_id = self._get_child_indices(idx)
        
        #set the parent's value
        super(CoupledFamilyObjectAttribute, self)._set_value(idx, value)

        #do we have any children?
        if len(children_id) > 0:
            #derive the children's values
            children_values = self._calculate_children_values(children_id, value, old_value)

            #set the values by recursively calling _set_value_subtree, in case
            #they have children of their own
            assert len(children_id) == len(children_values)
            for child_id, child_val in zip(children_id, children_values):
                self._set_value_subtree(child_id, child_val, self.get(child_id))

    def _calculate_children_values(self, children_id, new_value, old_value):
        """calculate the new attribute values to assign to a set of children,
        given the change in the parent's attribute value. subclasses may want to
        override this.

        IMPORTANT NOTE: this should never return a single value.
        
        Parameters
        ----------
        children_id : list
            the list of child indices
        new_value : self.dtype
            the parent's new attribute value
        old_value : self.dtype
            the parent's old attribute value
        
        Returns
        -------
        value : list
            the new attribute values to assign to the children
        """
        #give all children the value of the parent
        return len(children_id) * [new_value]
    
    def _calculate_child_value_given_parent(self, parent_id, parent_value):
        """calculate the default attribute value to assign to a child, given its
        parent value
        
        Parameters
        ----------
        parent_id : int | None
            the parent's id, or None if it does not have one
        parent_value : self.dtype
            the parent's value
        
        Returns
        -------
        value : self.dtype
            the child's value
        """
        return parent_value
    
    def _calculate_parent_value(self, parent_id, new_value=None, old_value=None):
        """calculate the new attribute value to assign to a parent, either given
        the change in a child's attribute value, or based on the set of
        children's current attribute values. if the parent has no children, the
        attribute's null value is used.
        
        Parameters
        ----------
        parent_id : int
            the parent's index
        new_value : self.dtype, optional
            the child's new attribute value. if this argument is omitted, the
            parent's value will be calculated based on the current set of
            child attribute values.
        old_value : self.dtype, optional
            the child's old attribute value
        
        Returns
        -------
        value : self.dtype
            the new attribute value to assign to the parent
        """
        assert not is_iterable(parent_id)

        if new_value is None or old_value is None:  #calculate based on leaf descendants
            descendants_id = self._get_leaf_descendant_indices(parent_id)
            
            if len(descendants_id) == 0:  # Object has no descendants, use the null value
                return self.null_value
            else:
                descendants_values = self.get(descendants_id)
                return self._calculate_parent_value_given_leaf_descendants(parent_id, descendants_values)
        else:
            return self._calculate_parent_value_given_change(parent_id, new_value, old_value)
    
    def _calculate_parent_value_given_leaf_descendants(self, parent_id, descendant_values):
        """calculate the new attribute value to assign to a parent, given the attribute
        values of its leaf descendants. subclasses that overload this method
        can assume that descendant_values is not empty.
        
        Parameters
        ----------
        parent_id : int
            the parent's index
        descendant_values : ndarray
            an array of the descendants' current attribute values
        
        Returns
        -------
        value : self.dtype
            the new attribute value to assign to the parent
        """
        #parent value is the mean of its descendants' value
        return self._mean(descendant_values)
    
    def _calculate_parent_value_given_change(self, parent_id, new_value, old_value):
        """calculate the new attribute value to assign to a parent, given the
        change in one of its children's attribute values
        
        Parameters
        ----------
        parent_id : int
            the parent's index
        new_value : self.dtype
            the child's new attribute value
        old_value : self.dtype
            the child's old attribute value
        
        Returns
        -------
        value : self.dtype
            the new attribute value to assign to the parent
        """
        #child's new value becomes parent's
        return new_value
    
    def _initialize_family(self, parent_id, change, parent_value=None):
        """this method is called by a parent Object when family membership changes, to
        ensure that the family's attribute values are properly coupled. It
        recursively calls _initialize_family if a child has been added and the
        parent object has a parent.
        
        Parameters
        ----------
        parent_id : int
            the parent's index
        change : string
            one of the following, to specify the type of change that occurred to
            the family's membership:
                'parent':   an Object is adopting a new child or children
                'child':    an Object has been added to an existing family, and we
                            are proppagating the change upward 
        parent_value : self.dtype, optional
            the new parent value. if unspecified, derives the parent value from
            the children / existing parent value. overrides 'change'.
        """
        #determine the parent's new value, given the change in the family
        if parent_value is not None:
            value = parent_value
        elif change=='parent' or self._child_beats_parent:
            #calculate the parent's new value based on the new family
            value = self._calculate_parent_value(parent_id)
        elif change=='child':
            #parent' value should stay the same
            value = self.get(parent_id)
        else:
            raise ValueError('"%s" is not a valid change' % (change))
        
        #set without coupling
        self.set([parent_id], value)

        #recursively go up the tree if we added a child and our parent has a
        #parent
        grandparent_id = getattr(self._world.objects[parent_id], 'parent', None)
        if change == 'child' and grandparent_id is not None:
            self._initialize_family(grandparent_id, 'child')

        #if _coupled_siblings is True, set with coupling
        if self._coupled_siblings:
            self.set(parent_id, value)
        

class ModeParentObjectAttribute(CoupledFamilyObjectAttribute):
    """coupling rules:
        siblings are uncoupled
        parent's value is the mode of its children's
    
    ColorObjectAttribute uses this, so that a CompoundObject's children can have
    different colors, while the "color" of the CompoundObject itself registers
    as the dominant color of the children
    """
    _name = 'mode_parent_attribute'
    
    _coupled_siblings = False
    
    _child_beats_parent = True
    
    _auto_state = False
    
    def _calculate_parent_value_given_leaf_descendants(self, parent_id, descendant_values):
        return self._mode(descendant_values)
    
    def _calculate_parent_value_given_change(self, parent_id, new_value, old_value):
        return self._calculate_parent_value(parent_id)


class SumParentObjectAttribute(CoupledFamilyObjectAttribute):
    """coupling rules:
        sibglings are uncoupled
        parent's value is the sum of its children's values
        changing a parent's value causes a proportional change in all children's
            values
    
    MassObjectAttribute uses this, so that the total mass of a CompoundObject is
    the sum of the masses of its children, and e.g. doubling the mass of the
    parent causes each of child's mass to double
    """
    _name = 'sum_parent_attribute'
    
    _coupled_siblings = False
    
    _child_beats_parent = True
    
    _auto_state = False
    
    def _calculate_children_values(self, children_id, new_value, old_value):
        values = self.get(children_id)
        num_children = len(children_id)
        
        if np.all(old_value == 0) or np.all(old_value == np.inf):
            old_value = np.sum(values)
            
        if np.all(old_value == 0) or np.all(old_value == np.inf):
            if num_children == 0:
                new_values = values
            else:
                new_values = copy(values)
                new_values[:] = new_value / num_children
        else:
            new_values = values * new_value / old_value
        
        return fix_float_integer(new_values)
    
    def _calculate_child_value_given_parent(self, parent_id, parent_value):
        if parent_id is None:
            return self.null_value
        else:
            num_children = len(self._get_child_indices(parent_id))
            return parent_value / num_children if num_children != 0 else self.null_value
    
    def _calculate_parent_value_given_leaf_descendants(self, parent_id, descendant_values):
        return self._sum(descendant_values)
    
    def _calculate_parent_value_given_change(self, parent_id, new_value, old_value):
        value = self.get(parent_id)
        
        return fix_float_integer(value + new_value - old_value)


class MeanParentObjectAttribute(CoupledFamilyObjectAttribute):
    """probably wins for my favorite class name ever.
    coupling rules:
        siblings are uncoupled
        parent's value is the mean if its children's values
        changing a parent's value causes all children's values to change
            according to the shift in the parent's value (e.g. if the parent's
            value increases by 2, then all children's values will also increase
            by 2)
    """
    _name = 'mean_parent_attribute'
    
    _coupled_siblings = False
    
    #understandable, considering the parent's temperament
    _child_beats_parent = True
    
    _auto_state = False
    
    def _calculate_children_values(self, children_id, new_value, old_value):
        values = self.get(children_id)
        return fix_float_integer(values + new_value - old_value)
    
    def _calculate_parent_value_given_leaf_descendants(self, parent_id, descendant_values):
        return self._mean(descendant_values)
    
    def _calculate_parent_value_given_change(self, parent_id, new_value, old_value):
        children_id = self._get_child_indices(parent_id)
        assert len(children_id) > 0
        value = self.get(parent_id)
        return fix_float_integer(value + (new_value - old_value)/len(children_id))


class RigidFamilyObjectAttribute(CoupledFamilyObjectAttribute):
    """coupling rules:
        values are potentially uncoupled at initialization (allows for arbitrary
            relative values)
        default parent value is mid-point of child extreme values
        changing any family member value causes all values to change according
            to the shift in the value (e.g. if the parent's value increases by
            2, then all children's values will also increase by 2)
    
    PositionObjectAttribute uses this, so that changing a parent's position
    maintains the relative positions of its children
    """
    _name = 'rigid_family_attribute'
    
    _uncoupled_at_initialization = True
    _coupled_siblings = True
    _child_beats_parent = False
    
    _auto_state = False
    
    def _calculate_children_values(self, children_id, new_value, old_value):
        values = self.get(children_id)
        return fix_float_integer(values + new_value - old_value)
    
    def _calculate_parent_value_given_leaf_descendants(self, parent_id, descendant_values):
        if descendant_values.size == 0:
            return self.null_value
        else:
            return (np.min(descendant_values, axis=0) + np.max(descendant_values, axis=0)) / 2
    
    def _calculate_parent_value_given_change(self, parent_id, new_value, old_value):
        value = self.get(parent_id)
        return fix_float_integer(value + new_value - old_value)


class RandomizingObjectAttribute(ObjectAttribute):
    """ObjectAttribute that can be randomized when the world resets. Subclasses
    should at a minimum overload the _get_random_values() method.
    """
    _name = 'randomizing_attribute'
    
    def randomize(self, x, value=None):
        """randomize the attribute value(s) for a set of Objects
        
        Parameters
        ----------
        x : Object | int | iterable | None
            an Object, object index, or array of object indices, or
            None. randomizes all Object attributes if x is None.
        value : self.dtype, optional
            if random values are being pre-computed, specify it/them here.
        """
        indices = to_vector(self._get_index(x))
        
        if value is None: value = self._get_random_values(indices)
        
        # break up set() calls so that it works with CoupledFamilyAttributes
        for idx,val in zip(indices, value):
            self.set(idx, val)
    
    def _add_world(self, world):
        """PixelWorld has a list attribute to keep track of attributes that
        randomize. add ourselves to that list."""
        super(RandomizingObjectAttribute, self)._add_world(world)
        
        world._add_randomizing_attribute(self)
    
    def _remove_world(self):
        self._world._remove_randomizing_attribute(self)
        
        super(RandomizingObjectAttribute, self)._remove_world()
    
    def _get_random_values(self, indices):
        """Override this to implement your chosen randomization strategy. By
        default, just return the current value.
        
        Parameters
        ----------
        indices : ndarray
            an array of Object indices
        
        Returns
        values : ndarray
            an array of random values
        """
        return self.get(indices)


class SteppingObjectAttribute(ObjectAttribute):
    """ObjectAttribute that does something when the world steps. subclasses
    should at a minimum overload the _step_object() method.
    """
    _name = 'stepping_attribute'
    
    #this causes each SteppingObjectAttribute to have _actions, _step_before,
    #and _step_after attributes, which are lists that are assembled based on
    #the SteppingObjectAttribute's superclasses (see TypeFamily and _actions,
    #_step_before, and _step_after below for details)
    _class_tree_attributes = ['_actions', '_step_before', '_step_after']
    
    #add a step ordering list
    _order_types = ['step']
    
    #if the SteppingObjectAttribute should step either before or after other
    #SteppingObjectAttributes, list them here
    _step_before = []
    _step_after = []
    
    #each SteppingObjectAttribute should define a list actions to which it
    #responds. NOTE that the attribute's step() method will still be called
    #regardless of which action was selected. this attribute just exists so that
    #the world knows which actions are available.
    _actions = []
    
    #this attribute can be used to divide the step into multiple substeps
    #(roughly _substep_factor substeps, but not quite because of floating point
    #error). NOTE that this is not included as a _state_attribute.
    _substep_factor = 1
    
    #new attributes to include in States
    _state_attributes = ['_actions']
    
    @property
    def actions(self):
        """the actions to which the SteppingObjectAttribute responds"""
        return self._actions
    
    def step(self, t, dt, agent_id, action):
        self._world.debug.log('%s step starting (t=%.3f)' % (self._name, t), level=Debug.INFO)
        
        #keep track of how many substeps have occurred
        num_steps = 0
        
        #time at the start of the step
        t_now = t
        
        #time at the end of the step
        t_end = t + dt
        
        #perform steps until we get to the end time
        while t_now < t_end:
            #time delta for the current substep. make sure we don't go past the
            #end time.
            dt_step = min(dt/self._substep_factor, t_end - t_now)
            
            self._step(t_now, dt_step, agent_id, action)
            
            t_now += dt_step
            
            num_steps += 1
        
        self._world.debug.log('%s step finished (%d substep%s)' % (self._name, num_steps, plural(num_steps)), level=Debug.INFO)
    
    def _add_world(self, world):
        """PixelWorld has a list attribute to keep track of attributes that
        step. add ourselves to that list."""
        super(SteppingObjectAttribute, self)._add_world(world)
        
        world._add_stepping_attribute(self)
    
    def _remove_world(self):
        self._world._remove_stepping_attribute(self)
        
        super(SteppingObjectAttribute, self)._remove_world()
    
    def _step(self, t, dt, agent_id, action):
        """override this method and/or _step_object() in subclasses. this is
        where the step processes actually take place.
        
        Parameters
        ----------
        t : float
            the current world time
        dt : float
            the amount of time by which the step should advance the world
        agent_id : int
            the id of the agent performing the action
        action : string
            the name of the action that is being performed
        """
        for obj in self.objects:
            if obj.exists:  # make sure the Object still exists
                self._step_object(obj, t, dt, agent_id, action)
    
    def _step_object(self, obj, t, dt, agent_id, action):
        """override this method and/or _step() in subclasses. this is handles
        the step for a single Object
        
        Parameters
        ----------
        obj : Object
            the Object to step
        t : float
            the current world time
        dt : float
            the amount of time by which the step should advance the world
        agent_id : int
            the id of the agent performing the action
        action : string
            the name of the action that is being performed
        """
        pass


class ChangeTrackingObjectAttribute(SteppingObjectAttribute):
    """ObjectAttribute that keeps track of immediately previous attribute values
    
    PositionObjectAttribute uses this to provide a pseudo-velocity in cases when
    an Object's position has been changed by some method other than a velocity
    step (e.g. by the PushObjectAttribute).
    """
    _name = 'change_tracking_attribute'
    
    #stores data about previous attribute values
    _previous_data = None
    
    #stores the value of the attribute at the beginning of a step
    #PositionObjectAttribute uses this to provide pseudo-velocities with a
    #limited lifespan, e.g. for Objects that have been pushed during the step,
    #but whose push shouldn't register as a movement in future steps.
    _step_previous_data = None
    
    #add fields for previous data (see _data_fields in ObjectAttribute)
    _data_fields = {
        '_previous_data': {'field': lambda (self): '%s_previous' % (self._name)},
        '_step_previous_data': {'field': lambda (self): '%s_step_previous' % (self._name)},
    }
    
    _auto_state = False
    
    def change(self, x=None, step=False):
        """calculate the change between the current and previous attribute
        values. the "change" of attribute values that haven't been set past
        their initial assignment will be 0.
        
        Parameters
        ----------
        x : Object | int | ndarray, optional
            an Object, Object index, or numpy array of Object indices. returns
            the change for all Objects if x is unspecified.
        step : bool, optional
            True to use the value of the attribute at the start of the step as
            the previous value
        
        Returns
        -------
        delta : self.dtype | ndarray
            the change(s) in attribute value(s)
        """
        idx = self._get_index(x)
        
        old_value = self.get_previous(idx, step=step)
        value = self.get(idx)
        
        if self.dtype == np.bool:
            if isinstance(value, np.ndarray):
                return value.astype(int) - old_value
            else:
                return int(value) - old_value
        else:
            return value - old_value
    
    def get_previous(self, x=None, step=False):
        """get the previous attribute value(s)
        
        Parameters
        ----------
        x : Object | int | ndarray
            an Object, Object index, or numpy array of Object indices. returns
            the previous values of all Objects if x is unspecified.
        step : bool, optional
            True to use the value of the attribute at the start of the step as
            the previous value
        
        Returns
        -------
        old_value : self.dtype | ndarray
            the previous attribute value(s)
        """
        idx = self._get_index(x)
        return self._get_previous_value(idx, step=step)
    
    def set_previous(self, x=None, step=False):
        """record the current value as the previous value.
        
        Parameters
        ----------
        x : Object | int | ndarray
            an Object, Object index, or numpy array of Object indices. sets
            the previous values of all Objects if x is unspecified.
        step : bool, optional
            True to record the step previous value, False to record the absolute
            previous value
        """
        idx = self._get_index(x)
        
        data = self._step_previous_data if step else self._previous_data
        data[idx] = self.get(idx)
    
    def set(self, x, value, validate=True):
        """stores the current attribute value before overwriting it"""
        idx = to_vector(self._get_index(x))
        
        #set the previous value if it isn't the first time setting it
        first = self._get_previous_mask(idx)
        idx_set = idx[~first]
        if idx_set.size !=0:
            self.set_previous(idx_set)
        
        super(ChangeTrackingObjectAttribute, self).set(x, value, validate=validate)
        
        #set the current value as the previous value for values being set for
        #the first time
        idx_first = idx[first]
        if idx_first.size != 0:
            self.set_previous(idx_first)
    
    def step(self, *args, **kwargs):
        """at the start of a step, record the current attribute values"""
        #reset the step previous values
        self.set_previous(step=True)
        
        return super(ChangeTrackingObjectAttribute, self).step(*args, **kwargs)
    
    def _get_previous_mask(self, idx=None, step=False):
        """return mask value(s) for the previous attribute data. mask will be
        True for indices whose previous data has not yet been set.
        
        Parameters
        ----------
        idx : int | ndarray
            an index or array of indices of mask elements to return. if
            unspecified, returns all mask values.
        step : bool
            True to return data from _step_previous_data rather than
            _previous_data
        
        Returns
        -------
        mask : bool | ndarray
            the previous data mask value(s)
        """
        data = self._step_previous_data if step else self._previous_data
        mask = data.mask if idx is None else data.mask[idx]
        
        if mask.ndim==2:  #attribute values are multi-dimensional
            mask = np.any(mask, axis=1)
        elif self._ndim > 1:  #single mask of a multi-dimensional attribute
            mask = np.any(mask)
        
        return mask
    
    def _get_previous_value(self, idx, step=False):
        """private method that actually gets the previous value. if no previous
        value exists, returns the current value (so change() becomes 0).
        
        Parameters
        ----------
        idx : int | ndarray
            an Object index or array of Object indices
        step : bool
            True to return data from _step_previous_data rather than
            _previous_data
        """
        data = self._step_previous_data if step else self._previous_data
        
        if isinstance(idx, Number):
            if self._get_previous_mask(idx, step=step):  # no previous value exists
                #just return the current value
                return self.get(idx)
            else:
                return data[idx]
        else:
            value = data[idx].data
            
            first = self._get_previous_mask(idx, step=step)
            idx_first = idx[first]
            if idx_first.size != 0:
                value[first] = self.get(idx_first)
        
            return value


class ConflictObjectAttribute(ObjectAttribute):
    """ObjectAttribute that checks for and resolves conflicts between Objects,
    e.g. when no two Objects can have the same value of the attribute.
    subclasses should at minimum override the _resolve_single_conflict() method,
    and possibly also _conflicts() and _conflicts_with_value(). by default, two
    Objects conflict if a _compare() between their attribute values is True.
    
    unless conflicts() is overriden, only Objects of type Object.SIMPLE can be
    involved in a conflict, and an Object cannot conflict with itself.
    
    PositionObjectAttribute uses this to detect and resolve Object collisions.
    """
    _name = 'conflict_attribute'
    
    #if more than this number of conflicts occur before all conflicts can be
    #resolved, then log a warning and abort conflict resolution. this avoids
    #infinite loops when resolving a conflict always leads to another conflict.
    _max_conflicts_before_abort = 50
    
    #keep track of new conflicts whose resolution is deferred until some process
    #finishes
    _deferred_conflicts = None
    _conflicts_are_deferred = False
    
    #new attributes to include in States
    _state_attributes = ['_deferred_conflicts', '_conflicts_are_deferred']
    
    def __init__(self, *args, **kwargs):
        self._clear_deferred_conflicts()
        
        super(ConflictObjectAttribute, self).__init__(*args, **kwargs)
    
    def set_unsafe(self, x, value, validate=True):
        """set attribute values without checking for conflicts"""
        super(ConflictObjectAttribute, self).set(x, value, validate=validate)
    
    def set(self, x, value, validate=True):
        """set attribute values, then check for and resolve conflicts involving
        the Objects whose attribute values were set"""
        idx = self._get_index(x)
        
        self.set_unsafe(idx, value, validate=validate)
        
        self.resolve_conflicts(idx)
    
    def resolve_conflicts(self, x=None, y=None):
        """check for and resolve conflicts between the specified Objects
        
        Parameters
        ----------
        x : Object | int | ndarray
            an Object, Object index, or array of Object indices. if unspecified,
            uses all Object indices.
        y : Object | int | ndarray, optional
            the set of Objects to check for conflicts with the Objects in x. if
            unspecified, compares x with all Objects with the attribute.
        """
        #get vectors of the indices to check for conflicts
        idx1 = self._get_simple_indices(x)
        idx2 = self._get_simple_indices(y)
        
        #construct each pair of indices to check
        idx1, idx2 = cartesian_product(idx1, idx2, matching_pairs=False)
        if idx1.size == 0:  # no objects
            return
        
        #check each pair for conflicts
        conflicts = self.conflicts(idx1, idx2)
        
        #resolve any conflicts
        if any(conflicts):
            idx1 = idx1[conflicts]
            idx2 = idx2[conflicts]
            if self._conflicts_are_deferred:
                #add the conflicting pairs to our list of conflicts
                self._append_conflicts(idx1, idx2)
            else:
                #resolve them immediately
                self._resolve_conflicts(idx1, idx2)
    
    def conflicts(self, idx1, idx2):
        """determine whether pairs of Objects are in conflict
        
        Parameters
        ----------
        idx1 : int | ndarray
            one Object index or array of Object indices in the pair
        idx2 : int | ndarray
            the other Object index or array of Object indices in the pair
        
        Returns
        -------
        conflicts : bool | ndarray
            True if the Object pair(s) conflict
        """
        idx1 = to_vector(idx1)
        idx2 = to_vector(idx2)
        
        #only simple objects can conflict
        simple1 = self._world.object_types[idx1] == Object.SIMPLE
        simple2 = self._world.object_types[idx2] == Object.SIMPLE
        conflicts = np.logical_and(simple1, simple2)
        
        #objects cannot conflict with themselves
        self_pair = idx1 == idx2
        np.logical_and(conflicts, ~self_pair, out=conflicts)
        
        #check the remainder for conflicts
        conflicts[conflicts] = self._conflicts(idx1[conflicts], idx2[conflicts])
        
        return conflicts
    
    def conflicts_with_value(self, idx, value):
        """determine whether Objects would conflict with other Objects that had
        the specified attribute value
        
        Parameters
        ----------
        idx : int | ndarray
            an Object index or array of Object indices
        value : self.dtype
            the value to check for conflicts
        """
        idx = to_vector(idx)
        value = self.coerce_value(value, num_values=len(idx))
        
        #only simple objects can conflict
        conflicts = self._world.object_types[idx] == Object.SIMPLE
        
        #check the remainder for conflicts
        conflicts[conflicts] = self._conflicts_with_value(idx[conflicts], value[conflicts])
        
        return conflicts
    
    def find_conflicts(self, value):
        """find Objects that would conflict with an Object that had thevalue
        
        Parameters
        ----------
        value : self.dtype
            the value to check for conflicts
        
        Returns
        -------
        indices : ndarray
            an array of indices of Objects that would conflict with the value
        """
        indices = self.object_indices
        
        return indices[self.conflicts_with_value(indices, value)]
    
    def _conflicts(self, idx1, idx2):
        """private version of conflicts(). subclasses may want to override this,
        if a conflict means something other than two Objects resulting in a
        True comparison.
        
        Parameters
        ----------
        idx1 : ndarray
            an array of Object indices
        idx2 : ndarray
            another array of Object indices that should be compared with the
            Objects in idx1. idx1 and idx2 can be assumed to be different and
            to refer to simple Objects.
        
        Returns
        -------
        conflicts : ndarray
            a boolean array indicating whether the Objects conflict
        """
        x1 = self.get(idx1)
        x2 = self.get(idx2)
        
        return self._compare(x1, x2)
    
    def _conflicts_with_value(self, idx, value):
        """private version of conflicts_with_value(). subclasses will probably
        want to override this if they also override _conflicts().
        
        Parameters
        ----------
        idx : ndarray
            an array of Object indices. idx can be assumed to refer to simple
            Objects
        value : ndarray
            an array of values to compare with the Objects' attribute values
        
        Returns
        -------
        conflicts : ndarray
            a boolean array indicating whether the Objects conflict with the
            specified values
        """
        x = self.get(idx)
        
        return self._compare(x, value) 
    
    def _resolve_conflicts(self, idx1, idx2):
        """resolve conflicts between Object pairs that have already been
        determined to be in conflict. this default version resolves conflicts
        sequentially, keeping track of and later resolving any new conflicts
        that occur during conflict resolution. subclasses that have some way of
        resolving conflicts in parallel may want to override this method.
        
        Parameters
        ----------
        idx1 : ndarray
            an array of indices of the first Objects in the conflicting pairs
        idx2 : ndarray
            an array of indices of the second Objects in the conflicting pairs
        """
        #should we print debug information?
        print_debug = self._world.debug.test(Debug.INFO)
        
        #defer any new conflicts that occur while we are resolving this set
        resolve_after = self._defer_conflicts()
        
        for i1, i2 in zip(idx1, idx2):
            #make sure we haven't reached our conflict limit
            if len(self._deferred_conflicts) >= self._max_conflicts_before_abort:
                self._world.debug.log('could not resolve %s conflicts (more than %d conflicts occurred)' % (self._name, self._max_conflicts_before_abort), level=Debug.WARN)
                
                self._clear_deferred_conflicts()
                
                break;
            
            #make sure the objects still conflict
            if self._conflicts(np.array([i1]), np.array([i2])):
                #finally, actually resolve the conflict
                resolved = self._resolve_single_conflict(i1, i2)
                
                if print_debug:
                    obj1 = self._world.objects[i1]
                    obj2 = self._world.objects[i2]
                    
                    if resolved:
                        self._world.debug.log('resolved %s conflict between %s and %s' % (self._name, obj1.unique_name, obj2.unique_name), level=Debug.INFO)
                    else:
                        self._world.debug.log('could not resolve %s conflict between %s and %s' % (self._name, obj1.unique_name, obj2.unique_name), level=Debug.INFO)
            elif print_debug:
                obj1 = self._world.objects[i1]
                obj2 = self._world.objects[i2]
                self._world.debug.log('%s conflict between %s and %s resolved itself' % (self._name, obj1.unique_name, obj2.unique_name), level=Debug.INFO)
        
        #check for and resolve deferred conflicts
        if resolve_after:
            self._resolve_deferred_conflicts()
    
    def _append_conflicts(self, idx1, idx2):
        """append conflicts to the list of deferred conflicts
        
        Parameters
        ----------
        idx1 : ndarray
            an array of indices of the first Objects in the conflicting pairs
        idx2 : ndarray
            an array of indices of the second Objects in the conflicting pairs
        """
        #encode the conflicts
        conflicts = np.column_stack((idx1, idx2))
        
        #append the conflicts to the deferred conflict list
        self._deferred_conflicts = np.vstack((self._deferred_conflicts, conflicts))
        
        #optionally print a debug message
        if self._world.debug.test(Debug.INFO):
            for i1, i2 in conflicts:
                obj1 = self._world.objects[i1]
                obj2 = self._world.objects[i2]
                self._world.debug.log('deferring %s conflict between %s and %s' % (self._name, obj1.unique_name, obj2.unique_name), level=Debug.INFO)
    
    def _defer_conflicts(self):
        """start (or continue) deferring conflict resolution
        
        Returns
        -------
        resolve_after : bool
            True if the calling process should call _resolve_deferred_conflicts
            after it finishes doing whatever needed conflict resolution deferred
        """
        #only resolve after if conflict resolution isn't already deferred
        resolve_after = not self._conflicts_are_deferred
        
        self._conflicts_are_deferred = True
        
        return resolve_after
    
    def _resolve_deferred_conflicts(self):
        """resolve conflicts that occurred while conflict resolution was
        deferred
        """
        #should we print debug information?
        print_debug = self._world.debug.test(Debug.INFO)
        
        idx_start = 0
        num_conflicts = len(self._deferred_conflicts)
        
        #while we still have conflicts
        while idx_start < num_conflicts:
            if print_debug:
                self._world.debug.log('resolving %d deferred %s conflict%s' % (num_conflicts, self._name, plural(num_conflicts)), level=Debug.INFO)
            
            idx1 = self._deferred_conflicts[idx_start:,0]
            idx2 = self._deferred_conflicts[idx_start:,1]
            
            #resolve the current set of deferred conflicts
            self._resolve_conflicts(idx1, idx2)
            
            #determine how many conflicts occur now (in case more conflicts
            #occurred during conflict resolution)
            idx_start += num_conflicts
            num_conflicts = len(self._deferred_conflicts)
        
        #clear the deferred conflicts list
        self._clear_deferred_conflicts()
        
        #conflicts are no longer deferred
        self._conflicts_are_deferred = False
    
    def _clear_deferred_conflicts(self):
        """clear the list of deferred conflicts"""
        self._deferred_conflicts = np.empty(shape=(0,2), dtype=int)
    
    def _resolve_single_conflict(self, idx1, idx2):
        """at a minimum, subclasses need to override this method in order to
        define how a single conflict between two Objects should be resolved
        
        Parameters
        ----------
        idx1 : int
            the index of the first conflicting Object
        idx2 : int
            the index of the second conflicting Object
        """
        raise NotImplementedError


class DerivedObjectAttribute(ObjectAttribute):
    """ObjectAttribute whose values are derived from other attributes, rather
    than providing their own field in the master data array. subclasses should
    at a minimum override _get_data_object(). writeable DerivedObjectAttribute's
    should additionally override _set_data_object() (or _set_data()).
    
    MomentumObjectAttribute and KineticEnergyObjectAttribute use this, since
    their values depend only on mass and velocity.
    """
    _name = 'derived_attribute'
    
    #remove the self data field, since we don't have it (see _data_fields in
    #ObjectAttribute)
    _data_fields_removed = ['_data']
    
    _read_only = True
    
    _auto_state = False
    
    @property
    def data(self):
        """mimics the data array of ObjectAttributes that have their own data"""
        idx = np.arange(self._world.num_objects, dtype=int)
        
        d = self.get(idx)
        m = self._get_mask(idx)
        
        if self._ndim > 1:
            m = np.tile(np.reshape(m, (-1, 1)), (1, self._ndim))
        
        return ma.array(d, mask=m)
    
    def get(self, x=None):
        """returns the derived attribute value for Objects that have the
        attribute, or the null value for Objects that do not have the attribute.
        """
        idx = self._get_index(x)
        
        #which Objects have this attribute?
        has_me = ~self._get_mask(idx)
        
        if is_iterable(idx):
            assert len(has_me.shape) == 1
            if self._ndim > 1:
                shape = (len(has_me), self._ndim)
            else:
                shape = has_me.shape

            value = np.empty(shape, dtype=self._dtype)

            #Objects with the attribute get the derived value
            value[has_me] = super(DerivedObjectAttribute, self).get(idx[has_me])
            
            #Objects without the attribute get the null value
            value[~has_me] = self._null_value
            
            return value
        elif has_me:  # Object has the attribute
            return super(DerivedObjectAttribute, self).get(idx)
        else:  # Object does not have the attribute
            return self.null_value
    
    def _get_mask(self, idx=None):
        """mimic the _get_mask behavior of ObjectAttributes with their own data
        """
        if idx is None:
            idx = np.arange(self._world.num_objects)
        elif not is_iterable(idx):
            idx = [idx]
        
        return ~np.array([self._name in self._world.objects[i]._attributes for i in idx], dtype=bool)
    
    def _get_data(self, idx):
        if isinstance(idx, Number):
            obj = self._world.objects[idx]
            return self._get_data_object(obj)
        else:
            objects = self._world.objects[idx]
            values = [self._get_data_object(obj) for obj in objects]
            return self._to_array(values, from_list=True)
    
    def _set_data(self, idx, value):
        if self._read_only:
            raise RuntimeError('%s is read-only' % (self._name))

        objects = self._world.objects[idx]
        assert len(objects) == len(value)
        for obj,v in zip(objects, value):
            self._set_data_object(obj, v)
    
    def _get_data_object(self, obj):
        """subclasses should override this (or alternatively _get_data()),
        defining how the derived value is computed for a single Object
        
        Parameters
        ----------
        obj : Object
            the Object
        
        Returns
        -------
        value : self.dtype
            the Object's derived attribute value
        """
        raise NotImplementedError
    
    def _set_data_object(self, obj, value):
        """writeable subclasses should override this (or alternatively
        _set_data()), defining what "setting" a derived value means for a single
        Object
        
        Parameters
        ----------
        obj : Object
            the Object
        value : self.dtype
            the value to set
        """
        raise NotImplementedError


class SyncObjectAttribute(ObjectAttribute):
    """an ObjectAttribute that keeps in sync with an actual attribute of the
    Object object.
    
    NOTE that syncing is one-sided. if the other attribute value is changed, the
    change will not be reflected in the SyncObjectAttribute. so,
    SyncObjectAttribute's should only be used when the SyncObjectAttribute
    itself will be the only attribute in the pair that is modified directly. see
    LinkObjectAttribute if bidirectional syncing is needed.
    
    see NameObjectAttribute below for an example.
    """
    _name = 'sync_attribute'
    
    #by default, the synced attribute is the SyncObjectAttribute's name preceded
    #by an underscore
    _synced_attribute = None
    
    _auto_state = False
    
    def __init__(self, *args, **kwargs):
        if self._synced_attribute is None:
            self._synced_attribute = '_%s' % (self._name)
        
        super(SyncObjectAttribute, self).__init__(*args, **kwargs)
    
    def _default_value(self, obj):
        return getattr(obj, self._synced_attribute)
    
    def _set_data(self, idx, value):
        #set the actual attribute for each Object
        for i,v in zip(idx, value):
            setattr(self._world.objects[i], self._synced_attribute, v)
        
        super(SyncObjectAttribute, self)._set_data(idx, value)


class LinkObjectAttribute(DerivedObjectAttribute):
    """an ObjectAttribute that is just a link to an actual attribute of the
    Object object.
    """
    _name = 'link_attribute'
    
    #by default, the linked attribute is the LinkObjectAttribute's name preceded
    #by an underscore
    _linked_attribute = None
    
    _read_only = False
    
    _auto_state = False
    
    def __init__(self, *args, **kwargs):
        if self._linked_attribute is None:
            self._linked_attribute = '_%s' % (self._name)
        
        super(LinkObjectAttribute, self).__init__(*args, **kwargs)
    
    def _default_value(self, obj):
        return getattr(obj, self._linked_attribute)
    
    def _set_data_object(self, obj, value):
        setattr(obj, self._linked_attribute, value)
    
    def _get_data_object(self, obj):
        return getattr(obj, self._linked_attribute)


class NonNegativeObjectAttribute(ObjectAttribute):
    """ObjectAttribute whose values must be non-negative"""
    _name = 'non_negative_attribute'
    
    _auto_state = False
    
    def _coerce_value(self, value, **kwargs):
        """add a check to ensure that values are >= 0"""
        valid = np.all(value >= 0)
        err = False if valid else TypeError('must be non-negative')
        return value, err


class ScalarObjectAttribute(ObjectAttribute):
    """ObjectAttribute whose values must be scalars. this actually doesn't do
    anything since the base ObjectAttribute is already a scalar int (oh well).
    """
    _name = 'scalar_attribute'
    
    _default_value = 0
    _null_value = 0
    
    _auto_state = False


class IntegerObjectAttribute(ScalarObjectAttribute):
    """ObjectAttribute whose values must be integers"""
    _name = 'integer_attribute'
    
    _dtype = np.dtype(int)
    
    _auto_state = False


class FloatObjectAttribute(ScalarObjectAttribute):
    """ObjectAttribute that must be floats"""
    _name = 'float_attribute'
    
    _dtype = np.dtype(float)
    
    _auto_state = False


class NonNegativeIntegerObjectAttribute(IntegerObjectAttribute, NonNegativeObjectAttribute):
    """convenience class for non-negative integer attributes"""
    _name = 'non_negative_integer_attribute'
    
    _auto_state = False


class NonNegativeFloatObjectAttribute(FloatObjectAttribute, NonNegativeObjectAttribute):
    """convenience class for non-negative float attributes"""
    _name = 'non_negative_float_attribute'
    
    _auto_state = False


class FractionalFloatObjectAttribute(FloatObjectAttribute, NonNegativeObjectAttribute):
    """class for float attributes between 0 and 1 (inclusive)"""
    _name = 'non_negative_float_attribute'
    
    _auto_state = False

    def _coerce_value(self, value, **kwargs):
        """add a check to ensure that values are <= 1"""
        valid = np.all(value <= 1)
        err = False if valid else TypeError('must be less than or equal to one')
        return value, err


class BooleanObjectAttribute(ScalarObjectAttribute):
    """ObjectAttribute whose values must be bools"""
    _name = 'boolean_attribute'
    
    _dtype = np.dtype(bool)
    
    _default_value = True
    _null_value = False
    
    _auto_state = False


class StringObjectAttribute(ObjectAttribute):
    """ObjectAttribute whose values must be strings. because of constraints in
    the master data array, memory must (or at least should, for efficiency) be
    pre-allocated. because of this, attribute values have a maximum length of
    255 characters."""
    _name = 'string_attribute'
    
    _dtype = np.dtype('S255')
    
    _default_value = ''
    _null_value = ''
    
    _auto_state = False


class ObjectObjectAttribute(ObjectAttribute):
    """ObjectAttribute that stores np.dtype('O') values. values can be coerced
    individually using _coerce_single_value()"""
    _name = 'object_attribute'
    
    _dtype = np.dtype('O')
    
    _default_value = None
    _null_value = None
    
    _auto_state = False
    
    def _coerce_value(self, value, num_values=None, **kwargs):
        if num_values is not None:
            for idx in xrange(len(value)):
                value[idx], err = self._coerce_single_value(value[idx])
                
                if err:
                    return value, err
        else:
            value, err = self._coerce_single_value(value)
            
            if err:
                return value, err
        
        return value, False
    
    def _coerce_single_value(self, value, **kwargs):
        return value, False


class ListObjectAttribute(ObjectObjectAttribute):
    """ObjectAttribute that whose values be lists (or None)."""
    _name = 'list_attribute'
    
    _auto_state = False
    
    @property
    def _default_value(self):
        return []
    
    def _coerce_single_value(self, value, **kwargs):
        """make sure the value is a list"""
        if value is None:
            return value, False
        else:
            try:
                value = list(value)
            except TypeError:
                return value, TypeError('must be convertible to a list')
        
        return value, False


class FloatVectorObjectAttribute(FloatObjectAttribute, CopyGetObjectAttribute):
    """ObjectAttribute that represents a vector of floats. this class also
    subclasses from CopyGetObjectAttribute so that returned values don't share
    data with the underlying master data array. subclasses should define the
    vector dimensionality via the _ndim attribute (see PointObjectAttribute
    below).
    """
    _name = 'float_vector_attribute'
    
    _auto_state = False


class PointObjectAttribute(FloatVectorObjectAttribute):
    """ObjectAttribute that represents points in a 2D plane"""
    _name = 'point_attribute'
    
    _ndim = 2
    
    _auto_state = False


class HasObjectAttribute(BooleanObjectAttribute, DerivedObjectAttribute):
    """boolean ObjectAttribute that indicates whether an Object has another
    attribute"""
    _name = 'has_attribute'
    
    #the name of the ObjectAttribute that this attribute should test for
    _has_attribute = None
    
    _auto_state = False
    
    def _get_data(self, idx):
        attr = self._world._object_attributes.get(self._has_attribute, None)
        
        if attr is None:
            if isinstance(idx, Number):
                return False
            else:
                return np.zeros(idx.shape, dtype=bool)
        else:
            return np.logical_not(attr._get_mask(idx))


class AbilityObjectAttribute(BooleanObjectAttribute, SteppingObjectAttribute):
    """use to implement agent abilities. the attribute indicates whether the
    ability is activated, and _execute_action executes activated abilities.
    actions are only registered with the world when an object has the attribute
    and it is set to True.

    If the object has the attribute 'controller', then the ability is only
    triggered when obj.controller is equal to the current agent's id, or when
    obj.controller is equal to -1 (which means that it is controlled by
    everyone).
    """
    _name = 'ability_attribute'
    
    @property
    def actions(self):
        if any(self.get()):
            return super(AbilityObjectAttribute, self).actions
        else:
            return []
    
    def _step_object(self, obj, t, dt, agent_id, action):
        if self.get(obj._id):
            if not hasattr(obj, 'controller') or obj.controller == agent_id or obj.controller == -1:
                self._execute_action(obj, t, dt, agent_id, action)
    
    def _execute_action(self, obj, t, dt, agent_id, action):
        raise NotImplementedError


class InteractsObjectAttribute(SteppingObjectAttribute, BooleanObjectAttribute):
    """Attribute that deals with interactions that should take place when two
    objects occupy the same position. When that happens, the _interact()
    function is called with the two objects in question.

    If either object has visible=False, then _interact will not be called.

    This attribute has a boolean value. If it is set to False, the behavior is
    disabled and _interact() will not be called

    Note that if two objects both have InteractsObjectAttributes, _interact
    will get called twice when they intersect, once as self._interact(a, b) and
    once as self._interact(b, a). (Unless the interaction has the effect that
    they are no longer intersecting after the first call.)

    Subclasses should override _interact().
    """
    _name = 'interacts_attribute'
    
    _depends_on = ['state_index']

    _class_tree_attributes = ['_interacts_types']

    # a list of classes that we can interact with. If empty, we can interact
    # with objects of any class. At preparation time, this is transformed from
    # a list of names to a list of classes.
    _interacts_types = []

    def prepare(self):
        self._interacts_types = [Object.get_class(x) for x in self._interacts_types]

    def _step_object(self, obj, t, dt, agent_id, action):
        """Check for intersections between the object and other objects and trigger
        _interact when those objects meet our criteria (visible, of the right
        class if _interacts_types is non-empty).

        Parameters
        ----------
        obj : Object
            The object that has the InteractsObjectAttributes
        t : number
            The simulation time.
        dt : number
            The time since the last step.
        agent_id : int
            Id of the agent currently moving.
        action : string
            The most recent action executed.
        """
        if not self.get(obj):
            return

        if not obj.visible:
            return

        objs = self.world.objects.find(state_index=obj.state_index, visible=True)
        for obj2 in objs:
            if not self._interacts_types or any(isinstance(obj2, cls) for cls in self._interacts_types):
                if obj2 is not obj and obj2.visible:
                    self._interact(obj, obj2)

    def _interact(self, obj1, obj2):
        """Override this function in subclasses.
                   
        Parameters
        ----------
        obj1 : Object
            The first object. This is the object that has the
            InteractsObjectAttribute 

        obj2 : Object
            The second object.
        """
        raise NotImplementedError


class ListeningObjectAttribute(BooleanObjectAttribute, SteppingObjectAttribute):
    """Object attribute that makes objects listen for events. Subclasses should
    override _process_event() and specify event names to respond to in
    _selected_events.

    Setting this attribute to False will disable the listening behavior.
    """
    _name = 'listening_attribute'
    
    _class_tree_attributes = ['_selected_events']

    # names of events to respond to. This is a class tree attribute.
    _selected_events = []

    def _step(self, t, dt, agent_id, action):
        """Step the attribute by scanning for relevant events.
        """
        for evt in self.world.step_events:
            if evt.name in self._selected_events:
                self._process_event(evt, t, dt, agent_id, action)

    def _process_event(self, evt, t, dt, agent_id, action):
        """Override this method in subclasses to respond to events in a vectorized
        fashion. (But remember to check if listening is disabled!)

        Parameters
        ----------
        evt : Event
            The event to respond to.
        t : number
            The simulation time.
        dt : number
            Time since last step.
        action : string
            Last selected action
        """
        for obj in self.objects:
            if self.get(obj):
                self._process_event_object(evt, obj, t, dt, agent_id, action)

    def _process_event_object(self, evt, obj, t, dt, agent_id, action):
        """Override this method in subclasses to respond to events correctly.

        Parameters
        ----------
        evt : Event
            The event to respond to.
        obj : Object
            The object that is responding.
        t : number
            The simulation time.
        dt : number
            Time since last step.
        action : string
            Last selected action
        """
        raise NotImplementedError


class StateMachineObjectAttribute(SteppingObjectAttribute, 
                                  NonNegativeIntegerObjectAttribute):
    """an attrubute for implementing state machine-like behavior in objects. At
    every step, we do:

    new_state = self.execute(obj, old_state)

    Subclasses should override _execute wth a function that branches on state
    and returns the new state. States are non-negative integers. Subclasses
    should raise an exception if an unknown state is passed into _execute().
    """
    _name = 'state_machine_attribute'
    
    def _step_object(self, obj, t, dt, agent_id, action):
        old_state = self.get(obj)
        new_state = self.execute(obj, old_state)
        self.set(obj, new_state)

    def execute(self, obj, old_state):
        """Override this function to implement your chosen behavior.

        Parameters
        ----------
        old_state : int
            The state at the beginning of execution.
        
        Returns
        -------
        new_state : int
            The state at the end of execution.
            
        """
        return self._execute(obj, old_state)

    def _execute(self, obj, old_state):
        """Override this function to implement your chosen behavior.

        Parameters
        ----------
        old_state : int
            The state at the beginning of execution.
        
        Returns
        -------
        new_state : int
            The state at the end of execution.
        """
        raise NotImplementedError


class NameObjectAttribute(SyncObjectAttribute, StringObjectAttribute):
    """the first ObjectAttribute that is actually used by Objects! this
    attribute keeps in sync with the Object object's _name attribute.
    """
    _auto_state = False


class IdObjectAttribute(SyncObjectAttribute, NonNegativeIntegerObjectAttribute):
    """ObjectAttribute that keeps in sync with an Object's _id
    """
    _auto_state = False


class ChildrenObjectAttribute(ObjectObjectAttribute, DerivedObjectAttribute):
    """provides an array of a parent's children's indices"""
    _auto_state = False
    
    def _get_data_object(self, obj):
        return np.array([child._id for child in obj._children], dtype=int)


class DescendantsObjectAttribute(ObjectObjectAttribute, LinkObjectAttribute):
    """provides an array of an object's descendants' indices"""
    _linked_attribute = '_descendants_ids'

    _read_only = True

    _auto_state = False


class LeafDescendantsObjectAttribute(ObjectObjectAttribute, LinkObjectAttribute):
    """provides an array of a parent's leaf descendants' indices"""
    _linked_attribute = '_leaf_descendants_ids'
    
    _read_only = True

    _auto_state = False


class SimpleLeafDescendantsObjectAttribute(ObjectObjectAttribute, LinkObjectAttribute):
    """provides an array of a parent's leaf descendants' indices, but only for
    those descendants which are of SIMPLE type"""
    _linked_attribute = '_simple_leaf_descendants_ids'

    _read_only = True

    _auto_state = False    

class ParentObjectAttribute(IntegerObjectAttribute, DerivedObjectAttribute):
    """provides a reference to the index of a child's parent"""
    _null_value = -1
    
    _auto_state = False
    
    def _get_data_object(self, obj):
        if obj._parent is None:
            return -1
        else:
            return obj._parent._id


class AncestorsObjectAttribute(ObjectObjectAttribute, LinkObjectAttribute):
    """provides an array of an object's ancestors' indices, not including the object"""
    _linked_attribute = '_ancestors_ids'

    _read_only = True

    _auto_state = False


class HeadOfFamilyObjectAttribute(IntegerObjectAttribute, LinkObjectAttribute):
    """provides the object's head of family's id"""
    _linked_attribute = '_head_of_family_id'

    _read_only = True

    _auto_state = False

class TopLevelObjectAttribute(DerivedObjectAttribute):
    """Whether an object is top-level (its own head of family)"""
    _read_only = True

    _depends_on = ['head_of_family', 'id']

    _dtype = np.dtype(bool)

    _auto_state = False

    def _get_data(self, idx):
        head_of_family = self._other_attr['head_of_family'].get(idx)
        id = self._other_attr['id'].get(idx)
        return (head_of_family == id)
    

class FamilyObjectAttribute(ObjectObjectAttribute, LinkObjectAttribute):
    """provides an array of all indices in an Object's family"""
    _linked_attribute = '_family_ids'

    _read_only = True

    _auto_state = False


class SimpleFamilyObjectAttribute(ObjectObjectAttribute, LinkObjectAttribute):
    """provides an array of the indices of all simple objects in an Object's family"""
    _linked_attribute = '_simple_family_ids'

    _read_only = True

    _auto_state = False


class MoveableObjectAttribute(HasObjectAttribute):
    """boolean ObjectAttribute that indicates whether an Object has the velocity
    attribute. by convention, Objects that have no velocity (e.g. WallObjects)
    are considered immoveable and are not affected by Object collisions.
    """
    _has_attribute = 'velocity'
    
    _auto_state = False


class PushableObjectAttribute(HasObjectAttribute):
    """boolean ObjectAttribute that indicates whether an Object has the mass
    attribute. by convention, Objects that have no mass (e.g. SelfObjects) are
    considered unpushable, essentially like infinitely massive Objects. they
    cannot be pushed, and are not affected by collisions with massive Objects,
    but they will bounce off of immoveable objects (i.e. the infinte mass of
    immoveable Objects is greater than the infinite mass of of unpushable
    Objects).
    """
    _has_attribute = 'mass'
    
    _auto_state = False


#---Objects--------------------------------------------------------------------#
class Object(Entity):
    """an Object is essentially just a collection of ObjectAttribute values.
    Objects should avoid defining their own behaviors. if an Object needs a
    specific behavior, the behavior should instead be implemented as an
    ObjectAttribute that the Object possesses.
    
    an Object's ObjectAttribute values can be set just like any other python
    attribute (e.g. obj.position = (2, 2)). however, this is relatively
    inefficient when setting multiple attribute values (e.g. incrementing all
    positions by velocity). in that case, the ObjectAttribute's set() method
    should be used instead.
    """
    class __metaclass__(type(Entity)):
        """this metaclass just sorts the Object's attributes according to the
        initialization order specified collectively by its ObjectAttributes"""
        def __new__(meta, name, bases, dct):
            cls = type(Entity).__new__(meta, name, bases, dct)
            
            #sort the attributes according to the initialization order
            cls._attributes = topological_sort(cls._attributes,
                                ObjectAttribute._attribute_order['initialize'])
            
            return cls
    
    #create a new class family
    _class_family = 'object'
    
    #the object's name. if subclasses do not assign this explicitly, the name
    #is derived sensibly from the class name, e.g. SelfObject becomes 'self' and
    #BasicSelfObject becomes 'basic_self'.
    _name = 'object'
    
    #this causes each Object class to have _attributes and _defaults attributes,
    #which are a list and dict (respectively) that are assembled based on the
    #Object's superclasses (see TypeFamily and _attributes and _defaults below)
    _class_tree_attributes = ['_attributes', '_defaults']
    
    #the names of attributes added by this object. note that at class
    #construction, TypeFamily replaces this with an aggregated list of all
    #attributes possessed by the Object. then, at object creation, this is
    #replaced with a dict of the ObjectAttribute objects themselves. Objects
    #that wish to remove some attributes added by superclasses should
    #additionally define a list attribute named _attributes_removed.
    _attributes = ['name', 'id', 'family', 'simple_family', 'head_of_family',
                    'ancestors',  'descendants', 'leaf_descendants',
                    'simple_leaf_descendants', 'top_level', 'moveable', 'pushable' ]
    
    #a dict mapping attribute names to overridden default attribute values. note
    #that at class construction, TypeFamily replaces this with an aggregated
    #dict of all overridden default attribute values defined by the Object class
    #or its superclasses.
    _defaults = {}  
    
    #the object's class type (to distinguish simple from compound objects)
    _class_type = 'SIMPLE'
    
    #the Object's id in the world (assigned by the world during creation). this
    #is the index of the row in the master data array that stores attribute
    #values for the Object.
    _id = None
    
    #if the Object is part of a CompoundObject, the parent Object is stored here
    _parent = None

    #if the object has children, they will be stored here. This should be empty
    #for non-compound objects
    _children = None

    #for efficiency we cache various family-related properties of the object
    _descendants_cached = None
    _descendants_ids_cached = None
    _leaf_descendants_cached = None
    _leaf_descendants_ids_cached = None
    _simple_leaf_descendants_cached = None
    _simple_leaf_descendants_ids_cached = None
    _ancestors_cached = None
    _ancestors_ids_cached = None
    _head_of_family_cached = None
    _head_of_family_id_cached = None
    _effective_head_of_family_cached = None
    
    #new attributes to include in States
    _state_attributes = ['_name', '_attributes', '_id', '_parent']
    
    def __init__(self, world, name=None, prepare=True, initialize=True, **kwargs):
        """
        Parameters
        ----------
        world : PixelWorld
            the world that the object will live in
        name : string, optional
            use to override the Object's default name
        prepare : bool, optional
            see Entity
        initialize : bool, optional
            True to set initial values for the Object's attributes. if this is
            False, _initialize() should be called manually at some later point.
        **kwargs
            use to specify initial values for the object's attributes. values
            can also be functions that take the Object as an input argument, for
            attribute values that should be computed lazily at the time of
            Object creation (e.g. if the value depends on the size of the
            world).
        """
        super(Object, self).__init__(world, name=name, prepare=False)
        
        self._children = []
        self._effective_head_of_family_cached = dict()

        #add our data
        self._add_data()
        
        #add the attributes
        self._add_attributes()
        
        #set the initial attribute values
        if initialize:
            self._initialize(kwargs, prepare)
    
    @property
    def specifications(self):
        spec = super(Object, self).specifications
        
        for name,attr in self._attributes.iteritems():
            if not attr._read_only and name not in ['id']:
                spec[1][name] = getattr(self, name)
        
        return spec

    @property
    def unique_name(self):
        """a unique name to identify the Object (since Object's can share the
        same value of the _name attribute). this used mainly for debugging
        messages."""
        parent_name = '' if self._parent is None else self._parent.unique_name + '/'
        idx = self._id if self._id is not None else -1
        
        return '%s%s(%d)' % (parent_name, self._name, idx)
    
    @property
    def attributes(self):
        """public access to the Object's attributes dict"""
        return self._attributes
    
    @property
    def state(self):
        """the current state of the Object, as a dict of observed attribute
        values"""
        if self._world._observed_attributes is None:
            return {name:self._get(name) for name in self._attributes}
        else:
            return {name:self._get(name) for name in self._attributes \
                    if name in self._world._observed_attributes}
    
    @property
    def parent_object(self):
        """the Object's parent Object, if it has one"""
        return self._parent

    @property
    def has_parent(self):
        """True if the object has children"""
        return (self._parent is not None)

    @property
    def has_children(self):
        """True if the object has children"""
        return len(self._children) > 0
    
    @property
    def _head_of_family(self):
        """provides the object's head of family as an object"""
        if self._head_of_family_cached is not None:
            return self._head_of_family_cached

        obj = self
        while obj._parent is not None:
            obj = obj._parent

        self._head_of_family_cached = obj
        return obj

    @property
    def _head_of_family_id(self):
        """provides the object's head of family's id"""
        if self._head_of_family_id_cached is not None:
            return self._head_of_family_id_cached

        obj = self._head_of_family

        self._head_of_family_id_cached = obj._id
        return obj._id


    @property
    def _ancestors(self):
        """provides a list of the ancestors of the object, not including the object,
        ordered by increasing remove from the object"""
        if self._ancestors_cached is not None:
            return self._ancestors_cached

        if self.has_parent:
            ancestors = [self._parent] + self._parent._ancestors
        else:
            ancestors = []

        self._ancestors_cached = ancestors
        return ancestors

    @property
    def _ancestors_ids(self):
        """provides an array of the ids of the ancestors of the object, not including
        the object, ordered by increasing remove from the object"""
        if self._ancestors_ids_cached is not None:
            return self._ancestors_ids_cached

        rv = np.array([x._id for x in self._ancestors])

        self._ancestors_ids_cached = rv
        return rv

    @property
    def _descendants(self):
        """provides a list of an object's descendants"""
        if self._descendants_cached is not None:
            return self._descendants_cached

        rv = [self] + sum([child._descendants for child in self._children], [])

        self._descendants_cached = rv
        return rv

    @property
    def _descendants_ids(self):
        """provides an array of an object's descendants' indices"""
        if self._descendants_ids_cached is not None:
            return self._descendants_ids_cached

        rv = np.array([x._id for x in self._descendants])
        self._descendants_ids_cached = rv
        return rv


    @property
    def _leaf_descendants(self):
        """provides a list of a parent's leaf descendants"""
        if self._leaf_descendants_cached is not None:
            return self._leaf_descendants_cached

        if not self.has_children:
            rv = [self]
        else:
            rv = sum([child._leaf_descendants for child in self._children], [])

        self._leaf_descendants_cached = rv
        return rv

    @property
    def _leaf_descendants_ids(self):
        """provides an array of an object's leaf descendants' indices"""
        if self._leaf_descendants_ids_cached is not None:
            return self._leaf_descendants_ids_cached

        rv = np.array([x._id for x in self._leaf_descendants])
        self._leaf_descendants_ids_cached = rv
        return rv

    @property
    def _simple_leaf_descendants(self):
        """provides a list of a parent's leaf descendants, but only those descendants
        which are of SIMPLE type"""
        if self._simple_leaf_descendants_cached is not None:
            return self._simple_leaf_descendants_cached

        descendants = self._leaf_descendants
        rv = [x for x in descendants if x._class_type == self.SIMPLE]

        self._simple_leaf_descendants_cached = rv
        return rv

    @property
    def _simple_leaf_descendants_ids(self):
        """provides an array of a parent's leaf descendants' indices, but only those
        descendants which are of SIMPLE type"""
        if self._simple_leaf_descendants_ids_cached is not None:
            return self._simple_leaf_descendants_ids_cached

        rv = np.array([x._id for x in self._simple_leaf_descendants])
        self._simple_leaf_descendants_ids_cached = rv
        return rv

    @property
    def _family(self):
        """provides a list of all objects in an Object's family"""
        ancestor = self._head_of_family
        family = ancestor._descendants
        return family

    @property
    def _family_ids(self):
        """provides an array of the indices of all objects in an Object's family"""
        ancestor = self._head_of_family
        family = ancestor._descendants_ids
        return family

    @property
    def _simple_family(self):
        """provides a list of all simple objects in an Object's family"""
        ancestor = self._head_of_family
        return ancestor._simple_leaf_descendants

    @property
    def _simple_family_ids(self):
        """provides an array of the indices of all simple objects in an Object's family"""
        ancestor = self._head_of_family
        return ancestor._simple_leaf_descendants_ids

    def is_related(self, other):
        """determine whether two Objects are part of the same family
        
        Parameters
        ----------
        other : Object
            the Object to test for a relationship
        
        Returns
        -------
        related : bool
            True if the Objects are part of the same family
        """
        return self._head_of_family is other._head_of_family
    
    def update(self, **kwargs):
        """update a set of the Object's attribute values. this mimics the
        behavior of dict.update().
        
        Parameters
        ----------
        **kwargs
            keyword arguments corresponding to ObjectAttribute names and their
            new values
        """
        for name,value in kwargs.iteritems():
            self._set(name, value)
    
    def remove(self):
        """remove the Object from the world. this destroys all record of the
        Object. it could also potentially change the structure of
        attribute-based observations, so beware."""
        #remove from the parent
        self._remove_parent()
        
        #remove the ObjectAttributes
        self._remove_attributes()
        
        #remove our data
        self._remove_data()
        
        super(Object, self).remove()
    
    def add_attribute(self, name, value=None, initialize=True, validate=True):
        """add an ObjectAttribute to the Object
        
        Parameters
        ----------
        name : string
            the name of the ObjectAttribute, which must have been imported
            already
        value : any, optional
            the initial value of the attribute. if unspecified, uses the default
            value
        initialize : bool, optional
            True to initialize the attribute value. the value will be
            initialized regardless of this value if the value keyword argument
            is specified.
        validate : bool, optional
            True to validate the value as conforming to the ObjectAttribute's
            requirements. only set this to False if the value is definitely a
            valid attribute value.
        """
        #get the ObjectAttribute object
        attr = self._world.get_object_attribute(name)
        self._attributes[name] = attr

        #clear outdated caches
        self._clear_descendants_effective_hof_cache()
        
        #initialize the attribute value
        if initialize or value is not None:
            data = {} if value is None else {name: value}
            self._initialize_attribute(attr, data, validate=validate)
    
    def remove_attribute(self, name):
        """remove an ObjectAttribute from the Object. it is okay to try to
        remove attributes that the Object doesn't have.
        
        Parameters
        ----------
        name : string
            the name of the ObjectAttribute
        """
        if name in self._attributes:  # only try to remove if we have the attribute
            attr = self._attributes[name]
            
            #mask out the Object's entry in the data array
            if attr._has_own_data:
                self._world.data[name].mask[self._id] = True
            
            del self._attributes[name]

    def has_attribute_class(self, cls):
        cls = ObjectAttribute.get_class(cls)
        for attr in self._attributes:
            if isinstance(self._attributes[attr], cls):
                return True
        return False
    
    def __dir__(self):
        """add the Object's ObjectAttributes to the dir() list"""
        d = set(super(Object, self).__dir__())
        d.update(self._attributes.keys())
        return sorted(d)
    
    def __getattr__(self, name):
        """overriding this method allows the Object's ObjectAttributes to be
        retrieved like a normal python attribute (e.g. p = obj.position). i
        believe this method only gets called if python couldn't resolve name as
        an attribute.
        """
        try:  # does name refer to an ObjectAttribute?
            return self._get(name)
        except KeyError:
            if name in self._world._object_attributes:
                #name is an ObjectAttribute, but not one that the Object has
                raise AttributeError('%s object does not have the attribute "%s"' % (self.unique_name, name))
            else:
                #i think an exception will always be raised at this point, but
                #hopefully this is just the result of something like a hasattr()
                #call. just send the request on to __getattribute__ so an
                #AttributeError is raised.
                return self.__getattribute__(name)
        except TypeError as err:
            #we probably got to this point because _attributes is still a list,
            #meaning _add_attributes() hasn't converted it into a dict yet, but
            #_get() is trying to access it as if it was a dict. this happens
            #e.g. if hasattr() is called during __init__(), before
            #_add_attributes().
            if isinstance(self._attributes, list):
                return self.__getattribute__(name)
            else:
                raise err
    
    def __setattr__(self, name, value):
        """overriding this method allows the Object's ObjectAttributes to be
        assigned like a normal python attribute (e.g. obj.position = (2, 2)).
        
        this is currently set up to be fairly restrictive: if the name of the
        attribute being assigned doesn't begin with an underscore, it raises an
        exception unless the attribute is one of the Object's ObjectAttributes.
        this prevents otherwise possiblly confusing situations, e.g. if the
        programmer thinks an Object has the 'position' ObjectAttribute when it
        actually doesn't, and then can't figure out why setting the Object's
        position doesn't lead to the usual CoupledFamilyObjectAttribute
        behaviors.
        """
        if name.startswith('_'):
            super(Object, self).__setattr__(name, value)
        elif name in self._attributes:
            self._set(name, value)
        elif hasattr(self, name):
            super(Object, self).__setattr__(name, value)
        else:
            raise AttributeError('%s object does not have the attribute "%s"' % (self.unique_name, name))

    def __repr__(self):
        return self.unique_name

    def _post_restore(self):
        """Clear the family caches after restoring from State."""
        self._clear_family_caches()
        
    def _clear_family_caches(self):
        """Clear the caches of family-related python attributes"""
        self._descendants_cached = None
        self._leaf_descendants_cached = None
        self._simple_leaf_descendants_cached = None
        self._ancestors_cached = None
        self._head_of_family_cached = None
        self._effective_head_of_family_cached.clear()
        self._clear_family_id_caches()

    def _clear_family_id_caches(self):
        """Clear the caches of family-related python attributes, but only those that
        return ids"""
        self._descendants_ids_cached = None
        self._leaf_descendants_ids_cached = None
        self._simple_leaf_descendants_ids_cached = None
        self._ancestors_ids_cached = None
        self._head_of_family_id_cached = None

    def _clear_ancestors_descendants_caches(self):
        """Clear the descendant-related caches of this object and all its ancestors.
        """
        obj = self
        while obj is not None:
            obj._descendants_cached = None
            obj._descendants_ids_cached = None
            obj._leaf_descendants_cached = None
            obj._leaf_descendants_ids_cached = None
            obj._simple_leaf_descendants_cached = None
            obj._simple_leaf_descendants_ids_cached = None
            obj = obj._parent

    def _clear_descendants_ancestors_caches(self):
        """Clear the ancestor-related caches of this object and all its descendants.
        """
        for obj in self._descendants:
            obj._head_of_family_cached = None
            obj._head_of_family_id_cached = None
            obj._ancestors_cached = None
            obj._ancestors_ids_cached = None
            obj._effective_head_of_family_cached.clear()

    def _clear_descendants_effective_hof_cache(self):
        """Clear the effective head of family caches of this object and all its descendants.
        """
        for obj in self._descendants:
            obj._effective_head_of_family_cached.clear()
    
    def _add_data(self):
        """add a row to the master data array for the Object's attribute data"""
        #do this here so _set_view already has the id when add_row is called
        self._id = self.world._get_new_object_id()
        
        if self._id >= len(self.world._data):  # add a row in the master data array for the object
            data = {
                'object': self,
                'object_type': self._class_type,
                }
            self.world._data.add_row(data=data)
        else:  # somebody must have already added a row for us (how nice)
            self.world._data[self._id]['object'] = self
            self.world._data[self._id]['object_type'] = self._class_type
        
    def _remove_data(self):
        """remove the Object's data row in the master data array"""
        #make sure we are who we think we are
        if self._world.num_objects > self._id and self._world.objects[self._id] is self:
            #remove the row
            self._world._data.remove_row(self._id)
            self._world._num_objects -= 1
        
        #update Object indices
        self._world._update_object_ids()
    
    def _add_attributes(self):
        """called during __init__. add all of the Object's ObjectAttributes.
        this is where _attributes is converted from a list to a dict."""
        attribute_names = self._attributes
        
        self._attributes = CustomOrderedDict()
        
        #add each attribute
        for name in attribute_names:
            self.add_attribute(name, initialize=False)
    
    def _remove_attributes(self):
        """remove all of the Object's ObjectAttributes. called in remove()."""
        for name in copy(self._attributes):
            self.remove_attribute(name)
    
    def _remove_parent(self):
        """remove the Object's parent, if it has one"""
        if self._parent is not None:
            self._parent.remove_child(self)
    
    def _initialize(self, data, prepare):
        """initialize the Object's presence in the world
        
        Parameters
        ----------
        data : dict
            a dict of initial attribute values. "values" can also be functions
            that take the world as an input argument and return the actual
            value, in which case the values is determined now.
        prepare : bool
            True to call the Object's prepare() method after setting initial
            attribute values
        """
        #initialize the attributes
        for attr in self._attributes.values():
            self._initialize_attribute(attr, data)
        
        #try to add leftover attributes
        for name,value in data.iteritems():
            self.add_attribute(name, value)
        
        #call the Object's prepare() method (see Entity.prepare())
        if prepare and self.world.populated:
            self.prepare()
    
    def _initialize_attribute(self, attr, data, validate=True):
        """initialize a single attribute
        
        Parameters
        ----------
        attr : ObjectAttribute
            the ObjectAttribute whose value should be initialized
        data : dict
            a dict of initialization data (see _initialize())
        validate : bool, optional
            True to validate the value
        """
        #only set initial values of writeable attributes
        if not attr._read_only:
            #get the initial value
            value = self._get_init_value(data, attr)
            
            #set it
            self._set(attr._name, value, validate=validate)
            
            #initialize the attribute family
            if isinstance(attr, CoupledFamilyObjectAttribute) and \
            self._parent is not None:
                attr._initialize_family(self._parent._id, change='child')
    
    def _get_init_value(self, data, attr, pop=True, evaluate=True):
        """get an initial attribute value from the initial values dict, or
        determine what the default initial value should be if none was specified
        
        Parameters
        ----------
        data : dict
            the dict of initial attribute values
        attr : ObjectAttribute
            the ObjectAttribute whose inital value should be computed
        pop : bool, optional
            True to remove the value from the initial values dict
        evaluate : bool, optional
            True to evaluate lazy values
        
        Returns
        -------
        value : attr.dtype
            the Object's initial attribute value for attr
        """
        if attr.name in data:  # default value is overridden
            if pop:
                value = data.pop(attr.name)
            else:
                value = data.get(attr.name)
        else:
            value = None
        
        if value is None:
            if attr.name in self._defaults:  # Object override attr's default value
                value = self._defaults[attr.name]
            else:  # just use attr's default value
                value = attr._default_value
        
        #resolve lazy attribute values
        if callable(value) and evaluate:
            value = value(self)
        
        return value
    
    def _get(self, name):
        """get the value of one of the Object's ObjectAttributes. this should
        primarily only be used by __getattr__.
        
        Parameters
        ----------
        name : string
            the name of the attribute. should be the name of an existing
            ObjectAttribute in _attributes
        
        Returns
        -------
        value : any
            the attribute value
        """
        return self._attributes[name].get(self)
    
    def _set(self, name, value, validate=True):
        """set the value of one of the Object's ObjectAttributes. this should
        primarily only be used by __setattr__.
        
        Parameters
        ----------
        name : string
            the name of the attribute. should be the name of an existing
            ObjectAttribute in _attributes
        value : any
            the new attribute value
        validate : bool, optional
            True to validate the value as conforming to the ObjectAttribute's
            requirements. only set this to False if the value is definitely a
            valid attribute value.
        """
        self._attributes[name].set(self, value, validate=validate)


class CompoundObject(Object):
    """CompoundObjects gather together a set of simple Object's into a family.
    this could be used e.g. to create Objects that involve multiple pixels.
    
    properties of CompoundObjects:
        -   they have their own attribute values, subject to the coupling rules
            of attributes that subclass off of CoupledFamilyObjectAttribute.
        -   they cannot conflict with other Objects under the rules of
            ConflictObjectAttribute (unless a ConflictObjectAttribute subclass
            overrides that behavior). e.g., the position of a CompoundObject
            that is made up of a set of simple Objects configured in a square
            will be the center of that square, but the CompoundObject won't
            "collide" with a simple Object that is actually at the center of the
            square.
        -   a CompoundObject gives each of its children an 'parent' attribute
            that references itself. a CompoundObject also has a 'children'
            attribute that references itself.
    
    usually, a CompoundObject subclass will want to additionally subclass off of
    the class of the simple Objects it takes as its children, so that it has the
    same attributes as its children.
    
    see FrameObject in objects.py for an example.
    """
    _attributes = ['children']
    
    #define a new class type to distinguish CompoundObjects from simple Objects
    _class_type = 'COMPOUND'
    
    #this will be a list of the CompoundObject's children
    _children = None
    
    #the default Object type to use for children
    _child_type = 'object'
    
    #a dict of default parameters to use when constructing children
    _child_params = {}
    
    #new attributes to include in States
    _state_attributes = ['_children']
    
    def __init__(self, world, name=None, children=None, child_type=None,
                    child_params=None, prepare=True, initialize=True, **kwargs):
        """
        Parameters
        ----------
        world : PixelWorld
            the host world
        name : string, optional
            use to override the Object's default name
        children : list[string | list], optional
            if the CompoundObject should be initialized with a set of children,
            define them here. this argument follows the same specifications as
            PixelWorld.create_objects(), i.e. it can describe either a set of
            existing Objects or specifications for creating new Objects.
        child_type : string | list_like, optional
            the name of the Object class to use for the child Objects, or a list
            of names, one for each child (overrides _child_type)
        child_params : dict | list[dict], optional
            a dict of parameters to pass when constructing the child Objects,
            or a list of dicts, one for each child (overrides _child_params)
        prepare : bool, optional
            see Entity
        initialize : bool, optional
            see Object
        **kwargs
            initial values for the CompoundObject's attributes. note that
            CoupledFamilyObjectAttribute values are only initialized if
            explicitly assigned here or if defined in _defaults. otherwise,
            the initial value will be determined from the CompoundObject's
            children.
        """
        #create the object, skip Object initialization for now
        super(CompoundObject, self).__init__(world, name=name, initialize=False,
                                                prepare=False, **kwargs)
        
        #create the children
        self._create_children(children, child_type, child_params, kwargs)
        
        #now initialize the object
        if initialize:
            self._initialize(kwargs, prepare)
    
    @property
    def specifications(self):
        spec = super(CompoundObject, self).specifications
        
        spec[1]['children'] = [obj.specifications for obj in self._children]
        del spec[1]['child_type']
        del spec[1]['child_params']
        del spec[1]['initialize']
        
        return spec
    
    @property
    def child_states(self):
        """a list of each child Object's state"""
        return [obj.state for obj in self._children]
    
    def remove(self, remove_children=False):
        """remove the CompoundObject from the world. additionally breaks up the
        family or deletes the CompoundObject's children.
        
        Parameters
        ----------
        remove_children : bool, optional
            True to also remove the CompoundObject's children
        """
        #remove the children
        if remove_children:
            for obj in copy(self._children):
                if isinstance(obj, CompoundObject):
                    obj.remove(remove_children=True)
                else:
                    obj.remove()
        else:
            for obj in copy(self._children):
                self.remove_child(obj)
        
        super(CompoundObject, self).remove()
    
    def add_child(self, child, initialize=True):
        """add a child to the family
        
        Parameters
        ----------
        child : Object
            a simple Object to add to the family
        initialize : bool, optional
            True to initiate the Object into the family, i.e. to recalculate
            CoupledFamilyObjectAttribute values.  this should always be True
            unless _initialize_family() will be called later for each
            CoupledFamilyObjectAttribute.
        """
        # get rid of cached family stuff in ancestors
        self._clear_ancestors_descendants_caches()

        # get rid of cached ancestor stuff in child's descendants
        child._clear_descendants_ancestors_caches()
        
        #remove the old parent
        child._remove_parent()
        
        #adopt the child
        self._children.append(child)
        child._parent = self
        child.add_attribute('parent', initialize=False)
        
        #reinitialize shared attributes
        if initialize:
            for name,attr in self._attributes.iteritems():
                if isinstance(attr, CoupledFamilyObjectAttribute):
                    attr._initialize_family(self._id, change='child')
    
    def remove_child(self, child):
        """remove a child from the family, if it is currently a family member
        
        Parameters
        ----------
        child : Object
            the child Object to remove
        """
        # get rid of cached family stuff in ancestors
        self._clear_ancestors_descendants_caches()

        # get rid of cached family stuff in child's descendants
        child._clear_descendants_ancestors_caches()

        if child in self._children:
            #remove the link to the child
            child.remove_attribute('parent')
            child._parent = None
            self._children.remove(child)
            
            #reinitialize shared attributes
            for name,attr in self._attributes.iteritems():
                if isinstance(attr, CoupledFamilyObjectAttribute):
                    attr._initialize_family(self._id, change='child')
    
    def __len__(self):
        """len(CompoundObject) := number of children"""
        return len(self._children)
    
    def _create_children(self, children, child_type, child_params, params):
        """create the CompoundObject's set of children. NOTE that
        _initialize_family is not called for the family's ObjectAttributes here,
        so must be done manually somewhere else.
        
        Parameters
        ----------
        children : list[string | list] | None
            see __init__()
        child_type : string | list[string] | None
            see __init__()
        child_params : dict | list[dict] | None
            see __init__()
        params : dict
            extra keyword arguments that might be needed by
            _construct_children_params()
        """
        children = self._construct_children_params(children, child_type,
                    child_params, params)
        
        if len(children) > 0:
            #create the children
            children = self._world.create_objects(children)
            
            #add the children to the family
            for child in children:
                self.add_child(child, initialize=False)
    
    def _construct_children_params(self, children, child_type, child_params,
                                    params, num_children=None):
        """construct the parameter list that defines the CompoundObject's child
        Objects
        
        Parameters
        ----------
        children : list[string | list] | None
            see __init__()
        child_type : string | list[string] | None
            see __init__()
        child_params : dict | list[dict] | None
            see __init__()
        params : dict
            a mix of explicitly specified parent attribute values and extra
            keyword arguments that might be used by subclasses
        num_children : Number, optional
            the number of children that should be created
        
        Returns
        ----------
        children : list[list[string, dict]]
            the list of child Object specifications (see world.create_object())
        """
        #make sure children is a list
        if children is None:
            children = []
        else:
            assert isinstance(children, list), 'children must be a list'
            children = copy(children)
        
        #make sure child_type is a list of strings
        if child_type is None:
            child_type = []
        else:
            child_type = copy(to_iterable(child_type))
            
        #make sure child_params is a list of dicts
        if child_params is None:
            child_params = []
        elif not isinstance(child_params, list):
            child_params = [child_params]
        else:
            child_params = copy(child_params)
        
        #get the number of children to construct
        if num_children is None:
            num_children = max(len(children), len(child_type), len(child_params))
        
        #make sure we have the correct number of children
        num_children_passed = len(children)
        if num_children_passed < num_children:
            num_needed = num_children - num_children_passed
            children.extend([ [None, {}] for _ in xrange(num_needed)])
        else:
            assert num_children_passed == num_children, 'expected %d children, got %d' % (num_children, num_children_passed)
        
        #make sure we have the correct number of child types
        num_child_type_passed = len(child_type)
        if num_child_type_passed < num_children:
            num_needed = num_children - num_child_type_passed
            child_type.extend([self._child_type for _ in xrange(num_needed)])
        else:
            assert num_child_type_passed == num_children, 'expected %d child_types, got %d' % (num_children, num_child_type_passed)
        
        #make sure we have the correct number of child params
        num_child_params_passed = len(child_params)
        if num_child_params_passed < num_children:
            num_needed = num_children - num_child_params_passed
            child_params.extend([self._child_params for _ in xrange(num_needed)])
        else:
            assert num_child_params_passed == num_children, 'expected %d child_params, got %d' % (num_children, num_child_params_passed)
        
        #fill in default values
        for idx,child in enumerate(children):
            #child should be a list or just a child type
            if isinstance(child, list):
                child = copy(child)
            else:
                child = [child]
            
            #make sure we have a parameter dict
            if len(child) < 2:
                child.append({})
            
            #make sure we have the proper length
            assert len(child) == 2, 'improper child length'
            
            #make sure we have a child type
            if child[0] is None:
                child[0] = child_type[idx]
            
            #add the non-overridden default parameters
            child[1] = dict(child_params[idx], **child[1])
            
            #fill in default values for CoupledFamilyObjectAttributes whose
            #parent value was specified
            unspecified_params = set(params.keys()) - set(child[1].keys())
            for unspecified_param in unspecified_params:
                try:
                    attr = self._world.get_object_attribute(unspecified_param)
                    if isinstance(attr, CoupledFamilyObjectAttribute):
                        child[1][attr._name] = attr._calculate_child_value_given_parent(self._id, params[unspecified_param])
                except ValueError:  # probably not an ObjectAttribute
                    pass
            
            children[idx] = child
        
        return children
    
    def _initialize_attribute(self, attr, data, validate=True):
        """initialize a single attribute
        
        Parameters
        ----------
        attr : ObjectAttribute
            the ObjectAttribute whose value should be initialized
        data : dict
            a dict of initialization data (see _initialize())
        validate : bool, optional
            True to validate the initial value
        """
        #only set initial values of writeable attributes
        if not attr._read_only:
            if isinstance(attr, CoupledFamilyObjectAttribute):
                value_specified = attr._name in data or attr._name in self._defaults
                set_parent_uncoupled = attr._uncoupled_at_initialization and value_specified

                #initialize the attribute family
                value = data[attr._name] if set_parent_uncoupled else None
                attr._initialize_family(self._id, change='parent', parent_value=value)
                
                #set a non-derived value only if an initial value was
                #specified, or an initial value was defined in _defaults, or
                #the CompoundObject has no children
                if value_specified or len(attr._get_child_indices(self._id)) == 0:
                    super(CompoundObject, self)._initialize_attribute(attr, data, validate=validate)
            else:  # unshared attribute; set the usual initial value
                super(CompoundObject, self)._initialize_attribute(attr, data, validate=validate)


class ObjectCollection(StateObject, np.ndarray):
    """PixelWorld uses this class to provide access to the world's Objects. it
    behaves like a numpy array, with some additional functionality for finding
    Objects based on attribute values.
    
    example usage:
        world.objects
        world.objects[0]
        world.objects(name='wall', position=(0, 0))
        world.objects.get(name='self')
        world.objects['self']
    """
    
    @property
    def specifications(self):
        """return a specifications list for the set of Objects in the
        collection. this could then e.g. be passed into world.create_objects.
        """
        return [obj.specifications for obj in self if obj._parent is None]
    
    @property
    def state(self):
        """return a list of states of the Objects in the collection"""
        return [obj.state for obj in self]
    
    def __call__(self, *args, **kwargs):
        """allows objects(...) usage to find objects
        
        Parameters
        ----------
        *args
            if one unnamed argument is passed, it is treated like a name
        **kwargs
            see find()
        
        Returns
        -------
        object : Object | ndarray
            if one Object matched, then that Object. otherwise, an array of
            matching Objects.
        """
        if len(args) > 0:
            assert len(args) == 1, 'at most one unnamed argument can be passed'
            
            kwargs['name'] = args[0]
        
        objects = self.find(**kwargs)
        
        return objects[0] if objects.size == 1 else objects
    
    def __getitem__(self, key):
        if isinstance(key, basestring):
            return self(key)
        else:
            return super(ObjectCollection, self).__getitem__(key)
    
    def get(self, **kwargs):
        """get the first Object that meets the specified criteria
        
        Parameters
        ----------
        **kwargs
            ObjectAttribute values to search for. an Object must have the
            attribute and its value must compare() True with the search value in
            order to match.
        
        Returns
        -------
        object : Object | None
            the first Object to meet the specified criteria, or None if no
            Object matches
        """
        objects = self.find(max_objects=1, **kwargs)
        
        if objects.size == 0:
            return None
        else:
            return objects[0]
    
    def find(self, max_objects=None, object_type=None, **kwargs):
        """find Objects that meet the specified criteria
        
        Parameters
        ----------
        max_objects : int, optional
            the maximum number of Objects to return. no limit if unspecified.
        object_type : string, optional
            only return Objects of the specified type
        **kwargs
            ObjectAttribute values to search for. an Object must have the
            attribute and its value must compare() True with the search value in
            order to match.
        
        Returns
        -------
        objects : ndarray
            a numpy array of Objects that met all of the specified criteria
        """
        num_objects = len(self)
        if num_objects == 0:
            return np.array([], dtype=Object)
        
        world = self[0]._world
        
        #start with all objects
        indices = np.arange(len(self))
        
        #keep only Objects of the specified type
        if object_type is not None and len(indices) > 0:
            object_type = getattr(world.objects[0], object_type)
            match = np.where(world.data['object_type'][indices] == object_type)[0]
            indices = indices[match]
        
        #find objects that have all of the search attributes
        for name in kwargs:
            attr = world._object_attributes.get(name, None)
            
            if attr is None:
                match = np.zeros(indices.shape, dtype=bool)
            else:
                match = ~attr._get_mask(indices)
            
            indices = indices[match]
            
            if indices.size == 0:
                return self[indices]
        
        #find objects that match all of the search criteria
        for name,value in kwargs.iteritems():
            attr = world._object_attributes[name]
            
            value = attr.coerce_value(value)
            
            values = attr.get(indices)
            
            match = attr.compare(values, value, validate=False)
            
            if indices.size == 0:
                return self[indices]
            
            indices = indices[match]
        
        #keep only the number of matches requested
        if max_objects is not None:
            indices = indices[:max_objects]
        
        #return the matching objects
        return self[indices]


#---Event----------------------------------------------------------------------#
class Event(Entity):
    """an Event records things that happen during the simulation. each Event is
    kept in a history of Events in PixelWorld._events, and a record of all
    Events that have occurred since the start of the last step is kept in
    PixelWorld._step_events. Events are used by a Judge to determine reward and
    to terminate the simulation.
    
    Events should attempt to require as little computing expense as possible.
    they should only record things that aren't easily determinable from
    inspecting the state of the world.
    
    as an example, PositionObjectAttribute adds a CollisionEvent each time a
    collision occurs between two Objects.
    
    an Event object is identified primarily just by being an instance of its
    particular Event class. beyond this, an Event class may want to define
    #two things:
        1) a list of parameters that the Event needs (see _parameters). this
          just stores extra data along with the Event.
        2) _get_description() which constructs a human-readable description of
          the event.
    
    see events.py for more example Events.
    """
    _class_family = 'event'
    
    #the event's name. if subclasses do not assign this explicitly, the name
    #is derived sensibly from the class name, e.g. CollisionEvent becomes
    #'collision' and LeaveScreenEvent becomes 'leave_screen'.
    _name = 'event'
    
    #this causes each Event class to have a _parameters attribute, which is a
    #list that is assembled based on the Event's superclasses (see TypeFamily
    #and _parameters below)
    _class_tree_attributes = ['_parameters']
    
    #an Event subclass should fill this with parameters that need to be passed
    #as keyword arguments when the Event is created. these parameters identify
    #particulars of the Event, such as which Objects were involved, or whether
    #an action was successful. note that at class construction, TypeFamily
    #replaces this with an aggregated list of all parameters required by the
    #Event class and its superclasses, and at instantiation, this is replaced
    #with a dict of parameter values.
    _parameters = []
    
    #if an Event should be associated with an intrinsic reward, define it here.
    #the default Judge incorporates intrinsic rewards, and other Judge's may or
    #may not. intrinsic rewards should probably only be used in special cases,
    #with rewards for the most part being defined via a Judge's _reward_events
    #attribute.
    _reward = 0
    
    #if an Event should intrinsically terminate the episode, specify so here.
    #the default Judge follows this while making termination judgments, and
    #other Judge's may or may not. intrinsic termination conditions should
    #probably only be used in special cases, with termination for the most part
    #being defined via a Judge's _termination_events attribute.
    _terminates = False
    
    #at object creation, a human-readable description of the event will be
    #assigned here
    _description = None
    
    #new attributes to include in States
    _state_attributes = ['_parameters', '_description']
    
    def __init__(self, world, name=None, prepare=True, **kwargs):
        """
        Parameters
        ----------
        world : PixelWorld
            the host world
        name : string, optional
            use to override the default Event name. this probably doesn't make
            much sense to do, but is required as part of being an Entity.
        prepare : bool, optional
            see Entity
        **kwargs
            the values of the required parameters for the Event
        """
        super(Event, self).__init__(world, name=name, prepare=False)
        
        #set the event parameters
        params = self._parameters
        self._parameters = {}
        for param in params:
            if param not in kwargs:
                raise TypeError('%s events require a %s parameter' % (self._name, param))
            
            self._parameters[param] = kwargs[param]
        
        #save the description. we do this now because e.g. Object ids may refer
        #to different Objects in the future.
        self._description = self._get_description()
        
        if prepare and self.world.populated:
            self.prepare()
    
    @property
    def specifications(self):
        spec = super(Event, self).specifications
        
        spec[1].update(**self._parameters)
        
        return spec
    
    @property
    def description(self):
        """a human-readable description of the event"""
        return self._description
    
    @property
    def reward(self):
        """the reward to (potentially) associated with the event"""
        return self._reward
    
    @property
    def terminates(self):
        """True if the reward intrinsically terminates the episode"""
        return self._terminates
    
    def __repr__(self):
        return self.description
    
    def __dir__(self):
        """add the Event's parameters to the dir() list"""
        d = set(super(Event, self).__dir__())
        d.update(self._parameters.keys())
        return sorted(d)
    
    def __getattr__(self, name):
        """this makes event parameters accessible as object attributes"""
        try:
            return self._parameters[name]
        except KeyError:
            return self.__getattribute__(name)
    
    def __getstate__(self):
        return self._parameters

    def __setstate__(self, d):
        self._parameters = d

    def __deepcopy__(self, memo):
        """Deep-copy the event. This is necessary because we define __getstate__ and
        __setstate__, which deepcopy attempts to use.

        Parameters
        ----------
        memo : dict
            Memoization dict
        """
        # create a new entity of the right class
        rv = self.__class__.__new__(self.__class__)
        memo[id(self)] = rv        

        # copy all of the attributes
        for attr, value in self.__dict__.iteritems():
            value_id = id(value)
            new_value = memo[value_id] if value_id in memo else deepcopy(value, memo)
            rv.__dict__[attr] = new_value

        assert rv._world is not None

        return rv
    
    def _add_world(self, world):
        """see Entity._add_world()"""
        super(Event, self)._add_world(world)
        
        world._events.append(self)
        world._step_events.append(self)
    
    def _remove_world(self):
        """see Entity._remove_world()"""
        try:
            self._world._events.remove(self)
        except ValueError:  #must have already been removed
            pass
        
        #if the event was from a previous step, it won't be in here
        try:
            self._world._step_events.remove(self)
        except ValueError:
            pass
        
        super(Event, self)._remove_world()
    
    def _get_description(self):
        """subclasses will probably want to override this to generate a more
        informative description of the event"""
        return self._name


#---Goal-----------------------------------------------------------------------#
class Goal(Entity):
    """a Goal evaluates whether it has been met given the current state of the
    world, and optionally also implements a method for achieving the goal from
    any world state. Goals can also be active or inactive, indicate that they
    should contribute to terminating the episode, and/or associate rewards with
    their achievement. Goals are used by the base Judge when determining rewards
    and termination conditions.
    
    subclasses should define _is_achieved() and optionally _achieve().
    """
    _class_family = 'goal'
    
    #default reward to associate with achieving the goal
    _reward = 1000
    
    #True if the goal should contribute to terminating the episode
    _terminates = True
    
    #True if the goal is active by default
    _active = True
    
    #True if the goal wants to be the only one that is active
    _exclusive = False
    
    def __init__(self, world, terminates=None, active=None, exclusive=None,
        **kwargs):
        """
        Parameters
        ----------
        world : PixelWorld
            the host PixelWorld
        terminates : bool
            override the default terminates attribute value
        active : bool
            override the default active attribute value
        exclusive : bool
            override the default exclusive attribute value
        """
        super(Goal, self).__init__(world, **kwargs)
        
        if terminates is not None: self.terminates = terminates
        if active is not None: self.active = active
        if exclusive is not None: self.exclusive = exclusive
    
    @property
    def achieved(self):
        """has the goal been achieved?"""
        return self.is_achieved()
    
    @property
    def reward(self):
        """reward associated with achieving the goal"""
        return self._reward
    
    @property
    def terminates(self):
        """True if the goal contributes toward terminating the episode"""
        return self._terminates
    
    @terminates.setter
    def terminates(self, terminates):
        self._terminates = terminates
    
    @property
    def active(self):
        """True if the goal is currently active"""
        return self._active
    
    @active.setter
    def active(self, active):
        """set the active state of the Goal. if the Goal is exclusive, set all
        other Goals to inactive."""
        active = bool(active)
        
        #deactivate the other goals if we are exclusive and becoming active
        if active == True and self.exclusive:
            for goal in self.world.get_goals(active=True).values():
                if goal is not self: goal.deactivate()
        
        self._active = active
    
    @property
    def exclusive(self):
        """True if the goal wants no other goals to be active when it is"""
        return self._exclusive
    
    @exclusive.setter
    def exclusive(self, exclusive):
        self._exclusive = exclusive
    
    def prepare(self):
        #make sure we handle exclusivity properly
        self.active = self.active
    
    def activate(self):
        """activate the goal"""
        self.set_activation(True)
    
    def deactivate(self):
        """deactivate the goal"""
        self.set_activation(False)
    
    def set_activation(self, state):
        """make the Goal either active or inactive
        
        Parameters
        ----------
        state : bool
            the new activation state
        
        Returns
        -------
        old_state : bool
            the old activation state
        """
        old_state = self.active
        
        self.active = state
        
        return old_state
    
    def match(self, other):
        """match the particulars of another Goal. by default this does nothing,
        but subclasses can override it if e.g. the concrete Goal depends on
        parameter values.
        
        Parameters
        ----------
        other : Goal
            the Goal to match
        """
        pass
    
    def is_achieved(self):
        """determine whether the goal is achieved in the current world state
        
        Returns
        -------
        achieved : bool
            True if the goal is achieved in the current state
        """
        return bool(self._is_achieved())
    
    def setup_preconditions(self, error=True):
        """do anything necessary to ensure that the goal is achievable
        
        Parameters
        ----------
        error : bool, optional
            True if an error should be raised if the pre-conditions cannot be
            achieved
        
        Returns
        -------
        success : bool | string
            True if the preconditions were met. if the preconditions could not
            be achieved, return either False or a string describing why.
        """
        success = self._setup_preconditions()
        
        if not self._test_success(success) and error:
            success = ': %s' % (success) if isinstance(success, basestring) else ''
            raise RuntimeError('preconditions could not be met%s' % (success))
        
        return success
    
    def achieve(self, setup_preconditions=True, error=True):
        """achieve the goal by manipulating the world
        
        Parameters
        ----------
        setup_preconditions : bool, optional
            True if setup_preconditions() should be called. this should probably
            only be False if preconditions are being set up somewhere else
            manually.
        error : bool, optional
            True if an error should be raised if the goal isn't achieved
        
        Returns
        -------
        success : bool | string
            True if the goal was achieved. if the goal could not be achieved,
            return either False or a string describing why.
        """
        #setup the preconditions
        if setup_preconditions:
            success = self.setup_preconditions(error=False)
        else:
            success = True
        
        if self._test_success(success):  # achieve the goal
            success = self._achieve()
        elif success is False:
            success = 'preconditions could not be met'
        
        #make sure we actually achieved it
        if self._test_success(success) and not self.achieved: success = 'goal was not actually achieved'
        
        if not self._test_success(success) and error:
            success = ': %s' % (success) if isinstance(success, basestring) else ''
            raise RuntimeError('goal could not be achieved%s' % (success))
        
        return success
    
    def test_achieve(self, n=100, render=False, error=False):
        """test achieve() multiple times to make sure it is always achievable
        
        Parameters
        ----------
        n : int, optional
            the number of times to test achieve()
        render : bool, optional
            True to render the world state immediately before and after
            achieving the goal
        error : bool, optional
            True to raise an error if either the preconditions or the goal could
            not be achieved
        
        Returns
        -------
        f : float
            the fraction of achieve() calls that were successful
        """
        num_success = 0
        for _ in xrange(n):
            #reset the world
            self.world.reset()
            seed = self.world.history.most_recent_seed()
            
            #set up the preconditions
            success = self.setup_preconditions(error=error)
            if render: self.world.render()
            if not self._test_success(success):
                self.world.debug.log('setup_preconditions() failed: %s (seed: %s)' % (success, seed), level=Debug.ERROR)
                continue
            
            #achieve the goal
            success = self.achieve(setup_preconditions=False, error=error)
            if render: self.world.render()
            if not self._test_success(success):
                self.world.debug.log('achieve() failed: %s (seed: %s)' % (success, seed), level=Debug.ERROR)
                continue
            
            num_success += 1
        
        return float(num_success) / n
    
    def _add_world(self, world):
        """see Entity._add_world()"""
        super(Goal, self)._add_world(world)
        
        #remove the old goal first
        old_goal = world._goals.get(self.name, None)
        if old_goal is not None:
            assert old_goal is not self
            old_goal.remove()
        
        world._goals[self.name] = self
    
    def _remove_world(self):
        """see Entity._remove_world()"""
        try:
            del self.world._goals[self.name]
        except KeyError:  # must have already been removed
            pass
        
        super(Goal, self)._remove_world()
    
    def _is_achieved(self):
        """private version of is_achieved(). override this to define the
        achievement condition.
        
        Returns
        -------
        achieved : bool
            True if the goals is achieved in the current state
        """
        return False
    
    def _setup_preconditions(self):
        """private version of setup_preconditions(). override this to implement
        any processes that are need to ensure that the goal can be met.
        
        Returns
        -------
        success : bool | string
            True if the preconditions were met. if the preconditions could not
            be achieved, return either False or a string describing why.
        """
        return True
    
    def _achieve(self):
        """private version of achieve(). override this to implement a method for
        achieving the goal.
        
        Returns
        -------
        success : bool | string
            True if the goal was achieved. if the goal could not be achieved,
            return either False or a string describing why.
        """
        return self.achieved
    
    def _test_success(self, success):
        """test whether a value represents success
        
        Parameters
        ----------
        success : bool | string
            the return value from one of the methods that returns success
        
        Returns
        -------
        test : bool
            True if the success represents a True value
        """
        return not isinstance(success, basestring) and success == True


#---Judge----------------------------------------------------------------------#
class Judge(Entity):
    """a Judge decides:
        1) the reward to give the agent(s) at each world step
        2) when to end the simulation.
    
    subclasses can define the Judge's behavior by changing the following
    attribute values and/or by overriding _calculate_reward() and _is_done().
    """
    _class_family = 'judge'
    
    #the judge's name. if subclasses do not assign this explicitly, the name
    #is derived sensibly from the class name, e.g. SoloTennisJudge becomes
    #'solo_tennis'
    _name = 'judge'
    
    #this causes each Judge class to have the listed attributes as lists that
    #are assembled based on the Judge's superclasses (see TypeFamily and the
    #attribute descriptions below)
    _class_tree_attributes = ['_reward_goals', '_reward_events',
        '_termination_events']
    
    #True if the judge judges. False if it always return 0 reward and not
    #terminate the episode.
    _active = True
    
    #unless the public calculate_reward() is overridden, this value will be
    #subtracted from the reward at each step
    _step_penalty = 1
    
    #if an episode should have a maximum time scale, override this attribute
    _end_time = np.inf
    
    #defines how achievement of terminating Goals translates to termination
    #conditions:
    #   'any': achieving any terminating Goal terminates the episode
    #   'all': achieving all terminating Goals terminates the episode
    #   'ignore': ignore terminating Goals when determining whether to terminate
    #               the episode
    _goal_termination_mode = 'any'
    
    #specify whether the Judge should take into account rewards and termination
    #conditions specified by Events/Goals themselves
    _use_goal_reward = True
    _use_event_reward = True
    _use_event_termination = True
    
    #keep track of goals that have been achieved
    _achieved_goals = None
    
    #if the Judge should assign rewards based on achieving particular Goals,
    #specify those Goals and their rewards in this list of dicts. each dict has
    #the following keys:
    #   goal:  the name of the Goal
    #   params: optional, one of the following:
    #               - a dict that specifies the particular values that the
    #                 Goals's attributes must have in order to qualify for the
    #                 reward. each dict key is a Goal attribute name, and each
    #                 dict value is either the value the attribute must be, or a
    #                 function that takes the Judge and the attribute value as
    #                 inputs and returns a boolean to indicate whether the Goal
    #                 qualifies for the reward based on that value. see the
    #                 solo_tennis universe environment for an example Judge that
    #                 uses this (for Events).
    #               - a function that takes the Judge and the Goal as inputs
    #                 and returns a boolean to indicate whether the Goal
    #                 qualifies for the reward
    #   reward: the reward to assign to achieving the Goal, or a function that
    #           takes the Judge and the Goal as inputs and returns the reward
    #NOTE that if _use_goal_reward is True, then intrinsic Goal rewards are
    #assigned on top of the rewards defined here.
    _reward_goals = []
    
    #the equivalent of _reward_goals for Events
    _reward_events = []
    
    #if the Judge should terminate the episode whenever specific Events occur,
    #regardless of whether they indicate that they are termination events,
    #specify them in the list of dicts below. each dict should follow the
    #specifications for _reward_events above, without the 'reward' key.
    _termination_events = []
    
    #attributes to save in States
    _state_attributes = ['_active', '_achieved_goals']
    
    def __init__(self, world, active=None, step_penalty=None, end_time=None,
                    use_goal_reward=None, goal_termination_mode=None,
                    use_event_reward=None, use_event_termination=None,
                    **kwargs):
        """
        Parameters
        ----------
        world : PixelWorld
            the host world
        step_penalty : Number, optional
            override the default reward to associate with each simulation step
        end_time : Number, optional
            override the default end time
        use_goal_reward : bool, optional
            override the default use goal reward attribute
        goal_termination_mode : string, optional
            override the default goal termination mode
        use_event_reward : bool, optional
            override the default use event reward attribute
        use_event_termination : bool, optional
            override the default use event termination attribute
        """
        self._achieved_goals = []
        
        if active is not None: self.active = active
        if step_penalty is not None: self._step_penalty = step_penalty
        if end_time is not None: self._end_time = end_time
        if use_goal_reward is not None: self._use_goal_reward = use_goal_reward
        if goal_termination_mode is not None: self._goal_termination_mode = goal_termination_mode
        if use_event_reward is not None: self._use_event_reward = use_event_reward
        if use_event_termination is not None: self._use_event_termination = use_event_termination
        
        super(Judge, self).__init__(world, **kwargs)
    
    @property
    def active(self):
        """is the judge actively judging?"""
        return self._active
    
    @active.setter
    def active(self, active):
        self._active = active
    
    def process_step(self):
        """judge the results of a step
        
        Returns
        -------
        reward : Number
            the reward to associate with the step
        done : bool
            True if the simulation should finish
        """
        if self.active:
            #find the newly achieved goals
            goals = self.world.active_goals.values()
            unachieved_goals = list(set(goals) - set(self._achieved_goals))
            newly_achieved_goals = [goal for goal in unachieved_goals if goal.achieved]
            self._achieved_goals.extend(newly_achieved_goals)
            
            #new events to consider
            step_events = self.world.step_events
            
            #compute the step results
            reward = self.calculate_reward(newly_achieved_goals, step_events)
            done = self.is_done(newly_achieved_goals, step_events)
            
            return reward, done
        else:
            return 0, False
    
    def calculate_reward(self, goals, events):
        """calculate the reward to associate with a set of goals that were
        achieved and events that occurred during a world step. since the judge
        also has a world attribute, it can make its judgment based on anything,
        not just the lists of goals and events that were passed.
        
        Parameters
        ----------
        goals : list[Goal]
            a list of Goals that were achieved during the step
        events : list[Event]
            a list of Events that occurred the step
        
        Return
        ------
        reward : Number
            the reward to associate with the step
        """
        #step penalty
        reward = -self._step_penalty
        
        #intrinsic Goal rewards
        if self._use_goal_reward:
            reward += sum([goal.reward for goal in goals])
        
        #Judge-specific Goal rewards
        for item in self._reward_goals:
            #goals that match the reward goal name
            candidate_goals = [goal for goal in goals if item['goal'] == goal.name]
            
            goal_reward = item['reward']
            
            params = item.get('params', True)
            for goal in candidate_goals:
                #determine whether the goal meets the reward criteria
                if self._resolve_conditions(goal, params):
                    reward += self._resolve_reward(goal, goal_reward)
        
        #intrinsic Event rewards
        if self._use_event_reward:
            reward += sum([event.reward for event in events])
        
        #Judge-specific Event rewards
        for item in self._reward_events:
            #events that match the reward event name
            candidate_events = [event for event in events if item['event'] == event.name]
            
            event_reward = item['reward']
            
            params = item.get('params', True)
            for event in candidate_events:
                #determine whether the event meets the reward criteria
                if self._resolve_conditions(event, params):
                    reward += self._resolve_reward(event, event_reward)
        
        return reward + self._calculate_reward(goals, events)
    
    def is_done(self, goals, events):
        """given a list of goals and events that occurred during a step,
        determine whether the episode has ended. again, anything in the world
        can potentially be used to determine termination criteria, not just
        the passed lists of goals and events.
        
        Parameters
        ----------
        goals : list[Goal]
            a list of goals that were achieved during the step
        events : list[Event]
            a list of Events that occurred during the step
        
        Returns
        -------
        done : bool
            True if the episode has finished
        """
        #check for timeout
        if self.world.time >= self._end_time:
            return True
        
        #Goal-based termination
        if self._goal_termination_mode != 'ignore':
            if self._goal_termination_mode == 'any':  # terminate if any goal achieved
                terminate = any([goal.terminates for goal in goals])
            elif self._goal_termination_mode == 'all':  # terminate if all goals achieved
                test_goals = self.world.get_goals(terminates=True, active=True).values()
                terminate = len(test_goals) > 0 and all([goal.achieved for goal in test_goals])
            else:
                raise ValueError('"%s" is not a valid goal termination mode' % (self._goal_termination_mode))
            
            if terminate: return True
        
        #intrinsic Event termination
        if self._use_event_termination and any([event.terminates for event in events]):
            return True
        
        #Judge-specific termination criteria
        for item in self._termination_events:
            candidate_events = [event for event in events if item['event'] == event.name]
            
            params = item.get('params', True)
            for event in candidate_events:
                if self._resolve_conditions(event, params):
                    return True
        
        #overridden Judge-specification termination criteria
        return self._is_done(goals, events)
    
    def _add_world(self, world):
        """see Entity._add_world()"""
        super(Judge, self)._add_world(world)
        
        #remove the old judge first
        if world._judge is not None:
            world._judge.remove()
        
        world._judge = self
    
    def _remove_world(self):
        """see Entity._remove_world()"""
        if self._world._judge is self:
            self._world._judge = None
        
        super(Judge, self)._remove_world()
    
    def _resolve_conditions(self, entity, params):
        """determine whether an Entity meets a set of criteria
        
        Parameters
        ----------
        entity : Entity
            an Entity to evaluate
        params : dict | boolean | callable
            see the _reward_goals documentation above
        
        Returns
        -------
        meets : bool
            True if the Entity meets the specified criteria
        """
        if callable(params):  # params is a function
            return params(self, entity)
        elif isinstance(params, bool):  # params is a boolean
            return params
        elif isinstance(params, dict):  # params is a dict of parameter values
            for param,test in params.iteritems():
                value = getattr(entity, param)
                
                if callable(test):
                    meets = test(self, value)
                else:
                    meets = value == test
                
                if not meets:
                    return False
            
            return True
        else:
            raise TypeError('"%s" is not a valid params specification' % (params))
    
    def _resolve_reward(self, event, reward):
        """resolve a reward that might actually be a function that returns a
        reward
        
        Parameters
        ----------
        event : Event
            the Event associated with the reward
        reward : Number | callable
            either a reward or a function that takes the Judge and the Event as
            inputs and returns the reward
        
        Returns
        -------
        reward : Number
            the reward
        """
        if callable(reward):
            return reward(self, event)
        elif isinstance(reward, Number):
            return reward
        else:
            raise TypeError('"%s" is not a valid reward specification' % (reward))
    
    def _calculate_reward(self, goals, events):
        """private version of calculate_reward. override this to return a custom
        reward on top of those defined by the step penalty, intrinsic Goals and
        Event rewards, and Judge-specified reward Events."""
        return 0
    
    def _is_done(self, goals, events):
        """private version of is_done. override this to implement custom
        termination conditions on top of those defined intrinsically by Goals,
        Events, or via _termination_events"""
        return False


#---Variant--------------------------------------------------------------------#
class Variant(Entity):
    """a Variant defines and brings about some variant of an existing world. for
    instance, it could turn colors on or off, or switch between variations of an
    Object. the default Randomizer chooses between states of each Variant when
    randomizing.
    
    a Variant is defined by three things:
        1) a list of possible variant state values
        2) the current variant state
        3) a function that brings about a variant state.
    
    if a Variant is active when its prepare() method is called, then its current
    variant state will be brought about.
    
    subclasses should define _states and override _set().
    """
    _class_family = 'variant'
    
    #a list of possible variant states. if this is a string, it will instead be
    #mapped to the world attribute with that name (which must exist).
    _states = None
    
    #the index of the current variant state. if this is None, then it will be
    #set to a random index when Variant.state is accessed.
    _state = None
    
    #True if the Variant is active
    _active = True
    
    def __init__(self, world, states=None, state=None, active=None, **kwargs):
        """
        Parameters
        ----------
        states : list, optional
            override the default list of variant states
        state : any
            override the initial variant state
        active : bool, optional
            override the initial active attribute value
        """
        self.states = states or self._states
        self.active = active or self._active
        
        super(Variant, self).__init__(world, **kwargs)
        
        if state is not None: self._state = self._get_state_index(state)
    
    @property
    def specifications(self):
        spec = super(Variant, self).specifications
        
        spec[1]['state'] = self.states[spec[1]['state']]
        
        return spec
    
    @property
    def active(self):
        """True if the Variant is currently active"""
        return self._active
    
    @active.setter
    def active(self, active):
        """set the active state of the Variant. NOTE that setting active to True
        will not cause the actual variant to be set."""
        self._active = bool(active)
    
    @property
    def state(self):
        """the value (not the index) of the current variant state"""
        if self._state is None: self._state = self.rng.randint(len(self.states))
        return self.states[self._state]
    
    @property
    def states(self):
        """a list of possible variant states"""
        if isinstance(self._states, basestring):  # linked to world attribute
            return getattr(self.world, self._states)
        else:
            return self._states
    
    @states.setter
    def states(self, states):
        """set the variant states. states must be a list of unique values."""
        if isinstance(self._states, basestring) and \
        not isinstance(states, basestring):  # linked to world attribute
            setattr(self.world, self._states, states)
        else:
            if not isinstance(states, (list, basestring)):
                raise TypeError('states must be a list or the name of a WorldAttribute to which to link')
            
            self._states = states
    
    def prepare(self):
        """set the current variant state if we are active"""
        if self.active: self.set()
    
    def set(self, state=None):
        """bring about the specified variant state
        
        Parameters
        ----------
        state : any
            the value of the variant state to bring about. must be a valid
            state.
        """
        if state is not None:
            self._state = self._get_state_index(state)
        
        self._set(self.state)
    
    def get_conditional_states(self, variant_states):
        """subclasses can override this to compute what the set of possible
        states would be, given the current states of a set of other Variants.
        
        by default, we don't care about other Variants.
        
        Parameters
        ----------
        variant_states : dict
            a dict mapping Variant names to Variant states
        """
        return self.states
    
    def _add_world(self, world):
        """see Entity._add_world()"""
        super(Variant, self)._add_world(world)
        
        #remove the old variant first
        old_variant = world._variants.get(self.name, None)
        if old_variant is not None:
            assert old_variant is not self
            old_variant.remove()
        
        world._variants[self.name] = self
    
    def _remove_world(self):
        """see Entity._remove_world()"""
        try:
            del self.world._variants[self.name]
        except KeyError:  # must have already been removed
            pass
        
        super(Variant, self)._remove_world()
    
    def _get_state_index(self, state):
        """get the index of a state in the states list
        
        Parameters
        ----------
        state : any
            a state value
        
        Returns
        -------
        idx : int
            the index of the state in Variant.states
        """
        try:
            return self.states.index(state)
        except ValueError:
            raise ValueError('"%s" is not a valid state' % (state))
    
    def _set(self, state):
        """private version of set()
        
        Parameters
        ----------
        state : any
            the value of the variant state to bring about
        """
        pass


class BooleanVariant(Variant):
    """a Variant that has True and False states"""
    _name = 'boolean_variant'
    
    _states = [True, False]


class IntegerVariant(Variant):
    """a Variant that takes on one of a set of integer values. states can either
    be a range of integers (set min_state and max_state) or a set of integers
    (set states directly)."""
    _name = 'integer_variant'
    
    #the inclusive range (min, max) of states. if these are strings, they will
    #instead be mapped to world attributes with that name (which must exist).
    _min_state = 1
    _max_state = 10
    
    def __init__(self, world, min_state=None, max_state=None, **kwargs):
        """
        Parameters
        ----------
        min_state : int, optional
            override the default min_state
        max_state : int, optional
            override the default max_state
        """
        self.min_state = min_state or self._min_state
        self.max_state = max_state or self._max_state
        
        super(IntegerVariant, self).__init__(world, **kwargs)
    
    @property
    def min_state(self):
        """minimum state value"""
        if isinstance(self._min_state, basestring):  # linked to world attribute
            return getattr(self.world, self._min_state)
        else:
            return self._min_state
    
    @min_state.setter
    def min_state(self, min_state):
        self.states = None
        
        if isinstance(self._min_state, basestring) and \
        not isinstance(min_state, basestring):  # linked to world attribute
            setattr(self.world, self._min_state, min_state)
        else:
            self._min_state = min_state
    
    @property
    def max_state(self):
        """maximum state value"""
        if isinstance(self._max_state, basestring):  # linked to world attribute
            return getattr(self.world, self._max_state)
        else:
            return self._max_state
    
    @max_state.setter
    def max_state(self, max_state):
        self.states = None
        
        if isinstance(self._max_state, basestring) and \
        not isinstance(max_state, basestring):  # linked to world attribute
            setattr(self.world, self._max_state, max_state)
        else:
            self._max_state = max_state
    
    @property
    def states(self):
        if self._states is None:
            return self.get_states_given_range(self.min_state, self.max_state)
        else:
            return super(IntegerVariant, self).states
    
    @states.setter
    def states(self, states):
        if states is not None:
            super(IntegerVariant, self.__class__).states.fset(self, states)

            if states:
                self._min_state = min(self.states)
                self._max_state = max(self.states)
        else:
            self._states = states
    
    def get_states_given_range(self, min_state, max_state):
        """get a list of possible states, given the minimum and maximum state
        value
        
        Parameters
        ----------
        min_state : int
            the minimum state value
        max_state : int
            the maximum state value
        
        Returns
        -------
        states : list
            the list of possible states
        """
        return range(min_state, max_state + 1)


class WorldAttributeVariant(Variant):
    """a Variant that sets the value of a WorldAttribute. the default
    WorldAttribute is the one with the same name as the Variant.
    
    see variants.ShowColorsVariant.
    """
    _name = 'world_attribute_variant'
    
    #the name of the WorldAttribute to set. if this is None, the Variant name
    #will be used.
    _world_attribute_name = None
    
    def __init__(self, world, world_attribute_name=None, **kwargs):
        """
        Parameters
        ----------
        world_attribute_name : string, optional
            override the default world attribute name
        """
        self.world_attribute_name = world_attribute_name or self._world_attribute_name
        
        super(WorldAttributeVariant, self).__init__(world, **kwargs)
    
    @property
    def world_attribute_name(self):
        """name of the WorldAttribute to set"""
        return self._world_attribute_name or self._name
    
    @world_attribute_name.setter
    def world_attribute_name(self, name):
        self._world_attribute_name = name
    
    def _set(self, state):
        """set the specified WorldAttribute to state"""
        setattr(self.world, self.world_attribute_name, state)


class SetWorldAttributeVariant(WorldAttributeVariant):
    """WorldAttributeVariant whose corresponding world attribute is a
    SetWorldAttribute. the variant's states default to the allowed values of the
    world attribute.
    """
    _name = 'set_world_attribute_variant'
    
    @property
    def states(self):
        if self._states is None:
            return self.world.get_world_attribute(self.world_attribute_name).values
        else:
            return WorldAttributeVariant.states.fget(self)
    
    @states.setter
    def states(self, states):
        if states is not None:
            WorldAttributeVariant.states.fset(self, states)


class ObjectAttributeVariant(Variant):
    """a Variant that sets the value of an ObjectAttribute. the default
    ObjectAttribute is the one with the same name as the Variant.
    
    the Variant can be restricted to specific Objects by specifying
    _object_attribute_filter as a dict that gets passed as **kwargs to
    world.objects.find().
    """
    _name = 'object_attribute_variant'
    
    _class_tree_attributes = ['_object_filter']
    
    #the name of the ObjectAttribute to set. if this is None, the Variant name
    #will be used.
    _object_attribute_name = None
    
    #a dict that will be used to find the Objects to which to apply the Variant.
    #this dict will be passed as the **kwargs argument to world.objects.find()
    _object_filter = None
    
    def __init__(self, world, object_attribute_name=None, object_filter=None,
                    **kwargs):
        """
        Parameters
        ----------
        object_attribute_name : string, optional
            override the default object attribute name
        object_filter : dict, optional
            override the default object filter
        """
        self.object_attribute_name = object_attribute_name or self._object_attribute_name
        self.object_filter = object_filter or self._object_filter
        
        super(ObjectAttributeVariant, self).__init__(world, **kwargs)
    
    @property
    def object_attribute_name(self):
        """name of the ObjectAttribute to set"""
        return self._world_attribute_name or self._name
    
    @object_attribute_name.setter
    def object_attribute_name(self, name):
        self._object_attribute_name = name
    
    @property
    def object_filter(self):
        """see _object_filter"""
        return self._object_filter
    
    @object_filter.setter
    def object_filter(self, object_filter):
        if object_filter is None:
            object_filter = {}
        
        if not isinstance(object_filter, dict):
            raise TypeError('object_filter must be a dict')
        
        self._object_filter = object_filter
    
    def _set(self, state):
        """set the specified ObjectAttribute to state for the specified
        Objects"""
        #ObjectAttribute to set
        name = self._object_attribute_name or self._name
        attr = self.world.get_object_attribute(name)
        
        #Objects to set
        if len(self._object_filter) == 0:
            objects = attr.objects
        else:
            objects = self.world.objects.find(**self._object_filter)
            objects = [obj for obj in objects if hasattr(obj, name)]
        
        #set the attribute values
        for obj in objects:
            setattr(obj, name, state)


class ObjectCountVariant(IntegerVariant):
    """a Variant that controls the number of Objects of a specific type that
    exist in the scene.
    """
    _name = 'object_count_variant'
    
    #the type of Object to create. can either be an Object name or Object class.
    _object_type = 'object'
    
    #a dict of parameters to use when creating each Object
    _object_params = None
    
    #this will keep track of the Objects that we have created.
    _objects = None
    
    def __init__(self, world, object_type=None, object_params=None, **kwargs):
        """
        Parameters
        ----------
        object_type : string | TypeFamily, optional
            override the default object type
        object_params : dict, optional
            override the default object params
        """
        self._objects = []
        
        self.object_type = object_type or self._object_type
        self.object_params = object_params or self._object_params
        
        super(ObjectCountVariant, self).__init__(world, **kwargs)
    
    @property
    def object_type(self):
        """the type of Object to create"""
        return self._object_type
    
    @object_type.setter
    def object_type(self, object_type):
        self._object_type = object_type
    
    @property
    def object_params(self):
        return self._object_params
    
    @object_params.setter
    def object_params(self, params):
        if params is None:
            params = {}
        
        if not isinstance(params, dict):
            raise TypeError('object_params must be a dict')
        
        self._object_params = params
    
    @property
    def object_specs(self):
        return [self.object_type, self.object_params]
    
    def _set(self, state):
        """remove any Objects we previously created, and create a new set of
        Objects"""
        #delete the old Objects
        for obj in self._objects:
            obj.remove()
        del self._objects[:]
        
        #create the new Objects
        object_specs = [self.object_specs for _ in xrange(state)]
        self._objects.extend(self.world.create_objects(object_specs))


#---Randomizer-----------------------------------------------------------------#
class Randomizer(Entity):
    """a Randomizer decides how to randomize the starting conditions of the
    world after a reset takes place. the default randomizer does nothing except
    choose randomly between the states of the world's active Variants.
    
    the base Randomizer class can be easily extended by defining
    _randomized_attributes as a list of the names of RandomizingObjectAttributes
    that should be randomized.
    
    a Randomizer can additionally define _excluded_objects as a list of the
    names of Objects to exclude when randomizing attributes defined in
    _randomized_attributes.  Objects, in turn, can define
    _exclude_randomize_attributes as a list of the names of ObjectAttributes to
    exclude when randomizing the Object's attributes, and/or _exclude_randomize
    as a boolean that tells the Randomizer whether to exclude the Object from
    randomization.
    
    finally, a Randomizer can override _randomize() or one of the methods that
    it calls to implement custom randomizations.
    
    Notes about the base Randomizer:
        - it randomizes only Objects without parents.
        - it randomizes attributes in the order defined by
            _initialize_before/after.
        - even though randomize() takes a seed argument, the base Randomizer
          doesn't use it. this means that the base Randomizer will randomize the
          world in exactly the same way with every reset, which might not be the
          desired behavior. if each reset() should lead to a different
          randomization, then use randomizers.ReseedingRandomizer.
    """
    _class_family = 'randomizer'
    
    #the randomizer's name. if subclasses do not assign this explicitly, the name
    #is derived sensibly from the class name, e.g. AllPositionsRandomizer becomes
    #'all_positions'
    _name = 'randomizer'
    
    #this causes each Randomizer to have the following attributes as lists that
    #are assembled based on the Randomizer's superclasses (see TypeFamily and
    #the attribute below)
    _class_tree_attributes = ['_randomize_attributes', '_exclude_objects']

    #a list of the names of attributes that should be randomized. these
    #attributes must be RandomizingObjectAttributes.
    _randomized_attributes = []
    
    #a list of the names of Objects that should be excluded from ObjectAttribute
    #randomization
    _excluded_objects = []
    
    #this will keep track of Objects that are generated during a call to
    #randomize()
    _generated_objects = None
    
    #attributes to exclude from States
    _auto_state_exclude = ['_randomized_attributes_removed',
        '_excluded_objects_removed']
    
    def __init__(self, world, randomized_attributes=None, excluded_objects=None,
                    **kwargs):
        """
        Parameters
        ----------
        world : PixelWorld
            the host world
        randomized_attributes : list[string], optional
            a list of names of ObjectAttributes to randomize
        excluded_objects : list[string], optional
            a list of names of Objects to exclude from randomization
        """
        self._generated_objects = []
        
        self.randomized_attributes = randomized_attributes or self._randomized_attributes
        self.excluded_objects = excluded_objects or self._excluded_objects
        
        super(Randomizer, self).__init__(world, **kwargs)
    
    @property
    def randomized_attributes(self):
        """a list of names of ObjectAttributes to randomize"""
        return self._randomized_attributes
    
    @randomized_attributes.setter
    def randomized_attributes(self, randomized_attributes):
        if not isinstance(randomized_attributes, list) or \
        not all([isinstance(name, basestring) for name in randomized_attributes]):
            raise TypeError('randomized_attributes must be a list of ObjectAttribute names')
        
        self._randomized_attributes = randomized_attributes
    
    @property
    def excluded_objects(self):
        """a list of names of Objects to exclude"""
        return self._excluded_objects
    
    @excluded_objects.setter
    def excluded_objects(self, excluded_objects):
        if not isinstance(excluded_objects, list) or \
        not all([isinstance(name, basestring) for name in excluded_objects]):
            raise TypeError('excluded_objects must be a list of Object names')
        
        self._excluded_objects = excluded_objects
    
    def pre_randomize(self, seed):
        """prepare for randomizing
        
        Parameters
        ----------
        seed : int
            the seed value to use
        """
        #record a randomize history event
        self.world.history.append_randomize(seed=seed)
        
        #block the history
        self.world.history.block()
    
    def randomize(self, seed=None):
        """randomize the state of the world
        
        Parameters
        ----------
        seed : int, optional
            the seed value to use. in unspecified, uses the current
            world.seed_value.
        """
        if seed is None: seed = self.world.seed_value
        
        self.world.debug.log('%s randomizing with seed=%s' % (self.name, seed), level=Debug.INFO)
        
        #pre-randomization stuff
        self.pre_randomize(seed)
        
        #perform the randomization
        self._randomize()
        
        #post-randomization stuff
        self.post_randomize()
    
    def post_randomize(self):
        """perform any necessary processes after randomizing"""
        #unblock the history
        self.world.history.unblock()
    
    def _add_world(self, world):
        """see Entity._add_world()"""
        super(Randomizer, self)._add_world(world)
        
        #remove the old randomizer first
        if world._randomizer is not None:
            world._randomizer.remove()
        
        world._randomizer = self
    
    def _remove_world(self):
        """see Entity._remove_world()"""
        if self._world._randomizer is self:
            self._world._randomizer = None
        
        super(Randomizer, self)._remove_world()
    
    def _randomize(self):
        """private version of randomize()"""
        #set each variant state
        self._randomize_variants()
        
        #randomize the Objects in the scene
        self._randomize_objects()
        
        #randomize each specified RandomizingObjectAttribute
        self._randomize_object_attributes()
    
    def _randomize_variants(self):
        """randomize the state of each active Variant"""
        variants = self.world.active_variants.values()
        
        for variant in variants:
            state = self._get_variant_state(variant)
            variant.set(state)
    
    def _randomize_objects(self):
        """delete the previously-generated Objects and generate a new set"""
        #delete the previously-generated Objects
        for obj in self._generated_objects:
            if isinstance(obj, CompoundObject):
                obj.remove(remove_children=True)
            else:
                obj.remove()
        del self._generated_objects[:]
        
        #generate the new set of Objects
        object_specs = self._get_object_specs()
        self._generated_objects.extend(self.world.create_objects(object_specs))
    
    def _randomize_object_attributes(self):
        """randomize each of the selected ObjectAttributes"""
        for name in self.world._randomizing_object_attributes:
            if name not in self.randomized_attributes:  # attribute not in our list
                continue
            
            attr = self.world.object_attributes[name]
            
            #get the indices of the Objects to randomize
            indices = self._get_object_attribute_indices(attr)
            
            #randomize the values
            values = self._get_object_attribute_values(attr, indices)
            attr.randomize(indices, value=values)
    
    def _get_variant_state(self, variant):
        """get the next states of a Variant
        
        Parameters
        ----------
        variant : Variant
            the Variant whose state should be assigned
        
        Returns
        -------
        state : any
            the Variant's new state
        """
        return self.rng.choice(variant.states)
    
    def _get_object_specs(self):
        """override this to return a list of Object specs if Objects should be
        generated during calls to randomize(), in between setting variants and
        randomizing ObjectAttributes.
        
        Returns
        -------
        specs : list
            a list of Object specs
        """
        return []
    
    def _get_object_attribute_indices(self, attr):
        """get the indices of the Objects that should be randomized for a
        particular ObjectAttribute
        
        Parameters
        ----------
        attr : ObjectAttribute
            the ObjectAttribute
        
        Returns
        -------
        indices : ndarray
            an array of the indices of Objects whose attribute values should be
            randomized
        """
        return np.array([obj._id for obj in attr.objects if
                        not hasattr(obj, 'parent') and  # doesn't have a parent
                        not getattr(obj, '_exclude_randomize', False) and  # doesn't exclude itself
                        attr.name not in getattr(obj, '_exclude_randomize_attributes', ()) and #doesn't exclude the attribute
                        obj.name not in self.excluded_objects  # not in excluded Objects
                        ]).astype(int)
    
    def _get_object_attribute_values(self, attr, indices):
        """get the next set of attribute values for the given set of Object
        indices
        
        Parameters
        ----------
        attr : RandomizingObjectAttribute
            the randomizing object attribute for which to generate values
        indices : ndarray
            an array of Object indices
        
        Returns
        -------
        values : ndarray
            an array of new attribute values
        """
        return attr._get_random_values(indices)


#---Agent----------------------------------------------------------------------#
class Agent(Entity):
    """it is not necessary to implement agents as subclasses of the Agent class
    here. PixelWorld can be treated exactly like any other OpenAI Gym
    environment, where the agent is kept strictly separate from the environment
    and must call the env.step() method itself.
    
    however, implementing an Agent provides a more concise means of running
    simulations. if an Agent is defined via this class. then an episode is
    executed just by calling world.run(). at each step(), the world will query
    the Agent for an action, perform the action, and then inform the Agent of
    the results.
    
    subclasses should override _get_action() and _process_action_result().
    
    NOTE that any pseudorandom behavior should be implemented via tools['rng'],
    so that episodes are reproducible.
    
    some basic Agents are implemented in agents.py. the default Agent defined
    here always chooses the _default_action as its action.
    """
    _class_family = 'agent'
    
    #the agent's name. if subclasses do not assign this explicitly, the name
    #is derived sensibly from the class name, e.g. SoloTennisAgent becomes
    #'solo_tennis'
    _name = 'agent'
    
    #default action, for processes that want one
    _default_action = 'NOOP'

    #set of actions allowed to this agent. If None, any action in world.actions
    #is allowed.
    _allowed_actions = None

    def __init__(self, world, allowed_actions=None, **kwargs):
        """
        Parameters
        ----------
        world : PixelWorld
            world in which the agent will reside
        allowed_actions : list of string (optional)
            List of actions allowed to the agent. If omitted, the agent can
            take any action in world.actions.
        """
        super(Agent, self).__init__(world, **kwargs)
        self._allowed_actions = allowed_actions
    
    def get_action(self, obs, tools):
        """public version of _get_action(). structured like this so agents can
        add behaviors here without worrying about overwriting the core
        functionality defined in _get_action().
        """
        tools = copy(tools)
        if self._allowed_actions is not None:
            tools['actions'] = [x for x in tools['actions'] if x in self._allowed_actions]

        action = self._get_action(obs, tools)

        if self._allowed_actions is not None:
            assert action is None or action in self._allowed_actions, \
                'Action %s should not be available to this agent' % action

        return action
    
    def process_action_result(self, obs, action, reward, done):
        """public version of _process_action_result(). structured like this so
        agents can add behaviors here without worrying about overwriting the
        core functionality defined in _process_action_result().
        """
        self._process_action_result(obs, action, reward, done)
    
    def _add_world(self, world):
        """see Entity._add_world()"""
        super(Agent, self)._add_world(world)

        self.world._add_agent(self)
    
    def _remove_world(self):
        """see Entity._remove_world()"""
        self.world._remove_agent(self)
        
        super(Agent, self)._remove_world()
    
    def _get_action(self, obs, tools):
        """determine the next action to take, given the current observation of
        the world state
        
        Parameters
        ----------
        obs : list | ndarray
            the current observation, the format of which depends on parameters
            specified during world creation
        tools : dict
            a set of tools that the agent can use:
                'actions': a list of available actions
                'rng': a numpy.random.RandomState that should be used for any
                    pseudorandom behavior
        
        Returns
        -------
        action : string
            the action to perform, from the set of actions in tools['actions']
        """
        return self._default_action
    
    def _process_action_result(self, obs, action, reward, done):
        """process the consequences of performing an action
        
        Parameters
        ----------
        obs : list | ndarray
            the observation that the action was based on. this will be the same
            observation as was passed to the previous _get_action() call.
        action : string
            the action that was performed
        reward : float
            the reward that resulted from performing the action
        done : bool
            True if the episode has finished
        """
        pass


#---WorldAttribute------------------------------------------------------------#
class WorldAttribute(Entity):
    """a WorldAttribute stores and manipulates data associated with a
    PixelWorld. this class defines some core WorldAttribute behaviors. below it,
    some generic subclasses define additional behaviors that some
    WorldAttributes might want. below that, some core WorldAttributes that all
    PixelWorld's should possess are defined. additional WorldAttributes are
    defined in world_attributes.py.
    
    many WorldAttribute methods have a corresponding _method. subclasses should
    probably override _method rather than method, since generic WorldAttribute
    classes may add functionality to method, but leave _method
    alone. for this base class, there are the following method pairs:
        set -> _set
        get -> _get
        compare -> _compare
    the generic subclasses below adopt the same convention of public/private
    method pairs (this is inspired by the OpenAI gym Environment class)
    """

    #define a new class family (see TypeFamily)
    _class_family = 'world_attribute'
    
    #the attribute's name. if subclasses do not assign this explicitly, the name
    #is derived sensibly from the class name, e.g. HeightWorldAttribute
    #becomes 'height'
    _name = 'world_attribute'
    
    #this causes each WorldAttribute to have the specified attributes, which
    #are lists that are assembled based on the WorldAttribute's superclasses
    #(see TypeFamily and the attributes below for details)
    _class_tree_attributes = ['_coerce_value']
    
    #the numpy dtype of the attribute's data
    _dtype = np.dtype('O')
    
    #the dimensionality of the attribute value
    _ndim = 1
    
    #the default value of an attribute. this can also be a function that takes
    #the PixelWorld as input and returns a value.
    _default_value = 0

    #the WorldAttribute's value
    _value = None
    
    #set this to True if the WorldAttribute's values should not be set. NOTE
    #that read-only WorldAttributes should also override set() to raise an
    #error. this is only here to provide a record that the ObjectAttribute
    #intends to be read-only.
    _read_only = False
    
    _state_attributes = ['_value']
    
    def __new__(cls, *args, **kwargs):
        attr = super(WorldAttribute, cls).__new__(cls, *args, **kwargs)
        
        #make bound methods out of the _coerce_value functions (see
        #_coerce_value for an explanation)
        attr._coerce_value = [f.__get__(attr, cls) for f in attr._coerce_value]
        
        return attr
    
    def __init__(self, world, value=None, prepare=True, **kwargs):
        """
        Parameters
        ----------
        world : PixelWorld
            the host PixelWorld
        value : self.dtype
            the initial value, if the default value should be overridden
        prepare : bool
            see Entity
        **kwargs
            extra arguments
        """
        super(WorldAttribute, self).__init__(world, prepare=False, **kwargs)
        
        #add the data
        self._add_data(value)
        
        if prepare and self.world.populated:
            self.prepare()
    
    @property
    def dtype(self):
        """the WorldAttribute's data array dtype"""
        return self._dtype
    
    @property
    def ndim(self):
        """the dimensionality of the attribute value"""
        return self._ndim
        
    def coerce_value(self, value, error=True, **kwargs):
        """coerce values to comply with the requirements of the WorldAttribute,
        or raise an error if it can't comply. this method passes values
        sequentially through all _coerce_value methods defined by the class and
        its superclasses, starting with the root WorldAttribute class.
        
        Parameters
        ----------
        value : any
            the value (or iterable of values) to coerce
        error : bool
            True if an exception should be raised if the value does not comply
            with the WorldAttribute's requirements. if this is False and the
            input does not comply, None will be returned.
        **kwargs
            extra keyword arguments defined by particular _coerce_value methods
        
        Returns
        -------
        value : self.dtype
            the coerced value(s)
        """
        for f in self._coerce_value:
            value, err = f(value, **kwargs)
            
            if err:
                if error:
                    raise type(err)('%s %s' % (self._name, err))
                else:
                    return None
        
        return value
    
    def remove(self):
        """remove the WorldAttribute from the world. this destroys all record
        of the WorldAttribute."""
        #remove our data
        self._remove_data()
        
        super(WorldAttribute, self).remove()
    
    def get(self):
        """return the current attribute value
                
        Returns
        -------
        value : self.dtype
            the attribute value
        """
        return self._get()
    
    def set(self, value, validate=True):
        """set the attribute value
        
        Parameters
        ----------
        value : self.dtype
            the new value of the attribute
        validate : bool, optional
            True to pass the value(s) through coerce_value(). if this is False,
            the method assumes value is valid for the WorldAttribute's dtype.
        """
        if validate:
            value = self.coerce_value(value)
        
        self._set(value)
    
    def compare(self, x, y, validate=True):
        """compare two values
        
        Parameters
        ----------
        x : self.dtype
            a value
        y : self.dtype
            another value
        validate : bool, optional
            True to validate x and y (see set())
        
        Returns
        -------
        b : bool
            the boolean comparison between x and y. essentially, whether x and
            y are "equal".
        """
        return self._compare(x, y)
    
    def __call__(self, *args, **kwargs):
        """calling the WorldAttribute directly mimics calling get()"""
        return self.get(*args, **kwargs)
    
    def _to_vector(self, value):
        """make sure a value is a 1D numpy array of the same length as the
        dimensionality of the WorldAttribute
        
        Parameters
        ----------
        value : any
            a value that is or can be converted to a 1D numpy array of length
            self.ndim. scalars are tiled.
        
        Returns
        -------
        value : ndarray
            value as a 1D numpy array of length self.ndim
        """
        #wrap the value in an object array if self._dtype == object
        if self._dtype == object and \
        (not isinstance(value, np.ndarray) or \
        not value.dtype == object):
            x = value
            value  = np.empty((1,), dtype=object)
            value[0] = x
        
        #make sure we have a numpy array
        try:
            if not isinstance(value, np.ndarray):
                value = np.array(value, dtype=self._dtype)
            elif not value.dtype == self._dtype:  # make sure we have the correct dtype
                value = value.astype(self._dtype)
        except TypeError:
            raise TypeError('must be convertible to %s' % (self._dtype.name))
        
        if value.ndim == 1 and value.size == self._ndim:  # we have what we need
            return value
        elif value.ndim == 0:  # tile if we got a scalar
            return np.tile(value, (self._ndim,))
        elif value.shape == (1, self._ndim) or value.shape == (self._ndim, 1): # convert 1xN to N
            return np.reshape(value, (self._ndim,))
        
        raise ValueError('must be a %d-dimensional vector' % (self._ndim))
    
    def _coerce_value(self, value, **kwargs):
        """each WorldAttribute can optionally define a _coerce_value method
        that checks and/or coerces values into values that are valid for the
        WorldAttribute's dtype. when a WorldAttribute's public coerce_value
        method is called, each private _coerce_value method in the chain of
        classes is called sequentially from the root member of the class family
        up until the WorldAttribute itself. i.e. the output of one _coerce_value
        method becomes the input to the next. the method defined here performs
        some basic shape checking on values, and also attempts to coerce the
        value to be of type self.dtype.
        
        a _coerce_value method should not raise an exception in case of an
        invalid value. instead, it should return an Exception describing the
        issue as the second output argument (see below).
        
        Parameters
        ----------
        value : any
            a potential attribute value, or iterable of attribute values
        
        Returns
        -------
        value : self.dtype
            value as a valid attribute value
            values.
        err : False | Exception
            if the value was successfully coerced into a valid attribute value,
            then should be False. if an exception occurred because the value
            could not be coerced into a valid attribute value, then this should
            be the exception that occurred. note that if coerce_value ends up
            raising an exception because of the failure, its message will be
            this Exception's message, prepended with the name of the
            ObjectAttribute (e.g. if the WorldAttribute's name is 'mass' and
            the error is 'must be non-negative', then the exception message will
            be 'mass must be non-negative').
        """
        if self._ndim != 1:
            #make sure we have a vector
            try:
                value = self._to_vector(value)
            except (TypeError, ValueError) as e:
                return value, e
        elif not self._dtype == object and not isinstance(value, self._dtype.type):
            #make sure we have the right data type
            try:
                value = self._dtype.type(value)
            except (TypeError, ValueError) as e:
                return value, TypeError('must be convertible to %s' % (self._dtype.type.__name__))
        
        return value, False
    
    def _add_world(self, world):
        """see Entity._add_world()"""
        super(WorldAttribute, self)._add_world(world)
        
        world._world_attributes[self._name] = self
    
    def _remove_world(self):
        """see Entity._remove_world()"""
        try:
            del self._world._world_attributes[self._name]
        except KeyError:  # must have already been removed
            pass
        
        super(WorldAttribute, self)._remove_world()
    
    def _add_data(self, value):
        """add the WorldAttribute's data
        
        Parameters
        ----------
        value : self.dtype
            the initial attribute value, or None to use the default
        """
        if value is None:
            if callable(self._default_value):
                value = self._default_value(self._world)
            else:
                value = self._default_value
        
        self.set(value)
    
    def _remove_data(self):
        """remove the data added by _add_data"""
        self._value = None
    
    def _get(self):
        """get the attribute value
        
        Returns
        -------
        value : self.dtype
            the data array value
        """
        return self._value
    
    def _set(self, value):
        """set the attribute value
        
        Parameters
        ----------
        value : self.dtype
            the value to set
        """
        self._value = value
    
    def _compare(self, x, y):
        """WorldAttributes may want to override this method to define what a
        "comparison" between two values means. usually this will define what it
        means for two values to be equal.
        
        Parameters
        ----------
        x : self.dtype
            a value
        y : self.dtype
            another value
        
        Returns
        -------
        b : bool
            the comparison between the two values
        """
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            return np.array_equal(x, y)
        else:
            return x == y


class UnchangeableWorldAttribute(WorldAttribute):
    """attribute whose value can only be set at initialization"""
    _name = 'unchangeable_attribute'
    
    _initialized = False
    
    _state_attributes = ['_initialized']
    
    def set(self, value):
        """only allows the value to be set once"""
        if self._initialized:
            raise RuntimeError('world %s cannot be set after initialization' % (self._name))
        else:
            super(UnchangeableWorldAttribute, self).set(value)
            self._initialized = True


class SetWorldAttribute(WorldAttribute):
    """attribute that is constrained to one of a set of values"""
    _name = 'set_attribute'
    
    #a list of values that the attribute can have
    _values = None
    
    _state_attributes = ['_values']
    
    def __init__(self, world, values=None, **kwargs):
        self.values = values if values is not None else \
                        self._values if self._values is not None else \
                        [None]
        
        super(SetWorldAttribute, self).__init__(world, **kwargs)
    
    @property
    def values(self):
        return self._values
    
    @values.setter
    def values(self, values):
        if not isinstance(values, list): raise TypeError('values must be a list')
        if len(values) == 0: raise ValueError('values must be non-empty')
        
        self._values = values
    
    def set(self, value):
        """make sure the value is valid"""
        if value not in self.values:
            raise ValueError('%s is not a valid value' % (value,))
        
        super(SetWorldAttribute, self).set(value)
    
    def _default_value(self, world):
        return self.values[0]


class NonNegativeWorldAttribute(WorldAttribute):
    """attribute whose values must be non-negative"""
    _name = 'non_negative_attribute'
    
    _auto_state = False
    
    def _coerce_value(self, value, **kwargs):
        """add a check to ensure that values are >= 0"""
        valid = np.all(value >= 0)
        err = False if valid else TypeError('must be non-negative')
        return value, err


class PositiveWorldAttribute(WorldAttribute):
    """attribute whose values must be positive"""
    _name = 'positive_attribute'
    
    _auto_state = False

    def _coerce_value(self, value, **kwargs):
        """add a check to ensure that values are > 0"""
        valid = np.all(value > 0)
        err = False if valid else TypeError('must be positive')
        return value, err


class ScalarWorldAttribute(WorldAttribute):
    """attribute whose values must be scalars. 
    """
    _name = 'scalar_attribute'
    
    _dtype = np.dtype(int)

    _default_value = 0
    
    _auto_state = False



class IntegerWorldAttribute(ScalarWorldAttribute):
    """WorldAttribute whose values must be integers"""
    _name = 'integer_attribute'
    
    _dtype = np.dtype(int)
    
    _auto_state = False


class FloatWorldAttribute(ScalarWorldAttribute):
    """attribute that must be floats"""
    _name = 'float_attribute'
    
    _dtype = np.dtype(float)
    
    _auto_state = False


class NonNegativeIntegerWorldAttribute(IntegerWorldAttribute, NonNegativeWorldAttribute):
    """convenience class for non-negative integer attributes"""
    _name = 'non_negative_integer_attribute'
    
    _auto_state = False


class PositiveIntegerWorldAttribute(IntegerWorldAttribute, PositiveWorldAttribute):
    """convenience class for positive integer attributes"""
    _name = 'positive_integer_attribute'
    
    _auto_state = False


class NonNegativeFloatWorldAttribute(FloatWorldAttribute, NonNegativeWorldAttribute):
    """convenience class for non-negative float attributes"""
    _name = 'non_negative_float_attribute'
    
    _auto_state = False


class BooleanWorldAttribute(ScalarWorldAttribute):
    """attribute whose values must be bools"""
    _name = 'boolean_attribute'
    
    _dtype = np.dtype(bool)
    
    _default_value = True
    
    _auto_state = False


class StringWorldAttribute(WorldAttribute):
    """attribute whose values must be strings"""
    _name = 'string_attribute'

    # numpy dtype for string of length 255
    _dtype = np.dtype(str)
    
    _default_value = ''
    
    _auto_state = False


class HeightWorldAttribute(PositiveIntegerWorldAttribute, UnchangeableWorldAttribute):
    """height of the visible portion of the world"""
    def _default_value(self, world):
        return world.width if 'width' in world._world_attributes else 20


class WidthWorldAttribute(PositiveIntegerWorldAttribute, UnchangeableWorldAttribute):
    """width of the visible portion of the world"""
    def _default_value(self, world):
        return world.height if 'height' in world._world_attributes else 20


class DepthWorldAttribute(NonNegativeIntegerWorldAttribute, UnchangeableWorldAttribute):
    """number of depth planes in the world. larger depths are "under" smaller
    depths. if defined, then Object depth attributes must be integers in the
    interval [0, world.depth). if world.depth == 0, then any depth is valid."""
    pass


class TimeWorldAttribute(IntegerWorldAttribute):
    """current simulation time"""
    pass


class NameWorldAttribute(StringWorldAttribute):
    """name of the world"""
    
    _default_value = 'world'
    
    def set(self, *args, **kwargs):
        """update the window caption if a window is open"""
        super(NameWorldAttribute, self).set(*args, **kwargs)
        
        if self.world._viewer is not None:
            self.world.render()


class StateCenterWorldAttribute(WorldAttribute):
    """how the center of the world.state is determined. either:
        - None: unmodified
        - (y, x): the specified position
        - name: the position of the first object with specified name
    
    e.g.
        world.state_center = (10, 10)
            -> world.state is centered on position (10, 10)
        world.state_center = 'self'
            -> world.state is centered on the object with name 'self'
    """
    _default_value = None


class History(StateObject):
    """the history of execution (clearing, seeding, populating, and actions) in
    a world. this allows you to repeat an exact sequence of events in a world
    and obtain the same results.
    
    Examples
    --------
    history.replay(world)
    
    num_actions = len(history['action'])
    
    history['action'].replay(world)
    
    history[3:].replay(world)
    
    history.filter(type='action', action=lambda a: a in ['LEFT', 'RIGHT']).replay(world)
    
    to use, call like this:
        history.replay(world)
    and the world will be stepped through all history events in order.

    IMPORTANT NOTE: if you are restoring from an initial snapshot, you probably
    want to do history[3:].replay(world) or history['action'].replay(world)
    rather than history.replay(world). if you do history.replay(world),
    it will reinitialize the world.
    """

    #a list of items in the history, each of which is a dict
    _items = None
    
    #keep track of how many processes have blocked the history
    _block_count = 0
    
    _state_attributes = ['_items', '_block_count']

    def __init__(self, items=None):
        """Create a History object from the provided list of history items, or
        an empty History if no list is provided
        
        Parameters
        ----------
        items : list, optional
            list of items to include in the history
        """
        self._items = [] if items is None else [copy(item) for item in items]
    
    @property
    def items(self):
        """a list of history items"""
        return self._items
    
    @property
    def blocked(self):
        return self._block_count > 0
    
    def __repr__(self):
        return repr(self.items)
    
    def __len__(self):
        return len(self.items)
    
    def __eq__(self, other):
        return self.items == other.items

    def __getitem__(self, key):
        """get a history item or subhistory"""
        """Return either a history item (if an integer is passed in) or a new
        History object that includes the history items in the slice (if a slice
        is passed in).

        Parameters
        ----------
        key : int | slice | string
            for int, the history item to return. for slice, the subhistory to
            return. for string, the history item to for which to construct a
            subhistory
        """
        if isinstance(key, int):
            return self.items[key]
        elif isinstance(key, basestring):
            return self.filter(type=key)
        else:
            return self.__class__(items=self.items[key])
    
    def clear(self):
        """clear the history"""
        del self.items[:]
    
    def block(self):
        """call this to block adding new items to the history. this is useful if
        a process is starting that involves subprocesses that add their own
        history items, in which case replaying the history would otherwise end
        up replaying those subprocesses multiple times.
        
        make sure to call history.unblock() after the process is completed.
        """
        self._block_count += 1
    
    def unblock(self):
        """unblock a previous call to block()"""
        assert self.blocked
        self._block_count -= 1
    
    def append(self, item_type, **kwargs):
        """record an event
        
        Parameters
        ----------
        item_type : string
            the type of history item
        **kwargs
            other info to associate with the item
        """
        if not self.blocked:
            kwargs.update(type=item_type)
            
            self.items.append(kwargs)
    
    def append_clear(self):
        """record an clear event"""
        self.append('clear')
    
    def append_seed(self, seed):
        """record a seeding event

        Parameters
        ----------
        seed : int
            The seed that was used.
        """
        self.append('seed', seed=seed)
    
    def append_populate(self, entities):
        """record a populate event
        
        Parameters
        ----------
        entities : dict
            a dict of Entities (see PixelWorld._init_entities)
        """
        self.append('populate', entities=copy(entities))
    
    def append_randomize(self, seed):
        """record a Randomizer.randomize() event"""
        self.append('randomize', seed=seed)
    
    def append_action(self, action, agent_chose_action):
        """record an action

        Parameters
        ----------
        action : any
            the action to be recorded
        agent_chose_action : bool
            whether the action was chosen by the world agent (True) or was
            supplied to step() (False)
        """
        self.append('action', action=action, agent_chose_action=agent_chose_action)
    
    def filter(self, **kwargs):
        """filter the history for specific items
        
        Parameters
        ----------
        type : string | list
            an item type or list of item types to return
        **kwargs
            extra parameters that items must have and match in order to be
            included. each kwarg value is either a value that the candiate
            item's parameter value must be equal to, or a function that takes
            the parameter value and returns a boolean
        
        Return
        ------
        history : History
            a sub-history of only the matching items
        """
        items = [item for item in self.items if self._item_matches(item, **kwargs)]
        return self.__class__(items=items)
    
    def replay(self, world):
        """Replay history by repeating actions in order.
        
        IMPORTANT NOTE: if you are restoring from an initial snapshot, you
        probably want to do history[3:].replay(world) or
        history['action'].replay(world) rather than history.replay(world). if
        you do history.replay(world), it will reinitialize the world.

        Parameters
        ----------
        world : PixelWorld
            world in which to replay the history
        """
        for item in copy(self.items):
            item_type = item['type']
            
            if item_type == 'clear':
                world.clear()
            elif item_type == 'seed':
                world.seed(item['seed'])
            elif item_type == 'populate':
                world.populate(**item['entities'])
            elif item_type == 'randomize':
                world.randomizer.randomize(seed=item['seed'])
            elif item_type == 'action':
                if isinstance(item['agent_chose_action'], list):  # multi-agent mode
                    actions = [None if chosen else action for (action, chosen) in zip(
                                item['action'], item['agent_chose_action'])]
                    
                    world.step(actions)
                else:  # single agent mode
                    if item['agent_chose_action']:
                        world.step()
                    else:
                        world.step(item['action'])
            else:
                raise ValueError('"%s" is not a valid history item type' % (item_type))
    
    def most_recent_seed(self):
        """return the most recently set seed value, or None if no seed has been
        set"""
        for item in self.items[::-1]:
            if 'seed' in item: return item['seed']
        
        return None
    
    def _item_matches(self, item, **kwargs):
        """test whether a history item matches a set of criteria
        
        Parameters
        ----------
        **kwargs
            see filter()
        
        Returns
        -------
        match : bool
            True if the item matches
        """
        for key,value in kwargs.iteritems():
            if key == 'type':
                if not is_iterable(value): value = [value]
                
                if item['type'] not in value: return False
            elif key not in item or \
            (callable(value) and not value(item[key])) or \
            (not callable(value) and not item[key] == value):
                return False
        
        return True


#---PixelWorld-----------------------------------------------------------------#
class PixelWorld(gym.Env, StateObject):
    """PixelWorld is a grid-based OpenAI gym environment that blends physics
    simulation and arcade environment with flexible object and attribute
    behaviors. it can be thought of as an infinite 2D plane, of which a small
    window is visible via world.state.
    
    at creation time, the world is optionally populated with a set of Objects
    that each have a set of ObjectAttributes, and also a Judge, an Agent, and a
    Randomizer. Objects can be created and deleted later.
    
    the usual step(), render(), etc. OpenAI Gym methods are available. step()
    can behave slightly differently, though: if no action is specified, the
    world will query the Agent for one.
    
    here is an example simulation of 10 bouncing objects:
    ============================================================================
    from pixelworld.envs.pixelworld import PixelWorld
    
    objects = ['frame'] + 10 * [['basic', {'velocity': (1, 1)}]]
    judge = ['judge', {'end_time': 100}]
    world = PixelWorld(20, objects=objects, judge=judge)
    
    world.run(render=True)
    ============================================================================
    
    see the example environments in library/world/. a demo of each can be run
    via the command line: python demo.py <world_name>
        e.g. python demo.py pixelzuma

    Note that world._data is a Data object (which is a thin wrapper around a
    structured, masked numpy array), while world.data is a view of the array.
    """
    _class_family = 'world'
    
    _class_tree_attributes = ['_world_attributes']
    
    #the entities that were passed in either to __init__ or populate()
    _init_entities = None
    
    #a list of WorldAttributes that this world should have. WorldAttributes can
    #also be added by specifying their values in the PixelWorld constructor.
    #at world creation, this attribute will become a dict of WorldAttributes.
    _world_attributes = ['height', 'width', 'time', 'name']
    
    #debug object
    _debug = None
    
    #for saving world state snapshots
    _snapshots = None

    #history of actions taken
    _history = None
    
    #object structure
    _data = None
    _object_attributes = None
    _objects = None
    _object_types = None
    _num_objects = 0
    
    #keep track of attributes that step
    _stepping_object_attribute_order = None
    _stepping_object_attributes = None
    _randomizing_object_attributes = None
    
    #events
    _events = None
    _step_events = None
    
    #goals
    _goals = None
    
    #variants
    _variants = None
    
    #other entities
    _judge = None
    _randomizer = None
    _agent = None
    # id of next agent to move
    _agent_id = 0
    _multi_agent_mode = False
    
    #state of the world
    _populated = False
    _state = None
    _total_reward = None
    
    #observations
    _obs_type = None
    _observed_attributes = None
    _observed_objects = None
    _observed_window = None
    _last_observation = None
    
    #keep track of possible actions
    _base_actions = ['NOOP']
    _actions = None
    
    #for pseudorandomness
    _seed_value = None
    _rng = None
    
    #attributes for the gym environment
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    #for rendering and capturing
    _viewer = None
    _lut = None
    _render_type = None
    _render_kwargs = None
    _render_size = None
    _capture = None
    _capture_paths = None
    _capture_idx = 0
    
    #this specifies a list of attributes that should be included when saving the
    #world's state (see datatypes.State)
    _state_attributes = ['_init_entities', '_data', '_world_attributes',
        '_object_attributes', '_stepping_object_attribute_order',
        '_stepping_object_attributes',  '_randomizing_object_attributes',
        '_events', '_step_events', '_goals', '_variants', '_judge',
        '_randomizer', '_agent', '_agent_id', '_multi_agent_mode', '_populated',
        '_total_reward', '_obs_type', '_observed_attributes',
        '_observed_objects', '_actions', '_seed_value', '_num_objects',
        '_history']
    
    @classmethod
    def get_init_args(cls):
        """construct a aggregate list of the arguments that the class and its
        superclasses accept in the __init__() methods"""
        args = []
        
        #compile the super class arguments
        for base_cls in cls.__bases__:
            if hasattr(base_cls, 'get_init_args'):
                args.extend(base_cls.get_init_args())
        
        #add our own
        args.extend(inspect.getargspec(cls.__init__).args)
        
        #return the unique items
        return list(set(args))
    
    def __init__(self, seed=None, populate=True, objects=None, goals=None,
        judge='judge', variants=None, randomizer='randomizer', agent='agent',
        obs_type='state', observed_attributes=None, observed_objects=None,
        observed_window=None, render_type='state', render_size=None,
        capture=False, debug=None, **kwargs):
        """
        Important note: if you specify the agent parameter as a list with one
        element, like agent=['random'], then PixelWorld will operate in
        multi-agent mode, which may not be what you want. You can get around
        this by specifying e.g., agent='random', or agent=['random', {}].

        Parameters
        ----------
        height : int, optional
            height of the world's visible region, in pixels. defaults to width
            if present, otherwise 20.
        width : int, optional
            width of the world's visible region, in pixels. defaults to height
            if present, otherwise 20.
        seed : int, optional
            the random number seed to use
        populate : bool, optional
            True to populate the world
        objects : list, optional
            the initial set of Objects. see create_objects().
        goals : list, optional
            a list of Goals to associate with the world
        judge : string | list[string | type, dict] | type | Judge, optional
            the Judge to use (see Entity.get_instance)
        variants : list, optional
            a list of Variants to associate with the world
        randomizer : string | list[string | type, dict] | type | Randomizer, optional
            the Randomizer to use (see Entity.get_instance)
        agent : list of (string | list[string | type, dict] | type | Agent), or
                only one of (string | list[string | type, dict] | type | Agent) (optional)
            the Agent(s) to use (see Entity.get_instance). these Agent(s) will
            have absolutely no effect on the world if an action is always
            specified when calling step(). If you supply a list (even if it
            only has one element), PixelWorld will operate in multi-agent mode.
        obs_type : string, optional
            the observation type. one of the following:
                state: the observable scene state
                objects: a list of dicts of object attribute values
                data: a numpy MaskedArray of attribute data
                vector: a flattened numpy MaskedArray of attribute data
                list: a list version of a vector observation
            for observation types involving attribute values, the attributes
            included depend on the state_attributes keyword argument
        observed_attributes : list, optional
            for all but state observation type, a list of the names of
            ObjectAttributes to include in observations. if unspecified, all
            ObjectAttributes are included.
        observed_objects : list, optional
            for all but state observation type, a list of the names of Objects
            to include in observations. if unspecified, all Objects are
            included.
        observed_window : ndarray, optional
            for state observation, optionally specify the (h, w) size of the
            centered sub-window to observe. e.g. (3, 3) observes the central
            3 x 3 portion of the state.
        state_center : None | tuple[x, x] | string
            a reference point for the center of the state observation (see
            StateCenterWorldAttribute)
        render_type : string, optional
            the type of data to render. either 'state', 'data', or 'vector' (see
            obs_type above).
        render_size : int | tuple[int, int], optional
            the size, in pixels, of a box that should contain the rendered world
            state, or a tuple specifying the height and width of that box. if a
            tuple is specified, elements may be None to indicate no preference.
        capture : bool | string, optional
            if an animated gif of the simulation should be saved, specify so
            here. this should either be a boolean to indicate whether or not to
            capture to the default output path, a name if the file should use a
            custom prefix, or a file path to specify where to save the gif.
        debug : int | string, optional
            the debug level (see utils.Debug)
        **kwargs:
            the initial values of extra WorldAttributes to give to the world
        """
        #initialize some objects
        self._init_entities = {}
        self._snapshots = {}
        self._history = History()
        self._data = Data(data_change_callback=self._set_views)
        self._state = np.zeros((0, 0), dtype=np.uint32)
        self._object_attributes = {}
        self._stepping_object_attribute_order = []
        self._stepping_object_attributes = []
        self._randomizing_object_attributes = []
        self._events = []
        self._step_events = []
        self._goals = CustomOrderedDict()
        self._variants = CustomOrderedDict()
        self._agent = []
        self._actions = copy(self._base_actions)
        self._total_reward = np.zeros(0)
        
        #set up the debug object
        self._debug = Debug(level=debug)
        
        #observation info
        assert obs_type in ['state', 'objects', 'data', 'vector', 'list'], 'invalid obs_type'
        self._obs_type = obs_type
        self._observed_attributes = observed_attributes
        self._observed_objects = observed_objects
        self._observed_window = observed_window
        
        #info for rendering and capturing
        assert render_type in ['state', 'data', 'vector'], 'invalid render_type'
        self._render_type = render_type
        self._render_size = render_size
        self._capture = capture
        
        #clear the world
        self.clear()
        
        #set the rng seed
        self.seed(seed)
        
        #save the Entity information
        self._init_entities.update(
                world_attributes=self._parse_world_attributes(self._world_attributes, kwargs),
                objects=objects,
                goals=goals,
                judge=judge,
                variants=variants,
                randomizer=randomizer,
                agent=agent,
                )
        
        if populate:
            self.populate()
    
    @property
    def specifications(self):
        """construct a specification list similar to Entity.specifications"""
        spec_params = {}
        
        #get the __init__ parameters and their current values
        for arg in self.get_init_args():
            if arg == 'seed':
                spec_params[arg] = self._seed_value
            elif arg == 'debug':
                spec_params[arg] = self.debug.level
            elif arg in self._init_entities:
                entity = getattr(self, arg)
                
                spec = getattr(entity, 'specifications', None)
                
                if spec is None:
                    if is_iterable(entity):
                        if isinstance(entity, dict):
                            entity = entity.values()
                        
                        spec = [e.specifications for e in entity]
                    else:
                        spec = entity.specifications
                
                spec_params[arg] = spec
            elif arg not in ['self', 'populate']:
                if hasattr(self, '_' + arg):
                    spec_params[arg] = getattr(self, '_' + arg)
                elif hasattr(self, arg):
                    spec_params[arg] = getattr(self, arg)
        
        #add the world attribute values
        for name,attr in self.world_attributes.iteritems():
            spec_params[name] = attr.get()
        
        return [self.name, spec_params]
    
    @property
    def shape(self):
        """shape of the world's visible region"""
        return (self.height, self.width)
    
    @property
    def actions(self):
        """the current list of actions. this accounts for the possibility
        that the set of available actions might change as the simulation
        progresses, guaranteeing a stable action order.
        """
        #get a list of the current actions
        current_actions = copy(self._base_actions)
        for name in self._stepping_object_attributes:
            actions = self._object_attributes[name].actions
            for action in actions:
                #append new actions to the current list
                if not action in current_actions:
                    current_actions.append(action)
                
                #append new actions to the master list
                if not action in self._actions:
                    self._actions.append(action)
        
        #return current actions in the order of the master list
        return [action for action in self._actions if action in current_actions]
    
    @property
    def rng(self):
        """a numpy.random.RandomState object that all pseudorandomness should go
        through"""
        return self._rng
    
    @property
    def seed_value(self):
        """the seed value for the current rng"""
        return self._seed_value
    
    @property
    def tools(self):
        """tools to provide an agent when making action decisions"""
        return {
            'actions': self.actions,
            'rng': self.rng,
            }
    
    @property
    def populated(self):
        """becomes True after the world is populated during creation"""
        return self._populated
    
    @property
    def snapshots(self):
        """a dict of world State snapshots"""
        return self._snapshots

    @property
    def history(self):
        """World's History object, see History documentation"""
        return self._history
    
    @property
    def data(self):
        """A view of the master data array that holds all Object and ObjectAttribute
        data. Note that world._data is a Data object (which is a thin wrapper
        around a structured, masked numpy array)."""
        return self._data[:self._num_objects]
    
    @property
    def world_attributes(self):
        """a dict of all WorldAttributes"""
        return self._world_attributes
    
    @property
    def object_attributes(self):
        """a dict of all ObjectAttributes"""
        return self._object_attributes

    @property
    def num_objects(self):
        return self._num_objects
    
    @property
    def objects(self):
        """a numpy object array of all Objects"""
        return self._objects[:self._num_objects]
    
    @property
    def object_types(self):
        """a numpy array of each Object's class_type (see TypeFamily). this may
        only ever be used to distinguish SIMPLE from COMPOUND Objects"""
        return self._object_types[:self._num_objects]
    
    @property
    def events(self):
        """a list of all Events that have ever occurred"""
        return self._events
    
    @property
    def step_events(self):
        """a list only of Events that have occurred since the start of the last
        step"""
        return self._step_events
    
    @property
    def goals(self):
        """a list of all Goals in the world"""
        return self._goals
    
    @goals.setter
    def goals(self, goals):
        """goals can be either a list or a dict of goal specs. if dict, then the
        dict keys are ignored in favor of the goal names."""
        if goals is None:
            goals = []
        elif isinstance(goals, dict):
            goals = goals.values()
        
        #get an instance of each goal
        goals = [Goal.get_instance(self, goal) for goal in goals]
        
        #remove the old goals
        for goal in copy(self._goals.values()):
            if goal not in goals:
                goal.remove()
        
        #set the new goals
        self._goals.clear()
        for goal in goals:
            self._goals[goal.name] = goal
    
    @property
    def active_goals(self):
        """a list of just the active Goals"""
        return self.get_goals(active=True)
    
    @property
    def variants(self):
        """a list of all Variants in the world"""
        return self._variants
    
    @variants.setter
    def variants(self, variants):
        """variants can be either a list or a dict of variant specs. if dict,
        then the dict keys are ignored in favor of the variant names."""
        if variants is None:
            variants = []
        elif isinstance(variants, dict):
            variants = variants.values()
        
        #get an instance of each variant
        variants = [Variant.get_instance(self, variant) for variant in variants]
        
        #remove the old variants
        for variant in copy(self._variants.values()):
            if variant not in variants:
                variant.remove()
        
        #set the new variants
        self._variants.clear()
        for variant in variants:
            self._variants[variant.name] = variant
    
    @property
    def active_variants(self):
        """a list of just the active Variants"""
        return self.get_variants(active=True)
    
    @property
    def god(self):
        return np.nan
    
    @god.setter
    def god(self, god):
        raise AttributeError("can't set god")
    
    @property
    def judge(self):
        """the world's Judge"""
        return self._judge
    
    @judge.setter
    def judge(self, judge):
        """set the world's Judge"""
        Judge.get_instance(self, judge)
    
    @property
    def randomizer(self):
        """the world's Randomizer"""
        return self._randomizer
    
    @randomizer.setter
    def randomizer(self, randomizer):
        """set the world's Randomizer"""
        Randomizer.get_instance(self, randomizer)
    
    @property
    def agent(self):
        """the world's Agent(s). the agent will never have any effect if an action is
        always specified when calling step(). When in multi-agent mode, this
        will return a list."""
        if self._multi_agent_mode:
            return self._agent
        else:
            assert len(self._agent) <= 1
            if len(self._agent) == 0:
                return None
            return self._agent[0]
    
    @agent.setter
    def agent(self, agent):
        """set the world's Agent(s)"""
        if agent is None:  # clear all agents
            multi_agent_mode = self._multi_agent_mode
            agent = []
        else:
            #we're in multi-agent mode if a list of agents was passed
            multi_agent_mode = is_iterable(agent) and (len(agent) != 2 or not isinstance(agent[1], dict))
        
            #make sure we have a list of agents
            agent = list(agent) if multi_agent_mode else [agent]
        
        #get an instance of each agent
        agent = [Agent.get_instance(self, agt) for agt in agent]
        
        #remove the old agents
        for agt in self._agent:
            if agt not in agent:
                agt.remove()
        
        #set the new agents
        del self._agent[:]
        for agt in agent:
            self._agent.append(agt)
        
        #set the multi agent mode
        self._multi_agent_mode = multi_agent_mode
        
        #make sure the total reward length matches
        num_reward = len(self._total_reward)
        num_agent = len(self._agent)
        if num_reward < num_agent:
            self._total_reward = np.concatenate([self._total_reward, np.zeros(num_agent - num_reward)])
        elif num_agent < num_reward:
            self._total_reward = self._total_reward[:num_agent]
    
    @property
    def multi_agent_mode(self):
        """True if we are operating in multi-agent mode, False otherwise. Note that we
        can be operating in multi-agent mode with a single agent, in which case
        the step() function expects to receive a list or tuple of actions of
        length 1.
        """
        return self._multi_agent_mode
    
    @property
    def state(self):
        """the current state of the visible portion of the world. this just
        encodes each Object's position and color as an integer numpy array. this
        is the observation when obs_type == 'state'."""
        self._state[:] = 0
        
        center = self.state_center
        
        #position determines the observed world state
        position = self._object_attributes.get('position', None)
        if position is not None:
            #get the indices of the objects to show
            if center is None:
                #get the objects that currently affect the state
                idx, loc = position.get_visible_indices()
            else:
                if isinstance(center, basestring):  # object name
                    obj = self.objects.get(name=center)
                    if obj is None:
                        raise RuntimeError('no object with name "%s" exists' % (center,))
                    
                    center = obj.position
                elif not isinstance(center, np.ndarray) or center.size != 2:
                    raise RuntimeError('invalid state_center')
                
                #get all objects in order of appearance
                idx, _ = position.get_visible_indices(within_window=False)
                
                #get all positions relative to the reference positions
                p = position.get(idx) - center + np.array(self.shape)/2
                
                #keep the visible objects
                is_visible = np.all(((0, 0) <= p) & (p < self.shape), axis=1)
                p = p[is_visible]
                idx = idx[is_visible]
                
                loc = position._position_to_index(p)
            
            #get the colors of the visible objects
            color = self._object_attributes.get('color', None)
            if color is None:
                c = 1
            else:
                c = color.get(idx)
                
                #monochrome!
                if self.has_world_attribute('show_colors') and not self.show_colors:
                    c[c != 0] = 1
        
            #show the visible simple objects
            self._state[loc] = c
        
        if self._observed_window is None:
            return self._state
        else:
            return get_centered_subarray(self._state, self._observed_window)
    
    @property
    def object_state(self):
        """a list of dicts of the value of every observed Object's observed
        attribute values, one dict per Object. this is the observation when
        obs_type == 'objects'."""
        if self._observed_objects is None:
            return [obj.state for obj in self.objects]
        else:
            return [obj.state for obj in self.objects if obj._name in self._observed_objects]
    
    @property
    def data_state(self):
        """world.object_state encoded as a num_objects x (sum of attribute dimensions)
        masked numpy array. elements representing the attribute value of an
        Object that does not have that attribute will be masked. just
        non-masked values can be retrieved with data_state.compressed(). this
        is the observation when obs_type == 'data'."""
        num_objects = self._data.shape[0]
        
        if num_objects == 0:
            return ma.empty((0, 0))
        
        #construct the full data array of observed attributes
        attributes = self._observed_attributes or sorted(self._object_attributes.keys())
        for name in attributes:
            assert len(self._object_attributes[name].data) == self._num_objects
        data = [ma.reshape(self._object_attributes[name].data, 
                           (self._num_objects,self._object_attributes[name]._ndim)) for name in attributes]
        data = ma.hstack(data)
        
        #keep only the observed objects
        if self._observed_objects is not None:
            indices = [obj._id for obj in self.objects if obj._name in self._observed_objects]
            data = data[indices]
        
        return data
    
    @property
    def vector_state(self):
        """a compressed, flattened (1D vector) version of world.data_state. this
        is the observation when obs_type == 'vector'."""
        return self.data_state.compressed().flatten()
    
    @property
    def list_state(self):
        """world.vector_state as a list"""
        return list(self.vector_state)
    
    @property
    def world_state(self):
        """a dict of all world attribute values"""
        return {name:attr.get() for name,attr in self._world_attributes.iteritems()}
    
    @property
    def obs_type(self):
        """the observation type"""
        return self._obs_type
    
    @property
    def last_observation(self):
        """the last observation that was made"""
        if self._last_observation is None:
            self.observe()
        
        return self._last_observation
    
    @property
    def total_reward(self):
        """total reward received by the agent(s) thus far. Will be an array if
        there are multiple agents."""
        if self._multi_agent_mode:
            return self._total_reward
        else:
            return self._total_reward[0]
    
    @property
    def energy(self):
        """the total energy of the world. used for debugging."""
        E = 0
        
        for obj in self.objects:
            if not hasattr(obj, 'children'):
                for name in obj._attributes:
                    if name.endswith('energy'):
                        E += obj._get(name)
        
        return E
    
    @property
    def observation_space(self):
        """compute the observation space based on the observation type. computed
        dynamically in case the observation space changes."""
        if self._obs_type == 'objects':
            return ObjectStateSpace()
        elif self._obs_type == 'list':
            return ListSpace(self, self.observe())
        else:
            if self._obs_type == 'state':
                min_value = 0
                max_value = np.iinfo(int).max
            else:
                min_value = float('-inf')
                max_value = float('inf')

            return spaces.Box(low=min_value, high=max_value,
                                shape=self.observe().shape)
    
    @property
    def action_space(self):
        """compute the space of actions that can be taken on each step,
        depending on the number of agents and the available actions. computed
        dynamically because the action space can change during the episode."""
        actions = self.actions
        
        if self.multi_agent_mode:
            return MultiListSpace(self, actions, len(self.agent))
        else:
            return ListSpace(self, actions)
    
    @property
    def reward_range(self):
        """range of rewards produced by each step"""
        return (-np.inf, np.inf)
    
    @property
    def viewer(self):
        """the rendering viewer object"""
        return self._viewer
    
    @property
    def render_size(self):
        """size, in pixels, of the rendered image"""
        return self._render_size
    
    @render_size.setter
    def render_size(self, sz):
        """set the size of the rendered image
        
        Parameters
        ----------
        sz : int | tuple[int, int]
            the size, in pixels, of a box that should contain the rendered world
            state, or a tuple specifying the height and width of that box. if a
            tuple is specified, elements may be None to indicate no preference.
        """
        if sz is None or sz == (None, None):
            sz = (None, 300)
        elif isinstance(sz, Number):
            sz = (sz, sz)
        
        assert isinstance(sz, tuple) and len(sz) == 2, 'invalid render size'
        
        if sz[0] is None:
            sz = (round(float(sz[1]) * self.height / self.width), sz[1])
        elif sz[1] is None:
            sz = (sz[0], round(float(sz[0]) * self.width / self.height))
        
        self._render_size = tuple([int(s) for s in sz])
    
    @property
    def render_location(self):
        """(left, top) location, in pixels, of the rendering window"""
        if self._viewer is not None:
            return self.viewer.window.get_location()
        else:
            return None, None
    
    @render_location.setter
    def render_location(self, loc):
        """set the location of the rendering window
        
        Parameters
        ----------
        loc : tuple[int, int]
            the new (left, top) location of the rendering window
        """
        if self._viewer is not None:
            self.viewer.window.set_location(*loc)
            self.render(capture=False)
    
    @property
    def render_location_centered(self):
        """(x, y) of the center of the window, relative to the center of the
        screen"""
        if self._viewer is not None:
            #screen dimensions
            ws = self.viewer.window.screen.width
            hs = self.viewer.window.screen.height
            
            #window dimensions
            ww = self.viewer.window.width
            hw = self.viewer.window.height
            left, top = self.render_location
            
            x = left - (ws - ww) / 2
            y = top - (hs - hw) / 2
            
            return x, y
        else:
            return None, None
    
    @render_location_centered.setter
    def render_location_centered(self, xy):
        """set the center-based location of the rendering window
        
        Parameters
        ----------
        xy : tuple[int, int]
            the new (x, y) center-based location of the rendering window
        """
        if self._viewer is not None:
            #screen dimensions
            ws = self.viewer.window.screen.width
            hs = self.viewer.window.screen.height
            
            #window dimensions
            ww = self.viewer.window.width
            hw = self.viewer.window.height
            
            left = xy[0] + (ws - ww) / 2
            top = xy[1] + (hs - hw) / 2
            
            self.render_location = (left, top)
    
    @property
    def debug(self):
        """the utils.Debug object used for printing debug messages"""
        return self._debug
    
    def step(self, action=None):
        """advance the world forward one step.
        
        Parameters
        ----------
        action : string or list of string, optional
            the action to execute. if no action is specified, the world queries
            the world's agent(s) for an action. if any agent still returns
            None, then the episode is aborted. this is mainly so the human
            agent can abort at any point. If there are multiple agents, this
            should be a list with one string per agent
        
        Returns
        -------
        obs : list | ndarray
            an observation of the world after all the actions are executed.
        reward : float or numpy array
            the reward received as a result of the step. This will be a numpy
            array in multi-agent mode.
        done : bool
            True if the episode is done
        info : dict
            a dict of info. currently only contains 'aborted', which indicates
            whether the agent indicated None as an action. In multi-agent mode,
            this will be a dict keyed on agent id containing one dict for each
            agent
        """
        #get the action(s) to perform
        if len(self._agent) == 1:
            if isinstance(action, basestring):
                action = [action]
            elif action == None:
                action = [None]
            else:
                assert isinstance(action, list) or isinstance(action, tuple)
        elif action is None:
            action = [None] * len(self._agent)
        else:
            assert is_iterable(action), "Action must be iterable in multi-agent mode"
            assert len(action) == len(self._agent), "Incorrect number of actions specified"
            assert not isinstance(action, str), "Action should not be a string when there are multiple agents"
        
        actions_taken, agent_chose_actions = [], []
        
        #add the history item here so other Entities have access to it as it is
        #filled in
        self._history.append_action(actions_taken, agent_chose_actions)
        
        rewards = np.zeros(len(self._agent))
        infos = dict()
        for id, (agent, agent_action) in enumerate(zip(self._agent, action)):
            agent_chose_action = False
            if agent_action is None:
                agent_chose_action = True
                agent_action = agent.get_action(self.last_observation, self.tools)

                #if the agent still returns None, we abort
                if agent_action is None:
                    if self._multi_agent_mode:
                        reward_rv = rewards
                        infos = defaultdict(dict, infos)
                        for i in xrange(self._agent_id):
                            infos[i].update({'aborted': True})
                        infos = dict(infos)
                    else:
                        assert len(self._agent) == 1
                        reward_rv = rewards[0]
                        infos = {'aborted': True}
                    return None, reward_rv, True, infos

            actions_taken.append(agent_action)
            agent_chose_actions.append(agent_chose_action)

            # call gym environment step() function
            self._agent_id = id
            obs, reward, done, info = super(PixelWorld, self).step(agent_action)
            rewards[id] = reward
            infos[id] = info
            if done:
                break
        
        if not self._multi_agent_mode:
            assert len(self._agent) == 1
            return obs, rewards[0], done, infos[0]
        else:
            return obs, rewards, done, infos
    
    def run(self, num_steps=None, render=None, end=None):
        """run the stimulation for num_steps steps or until a step returns
        done == True
        
        Parameters
        ----------
        num_steps : int, optional
            the number of steps to execute. if unspecified, runs until done ==
            True.
        render : bool, optional
            True to render the world at each step. defaults to True if the agent
            is human.
        end : bool, optional
            True to end the world after the run. defaults to False if num_steps
            is specified or True if it isn't. if the simulation is aborted, the
            world is not ended.
        """
        #default to rendering if we have a human agent
        if render is None:
            render = 'human' in [agent._name for agent in self._agent]
        
        #make sure we get captured images if they were specified
        if render:
            render = 'human'
        elif self._capture_paths:
            render = 'rgb_array'
        
        if end is None:
            end = num_steps is None
        
        if num_steps is None:
            num_steps = np.inf
        
        #render the initial scene
        if render:
            self.render(mode=render)
            self.bring_to_top()
        
        #run the simulation
        done = False
        step = 0
        while not done and step < num_steps:
            obs, reward, done, info = self.step()
            
            if render:
                self.render(mode=render)
            
            step += 1
        
        #end the world
        if end and not info['aborted']:
            self.end()
    
    def end(self):
        """end the world!
        
        Returns
        -------
        capture_path : string | None
            if the simulation was captured, the output gif path
        """
        return self.close()
    
    def close(self):
        """override the default gym Environment close so we can return the path
        to the captured gif
        
        Returns
        -------
        capture_path : string | None
            if the simulation was captured, the output gif path
        """
        if self._capture_paths:
            capture_path = self.save_gif()
        else:
            capture_path = None
        
        super(PixelWorld, self).close()
        
        return capture_path
    
    def clear(self):
        """depopulate the world of all Entities, and reset the world back to the
        beginning of time"""
        #depopulate the world
        self.depopulate()

        #clear the world state
        self._state = np.zeros((0, 0), dtype=np.uint32)
        if self._actions is not None:
            del self._actions[:]
            for action in self._base_actions:
                self._actions.append(action)
        self._total_reward = np.zeros(0)
        if self._snapshots is not None: self._snapshots.clear()
        if self._history is not None:
            self._history.clear()
        
            #append a clear event to the history
            self._history.append_clear()
        
        #add fields for object info
        if self._data is not None:
            self._data.add_field('object', Object, None)
            self._data.add_field('object_type', int, 0)
    
    def populate(self, **kwargs):
        """populate the world with Entities. this probably shouldn't be called
        unless the world is empty to begin with.
        
        Parameters
        ----------
        **kwargs
            Entities to override from the values determined in __init__()
        
        Returns
        -------
        obs : (see world.obs_type)
            an initial observation
        """
        self._init_entities.update(**kwargs)
        
        #append a populate event to the history
        self.history.append_populate(self._init_entities)
        
        #block the history until we're finished populating
        self.history.block()
        
        #create the world attributes
        self._add_world_attributes(self._init_entities['world_attributes'])
        
        #set some rendering info
        self.render_size = self.render_size  # so render size gets parsed
        self._set_capture_paths(self._capture)
        
        #initialize the state
        self._state = np.zeros((self.height, self.width), dtype=np.uint32)
        
        #create the objects
        objects = self._init_entities['objects']
        if objects is not None: self.create_objects(objects)
        
        #create the goals
        self.goals = self._init_entities['goals']
        
        #create the judge
        self.judge = self._init_entities['judge']

        #create the variants
        self.variants = self._init_entities['variants']
        
        #create the randomizer
        self.randomizer = self._init_entities['randomizer']
        
        #create the agent(s)
        self.agent = self._init_entities['agent']
        
        self._populated = True
        
        #now that everything is ready, call each entity's prepare() method
        self._prepare_entities()
        
        #display some debug info
        if self.debug.test(Debug.INFO):
            self.debug.log(ENERGY=self.energy, level=Debug.INFO)
        
        #temporarily unblock the history so the snapshot state is correct
        self.history.unblock()
        
        #save the initial state
        self.save_snapshot('initial')
        
        #block the history again
        self.history.block()
        
        #run post-reset processes
        self._post_reset()
        
        #unblock the history
        self.history.unblock()
    
    def depopulate(self):
        """remove all Entities from the world"""
        self._populated = False
        
        #clear the events
        if self._events is not None: del self._events[:]
        if self._step_events is not None: del self._step_events[:]
        
        #remove the agent(s)
        if self._agent is not None: del self._agent[:]
        self._multi_agent_mode = False
        self._total_reward = np.zeros(0)
        
        #remove the randomizer
        self._randomizer = None
        
        #clear the variants
        if self._variants is not None: self._variants.clear()
        
        #remove the judge
        self._judge = None
        
        #clear the goals
        if self._goals is not None: self._goals.clear()
        
        #clear all objects and object attributes
        if self._object_attributes is not None: self._object_attributes.clear()
        if self._stepping_object_attributes is not None: del self._stepping_object_attributes[:]
        if self._stepping_object_attribute_order is not None: del self._stepping_object_attribute_order[:]
        if self._randomizing_object_attributes is not None: del self._randomizing_object_attributes[:]
        if self._data is not None: self._data.null()
        self._num_objects = 0
        
        #clear all world attributes
        if isinstance(self._world_attributes, dict):  # world attributes have already been parsed
            self._world_attributes.clear()
    
    def save_snapshot(self, name):
        """save a named snapshot of the current world state
        
        Parameters
        ----------
        name : string
            a name to give to the snapshot
        
        Returns
        -------
        state : datatypes.State
            the saved State
        """
        state = State(self)
        
        self._snapshots[name] = state
        
        return state

    def load_snapshot(self, name):
        """load a previously saved snapshot
        
        Parameters
        ----------
        name : string
            the snapshot name
        """
        self._snapshots[name].restore()

    def observe(self):
        """observe the world. the form this takes depends on world.obs_type,
        which is set during world creation."""
        if self._obs_type == 'state':
            obs = self.state
        else:
            obs = getattr(self, '%s_state' % (self._obs_type))
        
        self._last_observation = obs
        return obs
    
    def has_world_attribute(self, name):
        """test whether the world has a WorldAttribute. this is implemented
        because hasattr(world, name) will always create the WorldAttribute if it
        doesn't exist, and return True.
        
        Parameters
        ----------
        name : string
            the name of the WorldAttribute to test for
        
        Returns
        -------
        b : bool
            True if the world has the specified WorldAttribute
        """
        return name in self._world_attributes
    
    def create_object(self, obj):
        """create an instance of an Object
        
        Parameters
        ----------
        obj : string | type | Object | list[string | type, dict]
            an Object name, class, or instance, or a one or two element list
            consisting of the Object class/name and optionally a dict of initial
            attribute values (e.g. ['self', {'color': 2}])
        
        Returns
        -------
        obj : Object
            the newly created Object
        """
        return Object.get_instance(self, obj)
    
    def create_objects(self, objects):
        """create multiple objects at once.
        
        Parameters
        ----------
        objects : list
            a list of Object specifications (see create_object())
        
        Returns
        -------
        objects : list[Object]
            a list of newly created Objects
        """
        if objects is None:
            return []
        
        #in case create_objects() is called like create_object()
        if not isinstance(objects, list) or (len(objects) == 2 and \
        isinstance(objects[1], dict)):
            objects = [objects]
        
        #reserve data rows for all of the objects
        self._data.add_rows(len(objects))
        
        return [self.create_object(obj) for obj in objects]
    
    def remove_objects(self, objects):
        """remove Objects from the world
        
        Parameters
        ----------
        objects : Object | string | list[Object | string]
            an Object or Object name, or list of Objects or Object names. if
            names are specified, then all Objects with that name are removed.
        """
        objects = to_iterable(objects)
        
        for obj in objects:
            if isinstance(obj, Object):
                obj.remove()
            elif isinstance(obj, basestring):
                self.remove_objects(self.objects(name=obj))
            else:
                raise TypeError('"%s" is not a valid object' % (obj))
        self._num_objects = len(self.objects)
    
    def get_goals(self, **kwargs):
        """return a list of Goals, possibly depending on a set of criteria
        
        Parameters
        ----------
        **kwargs
            use to specify filtering criteria for the list of Goals to return.
            each argument is the name of a Goal attribute, and its value is one
            of the following:
                value: the required value of the attribute
                f: a function that takes the Goal and the attribute value as
                inputs and returns a boolean to indicate whether to include the
                Goal.
            Goals must satisfy all of the criteria in order to be included.
        
        Returns
        -------
        goals : CustomOrderedDict
            a dict of Goals that matched the criteria
        
        Example
        --------
        goals = world.get_goals(terminates=True, active=True)
        """
        #start with all goals
        goals = copy(self.goals)
        
        #apply each filter
        for attr,value in kwargs.iteritems():
            for name,goal in copy(goals).iteritems():
                if not hasattr(goal, attr):
                    keep = False
                else:
                    attr_value = getattr(goal, attr)
                    
                    if callable(value):
                        keep = value(goal, attr_value)
                    else:
                        keep = attr_value == value
                
                if not keep:
                    del goals[name]
        
        return goals
    
    def get_variants(self, **kwargs):
        """return a list of Variants, possibly depending on a set of criteria
        
        Parameters
        ----------
        **kwargs
            use to specify filtering criteria for the list of Variants to
            return. each argument is the name of a Variant attribute, and its
            value is one of the following:
                value: the required value of the attribute
                f: a function that takes the Variant and the attribute value as
                inputs and returns a boolean to indicate whether to include the
                Variant.
            Variants must satisfy all of the criteria in order to be included.
        
        Returns
        -------
        variants : CustomOrderedDict
            a dict of Variants that matched the criteria
        
        Example
        --------
        variants = world.get_variants(active=True)
        """
        #start with all variants
        variants = copy(self.variants)
        
        #apply each filter
        for attr,value in kwargs.iteritems():
            for name,variant in copy(variants).iteritems():
                if not hasattr(variant, attr):
                    keep = False
                else:
                    attr_value = getattr(variant, attr)
                    
                    if callable(value):
                        keep = value(variant, attr_value)
                    else:
                        keep = attr_value == value
                
                if not keep:
                    del variants[name]
        
        return variants
    
    def get_world_attribute(self, attr):
        """get a WorldAttribute object. using this ensures that no more than one
        instance of each WorldAttribute is created.
        
        Parameters
        ----------
        attr : WorldAttribute | type | string | list[type | string, dict]
            a WorldAttribute name, class, or instance, or a one or two element
            list specifying an WorldAttribute name or class and a dict of
            keyword arguments to pass to its constructor
        
        Returns
        -------
        attr : WorldAttribute
            the WorldAttribute instance, ensuring that no more that one of each
            WorldAttribute class is created.
        """
        #make sure we have only one instance of each world attribute
        attr_spec = attr[0] if isinstance(attr, list) else attr
        if isinstance(attr_spec, basestring) and attr_spec in self._world_attributes:
            #WorldAttribute already exists
            return self._world_attributes[attr_spec]
        else:
            return WorldAttribute.get_instance(self, attr)
    
    def get_object_attribute(self, attr):
        """get an ObjectAttribute object. using this ensures that no more than
        one instance of each ObjectAttribute is created.
        
        Parameters
        ----------
        attr : ObjectAttribute | type | string | list[type | string, dict]
            an ObjectAttribute name, class, or instance, or a one or two element
            list specifying an ObjectAttribute name or class and a dict of
            keyword arguments to pass to its constructor
        
        Returns
        -------
        attr : ObjectAttribute
            the ObjectAttribute instance, ensuring that no more that one of each
            ObjectAttribute class is created.
        """
        #make sure we have only one instance of each object attribute
        attr_spec = attr[0] if isinstance(attr, list) else attr
        if isinstance(attr_spec, basestring) and attr_spec in self._object_attributes:
            #ObjectAttribute already exists
            return self._object_attributes[attr_spec]
        else:
            return ObjectAttribute.get_instance(self, attr)
    
    def render(self, capture=True, **kwargs):
        """adds capture mechanism to the base gym render() method
        
        Parameters
        ----------
        capture : bool, optional
            True to capture the rendered image to file (only if capturing is
            enabled)
        """
        #this is a hack since gym's render() doesn't support extra keyword arguments
        self._render_kwargs = {
            'capture': capture
        }
        
        return super(PixelWorld, self).render(**kwargs)
    
    def show(self):
        """Show an image of the visible region of the screen. (Opens in Preview on
        Mac.)
        """
        pic = self.render(capture=False, mode='rgb_array')
        toimage(pic).show()
    
    def bring_to_top(self):
        """bring the rendering window to the top"""
        self.viewer.window.activate()
        self.render(capture=False)
    
    def capture(self, im=None, path=None):
        """capture a state image to a file
        
        Parameters
        ----------
        im : ndarray | pyglet.image.ImageData, optional
            the image to capture. if unspecified, uses the current state image.
        path : string, optional
            the output image path
        
        Returns
        -------
        path : string
            the output image path
        """
        check_gui_exists()

        if im is None:
            im = self._get_render_image()
        
        if path is None:
            #get the path for the current capture image
            path = self._get_capture_temp_path()
        
        if isinstance(im, np.ndarray):
            imsave(path, im)
        elif isinstance(im, pyglet.image.ImageData):
            im.save(path)
        else:
            raise TypeError('invalid image type')
        
        return path
    
    def save_gif(self):
        """save a gif of the captured simulation (see capture in __init__)
        
        Returns
        -------
        output_path : string | None
            the path to the saved gif, or None if no gif was saved (either
            because it was already saved, or the simulation wasn't captured)
        """
        #construct the output path
        output_path = self._get_capture_output_path()
        if output_path is None:
            return None
        
        #make sure the temp directory still exists
        temp_dir = self._capture_paths['temp_dir']
        if temp_dir is None or not os.path.isdir(temp_dir):
            return None
        
        #make sure we have gifs to save
        if len(os.listdir(temp_dir)) > 0:
            #construct the gif conversion command
            cmd = 'convert -delay 10 -loop 0 -deconstruct -depth 8 "%s/*.png" %s' % (
                        temp_dir, output_path)
            
            #execute it
            exit_code = os.system(cmd)
            if exit_code:
                raise RuntimeError('gif saving failed with code: %d' % (exit_code))
        else:
            output_path = None
        
        #delete the capture directory
        shutil.rmtree(temp_dir)
        self._capture_paths['temp_dir'] = None
        
        return output_path
    
    def __dir__(self):
        """add the world's WorldAttributes to the dir() list"""
        #get class attributes
        d = set(dir(self.__class__))
        
        #add instance dict attributes
        d.update(self.__dict__.keys())
        
        #add the WorldAttributes
        d.update(self._world_attributes.keys())
        
        return sorted(d)
    
    def __getattr__(self, name):
        """overriding this method allows the world's WorldAttributes to be
        retrieved like a normal python attribute (e.g. h = world.height)
        """
        try:
            if name.startswith('__') or not isinstance(self._world_attributes, dict)    :
                return self.__getattribute__(name)
            else:
                return self._world_attributes[name].get()
        except KeyError:  # world attribute doesn't exist
            try: # try to create the world attribute
                attr = self.get_world_attribute(name)
                return attr.get()
            except ValueError: # no world attribute with that name
                return self.__getattribute__(name)
    
    def __setattr__(self, name, value):
        """overriding this method allows the world's WorldAttributes to be
        assigned like a normal python attribute (e.g. world.height = 10).
        """
        if name not in WorldAttribute._class_family:
            #we have to allow any attribute to be set, since gym sets attributes
            #(like 'spec') that are not in the class definition
            super(PixelWorld, self).__setattr__(name, value)
        else:
            self.get_world_attribute(name).set(value)
    
    def __deepcopy__(self, memo):
        """deepcopy everything except the viewer, since it involves ctypes

        Parameters
        ----------
        memo : dict
            Memoization dict
        """
        #make a new PixelWorld
        pw = self.__class__.__new__(self.__class__)
        memo[id(self)] = pw
        
        #deepcopy everything except the viewer
        for attr,value in self.__dict__.iteritems():
            if attr != '_viewer':
                value_id = id(value)
                new_value = memo[value_id] if value_id in memo else deepcopy(value, memo)
                setattr(pw, attr, new_value)

        # make sure that every attribute has _world set, even if not finished
        # copying, so that it's ready for _process_data_change()
        for attr in pw.object_attributes:
            pw.object_attributes[attr]._world = pw

        # call _process_data_change() to reset all the views into the master
        # array
        pw._data._process_data_change()
        
        return pw
    
    def _step(self, action):
        """OpenAI gym's step() calls this.
        
        Parameters
        ----------
        action : string
            the action to execute
        
        Returns
        -------
        obs : list | ndarray
            an observation of the world after the action is executed
        reward : float
            the reward received as a result of the step
        done : bool
            True if the episode is done
        info : dict
            a dict of info. currently only contains 'aborted', which indicates
            whether the agent indicated None as an action.
        """
        #clear the step event list
        del self._step_events[:]
        
        #current time info
        t = self.time
        dt = 1
        
        self.debug.log('action %s at time %.3f' % (action, t), level=Debug.INFO, hr='*')
        
        #step each stepping attribute
        for name in self._stepping_object_attributes:
            self._object_attributes[name].step(t, dt, self._agent_id, action)
        
        #increment the world time
        self.time += dt
        
        #compute the step results
        reward, done = self.judge.process_step()
        self._total_reward[self._agent_id] += reward

        #tell the agent to process the results
        self._agent[self._agent_id].process_action_result(self.last_observation, action, reward, done)
        
        #show some debug info
        if self.debug.test(Debug.INFO):
            event_descriptions = {idx:event.description
                                  for idx,event in enumerate(self._step_events)}
            
            self.debug.log(ENERGY=self.energy, level=Debug.INFO)
            self.debug.log('EVENTS', event_descriptions, level=Debug.INFO)
            self.debug.log(REWARD=reward, level=Debug.INFO)
            self.debug.log(DONE=done, level=Debug.INFO)

        #do post-step stuff if any
        self._post_step()
                
        return self.observe(), reward, done, {'aborted': False}

    def _post_step(self):
        """Function that runs after world has been stepped but before agent gets
        observation. Override this function to implement custom logic."""
        pass
    
    def _reset(self):
        """OpenAI gym's step() calls this. reset the world to the state it was
        in at the end of creation, then randomize.
        """
        #remember the old seed value
        seed_value = self.seed_value
        
        #load the initial state
        self.load_snapshot('initial')
        
        #run post-reset processes
        self._post_reset(seed=seed_value)
        
        #return an observation
        return self.observe()
    
    def _post_reset(self, seed=None):
        """this executes additional reset-related processes after loading the
        initial snapshot and before returning an initial observation.
        
        Parameters
        ----------
        seed : int, optional
            a custom seed value to use
        """
        #call the randomizer
        self._randomizer.randomize(seed=seed)
    
    def _render(self, mode='human', close=False):
        """render an observation. we use the same GUI as the OpenAI gym atari
        environments.
        
        Parameters
        ----------
        mode : string, optional
            the rendering mode:
                'rgb_array': return a numpy array representing an RGB image of
                    the observation
                'human': render the observation on screen
        close : bool, optional
            if this is True, then the GUI viewer should close
        
        Returns
        -------
        im : ndarray | pyglet.image.ImageData
            if mode == 'rgb_array', returns the array. otherwise, returns
            ImageData of the displayed image.
        """
        if close:  #close the GUI
            if self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            
            return
        
        #get an RGB image of the current render observation
        im = self._get_render_image()
        
        if mode == 'human':  # show on screen
            im = self._show_image(im)
        
        if self._render_kwargs['capture'] and self._capture_paths is not None:
            self.capture(im)
        
        return im
    
    def _close(self):
        """end the world!"""
        self.render(close=True, capture=False)
        self.clear()
    
    def _configure(self):
        """just to comply with the gym environment specifications"""
        pass

    def _seed(self, seed=None):
        """set the random number generator seed. called during world creation
        and resetting.
        
        Parameters
        ----------
        seed : int, optional
            a non-negative integer seed for the random number generator
        
        Returns
        -------
        seed : int
            the new seed value
        """
        #reseed the rng
        self._rng, seed = seeding.np_random(seed)
        
        self.debug.log('seeded the rng with seed=%s' % (seed), level=Debug.INFO)
        
        #append a seed event to the history
        self._history.append_seed(seed)
        
        #save the seed value
        self._seed_value = seed
        
        return seed
    
    def _parse_world_attributes(self, default, explicit):
        """construct a dict describing a set of world attributes
        
        Parameters
        ----------
        default : list
            a list of default world attribute names
        explicit : dict
            a dict of explicitly defined world attributes and values
        
        Returns
        -------
        world_attributes : list
            a list of WorldAttribute specifications
        """
        #construct the explicit specifications with values first
        world_attributes = [[name, {'value': value}] for name,value in explicit.iteritems()
                            if value is not None]
        
        #next construct the explicit specifications without values
        world_attributes += [[name, {'value': value}] for name,value in explicit.iteritems()
                             if value is None]
        
        #now add the defaults
        for name in default:
            if name not in explicit:
                world_attributes.append([name, {}])
        
        return world_attributes
    
    def _encode_state(self, state):
        """encode custom world information in a State"""
        #store the rng state
        rng_state = None if self._rng is None else self._rng.get_state()
        state.set('rng', rng_state)
    
    def _decode_state(self, state):
        """restore custom world information from a State"""
        #restore the rng state
        self._rng.set_state(state.get('rng'))
        
        #clear the state
        self._state[:] = 0
        
        #make an observation
        self.observe()
    
    def _add_world_attributes(self, world_attributes):
        """add a set of WorldAttributes. this is where _world_attributes is
        converted from a list to a dict.
        
        Parameters
        ----------
        world_attributes : list
            a list of WorldAttribute specifiers (see Entity.get_instance)
        """
        if isinstance(self._world_attributes, list):
            self._world_attributes = {}
        
        for spec in world_attributes:
            self.get_world_attribute(spec)
    
    def _add_stepping_attribute(self, attr):
        """add an ObjectAttribute to the list of stepping attributes, attempting
        to satisfy the specified order constraints
        
        Parameters
        ----------
        attr : SteppingObjectAttribute
            the SteppingObjectAttribute to add
        """
        #append the new attribute to the list
        self._stepping_object_attributes.append(attr._name)
        
        #sort the list according to step order
        order = ObjectAttribute._attribute_order['step']
        self._stepping_object_attributes = topological_sort(
                                            self._stepping_object_attributes,
                                            order)
    
    def _remove_stepping_attribute(self, attr):
        self._stepping_object_attributes.remove(attr._name)

    def _add_randomizing_attribute(self, attr):
        """add an ObjectAttribute to the list of randomizing attributes, attempting to
        satisfy the specified order constraints
        
        Parameters
        ----------
        attr : RandomizingObjectAttribute
            the RandomizingObjectAttribute to add
        """
        #append the new attribute to the list
        self._randomizing_object_attributes.append(attr._name)
        
        #sort the list according to initialize order
        order = ObjectAttribute._attribute_order['initialize']
        self._randomizing_object_attributes = topological_sort(
            self._randomizing_object_attributes,
            order)
    
    def _remove_randomizing_attribute(self, attr):
        self._randomizing_object_attributes.remove(attr._name)
    
    def _add_agent(self, agent):
        """add an Agent to the world. this is called by Agent, so you don't have
        to do it manually.

        Parameters
        ----------
        agent : Agent
            the agent to add
        """
        if len(self._agent) == 0 and not self.multi_agent_mode:
            self.agent = agent
        else:
            self.agent = self._agent + [agent]

    def _remove_agent(self, agent):
        """remove an Agent from the world. this is called by Agent, so you don't
        have to do it manually.

        Parameters
        ----------
        agent : Agent
            the agent to remove
        """
        agents = self._agent
        
        try:
            agents.remove(agent)
        except ValueError:
            return
        
        if len(agents) == 0 and not self.multi_agent_mode:
            self.agent = None
        else:
            self.agent = agents

    def _get_new_object_id(self):
        """Get the next available object id.
        """
        rv = self._num_objects
        self._num_objects += 1
        return rv
    
    def _update_object_ids(self):
        """update each Object's id after a change in the row structure of the
        master data array"""
        self.object_attributes['id'].set(None, np.arange(self._num_objects))

        # ids no longer valid, clear all family id caches
        for obj in self.objects:
            obj._clear_family_id_caches()
    
    def _prepare_entities(self):
        """call each Entity's prepare() method. called during world
        _initialize()."""
        for attr in self.world_attributes.values():
            attr.prepare()
        
        for attr in self.object_attributes.values():
            attr.prepare()
        
        for obj in self.objects:
            obj.prepare()
        
        for goal in self.goals.values():
            goal.prepare()
        
        self.judge.prepare()
        
        for variant in self.variants.values():
            variant.prepare()
        
        self.randomizer.prepare()
        
        for agent in self._agent:
            agent.prepare()
    
    def _set_views(self):
        """update all of the views in to the master data array. this method is
        set as a callback in world._data that gets called whenever the structure
        of the master data array changes.
        """
        if self._data is not None:
            self._objects = self._data.view_field('object', type=ObjectCollection) if \
                            'object' in self._data.fields else None
            
            self._object_types = self._data.view_field('object_type').data if \
                            'object_type' in self._data.fields else None
        
        if self._object_attributes is not None:
            for attr in self._object_attributes.values():
                attr._set_view()
    
    def _get_lut(self, state):
        """construct a lookup table for rendering the state
        
        Parameters
        ----------
        state : ndarray
            a world.state
        
        Returns
        -------
        lut : ndarray
            a num_colors x 3 array of colors in the lookup table
        """
        #number of colors in the LUT
        num_colors = np.max(state) + 1
        
        #only update the LUT if we have new colors
        if self._lut is None or self._lut.shape[0] < num_colors:
            self._lut = generate_color_palette(num_colors)
        
        return self._lut
    
    def _get_render_image(self):
        """construct an RGB image based on the current render observation
        
        Returns
        -------
        im : ndarray
            an RGB image of the current render observation
        """
        if self._render_type == 'state':
            obs = self.state
        else:
            obs = getattr(self, '%s_state' % (self._render_type))
        
        assert isinstance(obs, np.ndarray) and obs.dtype != object, 'invalid render observation'
        
        #convert index arrays to rgb images
        if np.issubdtype(obs.dtype, np.integer) or obs.dtype == bool:
            if obs.dtype != np.uint32:
                obs = obs.astype(np.uint32)
            obs = ind_to_rgb(obs, self._get_lut(obs))
        
        #resize for viewing
        rgb = imresize(obs, self.render_size)
        
        return rgb
    
    def _show_image(self, im):
        """show an image in the window, with extra info displayed. this modifies
        OpenAI gym's SimpleImageViewer imshow() method, in order to allow text
        to be displayed as well.
        
        Parameters
        ----------
        im : ndarray
            the RGB image array to show
        
        Returns
        -------
        im : pyglet.image.ImageData
            a pyglet ImageData image of the current screen (use im.save() to
            save the image to a file)
        """
        check_gui_exists()

        #parameters for the info display
        text_size = 14  # text size, in pixels
        padding = 4
        
        #add some padding to the bottom for the info display
        box_height = text_size + 2*padding
        box = np.zeros((box_height, im.shape[1], 3), dtype=im.dtype)
        im = np.vstack((im, box))
        
        #make sure the viewer is initialized
        if self._viewer is None:
            self._viewer = rendering.SimpleImageViewer()
        viewer = self._viewer
        
        #open the window if necessary
        if viewer.window is None:
            height, width, channels = im.shape
            viewer.window = pyglet.window.Window(width=width, height=height,
                                                 display=viewer.display)
            viewer.width = width
            viewer.height = height
            viewer.isopen = True
        
        #get the image to display
        assert im.shape == (viewer.height, viewer.width, 3), 'image has invalid shape'
        image = pyglet.image.ImageData(viewer.width, viewer.height, 'RGB',
                                        im.tobytes(), pitch=viewer.width * -3)
        
        #set the window caption
        viewer.window.set_caption(str(self.name))
        
        #prepare the display
        viewer.window.clear()
        viewer.window.switch_to()
        viewer.window.dispatch_events()
        
        #copy the image
        image.blit(0,0)
        
        #display some info along the bottom
        template = 'Reward: %s    Time: %%05d' % ('%s' if self.multi_agent_mode else '%06d')
        info = template % (self.total_reward, self.time)
        label = pyglet.text.Label(info,
                    x=padding,
                    y=padding + 2,
                    bold=True,
                    color=(255, 255, 255, 255),
                    )
        label.font_size = text_size * 72 / label.dpi
        label.draw()
        
        #get the image before we flip it
        im = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        
        #flip to the front buffer
        viewer.window.flip()
        
        return im
    
    def _set_capture_paths(self, capture):
        """generate path information for capturing state images to file and
        saving gifs
        
        Parameters
        ----------
        capture : bool | string
            either a boolean to indicate whether or not to capture to the
            default output path, a name if the file should use a custom prefix,
            or a file path to specify where to save the gif.
        """
        if not capture:
            self._capture_paths = None
        else:
            temp_dir = tempfile.mkdtemp()
            
            if isinstance(capture, bool):
                capture = self.name
            
            assert isinstance(capture, basestring), 'capture must be a bool or string'
            
            if capture.endswith('.gif'):
                [output_dir, output_name] = os.path.split(capture)
                output_prefix = None
            else:
                output_dir = os.path.join(module_path, 'output')
                output_prefix = capture
                output_name = None
        
            self._capture_paths = {
                'temp_dir': temp_dir,
                'output_dir': output_dir,
                'output_prefix': output_prefix,
                'output_name': output_name,
            }
    
    def _get_capture_temp_path(self):
        """construct the path to the current captured screen
        
        Returns
        -------
        path : string
            the path to which to save the current screen capture
        """
        if self._capture_paths is None:  # captures haven't been set up yet
            self._set_capture_paths(True)
        elif self._capture_paths['temp_dir'] is None:  # gif was already saved
            self._capture_paths['temp_dir'] = tempfile.mkdtemp()
        
        output_dir = self._capture_paths['temp_dir']
        
        #increment the capture counter
        capture_idx = self._capture_idx
        self._capture_idx += 1
        
        return os.path.join(output_dir, '%06d.png' % (capture_idx))
    
    def _get_capture_output_path(self):
        """construct the path for the output capture gif
        
        Returns
        -------
        path : string
            the path to which to save the capture gif
        """
        if self._capture_paths is None:  # don't capture
            return None
        
        #output directory
        output_dir = self._capture_paths['output_dir']
        
        #output file name
        if self._capture_paths['output_name'] is None:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            output_prefix = self._capture_paths['output_prefix']
            total_rewards = '_'.join('r%06d' % x for x in self._total_reward)
            output_name = '%s_%s_t%06d_%s.gif' % (output_prefix, timestamp,
                self.time, total_rewards)
        else:
            output_name = self._capture_paths['output_name']
        
        return os.path.join(output_dir, output_name)
    
    def __getstate__(self):
        """Custom pickling behavior: omit world._viewer, since it is not picklable."""
        d = copy(self.__dict__)
        try:
            del d['_viewer']
        except KeyError:
            pass
        return d
