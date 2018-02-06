"""
    a minimal, fast implementation of PixelWorld that supports the following:
    
    object attributes:
        id (int)
        name (string)
        shape (list[tuple(int, int)])
        color (int)
        position (tuple[int, int])
        meta (dict)
        head_of_family (always self)
    
    object classes:
        self
        basic
        complex
        immoveable
        immoveable_big
        frame
    
    actions:
        NOOP
        DOWN
        RIGHT
        UP
        LEFT
    
    behaviors:
        self moves down, right, up, left
        transitive pushing of moveable objects
"""
from __future__ import print_function
from __future__ import absolute_import

from copy import copy, deepcopy

from gym import Env
import numpy as np
import weakref


def test(objects=None, height=None, width=None):
    """basic interactive test of MinimalPixelWorld
    
    Parameters
    ----------
    see MinimalPixelWorld
    
    Returns
    -------
    pw : MinimalPixelWorld
        the environment
    """
    if height is None: height = 22
    if width is None: width = 22
    
    if objects is None:
        objects = [
            ['frame', {}],
            ['self', {
                'color': 2,
                'position': (4, 5),
            }],
            ['basic', {
                'position': (4, 6),
            }],
            ['complex', {
                'position': (4, 7),
                'color': 3,
                'shape': [(-1, 0), (0, 0), (1, 0), (0, 1)],
            }],
            ['immoveable', {
                'position': (10, 10),
            }],
        ]
    
    pw = MinimalPixelWorld(objects, height, width)
    
    print(pw.observe())
    
    done = False
    action_map = {'a': 'LEFT', 'd': 'RIGHT', 'w': 'UP', 's': 'DOWN', 'q': None}
    while not done:
        key = raw_input('action (a: LEFT, d: RIGHT, w: UP, s: DOWN, q: quit):')
        action = action_map[key]
        
        if action is None:
            done = True
        else:
            obs, reward, done, info = pw.step(action)
            print(obs)
    
    return pw


class Object(object):
    """represents an object that lives within a MinimalPixelWorld
    
    Parameters
    ----------
    pw : MinimalPixelWorld
        the host world
    idx : int
        the object index in the world
    cls : str
        the object class
    name : str, optional
        the object name
    color : int, optional
        the object color
    shape : list[tuple], optional
        a list of relative coordinates specifying the object shape
    position : tuple[int, int], optional
        the initial object position
    meta : dict, optional
        a dict of metadata
    """
    _pw = None
    
    _cls = None
    
    _idx = None
    _name = None
    _shape = None
    _color = None
    _position = None
    _meta = None
    
    _moveable = None
    
    _rows = None
    _cols = None
    
    def __init__(self, pw, idx, cls, name=None, color=None, shape=None,
        position=None, meta=None):
        
        self._pw = weakref.ref(pw)
        self._idx = idx
        self._set_class(cls)
        
        if cls == 'frame':
            if shape is None:
                rows = range(self.pw.height)
                cols = range(1, self.pw.width-1)
                
                shape = [
                    [(row, 0) for row in rows],
                    [(row, self.pw.width - 1) for row in rows],
                    [(0, col) for col in cols],
                    [(self.pw.height - 1, col) for col in cols],
                ]
                
                shape = [p for sub in shape for p in sub]
                
            if position is None:
                position = (0, 0)
        
        self.name = name
        self.shape = shape
        self.color = color
        self.position = position
        self.meta = meta
    
    @property
    def pw(self):
        return self._pw()
    
    @property
    def state(self):
        return {attr:getattr(self, attr) for attr in self.pw._observed_attributes}
    
    @property
    def id(self):
        return self._idx
    
    @property
    def head_of_family(self):
        return self.id
    
    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        if name is None: name = self._cls
        
        assert isinstance(name, basestring), 'bad name'
        
        self._name = name
    
    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, shape):
        if shape is None: shape = [(0, 0)]
        
        assert isinstance(shape, list) and \
            all(isinstance(p, tuple) and len(p) == 2 for p in shape), 'bad shape'
        
        rows, cols = zip(*shape)
        
        self._rows = np.array(rows)
        self._cols = np.array(cols)
        
        self._shape = shape
    
    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, color):
        if color is None: color = 1
        
        assert isinstance(color, int), 'bad color type'
        assert color > 0, 'bad color value'
        
        self._color = color
    
    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, position):
        position = np.array(position)
        
        assert position.dtype == int, 'bad position values'
        assert position.shape == (2,), 'bad position'
        
        self._set_position(position)
    
    @property
    def meta(self):
        return self._meta
    
    @meta.setter
    def meta(self, meta):
        if meta is None: meta = {}
        
        assert isinstance(meta, dict), 'bad meta'
        
        self._meta = meta
    
    @property
    def state_rows(self):
        return self._rows + self.position[0]
    
    @property
    def state_cols(self):
        return self._cols + self.position[1]
    
    def move(self, dr, dc):
        """attempt to move the object to a new location
        
        Parameters
        ----------
        dr : int
            the change in the object's row position
        dc : int
            the change in the object's column position
        
        Returns
        -------
        success : bool
            True if the object successfully moved
        """
        
        #can we be moved?
        if not self._moveable: return False
        
        #first make sure the move will work
        objects_to_move = []
        exclude = set()
        if not self._plan_move(dr, dc, objects_to_move, exclude): return False
        
        #execute the movement
        dp = (dr, dc)
        for obj in objects_to_move:
            obj._set_position(obj.position + dp, update_state=False)
        self.pw._update_state()
        
        return True
    
    def _set_class(self, cls):
        """set the object's class
        
        Parameters
        ----------
        cls : str
            the new class
        """
        assert cls in ['self', 'basic', 'complex', 'immoveable', 'immoveable_big', 'frame'], 'bad class: %s' % (cls,)
        self._cls = cls
        
        self._moveable = cls in ['self', 'basic', 'complex']
    
    def _set_position(self, position, update_state=True):
        """set the object's position
        
        Parameters
        ----------
        position : tuple[int, int]
            the new object position
        """
        self._position = position
        
        if update_state: self._update_state()
        
    def _update_state(self, unoccupy=True):
        """update the state of the parent pixelworld that pertains to the object
        
        Parameters
        ----------
        unoccupy : bool, optional
            True to first remove anything from the state that is currently
            occupying the object's position
        """
        rows = self.state_rows
        cols = self.state_cols
        
        if unoccupy: self.pw._unoccupy_state(rows, cols)
        self.pw._occupy_state(rows, cols, self._idx, self.color)
    
    def _plan_move(self, dr, dc, objects_to_move, exclude):
        """determine whether the object can perform a move
        
        Parameters
        ----------
        dr : int
            the planned change in row position
        dc : int
            the planned change in column position
        objects_to_move : list
            a running list of objects that need to move transitively
        exclude : list
            a list of object indices for objects to exclude from collision
            checking
        """
        #can we move?
        if not self._moveable: return False
        
        #process the exclude set
        if exclude is None: exclude = set()
        assert self._idx not in exclude, 'exclude ignored'
        exclude.add(self._idx)
        
        #add ourselves to the move plan
        objects_to_move.append(self)
        
        #find objects at the new location
        new_rows = self.state_rows + dr
        new_cols = self.state_cols + dc
        collision_indices = self.pw._objects_at(new_rows, new_cols) - exclude
        
        #propagate to collision objects
        for idx in collision_indices:
            obj = self.pw.objects._objects[idx]
            
            if not obj._plan_move(dr, dc, objects_to_move, exclude):
                return False
        
        return True


class Objects(object):
    """represents a set of objects that live within a MinimalPixelWorld
    
    Parameters
    ----------
    pw : MinimalPixelWorld
        the host world
    """
    _pw = None
    
    _objects = None
    
    def __init__(self, pw):
        self._objects = []
        
        self._pw = weakref.ref(pw)
    
    @property
    def pw(self):
        return self._pw()
    
    @property
    def positions(self):
        return [obj.position for obj in self._objects]
    
    @positions.setter
    def positions(self, positions):
        assert isinstance(positions, list), 'positions must be a list'
        assert len(self._objects) == len(positions), 'bad positions length'
        
        for obj,pos in zip(self._objects, positions):
            obj.position = pos

    def get(self, name):
        """get an object, given its name
        
        Parameters
        ----------
        name : str
            the object name
        
        Returns
        -------
        obj : Object
            the object
        """
        if name == 'self':
            return self.pw.self_obj
        else:
            for obj in self._objects:
                if obj.name == name:
                    return obj
        
        raise RuntimeError('object not found!')
    
    def __len__(self):
        return len(self._objects)
    
    def __getitem__(self, item):
        if isinstance(item, basestring):
            return self.get(name=item)
        else:
            return self._objects[item]
    
    def _add(self, specs):
        """add a new object to the world
        
        Parameters
        ----------
        specs : list[str, dict]
            specs for the new object
        """
        assert isinstance(specs, list) and len(specs) == 2, 'bad specs'
        
        cls, params = specs
        
        assert isinstance(params, dict), 'bad params'
        
        idx = len(self)
        obj = Object(self.pw, idx, cls, **params)
        
        self._objects.append(obj)
        
        if obj.name == 'self':
            assert self.pw._self_obj is None, 'more than one self'
            self.pw._self_obj = obj
    
    def _clear(self):
        """remove all objects from the world"""
        del self._objects[:]
        
        self.pw._self_obj = None


class MinimalPixelWorld(Env):
    """a minimal implementation of PixelWorld (see pixelworld.envs.pixelworld
    for the full implementation), built for speed.
    
    Parameters
    ----------
    objects : list[list[str, dict]]
        a list of object specs. each object spec is a two-element list
        specifying the object class name and a dict of attributes to assign to
        the object.
    height : int
        the height of the world, in pixels
    width : int
        the width of the world, in pixels
    do_aaai18_reset_behavior : bool, optional
        the experiments included in the AAAI '18 paper used a version of
        MinimalPixelWorld with a bug that caused objects' presence in the state
        array to not be cleared after a reset() until an action was performed.
        setting this parameter to True will restore that behavior if needed.
    """
    _actions = ['NOOP', 'DOWN', 'RIGHT', 'UP', 'LEFT']
    
    _observed_attributes = ['name', 'id', 'head_of_family', 'position', 'color', 'meta']
    
    _state = None
    _state_indices = None
    
    _objects = None

    _self_obj = None

    _init_objects = None
    
    _do_aaai18_reset_behavior = None

    def __init__(self, objects, height, width, do_aaai18_reset_behavior=False):
        
        self.do_aaai18_reset_behavior = do_aaai18_reset_behavior
        
        self._objects = Objects(self)
        
        self._init_objects = objects
        
        #initialize the state
        self._state = np.zeros((height, width), dtype=np.uint32)
        self._state_indices = np.full((height, width), -1, dtype=np.int32)
        
        self.reset()
    
    @property
    def actions(self):
        return copy(self._actions)
    
    @property
    def num_objects(self):
        return len(self.objects)
    
    @property
    def objects(self):
        return self._objects
    
    @property
    def state(self):
        return self._state.copy()
    
    @property
    def height(self):
        return self._state.shape[0]
    
    @property
    def width(self):
        return self._state.shape[1]
    
    @property
    def self_obj(self):
        return self._self_obj
    
    def observe(self):
        """observe the state of the environment
        
        Returns
        -------
        state : array
            the state array
        """
        return self.state
    
    def _reset(self):
        """private version of reset(), called by Env.reset()"""
        if not self._do_aaai18_reset_behavior:
            self._clear_state()
        
        self._set_objects(self._init_objects)
        
        return self.observe()
    
    def _step(self, action):
        """private version of step(), called by Env.step()"""
        if action == 'NOOP':
            pass
        elif action == 'DOWN':
            self.self_obj.move(1, 0)
        elif action == 'RIGHT':
            self.self_obj.move(0, 1)
        elif action == 'UP':
            self.self_obj.move(-1, 0)
        elif action == 'LEFT':
            self.self_obj.move(0, -1)
        else:
            raise ValueError('bad action: %s' % (action,))
        
        return self.observe(), 0, 0, {}
    
    def _set_objects(self, objects):
        """set the set of objects that occupy the world
        
        Parameters
        ----------
        objects : Objects
            the new set of objects
        """
        assert isinstance(objects, list), 'objects must be a list'
        
        self.objects._clear()
        
        for specs in objects:
            self.objects._add(specs)
    
    def _clear_state(self):
        """clear the state array"""
        self._state[:] = 0
        self._state_indices[:] = -1
    
    def _unoccupy_state(self, rows, cols):
        """remove any signature of objects from the specified locations of the
        state array
        
        Parameters
        ----------
        rows : list[int]
        cols : list[int]
            the (row, col) pairs of state locations to clear
        """
        self._state_indices[[rows, cols]] = -1
        self._state[[rows, cols]] = 0
    
    def _occupy_state(self, rows, cols, idx, color):
        """occupy the specified locations of the state array
        
        Parameters
        ----------
        rows : list[int]
        cols : list[int]
            the (row, col) pairs of state locations to occupy
        idx : int
            the index of the object that is occupying the locations
        color : int
            the color of the object that is occupying the locations
        """
        assert np.all(self._state_indices[[rows, cols]] == -1), 'object collision!'
        
        self._state_indices[[rows, cols]] = idx
        self._state[[rows, cols]] = color
    
    def _update_state(self):
        """refresh the state array according to the current object state"""
        self._clear_state()
        
        for obj in self.objects._objects:
            obj._update_state(unoccupy=False)
    
    def _objects_at(self, rows, cols):
        """find the objects that are currently occupying the specified locations
        of the state array
        
        Parameters
        ----------
        rows : list[int]
        cols : list[int]
            the (row, col) pairs of state locations to search
        
        Returns
        -------
        indices : list[int]
            a list of object indices at the specified state locations
        """
        indices = set(np.unique(self._state_indices[[rows, cols]]))
        
        if -1 in indices: indices.remove(-1)
        
        return indices
