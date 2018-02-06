'''
    basic set of Objects for PixelWorld
'''
import inspect
from copy import copy

import numpy as np

import core
from utils import is_iterable, to_iterable, roundup
from library.helpers import h


class VisibleObject(core.Object):
    """an Object that can be visible, i.e. can affect the world.state. this is
    the base class for Objects with a "physical" presence in the world."""
    _attributes = ['visible', 'zorder', 'position', 'state_index', 'extent',
                    'depth', 'color']


class ShapeObject(VisibleObject, core.CompoundObject):
    """a CompoundObject with visible children. the positions of a ShapeObject's
    children can be set via its shape attribute. this is the base class for
    CompoundObjects with a "physical" presence in the world.
    
    the following keyword arguments can be passed to the ShapeObject
    constructor, in addition to those for any CompoundObject:
    
    Parameters
    ----------
    shape : string | array_like, optional
        the initial shape of the ShapeObject (overrides _default_shape). string
        inputs are converted to shapes using the library.world._helpers.shape
        function.
    """
    _class_tree_attributes = ['_block_params']
    
    _attributes = ['extent', 'shape']
    
    #subclasses can define a default shape that can be overridden when
    #ShapeObjects are created
    _default_shape = None
    
    _child_type = 'visible'
    
    _auto_state = False
    
    def _construct_children_params(self, children, child_type, child_params,
        params, num_children=None):
        """overridden from CompoundObject. this incorporates the specified
        ShapeObject's shape into the child parameters.
        
        Parameters
        ----------
        children : list[string | list] | None
            see CompoundObject. if unspecified, the number of children will be
            determined from the specified shape.
        child_type : string | list[string] | None
            see CompoundObject
        child_params : dict | list[dict] | None
            see CompoundObject
        params : dict
            a dict of the keyword arguments passed to the ShapeObject
            constructor. this might include a 'shape' element that overrides the
            default shape.
        num_children : Number, optional
            the number of children that should be created. if unspecified,
            determined from the specified shape.
        
        Returns
        -------
        children : list[list[string, dict]]
            the children specification list
        """
        #process the shape
        shape = params.pop('shape', None)
        if shape is None:
            shape = copy(self._default_shape)
        if shape is not None:
            shape = self._attributes['shape'].coerce_value(shape)
            assert is_iterable(shape), 'shape must be iterable'
            
            num_shape = len(shape)
        
            if num_children is None:
                num_children = num_shape
            else:
                assert num_children == num_shape, 'number of children (%d) and shape length (%d) do not match' % (num_children, num_shape)
        
        #pop the parent position out of params (so the super method doesn't give
        #the children a position based on the parent)
        if not 'position' in params:
            position = self._attributes['position']
            params['position'] = self._get_init_value(params, position, evaluate=False)
        parent_pos = params.pop('position')
        
        #parse callable positions
        if callable(parent_pos):
            if 'shape' in inspect.getargspec(parent_pos)[0]:
                parent_pos = parent_pos(self, shape=shape)
            else:
                parent_pos = parent_pos(self)
        
        #get the children without positions specified
        children = super(ShapeObject, self)._construct_children_params(children,
                    child_type, child_params, params, num_children=num_children)
        
        #put position back in params
        params['position'] = parent_pos
        
        #parse the absolute child positions
        if shape is not None:
            #absolute shape positions
            children_pos = parent_pos + shape
            
            #make sure int coordinates are respected
            if isinstance(parent_pos, np.ndarray) and parent_pos.dtype == int:
                children_pos = children_pos.astype(int)
            
            for idx,child in enumerate(children):
                if not 'position' in child[1]:
                    child[1]['position'] = children_pos[idx]
        
        return children


class SpriteObject(ShapeObject):
    """a SpriteObject is a ShapeObject that stores a list of sprite
    specifications that define the appearance of the Object. a SpriteObject's
    active sprite can be selected via its sprite attribute, which is the index
    of the active sprite in the above list.
    
    each sprite defines three things:
        shape:      the shape of the sprite (i.e. the relative positions of the
                    SpriteObject's children)
        color:      the color of each child
        visible:    whether each child is visible when the sprite is active.
                    this is True whenever the child's color is non-zero.
    
    the animated attribute determines whether the SpriteObject's sprite cycles
    through its list of sprites as the world steps.
    
    the SpriteObject's sprite list is defined during Object construction by
    specifying the sprites keyword argument. each element of that list should be
    an integer numpy array that defines the sprite, a string to be parsed into
    said numpy array by library.helpers.sprite.from_string, or a per-parsed
    sprite dict. see _parse_sprite() below and
    library.helpers.sprite.from_string() for details on constructing sprite
    arrays.
    
    aliens.py in library/world shows an example of sprites in action.
    
    the following keyword arguments can be passed to the SpriteObject
    constructor, in addition to those for any ShapeObject:
    
    Parameters
    ----------
    sprite : list[ndarray]
        a list of sprite arrays (see _parse_sprite())
    """
    _attributes = ['sprite', 'animated']
    
    #the index of the active sprite
    _sprite = 0
    
    #a list of sprite specifications
    _sprites = None
    
    #a mapping from sprite names to sprite indices, if sprites is defined via a
    #dict
    _sprite_name_to_idx = None
    
    @property
    def specifications(self):
        spec = super(SpriteObject, self).specifications
        
        del spec[1]['shape']
        spec[1]['sprites'] = self._sprites
        
        return spec
    
    @property
    def sprites(self):
        """a list of sprite specifications"""
        return self._sprites
    
    @sprites.setter
    def sprites(self, sprites):
        """use to reconfigure the sprites. this will cause the SpriteObjects
        existing children to be deleted, and a new set to be created."""
        #get the parameters for create_children
        params = {
            'sprites': sprites,
            'sprite': self.sprite,
        }
        
        #delete the old children
        for child in copy(self._children):
            child.remove()
        
        #create the new children
        self._create_children(None, None, None, params)
    
    def set_sprite(self, sprite):
        """set the current sprite. this behaves like setting obj.sprite except
        when the sprites were defined using a dict, in which case the sprite can
        also be set by name.
        
        Parameters
        ----------
        sprite : int | string
            the index of the sprite to set, or the name of the sprite if the
            sprites were set via a dict.
        """
        if isinstance(sprite, basestring):
            try:
                sprite = self._sprite_name_to_idx[sprite]
            except KeyError:
                raise ValueError('no sprite with name "%s" exists' % (sprite))
        
        self.sprite = sprite
    
    def _parse_sprites(self, sprites):
        """convert a list of sprite definition arrays to the shape/color/visible
        attributes that the sprite defines
        
        Parameters
        ----------
        sprites : list[ndarray]
            a list of sprite arrays (see _parse_sprite())
        
        Returns
        -------
        params : list[dict]
            a list of sprite parameter dicts
        """
        #convert dict to list and store the sprite names
        if isinstance(sprites, dict):
            self._sprite_name_to_idx = {name:idx for idx,name in enumerate(sprites.keys())}
            sprites = sprites.values()
        
        return [self._parse_sprite(sprite) for sprite in sprites]
    
    def _parse_sprite(self, sprite):
        """convert a sprite definition array to the corresponding shape, color,
        and visible attributes that it defines
        
        Parameters
        ----------
        sprite : ndarray | string | dict
            ndarray:
                a sprite array, which is an integer numpy array that defines the
                sprite configuration. positive integer values denote a visible
                block with that integer color. 0 values indicate invisible
                blocks. negative values correspond to constants as defined in
                library.sprite._helpers:
                    SPRITE_IGNORE: no block at that location
                    SPRITE_DEFAULT: block should have the default color, which
                        is usually just the color attribute value of the
                        SpriteObject
                as an example:
                    -1 -2 -1
                    -2  0 -2
                    -1 -2 -1
                defines a plus-shaped sprite whose arms take on the
                SpriteObject's color and whose center is invisible.
            string:
                a sprite string to be parsed by
                library.helpers.sprite.from_string()
            dict:
                a pre-parsed sprite dict
        
        Returns
        -------
        params : dict
            a dict specifying the shape, color, and visible attributes defined
            by the sprite
        """
        if isinstance(sprite, np.ndarray):
            #find the block locations
            y, x = np.nonzero(sprite != h.sprite.SPRITE_IGNORE)
            
            #color of each block
            color = sprite[(y, x)]
            
            #center the block locations
            y = np.floor(y - np.mean(y))
            x = np.floor(x - np.mean(x))
            
            return {
                'shape': zip(y, x),
                'color': color,
                'visible': color != 0,
            }
        elif isinstance(sprite, basestring):
            return self._parse_sprite(h.sprite.from_string(sprite))
        elif isinstance(sprite, dict):
            return sprite
        else:
            raise TypeError('invalid sprite type (%s)' % (type(sprite)))
    
    def _parse_sprite_colors(self, colors, default_color=None):
        """parse a sprite color specification into the actual colors of the
        SpriteObject's blocks
        
        Parameters
        ----------
        colors : ndarray
            the sprite's color specification array
        default_color: int, optional
            the integer color to assign to blocks marked as having the default
            color (defaults to the SpriteObject's color)
        
        Returns
        colors : ndarray
            an array of the actual block colors
        """
        if default_color is None:
            default_color = self._get('color')
        
        colors = copy(colors)
        colors[colors == h.sprite.SPRITE_DEFAULT] = default_color
        
        return colors
    
    def _construct_children_params(self, children, child_type, child_params,
        params, num_children=None):
        """overridden from CompoundObject. this incorporates the specified
        ShapeObject's shape into the child parameters.
        
        Parameters
        ----------
        children : list[string | list] | None
            see CompoundObject. if unspecified, the number of children will be
            determined from the specified shape.
        child_type : string | list[string] | None
            see CompoundObject
        child_params : dict | list[dict] | None
            see CompoundObject
        params : dict
            a dict of the keyword arguments passed to the SpriteObject
            constructor. this should include a 'sprites' element that defines
            the SpriteObject's sprites.
        num_children : Number, optional
            the number of children that should be created. if unspecified,
            determined from the specified sprites.
        
        Returns
        -------
        children : list[list[string, dict]]
            the children specification list
        """
        #process the sprites
        sprites = params.pop('sprites', None)
        if callable(sprites):
            sprites = sprites(self)
        
        assert sprites is not None, 'sprites must be defined at Object creation'
        assert not 'shape' in params, 'shape cannot be defined explicitly for a SpriteObject'
        
        self._sprites = self._parse_sprites(sprites)
        
        #find the initial sprite
        sprite = self._attributes['sprite']
        sprite_idx = self._get_init_value(params, sprite, pop=False)
        assert sprite_idx >= 0 and sprite_idx < len(self._sprites), 'invalid sprite'
        
        #get the shape of the initial sprite
        shape = self._sprites[sprite_idx]['shape']
        
        #construct the child parameters based on the initial sprite
        params['shape'] = shape
        children = super(SpriteObject, self)._construct_children_params(children,
                    child_type, child_params, params, num_children=num_children)
        
        #set the child color and visibility based on the initial sprite
        sprite = self._sprites[sprite_idx]
        color = self._attributes['color']
        default_color = self._get_init_value(params, color, pop=False)
        colors = self._parse_sprite_colors(sprite['color'],
                                            default_color=default_color)
        for idx,child in enumerate(children):
            if not 'color' in child[1]:
                child[1]['color'] = colors[idx]
            if not 'visible' in child[1]:
                child[1]['visible'] = sprite['visible'][idx]
        
        return children


class BasicObject(VisibleObject):
    """an Object with a basic set of physical attributes"""
    _attributes = [
        'mass',
        'acceleration',
        'velocity',
        'kinetic_energy',
        'momentum',
    ]


class ComplexObject(BasicObject, ShapeObject):
    """the ShapeObject analog of a BasicObject"""
    _child_type = 'basic'


class ComplexSpriteObject(BasicObject, SpriteObject):
    """a ComplexObject that supports sprites"""
    _child_type = 'basic'


class ImmoveableObject(BasicObject):
    """a BasicObject that cannot be moved by other Objects"""
    _attributes_removed = [
        'mass',
        'acceleration',
        'velocity',
        'kinetic_energy',
        'momentum',
    ]


class ImmoveableBigObject(ImmoveableObject, ShapeObject):
    """the ShapeObject analog of an ImmoveableObject"""
    _child_type = 'immoveable'


class ImmoveableSpriteObject(ImmoveableObject, SpriteObject):
    """the SpriteObject analog of an ImmoveableObject"""
    _child_type = 'immoveable'


class UnpushableObject(BasicObject):
    """a BasicObject that cannot be pushed by other Objects"""
    _attributes_removed = [
        'mass',
        'kinetic_energy',
        'momentum',
    ]


class UnpushableBigObject(UnpushableObject, ShapeObject):
    """the ShapeObject analog of an UnpushableObject"""
    _child_type = 'unpushable'


class UnpushableSpriteObject(UnpushableObject, SpriteObject):
    """the SpriteObject analog of an UnpushableObject"""
    _child_type = 'unpushable'


class WallObject(ImmoveableObject):
    """the building block of walls (e.g. the frame)"""
    pass


class FrameObject(WallObject, ShapeObject):
    """a frame that surrounds the visible portion of the world. can be used to
    contain Objects within the visible world."""
    _child_type = 'wall'
    
    #don't randomize
    _exclude_randomize = True

    def __init__(self, world, sides=None, shape=None, **kwargs):
        """
        Parameters
        ----------
        world : PixelWorld
            the host world
        sides : list
            the list of sides ('top', 'right', 'bottom', and/or 'left') to
            include in the frame
        shape : string | array_like, optional
            see ShapeObject (overrides sides)
        **kwargs
            extra keyword arguments (see ShapeObject, CompoundObject)
        """
        if shape is None:
            shape = self._get_frame_shape(world, sides=sides)
        
        if not 'position' in kwargs:
            kwargs['position'] = (0, 0)
        
        super(FrameObject, self).__init__(world, shape=shape, **kwargs)
    
    def _get_frame_shape(self, world, sides=None):
        """get the shape of the frame, given the sides to include
        
        Parameters
        ----------
        world : PixelWorld
            the host world
        sides : list, optional
            a list of sides to include. defaults to all sides.
        
        Returns
        -------
        shape : list
            the frame shape
        """
        #default sides to include
        if sides is None:
            sides = ['top', 'right', 'bottom', 'left']
        
        shape = []
        
        if 'top' in sides:
            shape += [(0, x) for x in xrange(world.width)]
        if 'right' in sides:
            y_start = 1 if 'top' in sides else 0
            y_end = (world.height - 1) if 'bottom' in sides else world.height
            shape += [(y, world.width - 1) for y in xrange(y_start, y_end)]
        if 'bottom' in sides:
            shape += [(world.height - 1, x) for x in xrange(world.width-1, -1, -1)]
        if 'left' in sides:
            y_start = (world.height - 2) if 'bottom' in sides else (world.height - 1)
            y_end = 0 if 'top' in sides else -1
            shape += [(y, 0) for y in xrange(y_start, y_end, -1)]
        
        return shape


class SelfObject(UnpushableObject):
    """an Object that pushes but can't be pushed around by other Objects"""
    _attributes = ['pushes', 'orientation', 'orients']


class BasicSelfObject(BasicObject):
    """an Object that pushes and -can- be pushed around by other Objects"""
    _attributes = ['pushes']
    
    _defaults = {
        'mass': 10,
    }


class SelfBigObject(SelfObject, UnpushableBigObject):
    """the ShapeObject analog of a SelfObject"""
    pass


class SelfSpriteObject(SelfObject, UnpushableSpriteObject):
    """the SpriteObject analog of a SelfObject"""
    pass


class Bullet(BasicObject):
    """Object class for objects produced by shooters when they shoot.
    """

    _attributes = ['bullet_deadly', 'bullet_shooter', 'alive']

    _defaults = dict(color=3, mass=0.0)


class GripObject(core.CompoundObject):
    """See GripsObjectAttribute in object_attributes.py for more context.
    
    Object that is created whenever an object with the
    'grips' attribute makes use of the GRIP
    command. The object is destroyed when the gripper
    UNGRIPs.
    """
    _attributes = ['grip', 'position', 'velocity', 'acceleration']


