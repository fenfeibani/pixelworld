'''
    provides access to a set of pre-defined PixelWorlds
'''
import os
import importlib
import inspect

from core import PixelWorld, WorldAttribute
import library
from library import helpers as h


def get_pixelworld_defaults(cls=None, defaults=None):
    """get a dict of default arguments for a PixelWorld
    
    Parameters
    ----------
    cls : PixelWorld
        the PixelWorld class to get defaults for
    defaults : dict, optional
        a dict to add default values to, if they don't already exist
    
    Returns
    -------
    defaults : dict
        a dict of default values for the PixelWorld constructor
    """
    if cls is None:
        cls = PixelWorld
    
    if defaults is None:
        defaults = {}
    
    init = cls.__init__ if inspect.isclass(cls) else cls
    
    #get PixelWorld's constructor argument list
    args, varargs, varkw, arg_defaults = inspect.getargspec(init)
    if arg_defaults is None: arg_defaults = []
    
    #use PixelWorld's defaults when possible
    num_args = len(args)
    num_defaults = len(arg_defaults)
    values = [None]*(num_args - num_defaults) + list(arg_defaults)
    
    #add the defaults to the dict
    for arg,value in zip(args, values):
        if not arg in defaults and arg != 'self':
            defaults[arg] = value
    
    #add the world_attributes
    for name in getattr(cls, '_world_attributes', []):
        if not name in defaults:
            defaults[name] = None
    
    #add the defaults from the super classes
    if inspect.isclass(cls):
        for base_cls in cls.__bases__:
            if issubclass(base_cls, PixelWorld):
                defaults = get_pixelworld_defaults(cls=base_cls, defaults=defaults)
    
    return defaults

def load_parameters(world, defaults=None, ignore_world_agent=False,
                    reload_module=False, **kwargs):
    """load parameters for a preset world
    
    Parameters
    ----------
    world : string
        the name of a preset world. must be a module in the worlds package.
    defaults : dict, optional
        a dict of default parameters to use if the world doesn't override them.
        if neither the preset world nor defaults defines a parameter, the
        PixelWorld default is used.
    ignore_world_agent : bool, optional
        True to ignore any agent defined by the world
    reload_module : bool, optional
        True to reload the world module if it has already been imported
    **kwargs
        parameters to override
    
    Returns
    -------
    params : dict
        a dict of keyword arguments for the PixelWorld constructor
    """
    if defaults is None:
        defaults = {}
    
    #load the preset world module
    try:
        module = library.import_item('world', world, reload_module=reload_module)
    except ImportError:
        module = library.import_item('world', 'blank', reload_module=reload_module)
    
    #get the PixelWorld class
    if hasattr(module, 'world'):
        cls = module.world
    elif 'world' in defaults:
        cls = defaults['world']
    else:
        cls = PixelWorld
    
    #start with the default PixelWorld constructor parameters
    params = get_pixelworld_defaults(cls)
    params['world'] = cls
    
    #override with the manual defaults
    params.update(**defaults)
    
    #override with the world defaults
    for key in params:
        if hasattr(module, key) and (not ignore_world_agent or key != 'agent'):
            params[key] = h.Lazy.resolve(getattr(module, key), params=kwargs)
    
    #override with manually specified parameters
    params.update(**kwargs)
    
    return params

def create_world(world='world', defaults=None, ignore_world_agent=True,
                    reload_module=False, **kwargs):
    """create a PixelWorld based on a preset world from the library
    
    Parameters
    ----------
    world : string | list, optional
        the name of a preset world, or a two-element list of the form
        [name, params], name is the name of a PixelWorld and params is a dict of
        input parameters to the PixelWorld constructor
    defaults : dict, optional
        a dict of default parameters to use if the world doesn't override them
    ignore_world_agent : bool, optional
        True to ignore any agent defined by the world
    reload_module : bool, optional
        True to reload the world module if it has already been imported
    **kwargs
        keyword arguments to override in the PixelWorld constructor
    
    Returns
    -------
    world : PixelWorld
        the specified world
    """
    #parse the world specification
    if isinstance(world, list):
        if len(world) != 2:
            raise ValueError('world must be a two-element list')
        
        name = world[0]
        params = world[1]
    else:
        name = world
        params = {}
    
    #add the default parameters
    if defaults is not None:
        params.update(**defaults)
    
    #make sure the world has a name
    if 'name' not in params:
        params['name'] = name
    
    #parse the world parameters
    params = load_parameters(name, defaults=params,
                ignore_world_agent=ignore_world_agent,
                reload_module=reload_module, **kwargs)
    
    #get the PixelWorld class to use
    world = params.pop('world')
    
    return world(**params)
