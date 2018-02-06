'''
    functions for navigating the PixelWorld resource library
'''
import os, sys
import importlib

module_dir = os.path.dirname(__file__)

reload_modules_by_default = False

def sections():
    """get a list of sections in the library"""
    return [name for name in os.listdir(module_dir) if os.path.isdir(os.path.join(module_dir, name))]

def menu(section='world'):
    """get a list of the available library items
    
    Parameters
    ----------
    section : string, optional
        a section of the library
    
    Returns
    -------
    items : list
        a list of items in the section
    """
    path = os.path.join(module_dir, section)
    
    return [os.path.splitext(module)[0] for module in os.listdir(path)
            if module.endswith('.py') and not module.startswith('_')]

def import_item(section, item, reload_module=None):
    """import an item from the library
    
    Parameters
    ----------
    section : string
        the library section
    item : string
        the item
    reload_module : bool, optional
        True to reload the module if it was previously imported
    
    returns
    -------
    m : module
        the specified module
    """
    g = globals()
    
    #so we can restore the old default restore_module value
    old_reload_default = g['reload_modules_by_default']
    
    if reload_module is None:
        reload_module = old_reload_default
    else:
        #this is so modules that import other modules using this method will use
        #the same value for reload_module by default
        g['reload_modules_by_default'] = reload_module
    
    #the module name
    name = '%s.%s.%s' % (__package__, section, item)
    
    do_reload = reload_module and name in sys.modules
    
    m = importlib.import_module(name)
    
    if do_reload:
        reload(m)
    
    #restore the old reload default
    g['reload_modules_by_default'] = old_reload_default
    
    return m
