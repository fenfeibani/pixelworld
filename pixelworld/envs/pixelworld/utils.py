'''
    utility functions for PixelWorld
'''
import os
import re
from datetime import datetime
import time
from copy import copy, deepcopy
from collections import OrderedDict
from numbers import Number
from math import ceil, floor
import colorsys
import logging

import numpy as np
from numpy import ma
from scipy import stats
from scipy.misc import imresize as scipy_imresize

# put this import inside a try-catch block so that we can run on systems with
# no GUI
gui_exists = True
gui_exceptions = []
try:
    from pyglet.window import key as pkey
except Exception as e:
    gui_exists = False
    gui_exceptions.append(e)
if not gui_exists:
    logging.warn('Warning: proceeding in non-GUI mode: ' + '; '.join(str(e) for e in gui_exceptions))


def check_gui_exists():
    """Check that we successfully imported pyglet and rendering, and fail if we have not."""
    if not gui_exists:
        raise ImportError('a GUI function was called, but GUI-related imports failed: %s ' % 
                          ('; '.join(str(e) for e in gui_exceptions),))

eps = np.finfo(float).eps
eps_threshold = 10000*eps  # threshold for floats to be counted as "different"

#base lookup table for mapping integers to colors
base_lut = np.array([
        (0, 0, 0),
        (1, 1, 1),
        (1, 0, 0),
        (0, 0.5, 1),
        (0, 0.9, 0.5),
        (1, 0.8, 0.25),
        (0.75, 0, 1),
        (0.5, 0.5, 1),
        (1, 0.5, 0),
        (0, 0.75, 0),
    ])
    
#base_lut_hsv = matplotlib.colors.rgb_to_hsv(base_lut)
base_lut_hsv = np.array([[ 0.        ,  0.        ,  0.        ],
                         [ 0.        ,  0.        ,  1.        ],
                         [ 0.        ,  1.        ,  1.        ],
                         [ 0.58333333,  1.        ,  1.        ],
                         [ 0.42592593,  1.        ,  0.9       ],
                         [ 0.12222222,  0.75      ,  1.        ],
                         [ 0.79166667,  1.        ,  1.        ],
                         [ 0.66666667,  0.5       ,  1.        ],
                         [ 0.08333333,  1.        ,  1.        ],
                         [ 0.33333333,  1.        ,  0.75      ]])

def fix_float_integer(x):
    """try to fix floating point error around integers
    
    Parameters
    ----------
    x : Number | ndarray
        values that might be integers affected by floating point error
    """
    #check whether the number is almost the next integer up
    x1 = np.ceil(x)
    x2 = np.ceil(x + eps_threshold)
    if isinstance(x1, np.ndarray):
        do_fix = x1 != x2
        x[do_fix] = x1[do_fix]
    elif x1 != x2:
        return x1
    
    #check whether it is almost the next integer down
    x1 = np.floor(x)
    x2 = np.floor(x - eps_threshold)
    if isinstance(x1, np.ndarray):
        do_fix = x1 != x2
        x[do_fix] = x1[do_fix]
    elif x1 != x2:
        return x1
    
    return x

def compute_compounded_float_error(x, e, n, de):
    """compute the compounded floating point error that is introduced when a
    value is added to a floating point number over multiple steps rather than
    all at once
    
    Parameters
    ----------
    x : float
        a number
    e : float
        the value that should be added to x
    n : int
        the number of steps over which e is added to x
    de : float
        the value that is actually added to x at each step
    
    Returns
    -------
    err : float
        the difference between the desired and actual values added
    """
    y = x
    for _ in xrange(n):
        y += de
    
    return y - (x + e)

def roundup(x):
    """numpy round variant that rounds numbers falling exactly half way between
    two integers away from zero, instead of numpy.round, which rounds to the
    nearest even number
    
    Parameters
    ----------
    x : ndarray
        a numpy array
    
    Returns
    -------
    y : ndarray
        x rounded as described above
    """
    return np.sign(x) * np.floor(np.abs(x) + 0.5)

def mode(x):
    """return the mode of a set of numbers
    
    Parameters
    ----------
    x : array_like
        an array of values
    
    Returns
    -------
    m : Number
        the mode of the values in x
    """
    return stats.mode(x)[0][0]

def is_iterable(x):
    """is x iterable?
    
    Parameters
    ----------
    x : any
        the value to test
    
    Returns
    -------
    b : bool
        True if x is iterable
    """
    return hasattr(x, '__iter__')

def to_iterable(x, length=None):
    """make sure x is an iterable. wrap it in a list if it isn't.
    
    Parameters
    ----------
    x : object
        an object
    length : int, optional
        the desired length of x
    
    Returns
    -------
    x : object
        x as an iterable object
    """
    if not is_iterable(x):
        x = [x]
    
    if length is not None:
        if length == 0:
            return []
        elif len(x) == 1:
            x = [copy(x[0]) for _ in xrange(length)]
        elif len(x) == length:
            x = [copy(v) for v in x]
        else:
            raise ValueError('needed length %d but got %d' % (length, len(x)))
    
    return x

def switch(x, resolve_functions=True, **kwargs):
    """simplified switch statement that returns a value based on the input
    
    Parameters
    ----------
    x : any
        any hashable value
    resolve_functions : bool, optional
        True to resolve function switch values (see kwargs)
    **kwargs
        keyword arguments mapping possible values of x to return values. one of
        the keyword argument can be 'default' to indicate the default return
        value for non-matching x. values can also be functions that take no
        arguments and return the value.
    
    Returns
    -------
    value : any
        the matching return value, or kwargs['default'] if it was defined and
        x doesn't match, or None otherwise
    """
    value = kwargs.get(x, kwargs.get('default', None))
    
    return value() if resolve_functions and callable(value) else value

def plural(n, s=None, p=None):
    """return the proper suffix for a word, depending on whether it should be
    pluralized
    
    Parameters
    ----------
    n : int
        the number of things
    s : string, optional
        the suffix if the string should be singular
    p : string, optional
        the suffix if the string snould be pluralized
    """
    if s is None: s = ''
    if p is None: p = 's'
    
    return s if n==1 else p

def pluralize(word, n):
    """pluralize a word if n represents a plural number
    
    Parameters
    ----------
    word : string
        the world to pluralize
    n : int
        the number of that word that exist
    """
    #determine the plural suffix
    s = None
    p = None
    if len(word) > 0:
        if word[-1] == 'y':
            word = word[:-1]
            s = 'y'
            p = 'ies'
    
    return word + plural(n, s, p)

def generate_color_palette(n):
    """generate a color palette
    
    Parameters
    ----------
    n : int
        the number of colors needed
    
    Returns
    -------
    pal : ndarray
        a num_colors x 3 array of RGB colors
    """
    #number of colors needed on top of the base palette defined above
    n_needed = n - len(base_lut)
    
    if n_needed <= 0:  # already have what we need
        return base_lut[0:n,:]
    
    #sort the hues of the base palette
    sorted_hues = sorted(base_lut_hsv[:,0])
    
    #find maximally distinct hues until we have enough colors
    hues = []
    while n_needed > 0:
        #difference between adjacent hues
        df = [sorted_hues[i+1] - sorted_hues[i] for i in xrange(len(sorted_hues)-1)]
        
        #find the consecutive hues that have the maximum distance between them
        mx = max(df)
        idx = df.index(mx)
        
        #new hue is the mid-point between those hues
        new_hue = (sorted_hues[idx] + sorted_hues[idx+1])/2
        sorted_hues.insert(idx+1, new_hue)
        hues.append(new_hue)
        
        n_needed -= 1
    
    #append the generated colors to the base palette
    return np.vstack((base_lut, np.array([colorsys.hsv_to_rgb(hue, 1, 1) for hue in hues])))

def ind_to_rgb(ind, lut):
    """convert an index image array to a color RGB image array
    
    Parameters
    ----------
    ind : ndarray
        an H x W non-negative integer numpy array
    lut : ndarray
        a num_colors x nd array of colors
    
    Returns
    -------
    rgb : ndarray
        an H x W x nd colorized array
    """
    num_planes = np.shape(lut)[1]
    
    return np.dstack([lut[:,p][ind] for p in xrange(num_planes)])

def imresize(im, sz, interp='nearest', scale_values=True):
    """like scipy.misc's imresize, but doesn't do some of the strange things
    that scipy's version does (like setting constant arrays to zero, or
    screwing up colors in (1, 3, 3) arrays)
    
    Parameters
    ----------
    im : ndarray
        a 2D or 3D numpy array
    sz : int | float | tuple[int, int]
        the resize factor (see scipy.misc.imresize)
    interp : string
        the spline interpolation order (see scipy.ndimage.zoom)
    scale_values : bool
        True to scale the values to between 0 and 255
    
    Returns
    -------
    im : ndarray
        the resized image
    """
    #scale the image values
    if scale_values:
        mn = np.min(im)
        mx = np.max(im)
        if mn == mx:
            im = copy(im)
            im[:] = min(1, max(0, mx))
        else:
            im = (im - mn) / (mx - mn)
        
        im *= 255
    
    #make sure we have at least a 2D array
    if im.ndim < 2:
        im = np.reshape(im, (1, -1))
    
    #make sure we have a 3D array (RGB if 2D)
    if im.ndim < 3:
        shp = im.shape
        im = np.reshape(im, (shp[0], shp[1], 1))
        im = np.tile(im, (1, 1, 3))
    
    #fix for scipy using the 1st axis with size 3 as the channel axis
    if im.shape[0] == 3 or im.shape[1] == 3:
        im = np.transpose(im, (2, 0, 1))
    
    im = scipy_imresize(im, sz, interp=interp)
    
    return im

def to_vector(x):
    """make sure x is a vector numpy array
    
    Parameters
    ----------
    x : any
        the value to vectorize
    
    Returns
    -------
    x : ndarray
        x as a 1D numpy array
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    
    if x.ndim != 1:
        return np.reshape(x, (-1,))
    else:
        return x

def merge_structured_arrays(*arrays):
    """merge a set of structured numpy arrays
    
    Parameters
    ----------
    *arrays
        a set of masked structured arrays. all arrays must have the same number
        of rows.
    
    Returns
    -------
    x : MaskedArray
        the merged array
    """
    #concatenate all of the sub-arrays fields
    dtype = sum((array.dtype.descr for array in arrays), [])
    
    #empty array with all fields
    x = ma.empty(len(arrays[0]), dtype=dtype)
    x = SerializableMaskedArray(x)
    
    #assign each field
    for array in arrays:
        for name in array.dtype.names:
            x[name] = array[name]
    
    return x

def cartesian_product(x, y=None, self_pairs=True, matching_pairs=True):
    """construct the cartesian product of the columns of two ndarrays. x and y
    must be the same shape and either N-length vectors, or nd x N arrays.
    
    Parameters
    ----------
    x : ndarray
        either an N-length vector or a D x N array
    y : ndarray, optional
        either an M-length vector or a D x M array. shape must match that of x.
        if unspecified, x is used.
    self_pairs : bool
        when x.shape == y.shape, True to include pairs whose positions in their
        respective arrays are the same
    matching_pairs : bool
        False to delete pairs in which the two values are equal
    
    Returns
    -------
    p1 : ndarray
        an array representing the first value in each pair of the cartesian
        product
    p2 : ndarray
        an array representing the second value in each pair
    """
    assert x.ndim==y.ndim, 'x and y must both be either vectors or matrices'
    
    N = x.shape[-1]
    M = y.shape[-1]
    
    last_axis = y.ndim - 1
    
    #construct the full cartesian product
    p1 = np.tile(x, M)
    p2 = np.repeat(y, N, axis=last_axis)
    
    if x.shape == y.shape and not self_pairs:  # delete self pairs
        idx_delete = list(xrange(0, N*N, N+1))
        p1 = np.delete(p1, idx_delete, axis=last_axis)
        p2 = np.delete(p2, idx_delete, axis=last_axis)
    
    if not matching_pairs:  # delete matching pairs
        if last_axis==0:
            nonmatch = p1 != p2
            
            p1 = p1[nonmatch]
            p2 = p2[nonmatch]
        else:
            nonmatch = ~np.all(p1 == p2, axis=0)
            
            p1 = p1[:,nonmatch]
            p2 = p2[:,nonmatch]
    
    return p1, p2

def topological_sort(x, order):
    """topologically sort a partially ordered set. modified from original
    topological sort code written by Ofer Faigon (www.bitformation.com) and used
    with permission.
    
    Parameters
    ----------
    x : list
        the list to sort. items in x must be hashable.
    order : 
        a list of lists of ordered pairs of values in x. items not in x may be
        included in pairs, in which case those pairs are ignored.
    
    Returns
    -------
    y : list
        x sorted according to order
    """
    num_items = len(x)
    
    #eliminate duplicates
    order = list(set(tuple(x) for x in order))
    x = list(set(x))

    #construct lists of the descendants of each item
    num_parents = {item:0 for item in x}
    children = {item:[] for item in x}
    for pair in order:
        parent = pair[0]
        child = pair[1]
        if parent in x and child in x:
            num_parents[child] += 1
            children[parent].append(child)
    
    #construct the sorted list
    y = []
    roots = [item for item in x if num_parents[item] == 0]
    while len(roots) != 0:
        roots.sort(reverse=True)
        #add the first item with no parents to the sorted list
        item = roots.pop()
        y.append(item)
        
        #process its children
        for child in children[item]:
            num_parents[child] -= 1
            if num_parents[child] == 0:
                roots.append(child)
        del children[item]
    
    #make sure we didn't have a cyclic ordering
    if len(children) != 0:
        raise ValueError('the specified ordering is cyclic: %s' % order)
    
    return y

def setdiff2d(x, y):
    """for N x D array x and M x D array y, return a P x D array of the rows
    in x that aren't in y
    
    Parameters
    ----------
    x : ndarray
        an N x D array
    y : ndarray
        an M x D array
    
    Returns
    -------
    d : ndarray
        a P x D array of the rows in x that aren't in y
    """
    D = x.shape[1]
    
    #dtype to encode the arrays as 1D vectors, so we can use setdiff1d
    xdt = x.dtype
    dtype = [('f%d' % (i), xdt) for i in xrange(D)]
    
    xd = np.setdiff1d(x.view(dtype), y.view(dtype))
    
    return xd.view(xdt).reshape((-1,D))

def get_centered_subarray(x, shp):
    """construct a centered subarray of x with the specified shape
    
    Parameters
    ----------
    x : ndarray
        an array
    shp : tuple
        the shape of the output array to return. must be <= x.shape
    
    Returns
    -------
    sub_x : ndarray
        the specified subarray
    """
    c_x = np.round(np.array(x.shape) / 2)
    shp_half = np.round(np.array(shp) / 2)
    
    idx_start = [idx - h for idx,h in zip(c_x, shp_half)]
    idx_end = [idx_s + s for idx_s,s in zip(idx_start, shp)]
    
    return x[[slice(s, e) for s,e in zip(idx_start, idx_end)]]

def sub2ind(shape, *sub_indices):
    """like MATLAB's sub2ind: encode multiple-subscript indices in single
    linear index values
    
    Parameters
    ----------
    shape : tuple
        the shape of the array
    *sub_indices
        a set of subindex arrays (one for each dimension of the array)
    
    Returns
    -------
    idx : ndarray
        sub_indices encoded as a linear index array
    """
    #initialize the encoded index array
    idx = np.zeros(sub_indices[0].shape)
    
    #add each subindex with a multiplier
    multiplier = 1
    for sz,sub_idx in zip(shape, sub_indices):
        idx += multiplier*sub_idx
        multiplier *= sz
    
    return idx

def ind2sub(shape, idx):
    """like MATLAB's ind2sub: decode linear index values into multiple-subscript
    indices
    
    Parameters
    ----------
    shape : tuple
        the shape of the array
    idx : ndarray
        a linear index array, like that returned by sub2ind
    
    Returns
    -------
    *sub_indices
        the multiple-subscript index arrays encoded by idx
    """
    if isinstance(idx, ma.MaskedArray):
        idx = idx.data
    
    ndim = len(shape)
    
    #initialize the sub_indices list
    sub_indices = [None]*ndim
    
    #decode each dimension
    multiplier = np.prod(shape[:-1])
    for d,sz in enumerate(shape):
        sub_indices[ndim - d - 1] = (idx / multiplier).astype(int)
        idx %= multiplier
        multiplier /= sz
    
    return tuple(sub_indices)

def append_items(x, y, reorder=True):
    """append items (in place) to a list or dict. existing items are removed and
    added to the end.
    
    Parameters
    ----------
    x : list | dict
        the list or dict to append items to
    y : list | dict
        a list or dict (must match x) of items to append. items overwrite
        existing items in x.
    reorder : bool, optional
        True to force existing items to the end of lists/OrderedDicts
    """
    if isinstance(x, OrderedDict):
        for key,value in y.iteritems():
            if reorder:
                try:
                    del x[key]
                except KeyError:
                    pass
            
            x[key] = value
    if isinstance(x, dict):
        x.update(**y)
    else:
        for v in y:
            if reorder:
                try:
                    idx = x.index(v)
                    del x[idx]
                except ValueError:
                    pass
            
                x.append(v)
            elif v not in x:
                x.append(v)

def remove_items(x, y):
    """remove items (in place) from a list or dict. assumes the items exist.
    
    Parameters
    ----------
    x : list | dict
        the list or dict to remove items froms
    y : list
        for list x, a list of values to remove from x. for dict x, a list of
        keys to remove from x.
    """
    if isinstance(x, dict):
        for k in y:
            del x[k]
    else:
        for v in y:
            x.remove(v)

def ask(prompt, default=None, choices=None, num_type=float):
    """prompt the user for a response
    
    Parameters
    ----------
    prompt : string
        the prompt to show
    default : any, optional
        the default value, if no response is given
    choices : list, optional
        a list of acceptable responses. each item should either be a string
        response or one of the following:
            Number: numerical responses are acceptable
    num_type : type, optional
        the output type for numeric responses (only if Number is in choices)
    
    Returns
    -------
    response : string | num_type
        the response
    """
    if choices is not None:
        choices = to_iterable(choices)
        choices_string = ', '.join([str(choice) for choice in choices])
        number_response = Number in choices
    else:
        number_response = False
    
    err = False
    while True:
        err_msg = '(CHOOSE FROM: %s): ' % (choices_string) if err else ''
        default_msg = ' (%s)' % (default) if default is not None else ''
        
        response = raw_input('%s%s%s ' % (err_msg, prompt, default_msg))
        
        if len(response) == 0 and default is not None:
            return default
        elif choices:
            if response in choices:
                return response
            elif number_response:
                try:
                    return num_type(float(response))
                except ValueError:
                    pass
            
            err = True
        else:
            return response

def askyn(prompt, default=None):
    """prompt the user for a yes or no response
    
    Parameters
    ----------
    prompt : string
        the prompt to show
    default : bool, optional
        the default response
    
    Returns
    -------
    response : bool
        the boolean response
    """
    if default is not None:
        default = 'y' if default else 'n'
    
    response = ask(prompt, default=default, choices=['yes', 'no', 'y', 'n', 'Y', 'N'])
    
    return response.lower().startswith('y')

class PointMath(object):
    """utility math functions vectors. in each of the methods below, inputs can
    be with D-length vectors or N x D arrays of D-dimensional points
    """
    
    @classmethod
    def magnitude2(cls, p):
        """squared vector magnitude"""
        return np.sum( np.power(p, 2) , axis=-1)
    
    @classmethod
    def magnitude(cls, p):
        """vector magnitude"""
        return np.sqrt(cls.magnitude2(p))
    
    @classmethod
    def direction(cls, p):
        """unit-length vector in the same direction as p"""
        m = cls.magnitude(p)
        
        if p.ndim == 2:
            m[m==0] = 1
            m = np.tile(np.reshape(m, (-1,1)), (1, p.shape[1]))
        elif m==0:
            m = 1
        
        return p/m
    
    @classmethod
    def dot(cls, p1, p2):
        """dot product of p1 and p2"""
        return np.sum(p1*p2, axis=-1)
    
    @classmethod
    def det(cls, p1, p2):
        """determinant of p1 and p2"""
        if p1.ndim==1 and p2.ndim==1:
            p = np.vstack((p1, p2))
        else:
            if p1.ndim==1:
                p1 = np.reshape(p1, (1,-1))
            if p2.ndim==1:
                p2 = np.reshape(p2, (1,-1))
        
            shp = (p1.shape[0], 1, p1.shape[1])
            p1 = np.reshape(p1, shp)
            p2 = np.reshape(p2, shp)
            
            p = np.hstack((p1, p2))
        
        return np.linalg.det(p)
    
    @classmethod
    def angle_abs(cls, p1, p2):
        """absolute angle between p1 and p2"""
        cosa = cls.dot(cls.direction(p1), cls.direction(p2))
        return np.arccos(np.maximum(-1, np.minimum(1, cosa)))
    
    @classmethod
    def angle(cls, p1, p2):
        """counterclockwise angle between p1 and p2. i.e. if p2 is
        counterclockwise from p1, this will be positive. note that this is
        counterclockwise because we are treating points as (y,x) rather than
        (x,y).
        """
        a = cls.angle_abs(p1, p2)
        
        d = cls.det(p1, p2)
        
        if a.ndim == 0:
            return a if d < 0 else -a
        else:
            a[d<0] = -a[d<0]
            return a


class KeyboardController(object):
    """keyboard game controller that maps keys to actions"""
    #organizes groups of keys into schemes (e.g. QWERTY, NUMPAD, etc.)
    _schemes = None
    
    #records which schemes should be displayed in the legend
    _display_schemes = None
    
    #stores information about key->action mappings
    _keys = None
    
    #list of registered actions
    _actions = None
    
    #the pyglet window that listens for key events
    _window = None
    
    #line width (in characters) for printing things
    _print_width = 80
    
    #a list of actions waiting to be recognized
    _action_queue = None
    
    #list of pyglet key modifiers
    _key_modifiers = None
    _used_key_modifiers = None
    
    #generic keys to map to actions without explicitly-defined keys
    _generic_keys = '1234567890TYUIOPGHJKLBNMRFV'
    _generic_key_idx = 0
    
    #for keeping track of key-repeats
    _action_down = None
    _action_down_time = None
    _key_repeat_delay = 0.5
    
    def __init__(self, window, actions=None):
        """
        Parameters
        ----------
        window : pyglet.window.Window
            the pyglet Window that should listen for key events
        actions : list, optional
            a list of actions to watch for
        """
        check_gui_exists()

        #initialize the mapping attributes
        self._schemes = OrderedDict()
        self._display_schemes = []
        self._keys = OrderedDict()
        self._actions = []
        
        #initialize the action queue
        self._action_queue = []
        
        #get a list of pyglet key modifier attributes
        self._key_modifiers = [attr for attr in dir(pkey) if attr.startswith('MOD_')]
        self._used_key_modifiers = 0
        
        #register the default controller mapping
        self._register_controller_keys()
        
        #initial set of actions
        if actions is not None:
            self.register_actions(actions)
        
        #attach to the pyglet window
        self._window = window
        window.on_key_press = self.key_press
        window.on_key_release = self.key_release
    
    def get_action(self, timeout=None, default=None):
        """wait for an action key to be pressed, or return one from the queue
        
        Parameters
        ----------
        timeout : Number
            the maximum amount of time to wait, in seconds
        default : string
            the action to return if a timeout occurs
        
        Returns
        -------
        action : string
            the action
        """
        t_start = time.time()
        
        if timeout is None:
            timeout = float('inf')
        
        #mimic key repeat effect
        if self._action_down is not None and \
        t_start - self._action_down_time > self._key_repeat_delay:
            self._action_queue.append(self._action_down)
        
        #wait for an action key to be pressed or a timeout
        while len(self._action_queue) == 0 and time.time() < t_start + timeout:
            self._window.dispatch_events()
            time.sleep(0.001)
        
        if len(self._action_queue) == 0:  # timeout occurred
            return default
        else:  # user pressed a key
            return self._action_queue.pop()
    
    def register_key(self, key, action, mod=0, name=None, scheme='OTHER'):
        """add a key/action mapping
        
        Parameters
        ----------
        key : string | int
            the pyglet key name or code (see pyglet.window.key)
        action : string
            the mapped action name
        mod : int, optional
            the key modifier(s). see pyglet's Window.on_key_press.
        name : string, optional
            a user-friendly name for the key
        scheme : string, optional
            the key scheme to add the key to
        """
        #get the key name
        if name is None:
            if isinstance(key, basestring):
                name = key
            else:
                name = pkey.symbol_string(key)
        
        code = self._get_key_code(key, mod)
        
        #register the key
        self._keys[code] = {
            'description': self._get_key_description(name, mod),
            'name': name,
            'mod': mod,
            'action': action,
            }
        
        #mark the modifiers as used
        self._used_key_modifiers |= mod
        
        #add the key to its scheme
        self._add_key_to_scheme(code, scheme)
    
    def register_keys(self, params, scheme='OTHER'):
        """add a set of key/action mappings
        
        Parameters
        ----------
        params : list[dict]
            a list of argument dicts to pass to _register_key
        scheme : string, optional
            the key scheme to add the keys to
        """
        for param in params:
            self.register_key(scheme=scheme, **param)
    
    def register_action(self, action):
        """add an action to the action list
        
        Parameters
        ----------
        action : string
            the action to add to the list
        """
        if action in self._actions:
            return
        
        #append to the list
        self._actions.append(action)
        
        #make sure we have a key that maps to the action
        for code,param in self._keys.iteritems():
            if param['action'] == action:
                return
        
        #no existing key maps to the action. map a new key to the action.
        assert self._generic_key_idx < len(self._generic_keys), 'no more keys left to map!'
        name = self._generic_keys[self._generic_key_idx]
        key = '_'+name if name.isdigit() else name
        self.register_key(key, action, name=name)
        
        self._generic_key_idx += 1
    
    def register_actions(self, actions):
        """add actions to the action list
        
        Parameters
        ----------
        actions : list
            a list of actions to add to the list
        """
        for action in actions:
            self.register_action(action)
    
    def key_press(self, key, mod):
        """key press event handler. meant for a pyglet window's on_key_press.
        
        Parameter
        ---------
        key : int
            the key code
        mod : int
            the key modifiers that are down
        """
        mod = mod & self._used_key_modifiers
        
        action = self._get_key_action(key, mod)
        
        if action is not None and action in self._actions:
            self._action_down = action
            self._action_down_time = time.time()
            self._action_queue.append(action)
    
    def key_release(self, key, mod):
        """key release event handler. meant for a pyglet window's
        on_key_release.
        
        Parameter
        ---------
        key : int
            the key code
        mod : int
            the key modifiers that are down
        """
        self._action_down = None
    
    def print_legend(self):
        """print a key->action cheat sheet"""
        print ' KEYBOARD CONTROLS '.center(self._print_width, '=')
        
        for scheme in self._display_schemes:
            keys = self._schemes[scheme]
            self._print_key_scheme(scheme, keys)
        
        print ' KEYBOARD CONTROLS '.center(self._print_width, '=')
    
    def set_scheme_display(self, scheme, display):
        """set a scheme to either be displayed or hidden in the controller
        legend
        
        Parameters
        ----------
        scheme : string
            the name of the scheme
        display : bool
            True to show the scheme, False to hide it
        """
        if display:
            if scheme not in self._display_schemes:
                self._display_schemes.append(scheme)
        else:
            if scheme in self._display_schemes:
                self._display_schemes.remove(scheme)
    
    def _get_key_code(self, key, mod):
        """get the code associated with a keyboard key + modifiers
        
        Parameters
        ----------
        key : string | int
            the pyglet key name or code (see pyglet.window.key)
        mod : int
            the key modifier
        
        Parameters
        ----------
        code : string
            an encoding of the key + modifier
        """
        if isinstance(key, basestring):
            key = getattr(pkey, key)
        
        return '%d_%d' % (key, mod)
    
    def _get_key_description(self, name, mod):
        """get a user-friendly description of a key
        
        Parameters
        ----------
        name : string
            the key name
        mod : int
            the key modifier
        """
        desc = name
        
        #add text for the modifiers
        for key_mod in self._key_modifiers:
            if mod & getattr(pkey, key_mod):
                desc = '%s+%s' % (key_mod[4:], desc)
        
        return desc
    
    def _get_key_action(self, key, mod):
        """get the action associated with a key
        
        Parameters
        ----------
        key : string | int
            the pyglet key name or code (see pyglet.window.key)
        mod : int
            the key modifier
        """
        code = self._get_key_code(key, mod)
        
        try:
            return self._keys[code]['action']
        except KeyError:
            return None
    
    def _add_key_to_scheme(self, code, scheme):
        """add a key to a key scheme
        
        Parameters
        ----------
        key : string
            the key code
        """
        if not self._schemes.has_key(scheme):
            self._schemes[scheme] = []
            self.set_scheme_display(scheme, True)
        
        if code not in self._schemes[scheme]:
            self._schemes[scheme].append(code)
    
    def _register_controller_keys(self):
        """register the keyboard game controller keys
        """
        qwerty_scheme = [
            {'key': 'X', 'action': 'NOOP'},
            {'key': 'SPACE', 'action': 'FIRE'},
            {'key': 'W', 'action': 'UP'},
            {'key': 'S', 'action': 'DOWN'},
            {'key': 'A', 'action': 'LEFT'},
            {'key': 'D', 'action': 'RIGHT'},
            {'key': 'Q', 'action': 'UPLEFT'},
            {'key': 'E', 'action': 'UPRIGHT'},
            {'key': 'Z', 'action': 'DOWNLEFT'},
            {'key': 'C', 'action': 'DOWNRIGHT'},
            {'key': 'W', 'action': 'UPFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': 'S', 'action': 'DOWNFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': 'A', 'action': 'LEFTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': 'D', 'action': 'RIGHTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': 'Q', 'action': 'UPLEFTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': 'E', 'action': 'UPRIGHTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': 'Z', 'action': 'DOWNLEFTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': 'C', 'action': 'DOWNRIGHTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': 'SPACE', 'action': 'JUMP'},
            {'key': 'Q', 'action': 'JUMPLEFT'},
            {'key': 'E', 'action': 'JUMPRIGHT'},
        ]
        self.register_keys(qwerty_scheme, scheme='QWERTY')
        
        numpad_scheme = [
            {'key': pkey.NUM_0, 'action': 'NOOP'},
            {'key': pkey.NUM_5, 'action': 'FIRE'},
            {'key': pkey.NUM_8, 'action': 'UP'},
            {'key': pkey.NUM_2, 'action': 'DOWN'},
            {'key': pkey.NUM_4, 'action': 'LEFT'},
            {'key': pkey.NUM_6, 'action': 'RIGHT'},
            {'key': pkey.NUM_7, 'action': 'UPLEFT'},
            {'key': pkey.NUM_9, 'action': 'UPRIGHT'},
            {'key': pkey.NUM_1, 'action': 'DOWNLEFT'},
            {'key': pkey.NUM_3, 'action': 'DOWNRIGHT'},
            {'key': pkey.NUM_5, 'action': 'FIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_8, 'action': 'UPFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_2, 'action': 'DOWNFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_4, 'action': 'LEFTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_6, 'action': 'RIGHTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_7, 'action': 'UPLEFTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_9, 'action': 'UPRIGHTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_1, 'action': 'DOWNLEFTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_3, 'action': 'DOWNRIGHTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_5, 'action': 'JUMP'},
            {'key': pkey.NUM_7, 'action': 'JUMPLEFT'},
            {'key': pkey.NUM_9, 'action': 'JUMPRIGHT'},
        ]
        self.register_keys(numpad_scheme, scheme='NUMPAD')
        
        numpad_alt_scheme = [
            {'key': pkey.NUM_INSERT, 'action': 'NOOP'},
            {'key': pkey.NUM_BEGIN, 'action': 'FIRE'},
            {'key': pkey.NUM_UP, 'action': 'UP'},
            {'key': pkey.NUM_DOWN, 'action': 'DOWN'},
            {'key': pkey.NUM_LEFT, 'action': 'LEFT'},
            {'key': pkey.NUM_RIGHT, 'action': 'RIGHT'},
            {'key': pkey.NUM_HOME, 'action': 'UPLEFT'},
            {'key': pkey.NUM_PRIOR, 'action': 'UPRIGHT'},
            {'key': pkey.NUM_END, 'action': 'DOWNLEFT'},
            {'key': pkey.NUM_NEXT, 'action': 'DOWNRIGHT'},
            {'key': pkey.NUM_BEGIN, 'action': 'FIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_UP, 'action': 'UPFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_DOWN, 'action': 'DOWNFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_LEFT, 'action': 'LEFTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_RIGHT, 'action': 'RIGHTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_HOME, 'action': 'UPLEFTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_PRIOR, 'action': 'UPRIGHTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_END, 'action': 'DOWNLEFTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_NEXT, 'action': 'DOWNRIGHTFIRE', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.NUM_BEGIN, 'action': 'JUMP'},
            {'key': pkey.NUM_HOME, 'action': 'JUMPLEFT'},
            {'key': pkey.NUM_PRIOR, 'action': 'JUMPRIGHT'},
        ]
        self.register_keys(numpad_alt_scheme, scheme='NUMPAD_ALT')
        self.set_scheme_display('NUMPAD_ALT', False)
        
        arrow_scheme = [
            {'key': pkey.UP, 'action': 'UP'},
            {'key': pkey.DOWN, 'action': 'DOWN'},
            {'key': pkey.LEFT, 'action': 'LEFT'},
            {'key': pkey.RIGHT, 'action': 'RIGHT'},
            {'key': pkey.UP, 'action': 'JUMP', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.LEFT, 'action': 'JUMPLEFT', 'mod': pkey.MOD_SHIFT},
            {'key': pkey.RIGHT, 'action': 'JUMPRIGHT', 'mod': pkey.MOD_SHIFT},
        ]
        self.register_keys(arrow_scheme, scheme='ARROW')
        
        arrow_alt_scheme = [
            {'key': pkey.UP, 'action': 'UP', 'mod': pkey.MOD_FUNCTION},
            {'key': pkey.DOWN, 'action': 'DOWN', 'mod': pkey.MOD_FUNCTION},
            {'key': pkey.LEFT, 'action': 'LEFT', 'mod': pkey.MOD_FUNCTION},
            {'key': pkey.RIGHT, 'action': 'RIGHT', 'mod': pkey.MOD_FUNCTION},
            {'key': pkey.UP, 'action': 'JUMP', 'mod': pkey.MOD_SHIFT | pkey.MOD_FUNCTION},
            {'key': pkey.LEFT, 'action': 'JUMPLEFT', 'mod': pkey.MOD_SHIFT | pkey.MOD_FUNCTION},
            {'key': pkey.RIGHT, 'action': 'JUMPRIGHT', 'mod': pkey.MOD_SHIFT | pkey.MOD_FUNCTION},
        ]
        self.register_keys(arrow_alt_scheme, scheme='ARROW_ALT')
        self.set_scheme_display('ARROW_ALT', False)
    
    def _print_key_scheme(self, scheme, codes):
        """print the key->action mapping for a specific scheme
        
        Parameters
        ----------
        scheme : string
            the name of the key scheme
        codes : list
            a list of key codes
        """
        print '    %s:' % (scheme)
        for code in codes:
            desc = self._keys[code]['description']
            action = self._keys[code]['action']
            
            if action in self._actions:
                print '        %s: %s' % (desc, action)


class Debug(object):
    """use to print formatted debugging information.
    
    the primary printing method is log(), which will print the value of input
    arguments if the specified debug threshold is exceeded.
    
    test() can be used to test whether a threshold is exceeded.
    """
    class __metaclass__(type):
        """this metaclass just assigns the debug levels defined below as
        attribute constants (e.g. if 'INFO' is in levels, then every Debug
        object will have a Debug.INFO attribute)
        """
        def __new__(meta, name, bases, dct):
            cls = type.__new__(meta, name, bases, dct)
            
            for idx,level in enumerate(cls.levels):
                setattr(cls, level, idx)
            
            return cls
    
    #debug levels
    levels = ['INFO', 'WARN', 'ERROR', 'OFF']
    
    #horizontal rule characters to print for each of the above debug levels
    _hr = [None, '*', '!*', None]
    
    #the current debug printing threshold
    level = 0
    
    #keep track of some things during log printing
    _logging = False
    _logged = False
    _log_level = None
    
    #width of a line, in characters
    _line_width = 80
    
    def __init__(self, level=None):
        """
        Parameters
        ----------
        level : int | string, optional
            the debug threshold that must be met before a log message is
            displayed. must either be an integer, or a string from the levels
            list above. defaults to 'ERROR'.
        """
        if level is None:
            level = Debug.ERROR
        elif isinstance(level, basestring):
            level = self.levels.index(level)
        
        self.level = level
    
    def log(self, *args, **kwargs):
        """print a debug message
        
        Parameters
        ----------
        level : int, optional
            the threshold for printing. this level must be >= the current debug
            printing threshold for the message to be printed
        compact : bool, optional
            True to print values compactly
        hr : bool | string, optional
            True to surround the message in horizontal rules, or a string to use
            as the hr "character"
        *args
            a set of values to print
        **kwargs
            a set of values to print, with associated labels
        """
        #parse the keyword arguments
        level = kwargs.pop('level', Debug.INFO)
        compact = kwargs.pop('compact', False)
        hr = kwargs.pop('hr', False)
        
        if self.test(level):  #printing threshold is exceeded
            #start the message
            self._start_log(level, hr=hr)
            
            #print each argument
            for arg in args:
                self._print(arg, compact=compact)
            
            #print each keyword argument
            for key,value in kwargs.iteritems():
                self._print(value, prefix=key, compact=compact)
            
            #end the message
            self._end_log(hr=hr)
    
    def test(self, level):
        """test whether a debug print threshold is exceeded
        
        Parameters
        ----------
        level : int
            the treshold to test
        """
        return level >= self.level
    
    def _start_log(self, level, hr):
        """print the message header
        
        Parameters
        ----------
        level : int
            the log level
        hr : bool | string
            True to print the hr line, or the hr character
        """
        #record the start of the log
        self._logging = True
        self._logged = False
        self._log_level = level
        
        #print the default or custom hr
        if self._hr[self._log_level] is not None:  # hr defined by level
            self._print_hr(chr=self._hr[self._log_level])
        elif hr:  # custom hr
            self._print_hr(chr=hr)
    
    def _end_log(self, hr):
        """print the message footer
        
        Parameters
        ----------
        hr : bool | string
            True to print the hr line, or the hr character
        """
        if self._hr[self._log_level] is not None:  # hr defined by level
            self._print_hr(chr=self._hr[self._log_level])
        elif hr:  # custom hr
            self._print_hr(chr=hr)
        
        #record the end of logging
        self._logging = False
    
    def _get_time(self):
        """get a string-formatted timestamp"""
        return datetime.now().strftime('%H:%M:%S.%f')
    
    def _get_debug_prefix(self):
        """get the prefix for a debug message"""
        return '%s: DEBUG(%s)' % (self._get_time(), self.levels[self._log_level])
    
    def _print(self, x, prefix=None, indent=0, compact=False):
        """print a value
        
        Parameters
        ----------
        x : any
            the value
        prefix : string, optional
            a prefix to print before the value
        indent : int, optional
            the indentation level
        compact : bool, optional
            True to print the value compactly
        """
        if isinstance(x, dict) and not compact:
            self._print_dict(x, prefix=prefix, indent=indent)
        else:
            self._print_string(x, prefix=prefix, indent=indent)
    
    def _print_string(self, x, prefix=None, indent=0):
        """print a value as a string
        
        Parameters
        ----------
        x : any
            the value
        prefix : string, optional
            a prefix to print before the value
        indent : int, optional
            the indentation level
        """
        if not self._logged:  # print a message prefix first
            prefix = '%s%s' % (self._get_debug_prefix(), (' %s' % (prefix)) if prefix is not None else '')
        else:
            indent = indent+1
        
        prefix = '    '*indent + (('%s: ' % (prefix)) if prefix is not None else '')
        print '%s%s' % (prefix, self._to_string(x))
        
        self._logged = True
    
    def _print_dict(self, dct, prefix=None, indent=0):
        """print a dict
        
        Parameters
        ----------
        dct : dict
            the dict to print
        prefix : string, optional
            a prefix to print before the value
        indent : int, optional
            the indentation level
        """
        if not self._logged:  # print a message prefix line
                self._print_string('', indent=indent)
        
        if prefix is not None:  # print a dict prefix line
            self._print_string('', prefix=prefix, indent=indent)
            indent += 1
        
        #print each dict item
        for key,value in dct.iteritems():
            self._print(value, prefix=key, indent=indent)
    
    def _print_hr(self, chr=True):
        """print a horizontal rule
        
        Parameters
        ----------
        chr : bool | string
            True if the default hr character should be used, or a string to use
            as the hr "character"
        """
        if chr is True:
            chr = '-'
        
        print chr*(self._line_width/len(chr))
    
    def _to_string(self, x, **kwargs):
        """convert a value to a string for printing
        
        Parameters
        ----------
        x : any
            a value
        **kwargs
            see the *_to_string methods below
        """
        if isinstance(x, list):
            return self._list_to_string(x, **kwargs)
        elif isinstance(x, tuple):
            return self._tuple_to_string(x, **kwargs)
        elif isinstance(x, dict):
            return self._dict_to_string(x, **kwargs)
        else:
            return '%s' % (x)
    
    def _list_to_string(self, lst, list_brackets=False, **kwargs):
        """convert a list to a string representation
        
        Parameters
        ----------
        lst : list
            the list
        list_brackets : bool, optional
            True to print square brackets around the list items
        **kwargs
            ignored
        """
        x = ' '.join([self._to_string(x, list_brackets=isinstance(x, list)) for x in lst])
        
        if list_brackets:
            return '[%s]' % (x)
        else:
            return x
    
    def _tuple_to_string(self, tpl, tuple_brackets=True, **kwargs):
        """convert a tuple to a string representation
        
        Parameters
        ----------
        tpl : tuple
            the tuple
        tuple_brackets : bool, optional
            True to print parentheses around the tuple items
        **kwargs
            ignored
        """
        x = ' '.join([self._to_string(x, tuple_brackets=isinstance(x, tuple)) for x in tpl])
        
        if tuple_brackets:
            return '(%s)' % (x)
        else:
            return x
    
    def _dict_to_string(self, dct, dict_brackets=False, **kwargs):
        """convert a dict to a string representation
        
        Parameters
        ----------
        dct : dict
            the dict
        dict_brackets : bool, optional
            True to print curly brackets around the dict items
        **kwargs
            ignored
        """
        x = ', '.join(['%s: %s' % (key, self._to_string(x, dict_brackets=isinstance(x, dict))) for key,x in dct.iteritems()])
        
        if dict_brackets:
            return '{%s}' % (x)
        else:
            return x


class SerializableMaskedArray(ma.MaskedArray):
    """A masked array that can be pickled. Workaround for a numpy bug."""
    def __reduce__(self):
        """One way to define custom pickling behavior.

        Returns
        -------
        cls : a callable
        args : tuple
            Arguments to the callable.
        """
        return (SerializableMaskedArray, (self.data, self.mask, self.dtype, False, True, 0, self.fill_value))

    def __deepcopy__(self, memo):
        """Custom deepcopy() behavior.

        Parameters
        -------
        memo : the memoization dictionary
        """
        data = deepcopy(self.data, memo)
        mask = deepcopy(self.mask, memo)
        dtype = deepcopy(self.dtype, memo)
        fill_value = deepcopy(self.fill_value, memo)
        rv = SerializableMaskedArray(data, mask, dtype=dtype, fill_value=fill_value)
        memo[id(self)] = rv
        return rv


class CustomOrderedDict(OrderedDict):
    """An ordered dict that doesn't break deepcopy or pickling"""
    def __reduce__(self):
        """One way to define custom pickling behavior.

        Returns
        -------
        cls : a callable
        args : tuple
            Arguments to the callable. (Empty)
        state : dict
            (Empty) dict of values to be added to __dict__
        list_items : generator
            (Empty) iterator containing list items to be added
        dict_items : generator
            Iterator containing (key, value) dict items to be added
        """
        return (CustomOrderedDict, (), dict(), (x for x in ()), self.iteritems())

    def __deepcopy__(self, memo):
        """Deepcopy the ordered dict.

        Parameters
        ----------
        memo : dict
            The memoization dict.
        """
        rv = CustomOrderedDict()
        memo[id(self)] = rv

        for k, v in self.iteritems():
            rv[k] = deepcopy(v, memo)

        return rv

