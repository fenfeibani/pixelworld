from __future__ import print_function
from __future__ import absolute_import

import numpy as np

#from pixelworld.envs.pixelworld.core import ListSpace

from gym.spaces import Discrete, Box


class NamedDiscrete(Discrete):
    """
    Discrete, but with named elements.
    """
    def __init__(self, n, names=[]):
        names = list(names)
        if n != len(names):
            raise Exception("One name per element, please.")
        if n != len(set(names)):
            raise Exception("Names must be unique.")
        super(NamedDiscrete, self).__init__(n)
        self.names = names

    @staticmethod
    def from_unnamed(space, prefix=None):
        names = range(space.n)
        if prefix is not None:
            names = ["%s%s" % (prefix, name) for name in names]
        return NamedDiscrete(space.n, names=names)

    def __repr__(self):
        return "NamedDiscrete(%d, %s)" % (self.n, self.names)
    def __eq__(self, other):
        return self.n == other.n and self.names == other.names
    def __ne__(self, other):
        return not (self == other)


class NamedBox(Box):
    """
    Box, but with named dimensions.
    """
    def __init__(self, low, high, shape=None, names=[]):
        super(NamedBox, self).__init__(low, high, shape=shape)
        names = list(names)
        if self.low.size != len(names):
            raise Exception("One name per component, please.")
        if self.low.size != len(set(names)):
            raise Exception("Names must be unique.")
        self.names = names

    @staticmethod
    def from_unnamed(space, prefix=""):
        shape = space.low.shape
        if len(shape) == 1:
            names = range(shape[0])
            if len(prefix) > 0:
                names = ["%s%s" % (prefix, name) for name in names]
        else:
            names = zip(*np.unravel_index(range(np.prod(shape)), shape))
            if len(prefix) > 0:
                names = [prefix + '.'.join([str(c) for c in cs]) for cs in coords]
        return NamedBox(space.low, space.high, names=names)

    @property
    def shape(self):
        return self.low.shape
    def __repr__(self):
        return "NamedBox(%s, %s)" % (self.shape, self.names)
    def __eq__(self, other):
        return (self.shape == other.shape 
                and np.allclose(self.low, other.low) 
                and np.allclose(self.high, other.high) 
                and self.names == other.names)
    def __ne__(self, other):
        return not (self == other)


def ensure_named(space):
    """Ensure that the give space is named."""
    if isinstance(space, (NamedBox, NamedDiscrete)):
        return space
    elif isinstance(space, Box):
        return NamedBox.from_unnamed(space)
    elif isinstance(space, Discrete):
        return NamedDiscrete.from_unnamed(space)
#    elif isinstance(space, ListSpace):
#        return NamedDiscrete(len(space._list), names=space._list)
    else:
        raise Exception("Cannot convert space of type %s into a named gym space"
            % (type(space),))
