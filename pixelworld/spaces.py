from __future__ import print_function
from __future__ import absolute_import

import numpy as np

import gym
import rllab

from pixelworld.envs.pixelworld.core import ListSpace
from pixelworld.spaces_gym import Discrete, Box, NamedDiscrete, NamedBox
from pixelworld import spaces_gym, spaces_rllab


def flatcat_spaces(*spaces):
    """Flatten and concatenate the given spaces. Names are contagious."""
    discrete = None
    named = False
    names = []
    assert len(spaces) > 0

    # TODO: is there a reason to make the names of dimensions unique?
    # Perhaps Box's and Discrete's should have their names prepended,
    # or perhaps environments should already have done this.


    for space in spaces:
        if isinstance(space, NamedBox):
            if discrete == True:
                raise Exception("Spaces must be all Box or all Discrete.")
            discrete = False
            named = True
            names.extend(space.names)
        elif isinstance(space, NamedDiscrete):
            if discrete == False:
                raise Exception("Spaces must be all Box or all Discrete.")
            discrete = True
            named = True
            names.extend(space.names)
        elif isinstance(space, Box):
            if discrete == True:
                raise Exception("Spaces must be all Box or all Discrete.")
            discrete = False
            names.extend(NamedBox.from_unnamed(space).names)
            #names.extend(range(len(names), len(names) + len(space.low.flatten())))
        elif isinstance(space, Discrete):
            if discrete == False:
                raise Exception("Spaces must be all Box or all Discrete.")
            discrete = True
            names.extend(NamedDiscrete.from_unnamed(space).names)
            #names.extend(range(len(names), len(names) + space.n))
        else:    
            raise Exception("Space not a gym Box or Discrete.")
    for idx, name in enumerate(names):
        if isinstance(name, int):
            assert name == idx
    
    if discrete:
        n = sum([space.n for space in spaces])
        assert len(names) == n
        if named:
            return NamedDiscrete(n, names=names)
        else:
            return Discrete(n)
    else:
        assert sum([len(space.low.flatten()) for space in spaces]) == len(names)
        low = np.concatenate([space.low.flatten() for space in spaces])
        high = np.concatenate([space.high.flatten() for space in spaces])
        if named:
            return NamedBox(low=low, high=high, names=names)
        else:
            return Box(low=low, high=high)


def to_gym_space(space):
    if isinstance(space, spaces_rllab.NamedBox):
        return spaces_gym.NamedBox(low=space.low, high=space.high, names=space.names)
    elif isinstance(space, spaces_rllab.NamedDiscrete):
        return spaces_gym.NamedDiscrete(n=space.n, names=space.names)
    elif isinstance(space, rllab.spaces.Box):
        return gym.spaces.Box(low=space.low, high=space.high)
    elif isinstance(space, rllab.spaces.Discrete):
        return gym.spaces.Discrete(n=space.n)
    else:
        raise NotImplementedError


def to_rllab_space(space):
    if isinstance(space, spaces_gym.NamedBox):
        return spaces_rllab.NamedBox(low=space.low, high=space.high, names=space.names)
    elif isinstance(space, spaces_gym.NamedDiscrete):
        return spaces_rllab.NamedDiscrete(n=space.n, names=space.names)
    elif isinstance(space, gym.spaces.Box):
        return rllab.spaces.Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return rllab.spaces.Discrete(n=space.n)
    elif isinstance(space, ListSpace):
       return spaces_rllab.NamedDiscrete(n=len(space._list), names=space._list)
    else:
        raise NotImplementedError


def project_box(src_space=None, dest_space=None, src_val=None, bindings=None):
    """Project a value from one Box space to a smaller one.
    If src_val is not None, return dest_val using bindings if it is supplied.
    If src_val is None, check space compatibility and return the bindings.

    src_val may be an array or list."""
    
    if bindings is None:
        if src_space is None or not (isinstance(src_space, NamedBox) or isinstance(src_space, spaces_rllab.NamedBox)):
            raise Exception("Source space must be a NamedBox")
        if dest_space is None or not (isinstance(dest_space, NamedBox) or isinstance(dest_space, spaces_rllab.NamedBox)):
            raise Exception("Destination space must be a NamedBox")
        if not set(dest_space.names) <= set(src_space.names):
            raise Exception("Source space is missing components of destination space: %s." 
                                % (list(set(dest_space.names) - set(src_space.names)),))

        bindings = [src_space.names.index(name) for name in dest_space.names]
        bindings = np.array(bindings, np.intp)

    if src_val is None:
        return bindings
    else:
        dest_val = np.asarray(src_val)[...,bindings]
        return dest_val


def inject_discrete(src_space=None, dest_space=None, src_val=None, bindings=None):
    """Inject a value from one Discrete space into a larger one.
    If src_val is not None, return dest_val using bindings if it is supplied.
    If src_val is None, check space compatibility and return the bindings.

    src_val may be an array or list."""

    if bindings is None:
        if src_space is None or not (isinstance(src_space, NamedDiscrete) or isinstance(src_space, spaces_rllab.NamedDiscrete)):
            raise Exception("Source space must be a NamedDiscrete")
        if dest_space is None or not (isinstance(dest_space, NamedDiscrete) or isinstance(dest_space, spaces_rllab.NamedDiscrete)):
            raise Exception("Destination space must be a NamedDiscrete")
        if not set(src_space.names) <= set(dest_space.names):
            raise Exception("Destination space is missing elements of source space: %s."
                                % (list(set(src_space.names) - set(dest_space.names)),))

        bindings = [dest_space.names.index(name) for name in src_space.names]
        bindings = np.array(bindings, np.int)

    if src_val is None:
        return bindings
    else:
        dest_val = bindings[src_val]
        return dest_val


if __name__ == "__main__":
    src = NamedDiscrete(2, ["C", "A"])
    dest = NamedDiscrete(3, ["A", "B", "C"])

    print(inject_discrete(src, dest))
    print(inject_discrete(src, dest, 0))
    print(inject_discrete(src, dest, 1))
    print(inject_discrete(src, dest, [1,1,0,0,0,0,0,1,1])) 
    print(inject_discrete(src, dest, np.array([[1,1,1], [0,0,1]]))) 
    print(inject_discrete(src, dest, [1,1,0,0,0,0,0,1,1], bindings=inject_discrete(src, dest)))

    src2 = NamedBox(0, 1, (3,), ["A", "B", "C"])
    dest2 = NamedBox(0, 1, (2,), ["C", "A"])

    print(project_box(src2, dest2))
    print(project_box(src2, dest2, [3, 7, 21]))
    print(project_box(src2, dest2, [[1,2,3], [2,4,6], [5,6,7]]))
