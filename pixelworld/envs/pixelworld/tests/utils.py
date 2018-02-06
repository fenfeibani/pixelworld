"""
Utility functions for unit tests.
"""

from numbers import Number
import numpy as np
import sys
import pdb
import functools
import traceback


def debug_on(*exceptions):
    """A decorator for unit tests that enters pdb on any of the listed exceptions.

    Parameters
    ----------
    exceptions (as args) : Exception
        Exceptions to catch and enter pdb on. Defaults to just AssertionError.

    Returns
    -------
    decorator : decorator
        A function decorator to apply to a function.
    """
    if not exceptions:
        exceptions = (AssertionError, )

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except exceptions:
                info = sys.exc_info()
                traceback.print_exception(*info) 
                pdb.post_mortem(info[2])
        return wrapper
    return decorator


def check_deepcopy(a, b, memo=None, persistent_ids=True):
    """Check that deepcopy made two independent things which are isomorphic.

    Parameters
    ----------
    a : anything
        The first object to be compared.
    b : anything
        The second object to be compared, which should be a deepcopy of the first.
    memo : dict, optional
        Dict mapping id's of components of a to id's of components of b. This
        should be omitted if you are using this function non-recursively.
    persistent_ids : bool
        If True, enforce that ids persist over time. Only set to False when we
        are creating temporary objects by recursing into a numpy array.
    """
    if memo is None:
        memo = dict()

    if id(a) in memo:
        # check that mutable things have different ids and that they
        # persist over time
        if not hasattr(a, '__hash__') or (a is not np.ma.masked and a.__hash__ is None):
            assert id(a) != id(b)
            if persistent_ids:
                assert id(b) == memo[id(a)], '%s %s' % (a, b)
        return

    # remember that a and b are matched
    memo[id(a)] = id(b)

    # switch based on type of a, check that b is isomorphic to a and recurse
    # into their component parts if they have any

    # None
    if a is None:
        assert b is None

    # numpy masked
    elif a is np.ma.masked:
        assert b is np.ma.masked

    # bool
    elif isinstance(a, bool):
        assert isinstance(b, bool)
        assert a == b

    # numpy bool
    elif isinstance(a, np.bool_):
        assert isinstance(b, np.bool_)
        assert a == b

    # number
    elif isinstance(a, Number):
        assert isinstance(b, Number)
        assert a == b

    # string
    elif isinstance(a, str):
        assert isinstance(b, str)
        assert a == b

    # dict
    elif isinstance(a, dict):
        assert id(a) != id(b)
        assert isinstance(b, dict)
        assert sorted(a.keys()) == sorted(b.keys())

        # recurse into values
        for k in a:
            check_deepcopy(a[k], b[k], memo, persistent_ids=persistent_ids)

    # list
    elif isinstance(a, list):
        assert id(a) != id(b)
        assert isinstance(b, list)
        assert len(a) == len(b)

        # recurse into list elements
        for u, v in zip(a, b):
            check_deepcopy(u, v, memo, persistent_ids=persistent_ids)

    # tuple
    elif isinstance(a, tuple):
        assert isinstance(b, tuple)
        assert len(a) == len(b)

        # recurse into tuple elements
        for u, v in zip(a, b):
            check_deepcopy(u, v, memo, persistent_ids=persistent_ids)

    # numpy array
    elif isinstance(a, np.ndarray):
        assert id(a) != id(b)
        assert isinstance(b, np.ndarray)
        assert len(a) == len(b)

        # recurse into lower-dimensional slices or into elements, setting
        # persistent_ids to False because we're creating new arrays as we do
        # this
        if a.ndim > 0:
            for i in xrange(len(a)):
                check_deepcopy(a[i], b[i], memo, persistent_ids=False)

        # masked arrays produce mvoids of dimension 0, which we handle specially 
        else:
            check_deepcopy([x for x in a._data], [x for x in b._data], memo, persistent_ids=persistent_ids)
            check_deepcopy([x for x in a._mask], [x for x in b._mask], memo, persistent_ids=persistent_ids)
            check_deepcopy(a.dtype, b.dtype, memo, persistent_ids=persistent_ids)

    # numpy dtype
    elif isinstance(a, np.dtype):
        assert a == b

    # these objects have no __dict__ for some reason
    elif isinstance(a, np.random.mtrand.RandomState):
        assert isinstance(b, np.random.mtrand.RandomState)
        # different calls to get_state() can result in different objects which
        # have the same value, so set persistent_ids to False
        check_deepcopy(a.get_state(), b.get_state(), memo, persistent_ids=False)

    # standard objects with __dict__
    elif hasattr(a, '__dict__'):
        assert id(a) != id(b)
        assert a.__class__ == b.__class__, '%s %s' % (a.__class__, b.__class__)
        assert sorted(a.__dict__.keys()) == sorted(b.__dict__.keys()), '%s %s' % (
            sorted(a.__dict__.keys()), sorted(b.__dict__.keys()))

        for k in a.__dict__:
            check_deepcopy(a.__dict__[k], b.__dict__[k], memo, persistent_ids=persistent_ids)

    # default case, assume we're good
    else:
        pass  # assert False, 'Unrecognized thing: %s' % a
