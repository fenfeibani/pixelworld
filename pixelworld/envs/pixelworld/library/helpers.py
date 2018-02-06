'''
    helper classes and functions for defining library resources
'''
import sys
import importlib

import numpy as np

import core
from pixelworld.envs.pixelworld import base_entities
from pixelworld.envs.pixelworld.utils import is_iterable, pluralize

#for t parameters
from numpy import arange as t

class Lazy(object):
    """use to specify a value that is not resolved until world creation time
    and that optionally depends on parameters that may be overridden by the
    process that is creating the world. universe.load_parameters resolves these
    into values.
    """
    class __metaclass__(type):
        """this meta-class dynamically overloads a set of standard operators to
        support lazy resolution
        """
        __overloaded_binary_operators = ['eq', 'ne', 'lt' ,'gt', 'le', 'ge',
            'add', 'sub', 'mul', 'floordiv', 'div', 'truediv', 'mod', 'divmod',
            'pow', 'lshift', 'rshift', 'and', 'or', 'xor', 'radd', 'rsub',
            'rmul', 'rfloordiv', 'rdiv', 'rtruediv' ,'rmod', 'rdivmod', 'rpow',
            'rlshift', 'rand', 'ror', 'rxor', 'iadd', 'isub', 'imul',
            'ifloordiv', 'idiv', 'itruediv', 'imod', 'ipow', 'ilshift',
            'irshift', 'iand', 'ior', 'ixor', 'coerce']
        
        __overloaded_unary_operators = ['pos', 'neg', 'abs', 'invert', 'round',
            'floor', 'ceil', 'trunc', 'int', 'long', 'float', 'complex', 'oct',
            'hex', 'index']
        
        def __new__(meta, name, bases, dct):
            """this overrides a set of the class' operators with methods that
            delay the operation until the attribute value is resolved."""
            cls = type(name, bases, dct)
            
            def do_binary_op(op, x, other):
                """actually carry out a binary operation"""
                try:
                    op_func = getattr(x, '__%s__' % (op))
                    y = op_func(other)
                except (AttributeError, TypeError):
                    y = NotImplemented
                
                if y is NotImplemented:
                    try:
                        if op[0] == 'r' and op != 'round':
                            op_func = getattr(other, '__%s__' % (op[1:]))
                        else:
                            op_func = getattr(other, '__r%s__' % (op))
                        
                        y = op_func(x)
                    except (AttributeError, TypeError):
                        y = NotImplemented
                    
                    if y is NotImplemented:
                        raise TypeError('cannot perform %s on %s and %s' % (op, type(x).__name__, type(other).__name__))
                
                return y
            
            def do_unary_op(op, x):
                """actually carry out a unary operation"""
                try:
                    op_func = getattr(x, '__%s__' % (op))
                    y = op_func()
                except (AttributeError, TypeError):
                    y = NotImplemented
                
                if y is NotImplemented:
                    raise TypeError('cannot perform %s on %s' % (op, type(x).__name__))
                
                return y
            
            def get_operator(op, binary):
                """construct a lazy version of the specified operator
                
                Parameters
                ----------
                op : string
                    the operator name
                binary : bool
                    True if the operator is binary, False if it is unary
                
                Returns
                -------
                f : function
                    the overloaded operator
                """
                if binary:
                    def operator(self, other):
                        lazy_func = lambda params: do_binary_op(op, Lazy.resolve(self, params=params), other)
                        lazy_default = do_binary_op(op, self._default, other)
                        
                        return Lazy(lazy_func, default=lazy_default)
                else:
                    def operator(self):
                        lazy_func = lambda params: do_unary_op(op, Lazy.resolve(self, params=params))
                        lazy_default = do_unary_op(op, self._default)
                        
                        return Lazy(lazy_func, default=lazy_default)
                
                return operator
            
            #override each binary operator
            for op in meta.__overloaded_binary_operators:
                setattr(cls, '__%s__' % (op), get_operator(op, binary=True))
            
            #override each unary operator
            for op in meta.__overloaded_unary_operators:
                setattr(cls, '__%s__' % (op), get_operator(op, binary=False))
            
            return cls
    
    #the lazy value
    _value = None
    
    #the default resolved value
    _default = None
    
    @classmethod
    def resolve(cls, value, params=None):
        """resolve a value that may include Lazy values
        
        Parameters
        ----------
        value : any
            the value to resolve
        params : dict, optional
            a dict of parameters that may be used to determine resolved values
        
        Returns
        -------
        value : any
            the resolved value
        """
        if params is None:
            params = {}
        
        if isinstance(value, Lazy):  # Lazy value
            if isinstance(value._value, basestring):  # name of a parameter
                value = params.get(value._value, value._default)
            elif callable(value._value):  # function that returns a value
                default = value._default
                value = value._value(params)
                if value is None:
                    value = default
            else:
                value = value._value
            
            value = Lazy.resolve(value, params=params)
        elif isinstance(value, dict):  # dict of possibly Lazy values
            for key in value:
                value[key] = Lazy.resolve(value[key], params=params)
        elif isinstance(value, list):  # list of possibly Lazy values
            for idx,v in enumerate(value):
                value[idx] = Lazy.resolve(v, params=params)
        elif isinstance(value, tuple):  # tuple of possibly Lazy values
            value = tuple(Lazy.resolve(list(value), params=params))
        
        return value
    
    def __init__(self, value, default=None):
        """
        Parameters
        ----------
        value : string | callable
            either the name of a parameter that may be specified by the process
            that creates the world, or a function that takes a dict of
            parameters and returns a value. if a parameter name is specified and
            the parameter is not in the list of parameters passed when resolving
            the value, or a function is specified and it return None, then the
            default value is used instead.
        default : any
            the default parameter value
        """
        self._value = value
        self._default = default
    
    def __repr__(self):
        if callable(self._value):
            value = type(self._value)
        else:
            value = self._value
        
        return '%s: %s' % (self.__class__.__name__, value)


class LazyChoose(Lazy):
    """lazily choose between two values, based on a boolean parameter"""
    #the lazy test value
    _test = None
    
    #the value to choose if the test resolves to True
    _value_yes = None
    
    #the value to choose if the test resolves to False
    _value_no = None
    
    def __init__(self, test, yes, no, default=True):
        """
        Parameters
        ----------
        test : string | Lazy
            the name of the parameter to base the choice on, or a Lazy value
        yes : any
            the value to choose if the parameter is True
        no : any
            the value to choose if the parameter is False
        default : bool
            the default value for the parameter if it is unspecified
        """
        self._test = test
        self._value_yes = yes
        self._value_no = no
        
        default = self._value_yes if default else self._value_no
        
        super(LazyChoose, self).__init__(self._choose_value, default=default)
    
    def _choose_value(self, params):
        #resolve the test value
        test_value = Lazy.resolve(self._test, params=params)
        
        if test_value is None:  # use the default
            return None
        else:
            return self._value_yes if test_value else self._value_no


#helpers for compactly defining Entity classes
def define_entity(entity, name, bases=None, dct=None):
    """define an Entity class.
    
    Parameters
    ----------
    entity : string | TypeFamily
        the name or type of Entity to create
    name : string
        the name of the Entity
    bases : string | TypeFamily | list[string | TypeFamily], optional
        a list of Entity classes or names to use as bases for the new Entity
    dct : dict, optional
        a dict of class attributes to set
    
    Returns
    -------
    entity : TypeFamily
        the new Entity class
    """
    if isinstance(entity, basestring):
        entity = base_entities[entity]
    
    if bases is None:
        bases = [entity]
    if not is_iterable(bases):
        bases = [bases]
    
    if dct is None:
        dct = {}
    
    #resolve the base Entities to their classes
    bases = tuple([entity.get_class(cls) for cls in bases])
    
    #create the class
    return entity.__metaclass__(name, bases, dct)

def define_world_attribute(name, dct=None, **kwargs):
    """define a WorldAttribute class.
    
    Parameters
    ----------
    name : string
        the name of the WorldAttribute
    default : any, optional
        the default value of the WorldAttribute
    dct : dict, optional
        a dict of class attributes to set
    **kwargs
        extra keyword arguments to define_entity()
    
    Returns
    -------
    entity : TypeFamily
        the new Entity class
    """
    if dct is None: dct = {}
    
    if 'default' in kwargs and '_default_value' not in dct:
        dct['_default_value'] = kwargs.pop('default')
    
    return define_entity('world_attribute', name, dct=dct, **kwargs)

def define_entities(entity, specs):
    """define a set of Entity classes.
    
    Parameters
    ----------
    entity : string | TypeFamily
        the name or type of Entity to create
    specs : dict
        a dict that maps the new Entities' names to kwarg dicts to
        define_entity() above
    
    Returns
    -------
    entities : dict
        a dict mapping the new Entity names to their Entity classes
    """
    #try get a custom entity definer
    entity_name = entity if isinstance(entity, basestring) else entity._name
    entity_definer = globals().get('define_%s' % (entity_name), None)
    
    if entity_definer is None:  # use the default definer
        return {name:define_entity(entity, name, **kwargs) for name,kwargs in specs.iteritems()}
    else:
        return {name:entity_definer(name, **kwargs) for name,kwargs in specs.iteritems()}

#generate specialized define_entities Entity generators for each Entity type.
#these will be accessible as e.g. define_objects(specs), define_events(specs),
#etc.
def define_entities_generator(entity):
    def entity_definer(specs):
        return define_entities(entity, specs)
    return entity_definer
for name,entity in base_entities.iteritems():
    if name != 'entity':
        vars()['define_' + pluralize(name, 2)] = define_entities_generator(entity)


#import the section _helpers so they are accessible via their section name
#(e.g. sprite/_helpers.py is available as helpers.sprite)
for section in core.sections():
    vars()[section] = core.import_item(section, '_helpers')


#convenience synonyms
L = Lazy
LC = LazyChoose
#define h as a synonym for this module, allowing from helpers import h
h = sys.modules[__name__]
