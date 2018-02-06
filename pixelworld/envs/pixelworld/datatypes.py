'''
    data type definitions for PixelWorld
'''
import warnings
import inspect
import re
from copy import copy, deepcopy
from numbers import Number
from collections import OrderedDict
import cPickle as pickle, marshal, types

import numpy as np
from numpy import ma

from utils import is_iterable, to_iterable, fix_float_integer, \
    merge_structured_arrays, append_items, remove_items, SerializableMaskedArray


class TypeFamily(type):
    """this metaclass organizes classes into families that are accessible to
    each other via the _class_family dict attribute that all family members
    share. root family members should set the value of _class_family to the
    name of their family, which will be assigned to each class'
    _class_family_name attribute. at class construction time the class family
    dict will be assigned to _class_family. each family member assigns itself a
    name via the _name attribute, and at class construction time the class is
    added to the _class_family at the key corresponding to its name.
    
    each class can optionally assign itself a _class_type attribute, which by
    convention should be an uppercase string. at class construction time, the
    value of this attribute then itself becomes an integer attribute of all
    family members, and the _class_type attribute of the class is reassigned as
    this integer value. for instance, the Object class in core.py sets
    _class_type = 'SIMPLE'. this means that all classes in the same family as
    Object will have an attribute cls.SIMPLE, and Object and all of Object's
    subclasses that don't themselves set _class_type will have
    cls._class_type == Object.SIMPLE. similarly, CompoundObject defines
    _class_type = 'COMPOUND'. in this example, this provdes a simple way to test
    whether objects are simple or compound, e.g.:
        if obj._class_type == Object.COMPOUND: ...
    
    each root family member can define an attribute, named
    _class_tree_attributes, that specifies a list of attributes that are either
    lists or dicts and whose elements are aggregated from the first class to
    define the attribute down to each subclass, based on each class' own
    definition of that attribute. each class can also define an accompanying
    attribute with the suffix '_removed', which should be a list of values or
    keys that should be removed from the aggregate list/dict at that point in
    the class tree. e.g. Object defines such an attribute named _attributes,
    which defines a list of object attributes that each Object class adds beyond
    those added by the object's superclasses. so, if Object defines
    _attributes = ['name', 'id', ...], then BasicObject subclasses from Object
    and defines _attributes = ['color', 'mass', ...], then UnpushableObject
    subclasses from BasicObject and defines _attributes_removed = ['mass', ...],
    and finally SelfObject subclasses from UnpushableObject and defines
    _attributes = ['pushes'], then after class construction, SelfObject will
    have _attributes = ['name', 'id', 'color', 'pushes', ...]. note that
    _class_tree_attributes itself is a tree attribute. this feature is used
    extensively to define attributes that are defined collectively by the class
    tree.
    
    finally, each root family member can define an attribute, named
    _class_shared_attributes, that specifies a list of attributes that are
    either lists or dicts, and whose elements should be aggregated into a
    common list/dict that is shared by all members of the class family. at class
    construction, this common list/dict replaces those defined by each class. no
    classes use this feature in the current version.
    
    """
    _all_class_families = {}
    _all_class_types = {}
    
    _class_shared_attributes = []
    _class_tree_attributes = ['_class_tree_attributes']
    
    #keeps track of class-hierarchy snapshots and attributes that should be
    #included in them (see _save_class_hierarchy_snapshot() below). to include
    #an attribute in snapshots, append a list to _class_snapshot_attributes that
    #takes the form [metaclass, name]. this is done below immediately after the
    #TypeFamily definition (uck, messy, but this whole class family thing is
    #already bending the rules quite a bit).
    _class_hierarchy_snapshot_attributes = []
    _class_hierarchy_snapshots = None
    
    @classmethod
    def _generate_class_family(meta, cls):
        """generate a class family name for root member classes that did not
        explicitly assign the _class_family attribute
        
        Parameters
        ----------
        cls : TypeFamily
            the class
        
        Returns
        -------
        name : string
            the generated class family name
        """
        return cls.__name__.lower()
    
    @classmethod
    def _generate_class_name(meta, cls):
        """generate a class name for members that do not explicitly assign the
        _name attribute. if the class' defined name follows a camel-case
        convention with a name prepended onto the root family member name
        (e.g. KineticEnergyObjectAttribute, as a member of a family started by
        the ObjectAttribute class), then the generated name will be a snake
        case version of the just the prefix (in the current example,
        'kinetic_energy').
        
        Parameters
        ----------
        cls : TypeFamily
            the class
        
        Returns
        -------
        name : string
            the generated class name
        """
        #base the name of the class' defined name
        name = cls.__name__
        
        #remove the root member's name if it was added as a suffix
        if len(cls._class_family) > 0:
            root_name = cls._class_family.values()[0].__name__
            if name.endswith(root_name):
                name = name[:-len(root_name)]
        
        #convert camel to snake case
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
        
        return name
    
    @classmethod
    def _get_class_sequence(meta, cls, class_family=True, class_sequence=None):
        """generate a list of super classes that descend from either the root
        family member or the base class to the specified class
        
        Parameters
        ----------
        cls : TypeFamily
            the class
        class_family : bool, optional
            True to start at the root family member (False to start at the base
            class)
        class_sequence : list, optional
            used internally to compile the sequence
        
        Returns
        -------
        class_sequence : list
            the class sequence list
        """
        if class_sequence is None:
            class_sequence = []
        
        #get the class sequence of the base classes
        for base_cls in cls.__bases__:
            if not class_family or base_cls in cls._class_family.values():
                meta._get_class_sequence(base_cls, class_family=class_family, class_sequence=class_sequence)
        
        #add the class itself if it doesn't exist already
        if not cls in class_sequence:
            class_sequence.append(cls)
        
        return class_sequence
    
    @classmethod
    def _compile_tree_attribute(meta, cls, attr, dct):
        """compile a tree attribute from the values defined collectively by a
        class family (see description above), and assign the tree attribute to
        the class
        
        Parameters
        ----------
        cls : TypeFamily
            the class family member
        attr : string
            the tree attribute to compile. each family member should either not
            explicitly define the attribute, or define it as a list or a dict
            (consistent across all family members).
        dct : dict
            the dict passed to the metaclass
        """
        #make sure the attribute exists
        if not hasattr(cls, attr):
            return None
        
        #keep a record of the values defined by the class itself
        attr_added = '%s_added' % (attr)
        attr_removed = '%s_removed' % (attr)
        setattr(cls, attr_added, dct.get(attr, None))
        setattr(cls, attr_removed, dct.get(attr_removed, None))
        
        #get the sequence of classes from the base class to the class
        class_sequence = meta._get_class_sequence(cls, class_family=False)
        
        #compile the attribute over the class sequence
        value = None
        for cls in class_sequence:
            #add items specified by the class
            items_added = getattr(cls, '%s_added' % (attr), None)
            if items_added is not None:
                items_added = to_iterable(items_added)
                
                if value is None:
                    value = copy(items_added)
                else:
                    append_items(value, items_added, reorder=False)
            
            #remove items specified by the class
            items_removed = getattr(cls, '%s_removed' % (attr), None)
            if items_removed is not None:
                items_removed = to_iterable(items_removed)
                
                if value is not None:
                    remove_items(value, items_removed)
        
        #set the compiled value
        setattr(cls, attr, value)
    
    @classmethod
    def _compile_shared_attribute(meta, cls, attr):
        """compile a shared attribute from the values defined collectively by
        a class family (see description above), and assign the shared attribute
        to the class
        
        Parameters
        ----------
        cls : TypeFamily
            the class family member
        attr : string
            the shared attribute to compile. each family member should either
            not explicitly define the attribute, or define it as a list or a
            dict (consistent across all family members).
        """
        #make sure the attribute exists
        if not hasattr(cls, attr):
            return None
        
        #get the current value and delete the attribute
        old_value = to_iterable(getattr(cls, attr))
        try:
            delattr(cls, attr)
        except AttributeError:
            pass
        
        #if we still have the attribute, we inherited from an ancestor
        value = getattr(cls, attr, None)
        if value is not None:  # add any new items
            append_items(value, old_value)
        else:  # this is the first class to define the attribute
            value = old_value
            
            #go back and define the attribute for all ancestors
            for base_cls in cls._class_family.values():
                if base_cls is not cls:
                    setattr(base_cls, attr, value)
        
        #set the compiled value
        setattr(cls, attr, value)
    
    @classmethod
    def _save_class_hierarchy_state(meta, name='default'):
        """save a snapshot of the current state of the class hierarchy. this
        snapshot can later be restored with
        TypeFamily._restore_type_family_state(). use this if you want to be
        able to undo some process that pollutes the TypeFamily namespace.
        
        Parameters
        ----------
        name : string
            the name of the snapshot to save
        """
        #initialize the snapshot dict
        if meta._class_hierarchy_snapshots is None:
            meta._class_hierarchy_snapshots = {}
        
        #construct the snapshot
        state = {}
        for metaclass,attr_name in meta._class_hierarchy_snapshot_attributes:
            attr_value = getattr(metaclass, attr_name)
            
            if isinstance(attr_value, dict):
                state[attr_name] = {name:copy(value) for name,value in attr_value.iteritems()}
            elif isinstance(attr_value, list):
                state[attr_name] = [copy(value) for value in attr_value]
            else:
                state[attr_name] = copy(attr_value)
        
        #save it
        meta._class_hierarchy_snapshots[name] = state
    
    @classmethod
    def _restore_class_hierarchy_state(meta, name='default'):
        """restore a class hierarchy state previously saved with
        _save_class_hierarchy_state().
        
        NOTE: because python's import statements will not reload previously
        imported modules, this method could lead to problems if TypeFamily
        classes are imported after a call to _save_class_hierarchy_state(), then
        imported again after a call to _restore_class_hierarchy_state(). this is
        because, since said TypeFamily classes won't be reloaded, they won't
        re-establish themselves as part of the class family, and so e.g. won't
        be findable by name later. an example to illustrate, assuming there is
        a module named my_module.py including an Object class definition called
        MyAmazingObject:
            TypeFamily._save_class_hierarchy_state()
            import my_module
            world = PixelWorld(objects=['my_amazing_object'])
            TypeFamily._restore_class_hierarchy_state()
            import my_module
            world = PixelWorld(objects=['my_amazing_object'])  # ERROR!
        
        Parameters
        ----------
        name : string
            the name of the snapshot to restore
        """
        #get the snapshot
        snapshots = getattr(meta, '_class_hierarchy_snapshots', {})
        state = snapshots.get(name, None)
        if state is None:
            raise ValueError('no class hierarchy snapshot with name "%s" exists' % (name,))
        
        #restore the attributes
        for metaclass,attr_name in meta._class_hierarchy_snapshot_attributes:
            #get the current and former states of the attribute
            attr_value = state[attr_name]
            current_value = getattr(metaclass, attr_name)
            current_value_copy = copy(current_value)
            
            #for outer dict and lists, we want to repopulate the same variable
            #with the old values
            repopulate_outer = True
            if isinstance(attr_value, dict):
                items = attr_value.keys()
                values = attr_value.values()
                current_value.clear()
            elif isinstance(attr_value, list):
                items = range(len(attr_value))
                values = attr_value
                del current_value[:]
            else:
                repopulate_outer = False
                current_value = attr_value
            
            if repopulate_outer:  # repopulate the old values into the current outer variable
                for item,value in zip(items, values):
                    #if the old item currently exists in the variable, then we
                    #also want to repopulate it if it is a dict or list
                    if item in current_value_copy:
                        to_value = current_value_copy[item]
                        
                        #same as for the outer items, repopulate inner dicts and
                        #lists with the old inner items
                        if isinstance(to_value, dict):
                            to_value.clear()
                            for inner_item,inner_value in value.iteritems():
                                to_value[inner_item] = inner_value
                        elif isinstance(to_value, list):
                            del to_value[:]
                            to_value.extend(value)
                        else:
                            to_value = value
                        
                        current_value[item] = to_value
                    else:
                        current_value[item] = value
            
            setattr(metaclass, attr_name, current_value)
    
    def __new__(meta, name, bases, dct):
        """the TypeFamily metaclass manipulates the class definition as follows:
            - incorporates the class into a new or existing class family
            - assigns the class a unique name via its _name attribute
            - processes the _class_type attribute
            - compiles _class_tree_attributes
            - compiles _class_shared_attributes
        """
        cls = super(TypeFamily, meta).__new__(meta, name, bases, dct)
        
        #make sure the class has a family
        if not hasattr(cls, '_class_family'):
            dct['_class_family'] = meta._generate_class_family(cls)
        
        #make sure the class has a type
        if not hasattr(cls, '_class_type'):
            dct['_class_type'] = 'DEFAULT_TYPE'
        
        #potentially switch families or create a new family
        if '_class_family' in dct:
            family = dct['_class_family']
            if isinstance(family, basestring):
                class_family = meta._all_class_families.get(family, None)
                
                #make sure we register existing class types in adopted families
                if '_class_type' not in dct:
                    dct['_class_type'] = cls._class_types[cls._class_type]
                
                #get the family information
                if class_family is None:  # a new family is being created
                    class_family = OrderedDict()
                    class_types = copy(getattr(cls, '_class_types', []))
                    
                    meta._all_class_families[family] = class_family
                    meta._all_class_types[family] = class_types
                else:  # class is assigning itself to an existing family
                    #remove existing class types
                    if hasattr(cls, '_class_types'):
                        for attr in cls._class_types:
                            if hasattr(cls, attr):
                                delattr(cls, attr)
                    
                    class_types = meta._all_class_types[family]
                
                cls._class_family_name = family
                cls._class_family = class_family
                cls._class_types = class_types
            else:
                raise TypeError('_class_family must be a family name')
        
        #make sure the class has a unique name
        if '_name' not in dct:
            cls._name = meta._generate_class_name(cls)
        
        #add the class to the family (or overwrite the old member)
        cls._class_family[cls._name] = cls
        
        #process class types
        if '_class_type' in dct:  # class is changing its class type
            class_type = dct['_class_type']
            if isinstance(class_type, basestring):
                try:  #existing class type
                    class_type_idx = cls._class_types.index(class_type)
                except ValueError:  # new class type
                    #make sure existing classes don't already define the class type attribute
                    for c in cls._class_family.values():
                        if hasattr(c, class_type):
                            raise AttributeError('class type "%s" is already assigned as an attribute to %s' % (class_type, c.__name__))
                    
                    #register the new class type
                    cls._class_types.append(class_type)
                    class_type_idx = len(cls._class_types) - 1
                    
                    #add an attribute constant to all family members
                    for c in cls._class_family.values():
                        setattr(c, class_type, class_type_idx)
                
                #convert the class type name to the class type code
                cls._class_type = class_type_idx
            else:
                raise TypeError('_class_type must be a string (uppercase by convention)')
        
        #process tree attributes, starting with _class_tree_attributes itself
        meta._compile_tree_attribute(cls, '_class_tree_attributes', dct)
        
        if cls._class_tree_attributes is not None:
            for attr in cls._class_tree_attributes:
                meta._compile_tree_attribute(cls, attr, dct)
        
        #process shared attributes
        if getattr(cls, '_class_shared_attributes', None) is not None:
            for attr in cls._class_shared_attributes:
                meta._compile_shared_attribute(cls, attr)
        
        return cls


#see note above at definition of _class_snapshot_attributes
TypeFamily._class_hierarchy_snapshot_attributes.append([TypeFamily, '_all_class_families'])
TypeFamily._class_hierarchy_snapshot_attributes.append([TypeFamily, '_all_class_types'])


class AutoState(type):
    """for classes that do not explicitly assign their _state_attributes list
    (see StateObject below), this metaclass attempts to automatically determine
    which attributes should be included in it.
    
    unless the class explicitly assigns _auto_state = False, this metaclass
    assigns _state_attributes as a list of any non-method attributes that the
    class defines explicitly and which is not defined by one of its
    superclasses. note, however, that because _state_attributes is defined as a
    _class_tree_attribute by StateObject, all _auto_state = False does in
    practice is prevent any -new- attributes from being added to the
    _state_attributes list. the class will still inherit the _state_attributes
    defined by its superclasses.
    
    if a class explicitly defines a list attribute named _auto_state_exclude,
    then any attribute in that list is excluded from _state_attributes (again,
    unless that attribute has already been included as a state attribute by one
    of the class' superclasses).
    """
    @classmethod
    def _find_implicit_state_attributes(meta, bases, dct):
        """construct a list of candidate attributes for a class'
        _state_attributes list
        
        Parameters
        ----------
        bases : list
            the list of base classes
        dct : dict
            the class' attribute dict
        
        Returns
        -------
        candidates : list
            a list of non-method attributes that are defined by the class but
            not by any of its parents
        """
        #compile a list of inherited attributes
        attr_inherited = set()
        for base_cls in bases:
            attr_inherited = attr_inherited.union(dir(base_cls))
        
        #find the newly defined attributes
        candidates = set(dct.keys()).difference(attr_inherited)
        
        #remove methods
        for attr in copy(candidates):
            method_types = (types.FunctionType, classmethod, property)
            if isinstance(dct[attr], method_types):
                candidates.remove(attr)
        
        return list(candidates)
    
    def __new__(meta, name, bases, dct):
        if '_auto_state' in dct:  # user-specified auto_state
            auto_state = dct['_auto_state']
        else:  # auto state if _state_attributes is not defined
            auto_state = not '_state_attributes' in dct
        
        if auto_state:  # find state attributes
            dct['_state_attributes'] = meta._find_implicit_state_attributes(bases, dct)
        
        #TypeFamily should happen next, so we get compiled tree attributes
        cls = super(AutoState, meta).__new__(meta, name, bases, dct)
        
        if auto_state:  # remove explicitly excluded state attributes
            sa_current = getattr(cls, '_state_attributes')
            sa_exclude = getattr(cls, '_auto_state_exclude', [])
            sa = set(sa_current).difference(sa_exclude)
            
            setattr(cls, '_state_attributes', list(sa))
        
        return cls


class StateObject(object):
    """an object that supports being stored as a State.
    
    the attributes defined by the class' _state_attributes list are saved in the
    State. the class may additionally implement the _encode_state and
    _decode_state methods to define custom State storing/restoring behaviors.
    """
    #combine AutoState and TypeFamily metaclasses
    class __metaclass__(AutoState, TypeFamily):
        pass
    
    #this causes each StateObject to have _state_attributes and
    #_auto_state_exclude attributes, which are lists that are assembled based on
    #the StateObject's superclasses (see TypeFamily and _state_attributes and
    #_auto_state_exclude below for details)
    _class_tree_attributes = ['_state_attributes', '_auto_state_exclude']
    
    #this specifies a list of attributes that should be included when saving the
    #StateObject's state (see State in datatypes.py)
    _state_attributes = []
    
    #this specifies a list of attributes that should be excluded from the
    #automatically generated list of state attributes. there is no need to
    #define this if _state_attributes is defined.
    _auto_state_exclude = []
    
    def _encode_state(self, state):
        """define any custom state-storing code here
        
        Parameters
        ----------
        state : State
            the State, after state attributes have been encoded
        """
        pass
    
    def _decode_state(self, state):
        """define any custom state-restoring code here
        
        Parameters
        ----------
        state : State
            the State, after state attributes have been decoded
        """
        pass

    def _post_restore(self):
        """Perform any actions that need to take place after restoring from a State.
        """
        pass


class StateElement(object):
    """base class for classes that provide support for values that need special
    treatment when storing as States. StateElements should encode their values
    in __init__(), and should restore those encoded values in restore().
    """
    def restore(self, state):
        """restore an encoded value
        
        Parameters
        ----------
        state : State
            the State object of which this element is a part
        
        Returns
        -------
        value : any
            the decoded value
        """
        raise NotImplementedError


class StateKey(StateElement):
    """this acts as a placeholder for encoded values, primarily just pointing to
    the encoded value in the State._encoded_values dict
    """
    #the key in the _encoded_values dict
    key = None
    
    def __init__(self, key):
        self.key = key
    
    def restore(self, state):
        return state.restore_value(self)
    
    def __repr__(self):
        return '->%d' % (self.key)


class StateMethod(StateElement):
    """this encodes methods, which aren't picklable via cPickle"""
    
    #the key of the object whose method this is
    _object_key = None
    
    #the method's attribute name
    _attribute = None
    
    def __init__(self, state, method):
        self._object_key = state.store_value(method.im_self).key
        self._attribute = method.im_func.func_name
    
    def restore(self, state):
        object_key = self._object_key
        attr = self._attribute
        return getattr(state._reference_values[object_key], attr)
    
    def __repr__(self):
        return '(->%d).%s' % (self._object_key, self._attribute)


class StateFunction(StateElement):
    """this uses marshal to encode functions, as long as they don't have
    closures"""
    
    #the serialized encoding of the function
    _serialized = None
    
    #the function's name
    _name = None
    
    def __init__(self, state, func):
        assert func.func_closure is None, 'cannot save functions with closures'
        
        self._name = func.func_name
        self._serialized = marshal.dumps(func.func_code)
    
    def restore(self, state):
        code = marshal.loads(self._serialized)
        return types.FunctionType(code, globals())
    
    def __repr__(self):
        return 'StateFunction(%s)' % (self._name)


class State(object):
    """records the state of a system of StateObjects.
    
    a StateObject is encoded in a State by passing it to the State class
    constructor (i.e. state = State(state_object)). each of the StateObject's
    _state_attributes are then encoded, possibly leading to the creation of
    additional sub-States.
    
    a StateObject is restored from a State via the State.restore() method.
    """
    #the key of the StateObject for which this state was constructed
    _object_key = None
    
    #keep a list of state attributes that should be restored
    _state_attributes = None
    
    #a dict of encoded state attribute values (and possibly other custom data
    #handled by the StateObject's custom _encode_state() method)
    _state_data = None
    
    #the following dicts are shared in common by all States in the State system
    _root_state = None  # the root State
    _states = None  # a dict of States resulting from the root State
    _reference_values = None  # a dict of values needed for State restoration
    _encoded_values = None  # a dict of encoded values
    _decoded_values = None  # used during restore() to store decoded values
    
    def __init__(self, obj, root_state=None):
        self._state_attributes = copy(obj._state_attributes)
        self._state_data = {}
        
        if root_state is None:  # initialize some attributes
            self._root_state = self
            self._states = {}
            self._reference_values = {}
            self._encoded_values = {}
            self._decoded_values = {}
        else:  # share attributes with the root State
            self._root_state = root_state
            self._states = root_state._states
            self._reference_values = root_state._reference_values
            self._encoded_values = root_state._encoded_values
            self._decoded_values = root_state._decoded_values
        
        #keep a record of the StateObject and State
        self._object_key = self._get_key(obj)
        self._reference_values[self._object_key] = obj
        self._encoded_values[self._object_key] = self
        self._states[self._object_key] = self
        
        #store the object's state data
        for attr in self._state_attributes:
            value = getattr(obj, attr)
            self._state_data[attr] = self.store_value(value)
        
        #perform any custom encoding defined by the object
        obj._encode_state(self)

    def restore(self, obj=None, root=None):
        """restore the state of a StateObject encoded by this State
        
        Parameters
        ----------
        obj : StateObject, optional
            the StateObject to restore. if unspecified, uses the StateObject
            from which the State was constructed.
        root : bool, optional
            True if this is the first call to restore() (rather than a
            sub-State being restored during restoration of a super-State). this
            is used internally during the restore() process.
        
        Returns
        -------
        obj : StateObject
            the restored StateObject
        """
        if root is None:
            root = self._root_state is self
        
        if obj is None:
            obj = self._reference_values[self._object_key]
            if obj is None:
                return obj
        
        #restore state attributes
        for attr in self._state_attributes:
            state_key = self._state_data[attr]
            value = self.restore_value(state_key)
            setattr(obj, attr, value)
        
        #do things that only the root State.restore() should do
        if root:
            #perform custom decoding defined by encoded objects
            for key in self._states:
                state_obj = self._reference_values[key]
                state = self._states[key]
                if state_obj is not None:
                    state_obj._decode_state(state)
            
            #clear the decoded values
            self._decoded_values.clear()

        # do any necessary post-restore stuff
        obj._post_restore()
        
        return obj
    
    def store_value(self, value):
        """encode a value as part of the State
        
        Parameters
        ----------
        value : StateObject | method | function | dict | list | tuple | set |
                picklable
            the value to encode
        
        Returns
        -------
        key : StateKey
            a StateKey that can act as a placeholder for the value
        """
        key = self._get_key(value)
        state_key = StateKey(key)
        #only encode if we haven't already encoded the value
        if not key in self._reference_values:
            #remember the value so we don't have to encoded it again
            self._reference_values[key] = value
            
            if isinstance(value, StateObject):  # encode a StateObject in a State
                encoded_value = State(value, root_state=self._root_state)
            elif inspect.ismethod(value) and \
            isinstance(value.im_self, StateObject):  # encode a method in StateMethod
                encoded_value = StateMethod(self, value)
            elif isinstance(value, types.FunctionType):  # encode a function in a StateFunction
                encoded_value = StateFunction(self, value)
            elif isinstance(value, dict):  # encode the values of a dict
                encoded_value = copy(value)
                
                for sub_key,sub_value in value.iteritems():
                    encoded_value[sub_key] = self.store_value(sub_value)
            elif isinstance(value, (list, tuple, set)):  # encode the values of a list-like
                encoded_value = list(value)
                
                for idx,sub_value in enumerate(encoded_value):
                    encoded_value[idx] = self.store_value(sub_value)
                
                encoded_value = type(value)(encoded_value)
            else:  # make sure the object is pickleable and unpickleable
                try:
                    pickled = pickle.loads(pickle.dumps(value))
                    
                    #copy fails for classes that use custom metaclasses, so for
                    #those we will use deepcopy instead
                    try:
                        encoded_value = copy(value)
                    except TypeError:
                        encoded_value = deepcopy(value)
                except:
                    raise TypeError('cannot encode %s' % (self._get_object_name(value)))
            
            #save the encoded value
            self._encoded_values[key] = encoded_value
            
        return state_key
    
    def restore_value(self, state_key):
        """restore a previously-stored value
        
        Parameters
        ----------
        state_key : StateKey
            the StateKey that points to the encoded value
        
        Returns
        -------
        value : any
            the restored value
        """
        key = state_key.key
        
        #only decode if we haven't already decoded the value
        if not key in self._decoded_values:
            #the encoded value
            encoded_value = self._encoded_values[key]
            
            if isinstance(encoded_value, StateElement):  # restore a value encoded in a StateElement
                value = encoded_value.restore(self)
            elif isinstance(encoded_value, State):  # restore a StateObject
                #get the StateObject
                value = self._reference_values[key]
                self._decoded_values[key] = value

                #restore it
                encoded_value.restore(obj=value, root=False)
            elif isinstance(encoded_value, dict):  # restore a dict of values
                #get the original dict
                value = self._reference_values[key]
                self._decoded_values[key] = value
                
                #remove all of its values
                value.clear()
                
                #restore the old values
                for sub_key,sub_state_key in encoded_value.iteritems():
                    value[sub_key] = self.restore_value(sub_state_key)
            elif isinstance(encoded_value, list):  # restore a list of values
                #get the original list
                value = self._reference_values[key]
                self._decoded_values[key] = value
                
                #clear it
                del value[:]
                
                #add the encoded values (to initialize the list size)
                value.extend(encoded_value)
                
                #restore the old values
                for idx,sub_state_key in enumerate(encoded_value):
                    value[idx] = self.restore_value(sub_state_key)
            elif isinstance(encoded_value, (tuple, set)):  # restore a list-like of values
                #restore into a list
                value = [self.restore_value(sub_state_key) \
                            for sub_state_key in encoded_value]
                
                #convert to the output type
                value = type(encoded_value)(value)
            else:  # must have just been a pickleable object
                #copy fails for classes that use custom metaclasses, so for
                #those we will use deepcopy instead
                try:
                    value = copy(encoded_value)
                except TypeError:
                    value = deepcopy(encoded_value)
            
            #save the decoded value
            self._decoded_values[key] = value
        else:  # already decoded
            value = self._decoded_values[key]
        
        return value
    
    def set(self, name, value):
        """store extra information in the state. this can be used along with 
        get() by custom StateObject._encode_state() methods.
        
        Parameters
        ----------
        name : string
            an identifier
        value : any
            the information. must be pickleable.
        """
        self._state_data[name] = value
    
    def get(self, name):
        """retrieve information stored previously with set(). this can be used
        along with set() by custom StateObject._decode_state() methods.
        
        Parameters
        ----------
        name : string
            an identifier
        
        Returns
        -------
        value : any
            the previously stored information
        """
        return self._state_data[name]
    
    def __repr__(self):
        obj = self._reference_values[self._object_key]
        name = self._get_object_name(obj)
        
        return 'State(%s)' % (name)

    def __deepcopy__(self, memo):
        """Deep-copy the state. Unit tests fail without this, not entirely sure why.

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

        return rv
    
    def _get_object_name(self, obj):
        """get a friendly name for an object for representation in a string"""
        if hasattr(obj, '__name__'):
            return obj.__name__
        elif hasattr(obj, '__class__'):
            return '%s object' % (obj.__class__.__name__)
        else:
            return '%s' % (obj)
    
    def _get_key(self, obj):
        """get a key to identify an object
        
        Parameters
        ----------
        obj : object
            an object
        
        Returns
        -------
        key : int
            a hashable key to uniquely identify the object
        """
        return id(obj)


class Data(StateObject):
    """thin wrapper around a masked structured numpy array that supports
    dynamically adding and removing rows and fields
    """
    _class_family = 'data'
    
    #true if the data array has already been initialized with a set of fields
    _initialized = False
    
    #as masked structured numpy array that holds the complete data
    _data = None
    
    #view of the subset of rows of _data that are currently "defined". rows of
    #data are created in blocks in order to minimize copying of the data array.
    _data_view = None
    
    #a list of names of the data fields
    _fields = None
    
    #a list of each field's null value
    _null_values = None
    
    #stores a callback that is called whenever the data array changes
    _data_change_callback = None
    
    #create blocks of this many rows at a time
    _row_block_size = 100
    
    #number of "defined" rows
    _num_rows = 0
    
    #attributes to include in States
    _state_attributes = ['_initialized', '_fields', '_null_values',
        '_data_change_callback', '_num_rows']
    
    def __init__(self, data_change_callback=None):
        """
        Parameters
        ----------
        data_change_callback : callable
            a callback that gets called whenever the whenever the data array
            changes. e.g. can be used to update views into the data array. NOTE:
            if this Data object will be stored in a State, then this function
            must be an instance method of a StateObject.
        """
        self._fields = []
        self._null_values = {}
        
        self._data_change_callback = data_change_callback
        
        self.null()
    
    @property
    def initialized(self):
        """True if the data array has been initialized with a set of fields"""
        return self._initialized
    
    @property
    def shape(self):
        """shape of the data array"""
        return (self._num_rows, len(self._fields))
    
    @property
    def fields(self):
        """a list of names of the data fields"""
        return self._fields
    
    @property
    def dtype(self):
        """the data array dtype"""
        return self._data.dtype
    
    @property
    def data(self):
        """a view of the defined data part of the data array"""
        return self._data_view
    
    def null(self):
        """null the data array"""
        self._initialized = False
        
        del self._fields[:]
        self._null_values.clear()
        
        self._num_rows = 0
        
        self._set_data(ma.array([]))
    
    def view_field(self, name, type=None):
        """construct a view of one data field
        
        Parameters
        ----------
        name : string
            the name of the field
        type : type, optional
            the type of the returned array
        
        Returns
        -------
        view : MaskedArray
            a view of the specified field
        """
        view = self.data[name]
        
        if type is not None:
            return view.view(type=type)
        else:
            return view
    
    def view_row(self, idx):
        """construct a view of one data row
        
        Parameters
        ----------
        idx : int
            the row to view
        
        Returns
        -------
        view : mvoid
            a view of the specified row
        """
        return self.data[idx]
    
    def add_field(self, name, dtype, null_value, ndim=1):
        """add a field to the data array
        
        Parameters
        ----------
        name : string
            the name of the field
        dtype : type
            the dtype of the field
        null_value : any
            a value to use for unspecified (i.e. masked) field values
        ndim : int
            the dimensionality of a single field value
        """
        self._fields.append(name)
        self._null_values[name] = null_value
        
        #define the datatype of the new field
        dt = [(name, dtype, ndim)]
        
        #we need to add the field differently depending on whether it is the
        #first or a subsequent field
        if self.initialized:
            #initialize as an array of null values
            num_rows = len(self._data)
            field = ma.array([(null_value,)], dtype=dt)
            field = ma.array(np.tile(field, (num_rows,)), mask=[True])
            
            #merge the new field into the array
            self._set_data(merge_structured_arrays(self._data, field))
        else:
            #create an empty field
            self._set_data(ma.array([], dtype=dt))
            
            self._initialized = True
    
    def remove_field(self, name):
        """remove a field from the array
        
        Parameters
        ----------
        name : string
            the name of the field
        """
        self._fields.remove(name)
        del self._null_values[name]
        
        if len(self._fields)==0:
            self.null()
        else:
            with warnings.catch_warnings():
                #i don't think the "selecting multiple fields in a structured
                #array warning applies here
                warnings.simplefilter(action='ignore', category=FutureWarning)
                
                new_data = self._data[self._fields]
                
                self._set_data(new_data)
    
    def add_row(self, data=None):
        """add a row to the data. this only actually adds a row if we have
        reached the end of the previously added row block. otherwise, just
        increment the row counter by one.
        
        Parameters
        ----------
        data : dict, optional
            a dict of initial values for the row's fields
        
        Returns
        -------
        idx : int
            the row's index in the data array
        """
        idx = self._num_rows
        self.add_rows(1)

        #set any specified row data
        if data is not None:
            for key,value in data.iteritems():
                self._data[key][idx] = value
        
        return idx

    def add_rows(self, num):
        """add many rows to the data. this only actually adds a row if we have reached
        the end of the previously added row block. otherwise, just increment
        the row counter by num.
        
        Parameters
        ----------
        num : int
            number of rows to add
        
        Returns
        -------
        idx : int
            the row's index in the data array
        """
        if not self.initialized:
            raise RuntimeError('fields must be added before rows')

        while len(self._data) <= self._num_rows + num: # add a row block
            self._add_row_block()
        
        #increment the row counter
        self._num_rows += num
        self._process_data_change()
            
    def remove_row(self, idx):
        """remove a row from the data array
        
        Parameters
        ----------
        idx : int
            the index of the row to remove
        """
        #keep everything except the specified row
        keep = np.ones(len(self._data), dtype=bool)
        keep[idx] = False
        new_data = self._data[keep]
        
        #decrement the row counter
        self._num_rows -= 1
        
        self._set_data(new_data)
    
    def __repr__(self):
        return self.data.__repr__()
    
    def __len__(self):
        return self._num_rows
    
    def __getitem__(self, key):
        return self.data.__getitem__(key)
    
    def __setitem__(self, key, value):
        self.data.__setitem__(key, value)
    
    def _add_row_block(self):
        """add a block of rows to the data array
        """
        block = ma.masked_all((self._row_block_size,), dtype=self.dtype)
        
        self._set_data(ma.append(self._data, block))
    
    def _set_data(self, data):
        """set the data array
        
        Parameters
        ----------
        data : MaskedArray
            the new data array
        """
        if isinstance(data, ma.MaskedArray) and not isinstance(data, SerializableMaskedArray):
            self._data = SerializableMaskedArray(data)            
        elif isinstance(data, SerializableMaskedArray):
            self._data = data
        else:
            assert False, 'unknown data type'
        
        #set the fill value of each field
        for field in self.fields:
            #this fails with multi-dimensional fields, in which case we do it
            #the wonky way below
            try:
                self._data[field].set_fill_value(self._null_values[field])
            except TypeError:
                self._data._fill_value[field] = self._null_values[field]
        
        #process the change to the data array
        self._process_data_change()
    
    def _process_data_change(self):
        """process a change to the data array"""
        #update the data view
        self._data_view = self._data[:self._num_rows]
        
        #call the data change callback
        if callable(self._data_change_callback):
            self._data_change_callback()
    
    def _encode_state(self, state):
        """custom encoding of the Data into a State. we copy the data array and
        encode any StateObject element as a State.
        
        NOTE: we are assuming at this point that any array element is either a
        StateObject or is pickleable
        
        Parameters
        ----------
        state : State
            see StateObject
        """
        #copy the data array
        data = copy(self._data)
        
        #search for and encode StateObjects
        for field in self.fields:
            d = data[field]
            
            if d.dtype == np.object:
                for idx,x in enumerate(d):
                    if isinstance(x, StateObject):
                        d[idx] = state.store_value(x)

        #save the encoded data array
        state.set('data', data)
    
    def _decode_state(self, state):
        """custom decoding of a State. decode and set the previously encoded
        data array.
        
        Parameters
        ----------
        state : State
            see StateObject
        """
        #get the encoded data array
        data = copy(state.get('data'))

        #search for and decode States
        for field in self.fields:
            d = data[field]
            
            if d.dtype == np.object:
                for idx,x in enumerate(d):
                    if isinstance(x, StateKey):
                        d[idx] = state.restore_value(x)
        
        #set the decoded data array
        self._set_data(data)
