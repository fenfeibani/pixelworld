import unittest
from collections import OrderedDict

import numpy as np

from pixelworld.envs.pixelworld.datatypes import TypeFamily, AutoState, StateObject, StateFunction, StateMethod, State, \
    Data, StateKey
from pixelworld.envs.pixelworld.tests import test_core


#put this here because pickle doesn't like locally defined classes
class AForStateMethodClassMethod(StateObject):
    @classmethod
    def g(cls, x):
        return x + 4


class TestTypeFamily(test_core.TestCase):
    def test_type_family(self):
        """Check that type families behave as expected"""
        class A:
            __metaclass__ = TypeFamily
            _class_family = 'Fam2'
            _class_tree_attributes = []
            _name = 'Aname'

        class B(A):
            _name = 'Bname'

        class C(B):
            pass

        # check that _class_family is an OrderedDict
        self.assertEqual(type(A._class_family), OrderedDict)

        # check that everybody shares the same _class_family
        self.assertTrue(A._class_family is B._class_family is C._class_family)
        self.assertTrue(A._class_family_name is B._class_family_name is C._class_family_name is 'Fam2')

        # check that everybody's name in the class family is what we would
        # expect, and that there are no extra names
        cf = A._class_family 
        self.assertTrue(cf['Aname'] is A)
        self.assertTrue(cf['Bname'] is B)
        self.assertTrue(cf['c'] is C)
        self.assertTrue(len(cf) == 3, 'class family is: %s' % cf.keys())

        # check that everyone's name is what we expect
        self.assertEqual(A._name, 'Aname')
        self.assertEqual(B._name, 'Bname')
        self.assertEqual(C._name, 'c')

        # check that you can't have a _class_family that isn't a string
        with self.assertRaises(TypeError):
            class D(object):
                __metaclass__ = TypeFamily
                _class_family = 3

    def test_type_family_class_type(self):
        """Check that type family class types behave as expected"""
        class A:
            __metaclass__ = TypeFamily
            _class_family = 'Fam3'
            _class_type = 'CHIMNEY'
            _class_tree_attributes = []

            CHOWDER = 23

        class B(A):
            _class_type = 'FLAPJACK'

        class C(A):
            pass

        class D(object):
            __metaclass__ = TypeFamily
            _class_family = 'Fam3'        
            _class_type = 'CIRCUMSTANCES'
            _class_types = ['IGNORE', 'THIS']

            IGNORE = 17

        # check that everyone's _class_type is what we expect
        self.assertIs(A._class_type, A.CHIMNEY)
        self.assertIs(B._class_type, A.FLAPJACK)
        self.assertIs(C._class_type, A.CHIMNEY)
        self.assertIs(D._class_type, A.CIRCUMSTANCES)

        # check that everyone has all the classtypes defined as attributes and
        # that they are all the same
        self.assertTrue(A.CHIMNEY is B.CHIMNEY is C.CHIMNEY)
        self.assertTrue(A.FLAPJACK is B.FLAPJACK is C.FLAPJACK)
        self.assertTrue(A.CIRCUMSTANCES is B.CIRCUMSTANCES is C.CIRCUMSTANCES)

        # check that A has the correct list in _class_types
        self.assertItemsEqual(A._class_types, ['CHIMNEY', 'FLAPJACK', 'CIRCUMSTANCES'])

        # check that D's _class_types gets overwritten
        self.assertTrue(not hasattr(D, 'IGNORE'))

        # check that we can't define a classtype that would overwrite an
        # attribute of A
        with self.assertRaises(AttributeError):
            class E(A):
                __metaclass__ = TypeFamily
                _class_type = 'CHOWDER'

        # check that we can't define a classtype that isn't a string
        with self.assertRaises(TypeError):
            class F(A):
                __metaclass__ = TypeFamily
                _class_type = 49

    def test_type_family_shared_attributes(self):
        """Check that type family shared attributes behave as expected"""
        class A:
            __metaclass__ = TypeFamily
            _class_family = 'Fam4'
            _class_tree_attributes = []
            _class_shared_attributes = ['_wax', '_fungus']
            _wax = ['turtle']

        class B(A):
            _wax = ['ear']
            _fungus = {'mushroom': 1}

        class C(A):
            _wax = ['ball of']
            _fungus = {'lichen': 0.5}

        # check that the shared attribute is shared among all classes
        self.assertEqual(sorted(A._wax), ['ball of', 'ear', 'turtle'])
        self.assertEqual(sorted(B._wax), ['ball of', 'ear', 'turtle'])
        self.assertEqual(sorted(C._wax), ['ball of', 'ear', 'turtle'])

        # check that the dicts have the right values and are of the right
        # length
        self.assertEqual(A._fungus['mushroom'], 1)
        self.assertEqual(A._fungus['lichen'], 0.5)
        self.assertEqual(len(A._fungus), 2)
        self.assertEqual(B._fungus['mushroom'], 1)
        self.assertEqual(B._fungus['lichen'], 0.5)
        self.assertEqual(len(B._fungus), 2)
        self.assertEqual(C._fungus['mushroom'], 1)
        self.assertEqual(C._fungus['lichen'], 0.5)
        self.assertEqual(len(C._fungus), 2)


    def test_type_family_tree_attributes(self):
        """Check that type family tree attributes behave as expected"""
        class A:
            __metaclass__ = TypeFamily
            _class_family = 'Fam5'
            _class_tree_attributes = ['_wax', '_fungus']
            _class_shared_attributes = []
            _wax = ['turtle']
            _fungus = {'mushroom': 1}

        class B(A):
            _wax = ['ear']
            _fungus_removed = ['mushroom']

        class C(B):
            _wax_removed = ['ear']
            _fungus = {'lichen': 0.5}

        class D(C):
            _wax = ['ball of']
            _fungus_removed = ['lichen']

        # check that everyone has the right value for the list class tree
        # attribute
        self.assertEqual(A._wax, ['turtle'])
        self.assertEqual(sorted(B._wax), ['ear', 'turtle'])
        self.assertEqual(C._wax, ['turtle'])
        self.assertEqual(sorted(D._wax), ['ball of', 'turtle'])

        # check that everyone has the right value for the dict class tree
        # attribute
        self.assertEqual(A._fungus, {'mushroom': 1})
        self.assertEqual(B._fungus, {})
        self.assertEqual(C._fungus, {'lichen': 0.5})
        self.assertEqual(D._fungus, {})


class TestAutoState(test_core.TestCase):
    def test_auto_state(self):
        """Check that AutoState picks the right state attributes for classes"""
        class A:
            __metaclass__ = AutoState
            _class_family = 'Fam6'
            a = 3
            b = 4
            c = 'help i am trapped in a string'

        class B(A):
            a = 6
            d = 'fleek'

        class C(B):
            e = 9
            _auto_state_exclude = ['e']

        class D(A):
            f = 12
            _auto_state = False

        # check that everyone has the right state attributes
        self.assertItemsEqual(A._state_attributes, ['__metaclass__', '__module__', '_class_family', 'a', 'b', 'c'])
        self.assertItemsEqual(B._state_attributes, ['d'])
        self.assertItemsEqual(C._state_attributes, ['_auto_state_exclude'])
        self.assertItemsEqual(D._state_attributes, ['__metaclass__', '__module__', '_class_family', 'a', 'b', 'c'])

class TestState(test_core.TestCase):
    def test_state_object(self):
        """Check that StateObjects have attributes we expect"""
        class A(StateObject):
            pass

        # check that A has the attributes we expect
        self.assertTrue(hasattr(A, '_state_attributes'))
        self.assertTrue(hasattr(A, '_auto_state_exclude'))

    def test_state_method(self):
        """Check that StateMethods behave as expected"""
        class A(StateObject):
            def f(self, x):
                return x + 3

        # create a state object and a state of it
        a = A()
        s = State(a)
        
        # create a StateMethod based on state s
        f2 = StateMethod(s, a.f)
        
        # restore the StateMethod
        f3 = f2.restore(s)

        # check that the restored method does what we expect
        self.assertIs(f3(3), 6)

        # check that repr of the StateMethod is what we expect
        self.assertEqual(repr(f2), '(->%d).f' % id(a))

        # use store_value to store the method
        f4 = s.store_value(a.f)
        
        # restore the method
        f5 = f4.restore(s)

        # check that the restored method does what we expect
        self.assertIs(f5(4), 7)

    def test_state_key(self):
        """Check that state keys behave as expected"""
        class A(StateObject):
            foo = 3

        # create a state object and a state of it
        a = A()
        s = State(a)
        
        # look inside the state for the state key that represents foo
        key = s._state_data['foo']
        
        # check that the repr of the state key is what we expect
        self.assertEqual(repr(key), '->%d' % id(A.foo))
        
        # check that we can restore foo
        self.assertEqual(key.restore(s), 3)

    def test_auto_state_classmethod(self):
        """Check that class methods do not show up in state attributes"""
        class A(StateObject):
            @classmethod
            def blah(cls):
                return 'blah'

        # check that class method does not show up in A's state attributes
        self.assertTrue('blah' not in A._state_attributes)

    def test_auto_state_property(self):
        """Check that properties do not show up in state attributes"""
        class A(StateObject):
            @property
            def blah(self):
                return 'blah'

        # check that property does not show up in A's state attributes
        self.assertTrue('blah' not in A._state_attributes)

    def test_state_method_class_method(self):
        """Check that StateMethod can save class methods"""
        # create an object and save its state
        a = AForStateMethodClassMethod()
        s = State(a)

        # create a StateMethod from a's method
        g2 = StateMethod(s, a.g)
        
        # restore from state method
        g3 = g2.restore(s)

        # check that restored method does what we expect
        self.assertIs(g3(3), 7)

    def test_state_function(self):
        """Check that StateFunctions behave as expected"""
        def f(x):
            return x + 3

        # create a StateFunction from a function
        f2 = StateFunction(None, f)
        
        # check that structure of the StateFunction is what we expect
        self.assertIs(type(f2._serialized), str)

        # check that repr() of the StateFunction is what we expect
        self.assertEqual(repr(f2), 'StateFunction(f)')

        # restore the StateFunction
        f3 = f2.restore(None)
        
        # check that the restored function does what we expect
        self.assertIs(f3(3), 6)

        class Foo(StateObject):
            pass

        # create a state
        s = State(Foo())
        
        # create a StateFunction using store_value()
        f4 = s.store_value(f)
        
        # check that the return value of store_value() is a StateKey
        self.assertTrue(isinstance(f4, StateKey))
        
        # restore from the StateKey
        f5 = f4.restore(s)

        # check that the restored function does what we expect
        self.assertIs(f5(4), 7)

    def test_state(self):
        """Check that States behave as expected"""
        class A(StateObject):
            x = 3
            y = 5
            z = None

        # create an object
        a = A()
        
        # mutate the object
        a.x = 4

        # save the state of the object
        s = State(a)
        
        # restore from state
        a2 = s.restore()

        # check that attribute values are what we expect
        self.assertIs(a2.x, 4)
        self.assertIs(a2.y, 5)
        self.assertIs(a2.z, None)

        # store a value, check that we can restore it
        k = s.store_value('chimpanzee')
        self.assertIs(s.restore_value(k), 'chimpanzee')

        # store a None value, check that we can restore it
        k2 = s.store_value(None)
        self.assertIs(s.restore_value(k2), None)

        # set a value, check that we can get it back with get()
        s.set('rent', 'too high')
        self.assertIs(s.get('rent'), 'too high')

    def test_state2(self):
        """Check that States correctly handle the case where a state object has another
        state object as an attribute"""
        class A(StateObject):
            x = 3
            y = 4
            z = None

        class B(A):
            a = 2
            b = 8
            c = A()

        # create an object that has another state object as an attribute
        b = B()

        # save its state
        s = State(b)
        
        # restore from state
        b2 = s.restore()

        # check that all attribute values are what we would expect
        self.assertIs(b2.a, 2)
        self.assertIs(b2.b, 8)
        self.assertIs(b2.c.x, 3)
        self.assertIs(b2.c.y, 4)
        self.assertIs(b2.c.z, None)

    def test_unpicklable(self):
        """Check that unpicklable objects cannot be stored."""
        class Foo(StateObject):
            pass

        class Unpicklable(object):
            def __getstate__(self):
                assert False

        # create a state
        s = State(Foo())

        # check that we cannot store an unpicklable object
        with self.assertRaises(TypeError):
            s.store_value(Unpicklable())

    def test_store_list(self):
        """Check that we can save and restore a list"""
        class Foo(StateObject):
            pass

        # create a state
        s = State(Foo())

        # check that we can save and restore a list
        key = s.store_value([3, 4, 5])
        self.assertEqual(key.restore(s), [3, 4, 5])

    def test_store_tuple(self):
        """Check that we can save and restore a tuple"""
        class Foo(StateObject):
            pass

        # create a state
        s = State(Foo())

        # check that we can save and restore a tuple
        key = s.store_value((3, 4, 5))
        self.assertEqual(key.restore(s), (3, 4, 5))

    def test_state_repr(self):
        """Check that State repr is what we expect"""
        class Foo(StateObject):
            pass

        class Bar(StateObject):
            __name__ = 'Oz the great and powerful'

        # save state of object and check that repr of the state is what we
        # expect
        s = State(Foo())
        self.assertEqual(repr(s), 'State(Foo object)')

        # save state of object and check that repr of the state is what we
        # expect
        s = State(Bar())
        self.assertEqual(repr(s), 'State(Oz the great and powerful)')


class TestData(test_core.TestCase):
    def test_data(self):
        """Check that Data object acts as expected when we add and remove data"""
        # create a data object
        d = Data()

        # check that we can't add a row to a data array with no fields
        with self.assertRaises(RuntimeError):
            d.add_row({'fleebar': 'froofronk'})

        # add a float field
        d.add_field('flurbosity', float, -17.76)
        
        # add a row
        idx = d.add_row()
        
        # check that view_row has length one, since there is one field
        self.assertEqual(len(d.view_row(idx)), 1)
        # check that flurbosity is masked in the row that we have
        self.assertIs(d.view_row(idx)['flurbosity'], np.ma.masked)
        # check that d has the right shape and len
        self.assertEqual(d.shape, (1, 1))
        self.assertEqual(len(d), 1)

        # add a string field
        d.add_field('smilchritude', np.dtype('S100'), 'too smilchy for you')

        # add a row
        idx2 = d.add_row()

        # check that smilchritude is masked in the new row
        self.assertIs(d.view_row(idx)['smilchritude'], np.ma.masked)
        # check that view_row has length two, since there are two fields
        self.assertEqual(len(d.view_row(idx2)), 2)
        # check that d has the right shape and len
        self.assertEqual(d.shape, (2, 2))
        self.assertEqual(len(d), 2)

        # add a 2-D float field
        d.add_field('glexp', np.float64, 23, ndim=2)

        # add a new row with some data defined
        idx3 = d.add_row({'glexp': np.ones(2)})

        # set some data
        d[idx]['flurbosity'] = 1999
        d[idx2]['smilchritude'] = 'a little bit smilchy'

        # check that data we set is correct 
        self.assertEqual(d[idx]['flurbosity'], 1999.0)
        self.assertEqual(d[idx2]['smilchritude'], 'a little bit smilchy')
        self.assertTrue(np.array_equal(d[idx3]['glexp'], [1, 1]))

        # check that this setting data to the already-existing data does nothing
        d._set_data(d._data)
        self.assertEqual(d[idx]['flurbosity'], 1999.0)
        self.assertEqual(d[idx2]['smilchritude'], 'a little bit smilchy')
        self.assertTrue(np.array_equal(d[idx3]['glexp'], [1, 1]))

        # check that view_field('flurbosity') produces the correct value
        val = np.ma.zeros(3)
        val[0] = 1999
        val.mask = [False, True, True]
        val.set_fill_value(-17.76)
        self.assertTrue((d.view_field('flurbosity') == val).all())

        # check that view_field('flurbosity', type=np.float64) produces the correct value
        val = val.astype(np.float64)
        val.mask = [False, True, True]
        self.assertTrue((d.view_field('flurbosity', type=np.float64) == val).all())

        # save and restore the data
        s = State(d)
        d2 = s.restore()

        # check that the restored data has the right values
        self.assertEqual(d2[idx]['flurbosity'], 1999.0)
        self.assertEqual(d2[idx2]['smilchritude'], 'a little bit smilchy')
        self.assertTrue(np.array_equal(d2[idx3]['glexp'], [1, 1]))

        # check that we can set an entire field at once
        d['flurbosity'] = [23, 37, 43]
        self.assertTrue(np.array_equal(d.view_field('flurbosity') , [23, 37, 43]))

        # remove a field and a row
        d.remove_field('flurbosity')
        d.remove_row(idx2)

        # check that smilchritude is still masked
        self.assertIs(d.view_row(idx)['smilchritude'], np.ma.masked)

        # check that d has the right shape and len
        self.assertEqual(d.shape, (2, 2))
        self.assertEqual(len(d), 2)

        # add a row
        idx4 = d.add_row({'smilchritude': 'the smilchiest yet, by Jove!'})
        
        # check that d has the right shape and len
        self.assertEqual(d.shape, (3, 2))
        self.assertEqual(len(d), 3)
        self.assertEqual(d.view_row(idx4)['smilchritude'], 'the smilchiest yet, by Jove!')

        # deleting last field nulls the data array
        d.remove_field('smilchritude')
        d.remove_field('glexp')
        
        # check that d has the right shape and len after nulling
        self.assertEqual(d.shape, (0, 0))
        self.assertEqual(len(d), 0)

        # check that we can't _set_data() to something that isn't a masked array
        with self.assertRaises(AssertionError):
            d._set_data('wombat')

    def test_data_change_callback(self):
        """Check that the data change callback function gets called when we expect""" 
        log = []
        def f():
            log.append('foo')

        # check that log is empty
        self.assertEqual(len(log), 0)

        # create data and check that data_change_callback gets called once
        d = Data(data_change_callback=f)
        self.assertEqual(len(log), 1)

        # add a field and check that data_change_callback gets called once
        d.add_field('flurbosity', np.float64, -17.76)
        self.assertEqual(len(log), 2)

        # add a row and check that data_change_callback gets called
        idx = d.add_row()
        self.assertEqual(len(log), 4) # gets called twice for some reason

    def test_store_data(self):
        """Check that we can save and restore Data objects"""
        class Foo(StateObject):
            pass

        # create a Data object and add a field and a bunch of rows
        d = Data()
        d.add_field('flurbosity', np.float64, -17.76)
        d.add_row({'flurbosity': 19})
        d.add_row({'flurbosity': 23})
        d.add_row({'flurbosity': 47})

        # add a field of object type and put an object in the first row
        d.add_field('the_flaw_at_the_heart_of_it_all', np.dtype('O'), None)
        d[0]['the_flaw_at_the_heart_of_it_all'] = Foo()

        # save and restore
        s = State(d)
        d2 = s.restore()

        # check that flurbosity is what we expect
        self.assertTrue(np.array_equal(d2.view_field('flurbosity'), [19, 23, 47]))
        
        # check that object was restored correctly
        self.assertTrue(isinstance(d2.view_field('the_flaw_at_the_heart_of_it_all')[0], Foo))


if __name__ == '__main__':
    unittest.main()
