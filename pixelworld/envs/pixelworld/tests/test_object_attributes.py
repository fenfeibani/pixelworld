import unittest

import numpy as np

from pixelworld.envs.pixelworld import core, object_attributes, utils, universe, events, objects
from pixelworld.envs.pixelworld.tests import test_core
from pixelworld.envs.pixelworld.library.helpers import h

class TestObjectAttributeBlank(test_core.TestObjectAttribute):
    world = None
    
    def __init__(self, *args, **kwargs):
        super(TestObjectAttributeBlank, self).__init__(*args, **kwargs)
    
        self.world = kwargs.pop('world', 'blank')
        
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
        self.world = 'blank'


class TestMomentum(TestObjectAttributeBlank):
    def test_momentum(self):
        """Check that momentum.get() return value has the right shape"""
        self.world = universe.create_world('basic')
        momentum = self.world.object_attributes['momentum']
        values = momentum.get([3, 4, 5])
        self.assertEqual(values.shape, (3, 2))

    def test_momentum2(self):
        """Check that momentum of an object is what we expect"""
        x = self.world.create_object([['basic'], dict(velocity=(3, 3), mass=27)])
        momentum = self.world.get_object_attribute('momentum')
        np.testing.assert_allclose(x.momentum, (81, 81))
        np.testing.assert_allclose(momentum.get(x), (81, 81))


class TestColor(TestObjectAttributeBlank):
    def test_color(self):
        """Check that object color agrees with world state at object's index"""
        x = self.world.create_object([['basic'], dict(color=3)])
        si = tuple(utils.roundup(x.position).astype(np.int32))
        assert self.world.state[si] == 3

    def test_hierarchy(self):
        """Check that color coupling behaves as expected when hierarchies are
        assembled"""
        self.world.remove_objects(self.world.objects)

        # first we create a 2-level hierarchy:
        #         y     z
        #        / \   / \
        #       a   b c   d
        y = self.world.create_object('complex')
        z = self.world.create_object('complex')
        a = self.world.create_object('basic')
        b = self.world.create_object('basic')
        c = self.world.create_object('basic')
        d = self.world.create_object('basic')
        y.add_child(a)
        y.add_child(b)
        z.add_child(c)
        z.add_child(d)

        # set everyone's color to something unique
        a.color = 2
        b.color = 3
        c.color = 4
        d.color = 5

        # check that the parents' colors are what we expect
        self.assertEqual(y.color, 2)
        self.assertEqual(z.color, 4)

        # create a new top-level object that contains y and z
        x = self.world.create_object('complex')
        x.add_child(y)
        x.add_child(z)

        # check that everyone's color is still the same
        self.assertEqual(a.color, 2)
        self.assertEqual(b.color, 3)
        self.assertEqual(c.color, 4)
        self.assertEqual(d.color, 5)
        self.assertEqual(y.color, 2)
        self.assertEqual(z.color, 4)

        self.world.remove_objects([x])
        
        # check that y and z have no parent
        self.assertTrue(not hasattr(y, 'parent'))
        self.assertTrue(not hasattr(z, 'parent'))

        # check that everyone's color is still the same
        self.assertEqual(a.color, 2)
        self.assertEqual(b.color, 3)
        self.assertEqual(c.color, 4)
        self.assertEqual(d.color, 5)
        self.assertEqual(y.color, 2)
        self.assertEqual(z.color, 4)


class TestMass(TestObjectAttributeBlank):
    def test_mass(self):
        """Check that zero-mass objects can occupy the same space as another object,
        and that they no longer collide if they then gain mass."""
        x = self.world.create_object([['basic'], dict(color=3)])
        y = self.world.create_object([['basic'], dict(color=3, mass=0)])
        y.position = x.position
        self.assertTrue(np.array_equal(x.position, y.position))
        y.mass = 1
        self.assertTrue(np.linalg.norm(x.position - y.position) >= 1.0)


class TestDepth(TestObjectAttributeBlank):
    def test_depth(self):
        """Check that objects with different non-column depths can occupy the same
        position, and that they no longer collide if they then get the same
        depth."""
        x = self.world.create_object([['basic'], dict(color=3, depth=0, position=(3, 3))])
        y = self.world.create_object([['basic'], dict(color=3, depth=1, position=(3, 3))])
        self.assertTrue(np.array_equal(x.position, y.position), repr((x.position, y.position)))
        y.depth = 0
        self.assertTrue(np.linalg.norm(x.position - y.position) >= 1.0)


class TestVisible(TestObjectAttributeBlank):
    def test_visible(self):
        """Check that objects that become invisible get invisible color/zorder/mass,
        and that those properties are restored to their original values when
        the object becomes visible again."""
        x = self.world.create_object([['basic'], dict(color=3, zorder=17, mass=10.5)])
        x.visible = False
        self.assertEqual(x.color, 0)
        self.assertEqual(x.zorder, -np.inf)
        self.assertEqual(x.mass, 0)
        x.visible = True
        self.assertEqual(x.color, 3)
        self.assertEqual(x.zorder, 17)
        self.assertEqual(x.mass, 10.5)


class TestAcceleration(TestObjectAttributeBlank):
    def test_acceleration(self):
        """Check that acceleration has the expected effect on velocity and position"""
        x = self.world.create_object([['basic'], dict(position=(3, 3), acceleration=(1, 1))])
        self.assertTrue(np.array_equal(x.position, (3, 3)))
        self.assertTrue(np.array_equal(x.velocity, (0, 0)))
        self.assertTrue(np.array_equal(x.acceleration, (1, 1)))
        self.world.step()
        self.assertTrue(np.array_equal(x.position, (4, 4)))
        self.assertTrue(np.array_equal(x.velocity, (1, 1)))
        self.assertTrue(np.array_equal(x.acceleration, (1, 1)))
        self.world.step()
        self.assertTrue(np.array_equal(x.position, (6, 6)), repr(x.position))
        self.assertTrue(np.array_equal(x.velocity, (2, 2)))
        self.assertTrue(np.array_equal(x.acceleration, (1, 1)))
        self.world.step()
        np.testing.assert_allclose(x.position, (9, 9))
        self.assertTrue(np.array_equal(x.velocity, (3, 3)))
        self.assertTrue(np.array_equal(x.acceleration, (1, 1)))


class TestVelocity(TestObjectAttributeBlank):
    def test_velocity(self):
        """Check that even high-velocity objects cannot escape when there is a frame
        around them."""
        # create a frame
        self.world.create_object('frame')

        # get rid of any events that happened so far
        for evt in self.world.events:
            evt.remove()

        # create two high-velocity objects and aim them at each other
        x = self.world.create_object([['basic'], dict(position=(3, 3), velocity=(100, 100))])
        y = self.world.create_object([['basic'], dict(position=(4, 4), velocity=(-100, -100))])
        self.world.step()
        
        # check that first event is collision between x and y
        self.assertEqual(sorted(tuple(self.world.events[0].indices)), sorted((x.id, y.id)))

        # check that all events are collisions rather than leave-screen events
        for event in self.world.events:
            self.assertTrue(not isinstance(event, events.LeaveScreenEvent))
            self.assertTrue(isinstance(event, events.CollisionEvent))

        # check that x and y are still inside the frame
        self.assertTrue(1 <= x.position[0] <= self.world.height - 2)
        self.assertTrue(1 <= x.position[1] <= self.world.width - 2)
        self.assertTrue(1 <= y.position[0] <= self.world.height - 2)
        self.assertTrue(1 <= y.position[1] <= self.world.width - 2)


class TestPosition(TestObjectAttributeBlank):
    def test_position(self):
        """Check that when two objects occupy the same position, one of them gets
        moved. """
        x = self.world.create_object([['basic'], dict(position=(3, 3))])
        y = self.world.create_object([['basic'], dict(position=(3, 3))])
        self.assertTrue(np.linalg.norm(x.position - y.position) >= 1.0)

    def test_position_get_visible_indices(self):
        """Check that get_visible_indices() works as expected."""
        x = self.world.create_object([['basic'], dict(position=(3, 3))])
        y = self.world.create_object([['basic'], dict(position=(3, 3))])
        z = self.world.create_object([['basic'], dict(position=(100, 100))])
        
        position = self.world.object_attributes['position']

        # check that x and y are in visible indices but z is not
        self.assertTrue(x.id in position.get_visible_indices()[0])
        self.assertTrue(y.id in position.get_visible_indices()[0])
        self.assertTrue(z.id not in position.get_visible_indices()[0])

    def test_position_default_value(self):
        """Check that position._default_value() generates valid positions for simple
        and complex objects"""
        x = self.world.create_object([['basic'], dict(position=(3, 3))])
        position = self.world.object_attributes['position']

        # check that there is always a space of two between x and the edge of
        # the window
        for i in xrange(1000):
            posn = position._default_value(x)
            self.assertTrue(2 <= posn[0] <= 17)
            self.assertTrue(2 <= posn[1] <= 17)

        # create a 5x5 extent compound object
        w = self.world.create_object([['shape'], dict(shape="""
  X
 XXX
XXXXX
 XXX
  X
""")])

        # check that there is always a space of two between w and the edge of
        # the window
        for i in xrange(1000):
            posn = position._default_value(w, shape=w.shape)
            self.assertTrue(4 <= posn[0] <= self.world.height - 5)
            self.assertTrue(4 <= posn[1] <= self.world.width - 5)


class TestKineticEnergy(TestObjectAttributeBlank):
    def test_kinetic_energy(self):
        """Check that kinetic energy is what we expect"""
        x = self.world.create_object([['basic'], dict(velocity=(3, 3), mass=27)])
        self.assertEqual(x.kinetic_energy, 243.0)


class TestExtent(TestObjectAttributeBlank):
    def test_extent(self):
        """Check that extent of a complex object is what we expect"""

        # create a 5x5 extent compound object
        x = self.world.create_object([['shape'], dict(shape="""
  X
 XXX
XXXXX
 XXX
  X
""")])
        # set position of x's center
        x.position = (10, 10)

        # check extent
        self.assertTrue(np.array_equal(x.extent, (8, 8, 12, 12)))


class TestTopLeft(TestObjectAttributeBlank):
    def test_top_left(self):
        """Check that top-left of a complex object is what we expect and can be set"""

        # create a 5x5 extent compound object with top_left attribute
        x = self.world.create_object(['shape', dict(shape="""
  X
 XXX
XXXXX
 XXX
  X
""")])
        x.add_attribute('top_left')

        # set position of x's center
        x.position = (10, 10)

        # check top_left
        self.assertTrue(np.array_equal(x.top_left, (8, 8)))

        # check that setting top_left moves x's center
        x.top_left = (7, 7)
        self.assertTrue(np.array_equal(x.position, (9, 9)))

    def test_top_left_init(self):
        """Check that top-left of a complex object is what we expect and can be set
        during initialization"""

        # create a 5x5 extent compound object with top_left attribute
        x = self.world.create_object(['shape', dict(shape="""
  X
 XXX
XXXXX
 XXX
  X
""", top_left=(7, 7))])

        # check top_left has correct value
        self.assertTrue(np.array_equal(x.top_left, (7, 7)))

        # check that position has correct value
        self.assertTrue(np.array_equal(x.position, (9, 9)))


    def test_top_left_iterable(self):
        """Check that we can set and get top-left of an iterable of objects"""

        # create two 5x5 extent compound objects
        x = self.world.create_object(['shape', dict(shape="""
  X
 XXX
XXXXX
 XXX
  X
""", position=(9, 9))])
        y = self.world.create_object(['shape', dict(shape="""
  X
 XXX
XXXXX
 XXX
  X
""", position=(5, 5))])

        # add top_left attribute
        x.add_attribute('top_left')
        y.add_attribute('top_left')

        # set via iterable
        top_left = self.world.get_object_attribute('top_left')
        top_left.set([x, y], [(8, 8), (2, 2)])

        # check top_left has correct value
        self.assertTrue(np.array_equal(x.top_left, (8, 8)))

        # check that position has correct value
        self.assertTrue(np.array_equal(x.position, (10, 10)))

        # check top_left has correct value
        self.assertTrue(np.array_equal(y.top_left, (2, 2)))

        # check that position has correct value
        self.assertTrue(np.array_equal(y.position, (4, 4)))

        # get via iterable
        tls = top_left.get([x, y])
        self.assertTrue(np.array_equal(tls, ((8, 8), (2, 2))))


class TestShape(TestObjectAttributeBlank):
    def test_shape(self):
        """Check that shape of a complex object is what we expect"""
        self.world.remove_objects(self.world.objects)

        # create a compound object with 13 children
        x = self.world.create_object([['shape'], dict(shape="""
  X
 XXX
XXXXX
 XXX
  X
""")])

        # check that shape is what we expect
        self.assertEqual(len(x.shape), 13)
        shape = [tuple(row) for row in x.shape]
        for offset in [(-2, 0), (-1, -1), (-1, 0), (-1, 1), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2),
                       (1, -1), (1, 0), (1, 0), (2, 0)]:
            self.assertTrue(offset in shape)


class TestSprite(TestObjectAttributeBlank):
    def test_sprite(self):
        """Check that setting sprites causes the correct changes in visibility and
        color in all the child objects"""
        # create a sprite object
        x = self.world.create_object(['sprite', 
                                      {'sprites': h.sprite.load('aliens', 'alien'), 'animated': True}])

        # set sprite to zero, check that everything worked
        x.sprite = 0
        for yid, yvis, ycolor in zip(x.children, x._sprites[0]['visible'], x._sprites[0]['color']):
            y = self.world.objects[yid]
            self.assertEqual(y.visible, yvis)
            if yvis:
                self.assertEqual(y.color, ycolor)

        # set sprite to one, check that everything worked
        x.sprite = 1
        for yid, yvis, ycolor in zip(x.children, x._sprites[1]['visible'], x._sprites[1]['color']):
            y = self.world.objects[yid]
            self.assertEqual(y.visible, yvis)
            if yvis:
                self.assertEqual(y.color, ycolor)


class TestAnimated(TestObjectAttributeBlank):
    def test_animated(self):
        """Check that animated sprite objects cycle through their sprites when the
        world steps"""
        # create an animated sprite object
        x = self.world.create_object(['sprite', 
                                      {'sprites': h.sprite.load('aliens', 'alien'), 'animated': True}])

        # check that stepping the world cycles through the sprites
        self.assertEqual(x.sprite, 0)
        self.world.step()
        self.assertEqual(x.sprite, 1)
        self.world.step()
        self.assertEqual(x.sprite, 0)
        self.world.step()
        self.assertEqual(x.sprite, 1)
        self.world.step()
        self.assertEqual(x.sprite, 0)
        self.world.step()
        self.assertEqual(x.sprite, 1)
        self.world.step()


class TestPushes(TestObjectAttributeBlank):
    def test_pushes(self):
        """Check that pushing responds correctly to actions and that it generates the
        events we expect."""
        # create some objects
        x = self.world.create_object(['self', dict(position=(10, 10))])
        y = self.world.create_object(['basic', dict(position=(10, 11))])
        z = self.world.create_object(['basic', dict(position=(11, 10))])
        w = self.world.create_object(['immoveable', dict(position=(13, 10))])
        v = self.world.create_object(['self', dict(position=(1, 1))])

        # check that there are no events
        self.assertEqual(len(self.world.events), 0, repr(self.world.events))

        # check that stepping RIGHT has expected effects. x and v both have
        # successful pushes, and x pushes y.
        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(x.position, (10, 11)))
        self.assertTrue(np.array_equal(y.position, (10, 12)))
        self.assertTrue(np.array_equal(z.position, (11, 10)))
        self.assertTrue(np.array_equal(w.position, (13, 10)))
        self.assertTrue(np.array_equal(v.position, (1, 2)))
        self.assertEqual(len(self.world.events), 2, repr(self.world.events))
        self.assertTrue(isinstance(self.world.events[-2], events.PushEvent))
        self.assertTrue(isinstance(self.world.events[-1], events.PushEvent))
        self.assertTrue(self.world.events[-2].success)
        self.assertTrue(self.world.events[-1].success)
        self.assertEqual(self.world.events[-2].idx, x.id)
        self.assertEqual(self.world.events[-1].idx, v.id)

        # check that stepping LEFT has expected effects.  x and v both have
        # successful pushes
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(x.position, (10, 10)))
        self.assertTrue(np.array_equal(y.position, (10, 12)))
        self.assertTrue(np.array_equal(z.position, (11, 10)))
        self.assertTrue(np.array_equal(w.position, (13, 10)))
        self.assertTrue(np.array_equal(v.position, (1, 1)))
        self.assertEqual(len(self.world.events), 4)
        self.assertTrue(isinstance(self.world.events[-2], events.PushEvent))
        self.assertTrue(isinstance(self.world.events[-1], events.PushEvent))
        self.assertTrue(self.world.events[-2].success)
        self.assertTrue(self.world.events[-1].success)
        self.assertEqual(self.world.events[-2].idx, x.id)
        self.assertEqual(self.world.events[-1].idx, v.id)

        # check that stepping DOWN has expected effects. x and v both have
        # successful pushes, and x pushes z
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(x.position, (11, 10)))
        self.assertTrue(np.array_equal(y.position, (10, 12)))
        self.assertTrue(np.array_equal(z.position, (12, 10)))
        self.assertTrue(np.array_equal(w.position, (13, 10)))
        self.assertTrue(np.array_equal(v.position, (2, 1)))
        self.assertEqual(len(self.world.events), 6)
        self.assertTrue(isinstance(self.world.events[-2], events.PushEvent))
        self.assertTrue(isinstance(self.world.events[-1], events.PushEvent))
        self.assertTrue(self.world.events[-2].success)
        self.assertTrue(self.world.events[-1].success)
        self.assertEqual(self.world.events[-2].idx, x.id)
        self.assertEqual(self.world.events[-1].idx, v.id)

        # check taht stepping DOWN has expected effects. x's push fails because
        # w is immoveable, v's push succeeds
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(x.position, (11, 10)))
        self.assertTrue(np.array_equal(y.position, (10, 12)))
        self.assertTrue(np.array_equal(z.position, (12, 10)))
        self.assertTrue(np.array_equal(w.position, (13, 10)))
        self.assertTrue(np.array_equal(v.position, (3, 1)))
        self.assertEqual(len(self.world.events), 8)
        self.assertTrue(isinstance(self.world.events[-2], events.PushEvent))
        self.assertTrue(isinstance(self.world.events[-1], events.PushEvent))
        self.assertTrue(not self.world.events[-2].success)
        self.assertTrue(self.world.events[-1].success)
        self.assertEqual(self.world.events[-2].idx, x.id)
        self.assertEqual(self.world.events[-1].idx, v.id)
    
    def test_push_different_depths(self):
        """make sure objects don't push other objects at different depths"""
        x = self.world.create_object([['basic'], dict(color=3, depth=0, position=(3, 3))])
        y = self.world.create_object([['self'], dict(color=3, depth=1, position=(2, 3))])
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(x.position, y.position), repr((x.position, y.position)))
    
    def test_column_pushes_cube(self):
        """make sure a column can push a cube"""
        x = self.world.create_object([['basic'], dict(color=3, depth=0, position=(3, 3))])
        y = self.world.create_object([['self'], dict(color=3, position=(2, 3))])
        self.world.step('DOWN')
        self.assertFalse(np.array_equal(x.position, y.position), repr((x.position, y.position)))
    
    def test_cube_pushes_column(self):
        """make sure a cube can push a column"""
        x = self.world.create_object([['basic'], dict(color=3, position=(3, 3))])
        y = self.world.create_object([['self'], dict(color=3, depth=0, position=(2, 3))])
        self.world.step('DOWN')
        self.assertFalse(np.array_equal(x.position, y.position), repr((x.position, y.position)))
    
    def test_column_pushes_column(self):
        """make sure a column can push a column"""
        x = self.world.create_object([['basic'], dict(color=3, position=(3, 3))])
        y = self.world.create_object([['self'], dict(color=3, position=(2, 3))])
        self.world.step('DOWN')
        self.assertFalse(np.array_equal(x.position, y.position), repr((x.position, y.position)))
    
    


class TestShooting(TestObjectAttributeBlank):
    def test_shooting(self):
        """Check that shooting things generates the events we expect and that we can
        shoot in any direction."""

        # make a shooting self
        x = self.world.create_object(['self', dict(position=(10, 10))])
        x.add_attribute('orientation')
        x.add_attribute('self_shoots')

        # shoot to the right with killing_deletes = False
        x.orientation = 2 # right
        y = self.world.create_object(['basic', dict(position=(10, 12), alive=True)])
        self.world.step('SHOOT')
        self.assertEqual(len(self.world.events), 1)
        self.assertTrue(isinstance(self.world.events[0], events.BulletFireEvent))
        self.world.step('NOOP')
        self.world.step('NOOP')
        self.assertEqual(len(self.world.events), 3)
        # one kill event for the bullet, one for the target
        self.assertTrue(isinstance(self.world.events[-2], events.KillEvent))
        self.assertTrue(isinstance(self.world.events[-1], events.KillEvent))

        # shoot down with killing_deletes = True
        self.world.killing_deletes = True
        z = self.world.create_object(['basic', dict(position=(12, 10), alive=True)])
        x.orientation = 3 # down
        self.world.step('SHOOT')
        self.assertEqual(len(self.world.events), 4)
        self.assertTrue(isinstance(self.world.events[-1], events.BulletFireEvent))
        self.world.step('NOOP')
        self.world.step('NOOP')
        self.world.step('NOOP')
        self.assertEqual(len(self.world.events), 6)
        # one kill event for the bullet, one for the target
        self.assertTrue(isinstance(self.world.events[-2], events.KillEvent))
        self.assertTrue(isinstance(self.world.events[-1], events.KillEvent))

        # shoot left with killing_deletes = False
        self.world.killing_deletes = False
        # target is not alive and so cannot be killed
        z = self.world.create_object(['basic', dict(position=(10, 8))])
        x.orientation = 0 # left
        self.world.step('SHOOT')
        self.assertEqual(len(self.world.events), 7)
        self.assertTrue(isinstance(self.world.events[-1], events.BulletFireEvent))
        self.world.step('NOOP')
        self.world.step('NOOP')
        self.assertEqual(len(self.world.events), 8)
        # one kill event for the bullet
        self.assertTrue(isinstance(self.world.events[-1], events.KillEvent))

        # shoot up with killing_deletes = True
        self.world.killing_deletes = True
        # target is not alive and so cannot be killed
        z = self.world.create_object(['basic', dict(position=(8, 10))])
        x.orientation = 1 # up
        self.world.step('SHOOT')
        self.assertEqual(len(self.world.events), 9)
        self.assertTrue(isinstance(self.world.events[-1], events.BulletFireEvent))
        self.world.step('NOOP')
        self.world.step('NOOP')
        self.assertEqual(len(self.world.events), 10)
        # one kill event for the bullet
        self.assertTrue(isinstance(self.world.events[-1], events.KillEvent))


class TestGrips(TestObjectAttributeBlank):
    def test_gripper(self):
        """Check that gripper can pick things up and move them as expected"""
        # create some objects
        x = self.world.create_object(['self', dict(position=(10, 10))])
        y = self.world.create_object(['basic', dict(position=(10, 11))])
        z = self.world.create_object(['basic', dict(position=(11, 12))])

        # make x a gripper
        x.add_attribute('grips')

        # pick up y
        x.orientation = 2
        self.world.step('GRIP')

        # check that we picked up y and now x and y are both under a grip
        # object
        self.assertTrue(hasattr(x, 'parent'))
        self.assertTrue(hasattr(y, 'parent'))
        self.assertEqual(x.parent, y.parent)
        self.assertTrue(hasattr(x._parent, 'grip'))
        self.assertEqual(y.gripped, x)
        self.assertEqual(x._parent.grip, x)

        # check that we did not pick up z
        self.assertTrue(not hasattr(z, 'gripped'))

        # move within range of z
        self.world.step('RIGHT')
        self.world.step('RIGHT')

        # check that y moved with us
        self.assertTrue(np.array_equal(x.position + (0, 1), y.position))

        # check that z didn't move
        self.assertTrue(np.array_equal(z.position, (11, 12)))

        # pick up z
        x.orientation = 3
        self.world.step('GRIP')

        # move around
        self.world.step('UP')

        # check that y and z moved with us
        self.assertTrue(np.array_equal(x.position + (0, 1), y.position))
        self.assertTrue(np.array_equal(x.position + (1, 0), z.position))

        # move around
        self.world.step('LEFT')

        # check that y and z moved with us
        self.assertTrue(np.array_equal(x.position + (0, 1), y.position))
        self.assertTrue(np.array_equal(x.position + (1, 0), z.position))

        # move around
        self.world.step('DOWN')

        # check that y and z moved with us
        self.assertTrue(np.array_equal(x.position + (0, 1), y.position))
        self.assertTrue(np.array_equal(x.position + (1, 0), z.position))

        # move around
        self.world.step('RIGHT')

        # check that y and z moved with us
        self.assertTrue(np.array_equal(x.position + (0, 1), y.position))
        self.assertTrue(np.array_equal(x.position + (1, 0), z.position))

        # release everything
        self.world.step('UNGRIP')
        
        # remember y and z positions
        ypos = y.position
        zpos = z.position
        
        # move around
        self.world.step('UP')

        # check that y and z stayed put
        self.assertTrue(np.array_equal(y.position, ypos))
        self.assertTrue(np.array_equal(z.position, zpos))

    def test_gripper2(self):
        """Check that gripper can pick up complex objects and move them as expected
        without changing their color"""
        # create some objects
        x = self.world.create_object(['self', dict(position=(10, 10))])
        y = self.world.create_object(['complex', dict(shape='XXXX')])

        # make x a gripper
        x.add_attribute('grips')

        # move x adjacent to one of y's children
        x.position = y._children[0].position + (1, 0)

        # give everything colors
        x.color = 10
        for child, color in zip(y._children, range(2, 6)):
            child.color = color

        # pick up y
        x.orientation = 1
        self.world.step('GRIP')

        # check that parent is now a grip object
        self.assertEqual(x.parent, y.parent)
        self.assertEqual(x._parent.grip, x)

        # check that y is gripped
        self.assertEqual(y.gripped, x)

        # move around and check that y moves with us
        ypos = y.position
        self.world.step('UP')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, 0)))
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, -1)))
        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, 0)))
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(y.position, ypos))

        # check that we didn't mess up colors
        for child, color in zip(y._children, range(2, 6)):
            self.assertEqual(child.color, color)
        self.assertEqual(x.color, 10)

    def test_gripper3(self):
        """Check that complex gripper can pick up complex objects and move them as
        expected"""
        # create some objects
        x = self.world.create_object(['self_big', dict(shape='XXXX')])
        y = self.world.create_object(['complex', dict(shape='XXXX')])

        # make x a gripper
        x.add_attribute('grips')

        # move x adjacent to one of y's children
        x._children[-1].position = y._children[0].position + (1, 0)

        # pick up y
        x.orientation = 1
        self.world.step('GRIP')

        # check that parent is now a grip object
        self.assertEqual(x.parent, y.parent)
        self.assertEqual(x._parent.grip, x)

        # check that y is gripped
        self.assertEqual(y.gripped, x)

        # move around and check that y moves with us
        ypos = y.position
        self.world.step('UP')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, 0)))
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, -1)))
        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, 0)))
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(y.position, ypos))

    def test_gripper4(self):
        """Check that gripper cannot pick up immoveable objects"""
        # create some objects
        x = self.world.create_object(['self', dict(position=(10, 10))])
        y = self.world.create_object(['wall', dict(position=(11, 10))])
        
        # try to pick up y
        self.world.step('GRIP')

        # check that no grip object was created and y is not gripped
        self.assertIs(x._parent, None)
        self.assertIs(y._parent, None)
        self.assertTrue(not hasattr(y, 'gripped'))

        # move around and check that y does not move
        ypos = y.position
        self.world.step('UP')
        self.assertTrue(np.array_equal(y.position, ypos))

    def test_gripper5(self):
        """Check that complex gripper can pick up complex and simple objects as
        expected without changing their colors"""
        # create some objects
        x = self.world.create_object(['self_big', dict(shape='XXXX')])
        y = self.world.create_object(['complex', dict(shape='XXXX', color=3)])
        z = self.world.create_object('basic')

        # check the colors 
        self.assertEqual(z.color, 1)
        for a in y._children:
            self.assertEqual(a.color, 3)

        # make x a gripper
        x.add_attribute('grips')

        # move x adjacent to one of y's children
        x._children[-1].position = y._children[0].position + (1, 0)
        
        # move z adjacent to one of x's children
        z.position = x._children[0].position + (-1, 0)

        # pick up y and z
        x.orientation = 1
        self.world.step('GRIP')

        # check that colors didn't change
        self.assertEqual(z.color, 1)
        for a in y._children:
            self.assertEqual(a.color, 3)

        # check that parent is now a grip object
        self.assertEqual(x.parent, y.parent)
        self.assertEqual(x.parent, z.parent)
        self.assertEqual(x._parent.grip, x)

        # check that y and z are gripped
        self.assertEqual(y.gripped, x)
        self.assertEqual(z.gripped, x)

        # move around and check that y and z move with us
        ypos = y.position
        zpos = z.position
        self.world.step('UP')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, 0)))
        self.assertTrue(np.array_equal(z.position, zpos + (-1, 0)))
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, -1)))
        self.assertTrue(np.array_equal(z.position, zpos + (-1, -1)))
        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, 0)))
        self.assertTrue(np.array_equal(z.position, zpos + (-1, 0)))
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(y.position, ypos))
        self.assertTrue(np.array_equal(z.position, zpos))

        # release the objects
        self.world.step('UNGRIP')

        # check that colors didn't change
        self.assertEqual(z.color, 1)
        for a in y._children:
            self.assertEqual(a.color, 3)

    def test_gripper6(self):
        """Check that complex gripper can pick up complex and sprite objects as
        expected without changing their colors"""
        # create some objects
        x = self.world.create_object(['self_big', dict(shape='XXXX')])
        y = self.world.create_object(['complex', dict(shape='XXXX', color=3)])
        z = self.world.create_object(['complex_sprite', {'sprites': [h.sprite.from_string('44    44'),
                                                                     h.sprite.from_string('  4444  ')],
                                                         'animated': False}])

        # check the colors 
        for a in z._children:
            self.assertEqual(a.color, 4)
        for a in y._children:
            self.assertEqual(a.color, 3)

        # make x a gripper
        x.add_attribute('grips')

        # move x adjacent to one of y's children
        x._children[-1].position = y._children[0].position + (1, 0)
        
        # move z adjacent to one of x's children
        z._children[0].position = x._children[0].position + (1, 0)

        # pick up y
        x.orientation = 1
        self.world.step('GRIP')

        # pick up z
        x.orientation = 3
        self.world.step('GRIP')

        # check that colors didn't change
        for a in z._children:
            self.assertEqual(a.color, 4)
        for a in y._children:
            self.assertEqual(a.color, 3)

        # check that parent is now a grip object
        self.assertEqual(x.parent, y.parent)
        self.assertEqual(x.parent, z.parent)
        self.assertEqual(x._parent.grip, x)

        # check that y and z are gripped
        self.assertEqual(y.gripped, x)
        self.assertEqual(z.gripped, x)

        # move around and check that y and z move with us
        ypos = y.position
        zpos = z.position
        self.world.step('UP')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, 0)))
        self.assertTrue(np.array_equal(z.position, zpos + (-1, 0)))
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, -1)))
        self.assertTrue(np.array_equal(z.position, zpos + (-1, -1)))
        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(y.position, ypos + (-1, 0)))
        self.assertTrue(np.array_equal(z.position, zpos + (-1, 0)))
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(y.position, ypos))
        self.assertTrue(np.array_equal(z.position, zpos))

        # release the objects
        self.world.step('UNGRIP')

        # check that colors didn't change
        for a in z._children:
            self.assertEqual(a.color, 4)
        for a in y._children:
            self.assertEqual(a.color, 3)

    def test_gripper7(self):
        """Check that gripper with velocity can pick up objects as expected and that
        velocity coupling is respected"""
        # create some objects
        x = self.world.create_object(['self', dict(position=(10, 9))])
        y = self.world.create_object(['basic', dict(position=(10, 11))])

        # make x a gripper
        x.add_attribute('grips')

        # give x velocity
        x.velocity = (0, 1)

        # orient x towards y
        x.orientation = 2

        # try to pick up y
        self.world.step('GRIP')
        self.world.step('GRIP')
        self.world.step('GRIP')

        # check that we picked up y and now x and y are both under a grip
        # object
        self.assertTrue(hasattr(x, 'parent'))
        self.assertTrue(hasattr(y, 'parent'))
        self.assertEqual(x.parent, y.parent)
        self.assertTrue(hasattr(x._parent, 'grip'))
        self.assertEqual(y.gripped, x)
        self.assertEqual(x._parent.grip, x)

        self.assertTrue(np.array_equal(x.velocity, y.velocity))

        # move around
        self.world.step('UP')

        # check that y moved with us
        self.assertTrue(np.array_equal(x.position + (0, 1), y.position))

        # move around
        self.world.step('LEFT')

        # check that y moved with us
        self.assertTrue(np.array_equal(x.position + (0, 1), y.position))

        # move around
        self.world.step('DOWN')

        # check that y moved with us
        self.assertTrue(np.array_equal(x.position + (0, 1), y.position))

        # move around
        self.world.step('RIGHT')

        # check that y moved with us
        self.assertTrue(np.array_equal(x.position + (0, 1), y.position))

        # release everything
        self.world.step('UNGRIP')
        
        # remember y position
        ypos = y.position
        
        # move around
        self.world.step('UP')

        # check that y stayed put
        self.assertTrue(np.array_equal(y.position, ypos))

    def test_gripper8(self):
        """Check that gripper can take inappropriate UNGRIP actions without causing
        problems"""
        # create some objects
        x = self.world.create_object(['self', dict(position=(10, 10))])
        y = self.world.create_object(['basic', dict(position=(10, 11))])

        # make x a gripper
        x.add_attribute('grips')

        # orient x towards y
        x.orientation = 2

        # check that we can UNGRIP as first action
        self.world.step('UNGRIP')

        # grip y
        self.world.step('GRIP')

        # ungrip y
        self.world.step('UNGRIP')

        # check that we have a successful UNGRIP event
        self.assertEqual(len(self.world.step_events), 1)
        self.assertTrue(isinstance(self.world.step_events[0], events.UngripEvent))
        self.assertEqual(self.world.step_events[0].success, True)

        # check that we can UNGRIP when we aren't gripping anything
        self.world.step('UNGRIP')
        
        # check that we have a failed UNGRIP event
        self.assertEqual(len(self.world.step_events), 1)
        self.assertTrue(isinstance(self.world.step_events[0], events.UngripEvent))
        self.assertEqual(self.world.step_events[0].success, False)

    def test_gripper9(self):
        """Check that a complex sprite object can grip a complex object and move it as
        expected"""
        # create some objects
        y = self.world.create_object(['complex', dict(shape='XXXX', color=3, position=(10, 10))])
        x = self.world.create_object(['self_sprite', {'sprites': [h.sprite.from_string('44    44'),
                                                                  h.sprite.from_string('  4444  ')],
                                                      'position': (5, 5),
                                                      'animated': False}])

        # make x a gripper
        x.add_attribute('grips')

        # move y adjacent to one of x's children
        y._children[0].position = x._children[0].position + (1, 0)

        x.orientation = 3
        self.world.step('GRIP')
        self.world.step('GRIP')
        self.world.step('GRIP')

        # check that we picked up y and now x and y are both under a grip
        # object
        self.assertTrue(hasattr(x, 'parent'))
        self.assertTrue(hasattr(y, 'parent'))
        self.assertEqual(x.parent, y.parent)
        self.assertTrue(hasattr(x._parent, 'grip'))
        self.assertEqual(y.gripped, x)
        self.assertEqual(x._parent.grip, x)

        # remember y position
        ypos = y.position

        # move around
        self.world.step('UP')

        # check that y moved with us
        self.assertTrue(np.array_equal(ypos + (-1, 0), y.position))

        # move around
        self.world.step('LEFT')

        # check that y moved with us
        self.assertTrue(np.array_equal(ypos + (-1, -1), y.position))

        # move around
        self.world.step('DOWN')

        # check that y moved with us
        self.assertTrue(np.array_equal(ypos + (0, -1), y.position))

        # move around
        self.world.step('RIGHT')

        # check that y moved with us
        self.assertTrue(np.array_equal(ypos, y.position))

        # release everything
        self.world.step('UNGRIP')
        
        # move around
        self.world.step('UP')

        # check that y stayed put
        self.assertTrue(np.array_equal(y.position, ypos))

    def test_gripper_nonpushing(self):
        """Check that gripper without pushes can still pick up objects as expected and
        that velocity coupling is respected"""
        # create some objects
        x = self.world.create_object(['basic', dict(position=(10, 9), orientation=2)])
        y = self.world.create_object(['basic', dict(position=(10, 11))])

        # make x a gripper
        x.add_attribute('grips')

        # give x velocity
        x.velocity = (0, 1)

        # orient x towards y
        x.orientation = 2

        # try to pick up y
        self.world.step('GRIP')
        self.world.step('GRIP')
        self.world.step('GRIP')

        # check that we picked up y and now x and y are both under a grip
        # object
        self.assertTrue(hasattr(x, 'parent'))
        self.assertTrue(hasattr(y, 'parent'))
        self.assertEqual(x.parent, y.parent)
        self.assertTrue(hasattr(x._parent, 'grip'))
        self.assertEqual(y.gripped, x)
        self.assertEqual(x._parent.grip, x)

        self.assertTrue(np.array_equal(x.velocity, y.velocity))

        # release y
        self.world.step('UNGRIP')

        # check that x still doesn't have pushes
        self.assertFalse(hasattr(x, 'pushes'))

    def test_gripper_nonpushing2(self):
        """Check that gripper with pushes=False can still pick up objects as expected
        and that velocity coupling is respected"""
        # create some objects
        x = self.world.create_object(['self', dict(position=(10, 9), pushes=False)])
        y = self.world.create_object(['basic', dict(position=(10, 11))])

        # make x a gripper
        x.add_attribute('grips')

        # give x velocity
        x.velocity = (0, 1)

        # orient x towards y
        x.orientation = 2

        # try to pick up y
        self.world.step('GRIP')
        self.world.step('GRIP')
        self.world.step('GRIP')

        # check that we picked up y and now x and y are both under a grip
        # object
        self.assertTrue(hasattr(x, 'parent'))
        self.assertTrue(hasattr(y, 'parent'))
        self.assertEqual(x.parent, y.parent)
        self.assertTrue(hasattr(x._parent, 'grip'))
        self.assertEqual(y.gripped, x)
        self.assertEqual(x._parent.grip, x)

        self.assertTrue(np.array_equal(x.velocity, y.velocity))

        # release y
        self.world.step('UNGRIP')

        # check that x still has pushes=False
        self.assertFalse(x.pushes)


"""
def single_test(cls, test):
    def suite():
        suite = unittest.TestSuite()
        suite.addTest(cls(test))
        return suite

    s = suite()
    unittest.TextTestRunner(verbosity=100).run(s)
"""


if __name__ == '__main__':
    unittest.main()

    #single_test(TestTopLeft, 'test_top_left_iterable')
