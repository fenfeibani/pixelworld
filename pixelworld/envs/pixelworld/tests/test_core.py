"""Unit tests for PixelWorld
"""

import os, sys
import unittest
import numpy as np
from numbers import Number
from copy import copy, deepcopy
import dill

from pixelworld.envs.pixelworld import core, universe, agents, randomizers, utils
from pixelworld.envs.pixelworld.tests.utils import check_deepcopy


class TestCase(unittest.TestCase):
    def shortDescription(self):
        """Return None to make unit test descriptions use function name rather than
        docstring"""
        return None

    def setUp(self):
        pass
    
    def tearDown(self):
        pass


class TestWorld(TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestWorld, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
    
    def get_all_attributes(self, obj):
        """get all of an objects attributes to test for errors"""
        for attr in dir(obj):
            x = getattr(obj, attr)
    
    def examine_entity(self, entity):
        """examine an entity to test for errors"""
        self.get_all_attributes(entity)
    
    def examine_object_attribute(self, attr):
        """examine an object attribute to test for errors"""
        self.examine_entity(attr)
        
        num_objects = np.sum(~attr._get_mask())
        
        #get all values
        x = attr.get()
        self.assertEqual(len(x), num_objects, 'unexpected %s.get() length' % (attr.name))
        self.assertIsInstance(x, np.ndarray, 'unexpected %s.get() type' % (attr.name))
        self.assertEqual(x.dtype, attr.dtype, 'unexpected %s.get() type' % (attr.name))
        
        #set all values
        if not attr._read_only:
            attr.set(None, attr._null_value)
            attr.set(None, x)
            y = attr.get()
            
            attr.set(None, attr._null_value)
            attr.set(None, x)
            z = attr.get()
            
            for a,b in zip(y, z):
                if isinstance(a, np.ndarray):
                    ab_equal = np.allclose(a, b)
                else:
                    ab_equal = a == b
                
                if not ab_equal:
                    print a, b
                
                if attr._name not in ['position']:
                    self.assertTrue(ab_equal, 'unexpected %s.set() behavior (seed=%d)' % (attr.name, attr.world._seed_value))
    
    def examine_object(self, obj):
        """examine an object to test for errors"""
        self.examine_entity(obj)
    
    def examine_event(self, event):
        """examine an event to test for errors"""
        self.examine_entity(event)
    
    def examine_judge(self, judge):
        """examine a judge to test for errors"""
        self.examine_entity(judge)
    
    def examine_agent(self, agent):
        """examine an agent to test for errors"""
        self.examine_entity(agent)
    
    def test_world(self):
        """examine everything in the world to test for errors"""
        #world-specific tests
        self.get_all_attributes(self.world)
        
        #object attributes
        for attr in self.world.object_attributes.values():
            self.examine_object_attribute(attr)
        
        #objects
        for obj in self.world.objects:
            self.examine_object(obj)
        
        #events
        for event in self.world.events:
            self.examine_event(event)
        
        #judge
        self.examine_judge(self.world.judge)
        
        #agent
        for agent in self.world._agent:
            self.examine_agent(agent)

    def test_deletion(self):
        """Check that we can delete objects without causing problems"""
        objs = list(self.world.objects)
        while len(objs) > 0:
            obj = objs.pop()
            self.world.remove_objects([obj])

            #world-specific tests
            self.get_all_attributes(self.world)

            #object attributes
            for attr in self.world.object_attributes.values():
                self.examine_object_attribute(attr)

            #objects
            for obj in self.world.objects:
                self.examine_object(obj)

            #events
            for event in self.world.events:
                self.examine_event(event)

    def test_visible_state(self):
        """Check that every visible object shows up in the world state"""
        # basic world has no occlusion, so every simple object should
        # appear. this unit test is being stochastic though.
        self.world.randomizer.randomize(seed=0)

        for i, obj in enumerate(self.world.objects):
            # ignore compound objects
            if obj._class_type == core.Object.COMPOUND:
                continue
            
            # ignore invisible objects
            if not hasattr(obj, 'visible') or not obj.visible:
                continue

            # check that object color shows up in world state at object
            # position
            posn = tuple(np.int32(utils.roundup(obj.position)))
            self.assertTrue(self.world.state[posn] == obj.color)
            obj.color = i
            self.assertTrue(self.world.state[posn] == obj.color)

    def test_object_state(self):
        """Check that object-based state is in agreement with each object's
        attributes"""
        for d in self.world.object_state:
            obj = self.world.objects[d['id']]
            for k in d:
                if isinstance(d[k], np.ndarray):
                    self.assertTrue(np.array_equal(getattr(obj, k), d[k]), d[k])
                else:
                    self.assertTrue(getattr(obj, k) == d[k], d[k])

    def test_data_state(self):
        """Check that data-array state is exactly what we expect, including the
        mask."""
        ds = self.world.data_state
        nobj, nattr = ds.shape

        # make sure that we're observing all of the attributes
        self.assertEquals(self.world._observed_attributes, None)

        for i in xrange(nobj):
            obj = self.world.objects[i]
            # multi-dimensional attributes take up more than one column of the
            # data_state array, so column #j does not correspond to attribute
            # #j. finger keeps track of the starting column of the current
            # attribute. each attribute attr occupies columns
            # finger:finger+attr.ndim.
            finger = 0
            for attr in sorted(self.world._object_attributes.keys()):
                ndim = self.world.object_attributes[attr].ndim
                assert ndim > 0

                if hasattr(obj, attr):
                    if isinstance(getattr(obj, attr), np.ndarray):
                        # assert that the values we get from data_state are
                        # equal to those we get from obj.attr
                        attrval = getattr(obj, attr)
                        dsval = ds[i, finger:finger + ndim]
                        if isinstance(self.world._object_attributes[attr], core.ObjectObjectAttribute):
                            # ObjectObjectAttributes are wrapped in an extra
                            # dimension for some reason
                            self.assertEqual(attrval.tolist(), dsval[0].tolist())
                        else:
                            self.assertEqual(attrval.tolist(), dsval.tolist())
                        # assert that all the relevant columns are unmasked
                        self.assertFalse(ds.mask[i, finger:finger + ndim].any())
                    else:
                        # assert that data_state is unmaksed here, and that the
                        # value is equal to what we get from obj.attr
                        self.assertFalse(ds.mask[i, finger], attr)
                        self.assertTrue(ds[i, finger] == getattr(obj, attr))

                # we don't have the attr, so assert that all the values are
                # masked
                else:
                    if ndim > 1:
                        self.assertTrue(ds.mask[i, finger:finger + ndim].all())
                    else:
                        self.assertTrue(ds.mask[i, finger])

                # move finger to the next attribute
                finger += ndim

    def test_step(self):
        """Check that the simulation aborts when the embedded agent doesn't return an
        action."""
        class MyAgent(core.Agent):
            def get_action(self, *args):
                return None

        # replace the world agent with MyAgent
        self.world.agent = MyAgent(self.world)

        # check that we abort if the embedded agent returns None as its action
        obs, reward, done, info = self.world.step()
        self.assertTrue(done)
        self.assertTrue(info['aborted'] == True)

    def test_run(self):
        """Check that we can run the world for 100 steps"""
        self.world.run(num_steps=100)

    def test_clear(self):
        """Check that we have no objects or attributes after the world is cleared, and
        that the world state is a 0x0 array"""
        self.world.clear()
        self.assertTrue(len(self.world.objects) == 0)
        self.assertTrue(self.world.object_attributes == {})
        self.assertTrue(self.world.state.shape == (0, 0))
    
    def test_state(self):
        """Check that we can rewind by loading a snapshot and get the same state after
        the same number of steps."""
        #run for a bit and save a snapshot
        self.world.run(num_steps=10)
        self.world.save_snapshot('test')
        
        #run for a bit and make an observation
        self.world.run(num_steps=10)
        obs1 = self.world.observe()
        
        #rewind and make sure we get the same observation
        self.world.load_snapshot('test')
        self.world.run(num_steps=10)
        obs2 = self.world.observe()
        
        if isinstance(obs1, np.ndarray):
            test = np.array_equal(obs1, obs2)
        else:
            test = obs1 == obs2
        
        self.assertTrue(test, 'rewind did not work!')

    def test_get_object_attribute(self):
        """Check that get_object_attribute() adds the object attribute if it does not
         exist already, and does not add it if it does exist already."""
        class FooObjectAttribute(core.ObjectAttribute):
            pass

        nattr = len(self.world.object_attributes)
        foo = self.world.get_object_attribute('foo')
        self.assertEqual(len(self.world.object_attributes), nattr + 1)
        foo2 = self.world.get_object_attribute('foo')
        self.assertEqual(len(self.world.object_attributes), nattr + 1)

    def test_remove_object_attribute(self):
        """Check that we can remove object attributes from the world"""
        class FooObjectAttribute(core.ObjectAttribute):
            pass

        # add the attribute foo
        nattr = len(self.world.object_attributes)
        foo = self.world.get_object_attribute('foo')
        self.assertEqual(len(self.world.object_attributes), nattr + 1)

        # check that we can remove the attribute foo
        foo.remove()
        self.assertEqual(len(self.world.object_attributes), nattr)                

    def test_reset(self):
        """Check that we can reset the world without breaking it"""
        self.world.reset()
        #world-specific tests
        self.get_all_attributes(self.world)
        
        #object attributes
        for attr in self.world.object_attributes.values():
            self.examine_object_attribute(attr)
        
        #objects
        for obj in self.world.objects:
            self.examine_object(obj)
        
        #events
        for event in self.world.events:
            self.examine_event(event)
        
        #judge
        self.examine_judge(self.world.judge)
        
        #agent
        self.examine_agent(self.world.agent)
    
    def test_full_history(self):
        """Check that we can replay the history of the world and get the same state."""
        for _ in xrange(100): self.world.step()
        
        state = self.world.state.copy()
        
        self.world.history.replay(self.world)
        
        self.assertTrue(np.array_equal(state, self.world.state))
    
    def test_history(self):
        """Check that history replay behaves as we expect"""
        self.world.load_snapshot('initial')
        # replace the agent with a random agent
        self.world.agent = 'random'

        # step the world for a bit
        for i in xrange(100):
            self.world.step()
        # reseed the world
        self.world.seed(0)
        # step the world some more
        for i in xrange(100):
            self.world.step()
        # get state at end of all this
        state = self.world.state.copy()
        # get history
        history = deepcopy(self.world.history)

        # check that history starts with clear, then seed, then populate
        self.assertEqual(history[0]['type'], 'clear')
        self.assertEqual(history[1]['type'], 'seed')
        self.assertEqual(history[2]['type'], 'populate')

        # check that history replayed from after populate onwards results in
        # the same final state and the same history. here we replay the history
        # in two chunks.
        self.world.load_snapshot('initial')
        self.world.agent = 'random'
        history[3:50].replay(self.world)
        history[50:].replay(self.world)
        state2 = self.world.state.copy()
        history2 = deepcopy(self.world.history)
        self.assertTrue(np.array_equal(state, state2))
        self.assertEqual(history, history2)

        # check that history replayed from after populate onwards results in
        # the same final state and the same history. here we replay the history
        # in one chunk.
        self.world.load_snapshot('initial')
        self.world.agent = 'random'
        history[3:].replay(self.world)
        state3 = self.world.state.copy()
        history3 = deepcopy(self.world.history)
        self.assertTrue(np.array_equal(state, state3))
        self.assertEqual(history, history3)

        # here we check that replaying a copy of the new history from after
        # populate onwards results in the same final state and the same history.
        history4 = deepcopy(self.world.history)
        self.world.load_snapshot('initial')
        self.world.agent = 'random'
        history4[3:].replay(self.world)
        state4 = self.world.state.copy()
        history5 = deepcopy(self.world.history)
        self.assertTrue(np.array_equal(state, state4))
        self.assertEqual(history, history4)
        self.assertEqual(history, history5)

        # here we check that replaying the history from after populate onwards
        # results in the same final state and the same history. we use the
        # ability to set world.agent
        self.world.load_snapshot('initial')
        self.world.agent = 'random'
        history[3:].replay(self.world)
        state5 = self.world.state.copy()
        history6 = deepcopy(self.world.history)
        self.assertTrue(np.array_equal(state, state5))
        self.assertEqual(history, history6)

    def test_history_multi(self):
        """Check that history replay behaves as we expect when there are multiple
        agents."""
        self.world.load_snapshot('initial')
        # replace the agent with two random agents
        self.world.agent = ['random', 'random']

        # step the world for a bit
        for i in xrange(10):
            self.world.step()
        # reseed the world
        self.world.seed(0)
        # step the world some more
        for i in xrange(10):
            self.world.step()
        # get state at end of all this
        state = self.world.state.copy()
        # get history
        history = deepcopy(self.world.history)

        # check that history starts with clear, then seed, then populate
        self.assertEqual(history[0]['type'], 'clear')
        self.assertEqual(history[1]['type'], 'seed')
        self.assertEqual(history[2]['type'], 'populate')

        # check that history replayed from after populate onwards results in
        # the same final state and the same history. here we replay the history
        # in two chunks.
        self.world.load_snapshot('initial')
        self.world.agent = ['random', 'random']
        history[3:5].replay(self.world)
        history[5:].replay(self.world)
        state2 = self.world.state.copy()
        history2 = deepcopy(self.world.history)
        self.assertTrue(np.array_equal(state, state2))

        # check that history replayed from after populate onwards results in
        # the same final state and the same history. here we replay the history
        # in one chunk.
        self.world.load_snapshot('initial')
        self.world.agent = ['random', 'random']
        history[3:].replay(self.world)
        state3 = self.world.state.copy()
        history3 = deepcopy(self.world.history)
        self.assertTrue(np.array_equal(state, state3))

        # here we check that replaying a copy of the new history from after
        # populate onwards results in the same final state and the same history.
        history4 = deepcopy(self.world.history)
        self.world.load_snapshot('initial')
        self.world.agent = ['random', 'random']
        history4[3:].replay(self.world)
        state4 = self.world.state.copy()
        history5 = deepcopy(self.world.history)
        self.assertTrue(np.array_equal(state, state4))

        # here we check that replaying the history from after populate onwards
        # results in the same final state and the same history. we use the
        # ability to set world.agent
        self.world.load_snapshot('initial')
        self.world.agent = ['random', 'random']
        history[3:].replay(self.world)
        state5 = self.world.state.copy()
        history6 = deepcopy(self.world.history)
        self.assertTrue(np.array_equal(state, state5))

        # here we check that all the histories produced are the same
        self.assertEqual(history, history2)
        self.assertEqual(history, history3)
        self.assertEqual(history, history4)
        self.assertEqual(history, history5)
        self.assertEqual(history, history6)

    def test_multi_agent(self):
        """Check that we can add and remove agents in various ways and that we end up
        in multi-agent mode when we should."""
        # check that we start off in single-agent mode
        self.assertFalse(self.world.multi_agent_mode)

        # check that removing the single agent leaves us in single-agent mode
        self.world._remove_agent(self.world.agent)
        self.assertFalse(self.world.multi_agent_mode)
        
        # check that adding two agents puts us in multi-agent mode, and that
        # world.agent is now a list
        agent1 = agents.RandomAgent(self.world)
        agent2 = agents.RandomAgent(self.world)
        self.assertTrue(self.world.multi_agent_mode)
        self.assertEqual(len(self.world.agent), 2)

        # check that removing the second agent leaves us in multi-agent mode,
        # and that world.agent is still a list
        self.world._remove_agent(agent2)
        self.assertTrue(self.world.multi_agent_mode)
        self.assertEqual(len(self.world.agent), 1)

        # check that we can add additional agents and stay in multi-agent mode
        agent3 = agents.RandomAgent(self.world)
        agent4 = agents.RandomAgent(self.world)
        self.assertTrue(self.world.multi_agent_mode)
        self.assertEqual(len(self.world.agent), 3)

        # check that clearing the world puts us in single-agent mode and that
        # world.agent is None after a clear
        self.world.clear()
        self.assertFalse(self.world.multi_agent_mode)
        self.assertEqual(self.world.agent, None)

        # add two new agents and check that that puts us back in multi-agent
        # mode, and that world.agent is again a list
        agent5 = agents.RandomAgent(self.world)
        agent6 = agents.RandomAgent(self.world)
        self.assertTrue(self.world.multi_agent_mode)
        self.assertEqual(len(self.world.agent), 2)

        # check that removing the second agent leaves us in multi-agent mode,
        # and that world.agent is still a list
        agent6.remove()
        self.assertTrue(self.world.multi_agent_mode)
        self.assertEqual(len(self.world.agent), 1)

        # create a new world with 10 random agents, and check that that puts us
        # in multi-agent mode
        self.world = universe.create_world('basic', agent=['random'] * 10)
        self.assertTrue(self.world.multi_agent_mode)
        self.assertEqual(len(self.world.agent), 10)

        # create a new world with a non-list passed for the agent argument, and
        # check that that puts us in single-agent mode
        self.world = universe.create_world('basic', agent='random')
        self.assertFalse(self.world.multi_agent_mode)
        self.assertTrue(isinstance(self.world.agent, agents.RandomAgent))
        
        # check that we can set world.agent without leaving single-agent mode
        self.world.agent = agents.RandomAgent
        self.assertFalse(self.world.multi_agent_mode)

        # check that we can set world.agent to a single entity specification
        # without leaving single-agent mode
        self.world.agent = [agents.RandomAgent, {}]
        self.assertFalse(self.world.multi_agent_mode)

        # check that setting world.agent to a list of two agents puts us in
        # multi-agent mode
        self.world.agent = 2 * [agents.RandomAgent]
        self.assertTrue(self.world.multi_agent_mode)

    def test_multi_agent2(self):
        """Check that push actions work correctly in multi-agent mode with two
        agents."""
        # get rid of the world's agents and objects
        self.world.remove_objects(self.world.objects)
        self.world.agent.remove()
        
        # add two agents
        agent1 = core.Agent(self.world)
        agent2 = core.Agent(self.world)
        
        # add two self objects, one controlled by each agent
        x, y = self.world.create_objects([['self', dict(controller=0, position=(10, 10))], 
                                          ['self', dict(controller=1, position=(13, 13))]])

        # check that the self objects respond to supplied actions
        self.world.step(['LEFT', 'RIGHT'])
        self.assertTrue(np.array_equal(x.position, (10, 9)))
        self.assertTrue(np.array_equal(y.position, (13, 14)))
        self.world.step(['RIGHT', 'LEFT'])
        self.assertTrue(np.array_equal(x.position, (10, 10)))
        self.assertTrue(np.array_equal(y.position, (13, 13)))
        self.world.step(['RIGHT', 'LEFT'])
        self.assertTrue(np.array_equal(x.position, (10, 11)))
        self.assertTrue(np.array_equal(y.position, (13, 12)))
        self.world.step(['DOWN', 'LEFT'])
        self.assertTrue(np.array_equal(x.position, (11, 11)))
        self.assertTrue(np.array_equal(y.position, (13, 11)))
        self.world.step(['DOWN', 'NOOP'])
        self.assertTrue(np.array_equal(x.position, (12, 11)))
        self.assertTrue(np.array_equal(y.position, (13, 11)))

        # check that the self objects cannot push each other
        self.world.step(['DOWN', 'UP'])
        self.assertTrue(np.array_equal(x.position, (12, 11)))
        self.assertTrue(np.array_equal(y.position, (13, 11)))
        self.world.step(['DOWN', 'NOOP'])
        self.assertTrue(np.array_equal(x.position, (12, 11)))
        self.assertTrue(np.array_equal(y.position, (13, 11)))
        self.world.step(['NOOP', 'UP'])
        self.assertTrue(np.array_equal(x.position, (12, 11)))
        self.assertTrue(np.array_equal(y.position, (13, 11)))

    def test_serialization(self):
        """Check that we can serialize States and restore a world from them
        afterwards."""
        self.world.load_snapshot('initial')
        
        # get the initial snapshot
        state = self.world._snapshots['initial']

        # remember the initial world state
        world_state = deepcopy(self.world.state)
        
        # replace the world agent with a random agent
        self.world.agent = 'random'

        # step for a bit
        for i in xrange(100):
            self.world.step()
            
        # save a snapshot at the end of the simulation
        self.world.save_snapshot('foo')

        # get the final snapshot
        state2 = self.world._snapshots['foo']
        
        # remember the final world state
        world_state2 = deepcopy(self.world.state)
        
        # check that we can restore from the given states
        w = state.restore()
        self.assertTrue(np.array_equal(world_state, w.state))
        
        w = state2.restore()
        self.assertTrue(np.array_equal(world_state2, w.state))

        # serialize and deserialize the states
        state3 = dill.loads(dill.dumps(state, dill.HIGHEST_PROTOCOL))
        state4 = dill.loads(dill.dumps(state2, dill.HIGHEST_PROTOCOL))

        # check that we can restore from the initial snapshot, and that the
        # world state is equal to the original initial world state
        w = state3.restore()
        self.assertEqual(w.time, 0)
        self.assertTrue(np.array_equal(world_state, w.state))

        # check that we can restore from the final snapshot, and that the world
        # state is equal to the original final world state
        w = state4.restore()
        self.assertEqual(w.time, 100)
        self.assertTrue(np.array_equal(world_state2, w.state))
    
    def test_entity_setting(self):
        """make sure we can set Entities via their attributes"""
        for cls in [core.Judge, core.Randomizer, core.Agent]:
            name = cls._name
            
            setattr(self.world, name, cls)
            self.assertTrue(isinstance(getattr(self.world, name), cls))
            
            obj = cls(self.world)
            setattr(self.world, name, obj)
            self.assertIs(getattr(self.world, name), obj)

    def check_deepcopied_world(self, world2):
        """Check that deepcopied world is isomoprhic to old world, and that it behaves
        correctly."""
        # check that new world is isomorphic
        check_deepcopy(self.world, world2)

        # make sure that object attribute's view is still linked to the master
        # array by using object attribute's view to set position
        position = world2.object_attributes['position']
        position.set(0, (1, 1))

        # creating an object resets all the views to be linked to the master
        # array
        world2.create_object('basic')

        # check that position setting had the intended effect
        self.assertTrue(np.array_equal(position.get(0), (1, 1)))

    def test_deepcopy(self):
        """Make sure that we can deepcopy a world and get something isomorphic that
        behaves correctly"""
        world2 = deepcopy(self.world)

        # run checks
        self.check_deepcopied_world(world2)

    def test_deepcopy_object_attribute(self):
        """Make sure that we can deepcopy a world via an object attribute and get
        something isomorphic that behaves correctly"""
        position = self.world.object_attributes['position']
        position2 = deepcopy(position)
        world2 = position2.world

        # run checks
        self.check_deepcopied_world(world2)

    def test_deepcopy_object(self):
        """Make sure that we can deepcopy a world via an object and get something
        isomorphic that behaves correctly"""
        object = self.world.objects[0]
        object2 = deepcopy(object)
        world2 = object2.world

        # run checks
        self.check_deepcopied_world(world2)

    def test_deepcopy_judge(self):
        """Make sure that we can deepcopy a world via the judge and get something
        isomorphic that behaves correctly"""
        judge2 = deepcopy(self.world.judge)
        world2 = judge2.world

        # run checks
        self.check_deepcopied_world(world2)

    def test_deepcopy_event(self):
        """Make sure that we can deepcopy a world via an event and get something
        isomorphic that behaves correctly"""

        # use RIGHT action to generate a push event so that we have an event to
        # copy
        self.world.step('RIGHT')

        event2 = deepcopy(self.world.events[0])
        world2 = event2.world

        # run checks
        self.check_deepcopied_world(world2)

    def test_deepcopy_goal(self):
        """Make sure that we can deepcopy a world via a goal and get something
        isomorphic that behaves correctly"""
        self.world.goals = ['anything']

        goal2 = deepcopy(self.world.goals['anything'])
        world2 = goal2.world

        # run checks
        self.check_deepcopied_world(world2)

    def test_deepcopy_variant(self):
        """Make sure that we can deepcopy a world via a variant and get something
        isomorphic that behaves correctly"""
        self.world.variants = ['show_colors']

        variant2 = deepcopy(self.world.variants['show_colors'])
        world2 = variant2.world

        # run checks
        self.check_deepcopied_world(world2)

    def test_deepcopy_randomizer(self):
        """Make sure that we can deepcopy a world via the randomizer and get something
        isomorphic that behaves correctly"""
        randomizer2 = deepcopy(self.world.randomizer)
        world2 = randomizer2.world

        # run checks
        self.check_deepcopied_world(world2)

    def test_deepcopy_agent(self):
        """Make sure that we can deepcopy a world via the agent and get something
        isomorphic that behaves correctly"""
        agent2 = deepcopy(self.world.agent)
        world2 = agent2.world

        # run checks
        self.check_deepcopied_world(world2)

    def test_deepcopy_world_attribute(self):
        """Make sure that we can deepcopy a world via a world attribute and get
        something isomorphic that behaves correctly"""
        attr2 = deepcopy(self.world.world_attributes['height'])
        world2 = attr2.world

        # run checks
        self.check_deepcopied_world(world2)

    def test_check_deepcopy(self):
        """Check that check_deepcopy is correct in its judgments"""
        d = dict(a=3, b=4, c=5)
        check_deepcopy(d, deepcopy(d))
        a = [d, d, d]
        check_deepcopy(a, deepcopy(a))
        
        # check that non-isomorphic structure-sharing doesn't work
        a2 = [d, deepcopy(d), d]
        with self.assertRaises(AssertionError):
            check_deepcopy(a, a2)
        with self.assertRaises(AssertionError):
            check_deepcopy(a, deepcopy(a2))

        # check that tuples getting deepcopied works
        t = (3, 4, 5)
        t2 = deepcopy(t)
        check_deepcopy(t, t2)
        self.assertIs(t, t2)

        # check that arrays being deepcopied works
        arr = np.array([3, 4, 5])
        check_deepcopy(arr, deepcopy(arr))

        class Foo:
            def __init__(self, x=3, y=4, z=5):
                self.x = x
                self.y = y
                self.z = z

        # check that objects being deepcopied works
        f = Foo()
        check_deepcopy(f, deepcopy(f))
        # add some recursion
        f.z = f.x = f
        check_deepcopy(f, deepcopy(f))

        # check that we can build an isomorphic structure by hand
        g = Foo()
        g.z = g.x = g
        check_deepcopy(f, g)

        # check that a non-isomorphic structure fails
        g2 = Foo()
        g3 = Foo()
        g2.z = g2.x = g3
        g3.z = g3.x = g2
        with self.assertRaises(AssertionError):
            check_deepcopy(f, g2)


class TestState(TestCase):
    world = None
    
    def setUp(self):
        self.world = core.PixelWorld(objects=[
            ['self', {'position': (0, 0), 'color': 2}],
            ['basic', {'position': (0, 1), 'color': 1}],
            ['basic', {'position': (0, -1), 'color': 1}]
            ], state_center='self')
    
    def tearDown(self):
        self.world.end()
    
    def test_observation(self):
        """make sure the observation type is correct"""
        obs = self.world.observe()
        
        self.assertTrue(isinstance(obs, np.ndarray))
        self.assertTrue(obs.shape == self.world.shape)
    
    def test_self_is_centered(self):
        """make sure the self shows up at the center of the world"""
        obs = self.world.observe()
        
        self_pos = tuple(np.round(np.array(self.world.shape)/2))
        
        self.assertEqual(obs[self_pos], 2)
    
    def test_both_objects_visible(self):
        """make sure both objects show up in the self-centered view"""
        obs = self.world.observe()
        
        self.assertEqual(len(np.where(obs == 1)[0]), 2)
    
    def test_self_moves(self):
        """make sure the self stays in the center after moving"""
        self.world.step('LEFT')
        
        obs = self.world.observe()
        
        self_pos = tuple(np.round(np.array(self.world.shape)/2))
        
        self.assertEqual(obs[self_pos], 2)
    
    def test_observed_window(self):
        """make sure observed_window is respected"""
        self.world._observed_window = (3, 3)
        
        obs = self.world.observe()
        
        self.assertEqual(obs.shape, (3, 3))
    
    def test_explicit_state_center(self):
        """make sure we can explicitly set the center of the state"""
        self_obj = self.world.objects.get(name='self')
        self.world.state_center = self_obj.position
        self.world._observed_window = (3, 3)
        
        obs = self.world.observe()
        
        self.assertEqual(obs[1,1], self_obj.color)
        
        obs, _, _, _ = self.world.step('LEFT')
        
        self.assertEqual(obs[1, 0], self_obj.color)


class TestGymStuff(TestCase):
    world = None
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestGymStuff, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def test_state_space(self):
        """Check that ObjectStateSpace supports the expected interface"""
        space = core.ObjectStateSpace()

        # check that space.contains() only accepts lists of dicts
        self.assertTrue(not space.contains(3))
        self.assertTrue(space.contains([dict(x=3), dict(f=4)]))

        # check that sampling is not implemented
        with self.assertRaises(NotImplementedError):
            space.sample()

    def test_list_space(self):
        """Check that ListSpace support the expected interface"""
        # construct a list space
        ls = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'NOOP']
        space = core.ListSpace(self.world, ls)
        
        # construct a second list space with different actions
        ls2 = ['LEFT', 'RIGHT', 'NOOP']
        space2 = core.ListSpace(self.world, ls2)

        # check that when we sample from a list space we get something that is
        # contained in the original list
        for i in xrange(10):
            self.assertTrue(space.sample() in ls)

        # check that every element of the original list is contained in the
        # list space, and that elements are contained in the second list space
        # if and only if they were contained in the second list
        for x in ls:
            self.assertTrue(space.contains(x))
            self.assertTrue(space2.contains(x) == (x in ls2))

        # check that the spaces are not equal
        self.assertTrue(space != space2)
        self.assertFalse(space == space2)

        # make another list space that is the same as the first list space
        ls3 = ls[:]
        space3 = core.ListSpace(self.world, ls3)
        
        # check that the new list space is equal to the first list space
        self.assertTrue(space == space3)
        self.assertFalse(space != space3)

        # check that repr() of the list space is what we expect
        self.assertEqual(repr(space3), "List(['LEFT', 'RIGHT', 'UP', 'DOWN', 'NOOP'])")

    def test_multi_list_space(self):
        """Check that MultiListSpace supports the expected interface"""
        # make a bunch of list spaces with different action lists and different
        # numbers of agents
        ls = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'NOOP']
        ls2 = ['LEFT', 'RIGHT', 'NOOP']
        space = core.MultiListSpace(self.world, ls, 3)
        space2 = core.MultiListSpace(self.world, ls, 2)
        space3 = core.MultiListSpace(self.world, ls2, 3)
        space4 = core.MultiListSpace(self.world, ls[:], 3)

        # check that the sample has the right number of elements and that all
        # of them are strings
        sample = space.sample()
        self.assertEqual(len(sample), 3)
        for s in sample:
            self.assertTrue(isinstance(s, str))

        # check that repr() of the multi-list space is what we expect
        self.assertEqual(repr(space), "MultiList(['LEFT', 'RIGHT', 'UP', 'DOWN', 'NOOP'], 3)")

        # check that spaces are equal exactly when they have the same actions
        # and the same number of agents
        self.assertTrue(space != space2)
        self.assertFalse(space == space2)
        self.assertTrue(space != space3)
        self.assertFalse(space == space3)
        self.assertTrue(space == space4)
        self.assertFalse(space != space4)


class TestObject(TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestObject, self).__init__(*args, **kwargs)
    
    def setUp(self):
        class FooObjectAttribute(core.ObjectAttribute):
            _default_value = 1
            _null_value = 0
    
        class BarObjectAttribute(core.ObjectAttribute):
            _default_value = 1
            _null_value = 0
            _initialize_before = ['foo']

        class BazObjectAttribute(core.ObjectAttribute):
            _default_value = 1
            _null_value = 0

        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def test_removal(self):
        """Check that we can remove an object from the world"""
        n = self.world._num_objects
        x = self.world.objects[0]
        x.remove()
        self.assertTrue(self.world._num_objects == n-1)

    def test_object2(self):
        """Check that we can add, remove, and access attributes from objects"""
        # check that unique names are unique
        names = [x.unique_name for x in self.world.objects]
        names.sort()
        for n1, n2 in zip(names[:-1], names[1:]):
            self.assertTrue(n1 != n2)

        # check that x._head_of_family works correctly
        x = self.world.objects[3]
        self.assertTrue(x._head_of_family._id == x._id)
        for yid in x.children:
            self.assertTrue(self.world.objects[yid]._head_of_family._id == x._id)

        # check that we can add attributes
        for y in self.world.objects:
            y.add_attribute('foo')

        # check that we can access attributes in two different ways
        for y in self.world.objects:
            self.assertTrue(y.state['foo'] == y.foo)

        # check that we can update attribute values
        for y in self.world.objects:
            y.update(foo=y.id)
            self.assertTrue(y.foo == y.id)

        # check that we can remove attributes
        x.remove_attribute('foo')
        self.assertTrue(not hasattr(x, 'foo'))

        # check that we can remove a non-existent attribute
        for y in self.world.objects:
            y.remove_attribute('flux')

        # check that we can remove all the objects from the world
        for y in self.world.objects:
            y.remove()

    def test_object(self):
        """Check that we can access attribute as we expect through private methods _get
        and _set"""
        class MyObject(core.Object):
            _attributes = ['foo', 'bar']

        # create an object
        x = self.world.create_object(MyObject)

        # add attribute to existing object
        y = self.world.objects[3]
        y.add_attribute('foo')

        self.assertTrue(x._get('foo') == 1)
        self.assertTrue(y._get('foo') == 1)

        y._set('foo', 17)
        self.assertTrue(y.foo == 17)

    def test_remove_parent(self):
        """Check that the _remove_parent() method works as we expect"""
        # pick an object with children
        y = self.world.objects[3]

        # remove y's children
        for zid in y.children:
            self.world.objects[zid]._remove_parent()

        # check that y no longer has children
        self.assertTrue(y.children.size == 0)

    def test_initialize(self):
        """Check that the _initialize() method works as we expect"""
        class MyObject(core.Object):
            _attributes = ['foo', 'bar']

        # create an object
        x = self.world.create_object(MyObject)
        
        # initialize
        x._initialize(dict(foo=3, bar=6, baz=10), False)

        # check that values were set
        self.assertTrue(x.foo == 3)
        self.assertTrue(x.bar == 6)
        self.assertTrue(x.baz == 10)

    def test_copy_data_array(self):
        """Check that _add_data() and _remove_data() work as we expect"""
        class MyObject(core.Object):
            _attributes = ['foo', 'bar']

        # create an object
        x = self.world.create_object([MyObject, dict(foo=3, bar=6, baz=10)])

        # remember what the data array looks like
        data = deepcopy(self.world._data.data)

        # add a new row containing (some of) our data
        x._add_data()

        # remove the new row
        x._remove_data()

        # check that data array looks the same
        self.assertTrue(np.array_equal(data['foo'], self.world._data.data['foo']))
        self.assertTrue(np.array_equal(data['bar'], self.world._data.data['bar']))
        self.assertTrue(np.array_equal(data['baz'], self.world._data.data['baz']))

    def test_state(self):
        """Check that the object.state attribute works as we expect"""
        class MyObject(core.Object):
            _attributes = ['foo', 'bar']

        # create an object
        x = self.world.create_object(MyObject)

        # set observed attributes
        self.world._observed_attributes = ['foo']

        # check that state is what we expect
        self.assertEqual(x.state, {'foo':1})

    def test_is_related(self):
        """Test that is_related works correctly"""
        # pick an object with children
        x = self.world.objects[3]
        for y in x._children:
            self.assertTrue(x.is_related(y))
            self.assertTrue(y.is_related(x))

        # pick an object unrelated to x
        z = self.world.objects[0]
        for y in x._children:
            self.assertFalse(z.is_related(y))
            self.assertFalse(y.is_related(z))

    def test_has_attribute_class(self):
        """Test that has_attribute_class works correctly"""
        x = self.world.objects[3]

        # any attribute will trigger this, since all derive from
        # ObjectAttribute
        self.assertTrue(x.has_attribute_class(core.ObjectAttribute))

        # x has no foo attribute or foo-derived attribute
        self.assertFalse(x.has_attribute_class('foo'))

    def test_get_init_value(self):
        """Test that _get_init_value() works as expected"""
        # get existing object
        x = self.world.objects[3]

        # get attribute
        color = self.world.object_attributes['color']

        # check that _get_init_value() returns correct things and modifies d
        # correctly
        d = {'color': 3}
        self.assertEqual(x._get_init_value(d, color, pop=False), 3)
        self.assertEqual(d, {'color': 3})
        self.assertEqual(x._get_init_value(d, color, pop=True), 3)
        self.assertEqual(d, {})

        class MyObject(core.Object):
            _attributes = ['foo', 'bar']
            _defaults = {'foo': 3}

        # create a new object
        y = self.world.create_object(MyObject)

        # check that init value is what we expect
        foo = self.world.object_attributes['foo']
        bar = self.world.object_attributes['bar']
        self.assertEqual(y._get_init_value(d, foo), 3)
        self.assertEqual(y._get_init_value(d, bar), 1) # default value of bar


class TestObjectCollection(TestCase):
    world = None
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestObjectCollection, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def test(self):
        """Check that find/get/call methods work"""
        # check that objects(name) returns objects we expect to see
        for i in xrange(len(self.world.objects)):
            name = self.world.objects[i].name
            stuff = self.world.objects(name)
            if isinstance(stuff, core.ObjectCollection):
                self.assertTrue(self.world.objects[i] in stuff)
            else:
                self.assertTrue(self.world.objects[i] == stuff)

        # check that objects can accept keyword arguments
        x = self.world.objects[12]
        pos = x.position
        stuff = self.world.objects(position=pos)
        self.assertTrue(stuff == x)

        # check that we can get x by name
        self.assertTrue(x == self.world.objects.get(name=x.name))
        self.assertTrue(x == self.world.objects[name])

        # check that we can find children of an object using parent attribute
        x = self.world.objects[3]
        stuff = self.world.objects.find(parent=x.id)
        ids = [y.id for y in stuff]
        self.assertTrue(sorted(ids) == sorted(x.children))


class TestCompoundObject(TestCase):
    world = None
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestCompoundObject, self).__init__(*args, **kwargs)
    
    def setUp(self):
        class FooObjectAttribute(core.ObjectAttribute):
            _default_value = 1
            _null_value = 0

        class BarObjectAttribute(core.CoupledFamilyObjectAttribute):
            _default_value = 1
            _null_value = 0
            _initialize_before = ['foo']

        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def test_compound_object(self):
        """Check that we can create compound objects and use the add_child/remove_child
        functions as expected, and also that coupled family attributes work as
        expected."""
        class MyObject(core.Object):
            _attributes = ['foo', 'bar']
        
        class MyCompoundObject(MyObject, core.CompoundObject):
            pass

        # make a compound object with 10 children
        ys = ['my'] * 10
        x = MyCompoundObject(self.world, name='x', children=ys)
        ys = [self.world.objects[id] for id in x.children]
        
        # make sure that len(object) returns correct value
        self.assertTrue(len(x) == 10)
        
        # check that children have attributes we expect to see in child_states
        for s in x.child_states:
            self.assertTrue('foo' in s)
            self.assertTrue('bar' in s)

        # check that add_child and remove_child work as expected
        z = self.world.create_object(MyObject)
        x.add_child(z)
        self.assertTrue(z._parent is x)
        self.assertTrue(len(x) == 11)
        x.remove_child(z)
        self.assertTrue(z._parent is None)

        # check that len(x) is still 10
        self.assertTrue(len(x) == 10)

        # check coupling of x
        bar = self.world.object_attributes['bar']
        self.assertTrue(isinstance(bar, core.CoupledFamilyObjectAttribute))
        x.bar = 6
        self.assertTrue(x.bar == 6)

        # check that change propagated down to children
        for y in ys:
            self.assertTrue(y.bar == 6)

        # check that removing x from the world also removes it as a parent
        x.remove()
        for y in ys:
            self.assertTrue(y._parent is None, y)

    def test_family_caches(self):
        """Check that family caches in objects are populated and depopulated when we
        expect them to be. """
        self.world = universe.create_world('basic', depth=4)
        self.world.remove_objects(self.world.objects)

        # first we create a 5-level hierarchy
        x = self.world.create_object('complex')
        bottom_level = [x]
        everyone = [x]
        for lvl in xrange(4):
            new_bottom_level = []
            child_type = 'complex' if lvl < 3 else 'basic'
            for a in bottom_level:
                b = self.world.create_object(child_type)
                c = self.world.create_object(child_type)
                a.add_child(b)
                a.add_child(c)
                new_bottom_level.extend([b, c])
            # check that old nodes' descendants cache was cleared by
            # add_child. (leaf descendants and simple leaf descendants are
            # repopulated by position conflict checking)
            for a in bottom_level:
                self.assertEqual(a._descendants_cached, None)
                self.assertEqual(a._descendants_ids_cached, None)
            # check that new nodes' ancestors cache was cleared by
            # add_child. (head_of_family is repopulated by position conflict
            # checking)
            for a in new_bottom_level:
                self.assertEqual(a._ancestors_cached, None)
                self.assertEqual(a._ancestors_ids_cached, None)
            bottom_level = new_bottom_level
            everyone.extend(bottom_level)

        z = bottom_level[0]

        # repopulate all the caches and check that they are populated
        ls = []
        for a in everyone:
            ls.extend([a._descendants, a._descendants_ids, a._leaf_descendants, a._leaf_descendants_ids])
            ls.extend([a._simple_leaf_descendants, a._simple_leaf_descendants_ids])
            ls.extend([a._ancestors, a._ancestors_ids, a._head_of_family, a._head_of_family_id])

            self.assertFalse(a._descendants_cached is None)
            self.assertFalse(a._descendants_ids_cached is None)
            self.assertFalse(a._leaf_descendants_cached is None)
            self.assertFalse(a._leaf_descendants_ids_cached is None)
            self.assertFalse(a._simple_leaf_descendants_cached is None)
            self.assertFalse(a._simple_leaf_descendants_ids_cached is None)
            self.assertFalse(a._ancestors_cached is None)
            self.assertFalse(a._ancestors_ids_cached is None)
            self.assertFalse(a._head_of_family_cached is None)
            self.assertFalse(a._head_of_family_id_cached is None)

        # check that we can clear the ancestor caches of all of x's descendants
        x._clear_descendants_ancestors_caches()
        for a in bottom_level:
            self.assertEqual(a._ancestors_cached, None)
            self.assertEqual(a._ancestors_ids_cached, None)
            self.assertEqual(a._head_of_family_cached, None)
            self.assertEqual(a._head_of_family_id_cached, None)

        # repopulate all the caches and check that they are populated
        ls = []
        for a in everyone:
            ls.extend([a._descendants, a._descendants_ids, a._leaf_descendants, a._leaf_descendants_ids])
            ls.extend([a._simple_leaf_descendants, a._simple_leaf_descendants_ids])
            ls.extend([a._ancestors, a._ancestors_ids, a._head_of_family, a._head_of_family_id])

            self.assertFalse(a._descendants_cached is None)
            self.assertFalse(a._descendants_ids_cached is None)
            self.assertFalse(a._leaf_descendants_cached is None)
            self.assertFalse(a._leaf_descendants_ids_cached is None)
            self.assertFalse(a._simple_leaf_descendants_cached is None)
            self.assertFalse(a._simple_leaf_descendants_ids_cached is None)
            self.assertFalse(a._ancestors_cached is None)
            self.assertFalse(a._ancestors_ids_cached is None)
            self.assertFalse(a._head_of_family_cached is None)
            self.assertFalse(a._head_of_family_id_cached is None)

        # check that we can clear the descendant caches of all z's ancestors
        z._clear_ancestors_descendants_caches()
        a = z
        while a is not None:
            self.assertEqual(a._descendants_cached, None)
            self.assertEqual(a._descendants_ids_cached, None)
            self.assertEqual(a._leaf_descendants_cached, None)
            self.assertEqual(a._leaf_descendants_ids_cached, None)
            self.assertEqual(a._simple_leaf_descendants_cached, None)
            self.assertEqual(a._simple_leaf_descendants_ids_cached, None)
            a = a._parent

        # repopulate all the caches and check that they are populated
        ls = []
        for a in everyone:
            ls.extend([a._descendants, a._descendants_ids, a._leaf_descendants, a._leaf_descendants_ids])
            ls.extend([a._simple_leaf_descendants, a._simple_leaf_descendants_ids])
            ls.extend([a._ancestors, a._ancestors_ids, a._head_of_family, a._head_of_family_id])

            self.assertFalse(a._descendants_cached is None)
            self.assertFalse(a._descendants_ids_cached is None)
            self.assertFalse(a._leaf_descendants_cached is None)
            self.assertFalse(a._leaf_descendants_ids_cached is None)
            self.assertFalse(a._simple_leaf_descendants_cached is None)
            self.assertFalse(a._simple_leaf_descendants_ids_cached is None)
            self.assertFalse(a._ancestors_cached is None)
            self.assertFalse(a._ancestors_ids_cached is None)
            self.assertFalse(a._head_of_family_cached is None)
            self.assertFalse(a._head_of_family_id_cached is None)

        # save and restore from snapshot
        self.world.save_snapshot('foo')
        self.world.load_snapshot('foo')

        # check that caches were cleared
        for a in everyone:
            self.assertTrue(a._descendants_cached is None)
            self.assertTrue(a._descendants_ids_cached is None)
            self.assertTrue(a._leaf_descendants_cached is None)
            self.assertTrue(a._leaf_descendants_ids_cached is None)
            self.assertTrue(a._simple_leaf_descendants_cached is None)
            self.assertTrue(a._simple_leaf_descendants_ids_cached is None)
            self.assertTrue(a._ancestors_cached is None)
            self.assertTrue(a._ancestors_ids_cached is None)
            self.assertTrue(a._head_of_family_cached is None)
            self.assertTrue(a._head_of_family_id_cached is None)

    def test_construct_child_params(self):
        """Check that we can pass child_type into compound object constructor and have
        the right thing happen."""
        class MyCompoundObject(core.CompoundObject):
            pass

        x = MyCompoundObject(self.world, child_type=['my_compound', 'my_compound'])
        self.assertEqual(len(x.children), 2)
        for y in x._children:
            self.assertTrue(isinstance(y, MyCompoundObject))



class TestEntity(TestCase):
    world = None
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestEntity, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def test_get_class(self):
        """Check that get_class works as expected."""
        class FooEntity(core.Entity):
            _name = 'foo'
        
            def __eq__(self, other):
                return other.world == self.world
        
        # check various ways to get FooEntity class
        self.assertTrue(core.Entity.get_class('foo') == FooEntity)
        self.assertTrue(core.Entity.get_class(FooEntity) == FooEntity)
        self.assertTrue(core.Entity.get_class(FooEntity(self.world)) is FooEntity)
        self.assertTrue(core.Entity.get_class(['foo']) is FooEntity)

        # check that we can't pass in nonsense
        with self.assertRaises(TypeError):
            core.Entity.get_class(3)

    def test_get_instance(self):
        """Check that get_instance works as expected."""
        class FooEntity(core.Entity):
            _name = 'foo'
        
            def __eq__(self, other):
                return other.world == self.world

        # check various ways to get a FooEntity instance
        self.assertTrue(core.Entity.get_instance(self.world, 'foo') == FooEntity(self.world))
        self.assertTrue(core.Entity.get_instance(self.world, FooEntity) == FooEntity(self.world))
        self.assertTrue(core.Entity.get_instance(self.world, FooEntity(self.world)) == FooEntity(self.world))
        self.assertTrue(core.Entity.get_instance(self.world, ['foo']) == FooEntity(self.world))

        # if we pass an instance and a dict, the instance gets the dict's
        # key-value pairs assigned as attributes
        inst = core.Entity.get_instance(self.world, [FooEntity(self.world), {'fleek': 'on'}])
        self.assertEqual(inst.fleek, 'on')

        # check that we can't pass in nonsense
        with self.assertRaises(TypeError):
            core.Entity.get_instance(self.world, 3)

        # check that we can't pass in a list of the wrong length
        with self.assertRaises(TypeError):
            core.Entity.get_instance(self.world, ['foo', 'bar', 'baz'])

    def test_entity(self):
        """Check that entity properties work as expected."""
        class FooEntity(core.Entity):
            _name = 'foo'
        
            def __eq__(self, other):
                return other.world == self.world
        
        # create x
        x = core.Entity.get_instance(self.world, 'foo')

        # check that x has the properties we expect
        self.assertTrue(x.name == 'foo')
        self.assertTrue(x.world == self.world)
        self.assertTrue(x.rng == self.world._rng)

        # check that we can remove x
        x.remove()
        self.assertTrue(x.world == None)


class TestObjectAttribute(TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestObjectAttribute, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'


class TestObjectAttributeGeneric(TestObjectAttribute):
    def test_object_attribute_attributes(self):
        """Check that object attributes have the attributes we expect"""
        class FooObjectAttribute(core.ObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        self.assertTrue(hasattr(FooObjectAttribute, '_order_types'))
        self.assertTrue(hasattr(FooObjectAttribute, '_data_fields'))
        self.assertTrue(hasattr(FooObjectAttribute, '_depends_on'))
        self.assertTrue(hasattr(FooObjectAttribute, '_initialize_before'))
        self.assertTrue(hasattr(FooObjectAttribute, '_initialize_after'))
        self.assertTrue(hasattr(FooObjectAttribute, '_coerce_value'))

    def test_object_attribute2(self):
        """Check that initialize ordering is respected by object's attribute dict"""
        class FooObjectAttribute(core.ObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        class BarObjectAttribute(core.ObjectAttribute):
            _name = 'bar'
            _default_value = 1
            _null_value = 0
            _initialize_before = ['foo']

        class BazObjectAttribute(core.ObjectAttribute):
            _name = 'baz'
            _ndim = 2
            _default_value = 1
            _null_value = 0

        class MyObject(core.Object):
            _attributes = ['foo', 'bar', 'baz']

        x = self.world.objects[3]
        x.add_attribute('foo')
        x.add_attribute('bar')
        x.add_attribute('baz')

        attrs = MyObject._attributes
        self.assertTrue(attrs.index('bar') < attrs.index('foo'))

    def test_object_attribute_access_compare(self):
        """Check that we can access and compare object attributes as expected."""
        class FooObjectAttribute(core.ObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        class BarObjectAttribute(core.ObjectAttribute):
            _name = 'bar'
            _default_value = 1
            _null_value = 0
            _initialize_before = ['foo']

        class BazObjectAttribute(core.ObjectAttribute):
            _name = 'baz'
            _ndim = 2
            _default_value = 1
            _null_value = 0

        class MyObject(core.Object):
            _attributes = ['foo', 'bar', 'baz']

        # add attributes to an existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        x.add_attribute('bar')
        x.add_attribute('baz')

        # create an object
        y = self.world.create_object(MyObject)

        # get attributes
        foo = self.world.object_attributes['foo']
        bar = self.world.object_attributes['bar']
        baz = self.world.object_attributes['baz']

        # check that we can get, set, and compare foo
        y.foo = 3
        self.assertTrue(np.array_equal(foo.get([x, y]), [1, 3]))
        foo.set([x, y], [10, 12])
        self.assertTrue(np.array_equal(foo.get([x, y]), [10, 12]))
        self.assertFalse(foo.compare(x, y))
        y.foo = 10
        self.assertTrue(foo.compare(x, y))

        # check taht we can compare baz
        self.assertTrue(baz.compare(np.ones(2), np.ones((2,2)), validate=False).all())

        # check that we can compare foo
        self.assertTrue(foo.compare(x.foo, y.foo, validate=True))

    def test_object_attribute_conversion(self):
        """Check that we can use conversion methods of object attributes."""
        class FooObjectAttribute(core.ObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        class BarObjectAttribute(core.ObjectAttribute):
            _name = 'bar'
            _default_value = 1
            _null_value = 0
            _initialize_before = ['foo']

        class BazObjectAttribute(core.ObjectAttribute):
            _name = 'baz'
            _ndim = 2
            _default_value = 1
            _null_value = 0

        class MyObject(core.Object):
            _attributes = ['foo', 'bar', 'baz']

        # add attributes to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        x.add_attribute('bar')
        x.add_attribute('baz')

        # create new object
        y = self.world.create_object(MyObject)

        # get attributes
        foo = self.world.object_attributes['foo']
        bar = self.world.object_attributes['bar']
        baz = self.world.object_attributes['baz']

        # set value of foo
        foo.set([x, y], [10, 12])
        y.foo = 10

        # check that _to_numpy() raises an error on string input
        with self.assertRaises(TypeError):
            print foo._to_numpy('EXCELSIOR')

        # check that _to_vector() works
        self.assertTrue(np.array_equal(baz._to_vector(np.ones((1, 2))), [1, 1]))

        # check that _to_array raises an error when input has wrong length
        with self.assertRaises(ValueError):
            print baz._to_array(np.ones(3))

        # check that _to_array raises an error when input length is not
        # compatible with num_values
        with self.assertRaises(ValueError):
            print foo._to_array(np.ones(3), num_values=4)

        # check that calling an attribute results in an array of values
        self.assertTrue(np.array_equal(foo(), [10, 10]))

        # check that _to_numpy returns a numpy array
        self.assertTrue(type(foo._to_numpy([10, 10])) == np.ndarray)

        # check that _to_vector raises an error when input length is not the
        # same as the dimensionality of the attribute
        with self.assertRaises(ValueError):
            print foo._to_vector([[10, 10], [11, 12]])

    def test_remove_data(self):
        """Check that we can remove object attributes data and restore it."""
        class FooObjectAttribute(core.ObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        class BarObjectAttribute(core.ObjectAttribute):
            _name = 'bar'
            _default_value = 1
            _null_value = 0
            _initialize_before = ['foo']

        class BazObjectAttribute(core.ObjectAttribute):
            _name = 'baz'
            _ndim = 2
            _default_value = 1
            _null_value = 0

        class MyObject(core.Object):
            _attributes = ['foo', 'bar', 'baz']
            
        # add attributes to an existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        x.add_attribute('bar')
        x.add_attribute('baz')
        
        # create a new object
        y = self.world.create_object(MyObject)

        # get object attributes
        foo = self.world.object_attributes['foo']
        bar = self.world.object_attributes['bar']
        baz = self.world.object_attributes['baz']

        # set values to something non-null
        foo.set([x, y], [10, 12])
        y.foo = 10

        # check that removing and adding data results in everything having the
        # null value
        foo._remove_data()
        foo._add_data()
        self.assertTrue(np.array_equal(foo.get([x, y]), [0, 0]))

    def test_removal(self):
        """Check that we can remove object attributes from the world"""
        class FooObjectAttribute(core.SteppingObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        class BarObjectAttribute(core.RandomizingObjectAttribute):
            _name = 'bar'
            _default_value = 1
            _null_value = 0
            _initialize_before = ['foo']

        class MyObject(core.Object):
            _attributes = ['foo', 'bar']

        # create an object that has foo and bar attributes
        x = self.world.create_object('my')
        
        # get attributes
        foo, bar = [self.world.object_attributes[k] for k in ['foo', 'bar']]

        # remove attributes
        foo.remove()
        bar.remove()
        
        # check that attributes were removed from world lists of
        # stepping/randomizing attributes
        self.assertTrue('foo' not in self.world._stepping_object_attributes)
        self.assertTrue('bar' not in self.world._randomizing_object_attributes)

        # check that foo no longer has an objects attribute
        with self.assertRaises(AttributeError):
            print foo.objects

        # check that foo is no longer in world.object_attributes
        with self.assertRaises(KeyError):
            print self.world.object_attributes['foo']

    def test_object_attribute(self):
        """Check that we can access object attributes' data in a variety of ways."""
        class FooObjectAttribute(core.ChangeTrackingObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        class BarObjectAttribute(core.ChangeTrackingObjectAttribute):
            _name = 'bar'
            _default_value = 1
            _null_value = 0
            _initialize_before = ['foo']

        class MyObject(core.Object):
            _attributes = ['foo', 'bar', 'parent', 'children']

        # add attributes to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        x.add_attribute('bar')

        # create a new object
        y = core.CompoundObject(self.world, name='y', children=[])

        # get object attributes
        foo = self.world.object_attributes['foo']
        bar = self.world.object_attributes['bar']

        # test _get_index()
        self.assertTrue(foo._get_index(x) == x.id)
        self.assertTrue(np.array_equal(foo._get_index([x, y]), [x.id, y.id]))

        # test get()
        self.assertTrue(np.array_equal(foo.get([x, y]), [1, 0]), repr(foo.get([x, y])))

        # test _get_data()
        self.assertTrue(np.array_equal(foo._get_data([x.id, y.id]), [1, 0]), repr(foo._get_data([x.id, y.id])))

        # test _set_data()
        foo._set_data([x.id, y.id], [17, 23])
        self.assertTrue(np.array_equal(foo._get_data([x.id, y.id]), [17, 23]), repr(foo._get_data([x.id, y.id])))

        # test _sum() and _mean()
        self.assertTrue(foo._sum(np.array([12, 18, 20])) == 50)
        self.assertTrue(foo._mean(np.array([12, 18, 21])) == 17)
        self.assertTrue(foo._sum(np.array([])) == 0)
        self.assertTrue(foo._mean(np.array([])) == 0)

        # test _get_value(), _set_value(), and setting through attribute
        x.foo = 3
        self.assertTrue(foo._get_value(x.id) == 3)
        foo._set_value(x.id, 7)
        self.assertTrue(foo._get_value(x.id) == 7)
        self.assertTrue(x.foo == 7)
        
        # test _get_child_indices() and _get_parent_index()
        z = self.world.create_object(MyObject)
        y.add_child(z)
        self.assertTrue(foo._get_child_indices(y.id) == [z.id])
        self.assertTrue(foo._get_parent_index(z.id) == y.id)

        # test _get_objects() and _get_object_indices()
        self.assertTrue(sorted(foo._get_objects()) == sorted([x, y, z]))
        self.assertTrue(np.array_equal(foo._get_object_indices(), [x.id, y.id, z.id]))
        self.assertTrue(np.array_equal(foo._get_object_indices(core.Object.SIMPLE), [z.id]))

        # test _get_mask()
        mask = np.array([1 for obj in self.world.objects], dtype=bool)
        mask[x.id] = 0
        mask[y.id] = 0
        mask[z.id] = 0
        self.assertTrue(np.array_equal(foo._get_mask(), mask), repr(foo._get_mask()))

        # test get_previous()
        x.foo = 1
        self.assertTrue(x.foo == 1)
        x.foo = 3
        self.assertTrue(x.foo == 3)
        self.assertTrue(foo.get_previous(x) == 1, foo.get_previous(x))
        
        # test get_previous()
        x.foo = 6
        x.foo = 7
        self.assertTrue(x.foo == 7)
        self.assertTrue(foo.get_previous(x) == 6)

    def test_coerce_value(self):
        """Check that value coercion works as expected"""
        class FooObjectAttribute(core.ObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

            def _coerce_value(self, value, **kwargs):
                return 2 * value, False

        class BarObjectAttribute(FooObjectAttribute):
            _name = 'bar'
            _default_value = 1
            _null_value = 0

            def _coerce_value(self, value, **kwargs):
                return value % 2, False

        class BazObjectAttribute(core.ObjectAttribute):
            _name = 'baz'
            _default_value = 1
            _null_value = 0

            def _coerce_value(self, value, **kwargs):
                if value == 3:
                    return None, AssertionError()
                else:
                    return value, False

        # add attributes to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        x.add_attribute('bar')
        x.add_attribute('baz')
        
        # coercion goes 3 -> 6 -> 0
        x.bar = 3
        self.assertTrue(x.bar == 0, repr(x.bar))

        # check that baz complains about value 3
        baz = self.world.object_attributes['baz']
        self.assertEqual(baz.coerce_value(3, error=False), None)
        with self.assertRaises(AssertionError):
            baz.coerce_value(3)

    def test_hierarchy(self):
        """Check that various coupled family attributes behave as expected"""
        self.world.remove_objects(self.world.objects)

        # first we create a 3-level hierarchy:
        #            x
        #          /   \
        #         y     z
        #        / \   / \
        #       a   b c   d
        x = self.world.create_object('complex')
        y = self.world.create_object('complex')
        z = self.world.create_object('complex')
        a = self.world.create_object('basic')
        b = self.world.create_object('basic')
        c = self.world.create_object('basic')
        d = self.world.create_object('basic')
        x.add_child(y)
        x.add_child(z)
        y.add_child(a)
        y.add_child(b)
        z.add_child(c)
        z.add_child(d)

        # check that everybody has the right leaf descendants
        self.assertItemsEqual(x.leaf_descendants, [a.id, b.id, c.id, d.id])
        self.assertItemsEqual(y.leaf_descendants, [a.id, b.id])
        self.assertItemsEqual(z.leaf_descendants, [c.id, d.id])
        self.assertItemsEqual(a.leaf_descendants, [a.id])
        self.assertItemsEqual(b.leaf_descendants, [b.id])
        self.assertItemsEqual(c.leaf_descendants, [c.id])
        self.assertItemsEqual(d.leaf_descendants, [d.id])

        # check that everybody has the right parents
        self.assertTrue(y._parent == x)
        self.assertTrue(z._parent == x)
        self.assertTrue(a._parent == y)
        self.assertTrue(b._parent == y)
        self.assertTrue(c._parent == z)
        self.assertTrue(d._parent == z)

        # check that x's family is everyone
        self.assertItemsEqual(x.family, [x.id, y.id, z.id, a.id, b.id, c.id, d.id])

        # check that x's leaf descendants are a,b,c,d
        self.assertItemsEqual(x.leaf_descendants, [a.id, b.id, c.id, d.id])

        # check that everyone has the right children
        self.assertItemsEqual(x._children, [y, z])
        self.assertItemsEqual(y._children, [a, b])
        self.assertItemsEqual(z._children, [c, d])

        # check that everybody's positions translate when we move x
        ypos = y.position
        zpos = z.position
        apos = a.position
        bpos = b.position
        cpos = c.position
        dpos = d.position
        x.position += (100, -100)
        self.assertTrue(np.array_equal(y.position - ypos, (100, -100)))
        self.assertTrue(np.array_equal(z.position - zpos, (100, -100)))
        self.assertTrue(np.array_equal(a.position - apos, (100, -100)))
        self.assertTrue(np.array_equal(b.position - bpos, (100, -100)))
        self.assertTrue(np.array_equal(c.position - cpos, (100, -100)))
        self.assertTrue(np.array_equal(d.position - dpos, (100, -100)))

        # check that everybody's positions translate when we move y
        xpos = x.position
        zpos = z.position
        apos = a.position
        bpos = b.position
        cpos = c.position
        dpos = d.position
        y.position += (200, -17)
        self.assertTrue(np.array_equal(x.position - xpos, (200, -17)))
        self.assertTrue(np.array_equal(z.position - zpos, (200, -17)))
        self.assertTrue(np.array_equal(a.position - apos, (200, -17)))
        self.assertTrue(np.array_equal(b.position - bpos, (200, -17)))
        self.assertTrue(np.array_equal(c.position - cpos, (200, -17)))
        self.assertTrue(np.array_equal(d.position - dpos, (200, -17)))

        # check that everybody's positions translate when we move z
        xpos = x.position
        ypos = y.position
        apos = a.position
        bpos = b.position
        cpos = c.position
        dpos = d.position
        z.position += (23, 27)
        self.assertTrue(np.array_equal(x.position - xpos, (23, 27)))
        self.assertTrue(np.array_equal(y.position - ypos, (23, 27)))
        self.assertTrue(np.array_equal(a.position - apos, (23, 27)))
        self.assertTrue(np.array_equal(b.position - bpos, (23, 27)))
        self.assertTrue(np.array_equal(c.position - cpos, (23, 27)))
        self.assertTrue(np.array_equal(d.position - dpos, (23, 27)))

        # check that everybody's positions translate when we move a
        xpos = x.position
        ypos = y.position
        apos = a.position
        bpos = b.position
        cpos = c.position
        dpos = d.position
        a.position += (14, 49)
        self.assertTrue(np.array_equal(x.position - xpos, (14, 49)))
        self.assertTrue(np.array_equal(y.position - ypos, (14, 49)))
        self.assertTrue(np.array_equal(a.position - apos, (14, 49)))
        self.assertTrue(np.array_equal(b.position - bpos, (14, 49)))
        self.assertTrue(np.array_equal(c.position - cpos, (14, 49)))
        self.assertTrue(np.array_equal(d.position - dpos, (14, 49)))

        # check that setting x's color sets everyone's color
        x.color = 2
        self.assertEqual(x.color, y.color)
        self.assertEqual(x.color, z.color)
        self.assertEqual(x.color, a.color)
        self.assertEqual(x.color, b.color)
        self.assertEqual(x.color, c.color)
        self.assertEqual(x.color, d.color)

        # check that setting z's color sets its descendants' colors
        z.color = 3
        self.assertEqual(z.color, c.color)
        self.assertEqual(z.color, d.color)

        # check that setting 3/4 of the colors to a value changes x's value to
        # the mode
        a.color = 3
        self.assertEqual(x.color, 3)

        # check that setting z's velocity sets everyone's velocity
        z.velocity = (1, 0)
        for foo in [y, z, a, b, c, d]:
            self.assertTrue(np.array_equal(x.velocity, foo.velocity))

        # check that mass is additive
        a.mass, b.mass, c.mass, d.mass = range(4)
        self.assertTrue(np.array_equal(x.mass, y.mass + z.mass))
        self.assertTrue(np.array_equal(x.mass, a.mass + b.mass + c.mass + d.mass))

        # check that momentum is additive
        self.assertTrue(np.array_equal(x.momentum, y.momentum + z.momentum))
        self.assertTrue(np.array_equal(x.momentum, a.momentum + b.momentum + c.momentum + d.momentum))

        # check that kinetic energy is additive
        self.assertEqual(x.kinetic_energy, y.kinetic_energy + z.kinetic_energy)
        self.assertEqual(x.kinetic_energy, a.kinetic_energy + b.kinetic_energy + c.kinetic_energy + d.kinetic_energy)

        # change mass and check that it is still additive
        a.mass += 1
        self.assertTrue(np.array_equal(x.mass, y.mass + z.mass))
        self.assertTrue(np.array_equal(y.mass, a.mass + b.mass))
        self.assertTrue(np.array_equal(z.mass, c.mass + d.mass))

        # check that setting z's acceleration sets everyone's acceleration
        z.acceleration = (0, 1)
        self.assertTrue(np.array_equal(x.acceleration, y.acceleration))
        self.assertTrue(np.array_equal(x.acceleration, z.acceleration))
        self.assertTrue(np.array_equal(x.acceleration, a.acceleration))
        self.assertTrue(np.array_equal(x.acceleration, b.acceleration))
        self.assertTrue(np.array_equal(x.acceleration, c.acceleration))
        self.assertTrue(np.array_equal(x.acceleration, d.acceleration))

        # check that x's shape is the positions of a, b, c, d
        self.assertEqual(x.shape.shape, (4, 2))
        self.assertTrue(np.array_equal(x.shape[0] + x.position, a.position))
        self.assertTrue(np.array_equal(x.shape[1] + x.position, b.position))
        self.assertTrue(np.array_equal(x.shape[2] + x.position, c.position))
        self.assertTrue(np.array_equal(x.shape[3] + x.position, d.position))

        for foo in [y, z, a, b, c, d]:
            # check that depth is constant among the family
            self.assertEqual(x.depth, foo.depth)
            
            # check that everybody's ids are unique
            self.assertTrue(x.id != foo.id)

            # check that x's extent captures all the sub-extents
            self.assertTrue(x.extent[0] <= foo.extent[0])
            self.assertTrue(x.extent[1] <= foo.extent[1])
            self.assertTrue(x.extent[2] >= foo.extent[2])
            self.assertTrue(x.extent[3] >= foo.extent[3])

    def test_hierarchy3(self):
        """Check that coupled family attributes behave as expected in deep
        hierarchies"""
        for leaf_type in ['basic', 'complex']:
            self.world = universe.create_world('basic', depth=4)
            self.world.remove_objects(self.world.objects)

            # first we create a 5-level hierarchy
            x = self.world.create_object('complex')
            bottom_level = [x]
            everyone = [x]
            for lvl in xrange(4):
                new_bottom_level = []
                child_type = 'complex' if lvl < 3 else leaf_type
                for a in bottom_level:
                    b = self.world.create_object(child_type)
                    c = self.world.create_object(child_type)
                    a.add_child(b)
                    a.add_child(c)
                    new_bottom_level.extend([b, c])
                bottom_level = new_bottom_level
                everyone.extend(bottom_level)

            y = x._children[0]._children[0]
            z = bottom_level[0]

            self.assertTrue('children' in x._attributes)

            # check that x's leaf descendants are equal to bottom_level
            self.assertItemsEqual(x._leaf_descendants, bottom_level)

            # check that everybody's positions translate when we move x
            posns = [a.position for a in everyone]
            x.position += (100, -100)
            for posn, node in zip(posns, everyone):
                self.assertTrue(np.array_equal(node.position - posn, (100, -100)))

            # check that everybody's positions translate when we move y
            posns = [a.position for a in everyone]
            y.position = y.position + (200, -17)
            for posn, node in zip(posns[4:], everyone[4:]):
                self.assertTrue(np.array_equal(node.position - posn, (200, -17)), repr((node.id, node.position - posn)))

            # check that everybody's positions translate when we move z
            posns = [a.position for a in everyone]
            z.position = z.position + (23, 27)
            for posn, node in zip(posns, everyone):
                self.assertTrue(np.array_equal(node.position - posn, (23, 27)), repr((node.id, node.position - posn)))

            if leaf_type == 'basic':
                # check that setting x's color sets everyone's color
                x.color = 2
                for node in everyone:
                    self.assertEqual(node.color, 2)

                # check that setting y's color sets only its descendants' colors
                y.color = 3
                for node in everyone:
                    if node in y._descendants:
                        self.assertEqual(node.color, 3)
                    else:
                        self.assertEqual(node.color, 2)

                # check that setting z's color sets only its color
                z.color = 4
                for node in everyone:
                    if node is z:
                        self.assertEqual(node.color, 4)
                    elif node in y._descendants:
                        self.assertEqual(node.color, 3)
                    else:
                        self.assertEqual(node.color, 2)

            # check that setting x's velocity sets everyone's velocity
            x.velocity = (2, 0)
            for node in everyone:
                self.assertTrue(np.array_equal(node.velocity, (2, 0)))

            # check that setting y's velocity sets everyone's velocity
            y.velocity = (3, 0)
            for node in everyone:
                self.assertTrue(np.array_equal(node.velocity, (3, 0)))

            # check that setting z's velocity sets everyone's velocity
            z.velocity = (1, 0)
            for node in everyone:
                self.assertTrue(np.array_equal(node.velocity, (1, 0)))

            # check that setting x's acceleration sets everyone's acceleration
            x.acceleration = (2, 0)
            for node in everyone:
                self.assertTrue(np.array_equal(node.acceleration, (2, 0)))

            # check that setting y's acceleration sets everyone's acceleration
            y.acceleration = (3, 0)
            for node in everyone:
                self.assertTrue(np.array_equal(node.acceleration, (3, 0)))

            # check that setting z's acceleration sets everyone's acceleration
            z.acceleration = (1, 0)
            for node in everyone:
                self.assertTrue(np.array_equal(node.acceleration, (1, 0)))

            # check that mass is additive
            for node in bottom_level:
                self.assertEqual(node.mass, 1)
            for node in everyone:
                self.assertEqual(node.mass, len(node.leaf_descendants))

            # doubling x's mass should double everyone's mass
            x.mass *= 2
            for node in bottom_level:
                self.assertEqual(node.mass, 2)
            for node in everyone:
                self.assertEqual(node.mass, 2 * len(node.leaf_descendants))
            self.assertEqual(x.mass, 32)

            # doubling y's mass should double its simple leaf descendants
            y.mass *= 2
            for node in y._simple_leaf_descendants:
                self.assertEqual(node.mass, 4)
            self.assertEqual(x.mass, 40)

            # doubling z's mass should only double z's mass, but that affects its
            # ancestors
            z.mass *= 2
            self.assertEqual(z.mass, 8)
            self.assertEqual(y.mass, 20)
            self.assertEqual(x.mass, 44)

            # check that depth is constant among the family
            x.depth = 1
            for node in everyone:
                self.assertEqual(node.depth, 1)
            y.depth = 2
            for node in everyone:
                self.assertEqual(node.depth, 2)
            z.depth = 3
            for node in everyone:
                self.assertEqual(node.depth, 3)

    def test_hierarchy4(self):
        """Check that coupled family attributes behave as we assemble hierarchies"""
        self.world.remove_objects(self.world.objects)

        # first we create a 2-level hierarchy of objects with standard
        # attributes:
        #            x
        #
        #         y     z
        #        / \   / \
        #       a   b c   d
        x = self.world.create_object('compound')
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

        # set some colors
        y.color = 2
        c.color = 3
        d.color = 4

        # add x at the top of the hierarchy
        x.add_child(y)
        x.add_child(z)

        # check that colors didn't change
        self.assertEqual(y.color, 2)
        self.assertEqual(a.color, 2)
        self.assertEqual(b.color, 2)
        self.assertEqual(c.color, 3)
        self.assertEqual(d.color, 4)

        # check that x has no attributes it shouldn't
        self.assertTrue(not hasattr(x, 'color'))
        self.assertTrue(not hasattr(x, 'visible'))

        # check that children are still visible
        self.assertTrue(y.visible)
        self.assertTrue(z.visible)
        self.assertTrue(a.visible)
        self.assertTrue(b.visible)
        self.assertTrue(c.visible)
        self.assertTrue(d.visible)

    def test_hierarchy_conflict(self):
        """Check that conflicts are resolved as expected even when there are deep
        hierarchies."""
        self.world.remove_objects(self.world.objects)

        # first we create a 3-level hierarchy:
        #            x
        #          /   \
        #         y     z
        #        / \   / \
        #       a   b c   d
        x = self.world.create_object('complex')
        y = self.world.create_object('complex')
        z = self.world.create_object('complex')
        a = self.world.create_object('basic')
        b = self.world.create_object('basic')
        c = self.world.create_object('basic')
        d = self.world.create_object('basic')
        x.add_child(y)
        x.add_child(z)
        y.add_child(a)
        y.add_child(b)
        z.add_child(c)
        z.add_child(d)


        q = self.world.create_object('basic')
        
        # check that conflicts are resolved
        q.position = a.position
        self.assertTrue(np.sum(np.abs(q.position - a.position)) >= 1)
        q.position = b.position
        self.assertTrue(np.sum(np.abs(q.position - b.position)) >= 1)
        q.position = c.position
        self.assertTrue(np.sum(np.abs(q.position - c.position)) >= 1)
        q.position = d.position
        self.assertTrue(np.sum(np.abs(q.position - d.position)) >= 1)


class TestCopyGetObjectAttribute(TestObjectAttribute):
    def test_copy_get_attribute(self):
        """Check that CopyGetObjectAttribute behaves as expected"""
        class FooObjectAttribute(core.CopyGetObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0
            _ndim = 2
        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        
        # check that x.foo returns a copy, not the original
        thing = x.foo
        self.assertTrue(np.array_equal(thing, np.array([1, 1])))
        thing[1] = 17
        self.assertTrue(np.array_equal(thing, np.array([1, 17])))
        thing2 = x.foo
        self.assertTrue(np.array_equal(thing2, np.array([1, 1])))


class TestCoupledFamilyObjectAttribute(TestObjectAttribute):
    def test_coupled_family_attribute(self):
        """Check that coupled family attribute obeys coupling rule"""
        class FooObjectAttribute(core.CoupledFamilyObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # add attribute to existing object and its children
        x = self.world.objects[3]
        x.add_attribute('foo')
        for idx in x.children:
            self.world.objects[idx].add_attribute('foo')

        # set parent and check that it was set
        x.foo = 17
        self.assertTrue(x.foo == 17)
        
        # check that change propagated down to children
        for idx in x.children:
            self.assertTrue(self.world.objects[idx].foo == 17, '%d' % self.world.objects[idx].foo)

        # set foo on one of x's children, check that change is propagated to x and all other children
        self.world.objects[x.children[0]].foo = 18
        self.assertTrue(x.foo == 18)
        for idx in x.children:
            self.assertTrue(self.world.objects[idx].foo == 18, '%d' % self.world.objects[idx].foo)


class TestModeParentObjectAttribute(TestObjectAttribute):
    def test_mode_parent_attribute(self):
        """Check that mode parent attribute obeys coupling rule"""
        class FooObjectAttribute(core.ModeParentObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # get existing object
        x = self.world.objects[3]
        self.assertTrue(len(x.children) == 3)

        # add foo to x and its children
        x.add_attribute('foo')
        for idx in x.children:
            self.world.objects[idx].add_attribute('foo')

        # set foo on x, check that change propagates down to children
        x.foo = 17
        self.assertTrue(x.foo == 17)
        for idx in x.children:
            self.assertTrue(self.world.objects[idx].foo == 17, '%d' % self.world.objects[idx].foo)

        # set foo values of children to their ids
        for idx in x.children:
            self.world.objects[idx].foo = idx
        # set foo on first child to id of last child so that there is a mode
        self.world.objects[x.children[0]].foo = x.children[-1]
        
        # check that all children have the values we gave them
        for idx in x.children:
            self.assertTrue(self.world.objects[idx].foo == idx or idx == x.children[0])

        # check that x's value is the mode
        self.assertTrue(x.foo == x.children[-1])


class TestSumParentObjectAttribute(TestObjectAttribute):
    def test_sum_parent_attribute(self):
        """Check that sum parent attribute obeys coupling rule"""
        class FooObjectAttribute(core.FloatObjectAttribute, core.SumParentObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # get an existing object with children
        x = self.world.objects[3]
        self.assertTrue(len(x.children) == 3)

        # add foo to x and its children
        x.add_attribute('foo')
        for idx in x.children:
            self.world.objects[idx].add_attribute('foo')

        # check that children have the default value and x has the sum
        self.assertTrue(x.foo == 3)
        for idx in x.children:
            self.assertTrue(self.world.objects[idx].foo == 1, '%d' % self.world.objects[idx].foo)

        # check that increasing x increases children proportionally
        x.foo = 18
        self.assertTrue(x.foo == 18)
        for idx in x.children:
            self.assertTrue(self.world.objects[idx].foo == 6, '%d' % self.world.objects[idx].foo)

        # make values of children different from one another
        for i, idx in enumerate(x.children):
            self.world.objects[idx].foo += i

        # check that values are what we set them to
        for i, idx in enumerate(x.children):
            self.assertTrue(self.world.objects[idx].foo == 6 + i)
            
        # check that value of x.foo is sum of children
        self.assertTrue(x.foo == 21)

        # check that increasing x increases children proportionally
        x.foo = 30
        self.assertTrue(x.foo == 30)
        for i, idx in enumerate(x.children):
            foo_got = self.world.objects[idx].foo
            foo_expected = float(30)/21 * (6 + i)
            self.assertTrue(np.abs(foo_got - foo_expected) < 1e-5, 
                            'expected %f, got %f' % (foo_expected, foo_got))


class TestMeanParentObjectAttribute(TestObjectAttribute):
    def test_mean_parent_attribute(self):
        """Check that mean parent attribute obeys coupling rule"""
        class FooObjectAttribute(core.FloatObjectAttribute, core.MeanParentObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # get existing object with children
        x = self.world.objects[3]
        self.assertTrue(len(x.children) == 3)

        # add foo to x and its children
        x.add_attribute('foo')
        for idx in x.children:
            self.world.objects[idx].add_attribute('foo')

        # check that everyone has the default value
        self.assertTrue(x.foo == 1)
        for idx in x.children:
            self.assertTrue(self.world.objects[idx].foo == 1, '%d' % self.world.objects[idx].foo)

        # check that increasing x.foo increases children by same amount
        x.foo = 18
        self.assertTrue(x.foo == 18)
        for idx in x.children:
            self.assertTrue(self.world.objects[idx].foo == 18, '%d' % self.world.objects[idx].foo)

        # make children's values different
        for i, idx in enumerate(x.children):
            self.world.objects[idx].foo += i
        
        # check that x has the right value
        self.assertTrue(x.foo == 19, x.foo)

        # check that increasing x.foo increases children by same amount
        x.foo = 28
        for i, idx in enumerate(x.children):
            self.assertTrue(self.world.objects[idx].foo == 27 + i, '%d' % self.world.objects[idx].foo)


class TestSteppingObjectAttribute(TestObjectAttribute):
    def test_stepping_attribute_order(self):
        """Check that ordering cycles cause an exception to be raised."""
        log = []
        class FooObjectAttribute(core.SteppingObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0
            _step_before = ['bar']
            _actions = ['NOOP']

            def _step_object(self, obj, t, dt, agent_id, action):
                global log
                log.append(('foo', obj, t, dt, action))

        class BarObjectAttribute(core.SteppingObjectAttribute):
            _name = 'bar'
            _default_value = 1
            _null_value = 0
            _step_before = ['foo']
            _actions = ['NOOP', 'LEFT', 'RIGHT']

            def _step_object(self, obj, t, dt, agent_id, action):
                global log
                log.append(('bar', obj, t, dt, action))

        x = self.world.objects[3]

        # check that we can't have both foo and bar exist in the world, since
        # that would create an ordering cycle
        with self.assertRaises(Exception):
            x.add_attribute('foo')
            x.add_attribute('bar')

    def test_stepping_attribute(self):
        """Check that stepping object attributes step in the order expected"""
        log = []
        class FooObjectAttribute(core.SteppingObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0
            _step_before = ['bar']
            _actions = ['NOOP']

            def _step_object(self, obj, t, dt, agent_id, action):
                if action in self._actions:
                    log.append(('foo', obj, t, dt, action))

        class BarObjectAttribute(core.SteppingObjectAttribute):
            _name = 'bar'
            _default_value = 1
            _null_value = 0
            _actions = ['NOOP', 'LEFT', 'RIGHT']

            def _step_object(self, obj, t, dt, agent_id, action):
                if action in self._actions:
                    log.append(('bar', obj, t, dt, action))

        class BazObjectAttribute(core.SteppingObjectAttribute):
            _name = 'baz'
            _default_value = 1
            _null_value = 0
            _step_after = ['bar']
            _actions = ['LEFT', 'RIGHT']

            def _step_object(self, obj, t, dt, agent_id, action):
                if action in self._actions:
                    log.append(('baz', obj, t, dt, action))

        class BatObjectAttribute(core.SteppingObjectAttribute):
            _name = 'bat'
            _default_value = 1
            _null_value = 0
            _step_after = ['baz']
            _actions = ['LEFT']

            def _step(self, t, dt, agent_id, action):
                if action in self._actions:
                    log.append(('bat', t, dt, action))
        
        # add attributes to existing objects
        x = self.world.objects[3]
        x.add_attribute('foo')
        x.add_attribute('bar')
        y = self.world.objects[8]
        y.add_attribute('baz')
        y.add_attribute('bat')
        
        log_expected = []

        # step world and keep track of what log entries we expect to be created
        self.world.step('NOOP')
        log_expected.extend([('foo', x, 0, 1, 'NOOP'), ('bar', x, 0, 1, 'NOOP')])
        
        # step world and keep track of what log entries we expect to be created
        self.world.step('LEFT')
        log_expected.extend([('bar', x, 1, 1, 'LEFT'), ('baz', y, 1, 1, 'LEFT'), ('bat', 1, 1, 'LEFT')])
        
        # step world and keep track of what log entries we expect to be created
        self.world.step('NOOP')
        log_expected.extend([('foo', x, 2, 1, 'NOOP'), ('bar', x, 2, 1, 'NOOP')])
        
        # step world and keep track of what log entries we expect to be created
        self.world.step('RIGHT')
        log_expected.extend([('bar', x, 3, 1, 'RIGHT'), ('baz', y, 3, 1, 'RIGHT')])
        
        # step world and keep track of what log entries we expect to be created
        self.world.step('NOOP')
        log_expected.extend([('foo', x, 4, 1, 'NOOP'), ('bar', x, 4, 1, 'NOOP')])

        # check that log is equal to log_expected
        self.assertTrue(log == log_expected, repr(log))


class TestChangeTrackingObjectAttribute(TestObjectAttribute):
    def test_change_tracking_attribute(self):
        """Check that we can access change-tracking attributes and that they behave as
        expected."""
        class FooObjectAttribute(core.ChangeTrackingObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # add attribute to existing objects
        x = self.world.objects[3]
        y = self.world.objects[4]
        x.add_attribute('foo')
        y.add_attribute('foo')
        foo = self.world.object_attributes['foo']

        # step the world
        self.world.step()

        # check that x.foo has default value
        self.assertTrue(x.foo == 1)

        # set x.foo and check that previous values are what we expect
        x.foo = 17
        self.assertTrue(x.foo == 17)
        self.assertTrue(y.foo == 1)
        self.assertTrue(foo.get(3) == 17)
        self.assertTrue(foo.get_previous(3) == 1)
        self.assertTrue(foo.change(3) == 16)
        self.assertTrue(foo.get(x) == 17)
        self.assertTrue(foo.get_previous(x) == 1)
        self.assertTrue(foo.change(x) == 16)
        self.assertTrue(np.array_equal(foo.change(np.array([3, 4])), np.array([16, 0])))

        # set x.foo and check that previous values are what we expect
        x.foo = 18
        self.assertTrue(x.foo == 18)
        self.assertTrue(foo.get(3) == 18)
        self.assertTrue(foo.get_previous(3) == 17)
        self.assertTrue(foo.get_previous(3, step=True) == 1)
        self.assertTrue(foo.change(3) == 1)
        self.assertTrue(foo.change(3, step=True) == 17, foo.change(3, step=True))
        self.assertTrue(foo.get(x) == 18)
        self.assertTrue(foo.get_previous(x, step=True) == 1)
        self.assertTrue(foo.change(x, step=True) == 17)
        self.assertTrue(np.array_equal(foo.change(np.array([3, 4])), np.array([1, 0])))
        self.assertTrue(np.array_equal(foo.change(np.array([3, 4]), step=True), np.array([17, 0])))


class TestConflictObjectAttribute(TestObjectAttribute):
    def test_conflict_attribute(self):
        """Check that conflict attributes resolve conflicts, including conflicts that
        result from conflict resolution. Also check that conflict resolution
        ignores compound objects."""
        class FooObjectAttribute(core.ConflictObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

            def _resolve_single_conflict(self, idx1, idx2):
                self.set(idx1, 2 * self.get(idx1))

        # pick some existing objects and add attribute to them
        x = self.world.objects[3]
        y = self.world.objects[4]
        z = self.world.objects[5]
        w = self.world.objects[6]
        x.add_attribute('foo')
        y.add_attribute('foo')
        z.add_attribute('foo')
        w.add_attribute('foo')

        # check that x is complex and w,y,z are simple
        self.assertTrue(self.world._object_types[3] != core.Object.SIMPLE)
        self.assertTrue(self.world._object_types[4] == core.Object.SIMPLE)
        self.assertTrue(self.world._object_types[5] == core.Object.SIMPLE)
        self.assertTrue(self.world._object_types[6] == core.Object.SIMPLE)

        # set values so that there is conflict
        x.foo = 3
        y.foo = 3
        z.foo = 3
        w.foo = 6
        
        # step the world
        self.world.step()
        
        # check that x got set to value given, as it is complex
        self.assertTrue(x.foo == 3)
        
        # check that the simple objects all have different values for foo
        self.assertTrue(y.foo != z.foo)
        self.assertTrue(y.foo != w.foo)
        self.assertTrue(w.foo != z.foo)

        # check that conflicts_with_value works correctly
        foo = self.world.object_attributes['foo']
        self.assertTrue(np.array_equal(foo.conflicts_with_value(y.id, y.foo), (1,)))
        

class TestDerivedObjectAttribute(TestObjectAttribute):
    def test_derived_attribute(self):
        """Check that derived object attributes behave as expected."""
        class FooObjectAttribute(core.CopyGetObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        class BarObjectAttribute(core.DerivedObjectAttribute):
            _name = 'bar'
            _default_value = 1
            _null_value = 0
            _read_only = False

            def _get_data_object(self, obj):
                return obj.foo + 1

            def _set_data_object(self, obj, val):
                obj.foo = val - 1

        class BazObjectAttribute(core.DerivedObjectAttribute):
            _name = 'baz'
            _default_value = 1
            _null_value = 0
            _read_only = True

            def _get_data_object(self, obj):
                return obj.foo + 1

        class BatObjectAttribute(core.DerivedObjectAttribute):
            _name = 'bat'
            _default_value = 1
            _null_value = 0

        # add attributes to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        x.add_attribute('bar')
        x.add_attribute('baz')

        # check that getting and setting works as expected
        self.assertTrue(x.foo == 0)
        self.assertTrue(x.bar == 1)
        self.assertTrue(x.baz == 1)
        x.foo = 17
        self.assertTrue(x.foo == 17)
        self.assertTrue(x.bar == 18)
        self.assertTrue(x.baz == 18)
        x.bar = 23
        self.assertTrue(x.foo == 22)
        self.assertTrue(x.bar == 23)
        self.assertTrue(x.baz == 23)

        # check that trying to write a read-only attribute raises an error
        with self.assertRaises(Exception):
            x.baz = 10

        # if we try to get the value from an attribute the object doesn't have,
        # we get the null value
        y = self.world.create_object('basic')
        bar = self.world.object_attributes['bar']
        self.assertEqual(bar.get(y), 0)

        # bat hasn't implemented get_data or get_data_object, so accessing it
        # raises an error
        y.add_attribute('bat')
        with self.assertRaises(NotImplementedError):
            print y.bat


class TestSyncObjectAttribute(TestObjectAttribute):
    def test_sync_attribute(self):
        """Check that sync object attributes behave as expected"""
        class FooObjectAttribute(core.CopyGetObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        class BarObjectAttribute(core.SyncObjectAttribute):
            _name = 'bar'
            _default_value = 1
            _null_value = 0
            _synced_attribute = 'foo'

        # add attributes to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        x.add_attribute('bar')
        
        # check that getting and setting works as expected
        self.assertTrue(x.foo == 1)
        self.assertTrue(x.bar == 1)
        x.bar = 17
        self.assertTrue(x.foo == 17)
        self.assertTrue(x.bar == 17)
        x.bar = 23
        self.assertTrue(x.foo == 23)
        self.assertTrue(x.bar == 23)


class TestLinkObjectAttribute(TestObjectAttribute):
    def test_link_attribute(self):
        """Check that link object attributes behave as expected"""
        class FooObjectAttribute(core.CopyGetObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        class BarObjectAttribute(core.LinkObjectAttribute):
            _name = 'bar'
            _null_value = 0
            _linked_attribute = 'foo'

        # add attributes to an existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        x.add_attribute('bar')

        # check that getting and setting works as expected
        self.assertTrue(x.foo == 1)
        self.assertTrue(x.bar == 1)
        x.foo = 17
        self.assertTrue(x.foo == 17)
        self.assertTrue(x.bar == 17)
        x.bar = 23
        self.assertTrue(x.foo == 23)
        self.assertTrue(x.bar == 23)

        # check that _default_value() of link object attribute is the value of
        # the linked attribute
        bar = self.world.object_attributes['bar']
        self.assertEqual(bar._default_value(x), x.foo)


class TestNonnegativeObjectAttribute(TestObjectAttribute):
    def test_nonnegative_attribute(self):
        """Check that non-negative attributes can be set to non-negative values but not
        negative values"""
        class FooObjectAttribute(core.NonNegativeObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')

        # check that foo can have positive or zero values but not negative
        # values
        self.assertTrue(x.foo == 1)
        x.foo = 17
        self.assertTrue(x.foo == 17)
        x.foo = 0
        self.assertTrue(x.foo == 0)
        with self.assertRaises(TypeError):
            x.foo = -17
        self.assertTrue(x.foo == 0, x.foo)


class TestFractionalObjectAttribute(TestObjectAttribute):
    def test_fractional_attribute(self):
        """Check that fractional object attributes can be set to fractional values but
k        not values outside [0, 1]"""
        class FooObjectAttribute(core.FractionalFloatObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        
        # check that foo can have legal values but not illegal values
        self.assertTrue(x.foo == 1)
        with self.assertRaises(TypeError):
            x.foo = 17
        with self.assertRaises(TypeError):
            x.foo = -17
        x.foo = 0.5
        self.assertEqual(x.foo, 0.5)


class TestScalarObjectAttribute(TestObjectAttribute):
    def test_scalar_attribute(self):
        """Check that scalar object attributes behave as expected"""
        class FooObjectAttribute(core.ScalarObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')

        # check that getting and setting work
        self.assertTrue(x.foo == 1)
        x.foo = 17
        self.assertTrue(x.foo == 17)


class TestIntegerObjectAttribute(TestObjectAttribute):
    def test_integer_attribute(self):
        """Check that integer object attributes can only have integer values"""
        class FooObjectAttribute(core.IntegerObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')

        # check that foo can only have integer values and has the right type
        self.assertTrue(x.foo == 1)
        x.foo = 17
        self.assertTrue(x.foo == 17)
        x.foo = 17.3
        self.assertTrue(x.foo == 17, x.foo)
        self.assertTrue(type(x.foo) == np.int64, type(x.foo))


class TestFloatObjectAttribute(TestObjectAttribute):
    def test_float_attribute(self):
        """Check that float object attributes behave as expected"""
        class FooObjectAttribute(core.FloatObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')

        # check that getting and setting work, and that return value has the
        # right type
        self.assertTrue(x.foo == 1)
        x.foo = 17
        self.assertTrue(x.foo == 17.0)
        self.assertTrue(type(x.foo) == np.float64, type(x.foo))


class TestNonnegativeIntegerObjectAttribute(TestObjectAttribute):
    def test_nonnegative_integer_attribute(self):
        """Check that non-negative integer object attributes behave as expected"""
        class FooObjectAttribute(core.NonNegativeIntegerObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        
        # check that x.foo can only have non-negative values and has the right
        # type
        self.assertTrue(x.foo == 1)
        x.foo = 17
        self.assertTrue(x.foo == 17)
        with self.assertRaises(TypeError):
            x.foo = -17
        self.assertTrue(x.foo == 17, x.foo)
        self.assertTrue(type(x.foo) == np.int64, type(x.foo))


class TestNonnegativeFloatObjectAttribute(TestObjectAttribute):
    def test_nonnegative_float_attribute(self):
        """Check that non-negative float object attributes behave as expected"""
        class FooObjectAttribute(core.NonNegativeFloatObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        
        # check that x.foo can only have non-negative values and has the right
        # type
        self.assertTrue(x.foo == 1)
        x.foo = 17
        self.assertTrue(x.foo == 17.0)
        with self.assertRaises(TypeError):
            x.foo = -17
        self.assertTrue(x.foo == 17.0, x.foo)
        self.assertTrue(type(x.foo) == np.float64, type(x.foo))


class TestBooleanObjectAttribute(TestObjectAttribute):
    def test_boolean_attribute(self):
        """Check that boolean object attributes can be set and accessed, and that they
        have the right type"""
        class FooObjectAttribute(core.BooleanObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')

        # check that getting and setting work, and that x.foo has the right type
        self.assertTrue(x.foo == 1)
        x.foo = 0
        self.assertTrue(x.foo == 0)
        self.assertTrue(type(x.foo) == np.bool_, type(x.foo))


class TestStringObjectAttribute(TestObjectAttribute):
    def test_string_attribute(self):
        """Check that string object attribtues can be set and accessed, and that they
        have the right type"""
        class FooObjectAttribute(core.StringObjectAttribute):
            _name = 'foo'
            _default_value = 'doormat'
            _null_value = 'wendigo'

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')

        # check that x has the default value
        self.assertTrue(x.foo == 'doormat', repr(x.foo))

        # check that getting and setting works and that x.foo has the right
        # type
        x.foo = 'fluxion'
        self.assertTrue(x.foo == 'fluxion')
        self.assertTrue(type(x.foo) == np.string_, type(x.foo))


class TestObjectObjectAttribute(TestObjectAttribute):
    def test_object_object_attribute(self):
        """Check that object object attributes can be set and accessed as expected"""
        class Widget:
            next_id = 0
            def __init__(self, y=7):
                self.x = 3
                self.y = 7
                self.id = self.next_id
                self.next_id += 1
                
            def __eq__(self, other):
                return (self.x == other.x and self.y == other.y)

            def __repr__(self):
                return 'Widget(x=%d, y=%d, id=%d)' % (self.x, self.y, self.id)

        class FooObjectAttribute(core.ObjectObjectAttribute):
            _name = 'foo'
            _default_value = Widget(8)
            _null_value = Widget(3)

        # check that equality testing works
        self.assertTrue(Widget(8) == Widget(8))
        self.assertTrue(Widget(8) != Widget(9))

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')

        # check that getting and setting works and that we can access object
        # attributes
        self.assertTrue(x.foo == Widget(8), repr(x.foo))
        x.foo = Widget(9)
        self.assertTrue(x.foo == Widget(9), repr(x.foo))
        x.foo.x = 17
        self.assertTrue(x.foo != Widget(9), repr(x.foo))


class TestListObjectAttribute(TestObjectAttribute):
    def test_list_attribute(self):
        """Check that list object attributes can be set and accessed as expected"""
        class FooObjectAttribute(core.ListObjectAttribute):
            _name = 'foo'
            _default_value = [3, 4, 5]
            _null_value = [4, 5, 6]

        class BarObjectAttribute(core.ListObjectAttribute):
            _null_value = [4, 5, 6]

        # add attributes to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        x.add_attribute('bar')

        # check that x.foo has default value
        self.assertTrue(np.array_equal(x.foo, [3, 4, 5]), repr(x.foo))

        # check that getting and setting works as expected
        x.foo = [6, 7, 8]
        self.assertTrue(np.array_equal(x.foo, [6, 7, 8]))
        with self.assertRaises(TypeError):
            x.foo = 19
            
        # check that foo can be None
        x.foo = None
        self.assertTrue(x.foo is None)

        # check that unspecified default value is the empty list
        self.assertTrue(np.array_equal(x.bar, []))


class TestFloatVectorObjectAttribute(TestObjectAttribute):
    def test_float_vector_attribute(self):
        """Check that float vector attribute can be set and accessed as expected"""
        class FooObjectAttribute(core.FloatVectorObjectAttribute):
            _name = 'foo'
            _default_value = np.ones(2)
            _null_value = 0
            _ndim = 2

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')

        # check that getting and setting work
        self.assertTrue(np.array_equal(x.foo, np.ones(2)), repr(x.foo))
        x.foo = [4, 5]
        self.assertTrue(np.array_equal(x.foo, [4, 5]))


class TestPointObjectAttribute(TestObjectAttribute):
    def test_point_attribute(self):
        """Check that point object attribute can be set and accessed as expected"""
        class FooObjectAttribute(core.PointObjectAttribute):
            _name = 'foo'
            _default_value = np.ones(2)
            _null_value = 0

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')

        # check that getting and setting work
        self.assertTrue(np.array_equal(x.foo, np.ones(2)), repr(x.foo))
        x.foo = [4, 5]
        self.assertTrue(np.array_equal(x.foo, [4, 5]))


class TestHasObjectAttribute(TestObjectAttribute):
    def test_has_attribute(self):
        """Check that has attributes work correctly"""
        class FooObjectAttribute(core.FloatObjectAttribute):
            _name = 'foo'
            _default_value = 1
            _null_value = 0

        class BarObjectAttribute(core.HasObjectAttribute):
            _name = 'bar'
            _default_value = True
            _null_value = False
            _has_attribute = 'foo'

        class BazObjectAttribute(core.HasObjectAttribute):
            _name = 'baz'

        # add attributes to existing objects
        x = self.world.objects[3]
        y = self.world.objects[4]
        x.add_attribute('foo')
        x.add_attribute('bar')
        y.add_attribute('bar')
        y.add_attribute('baz')

        # check that has attributes have correct values
        self.assertTrue(x.bar)
        self.assertFalse(y.bar)
        self.assertFalse(y.baz)

        # check that baz.get() returns correct value
        baz = self.world.object_attributes['baz']
        self.assertTrue(np.array_equal(baz.get([x, y]), (False, False)))


class TestAbilityObjectAttribute(TestObjectAttribute):
    def test_ability_attribute(self):
        """Check that ability object attributes work as expected"""
        log = []
        class FooObjectAttribute(core.AbilityObjectAttribute):
            _name = 'foo'
            def _execute_action(self, obj, t, dt, agent_id, action):
                log.append((obj, t, dt, agent_id, action))

        # add attribute to existing object
        x = self.world.objects[3]
        x.add_attribute('foo')
        
        # set to True to enable behavior
        x.foo = True

        # step the world and check that log is correct
        self.world.step('NOOP')
        self.world.step('LEFT')
        self.world.step('RIGHT')
        self.assertTrue(log == [(x, 0, 1, 0, 'NOOP'),
                                (x, 1, 1, 0, 'LEFT'),
                                (x, 2, 1, 0, 'RIGHT')], repr(log))

        # check that the base AbilityObjectAttribute raises an exception if you
        # try to step it
        x.add_attribute('ability_attribute')
        with self.assertRaises(NotImplementedError):
            self.world.step('NOOP')


class TestInteractsObjectAttribute(TestObjectAttribute):
    def test_interacts_attribute(self):
        """Check that interacts object attribute processes interactions when objects
        overlap"""
        class FooObjectAttribute(core.InteractsObjectAttribute):
            def _interact(self, obj1, obj2):
                obj1.color = 2
                obj2.color = 3

        class BarObjectAttribute(core.InteractsObjectAttribute):
            pass

        # remove existing objects from world
        self.world.remove_objects(self.world.objects)

        # create objects with zero mass so they can overlap
        x = self.world.create_object(['basic', {'mass': 0}])
        y = self.world.create_object(['basic', {'mass': 0}])

        # check that color is default color
        self.assertEqual(x.color, 1)
        self.assertEqual(y.color, 1)

        # add interacts attribute to x
        x.add_attribute('foo')

        # cause an interaction
        y.position = x.position
        self.world.step()

        # check that interaction took place
        self.assertEqual(x.color, 2)
        self.assertEqual(y.color, 3)

        # check that invisible objects don't interact
        x.color = y.color = 1
        x.visible = False
        self.world.step()
        self.assertEqual(x.color, 0)
        self.assertEqual(y.color, 1)

        # check that setting x.foo to False disables interactions
        x.visible = True
        x.foo = False
        self.world.step()
        self.assertEqual(x.color, 1)
        self.assertEqual(y.color, 1)

        # check that we get a NotImplementedError if the attribute doesn't
        # implement _interact()
        x.add_attribute('bar')
        with self.assertRaises(NotImplementedError):
            self.world.step()


class TestListeningObjectAttribute(TestObjectAttribute):
    def test_listening_attribute(self):
        """Check that listening object attribute processes selected events as
        expected"""
        log = []

        class FleekEvent(core.Event):
            _parameters = ['strangeness']

        class FooObjectAttribute(core.ListeningObjectAttribute):
            _selected_events = ['fleek']
            def _process_event_object(self, evt, obj, t, dt, agent_id, action): 
                log.append((evt.strangeness, t))

        class CowObjectAttribute(core.SteppingObjectAttribute):
            """Exists solely to create an event"""
            _step_before = ['foo']
            def _step(self, t, dt, agent_id, action): 
                event = FleekEvent(self.world, strangeness=2)
                
        # add object to world and add attributes to it
        x = self.world.create_object(['basic', {'mass': 0}])
        x.add_attribute('foo')
        x.add_attribute('cow')

        # step the world and check that the log is what we expect
        self.world.step()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[-1], (2, 0))


class TestStateMachineObjectAttribute(TestObjectAttribute):
    def test_state_machine_attribute(self):
        """Check that state machine object attribute steps through states as expected
        when we step the world."""
        log = []
        class FooObjectAttribute(core.StateMachineObjectAttribute):
            def _execute(self, obj, old_state):
                if old_state == 0:
                    log.append(0)
                    return 1
                elif old_state == 1:
                    log.append(1)
                    return 2
                elif old_state == 2:
                    log.append(2)
                    return 0

        # create an object and add the attribute to it
        x = self.world.create_object('basic')
        x.add_attribute('foo')

        # set state machine state to 0
        x.foo = 0

        # step the world and check that log shows object stepping through
        # states in order
        self.world.step()
        self.assertEqual(log, [0])
        self.world.step()
        self.assertEqual(log, [0, 1])
        self.world.step()
        self.assertEqual(log, [0, 1, 2])
        self.world.step()
        self.assertEqual(log, [0, 1, 2, 0])


class TestNameObjectAttribute(TestObjectAttribute):
    def test_name_attribute(self):
        """Check that name object attribute can be set and accessed"""
        x = self.world.objects[3]
        self.assertTrue(x.name == 'self_big', x.name)
        y = self.world.create_object('basic')
        y.name = 'foo3'
        self.assertTrue(y.name == 'foo3', y.name)


class TestIdObjectAttribute(TestObjectAttribute):
    def test_id_attribute(self):
        """Check that id object attribute is the same as index in world objects
        array"""
        for idx, obj in enumerate(self.world.objects):
            self.assertTrue(obj.id == idx)


class TestChildrenObjectAttribute(TestObjectAttribute):
    def test_children_attribute(self):
        """Check that children object attribute is a list of the correct child ids"""
        # assemble a compound object
        x = self.world.create_object('compound')
        ids = []
        for _ in xrange(3):
            y = self.world.create_object('basic')
            x.add_child(y)
            ids.append(y.id)

        # check that x has the right children
        self.assertTrue(sorted(x.children) == sorted(ids))


class TestParentObjectAttribute(TestObjectAttribute):
    def test_parent_attribute(self):
        """Check that parent object attribute is correct"""
        # assemble a compound object
        x = self.world.create_object('compound')
        ids = []
        objs = []
        for _ in xrange(3):
            y = self.world.create_object('basic')
            x.add_child(y)
            ids.append(y.id)
            objs.append(y)

        # check that x has the right children
        self.assertTrue(sorted(x.children) == sorted(ids))

        # check that x's children have x as parent
        for y in objs:
            self.assertEqual(y.parent, x.id)

        # parent value of -1 means no parent
        x.add_attribute('parent')
        parent = self.world.object_attributes['parent']
        self.assertEqual(parent.get(x), -1)


class TestFamilyObjectAttribute(TestObjectAttribute):
    def test_family_attribute(self):
        """Check that family attribute returns correct value"""
        # assemble a compound object
        x = self.world.create_object('compound')
        ids = [x.id]
        objs = []
        for _ in xrange(3):
            y = self.world.create_object('basic')
            x.add_child(y)
            ids.append(y.id)
            objs.append(y)

        # check that x has the right family
        self.assertTrue(sorted(x.family) == sorted(ids))

        # check that x's children have the right family
        for y in objs:
            self.assertTrue(sorted(y.family) == sorted(ids))


class TestSimpleFamilyObjectAttribute(TestObjectAttribute):
    def test_simple_family_attribute(self):
        """Check that simple_family attribute returns correct value"""
        # assemble a compound object
        x = self.world.create_object('compound')
        ids = []
        objs = []
        for _ in xrange(3):
            y = self.world.create_object('basic')
            x.add_child(y)
            ids.append(y.id)
            objs.append(y)

        # check that x has the right simple family
        self.assertTrue(sorted(x.simple_family) == sorted(ids))

        # check that x's children have the right simple family
        for y in objs:
            self.assertTrue(sorted(y.simple_family) == sorted(ids))


class TestHeadOfFamilyObjectAttribute(TestObjectAttribute):
    def test_head_of_family_attribute(self):
        """Check that head_of_family attribute returns correct value"""
        # assemble a compound object
        x = self.world.create_object('compound')
        objs = []
        for _ in xrange(3):
            y = self.world.create_object('basic')
            x.add_child(y)
            objs.append(y)

        # check that x has the right head of family
        self.assertTrue(x.head_of_family == x.id)

        # check that x's children have the right head of family
        for y in objs:
            self.assertTrue(y.head_of_family == x.id)


class TestTopLevelObjectAttribute(TestObjectAttribute):
    def test_top_level_attribute(self):
        """Check that top_level attribute returns correct value"""
        # assemble a compound object
        x = self.world.create_object('compound')
        objs = []
        for _ in xrange(3):
            y = self.world.create_object('basic')
            x.add_child(y)
            objs.append(y)

        # check that x is top-level
        self.assertTrue(x.top_level)

        # check that x's children are not top-level
        for y in objs:
            self.assertTrue(not y.top_level)


class TestAncestorsObjectAttribute(TestObjectAttribute):
    def test_ancestors_attribute(self):
        """Check that ancestors attribute returns correct value"""
        # assemble a compound object
        x = self.world.create_object('compound')
        objs = []
        for _ in xrange(3):
            y = self.world.create_object('basic')
            x.add_child(y)
            objs.append(y)

        # check that x has the right ancestors
        self.assertTrue(np.array_equal(x.ancestors, []))

        # check that x's children have the right ancestors
        for y in objs:
            self.assertTrue(np.array_equal(y.ancestors, [x.id]))


class TestDescendantsObjectAttribute(TestObjectAttribute):
    def test_descendants_attribute(self):
        """Check that descendants attribute returns correct value"""
        # assemble a compound object
        x = self.world.create_object('compound')
        ids = []
        objs = []
        for _ in xrange(3):
            y = self.world.create_object('basic')
            x.add_child(y)
            objs.append(y)
            ids.append(y.id)

        # check that x has the right descendants
        self.assertTrue(sorted(x.descendants) ==  sorted([x.id] + ids))

        # check that x's children have the right descendants
        for y in objs:
            self.assertTrue(np.array_equal(y.descendants, [y.id]))


class TestLeafDescendantsObjectAttribute(TestObjectAttribute):
    def test_leaf_descendants_attribute(self):
        """Check that leaf_descendants attribute returns correct value"""
        # assemble a compound object
        x = self.world.create_object('compound')
        ids = []
        objs = []
        for _ in xrange(3):
            y = self.world.create_object('basic')
            x.add_child(y)
            objs.append(y)
            ids.append(y.id)

        # check that x has the right leaf descendants
        self.assertTrue(sorted(x.leaf_descendants) ==  sorted(ids))

        # check that x's children have the right leaf descendants
        for y in objs:
            self.assertTrue(np.array_equal(y.leaf_descendants, [y.id]))


class TestSimpleLeafDescendantsObjectAttribute(TestObjectAttribute):
    def test_simple_leaf_descendants_attribute(self):
        """Check that simple_leaf_descendants attribute returns correct value"""
        # assemble a compound object
        x = self.world.create_object('compound')
        ids = []
        objs = []
        for _ in xrange(3):
            y = self.world.create_object('basic')
            x.add_child(y)
            objs.append(y)
            ids.append(y.id)

        # check that x has the right simple leaf descendants
        self.assertTrue(sorted(x.simple_leaf_descendants) ==  sorted(ids))

        # check that x's children have the right simple leaf descendants
        for y in objs:
            self.assertTrue(np.array_equal(y.simple_leaf_descendants, [y.id]))


class TestMoveableObjectAttribute(TestObjectAttribute):
    def test_moveable_attribute(self):
        """Check that objects are moveable exactly when they have the velocity
        attribute"""
        for obj in self.world.objects:
            if obj.moveable:
                self.assertTrue(hasattr(obj, 'velocity'))
            else:
                self.assertTrue(not hasattr(obj, 'velocity'))
                with self.assertRaises(Exception):
                    obj.velocity = 3


class TestPushableObjectAttribute(TestObjectAttribute):
    def test_pushable_attribute(self):
        """Check that objects are pushable exactly when they have the mass attribute"""
        for obj in self.world.objects:
            if obj.pushable:
                self.assertTrue(hasattr(obj, 'mass'))
            else:
                self.assertTrue(not hasattr(obj, 'mass'))
                with self.assertRaises(Exception):
                    obj.mass = 3


class TestZorderObjectAttribute(TestObjectAttribute):
    def test_zorder_attribute(self):
        """Check that zorder determines which object's color appears in the world
        state"""
        x = self.world.create_object('basic')
        y = self.world.create_object('basic')
        x.color = 1
        y.color = 2
        x.zorder = 0
        y.zorder = 1
        x.mass = 0
        y.mass = 0
        x.position = y.position
        r, c = np.int32(x.position)
        self.assertTrue(y.color == self.world.state[r, c])
        x.zorder = 2
        self.assertTrue(x.color == self.world.state[r, c])


class TestEvent(TestCase):
    world = None
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestEvent, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def check_addition_removal(self, e):
        """Check that we can add and remove events"""
        # check that e was added correctly
        self.assertTrue(e in self.world.events)

        # check that e was removed correctly
        e._remove_world()
        self.assertTrue(e not in self.world.events)


class TestGoal(TestCase):
    world = None
    
    goal1 = None
    goal2 = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'blank')
        
        super(TestGoal, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
        
        class Goal1(core.Goal):
            _test_achieved = False
            
            _exclusive = True
            _terminates = True
            
            def _is_achieved(self):
                return self._test_achieved
            
            def _achieve(self):
                self._test_achieved = True
                
                return True
        
        class Goal2(Goal1):
            _exclusive = False
            _terminates = False

        class Goal3(Goal1):
            _exclusive = True
            _terminates = True
        
        self.goal1 = Goal1
        self.goal2 = Goal2
        self.goal3 = Goal3
    
    def tearDown(self):
        self.world.end()
        self.world = 'blank'
    
    def test(self):
        """Check that we can add a goal"""
        g = self.goal1(self.world)
        
        self.assertTrue(g.name in self.world.goals)
    
    def test_activate(self):
        """Check that we can activate and deactivate a goal"""
        g = self.goal1(self.world)
        
        #active by default
        self.assertTrue(g.active)
        self.assertTrue(g.name in self.world.active_goals)
        
        #deactivate works
        g.deactivate()
        self.assertFalse(g.active)
        self.assertFalse(g.name in self.world.active_goals)
        
        #activate works
        g.activate()
        self.assertTrue(g.active)
        self.assertTrue(g.name in self.world.active_goals)
        
        #set_activation works
        active = g.set_activation(False)
        self.assertTrue(active)
        self.assertFalse(g.active)
    
    def test_constructor_arguments(self):
        """Check that we can use constructor arguments to override default goal
        behavior"""
        g = self.goal1(self.world, terminates=True, active=True, exclusive=True)
        self.assertTrue(g.terminates)
        self.assertTrue(g.active)
        self.assertTrue(g.exclusive)
        
        g = self.goal1(self.world, terminates=False, active=False, exclusive=False)
        self.assertFalse(g.terminates)
        self.assertFalse(g.active)
        self.assertFalse(g.exclusive)
    
    def test_achieve(self):
        """Check that we can achieve the goal using goal.achieve()"""
        g = self.goal1(self.world)
        
        self.assertFalse(g.is_achieved())
        self.assertFalse(g.achieved)
        
        self.assertTrue(g.achieve())
        self.assertTrue(g.is_achieved())
        self.assertTrue(g.achieved)
    
    def test_exclusive(self):
        """Check that exclusive goals inactivate each other, and non-exclusive goals
        don't""" 
        g1 = self.goal1(self.world)
        g2 = self.goal2(self.world)
        g3 = self.goal3(self.world)
        
        #g3 should inactivate g1
        self.assertFalse(g1.active)
        self.assertTrue(g3.active)
        
        #g1 should inactivate g3
        g1.activate()
        self.assertFalse(g3.active)
        
        #g2 shouldn't inactivate g1
        g2.activate()
        self.assertTrue(g1.active)
    
    def test_termination(self):
        """Check that goals cause termination exactly when they should"""

        #achieving shouldn't cause the Judge to terminate
        g2 = self.goal2(self.world)
        g2.achieve()
        obs, reward, done, info = self.world.step('NOOP')
        self.assertFalse(done)
        
        #achieving should cause the Judge to terminate
        g1 = self.goal1(self.world)
        g1.achieve()
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(done)
    
    def test_any_termination(self):
        """Test that judge in any-goal-terminates mode terminates when any goal is
        achieved"""

        #either goal terminates
        g1 = self.goal2(self.world, name='goal1', terminates=True)
        g2 = self.goal2(self.world, terminates=True)

        judge = core.Judge(self.world, goal_termination_mode='any')

        obs, reward, done, info = self.world.step('NOOP')
        self.assertFalse(done)

        g1.activate()
        g1.achieve()
        self.assertTrue(g1.achieved)
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(done)
        
    def test_all_termination(self):
        """Test that judge in all-goals-required-to-terminate mode terminates only when
        all goals have been achieved"""
        #both goals need to be achieved before termination
        g1 = self.goal2(self.world, terminates=True)
        g2 = self.goal2(self.world, terminates=True)
        
        judge = core.Judge(self.world, goal_termination_mode='all')
        
        g1.achieve()
        obs, reward, done, info = self.world.step('NOOP')
        self.assertFalse(done)
        
        g2.achieve()
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(done)
    
    def test_active_termination(self):
        """Check that only active goals cause the judge to terminate"""
        #make sure inactive goals don't prevent termination
        g1 = self.goal2(self.world, terminates=True)
        g2 = self.goal2(self.world, terminates=True)
        
        judge = core.Judge(self.world, goal_termination_mode='all')
        
        g1.deactivate()
        
        obs, reward, done, info = self.world.step('NOOP')
        self.assertFalse(done)
        
        g2.achieve()
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(done)

    def test_addition_removal(self):
        """Check that we can add and remove goals from the world"""
        # check that we can add g1 to the world
        g1 = self.goal1(self.world)
        self.assertTrue(g1.name in self.world.goals)

        # check that we can add g2 to the world
        g2 = self.goal2(self.world)
        self.assertTrue(g2.name in self.world.goals)
        
        # check that we can remove g1 from the world
        g1._remove_world()
        self.assertTrue(g1.name not in self.world.goals)

        # check that we can remove g2 from the world
        g2._remove_world()
        self.assertTrue(g2.name not in self.world.goals)


class TestJudge(TestCase):
    world = None
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestJudge, self).__init__(*args, **kwargs)
    
    def setUp(self):
        class MyJudge(core.Judge):
            _reward_goals = [dict(goal='fleek', params=lambda judge, event: event.check == 13,
                                  reward=lambda judge, event: event.onness),
                             dict(goal='stronk', params=dict(strink=3), reward=19),
                             dict(goal='flee_floo', params=dict(flink=lambda judge, value: value == 6), reward=23),
                             ]
            _reward_events = [dict(event='my', params=lambda judge, event: event.check == 13,
                                   reward=lambda judge, event: event.flingrippery)]
            _termination_events = [dict(event='duck')]

            def _calculate_reward(self, goals, events):
                tot = 0
                for e in events:
                    if e.name == 'fling':
                        tot += 22
                return tot

            def _is_done(self, goals, events):
                for e in events:
                    if e.name == 'fling':
                        return True
                return False
        
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world, judge='my')
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def test_goals(self):
        """Check that judge terminates when it should based on goals being achieved."""
        # some simple goals
        class Goal1(core.Goal):
            _test_achieved = False
            
            _exclusive = True
            _terminates = True
            
            def _is_achieved(self):
                return self._test_achieved
            
            def _achieve(self):
                self._test_achieved = True
                
                return True

        class Goal2(Goal1):
            pass

        # set termination mode to all
        self.world.judge = core.Judge(self.world, goal_termination_mode='all')

        # create two goals
        g1 = Goal2(self.world)
        g2 = Goal2(self.world)
        
        # check that we don't terminate when neither goal is achieved
        obs, reward, done, stuff = self.world.step()
        self.assertEqual(done, False)

        g1.achieve()

        # check that we don't terminate when only one goal is achieved
        obs, reward, done, stuff = self.world.step()
        self.assertEqual(done, False)

        g2.achieve()

        # check that we do terminate when both goals are achieved
        obs, reward, done, stuff = self.world.step()
        self.assertEqual(done, True)

        # check that we can't set the goal termination mode to nonsense
        self.world.judge = core.Judge(self.world, goal_termination_mode='eleventy-seven')
        with self.assertRaises(ValueError):
            self.world.step()

    def test_rewards(self):
        """Check that judge gives correct rewards based on events and goals."""
        # some rewarding events
        class MyEvent(core.Event):
            _name = 'my'
            _parameters = ['flingrippery', 'check']
        
        class CowEvent(core.Event):
            _name = 'cow'
            _reward = 3
            
            _terminates = True
        
        class FlingEvent(core.Event):
            _name = 'fling'

        # some rewarding goals
        class FleekGoal(core.Goal):
            _name = 'fleek'
            onness = 17
            check = 13

        class StronkGoal(core.Goal):
            _name = 'stronk'
            strink = 19

        class FleeFlooGoal(core.Goal):
            _name = 'flee_floo'
            flink = 6
        
        # check that we get -1 for stepping with no goals and no rewards
        self.assertTrue(self.world.judge.calculate_reward([], []) == -1)

        # check that we get -1 for stepping
        obs, reward, done, stuff = self.world.step()
        self.assertTrue(reward == -1)

        e = MyEvent(self.world, flingrippery=12, check=13)
        # check that we get 12 for flingrippery and -1 for stepping
        self.assertTrue(self.world.judge.calculate_reward([], [e]) == 11)

        e = CowEvent(self.world)
        # check that we get 3 for cow and -1 for stepping
        self.assertTrue(self.world.judge.calculate_reward([], [e]) == 2)

        e = FlingEvent(self.world)
        # check that we get 22 for fling and -1 for stepping
        self.assertTrue(self.world.judge.calculate_reward([], [e]) == 21)

        g = FleekGoal(self.world)
        # check that we get 1000 for the goal, 17 extra, and -1 for stepping
        self.assertEqual(self.world.judge.calculate_reward([g], []), 1016)

        g = StronkGoal(self.world)
        # check that we get 1000 for the goal and -1 for stepping
        self.assertEqual(self.world.judge.calculate_reward([g], []), 999)

        g = FleeFlooGoal(self.world)
        # check that we get 1000 for the goal, 23 extra, and -1 for stepping
        self.assertEqual(self.world.judge.calculate_reward([g], []), 1022)

    def test_done(self):
        """Check that judge terminates when it should based on events"""
        class MyEvent(core.Event):
            _name = 'my'
            _parameters = ['flingrippery']
        
        class CowEvent(core.Event):
            _name = 'cow'
            _reward = 3
            
            _terminates = True

        # check that we don't terminate with no goals and no events
        self.assertTrue(self.world.judge.is_done([], []) == False)

        e = MyEvent(self.world, flingrippery=12)
        # check that MyEvent doesn't cause termination
        self.assertTrue(self.world.judge.is_done([], [e]) == False)

        e = CowEvent(self.world)
        # check that CowEvent does cause termination
        self.assertTrue(self.world.judge.is_done([], [e]) == True)

    def test_done2(self):
        """Check that judge terminates when it should based on events"""
        class MyEvent(core.Event):
            _name = 'my'
            _parameters = ['flingrippery']
        
        class DuckEvent(core.Event):
            _name = 'duck'
        
        # check that we don't terminate with no goals and no events
        self.assertTrue(self.world.judge.is_done([], []) == False)

        e = MyEvent(self.world, flingrippery=12)
        # check that MyEvent doesn't cause termination
        self.assertTrue(self.world.judge.is_done([], [e]) == False)

        e = DuckEvent(self.world)
        # check that DuckEvent does cause termination
        self.assertTrue(self.world.judge.is_done([], [e]) == True)

    def test_done3(self):
        """Check that judge terminates when it should based on events"""
        class MyEvent(core.Event):
            _name = 'my'
            _parameters = ['flingrippery']
        
        class FlingEvent(core.Event):
            _name = 'fling'
        
        
        # check that we don't terminate with no goals and no events
        self.assertTrue(self.world.judge.is_done([], []) == False)

        # check that MyEvent doesn't cause termination
        e = MyEvent(self.world, flingrippery=12)
        self.assertTrue(self.world.judge.is_done([], [e]) == False)

        # check that FlingEvent does cause termination
        e = FlingEvent(self.world)
        self.assertTrue(self.world.judge.is_done([], [e]) == True)

    def test_addition_removal(self):
        """Check that we can add and remove judge from the world"""
        judge = core.Judge(self.world)

        # check that we can remove the world's judge
        judge._remove_world()
        self.assertIs(self.world.judge, None)

    def test_end_time(self):
        """Check that judge ends episode when end time arrives"""
        self.world.judge._end_time = 2

        # check that we don't terminate when time=1
        obs, reward, done, stuff = self.world.step()
        self.assertEqual(done, False)

        # check that we do terminate when time=2
        obs, reward, done, stuff = self.world.step()
        self.assertEqual(done, True)


class TestVariant(TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestVariant, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'
    
    def check_addition_removal(self, v):
        """Check that we can add and remove variant from the world"""
        self.assertTrue(v.name in self.world.variants)
        v._remove_world()
        self.assertTrue(v.name not in self.world.variants)
        

class TestWorldAttributeVariant(TestVariant):
    def test(self):
        """Check that we can set a world attribute via a world attribute variant"""
        class TimeVariant(core.WorldAttributeVariant):
            _states = [11, 17]

        variant = TimeVariant(self.world)

        # set time, check
        variant.set(11)
        self.assertEqual(self.world.time, 11)

        # set time, check
        variant.set(17)
        self.assertEqual(self.world.time, 17)

        # set time to a state that doesn't exist
        with self.assertRaises(ValueError):
            variant.set(23)

    def test_addition_removal(self):
        """Check that we can add and remove variant from the world"""
        class TimeVariant(core.WorldAttributeVariant):
            _states = [11, 17]

        variant = TimeVariant(self.world)

        self.check_addition_removal(variant)


class TestObjectAttributeVariant(TestVariant):
    def test(self):
        """Check that we can change object attributes via a variant"""
        class ColorVariant(core.ObjectAttributeVariant):
            _states = [1, 2, 3, 4]
            _object_attribute_filter = {'name': 'x'}

        variant = ColorVariant(self.world)

        # create an object that matches the filter
        x = self.world.create_object(['basic', dict(name='x')])

        # set color, check
        variant.set(1)
        self.assertEqual(x.color, 1)

        # set color, check
        variant.set(2)
        self.assertEqual(x.color, 2)

        # set color, check
        variant.set(3)
        self.assertEqual(x.color, 3)

        # set color, check
        variant.set(4)
        self.assertEqual(x.color, 4)

        # set color to a state that doesn't exist
        with self.assertRaises(ValueError):
            variant.set(5)

    def test_all_objects(self):
        """Check that we can change object attributes of all objects via a variant"""

        # pick a world where everything has a color
        self.world = universe.create_world('blockpush')

        class ColorVariant(core.ObjectAttributeVariant):
            _states = [1, 2, 3, 4]

        variant = ColorVariant(self.world)

        # set color, check
        variant.set(1)
        for x in self.world.objects:
            self.assertEqual(x.color, 1)

        # set color, check
        variant.set(2)
        for x in self.world.objects:
            self.assertEqual(x.color, 2)

        # set color, check
        variant.set(3)
        for x in self.world.objects:
            self.assertEqual(x.color, 3)

        # set color, check
        variant.set(4)
        for x in self.world.objects:
            self.assertEqual(x.color, 4)

        # set color to a state that doesn't exist
        with self.assertRaises(ValueError):
            variant.set(5)

    def test_addition_removal(self):
        """Check that we can add and remove variant from the world"""
        class ColorVariant(core.ObjectAttributeVariant):
            _states = [1, 2, 3, 4]

        variant = ColorVariant(self.world)

        self.check_addition_removal(variant)


class TestBooleanVariant(TestVariant):
    def test_all_objects(self):
        """Check that we can change object attributes of all objects via a variant"""
        # pick a world where everything is visible
        self.world = universe.create_world('blockpush')

        class VisibleVariant(core.BooleanVariant, core.ObjectAttributeVariant):
            pass

        variant = VisibleVariant(self.world)

        # set visible, check
        variant.set(False)
        for x in self.world.objects:
            self.assertEqual(x.visible, False)

        # set visible, check
        variant.set(True)
        for x in self.world.objects:
            self.assertEqual(x.visible, True)

        # set visible to a state that doesn't exist
        with self.assertRaises(ValueError):
            variant.set(5)

    def test_addition_removal(self):
        """Check that we can add and remove variant from the world"""
        class VisibleVariant(core.BooleanVariant, core.ObjectAttributeVariant):
            pass

        variant = VisibleVariant(self.world)

        self.check_addition_removal(variant)


class TestRandomizer(TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestRandomizer, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()


class TestAgent(TestCase):
    world = None
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestAgent, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def test_legal_actions(self):
        """Check that the agent produces only legal actions and that it can process the
        action result."""
        for _ in xrange(20):
            # check that only legal actions are produced by get_action()
            action = self.world.agent.get_action(self.world.last_observation, self.world.tools)
            self.assertTrue(action in self.world.actions)
            
            # check that process_action_result() doesn't die
            self.world.agent.process_action_result(self.world.last_observation, action, 1, False)

    def test_removal(self):
        """Check that we can remove the agent from the world without problems."""
        self.world._remove_agent(self.world.agent)
        self.assertTrue(self.world.agent is None)

    def test_multi_agent_mode(self):
        """Check that we are not in multi-agent mode"""
        # check that we are NOT in multi-agent mode
        self.assertTrue(not self.world.multi_agent_mode)


class TestMultiAgent(TestCase):
    world = None
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestMultiAgent, self).__init__(*args, **kwargs)
    
    def setUp(self):
        class MyAgent(agents.RandomAgent):
            pass
        
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world, 
                                               agent=['my', ['my', dict(allowed_actions=['LEFT', 'RIGHT'])]])
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def test_multi_agent_mode(self):
        """Check that we are in multi-agent mode"""
        # check that we are in multi-agent mode
        self.assertTrue(self.world.multi_agent_mode)

    def test_removal(self):
        """Check that we can remove all agents from the world"""
        for agent in list(self.world.agent):
            self.world._remove_agent(agent)
        self.assertTrue(self.world.agent == [])


class TestSingleMultiAgent(TestMultiAgent):
    world = None
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestSingleMultiAgent, self).__init__(*args, **kwargs)
    
    def setUp(self):
        class MyAgent(core.Agent):
            _default_action = 'LEFT'
        
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world, 
                                               agent=[['my', dict(allowed_actions=['LEFT', 'RIGHT'])]])
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def test(self):
        """Check that single agent always uses default action."""
        # check that we world.agent is a single-element list
        self.assertEqual(len(self.world.agent), 1)

        # check that default action is what it should be
        self.assertEqual(self.world.agent[0]._default_action, 'LEFT')

        for _ in xrange(40):
            obs, reward, done, stuff = self.world.step()
            # check that agent uses default action
            self.assertEqual(self.world.agent[0].get_action(obs, self.world.tools), 'LEFT')
        
            # check that process_action_result() doesn't die
            self.world.agent[0].process_action_result(obs, 'LEFT', reward, done)


class TestTwoMultiAgent(TestMultiAgent):
    world = None
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestTwoMultiAgent, self).__init__(*args, **kwargs)
    
    def setUp(self):
        class MyAgent(agents.RandomAgent):
            pass
        
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world, 
                                               agent=['my', ['my', dict(allowed_actions=['LEFT', 'RIGHT'])]])
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def test(self):
        """Check that world.agent is what we expect and that agent with restricted
        actions obeys the restriction."""
        # check that world.agent is a two-element list
        self.assertEqual(len(self.world.agent), 2)

        for _ in xrange(40):
            obs, reward, done, stuff = self.world.step()
            # check that second agent uses allowed action
            self.assertTrue(self.world.agent[1].get_action(obs, self.world.tools) in ['LEFT', 'RIGHT'])


class TestWorldAttribute(TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestWorldAttribute, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
        self.world = 'basic'

    def test_removal(self):
        """Check that we can remove all world attributes"""
        for attr in self.world.world_attributes.keys():
            self.world.world_attributes[attr].remove()

    def test_compare(self):
        """Check that we can get and compare all world attributes"""
        for attr in self.world.world_attributes.keys():
            attr = self.world.world_attributes[attr]
            self.assertTrue(attr.compare(attr.get(), attr._get()))
            self.assertTrue(attr.compare(attr.get(), attr()))


class TestUnchangeableWorldAttribute(TestWorldAttribute):
    def setUp(self):
        super(TestUnchangeableWorldAttribute, self).setUp()

        class FooWorldAttribute(core.UnchangeableWorldAttribute):
            pass

        # add foo
        self.world.get_world_attribute(['foo', dict(value=3)])

    def test_unchangeable_world_attribute(self):
        """Check that world attribute was set during setup, and that it cannot be
        changed"""
        # check that foo was set
        self.assertEqual(self.world.foo, 3)

        # check that foo is unchangeable
        with self.assertRaises(RuntimeError):
            self.world.foo = 18


class TestNonNegativeWorldAttribute(TestWorldAttribute):
    def setUp(self):
        super(TestNonNegativeWorldAttribute, self).setUp()

        class FooWorldAttribute(core.NonNegativeWorldAttribute):
            pass

        # add foo
        self.world.get_world_attribute(['foo', dict(value=3)])

    def test_non_negative_world_attribute(self):
        """Check that world attribute can be set, but not to a negative value"""
        # check that foo was set
        self.assertEqual(self.world.foo, 3)

        # set foo
        self.world.foo = 17

        # check that foo was set
        self.assertEqual(self.world.foo, 17)

        # check that foo cannot be negative
        with self.assertRaises(TypeError):
            self.world.foo = -3


class TestPositiveWorldAttribute(TestWorldAttribute):
    def setUp(self):
        super(TestPositiveWorldAttribute, self).setUp()

        class FooWorldAttribute(core.PositiveWorldAttribute):
            pass

        # add foo
        self.world.get_world_attribute(['foo', dict(value=3)])

    def test_positive_world_attribute(self):
        """Check that world attribute can be set, but not to a non-positive value"""
        # check that foo was set
        self.assertEqual(self.world.foo, 3)

        # set foo
        self.world.foo = 17

        # check that foo was set
        self.assertEqual(self.world.foo, 17)

        # check that foo cannot be negative
        with self.assertRaises(TypeError):
            self.world.foo = -3

        # check that foo cannot be zero
        with self.assertRaises(TypeError):
            self.world.foo = 0


class TestScalarWorldAttribute(TestWorldAttribute):
    def setUp(self):
        super(TestScalarWorldAttribute, self).setUp()

    def test_scalar_world_attribute(self):
        """Check that scalar world attribute behaves as expected"""
        class FooWorldAttribute(core.ScalarWorldAttribute):
            pass

        # add foo
        self.world.get_world_attribute(['foo', dict(value=3)])

        # check that foo was set
        self.assertEqual(self.world.foo, 3)

        # check that foo has the right dtype and ndim
        self.assertEqual(type(self.world.foo), np.int64)
        self.assertEqual(self.world.get_world_attribute('foo').dtype, np.int64)
        self.assertEqual(self.world.get_world_attribute('foo').ndim, 1)

        # set foo
        self.world.foo = 17

        # check that foo was set
        self.assertEqual(self.world.foo, 17)

        # check that foo has the right dtype
        self.assertEqual(type(self.world.foo), np.int64)


class TestIntegerWorldAttribute(TestWorldAttribute):
    def setUp(self):
        super(TestIntegerWorldAttribute, self).setUp()

        class FooWorldAttribute(core.IntegerWorldAttribute):
            pass

        # add foo
        self.world.get_world_attribute(['foo', dict(value=3)])

    def test_integer_world_attribute(self):
        """Check that integer world attribute behaves as expected"""
        # check that foo was set
        self.assertEqual(self.world.foo, 3)

        # check that foo has the right dtype and ndim
        self.assertEqual(type(self.world.foo), np.int64)
        self.assertEqual(self.world.foo.dtype, np.int64)
        self.assertEqual(self.world.get_world_attribute('foo').dtype, np.int64)
        self.assertEqual(self.world.get_world_attribute('foo').ndim, 1)

        # set foo
        self.world.foo = 17

        # check that foo was set
        self.assertEqual(self.world.foo, 17)

        # check that foo has the right dtype
        self.assertEqual(type(self.world.foo), np.int64)


class TestFloatWorldAttribute(TestWorldAttribute):
    def setUp(self):
        super(TestFloatWorldAttribute, self).setUp()

        class FooWorldAttribute(core.FloatWorldAttribute):
            pass

        # add foo
        self.world.get_world_attribute(['foo', dict(value=3)])

    def test_float_world_attribute(self):
        """Check that float world attribute behaves as expected"""
        # check that foo was set
        self.assertEqual(self.world.foo, 3.0)

        # check that foo has the right dtype and ndim
        self.assertEqual(type(self.world.foo), np.float64)
        self.assertEqual(self.world.foo.dtype, np.float64)
        self.assertEqual(self.world.get_world_attribute('foo').dtype, np.float64)
        self.assertEqual(self.world.get_world_attribute('foo').ndim, 1)

        # set foo
        self.world.foo = 17

        # check that foo was set
        self.assertEqual(self.world.foo, 17)

        # check that foo has the right dtype
        self.assertEqual(type(self.world.foo), np.float64)


class TestBooleanWorldAttribute(TestWorldAttribute):
    def setUp(self):
        super(TestBooleanWorldAttribute, self).setUp()

        class FooWorldAttribute(core.BooleanWorldAttribute):
            pass

        # add foo
        self.world.get_world_attribute(['foo', dict(value=3)])

    def test_boolean_world_attribute(self):
        """Check that boolean world attribute behaves as expected"""
        # check that foo was set
        self.assertEqual(self.world.foo, True)

        # check that foo has the right dtype and ndim
        self.assertEqual(type(self.world.foo), np.bool_)
        self.assertEqual(self.world.foo.dtype, np.bool_)
        self.assertEqual(self.world.get_world_attribute('foo').dtype, np.bool_)
        self.assertEqual(self.world.get_world_attribute('foo').ndim, 1)

        # set foo
        self.world.foo = False

        # check that foo was set
        self.assertEqual(self.world.foo, False)

        # check that foo has the right dtype
        self.assertEqual(type(self.world.foo), np.bool_)


class TestStringWorldAttribute(TestWorldAttribute):
    def setUp(self):
        super(TestStringWorldAttribute, self).setUp()

        class FooWorldAttribute(core.StringWorldAttribute):
            pass

        # add foo
        self.world.get_world_attribute(['foo', dict(value=3)])

    def test_string_world_attribute(self):
        """Check that string world attribute behaves as expected"""
        # check that foo was set
        self.assertEqual(self.world.foo, '3')

        # check that foo has the right dtype and ndim
        self.assertTrue(isinstance(self.world.foo, str))
        self.assertEqual(self.world.get_world_attribute('foo').dtype, np.dtype(str))
        self.assertEqual(self.world.get_world_attribute('foo').ndim, 1)

        # set foo
        self.world.foo = False

        # check that foo was set
        self.assertEqual(self.world.foo, 'False')

        # check that foo has the right dtype
        self.assertTrue(isinstance(self.world.foo, str))


class TestVectorWorldAttribute(TestWorldAttribute):
    def setUp(self):
        super(TestVectorWorldAttribute, self).setUp()

        class FooWorldAttribute(core.FloatWorldAttribute):
            _ndim = 2

        # add foo
        self.world.get_world_attribute(['foo', dict(value=(3, 4))])

    def test_string_world_attribute(self):
        """Check that vector world attribute behaves as expected"""

        # check that foo was set
        self.assertTrue(np.array_equal(self.world.foo, (3, 4)))

        # check that foo has the right dtype and ndim
        self.assertTrue(isinstance(self.world.foo, np.ndarray))
        self.assertEqual(self.world.get_world_attribute('foo').dtype, np.float64)
        self.assertEqual(self.world.get_world_attribute('foo').ndim, 2)

        # set foo
        self.world.foo = 17

        # check that foo was set
        self.assertTrue(np.array_equal(self.world.foo, (17, 17)))

        # check that foo has the right dtype
        self.assertTrue(isinstance(self.world.foo, np.ndarray))

    def test_to_vector(self):
        """Check that _to_vector() works correctly"""
        foo = self.world.get_world_attribute('foo')

        # check that _to_vector fixes dtype
        self.assertEqual(foo._to_vector(np.array([3.3, 3.4], dtype=np.float32)).dtype, np.float64)

        # check that _to_vector squashes 1x2 and 2x1 to just 2
        self.assertTrue(np.array_equal(foo._to_vector(np.array([[3.3, 3.4]])), (3.3, 3.4)))
        self.assertTrue(np.array_equal(foo._to_vector(np.array([[3.3], [3.4]])), (3.3, 3.4)))


class TestHeightWorldAttribute(TestWorldAttribute):
    def test_height(self):
        """Check that height world attribute behaves as expected"""
        self.world = universe.create_world('basic', height=17)

        # check that the height was set
        self.assertEqual(self.world.height, 17)

        # check that height is unchangeable
        with self.assertRaises(RuntimeError):
            self.world.height = 18

        # check that we can't have a negative height
        with self.assertRaises(TypeError):
            self.world = universe.create_world('basic', height=-17)

        # check that world state is the right shape
        self.assertEqual(self.world.state.shape, (17, 17))
    

class TestWidthWorldAttribute(TestWorldAttribute):
    def test_width(self):
        """Check that width world attribute behaves as expected"""
        self.world = universe.create_world('basic', width=17)

        # check that the width was set
        self.assertEqual(self.world.width, 17)

        # check that the width is unchangeable
        with self.assertRaises(RuntimeError):
            self.world.width = 18

        # check that we can't have a negative width
        with self.assertRaises(TypeError):
            self.world = universe.create_world('basic', width=-17)

        # check that world state is the right shape
        self.assertEqual(self.world.state.shape, (17, 17))


class TestDepthWorldAttribute(TestWorldAttribute):
    def test_depth(self):
        """Check that depth world attribute behaves as expected"""
        self.world = universe.create_world('basic', depth=17)

        # check that the depth was set
        self.assertEqual(self.world.depth, 17)

        # check that the depth is unchangeable
        with self.assertRaises(RuntimeError):
            self.world.depth = 18

        # check that we can't have a negative depth
        with self.assertRaises(TypeError):
            self.world = universe.create_world('basic', depth=-17)

        # check that objects respect the depth
        with self.assertRaises(TypeError):
            x = self.world.create_object(['basic', dict(depth=20)])


class TestTimeWorldAttribute(TestWorldAttribute):
    def test_time(self):
        """Check that time world attribute behaves as expected"""
        # check that time starts at 0
        self.assertEqual(self.world.time, 0)

        # set time
        self.world.time = 17

        # check that the time was set
        self.assertEqual(self.world.time, 17)

        # set time
        self.world.time = -17

        # check that the time was set
        self.assertEqual(self.world.time, -17)

        # check that stepping the world increments the time
        self.world.step()
        self.assertEqual(self.world.time, -16)


class TestNameWorldAttribute(TestWorldAttribute):
    def test_name(self):
        """Check that name world attribute behaves as expected"""
        # check that name is correct
        self.assertEqual(self.world.name, 'basic')

        # set name
        self.world.name = 'fliggedy floo'

        # check that the name was set
        self.assertEqual(self.world.name, 'fliggedy floo')


class BehaviorTest(TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(BehaviorTest, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
    
    def test_collision(self):
        # check that we can collide complex objects without them falling apart
        self.world.remove_objects(self.world.objects)
        shape = """
 XXX
XXXXX
 XXX
"""
        x = self.world.create_object(['complex', dict(shape=shape, position=(10, 0), velocity=(0, 10))])
        y = self.world.create_object(['complex', dict(shape=shape, position=(10, 20), velocity=(0, -10))])

        family = self.world.object_attributes['family']

        for i in xrange(10):
            self.world.step()
        for id in x.children:
            self.assertLess(np.linalg.norm(self.world.objects[id].position - x.position), 3)
        for id in y.children:
            self.assertLess(np.linalg.norm(self.world.objects[id].position - y.position), 3)
    
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

    #single_test(TestCompoundObject, 'test_family_caches')
