import unittest
import random

import numpy as np

from pixelworld.envs.pixelworld import agents, universe, library
from pixelworld.envs.pixelworld.tests import test_core
from pixelworld.envs.pixelworld.tests.library.world import test_library_world

taxi = library.import_item('world', 'taxi')

class TestTaxi(test_library_world.TestLibraryWorld):
    world = None
    
    def __init__(self, *args, **kwargs):
        super(TestTaxi, self).__init__(*args, **kwargs)

        self.world = kwargs.pop('world', 'taxi')
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()

    def test_spawning(self):
        # check that we start with no passengers
        self.assertEqual(len(self.world.objects.find(name='passenger')), 0)

        # check that there is one indicator and that it is black
        self.assertEqual(len(self.world.objects.find(name='indicator')), 1)
        self.assertEqual(self.world.objects['indicator'].color, 0)

        # step the world to spawn a passenger
        self.world.step()

        for _ in xrange(20):
            # check that we have exactly one passenger
            self.assertEqual(len(self.world.objects.find(name='passenger')), 1)

            # check that the indicator color is the same as the passenger's destination
            self.assertEqual(self.world.objects['passenger'].destination.color,
                             self.world.objects['indicator'].color)
            
            # since the passenger was just spawned, check that their position
            # is not their destination
            self.assertTrue((self.world.objects['passenger'].position != 
                             self.world.objects['passenger'].destination.position).any())
            
            # move the taxi to the passenger
            self.world.objects['taxi'].position = self.world.objects['passenger'].position
            
            # check that the non-picked-up passenger is in front of the taxi and is red
            self.assertTrue(self.world.objects['passenger'].zorder > self.world.objects['taxi'].zorder) 
            self.assertEqual(self.world.objects['passenger'].color, 2)
            si = self.world.objects['passenger'].state_index.tolist()
            self.assertEqual(self.world.state[tuple(si)], 2)

            # pick up the passenger
            obs, reward, done, info = self.world.step('PICKUP')
            self.assertEqual(reward, 9)
            self.assertEqual(done, False)

            # check that the picked-up passenger is in front of the taxi and is orange
            self.assertTrue(self.world.objects['passenger'].zorder > self.world.objects['taxi'].zorder) 
            self.assertEqual(self.world.objects['passenger'].color, 8)
            si = self.world.objects['passenger'].state_index.tolist()
            self.assertEqual(self.world.state[tuple(si)], 8)

            # move the taxi and the passenger to the destination
            self.world.objects['taxi'].position = self.world.objects['passenger'].destination.position
            self.world.objects['passenger'].position = self.world.objects['passenger'].destination.position

            # check that the picked-up passenger is in front of the taxi and is orange
            self.assertTrue(self.world.objects['passenger'].zorder > self.world.objects['taxi'].zorder) 
            self.assertEqual(self.world.objects['passenger'].color, 8)
            si = self.world.objects['passenger'].state_index.tolist()
            self.assertEqual(self.world.state[tuple(si)], 8)

            # drop off the passenger and check that we get reward and that
            # episode is not done
            obs, reward, done, info = self.world.step('DROPOFF')
            self.assertEqual(reward, 9)
            self.assertEqual(done, False)

            # check that the newly spawned passenger is red and visible
            self.assertEqual(self.world.objects['passenger'].color, 2)
            si = self.world.objects['passenger'].state_index.tolist()
            self.assertEqual(self.world.state[tuple(si)], 2)

            # check that the taxi is yellow and visible if the passenger wasn't
            # spawned on top of it
            self.assertEqual(self.world.objects['taxi'].color, 5)
            si2 = self.world.objects['passenger'].state_index.tolist()
            if si2 != si:
                self.assertEqual(self.world.state[tuple(si)], 2)

    def test_moving(self):
        # step the world to spawn a passenger
        self.world.step()

        # move the taxi to the passenger
        self.world.objects['taxi'].position = self.world.objects['passenger'].position

        # pick up the passenger
        obs, reward, done, info = self.world.step('PICKUP')
        self.assertEqual(reward, 9)
        self.assertEqual(done, False)

        push_actions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        for _ in xrange(100):
            # take a random push action and check that we got the reward we
            # expected and that the episode is not done
            action = random.choice(push_actions)
            obs, reward, done, info = self.world.step(action)
            self.assertEqual(reward, -1)
            self.assertEqual(done, False)

            # check that the picked-up passenger moved with the taxi, is in
            # front of the taxi and is orange
            self.assertTrue((self.world.objects['taxi'].position == self.world.objects['passenger'].position).all())
            self.assertTrue(self.world.objects['passenger'].zorder > self.world.objects['taxi'].zorder) 
            self.assertEqual(self.world.objects['passenger'].color, 8)
            si = self.world.objects['passenger'].state_index.tolist()
            self.assertEqual(self.world.state[tuple(si)], 8)

    def test_scoring(self):
        # step the world to spawn a passenger
        self.world.step()

        for _ in xrange(20):
            # move the taxi to the passenger
            self.world.objects['taxi'].position = self.world.objects['passenger'].position

            # pick up the passenger
            obs, reward, done, info = self.world.step('PICKUP')
            self.assertEqual(reward, 9)
            self.assertEqual(done, False)
            self.assertTrue(isinstance(self.world.events[-1], taxi.GoodPickup))

            # choose a random position that's not the destination
            posn = np.array([random.randint(1, self.world.height - 1), random.randint(1, self.world.width - 1)])
            while ((posn == self.world.objects['passenger'].destination.position).all() or
                   self.world.state[tuple(posn.tolist())] == 1):
                posn = np.array([random.randint(1, self.world.height - 1), random.randint(1, self.world.width - 1)])

            # move the taxi and the passenger to the position
            self.world.objects['taxi'].position = posn
            self.world.objects['passenger'].position = posn

            # try to drop off the passenger in a bad position
            obs, reward, done, info = self.world.step('DROPOFF')
            self.assertEqual(reward, -101)
            self.assertEqual(done, False)
            self.assertTrue(isinstance(self.world.events[-1], taxi.BadDropoff))

            # try to pick up a non-existent passenger
            obs, reward, done, info = self.world.step('PICKUP')
            self.assertEqual(reward, -101)
            self.assertEqual(done, False)
            self.assertTrue(isinstance(self.world.events[-1], taxi.BadPickup))

            # move the taxi and the passenger to the destination
            self.world.objects['taxi'].position = self.world.objects['passenger'].destination.position
            self.world.objects['passenger'].position = self.world.objects['passenger'].destination.position

            # try to pick up a non-existent passenger
            obs, reward, done, info = self.world.step('PICKUP')
            self.assertEqual(reward, -101)
            self.assertEqual(done, False)
            self.assertTrue(isinstance(self.world.events[-1], taxi.BadPickup))

            # drop off the passenger
            obs, reward, done, info = self.world.step('DROPOFF')
            self.assertEqual(reward, 9)
            self.assertEqual(done, False)
            self.assertTrue(isinstance(self.world.events[-1], taxi.GoodDropoff))

            # choose a random position that's not the new passenger's location
            posn = np.array([random.randint(1, self.world.height - 1), random.randint(1, self.world.width - 1)])
            while ((posn == self.world.objects['passenger'].position).all() or
                   self.world.state[tuple(posn.tolist())] == 1):
                posn = np.array([random.randint(1, self.world.height - 1), random.randint(1, self.world.width - 1)])

            # move the taxi and the passenger to the position
            self.world.objects['taxi'].position = posn

            # try to pick up a non-existent passenger
            obs, reward, done, info = self.world.step('PICKUP')
            self.assertEqual(reward, -101)
            self.assertEqual(done, False)
            self.assertTrue(isinstance(self.world.events[-1], taxi.BadPickup))

            # try to drop off a non-existent passenger
            obs, reward, done, info = self.world.step('DROPOFF')
            self.assertEqual(reward, -101)
            self.assertEqual(done, False)
            self.assertTrue(isinstance(self.world.events[-1], taxi.BadDropoff))

    def test_vector_state(self):
        self.world._observed_attributes = ['position', 'color']

        # step the world to spawn a passenger
        self.world.step()

        shape = self.world.vector_state.shape

        push_actions = ['LEFT', 'RIGHT', 'UP', 'DOWN']
        for _ in xrange(20):
            # check that vector shape stayed the same
            self.assertEqual(shape, self.world.vector_state.shape)

            # move the taxi to the passenger
            self.world.objects['taxi'].position = self.world.objects['passenger'].position

            # pick up the passenger
            obs, reward, done, info = self.world.step('PICKUP')
            self.assertEqual(reward, 9)
            self.assertEqual(done, False)
            self.assertTrue(isinstance(self.world.events[-1], taxi.GoodPickup))

            # check that vector shape stayed the same
            self.assertEqual(shape, self.world.vector_state.shape)

            for _ in xrange(5):
                # take a random push action and check that we got the reward we
                # expected and that the episode is not done
                action = random.choice(push_actions)
                obs, reward, done, info = self.world.step(action)
                self.assertEqual(reward, -1)
                self.assertEqual(done, False)

                # check that vector shape stayed the same
                self.assertEqual(shape, self.world.vector_state.shape)

            # move the taxi and the passenger to the destination
            self.world.objects['taxi'].position = self.world.objects['passenger'].destination.position
            self.world.objects['passenger'].position = self.world.objects['passenger'].destination.position

            # drop off the passenger
            obs, reward, done, info = self.world.step('DROPOFF')
            self.assertEqual(reward, 9)
            self.assertEqual(done, False)
            self.assertTrue(isinstance(self.world.events[-1], taxi.GoodDropoff))

            # check that vector shape stayed the same
            self.assertEqual(shape, self.world.vector_state.shape)

if __name__ == '__main__':
    unittest.main()
