import unittest

import numpy as np

from pixelworld.envs.pixelworld import agents, universe
from pixelworld.envs.pixelworld.tests.library.world import test_library_world

class TestPushBlock(test_library_world.TestLibraryWorld):
    world = None
    
    def __init__(self, *args, **kwargs):
        super(TestPushBlock, self).__init__(*args, **kwargs)

        self.world = kwargs.pop('world', 'pushblock')
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()

    def test_scoring(self):
        block = self.world.objects['block']
        player = self.world.objects['agent']
        goal = self.world.objects['goal']

        goal.position = (10, 10)
        block.position = goal.position + (1,0)
        player.position = goal.position + (2,0)

        # check that we don't get points and the episode doesn't end when the
        # block is not on top of the goal
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1)
        self.assertFalse(done)

        # check that we get points and the episode does end when the block is
        # on top of the goal
        obs, reward, done, info = self.world.step('UP')
        self.assertEqual(reward, 999)
        self.assertTrue(done)

    def test_warmer_colder(self):
        self.world = universe.create_world('pushblock', warmercolder=True)
        block = self.world.objects['block']
        player = self.world.objects['agent']
        goal = self.world.objects['goal']
        
        goal.position = (10, 10)
        block.position = goal.position + (2,0)
        player.position = goal.position + (3,0)

        # check that we lose a point when the block doesn't get closer to the
        # goal
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1)
        self.assertFalse(done)

        # check that we get a point when the block does get closer to the goal
        obs, reward, done, info = self.world.step('UP')
        self.assertEqual(reward, 1)
        self.assertFalse(done)

        # check that we get points and the episode ends when the block is on
        # top of the goal
        obs, reward, done, info = self.world.step('UP')
        self.assertEqual(reward, 1001)
        self.assertTrue(done)

    def test_actions(self):
        player = self.world.objects['agent']
        posn = player.position

        # check that actions do what we expect
        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(player.position, posn + (0, 1)))
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(player.position, posn + (1, 1)))
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(player.position, posn + (1, 0)))
        self.world.step('UP')
        self.assertTrue(np.array_equal(player.position, posn))
        


if __name__ == '__main__':
    unittest.main()
