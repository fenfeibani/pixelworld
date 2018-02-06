import unittest

import numpy as np

from pixelworld.envs.pixelworld import agents, universe, utils
from pixelworld.envs.pixelworld.tests import test_core

class TestTouchBlock(test_core.TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'touchblock')
        
        super(TestTouchBlock, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()

    def test_no_touching(self):
        """Check that we can take random actions and get the reward/termination we
        expect."""
        #replace the world agent with a random agent
        self.world.agent = agents.RandomAgent(self.world)
        
        agent = self.world.objects['agent']
        block = self.world.objects['block']

        for _ in xrange(100):
            obs, reward, done, info = self.world.step()

            #check that we got a reward if we're touching the block and not
            #otherwise
            if np.sum(utils.roundup(np.abs(agent.position - block.position))) <= 1:
                self.assertEqual(reward, 999)
                self.assertTrue(done)
                break
            else:
                self.assertEqual(reward, -1)
                self.assertFalse(done)

    def test_touching(self):
        """Check that we can take optimal actions and get the reward/termination we
        expect."""
        agent = self.world.objects['agent']
        block = self.world.objects['block']

        for _ in xrange(100):
            #pick an optimal action
            if agent.position[0] < block.position[0]:
                action = 'DOWN'
            elif agent.position[0] > block.position[0]:
                action = 'UP'
            elif agent.position[1] < block.position[1]:
                action = 'RIGHT'
            elif agent.position[1] > block.position[1]:
                action = 'LEFT'

            #check that we got a reward if we're touching the block and not
            #otherwise
            obs, reward, done, info = self.world.step(action)
            if np.sum(utils.roundup(np.abs(agent.position - block.position))) <= 1:
                self.assertEqual(reward, 999)
                self.assertTrue(done)
                break
            else:
                self.assertEqual(reward, -1)
                self.assertFalse(done)


if __name__ == '__main__':
    unittest.main()
