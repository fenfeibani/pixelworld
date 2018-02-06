import unittest

from pixelworld.envs.pixelworld import universe, core, agents
from pixelworld.envs.pixelworld.tests import test_core


class TestRandomAgent(test_core.TestAgent):
    def setUp(self):
        class MyAgent(core.Agent):
            _default_action = 'LEFT'
        
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world, 
                                               agent='random')

    def test_not_noop(self):
        """Check that a random agent doesn't always do NOOP"""
        # replace the world agent with a random agent
        self.world.agent = agents.RandomAgent(self.world)

        # check that random agent doesn't always do NOOP
        actions = [self.world.agent._get_action(self.world.last_observation, self.world.tools)
                   for _ in xrange(40)]
        self.assertTrue(any(x != 'NOOP' for x in actions))

    def test_restricted_actions(self):
        """Check that a random agent respects allowed_actions list.""" 
        # replace the world agent with a random agent with restricted actions
        self.world.agent = agents.RandomAgent(self.world, allowed_actions=['LEFT', 'RIGHT'])

        # check that agent always produces allowed actions
        for _ in xrange(40):
            action = self.world.agent.get_action(self.world.last_observation, self.world.tools)
            self.assertTrue(action in ['LEFT', 'RIGHT'])


if __name__ == '__main__':
    unittest.main()

