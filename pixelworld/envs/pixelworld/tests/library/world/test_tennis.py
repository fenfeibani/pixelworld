import unittest

import numpy as np

from pixelworld.envs.pixelworld import agents, universe
from pixelworld.envs.pixelworld.tests import test_core
import pixelworld.envs.pixelworld.library.helpers as h
from pixelworld.envs.pixelworld.tests.library.world import test_library_world

class TestTennis(test_library_world.TestLibraryWorld):
    world = None
    
    def __init__(self, *args, **kwargs):
        super(TestTennis, self).__init__(*args, **kwargs)

        self.world = kwargs.pop('world', 'tennis')
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()

    def test_perfect_enemy(self):
        ball = self.world.objects['ball']
        self.world.objects['paddle'].add_attribute('tracks_ball', 1.0)
        self.world.objects['enemy'].tracks_ball = 1.0

        # check that nobody ever misses a point
        for _ in xrange(100):
            obs, reward, done, info = self.world.step()
            self.assertTrue(0 <= ball.position[1] <= self.world.width - 1)
            self.assertFalse(done)
            self.assertEqual(reward, 0)

    def test_enemy_loses(self):
        self.world.objects['enemy'].position = (4, self.world.width - 1)

        # put ball where enemy can't get to it
        ball = self.world.objects['ball']
        ball.position = (self.world.height-3, self.world.width - 2)
        ball.velocity = (1.01, 1)

        self.world.step('NOOP')
        obs, reward, done, info = self.world.step('NOOP')

        # check that we got a reward and the episode did not terminate and the
        # ball reset
        self.assertEqual(reward, 1000)
        self.assertFalse(done)
        self.assertTrue(np.array_equal(ball.position, (self.world.height // 2, 1)))

    def test_player_loses(self):
        self.world.objects['paddle'].position = (4, 0)

        # put ball where the player can't get to it
        ball = self.world.objects['ball']
        ball.position = (self.world.height-3, 1)
        ball.velocity = (1.01, -1)

        self.world.step('NOOP')
        obs, reward, done, info = self.world.step('NOOP')

        # check that we got a penalty and the episode did not terminate and the
        # ball reset
        self.assertEqual(reward, -1000)
        self.assertFalse(done)
        self.assertTrue(np.array_equal(ball.position, (self.world.height // 2, 1)))

    def test_bounce(self):
        ball = self.world.objects['ball']
        self.world.objects['paddle'].position = (4, 0)

        # put ball on a collision course with the frame
        ball.position = (self.world.height-3, 1)
        ball.velocity = (1.01, 1)

        # check that we bounce off the frame as we expect
        self.world.step('NOOP')
        self.assertTrue((np.abs(ball.position - (17.99, 2)) < 0.01).all())
        self.assertTrue((np.abs(ball.velocity - (-1.01, 1)) < 0.01).all())

        self.world.step('NOOP')
        self.assertTrue((np.abs(ball.position - (16.98, 3)) < 0.01).all())
        self.assertTrue((np.abs(ball.velocity - (-1.01, 1)) < 0.01).all())

    def test_actions(self):
        player = self.world.objects['paddle']
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
