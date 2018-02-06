import unittest

import numpy as np

from pixelworld.envs.pixelworld import agents, universe, objects, library
from pixelworld.envs.pixelworld.tests.library.world import test_library_world
import pixelworld.envs.pixelworld.library.helpers as h

block_breaker = library.import_item('world', 'block_breaker')

class TestBlockBreaker(test_library_world.TestLibraryWorld):
    world = None
    
    def __init__(self, *args, **kwargs):
        super(TestBlockBreaker, self).__init__(*args, **kwargs)

        self.world = kwargs.pop('world', 'block_breaker')
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()

    def test_bricks_disappear_and_goal_achieved(self):
        # create the obejcts we want
        objects, height, width = h.world.screen("""
WWWW
W BW
W* W
W  W
""", block_breaker.legend)

        self.world = universe.create_world('block_breaker', objects=objects, height=height, width=width)

        # point the ball up and to the right
        self.world.objects['ball'].velocity = (-1.01, 1)

        # get the brick
        brick = self.world.objects['brick']

        # get the goal
        self.assertEqual(len(self.world.goals), 1)
        goal = self.world.goals['bricks_destroyed']

        # check that goal is not achieved
        self.assertFalse(goal.is_achieved())

        # step
        obs, reward, done, info = self.world.step('NOOP')

        # check that we got a reward (100 for brick, 1000 for winning, -1 for
        # stepping) and that the episode terminated
        self.assertEqual(reward, 1099)
        self.assertTrue(done)

        # check that brick disappeared 
        self.assertFalse(brick.visible)

        # check that goal is achieved
        self.assertTrue(goal.is_achieved())

    def test_ball_leaves_screen(self):
        # create the obejcts we want
        objects, height, width = h.world.screen("""
WWWW
W BW
W* W
""", block_breaker.legend)

        self.world = universe.create_world('block_breaker', objects=objects, height=height, width=width)

        # point the ball down and to the right
        self.world.objects['ball'].velocity = (1.01, 1)

        # get the goal
        self.assertEqual(len(self.world.goals), 1)
        goal = self.world.goals['bricks_destroyed']

        # check that goal is not achieved
        self.assertFalse(goal.is_achieved())

        # step
        obs, reward, done, info = self.world.step('NOOP')

        # check that we got a penalty and that the episode terminated
        self.assertEqual(reward, -1001)
        self.assertTrue(done)

        # check that goal is not achieved
        self.assertFalse(goal.is_achieved())

    def test_paddle_leaves_screen(self):
        # create the obejcts we want
        objects, height, width = h.world.screen("""
WWWW
W BW
W* W
W  W
WP W
""", block_breaker.legend)

        self.world = universe.create_world('block_breaker', objects=objects, height=height, width=width)

        # point the ball down and to the right
        self.world.objects['ball'].velocity = (1.01, 1)

        # get the goal
        self.assertEqual(len(self.world.goals), 1)
        goal = self.world.goals['bricks_destroyed']

        # check that goal is not achieved
        self.assertFalse(goal.is_achieved())

        # step
        obs, reward, done, info = self.world.step('DOWN')

        # check that we got a penalty and that the episode terminated
        self.assertEqual(reward, -1001)
        self.assertTrue(done)

        # check that goal is not achieved
        self.assertFalse(goal.is_achieved())

    def test_bounce(self):
        # create the obejcts we want
        objects, height, width = h.world.screen("""
WWWW
W  W
W*BW
WBBW
WPPW
""", block_breaker.legend)

        self.world = universe.create_world('block_breaker', objects=objects, height=height, width=width)

        ball = self.world.objects['ball']

        # point the ball up and to the right
        ball.velocity = (-1.01, 1)

        self.assertTrue(np.array_equal(ball.state_index, (2, 1)), repr(ball.state_index))

        # step and check ball position
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(np.array_equal(ball.state_index, (1, 2)))

        # step and check ball position
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(np.array_equal(ball.state_index, (2, 1)))

        # step and check ball position
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(np.array_equal(ball.state_index, (1, 2)))

        # step and check ball position
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(np.array_equal(ball.state_index, (2, 1)))

        # step and check ball position
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(np.array_equal(ball.state_index, (3, 2)))

        # step and check ball position
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(np.array_equal(ball.state_index, (2, 1)))

        # step and check ball position
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(np.array_equal(ball.state_index, (1, 2)))

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
        
    def test_collide_callback(self):
        log = []
        class FooCollideObjectAttribute(block_breaker.CollisionListeningObjectAttribute):
            _step_after = ['velocity']
            def _process_collision(self, evt, t, dt, agent_id, action):
                log.append(evt)

        class CollisionObject(objects.BasicObject):
            _attributes = ['foo_collide']

        self.world.create_object(['collision', dict(position=(10, 10), velocity=(0, 1))])
        self.world.create_object(['wall', dict(position=(10, 12))])
        self.world.create_object(['wall', dict(position=(10, 9))])

        self.world.step('NOOP')
        self.world.step('NOOP')
        self.assertEqual(len(log), 1)
        self.assertIn(self.world.objects['collision']._id, log[0].indices)
        self.world.step('NOOP')
        self.assertEqual(len(log), 2)
        self.assertIn(self.world.objects['collision']._id, log[1].indices)
        self.world.step('NOOP')
        self.assertEqual(len(log), 3)
        self.assertIn(self.world.objects['collision']._id, log[2].indices)


if __name__ == '__main__':
    unittest.main()
