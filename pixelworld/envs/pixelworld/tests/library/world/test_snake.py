import unittest

import numpy as np

from pixelworld.envs.pixelworld import agents, events, universe, library
from pixelworld.envs.pixelworld.tests import test_core
from pixelworld.envs.pixelworld.tests.library.world import test_library_world

snake = library.import_item('world', 'snake')

class TestSnake(test_library_world.TestLibraryWorld):
    world = None
    
    def __init__(self, *args, **kwargs):
        super(TestSnake, self).__init__(*args, **kwargs)

        self.world = kwargs.pop('world', 'snake')
        
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()

    def _test_solver(self):
        self.world = universe.create_world('snake', width=12, height=12)
        done = False
        head = self.world.objects['head']
        self.world.render()
        body_positions = dict()
        num_eaten = 0
        while not done:
            # check that we're the right length
            self.assertEqual(len(self.world.objects.find(name='body')), 5 + num_eaten)

            # check that food wasn't spawned on top of body
            for food in self.world.objects.find(name='food'):
                self.assertEqual(len(self.world.objects.find(name='body', position=food.position)), 0)

            # check that food wasn't spawned on top of other food
            for food in self.world.objects.find(name='food'):
                self.assertEqual(len(self.world.objects.find(name='food', position=food.position)), 1)

            # check that food wasn't spawned on top of the frame
            for food in self.world.objects.find(name='food'):
                self.assertTrue((1 <= food.position).all())
                self.assertTrue((food.position <= np.array([self.world.height - 2, self.world.width - 2])).all())

            # check that body segments don't move
            for birth in body_positions:
                body_segments = self.world.objects.find(name='body', birth=birth)
                self.assertTrue(len(body_segments) <= 1)
                if len(body_segments) == 1:
                    body = body_segments[0]
                    self.assertTrue(np.array_equal(body.position, body_positions[birth]),
                                    repr(birth) + repr(body.position) + repr(body_positions[birth]))

            # remember body positions for next time
            for body in self.world.objects.find(name='body'):
                body_positions[body.birth] = body.position

            # do the right thing
            if ((head.position[0] == self.world.height - 3 or head.position[0] == self.world.height - 2)
                    and head.position[1] == self.world.width - 3):
                obs, reward, done, info = self.world.step('NOOP')
                if reward > 0:
                    num_eaten += 1
                if done:
                    break
                obs, reward, done, info = self.world.step('UP')
                if reward > 0:
                    num_eaten += 1
            elif head.position[0] == 1 and head.position[1] == self.world.width - 2:
                obs, reward, done, info = self.world.step('LEFT')
                if reward > 0:
                    num_eaten += 1
            elif head.position[1] == self.world.width - 3 and head.position[0] != 1:
                obs, reward, done, info = self.world.step('DOWN')
                if reward > 0:
                    num_eaten += 1
                if done:
                    break
                obs, reward, done, info = self.world.step('LEFT')
                if reward > 0:
                    num_eaten += 1
            elif head.position[1] == 1:
                obs, reward, done, info = self.world.step('DOWN')
                if reward > 0:
                    num_eaten += 1
                if done:
                    break
                obs, reward, done, info = self.world.step('RIGHT')
                if reward > 0:
                    num_eaten += 1
            else:
                obs, reward, done, info = self.world.step('NOOP')
                if reward > 0:
                    num_eaten += 1

            self.world.render()


        self.assertTrue(done)

        # check that we never died
        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 0)

        # check that we won
        win_events = [x for x in self.world.events if isinstance(x, snake.WinEvent)]
        self.assertEqual(len(win_events), 1)

    def test_ouroborous(self):
        # run into ourselves
        obs, reward, done, info = self.world.step('UP')
        self.assertFalse(done)
        obs, reward, done, info = self.world.step('RIGHT')
        self.assertFalse(done)
        obs, reward, done, info = self.world.step('DOWN')
        self.assertEqual(reward, -1001)
        self.assertTrue(done)

        # check that we died exactly once
        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 1)

    def test_wall(self):
        # run into wall
        for i in xrange(9):
            obs, reward, done, info = self.world.step('NOOP')
            self.assertTrue(reward >= -1)
            self.assertFalse(done)
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1001)
        self.assertTrue(done)

        # check that we died exactly once
        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 1)

    def test_food(self):
        # food is stochastic
        self.world.seed(0)

        # spawn food in our path and check that we got a reward for eating it
        x = self.world.create_object(['food', dict(position=self.world.objects['head'].position + (0, -1))])
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 9)
        self.assertFalse(done)

        # check that food was removed
        self.assertIs(x._world, None)

    def test_actions(self):
        head = self.world.objects['head']
        posn = head.position

        # check that actions have expected consequences
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(head.position, posn + (0, -1)))

        # check that actions have expected consequences
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(head.position, posn + (1, -1)))

        # check that actions have expected consequences
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(head.position, posn + (1, -2)))

        # check that actions have expected consequences
        self.world.step('UP')
        self.assertTrue(np.array_equal(head.position, posn + (0, -2)))

        # check that actions have expected consequences
        self.world.step('UP')
        self.assertTrue(np.array_equal(head.position, posn + (-1, -2)))

        # check that actions have expected consequences
        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(head.position, posn + (-1, -1)))


if __name__ == '__main__':
    unittest.main()
