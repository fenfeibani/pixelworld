import unittest

import numpy as np

from pixelworld.envs.pixelworld import universe, events, agents
from pixelworld.envs.pixelworld.tests.library.world import test_library_world
import pixelworld.envs.pixelworld.library.helpers as h

legend = {'*': 'player',
          'C': 'crash',
          'R': 'robot',
          'W': 'wall',
          }


class TestRobots(test_library_world.TestLibraryWorld):
    world = None
    
    def __init__(self, *args, **kwargs):
        super(TestRobots, self).__init__(*args, **kwargs)

        self.world = kwargs.pop('world', 'robots')
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()

    def test_actions(self):
        objects, width, height = h.world.screen("""
WWWWWW
W    W
W *  W
W    W
W    W
WWWWWW
""", legend)

        self.world = universe.create_world('robots', objects=objects, width=width, height=height)

        player = self.world.objects['player']
        posn = player.position

        # check that actions have expect effects
        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(player.position, posn + (0, 1)))
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(player.position, posn + (1, 1)))
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(player.position, posn + (1, 0)))
        self.world.step('UP')
        self.assertTrue(np.array_equal(player.position, posn))

    def test_teleport(self):
        objects, width, height = h.world.screen("""
WWWWWW
W  R W
W *  W
W    W
W    W
WWWWWW
""", legend)

        self.world = universe.create_world('robots', objects=objects, width=width, height=height)

        player = self.world.objects['player']

        # check that we eventually teleport into a robot (4 possible locations,
        # so 40 tries seems good enough, about 1e-5 probability of failure)
        for _ in xrange(40):
            obs, reward, done, info = self.world.step('TELEPORT')

            # check that we never teleport outside viewable region
            self.assertTrue(0 <= player.position[0] < self.world.height)
            self.assertTrue(0 <= player.position[1] < self.world.width)
            
        self.assertTrue(done)

    def test_sometimes_robots_spawn_same(self):
        # check that sometimes robots spawn on top of each other, leading to
        # their deaths
        robots_died = False
        for _ in xrange(100):
            self.world = universe.create_world('robots', width=8, height=8)
            
            if not all(robot.alive for robot in self.world.objects['robot']):
                robots_died = True
                break

        self.assertTrue(robots_died)
            

    def test_robots_get_closer(self):
        player = self.world.objects['player']
        robots = self.world.objects['robot']
        dists = [np.sum(np.abs(player.position - robot.position)) for robot in robots]

        for _ in xrange(10):
            alive_before_step = [robot.alive for robot in robots]
            obs, reward, done, info = self.world.step('NOOP')

            new_dists = [np.sum(np.abs(player.position - robot.position)) for robot in robots]

            # check that robots get one or two steps closer if the player isn't
            # dead yet and the robot isn't dead yet
            for d, nd, robot, alive in zip(dists, new_dists, robots, alive_before_step):
                self.assertTrue(nd <= d <= nd + 2)
                if not alive:
                    self.assertTrue(nd == d)
                elif not done and d > 0:
                    self.assertTrue(nd <= d - 1, repr((nd, d)))

            dists = new_dists

    def test_player_moves_before_robots(self):
        self.world.remove_objects(self.world.objects)

        objects, _, _ = h.world.screen("""
R* 
""", legend)
        self.world.create_objects(objects)

        player, robot = self.world.objects['player'], self.world.objects['robot']

        # check that the player always survives by moving right, because the
        # player moves before the robot
        for _ in xrange(10):
            obs, reward, done, info = self.world.step('RIGHT')
            self.assertEqual(reward, -1)
            self.assertFalse(done)

            # check that the robot follows close behind
            self.assertTrue(np.array_equal(player.position + (0, -1), robot.position))

    def test_death(self):
        self.world.remove_objects(self.world.objects)

        objects, _, _ = h.world.screen("""
 R    
 * 
""", legend)
        self.world.create_objects(objects)

        # check that the player dies when the robot catches them
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1001)
        self.assertTrue(done)
        self.assertTrue(isinstance(self.world.events[-1], events.KillEvent))
        self.assertEqual(self.world.events[-1].reason, 'robot')

    def test_three_robots(self):
        self.world.remove_objects(self.world.objects)

        objects, _, _ = h.world.screen("""
RRR
   
 * 
""", legend)
        self.world.create_objects(objects)

        # check that three robots all die in the same position when they crash
        # into each other
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 299)
        self.assertTrue(done)
        player = self.world.objects['player']
        for robot in self.world.objects['robot']:
            self.assertTrue(np.array_equal(player.position + (-1, 0), robot.position))
            self.assertTrue(not robot.visible)
            self.assertTrue(not robot.alive)
            self.assertTrue(not hasattr(robot, 'chases_player'))

        # check that only one crash is generated
        self.assertEqual(len(self.world.objects.find(name='crash')), 1)

    def test_two_robots(self):
        self.world.remove_objects(self.world.objects)

        objects, _, _ = h.world.screen("""
RR 
   
 * 
""", legend)
        self.world.create_objects(objects)
        
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 199)
        self.assertTrue(done)

        # check that two robots die in the same position when they crash into
        # each other
        player = self.world.objects['player']
        for robot in self.world.objects['robot']:
            self.assertTrue(np.array_equal(player.position + (-1, 0), robot.position))
            self.assertTrue(not robot.visible)
            self.assertTrue(not robot.alive)
            self.assertTrue(not hasattr(robot, 'chases_player'))

        self.assertEqual(len(self.world.objects.find(name='crash')), 1)

    def test_noop(self):
        self.world.remove_objects(self.world.objects)

        objects, _, _ = h.world.screen("""
RRRRRRR
 RRRRR 
  CCC  
   *   
""", legend)
        self.world.create_objects(objects)

        # check that we can win by doing nothing for two steps
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 799)
        self.assertFalse(done)
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 399)
        self.assertTrue(done)

        # check that all the robots are dead and that they died where we expect
        player = self.world.objects['player']
        for robot in self.world.objects['robot']:
            self.assertTrue(np.array_equal(player.position + (-1, 0), robot.position) or
                            np.array_equal(player.position + (-1, -1), robot.position) or
                            np.array_equal(player.position + (-1, 1), robot.position) or
                            np.array_equal(player.position + (-2, 0), robot.position))
            self.assertTrue(not robot.visible)
            self.assertTrue(not robot.alive)
            self.assertTrue(not hasattr(robot, 'chases_player'))

        # check that there are only as many crashes as we expect
        self.assertEqual(len(self.world.objects.find(name='crash')), 4)

    def test_crash(self):
        self.world.remove_objects(self.world.objects)

        objects, _, _ = h.world.screen("""
 R  
 C 
 *
""", legend)
        self.world.create_objects(objects)

        # check that robot dies by running into crash
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 99)
        self.assertTrue(done)
        player = self.world.objects['player']
        robot = self.world.objects['robot']
        self.assertTrue(np.array_equal(player.position + (-1, 0), robot.position))
        self.assertTrue(not robot.visible)
        self.assertTrue(not robot.alive)
        self.assertTrue(not hasattr(robot, 'chases_player'))

        self.assertEqual(len(self.world.objects.find(name='crash')), 1)


if __name__ == '__main__':
    unittest.main()
