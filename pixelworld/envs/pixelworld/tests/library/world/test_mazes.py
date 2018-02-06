import unittest

import numpy as np
import networkx as nx

from pixelworld.envs.pixelworld import agents, core, universe
from pixelworld.envs.pixelworld.tests import test_core
from pixelworld.envs.pixelworld.tests.library.world import test_library_world
import pixelworld.envs.pixelworld.library.helpers as h


class TestMazes(test_library_world.TestLibraryWorld):
    world = None
    
    def __init__(self, *args, **kwargs):
        super(TestMazes, self).__init__(*args, **kwargs)

        self.world = kwargs.pop('world', 'mazes')
    
    def tearDown(self):
        if isinstance(self.world, core.PixelWorld):
            self.world.end()

    def check_solvable(self):
        # create the grid graph
        gf = nx.Graph()
        for r in xrange(self.world.height):
            for c in xrange(self.world.width):
                if c < self.world.width - 1:
                    gf.add_edge((r, c), (r, c + 1))
                if r < self.world.height - 1:
                    gf.add_edge((r, c), (r + 1, c))

        # remove nodes that are wall pixels
        for wall in self.world.objects['wall']:
            gf.remove_node(tuple(wall.state_index.tolist()))

        # find player and goal
        player = self.world.objects['self']
        goal = self.world.objects['goal']
        player_node = tuple(player.state_index.tolist())
        goal_node = tuple(goal.state_index.tolist())

        # check for existence of path
        self.assertTrue(nx.has_path(gf, player_node, goal_node))

    def test_squares_solvable(self):
        self.world = universe.create_world('mazes', seed=0, method='squares')

        self.check_solvable()

    def test_subdivide_solvable(self):
        self.world = universe.create_world('mazes', seed=0, method='squares')

        self.check_solvable()

    def test_kruskal_solvable(self):
        self.world = universe.create_world('mazes', seed=0, method='squares')

        self.check_solvable()

    def test_longest_solvable(self):
        self.world = universe.create_world('mazes', seed=0, method='squares')

        self.check_solvable()

    def test_walls_unpushable_and_goal_rewards(self):
        self.world = universe.create_world('mazes', seed=0, method='subdivide', width=7, height=7)

        self.world.remove_objects(self.world.objects['wall'])
        self.world.remove_objects(self.world.objects['goal'])

        objects, _, _ = h.world.screen("""
WWWWWWW
W W   W
W W   W
W    WW
W  W  W
W  W GW
WWWWWWW
""", self.world._legend)

        self.world.create_objects(objects)

        player = self.world.objects['self']
        posn = player.position

        # check that we can't push the walls
        self.world.step('UP')
        self.assertTrue(np.array_equal(player.position, posn))
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(player.position, posn))
        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(player.position, posn))
        
        # solve the maze
        self.world.step('DOWN')
        self.world.step('DOWN')
        self.world.step('RIGHT')
        self.world.step('RIGHT')
        self.world.step('RIGHT')
        self.world.step('DOWN')
        self.world.step('DOWN')
        obs, reward, done, info = self.world.step('RIGHT')

        # check that the episode terminated and we got a reward
        self.assertTrue(done)
        self.assertEqual(reward, 999)

    def test_actions(self):
        self.world = universe.create_world('mazes', seed=0, method='subdivide', width=7, height=7)

        self.world.remove_objects(self.world.objects['wall'])

        objects, _, _ = h.world.screen("""
WWWWWWW
W  W  W
W  W  W
W    WW
W  W  W
W  W  W
WWWWWWW
""", self.world._legend)

        self.world.create_objects(objects)

        player = self.world.objects['self']
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
        
    def test_energy(self):
        self.world = universe.create_world('mazes', seed=0, method='subdivide', width=7, height=7)

        self.world.agent = agents.RandomAgent(self.world)
        energy = self.world.energy
        for _ in xrange(100):
            self.world.step()
            self.assertEqual(self.world.energy, energy)


if __name__ == '__main__':
    unittest.main()
