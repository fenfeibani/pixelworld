import unittest

import numpy as np

from pixelworld.envs.pixelworld import universe, events, agents, library
from pixelworld.envs.pixelworld.tests import test_core

sokoban = library.import_item('world', 'sokoban')

class TestSokoban(test_core.TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'sokoban')
        
        super(TestSokoban, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()
    
    def test_walls(self):
        # switch to provided screen
        screen = """
----
-* -
-  -
----
"""
        self.world.change_to_new_screen(screen)

        # remember position
        player = self.world.objects['sokoban_player']
        posn = player.position
        
        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(player.position, posn + (0, 1)))
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(player.position, posn + (1, 1)))
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(player.position, posn + (1, 0)))
        self.world.step('UP')
        self.assertTrue(np.array_equal(player.position, posn))

    def test_walls(self):
        # switch to provided screen
        screen = """
---
-*-
---
"""
        self.world.change_to_new_screen(screen)

        # remember position
        posn = self.world.objects['sokoban_player'].position

        # check that we cannot move through wall
        self.world.step('LEFT')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())

        # check that we cannot move through wall
        self.world.step('UP')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())

        # check that we cannot move through wall
        self.world.step('RIGHT')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())

        # check that we cannot move through wall
        self.world.step('DOWN')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())

    def test_boulders_movable(self):
        # switch to provided screen
        screen = """
-------
-     -
-  0  -
- 0*0 -
-  0  -
-     -
-------
"""
        self.world.change_to_new_screen(screen)

        # remember position
        posn = self.world.objects['sokoban_player'].position

        # check that we can move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (0, -1)))[0]
        boulder_posn = boulder.position
        self.world.step('LEFT')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn + (0, -1)).all())
        self.assertTrue((boulder.position == boulder_posn + (0, -1)).all())
        self.world.step('RIGHT')

        # check that we can move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (0, 1)))[0]
        boulder_posn = boulder.position
        self.world.step('RIGHT')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn + (0, 1)).all())
        self.assertTrue((boulder.position == boulder_posn + (0, 1)).all())
        self.world.step('LEFT')

        # check that we can move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (-1, 0)))[0]
        boulder_posn = boulder.position
        self.world.step('UP')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn + (-1, 0)).all())
        self.assertTrue((boulder.position == boulder_posn + (-1, 0)).all())
        self.world.step('DOWN')

        # check that we can move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (1, 0)))[0]
        boulder_posn = boulder.position
        self.world.step('DOWN')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn + (1, 0)).all())
        self.assertTrue((boulder.position == boulder_posn + (1, 0)).all())
        self.world.step('UP')

    def test_boulders_not_movable(self):
        # switch to provided screen
        screen = """
---------
-       -
-   0   -
-   0   -
- 00*00 -
-   0   -
-   0   -
-       -
---------
"""
        self.world.change_to_new_screen(screen)

        # remember position
        posn = self.world.objects['sokoban_player'].position

        # check that we cannot move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (0, -1)))[0]
        boulder_posn = boulder.position
        self.world.step('LEFT')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())
        self.assertTrue((boulder.position == boulder_posn).all())
        self.world.step('RIGHT')

        # check that we cannot move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (0, 1)))[0]
        boulder_posn = boulder.position
        self.world.step('RIGHT')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())
        self.assertTrue((boulder.position == boulder_posn).all())
        self.world.step('LEFT')

        # check that we cannot move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (-1, 0)))[0]
        boulder_posn = boulder.position
        self.world.step('UP')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())
        self.assertTrue((boulder.position == boulder_posn).all())
        self.world.step('DOWN')

        # check that we cannot move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (1, 0)))[0]
        boulder_posn = boulder.position
        self.world.step('DOWN')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())
        self.assertTrue((boulder.position == boulder_posn).all())
        self.world.step('UP')

    def test_boulders_not_movable2(self):
        # switch to provided screen
        screen = """
-----
- 0 -
-0*0-
- 0 -
-----
"""
        self.world.change_to_new_screen(screen)

        # remember position
        posn = self.world.objects['sokoban_player'].position

        # check that we cannot move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (0, -1)))[0]
        boulder_posn = boulder.position
        self.world.step('LEFT')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())
        self.assertTrue((boulder.position == boulder_posn).all())
        self.world.step('RIGHT')

        # check that we cannot move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (0, 1)))[0]
        boulder_posn = boulder.position
        self.world.step('RIGHT')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())
        self.assertTrue((boulder.position == boulder_posn).all())
        self.world.step('LEFT')

        # check that we cannot move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (-1, 0)))[0]
        boulder_posn = boulder.position
        self.world.step('UP')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())
        self.assertTrue((boulder.position == boulder_posn).all())
        self.world.step('DOWN')

        # check that we cannot move the boulder
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (1, 0)))[0]
        boulder_posn = boulder.position
        self.world.step('DOWN')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn).all())
        self.assertTrue((boulder.position == boulder_posn).all())
        self.world.step('UP')

    def test_boulders_fill_holes(self):
        # switch to provided screen
        screen = """
-------
-  ^  -
-  0  -
-^0*0^-
-  0  -
-  ^  -
-------
"""
        self.world.change_to_new_screen(screen)

        # remember position
        posn = self.world.objects['sokoban_player'].position
        num_events = len(self.world.events)

        # check that we can move the boulder and it fills a hole and that looks
        # like it should
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (0, -1)))[0]
        boulder_posn = boulder.position
        self.world.step('LEFT')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn + (0, -1)).all())
        self.assertTrue((boulder.position == boulder_posn + (0, -1)).all())
        self.assertTrue(len(self.world.events) == num_events + 2)
        self.assertTrue(isinstance(self.world.events[-2], events.PushEvent))
        self.assertTrue(isinstance(self.world.events[-1], sokoban.HoleFillEvent))
        self.assertTrue(not boulder.visible)
        self.assertEqual(self.world.state[tuple(boulder.state_index.tolist())], 0)
        self.world.step('RIGHT')

        # check that we can move the boulder and it fills a hole and that looks
        # like it should
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (0, 1)))[0]
        boulder_posn = boulder.position
        self.world.step('RIGHT')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn + (0, 1)).all())
        self.assertTrue((boulder.position == boulder_posn + (0, 1)).all())
        self.assertTrue(len(self.world.events) == num_events + 5)
        self.assertTrue(isinstance(self.world.events[-2], events.PushEvent))
        self.assertTrue(isinstance(self.world.events[-1], sokoban.HoleFillEvent))
        self.assertTrue(not boulder.visible)
        self.assertEqual(self.world.state[tuple(boulder.state_index.tolist())], 0)
        self.world.step('LEFT')

        # check that we can move the boulder and it fills a hole and that looks
        # like it should
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (-1, 0)))[0]
        boulder_posn = boulder.position
        self.world.step('UP')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn + (-1, 0)).all())
        self.assertTrue((boulder.position == boulder_posn + (-1, 0)).all())
        self.assertTrue(len(self.world.events) == num_events + 8)
        self.assertTrue(isinstance(self.world.events[-2], events.PushEvent))
        self.assertTrue(isinstance(self.world.events[-1], sokoban.HoleFillEvent))
        self.assertTrue(not boulder.visible)
        self.assertEqual(self.world.state[tuple(boulder.state_index.tolist())], 0)
        self.world.step('DOWN')

        # check that we can move the boulder and it fills a hole and that looks
        # like it should
        boulder = self.world.objects.find(position=(self.world.objects['sokoban_player'].position + (1, 0)))[0]
        boulder_posn = boulder.position
        self.world.step('DOWN')
        self.assertTrue((self.world.objects['sokoban_player'].position == posn + (1, 0)).all())
        self.assertTrue((boulder.position == boulder_posn + (1, 0)).all())
        self.assertTrue(len(self.world.events) == num_events + 11)
        self.assertTrue(isinstance(self.world.events[-2], events.PushEvent))
        self.assertTrue(isinstance(self.world.events[-1], sokoban.HoleFillEvent))
        self.assertTrue(not boulder.visible)
        self.assertEqual(self.world.state[tuple(boulder.state_index.tolist())], 0)
        self.world.step('UP')

    def test_holes_kill_player(self):
        # switch to provided screen
        screen = """
-----
- ^ -
- * -
-----
"""
        self.world.change_to_new_screen(screen)

        num_events = len(self.world.events)

        obs, reward, done, info = self.world.step('UP')
        self.assertEqual(reward, -1001)
        self.assertTrue(done)

        player = self.world.objects['sokoban_player']
        self.assertEqual(player.visible, False)
        self.assertEqual(self.world.state[tuple(player.state_index.tolist())], 4)
        self.assertTrue(isinstance(self.world.events[-1], events.KillEvent))

    def test_load_screen(self):
        # check that we can load every screen
        screen_ids = self.world._screens.keys()
        for id in screen_ids:
            self.world.screen = id

    def test_give_up(self):
        # check that we can give up
        obs, reward, done, info = self.world.step('GIVE_UP')
        self.assertEqual(reward, -1001)
        self.assertTrue(done)
        self.assertTrue(isinstance(self.world.events[-1], events.KillEvent))
        self.assertEqual(self.world.events[-1].reason, 'forfeit')

    def test_exit(self):
        # define two new screens
        screen = """
-----
- X -
- * -
-----
"""
        idx = self.world.change_to_new_screen(screen)

        screen2 = """
-----
-   -
- *X-
-----
"""
        idx2 = self.world.change_to_new_screen(screen2)

        # switch to the first screen we defined
        self.world.screen = idx

        # check that reaching the exit gives us a reward and causes a screen
        # change event
        obs, reward, done, info = self.world.step('UP')
        self.assertEqual(reward, 999)
        self.assertFalse(done)
        self.assertTrue(isinstance(self.world.events[-1], events.ChangeScreenEvent))

        # check that reaching the exit gives us a reward and terminates the
        # episode
        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, 999)
        self.assertTrue(done)
        self.assertTrue(isinstance(self.world.events[-1], sokoban.WinEvent))

    def test_energy(self):
        self.world.agent = agents.RandomAgent(self.world)
        energy = self.world.energy
        for _ in xrange(100):
            self.world.step()
            self.assertEqual(self.world.energy, energy)

if __name__ == '__main__':
    unittest.main()
