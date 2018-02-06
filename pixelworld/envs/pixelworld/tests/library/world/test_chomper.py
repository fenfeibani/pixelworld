import unittest

import numpy as np

from pixelworld.envs.pixelworld import agents, universe, library
from pixelworld.envs.pixelworld.tests import test_core
import pixelworld.envs.pixelworld.library.helpers as h
from pixelworld.envs.pixelworld.tests.library.world import test_library_world

chomper = library.import_item('world', 'chomper')

class TestChomper(test_library_world.TestLibraryWorld):
    world = None
    
    def __init__(self, *args, **kwargs):
        super(TestChomper, self).__init__(*args, **kwargs)

        self.world = kwargs.pop('world', 'chomper')
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()

    def test_loop(self):
        """Check that ghost goes around a loop when there are no branching choices"""
        objects, height, width = h.world.screen("""
XXXXXXXXX
X    BX*X
X XXX XXX
X     XXX
XXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        blinky = self.world.objects['blinky']
        blinky.orientation = 1 #UP
        self.assertTrue(np.array_equal(blinky.position, (1, 5)))

        # the squares of the track
        track = ([(1, x) for x in xrange(5, 0, -1)] +
                 [(y, 1) for y in xrange(2, 4)] +
                 [(3, x) for x in xrange(2, 6)] +
                 [(2, 5)])

        # go around the track two times
        for _ in xrange(2):
            for t in track:
                # check that Blinky is where we expect
                self.assertTrue(np.array_equal(blinky.position, t))
                obs, reward, done, info = self.world.step('NOOP')

    def test_chomper_wall(self):
        """Check that Chomper can't go through walls"""
        objects, height, width = h.world.screen("""
XXX
X*X
XXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        player = self.world.objects['chomper_player']
        posn = player.position

        for action in ['LEFT', 'UP', 'RIGHT', 'DOWN']:
            obs, reward, done, info = self.world.step(action)

            # check that we didn't move
            self.assertTrue(np.array_equal(player.position, posn))

    def test_ghost_wall(self):
        """Test that ghosts cannot go through walls"""
        objects, height, width = h.world.screen("""
XXX*
XBXX
XXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        blinky = self.world.objects['blinky']
        posn = blinky.position

        for _ in xrange(10):
            obs, reward, done, info = self.world.step('NOOP')

            # check that blinky didn't move
            self.assertTrue(np.array_equal(blinky.position, posn))

    def test_chomper_cannot_go_through_ghost(self):
        """Test that Chomper cannot go through a ghost"""
        objects, height, width = h.world.screen("""
XXXX
XB*X
XXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        # point blinky right
        blinky = self.world.objects['blinky']
        blinky.orientation = 2

        obs, reward, done, info = self.world.step('LEFT')

        self.assertTrue(done)
        self.assertEqual(reward, -1001)

    def test_ghost_cannot_go_through_chomper(self):
        """Test that a frightened ghost cannot go through Chomper"""
        objects, height, width = h.world.screen("""
XXXX
XB*X
XXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        # give player the powerup
        player = self.world.objects['chomper_player']
        player.powerup = 10

        # point blinky right
        blinky = self.world.objects['blinky']
        blinky.orientation = 2

        obs, reward, done, info = self.world.step('LEFT')

        self.assertFalse(done)
        self.assertEqual(reward, 199)

    def test_power_pellet(self):
        """Test that power pellets power us up, give us points, and that powerup
        decrements over time."""
        objects, height, width = h.world.screen("""
XXXXXXXXXXX
X*O       X
XXXXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        player = self.world.objects['chomper_player']

        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, 49)
        self.assertTrue(done)

        for t in xrange(11):
            # check that powerup decrements
            self.assertTrue(player.powerup, 10 - t)
            obs, reward, done, info = self.world.step('RIGHT')

    def test_eating_score(self):
        """Test that eating things gives us points"""
        objects, height, width = h.world.screen("""
XXXXXXXXXXX
X*O.......X
XXXXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, 49)
        self.assertFalse(done)

        for t in xrange(7):
            obs, reward, done, info = self.world.step('RIGHT')
            self.assertEqual(reward, 9)
            self.assertEqual(done, t==6)

    def test_ghost_score(self):
        """Test that eating ghosts gives us the right number of points"""
        objects, height, width = h.world.screen("""
XXXXXXXXXXXXXX
X* B  P  I  CX
XXXXXXXXXXXXXX
X            X
XXXXXXXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        # ghosts are stochastic when chomper is powered up
        self.world.seed(0)

        player = self.world.objects['chomper_player']
        blinky = self.world.objects['blinky']
        pinky = self.world.objects['pinky']
        inky = self.world.objects['inky']
        clyde = self.world.objects['clyde']

        # give the player the powerup
        player.powerup = 10

        # point ghosts left
        blinky.orientation = 0
        pinky.orientation = 0
        inky.orientation = 0
        clyde.orientation = 0

        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, 199)
        self.assertFalse(done)

        blinky.position = (3, 1)

        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)

        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, 399)
        self.assertFalse(done)

        pinky.position = (3, 3)

        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, 799)
        self.assertFalse(done)

        inky.position = (3, 5)

        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)

        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)

        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, 1599)
        self.assertFalse(done)

    def test_ghosts_reverse_on_mode_change(self):
        """Test that ghosts reverse direction when the mode changes"""
        objects, height, width = h.world.screen("""
XXXXXXXXXXXXXXX
X  B  P  I  C X
XXXXXXXXXXXXXXX
X*            X
XXXXXXXXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        player = self.world.objects['chomper_player']
        blinky = self.world.objects['blinky']
        pinky = self.world.objects['pinky']
        inky = self.world.objects['inky']
        clyde = self.world.objects['clyde']

        # point ghosts left
        blinky.orientation = 0
        pinky.orientation = 0
        inky.orientation = 0
        clyde.orientation = 0

        self.world.time = 29

        obs, reward, done, info = self.world.step('NOOP')

        self.assertEqual(blinky.orientation, 2)
        self.assertEqual(pinky.orientation, 2)
        self.assertEqual(inky.orientation, 2)
        self.assertEqual(clyde.orientation, 2)

    def test_ghosts_reverse_on_pellet(self):
        """Test that ghosts reverse direction when Chomper eats a power pellet"""
        objects, height, width = h.world.screen("""
XXXXXXXXXXXXXX
X  B  P  I  CX
XXXXXXXXXXXXXX
X*O          X
XXXXXXXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        player = self.world.objects['chomper_player']
        blinky = self.world.objects['blinky']
        pinky = self.world.objects['pinky']
        inky = self.world.objects['inky']
        clyde = self.world.objects['clyde']

        # point ghosts left
        blinky.orientation = 0
        pinky.orientation = 0
        inky.orientation = 0
        clyde.orientation = 0

        obs, reward, done, info = self.world.step('RIGHT')

        self.assertEqual(blinky.orientation, 2)
        self.assertEqual(pinky.orientation, 2)
        self.assertEqual(inky.orientation, 2)
        self.assertEqual(clyde.orientation, 2)

    def test_teleport(self):
        """Test that the teleporter teleports Chomper"""
        objects, height, width = h.world.screen("""
XXXXXXXXXXXXXX
T*           T
XXXXXXXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        player = self.world.objects['chomper_player']

        self.world.step('LEFT')
        
        self.assertTrue(np.array_equal(player.position, (1, 13)), repr(player.position))

        self.world.step('LEFT')
        self.world.step('RIGHT')

        self.assertTrue(np.array_equal(player.position, (1, 0)), repr(player.position))

    def test_teleport_ghost(self):
        """Test that the teleporter teleports ghosts"""
        objects, height, width = h.world.screen("""
XXXXXXXXXXXXXX
TB           T
XXXXXXXXXXXXXX
X*XXXXXXXXXXXX
XXXXXXXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        blinky = self.world.objects['blinky']
        blinky.orientation = 0

        self.world.step('NOOP')
        
        self.assertTrue(np.array_equal(blinky.position, (1, 13)), repr(blinky.position))

        self.world.step('NOOP')
        blinky.orientation = 2
        self.world.step('NOOP')

        self.assertTrue(np.array_equal(blinky.position, (1, 0)), repr(blinky.position))

    def test_blinky_choice(self):
        """Test that Blinky always makes the right choice when turning"""
        objects, height, width = h.world.screen("""
XXXXXXXXXXXXXX
X            X
XXXXXXXBXXXXXX
X           *X
XXXXXXXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        player = self.world.objects['chomper_player']
        player.orientation = 0 # left

        blinky = self.world.objects['blinky']
        blinky.orientation = 3 # down

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(blinky.position, (3, 7)), repr(blinky.position))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(blinky.position, (3, 8)), repr(blinky.position))

        blinky.position = blinky.initial_position
        blinky.orientation = 1 # up, so blinky will go down after flip

        self.world.time = 29        

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(blinky.position, (3, 7)), repr(blinky.position))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(blinky.position, (3, 8)), repr(blinky.position))

    def test_pinky_choice(self):
        """Test that Pinky always makes the right choice when turning"""
        objects, height, width = h.world.screen("""
XXXXXXXXXXXXXX
X            X
XXXXXXXPXXXXXX
X           *X
XXXXXXXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        player = self.world.objects['chomper_player']
        player.orientation = 0 # left

        pinky = self.world.objects['pinky']
        pinky.orientation = 3 # down

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(pinky.position, (3, 7)), repr(pinky.position))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(pinky.position, (3, 6)), repr(pinky.position))

        pinky.position = pinky.initial_position
        pinky.orientation = 1 # up, so pinky will go down after flip

        self.world.time = 29        

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(pinky.position, (3, 7)), repr(pinky.position))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(pinky.position, (3, 8)), repr(pinky.position))


    def test_inky_choice(self):
        """Test that Inky always makes the right choice when turning"""
        objects, height, width = h.world.screen("""
XXXXXXXXXXXXXX
XB           X
XXXXXXXIXXXXXX
X*           X
XXXXXXXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        player = self.world.objects['chomper_player']
        player.orientation = 0 # left

        inky = self.world.objects['inky']
        inky.orientation = 3 # down

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(inky.position, (3, 7)), repr(inky.position))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(inky.position, (3, 8)), repr(inky.position))

        inky.position = inky.initial_position
        inky.orientation = 1 # up, so inky will go down after flip

        self.world.time = 29        

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(inky.position, (3, 7)), repr(inky.position))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(inky.position, (3, 6)), repr(inky.position))

    def test_clyde_choice(self):
        """Test that Clyde always makes the right choice when turning"""
        objects, height, width = h.world.screen("""
XXXXXXXXXXXXXX
X            X
XXXXXXXCXXXXXX
X           *X
XXXXXXXXXXXXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        player = self.world.objects['chomper_player']
        player.orientation = 0 # left

        clyde = self.world.objects['clyde']
        clyde.orientation = 3 # down

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(clyde.position, (3, 7)), repr(clyde.position))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(clyde.position, (3, 6)), repr(clyde.position))

        clyde.position = clyde.initial_position
        clyde.orientation = 1 # up, so clyde will go down after flip

        self.world.time = 29        

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(clyde.position, (3, 7)), repr(clyde.position))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(clyde.position, (3, 6)), repr(clyde.position))

    def test_actions(self):
        """Test that actions have the right effect"""
        # create the objects we want
        objects, height, width = h.world.screen("""
XXXX
X* X
X  X
XXXX
""", chomper.legend)
        self.world = universe.create_world('chomper', objects=objects, width=width, height=height)

        player = self.world.objects['chomper_player']
        posn = player.position

        # check that actions do what we think they should
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
