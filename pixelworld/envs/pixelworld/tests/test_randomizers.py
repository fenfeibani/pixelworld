import unittest

import numpy as np

from pixelworld.envs.pixelworld import core, randomizers, universe, variants
from pixelworld.envs.pixelworld.tests import test_core


class TestReseedingRandomizer(test_core.TestRandomizer):
    def setUp(self):
        super(TestReseedingRandomizer, self).setUp()

        self.world.randomizer = randomizers.ReseedingRandomizer(self.world)

    def test_reset(self):
        """Check that reseeding randomizer actually reseeds when it should and not when
        it shouldn't"""
        # reset the world and generate a random number
        self.world.reset()
        u1 = self.world.rng.rand()

        # reset the world and generate a random number
        self.world.reset()
        u2 = self.world.rng.rand()

        # check that random numbers are different
        self.assertTrue(u1 != u2)

        # save a snapshot and generate a random number
        self.world.save_snapshot('foo')
        u3 = self.world.rng.rand()        

        # load the snapshot and generate a random number
        self.world.load_snapshot('foo')
        u4 = self.world.rng.rand()        

        # check that the random numbers are the same
        self.assertEqual(u3, u4)


class TestRandomPositionsRandomizer(test_core.TestRandomizer):
    def test_randomizer(self):
        """Check that randomizer randomizes positions of non-excluded objects and
        leaves excluded objects alone. """
        # get all positions
        position = self.world.object_attributes['position']
        positions = position.get()

        # set randomizer to default randomizer and randomize
        self.world.randomizer = core.Randomizer(self.world)
        self.world.randomizer.randomize()

        # check that positions didn't change
        positions2 = position.get()
        self.assertTrue(np.array_equal(positions, positions2))

        # set randomizer to RandomPositionsRandomizer and randomize
        self.world._randomizer = randomizers.RandomPositionsRandomizer(self.world)
        self.world.randomizer.randomize()

        # check that positions did change
        positions3 = position.get()
        # technically this might fail but the odds are pretty astronomical
        # since every object would have to get the same position
        self.assertFalse(np.array_equal(positions, positions3))

        # create a frame, check that randomization leaves it alone
        self.world.create_object('frame')
        frame = self.world.objects['frame']
        frame_posn = frame.position
        self.world.randomizer.randomize()
        self.assertTrue(np.array_equal(frame_posn, frame.position))

        # create a basic object with _exclude_randomize=True, check that
        # randomization leaves it alone
        y = self.world.create_object('basic')
        y._exclude_randomize = True
        y_posn = y.position
        y.mass = 0 # otherwise we might get shoved by someone else
        self.world.randomizer.randomize()
        self.assertTrue(np.array_equal(y_posn, y.position))

        # create a basic object with _exclude_randomize_attributes=position,
        # check that randomization leaves it alone
        z = self.world.create_object('basic')
        z._exclude_randomize_attributes = ['position']
        z_posn = z.position
        z.mass = 0 # otherwise we might get shoved by someone else
        self.world.randomizer.randomize()
        self.assertTrue(np.array_equal(z_posn, z.position))


if __name__ == '__main__':
    unittest.main()
