import unittest

import numpy as np

from pixelworld.envs.pixelworld import core, universe, world_attributes
from pixelworld.envs.pixelworld.tests import test_core


class TestShowColorsWorldAttribute(test_core.TestWorldAttribute):
    def test_colors(self):
        """Check that show_colors world attribute has the expected effect on colors"""
        x = self.world.create_object(['basic', dict(color=2)])

        # check that x has color 2 and that it shows up in world state
        self.world.show_colors = True
        self.assertEqual(x.color, 2)
        self.assertEqual(self.world.state[tuple(x.state_index.tolist())], 2)

        # check that x has the same color but it doesn't show up in world state
        self.world.show_colors = False
        self.assertEqual(x.color, 2)
        self.assertEqual(self.world.state[tuple(x.state_index.tolist())], 1)


class TestKillingDeletesWorldAttribute(test_core.TestWorldAttribute):
    def test_killing_deletes(self):
        """Check that killing_deletes world attribute has the expected effect"""
        x = self.world.create_object(['basic', dict(killed_by='expediency', alive=True)])

        # check that x does not get deleted when killing_deletes=False
        self.world.killing_deletes = False
        x.alive = False
        self.assertFalse(x.alive)
        self.assertTrue(x in self.world.objects)

        # check that x gets deleted when killing_deletes=True
        self.world.killing_deletes = True
        x.alive = True
        x.alive = False
        self.assertTrue(x not in self.world.objects)


if __name__ == '__main__':
    unittest.main()
