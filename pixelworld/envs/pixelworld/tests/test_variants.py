import unittest

from pixelworld.envs.pixelworld import core, universe, variants
from pixelworld.envs.pixelworld.tests import test_core

class TestShowColorsVariant(test_core.TestVariant):
    def test_colors(self):
        """Check that show_colors variant has the intended effect on colors."""
        x = self.world.create_object(['basic', dict(color=2)])
        
        variant = variants.ShowColorsVariant(self.world)

        # check that x has color 2 and that it shows up in world state
        variant.set(True)
        self.assertEqual(x.color, 2)
        self.assertEqual(self.world.state[tuple(x.state_index.tolist())], 2)

        # check that x has the same color but it doesn't show up in world state
        variant.set(False)
        self.assertEqual(x.color, 2)
        self.assertEqual(self.world.state[tuple(x.state_index.tolist())], 1)

    def test_addition_removal(self):
        """Check that we can add and remove variant from the world"""
        variant = variants.ShowColorsVariant(self.world)

        self.check_addition_removal(variant)

if __name__ == '__main__':
    unittest.main()
