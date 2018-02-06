import unittest

import numpy as np

from pixelworld.envs.pixelworld import utils
from pixelworld.envs.pixelworld.tests import test_core

class TestUtils(test_core.TestCase):
    def test_fix_float_integer(self):
        """Check that fix_float_integer works as expected"""
        x = 2.0 - 2 * utils.eps_threshold
        self.assertEqual(utils.fix_float_integer(x), x)
        x = 2.0 - 0.5 * utils.eps_threshold
        self.assertEqual(utils.fix_float_integer(x), 2.0)

        x = 2.0 + 2 * utils.eps_threshold
        self.assertEqual(utils.fix_float_integer(x), x)
        x = 2.0 + 0.5 * utils.eps_threshold
        self.assertEqual(utils.fix_float_integer(x), 2.0)

        x = np.array([2.0 - 0.5 * utils.eps_threshold, 2.0 + 0.5 * utils.eps_threshold])
        self.assertTrue(np.array_equal(utils.fix_float_integer(x), [2, 2]))

    def test_roundup(self):
        """Check that roundup works as expected"""
        self.assertEqual(utils.roundup(-1.5), -2)
        self.assertEqual(utils.roundup(1.5), 2)
        self.assertEqual(utils.roundup(-2.5), -3)
        self.assertEqual(utils.roundup(2.5), 3)

    def test_mode(self):
        """Check that mode works as expected"""
        self.assertEqual(utils.mode([1, 2, 3, 4]), 1)
        self.assertEqual(utils.mode([1, 1, 3, 4]), 1)
        self.assertEqual(utils.mode([1, 2, 2, 4]), 2)
        self.assertEqual(utils.mode([1, 1, 2, 2]), 1)

    def test_is_iterable(self):
        """Check that is_iterable works as expected"""
        self.assertTrue(utils.is_iterable(xrange(10)))
        self.assertTrue(utils.is_iterable(range(10)))
        self.assertTrue(utils.is_iterable(np.arange(10)))
        self.assertTrue(utils.is_iterable(dict(x=3)))

        self.assertFalse(utils.is_iterable(17))
        self.assertFalse(utils.is_iterable('fleek'))

    def test_to_iterable(self):
        """Check that to_iterable works as expected"""
        self.assertEqual(utils.to_iterable(17, length=0), [])
        self.assertEqual(utils.to_iterable(17, length=3), [17] * 3)
        self.assertEqual(utils.to_iterable([17] * 3, length=3), [17] * 3)
        with self.assertRaises(ValueError):
            print utils.to_iterable([17] * 3, length=4)

    def test_switch(self):
        """Check that switch works as expected"""
        self.assertEqual(utils.switch('foo',
                                      foo=3,
                                      bar=6), 3)
        self.assertEqual(utils.switch('blonk',
                                      foo=3,
                                      bar=6,
                                      default=7), 7)
        self.assertEqual(utils.switch('foo', resolve_functions=True,
                                      foo=lambda: 7,
                                      bar=lambda: 11), 7)

    def test_plural(self):
        """Check that plural works as expected"""
        self.assertEqual(utils.plural(1, 'child', 'children'), 'child')
        self.assertEqual(utils.plural(11, 'child', 'children'), 'children')

    def test_generate_color_palette(self):
        """Check that generate_color_palette generates distinct colors"""
        palette = utils.generate_color_palette(2)
        self.assertEqual(len(palette), 2)

        palette = utils.generate_color_palette(200)
        self.assertEqual(len(palette), 200)
        for i, color in enumerate(palette):
            for color2 in palette[i + 1:]:
                self.assertTrue((color != color2).any())

    def test_ind_to_rgb(self):
        """Check that ind_to_rgb works as expected"""
        val = np.zeros((2, 2), dtype=int)
        val[0, 1] = 1
        val[1, 0] = 2

        val2 = utils.ind_to_rgb(val, utils.base_lut)
        self.assertEqual(val2.shape, (2, 2, 3))
        self.assertTrue(np.array_equal(val2[0, 0, :], utils.base_lut[0]))
        self.assertTrue(np.array_equal(val2[0, 1, :], utils.base_lut[1]))
        self.assertTrue(np.array_equal(val2[1, 0, :], utils.base_lut[2]))
        self.assertTrue(np.array_equal(val2[1, 1, :], utils.base_lut[0]))


if __name__ == '__main__':
    unittest.main()
