import unittest
import sys

import pixelworld.envs.pixelworld.demo as demo
from pixelworld.envs.pixelworld.tests import test_core

class TestDemo(test_core.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDemo, self).__init__(*args, **kwargs)

    def test_parse_args(self):
        """Check that parse_args works correctly"""
        # pretend that there are command line arguments
        sys.argv[1:] = ['basic', 'thing=3', 'floor=ceiling', 'ceiling=floor']

        # pass in more args (overridden by command line args) 
        world_name, kwargs = demo.parse_args(thing=2, foo=6, bar=7, baz=8)

        # check return values
        self.assertEqual(world_name, 'basic')
        self.assertEqual(kwargs, dict(thing=3, floor='ceiling', ceiling='floor',
                                      foo=6, bar=7, baz=8))
        

if __name__ == '__main__':
    unittest.main()
