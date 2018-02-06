import unittest

import numpy as np

from pixelworld.envs.pixelworld import core, events, objects, pixel_worlds
from pixelworld.envs.pixelworld.tests import test_core

class TestScreenBasedPixelWorld(test_core.TestCase):
    world = None

    def __init__(self, *args, **kwargs):
        super(TestScreenBasedPixelWorld, self).__init__(*args, **kwargs)
        

    def test_screen_changing(self):
        """Test that we can change screens through the screen world attribute and also
        by generating ChangeScreenEvents."""

        class ChangeScreenAbilityObjectAttribute(core.AbilityObjectAttribute):
            """Attribute that allows object to change screen at will. Just for testing."""
            _actions = ['CHANGE_SCREEN']
            def _execute_action(self, obj, t, dt, agent_id, action):
                """Listen for CHANGE_SCREEN commands and generate ChangeScreenEvents.

                Parameters
                ----------
                obj : Object
                    The object that has the 'gripper' attribute.
                t : number
                    The simulation time.
                dt : number
                    The time since the last step.
                agent_id : int
                    The id of the agent that is currently stepping.
                action : string
                    The most recent action executed.
                """
                if action == 'CHANGE_SCREEN':
                    event = events.ChangeScreenEvent(self.world, new_screen=1)

        class ScreenChangingSelf(objects.SelfObject):
            """A self with the ability to change screens"""
            _attributes = ['change_screen_ability']


        self.world = pixel_worlds.ScreenBasedPixelWorld(
            screens = ["""
WWWW
W  W
WSBW 
WWWW
""",
                       """
WWWW
WSBW
    
WWWW
"""],
            legend = dict(W='wall', S=['screen_changing_self', dict(name='self')], B='basic'))

        self.assertEqual(self.world.width, 4)
        self.assertEqual(self.world.height, 4)

        # check that we are on screen 0
        self.assertTrue(np.array_equal(self.world.objects['self'].position, (2, 1)))
        self.assertTrue(np.array_equal(self.world.objects['basic'].position, (2, 2)))
        self.assertEqual(len(self.world.objects['wall']), 12)

        self.world.screen = 1

        # check that we are on screen 1
        self.assertTrue(np.array_equal(self.world.objects['self'].position, (1, 1)))
        self.assertTrue(np.array_equal(self.world.objects['basic'].position, (1, 2)))
        self.assertEqual(len(self.world.objects['wall']), 10)

        self.world.screen = 0

        # check that we are on screen 0
        self.assertTrue(np.array_equal(self.world.objects['self'].position, (2, 1)))
        self.assertTrue(np.array_equal(self.world.objects['basic'].position, (2, 2)))
        self.assertEqual(len(self.world.objects['wall']), 12)

        self.world.step('CHANGE_SCREEN')

        # check that we are on screen 1
        self.assertTrue(np.array_equal(self.world.objects['self'].position, (1, 1)))
        self.assertTrue(np.array_equal(self.world.objects['basic'].position, (1, 2)))
        self.assertEqual(len(self.world.objects['wall']), 10)

    def test_change_to_new_screen(self):
        """Test that we can change to a new screen by using change_to_new_screen()."""
        self.world = pixel_worlds.ScreenBasedPixelWorld(
            screens = ["""
WWWW
W  W
WSBW 
WWWW
""",
                       """
WWWW
WSBW
    
WWWW
"""],
            legend = dict(W='wall', S='self', B='basic'))

        self.assertEqual(self.world.width, 4)
        self.assertEqual(self.world.height, 4)

        # check that we are on screen 0
        self.assertTrue(np.array_equal(self.world.objects['self'].position, (2, 1)))
        self.assertTrue(np.array_equal(self.world.objects['basic'].position, (2, 2)))
        self.assertEqual(len(self.world.objects['wall']), 12)

        self.world.screen = 1

        # check that we are on screen 1
        self.assertTrue(np.array_equal(self.world.objects['self'].position, (1, 1)))
        self.assertTrue(np.array_equal(self.world.objects['basic'].position, (1, 2)))
        self.assertEqual(len(self.world.objects['wall']), 10)

        idx = self.world.change_to_new_screen("""
WWWW
W SW
W BW
WWWW
""")

        # check that index is what we expect
        self.assertEqual(idx, 2)

        # check that we are on new screen
        self.assertTrue(np.array_equal(self.world.objects['self'].position, (1, 2)))
        self.assertTrue(np.array_equal(self.world.objects['basic'].position, (2, 2)))
        self.assertEqual(len(self.world.objects['wall']), 12)

if __name__ == '__main__':
    unittest.main()
