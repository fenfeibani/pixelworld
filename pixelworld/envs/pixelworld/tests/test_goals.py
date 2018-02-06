import unittest

from pixelworld.envs.pixelworld import universe, core, goals, objects
from pixelworld.envs.pixelworld.tests import test_core


class TestAnythingGoal(test_core.TestGoal):
    def setUp(self):
        super(TestAnythingGoal, self).setUp()

        class Goal1(goals.AnythingGoal):
            _exclusive = True
            _terminates = True
                    
        class Goal2(goals.AnythingGoal):
            _exclusive = False
            _terminates = False

        class Goal3(core.Goal):
            _test_achieved = False
            
            _exclusive = True
            _terminates = False
            
            def _is_achieved(self):
                return self._test_achieved
            
            def _achieve(self):
                self._test_achieved = True
                
                return True
        
        self.goal1 = Goal1
        self.goal2 = Goal2
        self.goal3 = Goal3

    def test_achieve(self):
        """Check that the goal is always achieved"""
        g = self.goal1(self.world)
        
        self.assertTrue(g.is_achieved())
        self.assertTrue(g.achieved)
        
        self.assertTrue(g.achieve())
        self.assertTrue(g.is_achieved())
        self.assertTrue(g.achieved)

    def test_active_termination(self):
        """Check that inactive goals don't prevent termination"""
        #make sure inactive goals don't prevent termination
        g1 = self.goal3(self.world, terminates=True)
        g2 = self.goal2(self.world, terminates=True)

        # set the judge termination mode to all
        judge = core.Judge(self.world, goal_termination_mode='all')

        # deactivate g1
        g1.deactivate()

        # step the world and check that we're done
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(done)

    def test_any_termination(self):
        """Check that any-goal-terminates mode works correctly"""
        #either goal terminates, and both are already achieved
        g1 = self.goal2(self.world, terminates=True)
        g2 = self.goal2(self.world, terminates=True)

        # step the world and check that we're done
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(done)

    def test_all_termination(self):
        """Check that all-goals-terminates mode works correctly"""
        #both goals need to be achieved before termination, but g2 is already
        #achieved
        g1 = self.goal3(self.world, terminates=True)
        g2 = self.goal2(self.world, terminates=True)
        
        # set the judge termination mode to all
        judge = core.Judge(self.world, goal_termination_mode='all')
        
        # achieve g1
        g1.achieve()

        # step the world and check that we're done
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(done)


class TestActionGoal(test_core.TestGoal):
    def test(self):        
        """Check that we can achieve the goal using goal.achieve()"""
        g = goals.ActionGoal(self.world, actions=['NOOP'])

        # check that goal is not yet achieved
        self.assertFalse(g.achieved)
        self.assertFalse(g.is_achieved())
        self.assertFalse(g._is_achieved())
        
        # check that goal is achieved by g.achieve()
        self.assertTrue(g.achieve(error=False))
        self.assertTrue(g._achieve())
        self.assertTrue(g.achieve(error=True))
        self.assertTrue(g.achieved)
        self.assertTrue(g.is_achieved())
        self.assertTrue(g._is_achieved())

    def test_no_actions(self):        
        """Check that goal is unachievable when it has no actions"""
        g = goals.ActionGoal(self.world, actions=[])

        # check that goal is not achieved
        self.assertFalse(g.achieved)
        self.assertFalse(g.is_achieved())
        self.assertFalse(g._is_achieved())
        
        # check that goal is not achieved by g.achieve()
        self.assertEqual(g.achieve(error=False), 'no actions are defined')
        self.assertEqual(g._achieve(), 'no actions are defined')
        with self.assertRaises(RuntimeError):
            self.assertFalse(g.achieve(error=True))
        self.assertFalse(g.achieved)
        self.assertFalse(g.is_achieved())
        self.assertFalse(g._is_achieved())

    def test_malformed_actions(self):
        """Check that we have to supply a list for the actions"""
        with self.assertRaises(TypeError):
            g = goals.ActionGoal(self.world, actions=17)

    def test_any_termination(self):
        """Disable this test"""
        pass

    def test_exclusive(self):
        """Disable this test"""
        pass


if __name__ == '__main__':
    unittest.main()

