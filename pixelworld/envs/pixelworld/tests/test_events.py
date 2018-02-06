import unittest

from pixelworld.envs.pixelworld import universe, core, events, objects
from pixelworld.envs.pixelworld.tests import test_core

class TestEventGeneric(test_core.TestEvent):
    def test(self):
        """Check that event looks as expected"""
        class MyEvent(core.Event):
            _name = 'my'
            _parameters = ['flingrippery']
        
        e = MyEvent(self.world, flingrippery=12)
        
        # check that parameter flingrippery exists and has the right value and
        # shows up in dir
        self.assertTrue('flingrippery' in e._parameters)
        self.assertTrue(e.flingrippery == 12)
        self.assertTrue('flingrippery' in dir(e))

    def test_addition_removal(self):
        """Check that we can add and remove event"""
        class MyEvent(core.Event):
            _name = 'my'
            _parameters = ['flingrippery']

        e = MyEvent(self.world, flingrippery=12)
        self.check_addition_removal(e)


class TestCollisionEvent(test_core.TestEvent):
    def test(self):
        """Check that event looks as expected"""
        e = events.CollisionEvent(self.world, indices=[2, 4])
        
        # check that parameter indices exists and has the right value and shows
        # up in dir
        self.assertTrue('indices' in e._parameters)
        self.assertTrue(e.indices == [2, 4])
        self.assertTrue('indices' in dir(e))

    def test_addition_removal(self):
        """Check that we can add and remove event"""
        e = events.CollisionEvent(self.world, indices=[2, 4])

        self.check_addition_removal(e)

    def test_description(self):
        """Check that description is what we would expect"""
        e = events.CollisionEvent(self.world, indices=[2, 4])
        names = self.world.objects[2], self.world.objects[4]
        self.assertEqual(e.description, 'collision between %s and %s' % names)


class TestLeaveScreenEvent(test_core.TestEvent):
    def test(self):
        """Check that event looks as expected"""
        e = events.LeaveScreenEvent(self.world, indices=[2, 4, 6])
        
        # check that parameter indices exists and has the right value and shows
        # up in dir
        self.assertTrue('indices' in e._parameters)
        self.assertTrue(e.indices == [2, 4, 6])
        self.assertTrue('indices' in dir(e))

    def test_addition_removal(self):
        """Check that we can add and remove event"""
        e = events.LeaveScreenEvent(self.world, indices=[2, 4, 6])
        self.check_addition_removal(e)

    def test_description(self):
        """Check that description is what we would expect"""
        e = events.LeaveScreenEvent(self.world, indices=[2, 4, 6])
        names = self.world.objects[2], self.world.objects[4], self.world.objects[6]
        self.assertEqual(e.description, 'object(s) left the screen: %s, %s, %s' % names)


class TestPushEvent(test_core.TestEvent):
    def test(self):
        """Check that event looks as expected"""
        e = events.PushEvent(self.world, idx=3, success=True)
        
        # check that parameter idx exists and has the right value and shows up
        # in dir
        self.assertTrue('idx' in e._parameters)
        self.assertTrue(e.idx == 3)
        self.assertTrue('idx' in dir(e))

        # check that parameter success exists and has the right value and shows
        # up in dir
        self.assertTrue('success' in e._parameters)
        self.assertTrue(e.success == True)
        self.assertTrue('success' in dir(e))

    def test_addition_removal(self):
        """Check that we can add and remove event"""
        e = events.PushEvent(self.world, idx=3, success=True)

        self.check_addition_removal(e)

    def test_description(self):
        """Check that description is what we would expect"""
        # check that description is what we would expect
        e = events.PushEvent(self.world, idx=3, success=True)
        name = self.world.objects[3]
        self.assertEqual(e.description, 'successful push by %s' % name)

        # check that description is what we would expect
        e = events.PushEvent(self.world, idx=3, success=False)
        self.assertEqual(e.description, 'failed push by %s' % name)


class TestBulletFireEvent(test_core.TestEvent):
    def test(self):
        """Check that event looks as expected"""
        bullet = objects.Bullet(self.world, bullet_shooter=self.world.objects[3])
        e = events.BulletFireEvent(self.world, position=(10, 10), dir=2, bullet_id=bullet.id)
        
        # check that parameter position exists and has the right value and shows
        # up in dir
        self.assertTrue('position' in e._parameters)
        self.assertTrue(e.position == (10, 10))
        self.assertTrue('position' in dir(e))

        # check that parameter dir exists and has the right value and shows up
        # in dir
        self.assertTrue('dir' in e._parameters)
        self.assertTrue(e.dir == 2)
        self.assertTrue('dir' in dir(e))

        # check that parameter bullet_id exists and has the right value and
        # shows up in dir
        self.assertTrue('bullet_id' in e._parameters)
        self.assertTrue(e.bullet_id == bullet.id)
        self.assertTrue('bullet_id' in dir(e))

    def test_addition_removal(self):
        """Check that we can add and remove event"""
        bullet = objects.Bullet(self.world, bullet_shooter=self.world.objects[3])
        e = events.BulletFireEvent(self.world, position=(10, 10), dir=2, bullet_id=bullet.id)

        self.check_addition_removal(e)

    def test_description(self):
        """Check that description is what we would expect"""
        # check that description is what we would expect
        bullet = objects.Bullet(self.world, bullet_shooter=self.world.objects[3])
        e = events.BulletFireEvent(self.world, position=(10, 10), dir=2, bullet_id=bullet.id)
        name = self.world.objects[3]
        self.assertEqual(e.description, 'bullet fired by %s' % name)


class TestKillEvent(test_core.TestEvent):
    def test(self):
        """Check that event looks as expected"""
        e = events.KillEvent(self.world, victim=3, reason='misadventure')
        
        # check that parameter victim exists and has the right value and shows
        # up in dir
        self.assertTrue('victim' in e._parameters)
        self.assertTrue(e.victim == 3)
        self.assertTrue('victim' in dir(e))

        # check that parameter reason exists and has the right value and shows up
        # in dir
        self.assertTrue('reason' in e._parameters)
        self.assertTrue(e.reason == 'misadventure')
        self.assertTrue('reason' in dir(e))

    def test_addition_removal(self):
        """Check that we can add and remove event"""
        e = events.KillEvent(self.world, victim=3, reason='misadventure')

        self.check_addition_removal(e)

    def test_description(self):
        """Check that description is what we would expect"""
        e = events.KillEvent(self.world, victim=3, reason='misadventure')

        # check that description is what we would expect
        name = self.world.objects[3]
        self.assertEqual(e.description, '%s died by misadventure' % name)


if __name__ == '__main__':
    unittest.main()

