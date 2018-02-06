from noseperf.testcases import PerformanceTest
import copy

import pixelworld.envs.pixelworld as pw

class MyTest(PerformanceTest):
    def test_creation(self):
        """Test the speed of world creation"""
        for i in xrange(10):
            pw.universe.create_world('basic', seed=i)

    def test_object_creation(self):
        """Test the speed of object creation"""
        world = pw.universe.create_world('blank', seed=0)
        for _ in xrange(100):
            world.create_object('basic')

    def test_copy(self):
        """Test the speed of world copying"""
        world = pw.universe.create_world('basic', seed=0)
        for _ in xrange(10):
            world2 = copy.deepcopy(world)

    def test_reset(self):
        """Test the speed of world resetting"""
        world = pw.universe.create_world('basic', seed=0)
        for _ in xrange(10):
            world.reset()

    def test_seachase(self):
        """Test the speed of running Seachase world"""
        world = pw.universe.create_world('seachase', seed=0)
        world.agent = 'random'
        for _ in xrange(1000):
            world.step()

    def test_collision3(self):
        """Test the speed of running collision3 world"""
        world = pw.universe.create_world('collision3', seed=0)
        for _ in xrange(1000):
            world.step()
