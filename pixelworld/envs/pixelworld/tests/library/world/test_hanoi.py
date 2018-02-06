import unittest

from pixelworld.envs.pixelworld import universe, agents, library
from pixelworld.envs.pixelworld.tests import test_core
from pixelworld.envs.pixelworld.tests.library.world import test_library_world

hanoi = library.import_item('world', 'hanoi')

class TestHanoi(test_library_world.TestLibraryWorld):
    world = None
    
    def __init__(self, *args, **kwargs):
        super(TestHanoi, self).__init__(*args, **kwargs)
    
    def setUp(self):
        self.world = universe.create_world('hanoi', num_disks=2)
    
    def tearDown(self):
        self.world.end()

    def validate_arrangement(self):
        """Check that all the disks are in the right position"""
        # for each tower
        for location in xrange(3):
            disks = self.world.objects.find(name='disk', location=location)

            # sort disks by size
            disks = sorted(disks, key=lambda disk: disk.size, reverse=True)
            for i, disk in enumerate(disks):
                # check that disks are at the right y-position based on their relative sizes
                assert disk.position[0] == self.world.height - 1 - i, repr(disk.position) + repr(i)
                # check that the disks are at the right x-position based on tower location
                assert disk.position[1] == 20 * (location + 1)

    def test_solver(self):
        # adapted from wikipedia
        def solve(n, source, target, aux):
            if n > 0:
                solve(n-1, source, aux, target)
                
                self.world.step('_'.join([source, 'TO', target]))
                self.validate_arrangement()

                solve(n-1, aux, target, source)

        for n in xrange(2, 7):
            # create a puzzle with n disks
            self.world = universe.create_world('hanoi', num_disks=n)
            self.validate_arrangement()
            
            # check that the solver solves it in the right amount of time
            solve(n, 'LEFT', 'RIGHT', 'CENTER')
            self.assertEqual(self.world.time, (2 ** n) - 1)

            # check that we won
            self.assertTrue(isinstance(self.world.events[-1], hanoi.WinEvent), repr(self.world.events))

    def test_composite_actions(self):
        # get the disks
        big_disk = self.world.objects.find(name='disk', size=4)[0]
        small_disk = self.world.objects.find(name='disk', size=2)[0]

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('CENTER_TO_RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        for disk in self.world.objects['disk']:
            self.assertEqual(disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('RIGHT_TO_CENTER')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        for disk in self.world.objects['disk']:
            self.assertEqual(disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('LEFT_TO_RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 2)
        self.assertEqual(big_disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('LEFT_TO_RIGHT')
        self.assertEqual(reward, -101)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 2)
        self.assertEqual(big_disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('RIGHT_TO_CENTER')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 1)
        self.assertEqual(big_disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('LEFT_TO_CENTER')
        self.assertEqual(reward, -101)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 1)
        self.assertEqual(big_disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('LEFT_TO_RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 1)
        self.assertEqual(big_disk.location, 2)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('RIGHT_TO_LEFT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 1)
        self.assertEqual(big_disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('LEFT_TO_RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 1)
        self.assertEqual(big_disk.location, 2)

        # check that reward is what we expect, the episode DOES terminate, and
        # the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('CENTER_TO_RIGHT')
        self.assertEqual(reward, 999)
        self.assertTrue(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 2)
        self.assertEqual(big_disk.location, 2)

    def test_click_actions(self):
        # get the disks
        big_disk = self.world.objects.find(name='disk', size=4)[0]
        small_disk = self.world.objects.find(name='disk', size=2)[0]

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('DOWN')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        for disk in self.world.objects['disk']:
            self.assertEqual(disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        obs, reward, done, info = self.world.step('DOWN')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        for disk in self.world.objects['disk']:
            self.assertEqual(disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 2)
        self.assertEqual(big_disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, -101)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 2)
        self.assertEqual(big_disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        obs, reward, done, info = self.world.step('DOWN')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 1)
        self.assertEqual(big_disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        obs, reward, done, info = self.world.step('DOWN')
        self.assertEqual(reward, -101)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 1)
        self.assertEqual(big_disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 1)
        self.assertEqual(big_disk.location, 2)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 1)
        self.assertEqual(big_disk.location, 0)

        # check that reward is what we expect, the episode doesn't terminate,
        # and the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 1)
        self.assertEqual(big_disk.location, 2)

        # check that reward is what we expect, the episode DOES terminate, and
        # the arrangement is still valid, and that everything is where we
        # expect it to be
        obs, reward, done, info = self.world.step('DOWN')
        self.assertEqual(reward, -1)
        self.assertFalse(done)
        self.validate_arrangement()
        obs, reward, done, info = self.world.step('RIGHT')
        self.assertEqual(reward, 999)
        self.assertTrue(done)
        self.validate_arrangement()
        self.assertEqual(small_disk.location, 2)
        self.assertEqual(big_disk.location, 2)

    def test_only_top_moves(self):
        # get the disks
        big_disk = self.world.objects.find(name='disk', size=4)[0]
        small_disk = self.world.objects.find(name='disk', size=2)[0]

        # try to move the big disk even though it's not top disk
        big_disk.location = 1

        # check that nothing moved
        self.assertEqual(big_disk.location, 0)
        self.assertEqual(small_disk.location, 0)

        # check that we generated a cheat event
        self.assertTrue(isinstance(self.world.events[-1], hanoi.CheatEvent))

    def test_top_disk(self):
        # check that we can pass an array of idxs into top_disk.get()
        idxs = [tower.id for tower in sorted(self.world.objects['tower'], key=lambda tower: tower.label)]
        top_disk = self.world.object_attributes['top_disk']
        vals = top_disk.get(idxs)
        self.assertEqual(vals[0].size, 2)
        self.assertEqual(vals[1], None)
        self.assertEqual(vals[2], None)

    def test_size(self):
        # check that we can pass an array of idxs into top_disk.get()
        idxs = [disk.id for disk in sorted(self.world.objects['disk'], key=lambda disk: disk.size)]
        size = self.world.object_attributes['size']
        vals = size.get(idxs)
        self.assertEqual(vals[0], 2)
        self.assertEqual(vals[1], 4)


if __name__ == '__main__':
    unittest.main()
