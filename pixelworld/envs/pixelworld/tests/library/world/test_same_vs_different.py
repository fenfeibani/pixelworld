import unittest

import numpy as np

from pixelworld.envs.pixelworld import agents, universe
from pixelworld.envs.pixelworld.tests import test_core

class TestSameVsDifferent(test_core.TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'same_vs_different')
        
        super(TestSameVsDifferent, self).__init__(*args, **kwargs)
    
    def setUp(self):
        pass

    def test_color(self):
        seen_same = seen_different = False

        for seed in xrange(100):
            if seen_same and seen_different:
                break

            self.world = universe.create_world('same_vs_different', seed=seed)
        
            objects = [x for x in self.world.objects if x.name != 'self']
            if objects[0].color == objects[1].color:
                seen_same = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'SAME')

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, -101)
                self.assertTrue(done)
            else:
                seen_different = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'DIFFERENT')

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

        # check that we've seen both cases
        self.assertTrue(seen_same)
        self.assertTrue(seen_different)

    def test_depth(self):
        seen_same = seen_different = False

        for seed in xrange(100):
            if seen_same and seen_different:
                break

            self.world = universe.create_world('same_vs_different', seed=seed, attribute='depth', depth=4)
        
            objects = [x for x in self.world.objects if x.name != 'self']
            if objects[0].depth == objects[1].depth:
                seen_same = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'SAME')

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, -101)
                self.assertTrue(done)
            else:
                seen_different = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'DIFFERENT')

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

        # check that we've seen both cases
        self.assertTrue(seen_same)
        self.assertTrue(seen_different)

    def test_mass(self):
        seen_same = seen_different = False

        for seed in xrange(100):
            if seen_same and seen_different:
                break

            self.world = universe.create_world('same_vs_different', seed=seed, attribute='mass')
        
            objects = [x for x in self.world.objects if x.name != 'self']
            if objects[0].mass == objects[1].mass:
                seen_same = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'SAME')

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, -101)
                self.assertTrue(done)
            else:
                seen_different = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'DIFFERENT')

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

        # check that we've seen both cases
        self.assertTrue(seen_same)
        self.assertTrue(seen_different)

    def test_velocity(self):
        seen_same = seen_different = False

        for seed in xrange(100):
            if seen_same and seen_different:
                break

            self.world = universe.create_world('same_vs_different', seed=seed, attribute='velocity')
        
            objects = [x for x in self.world.objects if x.name != 'self']
            if (objects[0].velocity == objects[1].velocity).all():
                seen_same = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'SAME')

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, -101)
                self.assertTrue(done)
            else:
                seen_different = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'DIFFERENT')

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

        # check that we've seen both cases
        self.assertTrue(seen_same)
        self.assertTrue(seen_different)

    def test_position_x(self):
        seen_same = seen_different = False

        for seed in xrange(100):
            if seen_same and seen_different:
                break

            self.world = universe.create_world('same_vs_different', seed=seed, attribute='position_x')
        
            objects = [x for x in self.world.objects if x.name != 'self']
            if objects[0].position_x == objects[1].position_x:
                seen_same = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'SAME')

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we can change the true class
                objects[0].position_x = objects[1].position_x + 1
                self.assertEqual(self.world.svd_true_class, 'DIFFERENT')
            else:
                seen_different = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'DIFFERENT')

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we can change the true class
                objects[0].position_x = objects[1].position_x
                self.assertEqual(self.world.svd_true_class, 'SAME')

        # check that we've seen both cases
        self.assertTrue(seen_same)
        self.assertTrue(seen_different)

    def test_position_y(self):
        seen_same = seen_different = False

        for seed in xrange(100):
            if seen_same and seen_different:
                break

            self.world = universe.create_world('same_vs_different', seed=seed, attribute='position_y')
        
            objects = [x for x in self.world.objects if x.name != 'self']
            if objects[0].position_y == objects[1].position_y:
                seen_same = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'SAME')

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we can change the true class
                objects[0].position_y = objects[1].position_y + 1
                self.assertEqual(self.world.svd_true_class, 'DIFFERENT')
            else:
                seen_different = True

                # check that true class is correct
                self.assertEqual(self.world.svd_true_class, 'DIFFERENT')

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_SAME')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_DIFFERENT')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we can change the true class
                objects[0].position_y = objects[1].position_y
                self.assertEqual(self.world.svd_true_class, 'SAME')

        # check that we've seen both cases
        self.assertTrue(seen_same)
        self.assertTrue(seen_different)

    def test_actions(self):
        self.world = universe.create_world('same_vs_different', seed=0)

        player = self.world.objects['self']
        posn = player.position

        # check that actions do what we expect
        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(player.position, posn + (0, 1)))
        self.world.step('DOWN')
        self.assertTrue(np.array_equal(player.position, posn + (1, 1)))
        self.world.step('LEFT')
        self.assertTrue(np.array_equal(player.position, posn + (1, 0)))
        self.world.step('UP')
        self.assertTrue(np.array_equal(player.position, posn))
        
    def test_energy(self):
        self.world = universe.create_world('same_vs_different', seed=0)

        self.world.agent = agents.RandomAgent(self.world)
        energy = self.world.energy
        for _ in xrange(100):
            self.world.step()
            self.assertEqual(self.world.energy, energy)


if __name__ == '__main__':
    unittest.main()
