import unittest

import numpy as np

from pixelworld.envs.pixelworld import agents, universe
from pixelworld.envs.pixelworld.tests import test_core

class TestTouching(test_core.TestCase):
    world = None
    
    def test_touching(self):
        seen_touching = seen_not_touching = False

        for seed in xrange(100):
            if seen_touching and seen_not_touching:
                break

            self.world = universe.create_world('touching', seed=seed)
        
            objects = [x for x in self.world.objects if x.name == 'basic']
            if np.sum(np.abs(objects[0].position - objects[1].position)) <= 1.0:
                seen_touching = True

                # check that true class is correct
                self.assertEqual(self.world.touching_true_class, 'TOUCHING')

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_TOUCHING')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_NOT_TOUCHING')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we can change the true class
                objects[0].position = objects[1].position + (2, 0)
                self.assertEqual(self.world.touching_true_class, 'NOT_TOUCHING')
            else:
                seen_not_touching = True

                # check that true class is correct
                self.assertEqual(self.world.touching_true_class, 'NOT_TOUCHING')

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_TOUCHING')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_NOT_TOUCHING')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we can change the true class
                objects[0].position = objects[1].position + (1, 0)
                self.assertEqual(self.world.touching_true_class, 'TOUCHING')

        # check that we've seen both cases
        self.assertTrue(seen_touching)
        self.assertTrue(seen_not_touching)

    def test_actions(self):
        self.world = universe.create_world('touching', seed=0)

        player = self.world.objects['classifier']
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
        self.world = universe.create_world('touching', seed=0)

        # check that energy is conserved
        self.world.agent = agents.RandomAgent(self.world)
        energy = self.world.energy
        for _ in xrange(100):
            self.world.step()
            self.assertEqual(self.world.energy, energy)


if __name__ == '__main__':
    unittest.main()
