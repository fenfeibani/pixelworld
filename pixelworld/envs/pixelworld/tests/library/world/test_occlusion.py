import unittest

import numpy as np

from pixelworld.envs.pixelworld import agents, universe
from pixelworld.envs.pixelworld.tests import test_core

class TestOcclusion(test_core.TestCase):
    world = None
    
    def test_occlusion(self):
        seen_occlusion = seen_not_occlusion = False

        for seed in xrange(100):
            if seen_occlusion and seen_not_occlusion:
                break

            self.world = universe.create_world('occlusion', seed=seed)
        
            objects = [x for x in self.world.objects if x.name == 'basic']
            if np.sum(np.abs(objects[0].position - objects[1].position)) <= 1.0:
                seen_occlusion = True

                # check that true class is correct
                self.assertEqual(self.world.occlusion_true_class, 'OCCLUSION')

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_OCCLUSION')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_NOT_OCCLUSION')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we can change the true class
                objects[0].position = objects[1].position + (2, 0)
                self.assertEqual(self.world.occlusion_true_class, 'NOT_OCCLUSION')
            else:
                seen_not_occlusion = True

                # check that true class is correct
                self.assertEqual(self.world.occlusion_true_class, 'NOT_OCCLUSION')

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_OCCLUSION')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_NOT_OCCLUSION')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we can change the true class
                objects[0].position = objects[1].position
                self.assertEqual(self.world.occlusion_true_class, 'OCCLUSION')

        # check that we've seen both cases
        self.assertTrue(seen_occlusion)
        self.assertTrue(seen_not_occlusion)

    def test_actions(self):
        self.world = universe.create_world('occlusion', seed=0)

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
        self.world = universe.create_world('occlusion', seed=0)

        # check that energy is conserved
        self.world.agent = agents.RandomAgent(self.world)
        energy = self.world.energy
        for _ in xrange(100):
            self.world.step()
            self.assertEqual(self.world.energy, energy)


if __name__ == '__main__':
    unittest.main()
