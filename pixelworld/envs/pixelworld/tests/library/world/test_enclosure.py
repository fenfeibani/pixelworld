import unittest

import numpy as np

from pixelworld.envs.pixelworld import agents, universe
from pixelworld.envs.pixelworld.tests.library.world import test_library_world

class TestEnclosure(test_library_world.TestLibraryWorld):
    world = None
    
    def test_enclosure(self):
        seen_enclosed = seen_not_enclosed = False

        for seed in xrange(100):
            if seen_enclosed and seen_not_enclosed:
                break

            self.world = universe.create_world('enclosure', seed=seed)

            block = self.world.objects['block']
            container = self.world.objects['container']
            tlbr = container.extent
            objects = [x for x in self.world.objects if x.name == 'basic']
            if tlbr[0] <= block.position[0] <= tlbr[2] and tlbr[1] <= block.position[1] <= tlbr[3]:
                seen_enclosed = True

                # check that true class is correct
                self.assertEqual(self.world.enclosure_true_class, 'ENCLOSED')

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_ENCLOSED')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_NOT_ENCLOSED')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we can change the true class
                block.position = (1, 1)
                self.assertEqual(self.world.enclosure_true_class, 'NOT_ENCLOSED')
            else:
                seen_not_enclosed = True

                # check that true class is correct
                self.assertEqual(self.world.enclosure_true_class, 'NOT_ENCLOSED')

                # check that we get a penalty for incorrect classification and
                # the episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_ENCLOSED')
                self.assertEqual(reward, -101)
                self.assertTrue(done)

                # check that we get a reward for correct classification and the
                # episode terminates
                obs, reward, done, info = self.world.step('CLASSIFY_NOT_ENCLOSED')
                self.assertEqual(reward, 99)
                self.assertTrue(done)

                # check that we can change the true class
                self.world.objects['classifier'].position = (10, 10)
                block.position = container.position
                self.assertEqual(self.world.enclosure_true_class, 'ENCLOSED')

        # check that we've seen both cases
        self.assertTrue(seen_enclosed)
        self.assertTrue(seen_not_enclosed)

    def test_actions(self):
        self.world = universe.create_world('enclosure', seed=0)

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
        self.world = universe.create_world('enclosure', seed=0)

        # check that energy is conserved
        self.world.agent = agents.RandomAgent(self.world)
        energy = self.world.energy
        for _ in xrange(100):
            self.world.step()
            self.assertEqual(self.world.energy, energy)


if __name__ == '__main__':
    unittest.main()
