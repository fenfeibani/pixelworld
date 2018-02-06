import numpy as np

from pixelworld.envs.pixelworld.tests import test_core
from pixelworld.envs.pixelworld import agents, universe

class TestLibraryWorld(test_core.TestCase):
    world = None

    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'basic')
        
        super(TestLibraryWorld, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()

    def test_energy(self):
        self.world.agent = agents.RandomAgent(self.world)
        energy = self.world.energy
        for _ in xrange(100):
            self.world.step()
            self.assertTrue(np.abs(self.world.energy - energy) < 1e-3)
