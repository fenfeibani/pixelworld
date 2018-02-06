import pixelworld.envs.pixelworld.core as core

class CrazyRandomizer(core.Randomizer):
    _randomize_attributes = ['position', 'mass', 'acceleration', 'velocity']

#randomly placed moving balls
balls = 10*[['basic', {'velocity': lambda obj: 3 - 6*obj.world.rng.rand(2)}]]

objects = ['frame'] + balls
agent = 'agent'

randomizer = 'crazy'
