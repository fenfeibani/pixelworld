import pixelworld.envs.pixelworld as pw

shape = """
 XXX
XXXXX
 XXX
"""

red_balls = [['complex', {'name': 'red_ball', 'shape': shape, 'color': 2, 'mass': 1, 'depth': 1, 'velocity': (-1, 1)}]] * 3
blue_balls = [['complex', {'name': 'blue_ball', 'shape': shape, 'color': 3, 'mass': 1, 'depth': 0, 'velocity': (-1, 1)}]] * 3

white_ball = [['complex', {'name': 'white_ball', 'shape': shape, 'color': 1, 'mass': 1, 'velocity':(0, .5)}]]

class DepthWorld(pw.PixelWorld):
    _world_attributes = ['depth']

world = DepthWorld
objects = ['frame'] + white_ball + red_balls + blue_balls
agent = 'agent'
width = 30
height = 30
depth = 2
