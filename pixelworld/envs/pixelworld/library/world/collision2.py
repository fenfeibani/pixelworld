import numpy as np

from ..helpers import h
 
#triangle with holes at the corners
t = h.t(4)
line1 = h.world.shape(11, 8+t, center=False)
line2 = h.world.shape(8-t, 5+t, center=False)
line3 = h.world.shape(5+t, 11+t, center=False)
triangle = [['complex', {
                'name': 'triangle',
                'shape': line1 + line2 + line3,
                'velocity': (0, -1),
                }]]
 
#bouncing red ball
ball = [['basic', {
            'name': 'ball',
            'position': (9,9),
            'velocity': (-1,1),
            'color': 2,
            }]]
 
objects = ['frame'] + triangle + ball
agent = 'agent'

randomizer = 'random_positions'
