#simple compound shape
thing = [['complex', {'children': 3*['basic'], 'velocity': (0, -1)}]]
 
#bouncing red ball
ball = [['basic', {'velocity': (-1,1), 'color': 2}]]
 
objects = ['frame'] + thing + ball

seed = 12346
agent = 'agent'

randomizer = 'random_positions'
