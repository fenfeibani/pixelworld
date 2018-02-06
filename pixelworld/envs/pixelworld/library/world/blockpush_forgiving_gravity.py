#give everything a positive y-direction acceleration
base_prop = {'acceleration': (0.025, 0)}

#randomly placed blocks
blocks = 10*[['basic', base_prop]]

#moveable "agent"
me = [['basic_self', dict(base_prop, color=2)]]

objects = ['frame'] + blocks + me
randomizer = 'random_positions'
