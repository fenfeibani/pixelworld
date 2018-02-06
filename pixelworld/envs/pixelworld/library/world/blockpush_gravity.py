#randomly placed blocks
blocks = 10*[['basic', {'acceleration': (1, 0)}]]

#moveable "agent"
me = [['basic_self', {'color': 2, 'acceleration': (1, 0)}]]

objects = ['frame'] + blocks + me
randomizer = 'random_positions'
