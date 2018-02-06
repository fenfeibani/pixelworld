from ..helpers import h, L

height = 100
width = 50

invader = [['self_sprite', {'name': 'invader', 'sprites': h.sprite.load('aliens', 'alien'), 'animated': True}]]

objects = ['frame'] + invader
rate = L('rate', 3)
agent = ['human', {'rate': rate}]

randomizer = 'random_positions'
