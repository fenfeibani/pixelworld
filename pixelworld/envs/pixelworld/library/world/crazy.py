import numpy as np

objects = []

#specify the objects to add
me_position = np.array((5,5))
objects += ['frame',
    ['self', {
        'color': 2,
        'position': me_position,
        }],
    ['basic', {
        'name': 'heavy4',
        'color': 4,
        'mass': 10,
        'velocity': (0,1),
        'position': (3,4),
        }],
    ['basic', {
        'name': 'heavy5',
        'color': 5,
        'mass': 10,
        'velocity': (0,-1),
        'position': (3,4),
        }],
    ['basic', {
        'name': 'light6',
        'color': 6,
        'mass': 1,
        'position': (10,6),
        }],
    ['complex', {
        'children': [
            ['basic', {
                'mass': 10,
                'position': (19, 10),
                }],
            ['basic', {
                'position': (19, 11),
                }],
        ],
        'color': 7,
        'velocity': (-1,0),
        'name': 'c1',
    }],
]

#add some randomly placed objects
objects += 5*[['basic', {'name': 'random'}]]

#create a line of objects to the right of the self
objects += [['complex', {
    'children': [ ['basic', {'color': 3, 'position': me_position + (0, i)}] for i in xrange(1,5)],
    'name': 'line',
}]]

agent = 'random'

randomizer = 'random_positions'
