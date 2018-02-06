import numpy as np

import pixelworld.envs.pixelworld as pw
from ..helpers import h, L


class GripperSelf(pw.objects.SelfObject):
    """The self that grips"""
    _attributes = ['grips']


objects = [['gripper_self', {'color': 2}]] + ['basic'] * 10

objects += [['complex', {
    'children': [ ['basic', {'color': 3, 'position': (10 + 3 * j, 5 + i)}] for i in xrange(1,6)],
    'name': 'line',
}] for j in xrange(3)]


objects += [['complex_sprite', {'sprites': [h.sprite.from_string('44    44'),
                                            h.sprite.from_string('  4444  ')],
                                'animated': True}]]

objects += ['frame']
