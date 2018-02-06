import numpy as np

from pixelworld.envs.pixelworld import core, objects, randomizers


class Packed(core.BooleanObjectAttribute, core.DerivedObjectAttribute, core.ChangeTrackingObjectAttribute):
    """indicates whether a product is packed"""
    _default_value = False
    
    _depends_on = ['position']
    
    _step_before = 'position'
    
    def _get_data_object(self, obj):
        return obj.position[1] == self.world.width - 2


class Product(objects.BasicObject):
    """a product that needs packing"""
    _attributes = ['packed']
    _defaults = {'color': lambda obj: obj.rng.randint(3, 10)}


class PackedGoal(core.Goal):
    packed = None
    
    def prepare(self):
        self.packed = self.world.object_attributes['packed']
    
    def _is_achieved(self):
        return np.all(self.packed())


class PixelPackJudge(core.Judge):
    """a judge that rewards packed products and penalizes inefficiency"""
    packed = None

    def prepare(self):
        self.packed = self.world.object_attributes['packed']
        
    def _calculate_reward(self, goals, events):
        return 100 * np.sum(self.packed.change(step=True))


class PixelPackRandomizer(randomizers.RandomPositionsRandomizer):
    _excluded_objects = ['wall']

height = width = 20

shelves = [ ['wall', {'position': (y, width - 2)}] for y in xrange(2, height - 1, 2)]
products = ['product'] * ((height - 1) / 2)
worker = [['self', {'color': 2}]]

objects = ['frame'] + shelves + products + worker
goals = ['packed']
judge = PixelPackJudge
randomizer = PixelPackRandomizer
