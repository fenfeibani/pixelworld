import numpy as np

from pixelworld.envs.pixelworld import core, library, objects, object_attributes, randomizers

pixelpack = library.import_item('world', 'pixelpack')

from pixelpack import Packed, Product, height, width, shelves, products, \
        worker, goals, judge, randomizer

class Progress(core.FloatObjectAttribute, core.DerivedObjectAttribute, core.SteppingObjectAttribute):
    """keeps track of the worker's packing progress"""
    _depends_on = ['packed', 'color']
    
    _step_after = ['pushes']
    
    def prepare(self):
        for obj in self.objects:
            self._set_indicator_colors(obj)
    
    def _get_normalized_height(self, obj):
        y, x = obj.position
        h = self.world.height
        return (h - y - 1)/(h - 2)
    
    def _set_indicator_colors(self, obj):
        progress = self.get(obj)
        
        for child in obj._children:
            height = self._get_normalized_height(child)
            child.color = 2 if height <= progress else 0
    
    def _get_data_object(self, obj):
        packed = self._other_attr['packed'].get()
        return np.sum(packed) / float(len(packed))
    
    def _step_object(self, obj, t, dt, agent_id, action):
        self._set_indicator_colors(obj)


class Indicator(core.CompoundObject):
    """indicates packing progress with a color bar"""
    _attributes = ['progress']
    
    def __init__(self, world, *args, **kwargs):
        kwargs['children'] = [ ['immoveable', {'position': (y, 0)}] for y in xrange(1, world.height - 1)]
        
        super(Indicator, self).__init__(world, *args, **kwargs)


frame = [['frame', {'sides': ['top', 'right', 'bottom']}]]

objects = frame + shelves + products + worker + ['indicator']
