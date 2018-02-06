import numpy as np

import pixelworld.envs.pixelworld as pw

class EnclosureTrueClassWorldAttribute(pw.core.StringWorldAttribute, pw.core.UnchangeableWorldAttribute):
    """The true class of an enclosure problem. Computed on the fly so that if
    e.g. the agent manipulates the environment, the true class can change if
    the right answer changes."""
    def _get(self):
        """Get the true class by doing the comparison.
        """
        block = self.world.objects['block']
        container = self.world.objects['container']
        tlbr = container.extent
        if tlbr[0] <= block.position[0] <= tlbr[2] and tlbr[1] <= block.position[1] <= tlbr[3]:
            return 'ENCLOSED'
        else:
            return 'NOT_ENCLOSED'


class ClassifyObjectAttribute(pw.core.AbilityObjectAttribute):
    _actions = ['CLASSIFY_ENCLOSED', 'CLASSIFY_NOT_ENCLOSED']
    def _execute_action(self, obj, t, dt, agent_id, action):
        if action.startswith('CLASSIFY_'):
            cls = action[len('CLASSIFY_'):]
            if cls == self.world.enclosure_true_class:
                event = pw.events.CorrectClassificationEvent(self.world, truth=self.world.enclosure_true_class,
                                                             guess=cls)
            else:
                event = pw.events.IncorrectClassificationEvent(self.world, truth=self.world.enclosure_true_class,
                                                             guess=cls)
            

class Classifier(pw.objects.SelfObject):
    _attributes = ['classify']

class BlockObject(pw.objects.BasicObject):
    pass

class EnclosureRandomizer(pw.randomizers.ReseedingRandomizer):
    def randomize(self, seed=None):
        super(EnclosureRandomizer, self).randomize(seed=seed)

        self.world.remove_objects(self.world.objects.find(name='block'))
        self.world.remove_objects(self.world.objects.find(name='classifier'))
        self.world.remove_objects(self.world.objects.find(name='container'))

        width, height = np.random.randint(3, 8, size=2)
        shape_string = '\n'.join(['X' * width] + (height - 2) * ['X' + (width - 2) * ' ' + 'X'] + ['X' * width])
        container = self.world.create_object(['complex', dict(name='container', shape=shape_string)])

        block = self.world.create_object(['block', dict(color=3)])

        classifier = self.world.create_object(['classifier', dict(color=2)])

        if self.world.rng.rand() < 0.5:
            while self.world.enclosure_true_class == 'NOT_ENCLOSED':
                block.position = self.world.object_attributes['position']._get_random_value_object(block)
        else:
            while self.world.enclosure_true_class == 'ENCLOSED':
                block.position = self.world.object_attributes['position']._get_random_value_object(block)


class EnclosureWorld(pw.core.PixelWorld):
    def __init__(self, objects=None, randomizer=None, **kwargs):
        objects = ['frame']

        if randomizer is None:
            randomizer = 'enclosure'
        super(EnclosureWorld, self).__init__(objects=objects, randomizer=randomizer, **kwargs)



world = EnclosureWorld
