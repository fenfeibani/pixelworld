import numpy as np

import pixelworld.envs.pixelworld as pw

class TouchingTrueClassWorldAttribute(pw.core.StringWorldAttribute, pw.core.UnchangeableWorldAttribute):
    """The true class of a Same Vs. Different Problem. Computed on the fly so that
    if e.g. the agent manipulates the environment, the true class can change if
    the right answer changes."""
    def _get(self):
        """Get the true class by doing the comparison.
        """
        objects = [x for x in self.world.objects if x._name == 'basic']
        assert len(objects) == 2
        if np.sum(np.abs(objects[0].position - objects[1].position)) <= 1.0:
            return 'TOUCHING'
        else:
            return 'NOT_TOUCHING'


class ClassifyObjectAttribute(pw.core.AbilityObjectAttribute):
    _actions = ['CLASSIFY_TOUCHING', 'CLASSIFY_NOT_TOUCHING']
    def _execute_action(self, obj, t, dt, agent_id, action):
        if action.startswith('CLASSIFY_'):
            cls = action[len('CLASSIFY_'):]
            if cls == self.world.touching_true_class:
                event = pw.events.CorrectClassificationEvent(self.world, truth=self.world.touching_true_class,
                                                             guess=cls)
            else:
                event = pw.events.IncorrectClassificationEvent(self.world, truth=self.world.touching_true_class,
                                                             guess=cls)
            

class Classifier(pw.objects.SelfObject):
    _attributes = ['classify']


class TouchingRandomizer(pw.randomizers.ReseedingRandomizer):
    def randomize(self, seed=None):
        super(TouchingRandomizer, self).randomize(seed=seed)

        blocks = self.world.objects['basic']
        assert len(blocks) == 2
        classifier = self.world.objects['classifier']
        color1 = color2 = self.world.rng.randint(3, 10)
        while color1 == color2:
            color2 = self.world.rng.randint(3, 10)
        blocks[0].color = color1
        blocks[1].color = color2
        blocks[0].position = self.world.object_attributes['position']._get_random_value_object(blocks[0])
        blocks[1].position = blocks[0].position + (0, 1)

        if self.world.rng.rand() < 0.5:
            pass
        else:
            while np.sum(np.abs(blocks[0].position - blocks[1].position)) <= 1.0:
                blocks[0].position = self.world.object_attributes['position']._get_random_value_object(blocks[0])

class TouchingWorld(pw.core.PixelWorld):
    def __init__(self, objects=None, randomizer=None, **kwargs):
        objects = ['frame', ['basic', dict(color=3)], ['basic', dict(color=4)],
                   ['classifier', dict(color=2)]]

        if randomizer is None:
            randomizer = 'touching'
        super(TouchingWorld, self).__init__(objects=objects, randomizer=randomizer, **kwargs)

        self.randomizer.randomize()


world = TouchingWorld
