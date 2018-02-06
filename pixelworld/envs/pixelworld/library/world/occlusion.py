import numpy as np

import pixelworld.envs.pixelworld as pw

class OcclusionTrueClassWorldAttribute(pw.core.StringWorldAttribute, pw.core.UnchangeableWorldAttribute):
    """The true class of a Same Vs. Different Problem. Computed on the fly so that
    if e.g. the agent manipulates the environment, the true class can change if
    the right answer changes."""
    def _get(self):
        """Get the true class by doing the comparison.
        """
        objects = [x for x in self.world.objects if x._name == 'basic']
        assert len(objects) == 2
        if (objects[0].state_index == objects[1].state_index).all():
            return 'OCCLUSION'
        else:
            return 'NOT_OCCLUSION'


class ClassifyObjectAttribute(pw.core.AbilityObjectAttribute):
    _actions = ['CLASSIFY_OCCLUSION', 'CLASSIFY_NOT_OCCLUSION']
    def _execute_action(self, obj, t, dt, agent_id, action):
        if action.startswith('CLASSIFY_'):
            cls = action[len('CLASSIFY_'):]
            if cls == self.world.occlusion_true_class:
                event = pw.events.CorrectClassificationEvent(self.world, truth=self.world.occlusion_true_class,
                                                             guess=cls)
            else:
                event = pw.events.IncorrectClassificationEvent(self.world, truth=self.world.occlusion_true_class,
                                                             guess=cls)
            

class Classifier(pw.objects.SelfObject):
    _attributes = ['classify']


class OcclusionRandomizer(pw.randomizers.ReseedingRandomizer):
    def randomize(self, seed=None):
        super(OcclusionRandomizer, self).randomize(seed=seed)

        blocks = self.world.objects['basic']
        assert len(blocks) == 2
        classifier = self.world.objects['classifier']
        color1 = color2 = self.world.rng.randint(3, 10)
        while color1 == color2:
            color2 = self.world.rng.randint(3, 10)
        blocks[0].color = color1
        blocks[1].color = color2
        blocks[0].position = self.world.object_attributes['position']._get_random_value_object(blocks[0])
        blocks[0].depth = 0
        blocks[1].depth = 1
        blocks[1].position = blocks[0].position

        if self.world.rng.rand() < 0.5:
            pass
        else:
            while (blocks[0].position == blocks[1].position).all():
                blocks[0].position = self.world.object_attributes['position']._get_random_value_object(blocks[0])

class OcclusionWorld(pw.core.PixelWorld):
    def __init__(self, objects=None, randomizer=None, **kwargs):
        if objects is None:
            objects = []

        objects += ['frame', ['basic', dict(color=3)], ['basic', dict(color=4)],
                   ['classifier', dict(color=2)]]

        if randomizer is None:
            randomizer = 'occlusion'
        super(OcclusionWorld, self).__init__(objects=objects, randomizer=randomizer, depth=2, **kwargs)

        self.randomizer.randomize()


world = OcclusionWorld
