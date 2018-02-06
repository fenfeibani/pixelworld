import pixelworld.envs.pixelworld as pw


class ClassifiesSameVsDifferentObjectAttribute(pw.core.AbilityObjectAttribute):
    _actions = ['CLASSIFY_SAME', 'CLASSIFY_DIFFERENT']
    """Object attribute that, when added to an object, adds actions to classify as
    SAME or DIFFERENT, and generates appropriate classification events.
    """
    def _execute_action(self, obj, t, dt, agent_id, action):
        """Respond to CLASSIFY_FOO actions, ignore other actions.
        
        Parameters
        ----------
        obj : Object
            The object that has the attribute. If there is more than one you
            will have problems in this case.
        t : number
            The simulation time.
        dt : number
            Time since last step
        action : string
            Action just taken
        """
        if action.startswith('CLASSIFY_'):
            cls = action[len('CLASSIFY_'):]
            if cls == self.world.svd_true_class:
                event = pw.events.CorrectClassificationEvent(self.world, truth=self.world.svd_true_class, guess=cls)
            else:
                event = pw.events.IncorrectClassificationEvent(self.world, truth=self.world.svd_true_class, guess=cls)


class SvdTrueClassWorldAttribute(pw.core.StringWorldAttribute):
    """The true class of a Same Vs. Different Problem. Computed on the fly so that
    if e.g. the agent manipulates the environment, the true class can change if
    the right answer changes."""
    def _get(self):
        """Get the true class by doing the comparison.
        """
        # get the test objects
        objects = [x for x in self.world.objects if isinstance(x, SameVsDifferentTestObject)]
        assert len(objects) == 2

        if self.world._comparison_attribute.compare(
            self.world._comparison_attribute.get(objects[0]), 
            self.world._comparison_attribute.get(objects[1])):
            return 'SAME'
        else:
            return 'DIFFERENT'


class SameVsDifferentRandomizer(pw.core.Randomizer):
    """Given a particular comparison attribute, this class generates random Same
    Vs. Different problems for that attribute.
    """
    _comparison_attribute = None
    def __init__(self, world, attr):
        """
        Parameters
        ----------
        world : PixelWorld
        attr : string
            Name of attribute to compare.
        """
        self._comparison_attribute = attr
        super(SameVsDifferentRandomizer, self).__init__(world)

    def randomize(self):
        """Generate a random Same Vs. Different problem. Assumes that the test objects
        already exist.
        """
        attr = self.world.object_attributes[self._comparison_attribute]

        # get the test objects
        objects = [x for x in self.world.objects if isinstance(x, SameVsDifferentTestObject)]
        assert len(objects) == 2

        # generate a SAME problem and assert that we succeeded
        if self.world.rng.rand() < 0.5:
            attr.set(objects[0], attr._get_random_values([objects[0].id]))
            attr.set(objects[1], attr.get(objects[0]))
            assert self.world.svd_true_class == 'SAME'

        # generate a DIFFERENT problem and assert that we succeeded
        else:
            for obj in objects:
                attr.set(obj, attr._get_random_values([obj.id]))
            while attr.compare(attr.get(objects[0]), attr.get(objects[1])):
                attr.set(objects[1], attr._get_random_values([objects[1].id]))
            assert self.world.svd_true_class == 'DIFFERENT'


class SameVsDifferentClassifierObject(pw.core.Object):
    """A non-physical object that will add SAME vs. DIFFERENT classification to the
    actions and handle them appropriately. You should only create one of these."""
    _attributes = ['classifies_same_vs_different']


class SameVsDifferentTestObject(pw.core.Object):
    """An object that should be considered one of the test objects in a Same
    Vs. Different problem. It is assumed (with asserts) in several places that
    exactly two of these exist.
    """
    pass


class SameVsDifferentPixelWorld(pw.core.PixelWorld):
    """A PixelWorld that makes it easy to define Same Vs. Different problems. 

    The answer to the problem is available as world.svd_true_class.
    """
    __bases__ = (pw.core.PixelWorld,)

    # the class of objects to create for experiments
    _object_class = None

    # the attribute that should be compared between objects
    _comparison_attribute = None

    def __init__(self, objects=None, object_class=None, attribute=None, same=None, **kwargs):
        """
        Parameters
        ----------
        objects : list of object specifications (see Entity.get_instance)
            Objects to create in addition to the test objects and a classifier object.
        object_class : string or class
            The class of object you want the two test objects to be.
        attribute : string or class
            The attribute that you want to use to compare the objects.
        same : bool or "SAME" or "DIFFERENT"
            Whether to generate a 'same' instance or a 'different'
            instance. This will not be respected during reset.
        """

        if objects is None:
            objects = []

        # if object_class is not specified, use the one defined in the class or
        # subclass
        if object_class is None:
            if self._object_class is None:
                object_class = self._object_class = 'basic'
            else:
                object_class = self._object_class
        else:
            self._object_class = object_class

        # if attribute is not specified, use the one defined in the class or
        # subclass
        if attribute is None:
            if self._comparison_attribute is None:
                attribute = self._comparison_attribute = 'color'
            else:
                attribute = self._comparison_attribute
        else:
            self._comparison_attribute = attribute

        # parse the attribute
        if type(attribute) != str:
            self._comparison_attribute = pw.core.ObjectAttribute.get_instance(self, attribute)
            attribute = self._comparison_attribute.name

        # create a special class that is marked as being a test object
        object_class = pw.core.Object.get_class(object_class)
        class SpecializedSameVsDifferentTestObject(object_class, SameVsDifferentTestObject):
            _name = 'svd_test'
            _attributes = [attribute]

        # add the class to globals so that we can be pickled
        globals().update(dict(SpecializedSameVsDifferentTestObject=SpecializedSameVsDifferentTestObject))

        # assemble list of objects
        objects = objects + [SpecializedSameVsDifferentTestObject] * 2 + ['same_vs_different_classifier']

        # initialize
        super(SameVsDifferentPixelWorld, self).__init__(objects=objects, **kwargs)

        # get the attribute
        if type(attribute) == str:
            try:
                self._comparison_attribute = self.object_attributes[attribute]
            except KeyError:
                self._comparison_attribute = pw.core.ObjectAttribute.get_instance(self, attribute)
        else:
            self._comparison_attribute = pw.core.ObjectAttribute.get_instance(self, attribute)
            attribute = self._comparison_attribute.name

        # create the randomizer
        self._randomizer = SameVsDifferentRandomizer(self, attribute)

        # if same is None, flip a coin
        if same is None:
            same = (self.rng.rand() < 0.5)

        # if same is not a string, evaluate it as a bool where True means SAME
        # and False means DIFFERENT
        if not isinstance(same, str):
            same = 'SAME' if same else 'DIFFERENT'

        # check that string is valid
        assert same == 'SAME' or same == 'DIFFERENT', \
            'Only valid string values for "same" are "SAME" and "DIFFERENT"'

        # rerandomize until we get the right true class
        self.svd_true_class = ''
        self._randomizer.randomize()
        while self.svd_true_class != same:
            self._randomizer.randomize()



world = SameVsDifferentPixelWorld
object_class = 'basic'
attribute = 'color'
objects = ['self']
