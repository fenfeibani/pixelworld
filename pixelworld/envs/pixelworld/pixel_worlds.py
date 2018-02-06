'''
    basic set of PixelWorld subclasses
'''
from copy import copy, deepcopy

import numpy as np

import core
import objects as object_mod
import events
import randomizers
from library.helpers import h, L


class ScreenBasedPixelWorld(core.PixelWorld):
    """A pixel world that allows you to switch screens. When you switch screens,
    all objects are deleted and the new screen's objects are created.
    """
    #a dict of strings indicating screen layouts
    _screens = None
    
    #a dict mapping characters to object specifications
    _legend = None

    #the screen chosen at time of construction, or None if no screen was chosen
    _chosen_screen = None

    def __init__(self, screens=None, legend=None, screen=None, objects=None, **kwargs):
        """
        Parameters
        ----------

        screens : list or dict
            a list of screen layout strings, or a dict whose values are screen
            layout strings and whose keys are ints. See
            library.world._helpers.layout for more information.

        legend : dict
            a dict mapping characters to object specifications. See
            library.world._helpers.layout for more information.

        screen : int or string or None
            The index or key of the screen you want to start with, or None to
            start at screen 0.

        objects : list of object specifications
            Objects to create in addition to those specified by screens.

        width, height : ints (optional)
            If not supplied, will be inferred as the maximum height or width
            among the screens provided. If you supply a height or width that is
            less than this, the screens will only be partially visible.
        """
        if screens is None:
            screens = dict()

        if legend is None:
            legend = dict()

        # if screens is a list, replace it with a dict that maps integer
        # indices to screens.
        if isinstance(screens, list):
            screens = dict(enumerate(screens))
        else:
            for k in screens:
                assert isinstance(k, int), 'dict keys must be integers'

        self._chosen_screen = screen
        self._screens = screens
        self._legend = legend

        if screen is None:
            screen = 0

        # if width not supplied, set width to max of screen widths
        need_width = False
        if 'width' in kwargs and kwargs['width'] is not None:
            width = kwargs['width']
        else:
            need_width = True
            width = 0
                
        # if height not supplied, set height to max of screen heights
        need_height = False
        if 'height' in kwargs and kwargs['height'] is not None:
            height = kwargs['height']
        else:
            need_height = True
            height = 0

        # loop over screens finding max width and max height if we don't have a
        # width or height specified, respectively. Also get the new objects
        # from the selected screen.
        for screen2 in screens:
            screen_objects, screen_height, screen_width = h.world.screen(screens[screen2], legend)
            if need_width:
                width = max(width, screen_width)
            if need_height:
                height = max(height, screen_height)
            if screen2 == screen:
                new_objects = screen_objects

        # take the new objects from the screen and append them to the other
        # objects if any
        if objects is None:
            objects = new_objects
        else:
            objects = objects + new_objects

        kwargs['width'] = width
        kwargs['height'] = height
        kwargs['screen'] = screen

        super(ScreenBasedPixelWorld, self).__init__(objects=objects, **kwargs)

    def change_to_new_screen(self, screen):
        """Add a new screen to the world and immediately change to it.

        Parameters
        ----------
        screen : string
            The screen string. See library.world._helpers.screen for more
            information.

        Returns
        -------
        new_idx : int
            The index of the newly created screen.
        """
        new_idx = len(self._screens)
        while new_idx in self._screens:
            new_idx += 1
        assert new_idx not in self._screens
        self._screens[new_idx] = (screen)
        self.screen = new_idx
        return new_idx

    def _post_step(self):
        """Function that runs after world has been stepped but before we compute new
        observation. In this case, we look for screen change events and do what
        they indicate."""
        for event in self._step_events:
            if isinstance(event, events.ChangeScreenEvent):
                self.screen = event.new_screen

