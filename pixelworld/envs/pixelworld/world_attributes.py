'''
    basic set of WorldAttributes for PixelWorld
'''
import core, pixel_worlds, objects as objects_mod
import library.helpers as h

class ShowColorsWorldAttribute(core.BooleanWorldAttribute):
    """True to show colors in the world.state, False to show everything as if
    color == 1 or 0"""
    _default_value = True


class KillingDeletesWorldAttribute(core.BooleanWorldAttribute):
    """If True, delete objects when they get 'killed' by setting alive to False."""
    _default_value = False


class ScreenWorldAttribute(core.IntegerWorldAttribute):
    """For screen-based worlds, the curent screen. Setting this attribute causes
    the world to remove all objects and create the objects for the newly chosen
    screen."""
    def _set(self, value):
        """set the attribute value
        
        Parameters
        ----------
        value : int
            the id of the new screen
        """
        assert isinstance(self.world, pixel_worlds.ScreenBasedPixelWorld), \
            'Only ScreenBasedPixelWorlds can have the "screen" world attribute'

        super(ScreenWorldAttribute, self)._set(value)

        if self.world._populated:
            self.world.remove_objects(self.world.objects)

            objects, width, height = h.world.screen(self.world._screens[value], self.world._legend)

            self.world.create_objects(objects)
        else:
            # do nothing, assume the ScreenBasedPixelWorld constructor does all
            # the work
            pass



