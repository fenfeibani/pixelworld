'''
    basic set of Variants for PixelWorld
'''
import core


class ShowColorsVariant(core.BooleanVariant, core.WorldAttributeVariant):
    """determines whether Object colors affect world.state
    
    True: show colors
    False: everything is monochrome
    """
    pass
