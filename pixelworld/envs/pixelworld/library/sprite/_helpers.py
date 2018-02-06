import numpy as np

from ..core import import_item

SPRITE_IGNORE = -1
SPRITE_DEFAULT = -2


def from_string(string):
    """generate a sprite from a string
    
    Parameters
    ----------
    string : string
        a string describing a sprite. \n characters separate rows. each sprite
        pixel should be a digit (the color of the pixel). spaces are ignored. 0
        indicates a hidden sprite. 'X' indicates a pixel that should take on the
        color of the object. if the leading or trailing lines are blank, they
        are ignored.
    
    Returns
    -------
    sprite : ndarray
        the sprite array
    """
    #split by row
    rows = string.split("\n")
    
    #remove leading and trailing empty rows
    if rows[0] == '':
        del rows[0]
    if rows[-1] == '':
        del rows[-1]
    
    for idx,row in enumerate(rows):
        rows[idx] = [   SPRITE_IGNORE if char == ' ' else
                        SPRITE_DEFAULT if char == 'X' else
                        int(char) for char in row]
    
    return np.array(rows, dtype=int)

def load(category, name='sprite'):
    """load a sprite from the library
    
    Parameters
    ----------
    category : string
        the name of the sprite module that contains the sprite
    name : string, optional
        the name of the sprite
    """
    module = import_item('sprite', category)
    return getattr(module, name)
