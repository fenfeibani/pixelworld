from copy import copy
import numpy as np


def layout(layout_string, legend=None):
    """construct an entity layout from a string
    
    Parameters
    ----------
    layout_string : string
        a string describing a layout of entities in a 2D grid. \n characters
        separate rows. spaces are ignored. if the leading or trailing lines are
        blank, they are ignored.
    legend : dict, optional
        an optional mapping from characters to entity descriptions
    
    Returns
    -------
    entities : list
        a list of the entities in the layout
    positions : list[(y, x)]
        a list of tuples specifying the grid positions of the entities
    """
    assert isinstance(layout_string, basestring), 'layout_string must be a string'
    
    if legend is None:
        legend = {}
    assert isinstance(legend, dict), 'legend must be a dict'
    
    #split by row
    rows = layout_string.split("\n")
    
    #remove leading and trailing empty rows
    if rows[0] == '':
        del rows[0]
    if rows[-1] == '':
        del rows[-1]
    
    #construct the lists of entities and positions
    entities = []
    positions = []
    for r,row in enumerate(rows):
        for c,char in enumerate(row):
            if char != ' ':
                positions.append((r, c))
                entities.append(legend.get(char, char))
    
    return entities, positions

def screen(layout_string, object_specs):
    """construct the specifications for a screen full of Objects. see pixelzuma
    for an example.
    
    Parameters
    ----------
    layout_string : string
        see layout()
    object_specs : dict
        a dict mapping characters to Object specifications
    
    Returns
    -------
    objects : list
        a list of specifications for each Object on the screen
    height : int
        the height of the screen
    width : int
        the width of the screen
    """
    #flesh out the object specifications
    object_specs = object_specs.copy()
    for key,spec in object_specs.iteritems():
        if isinstance(spec, basestring):
            spec = [spec, {}]
        
        assert isinstance(spec, list) and len(spec) > 0, 'invalid object specification'
        
        if spec[0] == 1:
            spec.append({})
        
        assert len(spec) == 2, 'invalid object specification'
        
        object_specs[key] = spec
    
    #parse the layout
    objects, positions = layout(layout_string, object_specs)
    height = max([p[0] for p in positions]) + 1
    width = max([p[1] for p in positions]) + 1
    
    #get a unique copy for each object
    objects = [copy(obj) for obj in objects]
    
    #add the positions to the object specifications
    for obj,pos in zip(objects, positions):
        obj[1] = obj[1].copy()
        obj[1]['position'] = pos
    
    return objects, height, width

def shape(y, x=None, center=True):
    """generate the positions for a shape
    
    Parameters
    ----------
    y : int | ndarray | string
        one of the following:
            int: a fixed y position
            ndarray: a range of y positions
            string: an input to layout(). the characters don't matter.
    x : int | ndarray, optional
        if y is not a string, then the x positions corresponding to y
    center : bool, optional
        True to center the shape around (0, 0)
    
    Returns
    -------
    shp : list[(y, x)]
        a list of (y, x) position tuples
    """
    if isinstance(y, basestring):  # layout input
        #get the uncentered points
        entities, shp = layout(y)
        
        #center the shape around (0, 0)
        if center:
            shp = shp - ((np.min(shp, axis=0) + np.max(shp, axis=0)) / 2).astype(int)
            #shp = shp - (np.ptp(shp, axis=0) / 2).astype(int)
        
        return tuple(shp)
    else:
        if np.isscalar(y):
            if np.isscalar(x):
                x = np.array([x], dtype=type(x))
            
            y = np.tile(y, len(x))
        if np.isscalar(x):
            x = np.tile(x, len(y))
        
        #center the shape around (0, 0)
        if center:
            y -= ((np.min(y) + np.max(y)) / 2).astype(int)
            x -= ((np.min(x) + np.max(x)) / 2).astype(int)
        
        return zip(y, x)
