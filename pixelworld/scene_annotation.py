from __future__ import print_function
from __future__ import absolute_import

import numpy as np



### API ###
class Annotation(np.ndarray):
    """
    An annotation is a thin wrapper around a numpy array that
    associates string `labels` with the integer values of the
    array.

    Parameters
    ----------
    labels : { str : int }
    """
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, labels=None):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides,
                         order)
        obj.labels = labels
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.labels = getattr(obj, 'labels', None)

    def get_points(self, label):
        v = self.labels[label]
        return np.argwhere(self == v)



class SceneAnnotator(object):
    """
    Given a numpy array representing a scene, produce a
    numpy array of the same shape with integer labels
    annotating the scene.
    """
    def annotate(self, scene):
        """
        Parameters
        ----------
        scene : np.ndarray

        Returns
        -------
        annotation : Annotation
        """
        raise NotImplementedError()


### Custom Scene Annotators ###
class ContainmentAnnotator(SceneAnnotator):
    """
    A container is an object that intuitively could hold liquid.

    Annotations are performed by literally simulating "liquid" flowing
    throughout the scene. After no changes are observed, pixels
    where liquid remains are considered to be in the interior of a
    container. Any adjacent solids to the interior are marked as part
    of a container, and any adjacent solids to the container are marked
    as part of the container as well.

    Labels
    ------
    2 : not involved in containment
    3 : part of the container
    4 : liquid contained
    5 : contained liquid is directly adjacent to the floor 
    """
    ORIG_SPACE = 0
    ORIG_SOLID = 1

    TEMP_LIQUID = -2

    LABEL_SPACE = 2
    LABEL_CONTAINER = 3
    LABEL_INTERIOR = 4
    LABEL_FLOOR_ADJACENT = 5

    LABELS = {
        'space' : LABEL_SPACE,
        'container' : LABEL_CONTAINER,
        'interior' : LABEL_INTERIOR,
        'inside_supported' : LABEL_FLOOR_ADJACENT
    }

    def __init__(self, flow_dirs, label_floor_adjacent=True):
        self.flow_dirs = flow_dirs
        self.label_floor_adjacent = label_floor_adjacent

    def _plot_scene(self, scene):
        from matplotlib import colors
        from matplotlib import pyplot as plt

        cdict = {
            self.ORIG_SPACE : 'white',
            self.LABEL_SPACE : 'gray',
            self.ORIG_SOLID : 'black',
            self.LABEL_CONTAINER : 'red',
            self.TEMP_LIQUID : 'blue',
            self.LABEL_INTERIOR : 'green',
            self.LABEL_FLOOR_ADJACENT : 'yellow'
        }

        clist = list(set(cdict.values()))

        cmap = colors.ListedColormap(clist)
        bounds= range(len(clist)+1)
        norm = colors.BoundaryNorm(bounds, cmap.N)

        img = scene.copy()

        for i, c in cdict.iteritems():
            c_idx = clist.index(c)
            img[scene == i] = c_idx

        plt.imshow(img, cmap=cmap, norm=norm, interpolation='nearest')

    def _spread_liquid(self, scene, r, c):
        """
        Helper method for _flow.
        """
        assert scene[r, c] == self.TEMP_LIQUID

        # Spread the liquid randomly.
        np.random.shuffle(self.flow_dirs)

        for i,j in self.flow_dirs:
            try:
                if r+i < 0 or c+j < 0:
                    raise IndexError()

                if scene[r+i, c+j] == self.LABEL_SPACE:
                    return (r+i, c+j)

            # If the position is out of bounds, let the liquid flow away
            except IndexError:
                return None

        return (r, c)

    def _flow(self, scene):
        """
        Allow liquid to flow throughout the scene
        under the pressure of gravity for one timestep.

        For each liquid pixel, try to move down, left, right.
    
        Returns
        -------
        flow_change : bool
            Whether or not any liquid flowed.
        """
        new_scene = np.copy(scene)
        new_scene[new_scene == self.TEMP_LIQUID] = self.LABEL_SPACE

        for r in xrange(scene.shape[0]):
            for c in xrange(scene.shape[1]):
                # Found a liquid pixel.
                if scene[r,c] == self.TEMP_LIQUID:
                    next_pos = self._spread_liquid(scene, r, c)
                    if next_pos is not None:
                        rn, cn = next_pos
                        new_scene[rn, cn] = self.TEMP_LIQUID

        flow_change = not np.array_equal(scene, new_scene)
        scene[:] = new_scene[:]
        return flow_change

    def _mark_adjacent_pixels(self, scene, r, c):
        """
        Helper method for _mark_adjacent_solids.
        """
        for i,j in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            try:
                if r+i < 0 or c+j < 0:
                    raise IndexError()

                if scene[r+i, c+j] == self.ORIG_SOLID:
                    scene[r+i, c+j] = self.LABEL_CONTAINER

            # Ignore out of bounds positions
            except IndexError:
                pass

    def _mark_adjacent_solids(self, scene, adjacent_to=None):
        if not isinstance(adjacent_to, list):
            adjacent_to = [adjacent_to]

        for r in xrange(scene.shape[0]):
            for c in xrange(scene.shape[1]):
                for adj_to in adjacent_to:
                    # Found an adjacent_to pixel
                    if scene[r, c] == adj_to:
                        self._mark_adjacent_pixels(scene, r, c)
                        break

    def annotate(self, scene, debug=False):
        # -1. Convert to numpy array.
        scene = np.array(scene, dtype=np.int)

        if debug:
            from matplotlib import pyplot as plt

            plt.ion()
            self._plot_scene(scene)
            plt.pause(0.2)

        # 0. Make sure the scene is binary.
        assert ((scene==self.ORIG_SPACE) | (scene==self.ORIG_SOLID)).all()

        annotation = np.copy(scene).view(Annotation)
        annotation.labels = self.LABELS

        # 1. Fill the spaces with 'liquid'.
        annotation[annotation == self.ORIG_SPACE] = self.TEMP_LIQUID

        if debug:
            self._plot_scene(annotation)
            plt.pause(0.5)

        # 2. Let the liquid flow.
        flow_change = True
        while flow_change:
            flow_change = self._flow(annotation)

            if debug:
                self._plot_scene(annotation)
                plt.pause(0.0005)

        # 2.5. Mark liquid with no other liquid below it as 'floor adjacent'.
        if self.label_floor_adjacent:
            liquid_floor_mask = annotation == self.TEMP_LIQUID
            liquid_floor_mask[:-1] = np.logical_and(liquid_floor_mask[:-1], 
                                                    np.logical_not(liquid_floor_mask[1:]))
            annotation[liquid_floor_mask] = self.LABEL_FLOOR_ADJACENT

            if debug:
                self._plot_scene(annotation)
                plt.pause(0.0005)

        # 3. Mark the liquid that remains as part of the interior.
        annotation[annotation == self.TEMP_LIQUID] = self.LABEL_INTERIOR

        if debug:
            self._plot_scene(annotation)
            plt.pause(0.5)

        # 4. Mark solids adjacent to the interior.
        self._mark_adjacent_solids(annotation, adjacent_to=[self.LABEL_INTERIOR, self.LABEL_FLOOR_ADJACENT])

        if debug:
            self._plot_scene(annotation)
            plt.pause(0.5)

        # 5. Mark solids adjacent to the already marked solids.
        self._mark_adjacent_solids(annotation, adjacent_to=self.LABEL_CONTAINER)

        if debug:
            self._plot_scene(annotation)
            plt.pause(0.5)        

        # 6. Remove any other solids.
        annotation[annotation == self.ORIG_SOLID] = self.LABEL_SPACE

        if debug:
            self._plot_scene(annotation)
            plt.pause(0.5)

        return annotation


class HoleyContainmentAnnotator(ContainmentAnnotator):
    """
    An annotator that permits at least one hole in a container,
    and otherwise labels it in the same manner as the ContainmentAnnotator.
    """
    def _plug_holes(self, scene):
        """
        Assumes that the scene originally contained exactly one container,
        but then at least one hole was poked in it.

        Recovers the original container by identifying the dimensions of the walls,
        which are assumed to exist in the directions of the flow.

        Also critically assumes that the base of the container is at the bottom of
        the scene.
        """

        plugged_scene = np.copy(scene)
        base_row = plugged_scene.shape[0]-1

        # Fill in missing left or right wall pixel first
        if sum(plugged_scene[:,-1]) == 0:  # empty right wall
            plugged_scene[base_row-1,-1] = 1
        if sum(plugged_scene[:,0]) == 0: # empty left wall
            plugged_scene[base_row-1,0] = 1    

        container_pixels = np.transpose(np.nonzero(plugged_scene))
        
        # 1. Find the base       
        min_col = np.min(container_pixels, axis=0)[1]
        max_col = np.max(container_pixels, axis=0)[1]
        base = [(base_row, col) for col in range(min_col, max_col+1)]

        # 2. Find the left side
        left_side_corner = min([r for r, c in container_pixels if c == min_col])
        left_side_corner = min(left_side_corner,base_row-1) 
        left_line = [(row, min_col) for row in range(left_side_corner, base_row+1)]

        # 3. Find the right side
        right_side_corner = min([r for r, c in container_pixels if c == max_col])
        right_side_corner = min(right_side_corner,base_row-1)
        right_line = [(row, max_col) for row in range(right_side_corner, base_row+1)]

        # 4. Draw the lines
        for r,c in base + left_line + right_line:
            plugged_scene[r,c] = 1

        # Fix corners
        plugged_scene[base_row,0] = scene[base_row,0]
        plugged_scene[base_row,-1] = scene[base_row,-1]

        return plugged_scene

    def annotate(self, scene, debug=False):
        plugged_scene = self._plug_holes(scene)
        annotation = super(HoleyContainmentAnnotator, self).annotate(plugged_scene, debug=debug)
        annotation[(np.array(scene) == self.ORIG_SPACE) & (annotation == self.LABEL_CONTAINER)] = self.LABEL_SPACE  
        annotation[(np.array(scene) == self.ORIG_SOLID)] = self.LABEL_CONTAINER

        # Make sure that inside_supported is not on top of a hole
        for j in range(0,annotation.shape[1]): 
            if annotation[-2,j] == self.LABEL_FLOOR_ADJACENT and annotation[-1,j] == self.LABEL_SPACE:
                annotation[-2,j] = self.LABEL_INTERIOR  

        # Add floor-adjacent pixels for corners
        if self.label_floor_adjacent:
            left_wall = annotation[0:-1,0]
            if all(left_wall == self.LABEL_SPACE):  # shape is a lower right corner
                for j in range(0,annotation.shape[1]):
                    if annotation[-1,j] == self.LABEL_CONTAINER: annotation[-2,j] = self.LABEL_FLOOR_ADJACENT                 
            right_wall = annotation[0:-1,-1]
            if all(right_wall == self.LABEL_SPACE):  # shape is a lower left corner
                for j in range(0,annotation.shape[1]):
                    if annotation[-1,j] == self.LABEL_CONTAINER: annotation[-2,j] = self.LABEL_FLOOR_ADJACENT    
           
        return annotation



class HoleyOrientedContainmentAnnotator(ContainmentAnnotator):

    def __init__(self, flow_dirs, orientation_dir):
        self.orientation_dir = orientation_dir
        super(HoleyOrientedContainmentAnnotator,self).__init__(flow_dirs,label_floor_adjacent=False)

    def _plug_holes(self,scene):
        # Plug holes in the oriented bottom and 1 pixel up on each side wall
        plugged_scene = np.copy(scene)
        # 1. Fill in "bottom" of container  (-1,0)=top, (0,-1)=left (0,1)=right  
        if self.orientation_dir == (-1,0): # top
            plugged_scene[0,1:-1] = 1
            plugged_scene[1,0] = 1
            plugged_scene[1,-1] = 1
        elif self.orientation_dir == (0,-1): # left
            plugged_scene[1:-1,0] = 1
            plugged_scene[0,1] = 1
            plugged_scene[-1,1] = 1
        elif self.orientation_dir == (0,1): # right
            plugged_scene[1:-1,-1] = 1 
            plugged_scene[0,-2] = 1
            plugged_scene[-1,-2] = 1   
        return plugged_scene        

    def annotate(self,scene,debug=False):    
        plugged_scene = self._plug_holes(scene)
        annotation = super(HoleyOrientedContainmentAnnotator, self).annotate(plugged_scene, debug=debug)
        annotation[(np.array(scene) == self.ORIG_SPACE) & (annotation == self.LABEL_CONTAINER)] = self.LABEL_SPACE  
        annotation[(np.array(scene) == self.ORIG_SOLID)] = self.LABEL_CONTAINER
        return annotation


class HoleyEnclosureAnnotator(ContainmentAnnotator):

    def _plug_holes(self,scene):
        plugged_scene = np.copy(scene)
        plugged_scene[1:-1,0] = 1   # left wall
        plugged_scene[1:-1,-1] = 1  # right wall
        plugged_scene[0,1:-1] = 1   # top wall
        plugged_scene[-1,1:-1] = 1  # bottom wall
        return plugged_scene

    def annotate(self,scene,debug=False):
        plugged_scene = self._plug_holes(scene)
        annotation = super(HoleyEnclosureAnnotator,self).annotate(plugged_scene,debug=debug)
        # print(annotation)
        # import pdb; pdb.set_trace()        
        annotation[(np.array(scene) == self.ORIG_SPACE) & (annotation == self.LABEL_CONTAINER)] = self.LABEL_SPACE     
        annotation[(np.array(scene) == self.ORIG_SOLID)] = self.LABEL_CONTAINER
        return annotation



### Library ###
ANNOTATOR_LIBRARY = {
    'containment' : [ContainmentAnnotator, {'flow_dirs' : [(1, 0), (0, -1), (0, 1)]}],
    'enclosure' : [ContainmentAnnotator, {'flow_dirs' : [(-1, 0), (1, 0), (0, -1), (0, 1)]}],
    'left_containment' : [ContainmentAnnotator, {'flow_dirs' : [(-1, 0), (1, 0), (0, -1)]}],
    'right_containment' : [ContainmentAnnotator, {'flow_dirs' : [(-1, 0), (1, 0), (0, 1)]}],
    'top_containment' : [ContainmentAnnotator, {'flow_dirs' : [(-1, 0), (0, -1), (0, 1)]}],
    'holey_containment' : [HoleyContainmentAnnotator, {'flow_dirs' : [(1, 0), (0, -1), (0, 1)]}],
    'holey_enclosure' : [HoleyEnclosureAnnotator, {'flow_dirs' : [(-1, 0), (1, 0), (0, -1), (0, 1)]}],
    'left_noncontainment' : [HoleyOrientedContainmentAnnotator, 
                                {'flow_dirs' : [(-1, 0), (1, 0), (0, -1)], 'orientation_dir': (0, -1)}],
    'right_noncontainment' : [HoleyOrientedContainmentAnnotator, 
                                {'flow_dirs' : [(-1, 0), (1, 0), (0, 1)], 'orientation_dir': (0, 1)}],
    'top_noncontainment' : [HoleyOrientedContainmentAnnotator, 
                                {'flow_dirs' : [(-1, 0), (0, -1), (0, 1)], 'orientation_dir': (-1, 0)}],    
}


### Helper functions ###
def generate_annotations(scenes, annotator_name, **kwargs):
    annotator_class = ANNOTATOR_LIBRARY[annotator_name]

    if isinstance(annotator_class, list):
         annotator_class, lib_kwargs = annotator_class
         kwargs.update(lib_kwargs)
    assert isinstance(annotator_class, SceneAnnotator.__class__)

    annotator = annotator_class(**kwargs)
    return [annotator.annotate(scene) for scene in scenes]


### Tests ###
def check_annotations(target, annotation):
    check = np.array_equal(target, annotation)
    if not check:
        print("target is")
        print(target)
        print("annotation is")
        print(annotation)
        raise Exception("Annotation check failed")


def test_containment_annotator(visualize=False):

    annotator = ContainmentAnnotator([(1, 0), (0, -1), (0, 1)])

    # Most straightforward test
    scene1 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 0]]

    target1 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 3, 3, 3, 3, 3, 2]]

    annotation1 = annotator.annotate(scene1, debug=visualize)

    check_annotations(target1, annotation1)

    # One wall lower than the other
    scene2 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 0]]

    target2 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 2, 2, 2, 2, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 3, 3, 3, 3, 3, 2]]

    annotation2 = annotator.annotate(scene2, debug=visualize)

    check_annotations(target2, annotation2)

    # Clipped corners
    scene3 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 0, 1, 1, 1, 0, 0]]

    target3 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 2, 2, 2, 2, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 2, 3, 3, 3, 2, 2]]

    annotation3 = annotator.annotate(scene3, debug=visualize)

    check_annotations(target3, annotation3)

    # Closed tunnel in the floor
    scene4 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 0, 1, 0, 1, 0, 0],
              [0, 1, 1, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 0]]

    target4 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 2, 2, 2, 2, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 2, 3, 4, 3, 2, 2],
               [2, 3, 3, 4, 3, 2, 2],
               [3, 4, 4, 4, 4, 3, 2],
               [2, 3, 3, 3, 3, 3, 2]]

    annotation4 = annotator.annotate(scene4, debug=visualize)

    check_annotations(target4, annotation4)

    # Multiple containers
    scene5 = [[0, 1, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1, 0, 0],
              [0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 1, 1, 1, 0]]

    target5 = [[2, 3, 4, 4, 3, 2, 2],
               [2, 3, 4, 4, 3, 2, 2],
               [2, 3, 3, 3, 3, 2, 2],
               [2, 2, 2, 2, 2, 2, 2],
               [2, 2, 2, 3, 4, 3, 2],
               [2, 2, 2, 3, 4, 3, 2],
               [2, 2, 2, 3, 3, 3, 2]]

    annotation5 = annotator.annotate(scene5, debug=visualize)

    check_annotations(target5, annotation5)

    # Multiple containers + noise
    scene6 = [[0, 1, 0, 0, 1, 0, 1],
              [0, 1, 0, 0, 1, 0, 0],
              [0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 1, 0],
              [0, 1, 0, 1, 0, 1, 0],
              [1, 1, 0, 1, 1, 1, 0]]

    target6 = target5

    annotation6 = annotator.annotate(scene6, debug=visualize)

    check_annotations(target6, annotation6)

    # Open tunnel in the floor
    scene7 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 0, 1, 0, 1, 0, 0],
              [0, 1, 1, 0, 1, 1, 0],
              [0, 0, 0, 0, 0, 0, 1],
              [0, 1, 1, 1, 1, 1, 0]]

    target7 = np.ones_like(scene7) * 2

    annotation7 = annotator.annotate(scene7, debug=visualize)

    check_annotations(target7, annotation7)


def test_enclosure_annotator(visualize=False):

    annotator = ContainmentAnnotator([(-1, 0), (1, 0), (0, -1), (0, 1)])

    # Most straightforward test
    scene1 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 0]]

    target1 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 3, 3, 3, 3, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 3, 3, 3, 3, 3, 2]]

    annotation1 = annotator.annotate(scene1, debug=visualize)

    check_annotations(target1, annotation1)

    # One wall lower than the other
    scene2 = [[0, 1, 1, 1, 0, 0, 0],
              [0, 1, 0, 1, 1, 1, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 0]]

    target2 = [[2, 3, 3, 3, 2, 2, 2],
               [2, 3, 4, 3, 3, 3, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 3, 3, 3, 3, 3, 2]]

    annotation2 = annotator.annotate(scene2, debug=visualize)

    check_annotations(target2, annotation2)

    # Clipped corners
    scene3 = [[0, 0, 1, 1, 1, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 0, 1, 1, 1, 0, 0]]

    target3 = [[2, 2, 3, 3, 3, 2, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 2, 3, 3, 3, 2, 2]]

    annotation3 = annotator.annotate(scene3, debug=visualize)

    check_annotations(target3, annotation3)

    # Closed tunnel in the floor
    scene4 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 0, 1, 0, 1, 0, 0],
              [0, 1, 1, 0, 1, 0, 0],
              [1, 0, 0, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 0]]

    target4 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 3, 3, 3, 3, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 2, 3, 4, 3, 2, 2],
               [2, 3, 3, 4, 3, 2, 2],
               [3, 4, 4, 4, 4, 3, 2],
               [2, 3, 3, 3, 3, 3, 2]]

    annotation4 = annotator.annotate(scene4, debug=visualize)

    check_annotations(target4, annotation4)

    # Multiple containers
    scene5 = [[0, 1, 1, 1, 1, 0, 0],
              [0, 1, 0, 0, 1, 0, 0],
              [0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 0],
              [0, 0, 0, 1, 0, 1, 0],
              [0, 0, 0, 1, 1, 1, 0]]

    target5 = [[2, 3, 3, 3, 3, 2, 2],
               [2, 3, 4, 4, 3, 2, 2],
               [2, 3, 3, 3, 3, 2, 2],
               [2, 2, 2, 2, 2, 2, 2],
               [2, 2, 2, 3, 3, 3, 2],
               [2, 2, 2, 3, 4, 3, 2],
               [2, 2, 2, 3, 3, 3, 2]]

    annotation5 = annotator.annotate(scene5, debug=visualize)

    check_annotations(target5, annotation5)

    # Multiple containers + noise
    scene6 = [[0, 1, 1, 1, 1, 0, 1],
              [0, 1, 0, 0, 1, 0, 0],
              [0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 1, 1, 0],
              [0, 1, 0, 1, 0, 1, 0],
              [1, 1, 0, 1, 1, 1, 0]]

    target6 = target5

    annotation6 = annotator.annotate(scene6, debug=visualize)

    check_annotations(target6, annotation6)

    # Open tunnel in the ceiling
    scene7 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 0, 1, 0, 1, 0, 0],
              [0, 1, 1, 0, 1, 1, 0],
              [0, 1, 0, 0, 0, 0, 1],
              [0, 1, 1, 1, 1, 1, 0]]

    target7 = np.ones_like(scene7) * 2

    annotation7 = annotator.annotate(scene7, debug=visualize)

    check_annotations(target7, annotation7)

def test_left_containment_annotator(visualize=False):

    annotator_class, kwargs = ANNOTATOR_LIBRARY['left_containment']
    annotator = annotator_class(**kwargs)

    # Most straightforward test
    scene1 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 0, 0]]

    target1 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 3, 3, 3, 2, 2],
               [2, 3, 4, 4, 4, 2, 2],
               [2, 3, 3, 3, 3, 2, 2]]

    annotation1 = annotator.annotate(scene1, debug=visualize)

    check_annotations(target1, annotation1)


    # Most straightforward test
    scene2 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1, 0, 0],
              [0, 1, 1, 1, 1, 0, 0]]

    target2 = 2 * np.ones_like(scene2)

    annotation2 = annotator.annotate(scene2, debug=visualize)

    check_annotations(target2, annotation2)

def test_right_containment_annotator(visualize=False):

    annotator_class, kwargs = ANNOTATOR_LIBRARY['right_containment']
    annotator = annotator_class(**kwargs)

    # Most straightforward test
    scene1 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 0]]

    target1 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 3, 3, 3, 3, 2],
               [2, 4, 4, 4, 4, 3, 2],
               [2, 3, 3, 3, 3, 3, 2]]

    annotation1 = annotator.annotate(scene1, debug=visualize)

    check_annotations(target1, annotation1)


    # Most straightforward test
    scene2 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1, 0, 0],
              [0, 1, 1, 1, 1, 0, 0]]

    target2 = 2 * np.ones_like(scene2)

    annotation2 = annotator.annotate(scene2, debug=visualize)

    check_annotations(target2, annotation2)

def test_top_containment_annotator(visualize=False):

    annotator_class, kwargs = ANNOTATOR_LIBRARY['top_containment']
    annotator = annotator_class(**kwargs)

    # Most straightforward test
    scene1 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1, 0]]

    target1 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 3, 3, 3, 3, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 3, 4, 4, 4, 3, 2]]

    annotation1 = annotator.annotate(scene1, debug=visualize)

    check_annotations(target1, annotation1)


    # Most straightforward test
    scene2 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1, 0, 0],
              [0, 1, 1, 1, 1, 0, 0]]

    target2 = 2 * np.ones_like(scene2)

    annotation2 = annotator.annotate(scene2, debug=visualize)

    check_annotations(target2, annotation2)

def test_floor_adjacent_containment_annotator(visualize=False):

    annotator_class, kwargs = ANNOTATOR_LIBRARY['containment']
    annotator = annotator_class(label_floor_adjacent=True, **kwargs)

    # Most straightforward test
    scene1 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 0]]

    target1 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 3, 5, 5, 5, 3, 2],
               [2, 3, 3, 3, 3, 3, 2]]

    annotation1 = annotator.annotate(scene1, debug=visualize)

    check_annotations(target1, annotation1)

    scene2 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 0, 1, 1, 1, 0],
              [0, 1, 1, 0, 0, 0, 0]]

    target2 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 3, 4, 5, 5, 3, 2],
               [2, 3, 5, 3, 3, 3, 2],
               [2, 3, 3, 2, 2, 2, 2]]

    annotation2 = annotator.annotate(scene2, debug=visualize)

    check_annotations(target2, annotation2)

def test_holey_containment_annotator(visualize=False):

    annotator = HoleyContainmentAnnotator([(1, 0), (0, -1), (0, 1)])

    # Most straightforward test
    scene1 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 1, 1, 0, 1, 1, 0]]

    target1 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 3, 3, 2, 3, 3, 2]]

    annotation1 = annotator.annotate(scene1, debug=visualize)

    check_annotations(target1, annotation1)

    # One wall lower than the other
    scene2 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 0]]

    target2 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 2, 2, 2, 2, 2],
               [2, 2, 4, 4, 4, 3, 2],
               [2, 3, 3, 3, 3, 3, 2]]

    annotation2 = annotator.annotate(scene2, debug=visualize)

    check_annotations(target2, annotation2)

    # Clipped corners
    scene3 = [[0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 1, 0],
              [0, 0, 1, 1, 0, 0, 0]]

    target3 = [[2, 2, 2, 2, 2, 2, 2],
               [2, 3, 2, 2, 2, 2, 2],
               [2, 3, 4, 4, 4, 3, 2],
               [2, 2, 3, 3, 2, 2, 2]]

    annotation3 = annotator.annotate(scene3, debug=visualize)

    check_annotations(target3, annotation3)


    
if __name__ == '__main__':
    test_containment_annotator(visualize=False)
    test_enclosure_annotator(visualize=False)
    test_left_containment_annotator(visualize=False)
    test_right_containment_annotator(visualize=False)
    test_top_containment_annotator(visualize=False)
    test_floor_adjacent_containment_annotator(visualize=False)
    test_holey_containment_annotator(visualize=False)
    
    print("Passed.")

