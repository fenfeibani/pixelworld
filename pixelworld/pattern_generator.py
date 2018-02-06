"""
Randomly generates various patterns for use in PixelWorld environments.

Here 'patterns' are simply binary 2D (numpy) arrays. The width and height 
of the arrays is fixed for a given pattern generator.

A pattern generator is implicitly characterized by a domain of patterns, which
is a subset of all possible binary arrays of the given width and height.
Explicitly, the generator implements a `sample` method that produces
samples from the domain *uniformly* at random.
"""
from __future__ import print_function
from __future__ import absolute_import

from bresenham import bresenham
from scipy.ndimage.measurements import label as scipy_label
from scipy.ndimage.filters import maximum_filter
from scipy.spatial import ConvexHull

import numpy as np
from sys import stdout


### API ###
class PatternGenerator(object):
    """
    Generates all possible 'patterns' of the given max_width and max_height.

    Note that the "patterns" generated here are not necessarily contiguous shapes!

    The samples produced by this generator are a superclass of 
    all other pattern generators with the same max_width and max_height.

    Parameters
    ----------
    max_width : int
    max_height : int
    seed : int
    """
    def __init__(self, max_width, max_height, seed=42):
        self.max_width = max_width
        self.max_height = max_height
        self.seed = seed

        self.random = np.random.RandomState(seed)

    def sample(self):
        # Sample randomly until we find a pattern that touches its boundary.
        while True:
            sample = self.random.randint(2, size=(self.max_height, self.max_width), dtype=np.int)
            try:
                check_sample_validity(sample)
                return sample
            except Exception:
                pass



### Custom Shape Generators ###
class ContainerGenerator(PatternGenerator):
    """
    A container is a pattern with two vertical 'walls' that start in the same bottom
    row, and a horizontal floor that connects their bottoms.

    A container is characterized by:
        -the length of the floor
        -the heights of the two walls

    To avoid degenerate cases, we insist:
        -the length of the floor is at least 3
        -the heights of the walls are at least 1

    Parameters
    ----------
    max_width : int
    max_height : int
    seed : int
    symmetric_walls : bool
    """
    ROTATIONS = {"bottom":0,"right":1,"top":2,"left":3}

    def __init__(self, max_width, max_height, seed=42, symmetric_walls=False, clip_corners=False, has_hole=False, orientation="bottom"):
        self.symmetric_walls = symmetric_walls
        self.clip_corners = clip_corners
        self.has_hole = has_hole
        self.rotations = self.ROTATIONS[orientation]
        super(ContainerGenerator, self).__init__(max_width, max_height, seed=seed)

    def _create_container(self, floor_length, left_height, right_height):
        container_height = max(left_height, right_height)
        floor_row = container_height - 1
        container = np.zeros((container_height, floor_length), dtype=np.int)

        left_wall_top_row = container_height - left_height
        right_wall_top_row = container_height - right_height

        assert floor_row > 0
        assert left_wall_top_row == 0 or right_wall_top_row == 0
        assert floor_row - left_wall_top_row > 0 and floor_row - right_wall_top_row > 0

        # Add the floor
        container[floor_row, :] = 1

        # Add the left wall
        container[left_wall_top_row:floor_row, 0] = 1

        # Add the right wall
        container[right_wall_top_row:floor_row, floor_length-1] = 1

        if self.has_hole:
            n = self.random.randint(floor_length)
            if n > 0 and n < floor_length-1:
                container[floor_row,n] = 0
            else:
                container[floor_row-1,n] = 0    

        if self.clip_corners:
            for corner in [(floor_row,0), (floor_row,floor_length-1)]:
                if self.random.uniform() < 0.5:
                    r, c = corner
                    container[r, c] = 0            
        
        for _ in range(self.rotations):
            container = np.rot90(container)


        return container

    def sample(self):
        floor_length = self.random.randint(3, self.max_width+1)
        left_height = self.random.randint(2, self.max_height+1)
        if self.symmetric_walls:
            right_height = left_height
        else:
            right_height = self.random.randint(2, self.max_height+1)
        return self._create_container(floor_length, left_height, right_height)



class BlobGenerator(PatternGenerator):
    """
    A blob is a contiguous pattern.

    The sampling process works as follows:
        1. 2 starting pixels are randomly placed within a canvas of
            `max_width` and `max_height`, within the `padding`.
        2. While the pixels are not contiguous: 
            2a. Apply the local rule that random neighboring pixels of an active pixel is activated.
            2b. With probability `starting_pixel_prob`, add another randomly place pixel.
        3. The canvas is cropped to remove any rows or columns that do not touch the blob.
    """
    def __init__(self, max_width, max_height, seed=42, starting_pixel_prob=0.1, padding=0):
        self.starting_pixel_prob = starting_pixel_prob
        self.padding = padding

        self._footprint = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        super(BlobGenerator, self).__init__(max_width, max_height, seed=seed)

    def _is_contiguous(self, arr):
        _, num_features = scipy_label(arr)
        return num_features == 1

    def _grow(self, arr):
        return maximum_filter(arr, footprint=self._footprint)

    def _add_random_pixel(self, arr):
        max_height = self.max_height - self.padding
        max_width = self.max_width - self.padding
        row, col = self.random.randint(self.padding, max_height), self.random.randint(self.padding, max_width)
        arr[row, col] = 1

    def sample(self):
        canvas = np.zeros((self.max_height, self.max_width), dtype=np.int)
        for _ in xrange(2):
            self._add_random_pixel(canvas)

        while not self._is_contiguous(canvas):
            canvas = self._grow(canvas)
            if self.random.uniform() <= self.starting_pixel_prob:
                self._add_random_pixel(canvas)

        return crop(canvas)



class ConvexBlobGenerator(PatternGenerator):
    """
    The sampling process works as follows:
        1. Randomly sample 5-10 points in the canvas.
        2. Draw a convex hull around those points.
        3. Fill in the convex hull.
        4. Crop.
    """
    def __init__(self, max_width, max_height, seed=42, padding=0):
        self.padding = padding
        self._closing = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

        super(ConvexBlobGenerator, self).__init__(max_width, max_height, seed=seed)

    def _add_random_pixel(self, arr):
        max_height = self.max_height - self.padding
        max_width = self.max_width - self.padding
        row, col = self.random.randint(self.padding, max_height), self.random.randint(self.padding, max_width)
        arr[row, col] = 1

    def _draw_line(self, arr, line_start, line_end):
        for x,y in bresenham(line_start[0], line_start[1], line_end[0], line_end[1]):
            arr[x,y] = 1
        return arr

    def _fill_interior(self, arr):
        for i, row in enumerate(arr):
            edges = np.transpose(np.nonzero(row))
            if len(edges) <= 1:
                continue
            left_idx, right_idx = edges[0][0], edges[-1][0]
            arr[i, left_idx:right_idx] = 1
        return arr

    def _convex_completion(self, arr):
        blob_points = np.transpose(np.nonzero(arr))
        assert len(blob_points) >= 3

        hull = ConvexHull(blob_points)

        for simplex in hull.simplices:
            line_start, line_end = blob_points[simplex[0]], blob_points[simplex[1]]
            arr = self._draw_line(arr, line_start, line_end)

        arr = crop(arr)
        arr = self._fill_interior(arr)

        return arr

    def sample(self):
        canvas = np.zeros((self.max_height, self.max_width), dtype=np.int)
        num_points = self.random.randint(5, 11)
        for _ in xrange(num_points):
            self._add_random_pixel(canvas)

        canvas = self._convex_completion(canvas)

        return canvas



class CornerGenerator(PatternGenerator):
    """
    An L shape, in any of four orientations
    """
    DIRECTIONS = {"lower_left":(-1,0),"lower_right":(-1,-1),"upper_left":(0,0),"upper_right":(0,-1)}

    def __init__(self,max_width,max_height,seed=42,orientation=None,clip_corners=False):
        self.orientation = orientation
        self.clip_corners = clip_corners
        super(CornerGenerator,self).__init__(max_width,max_height,seed=seed)

    def sample(self):
        height = self.random.randint(2,self.max_height)
        width = self.random.randint(2,self.max_width)
        corner = np.zeros((height,width), dtype=np.int)
        if self.orientation is None:
            direction = self.DIRECTIONS.values()[self.random.randint(4)]
        else:
            direction = self.DIRECTIONS[self.orientation]    
        corner[direction[0],:] = 1
        corner[:,direction[1]] = 1    
        if self.clip_corners and self.random.random_sample() < 0.5:
            corner[direction[0],direction[1]] = 0

        return corner




class RectangularEnclosureGenerator(PatternGenerator):
    """
    An rectangular enclosure is the perimeter of a rectangle.

    Randomly remove corners if `clip_corners` is True.
    """
    def __init__(self, max_width, max_height, min_width=3, min_height=3, seed=42, clip_corners=False, has_hole=False):
        self.clip_corners = clip_corners
        self.min_width = min_width
        self.min_height = min_height
        self.has_hole = has_hole
        super(RectangularEnclosureGenerator, self).__init__(max_width, max_height, seed=seed)

    def _create_enclosure(self, floor_length, wall_height):
        enclosure = np.zeros((wall_height, floor_length), dtype=np.int)

        # Add the ceiling
        enclosure[0, :] = 1

        # Add the floor
        enclosure[-1, :] = 1

        # Add the left wall
        enclosure[:, 0] = 1

        # Add the right wall
        enclosure[:, -1] = 1

        if self.has_hole:
            if self.random.uniform() < 0.5: # left or rigtht
                n = self.random.randint(1,wall_height-1)
                if self.random.uniform() < 0.5: 
                    enclosure[n,0] = 0
                else:
                    enclosure[n,-1] = 0
            else: # top or bottom
                n = self.random.randint(1,floor_length-1)
                if self.random.uniform() < 0.5: 
                    enclosure[0,n] = 0
                else:
                    enclosure[-1,n] = 0               

        # Remove corners, each with probability 0.5.
        if self.clip_corners:
            for corner in [(0, 0), (0, -1), (-1, -1), (-1, 0)]:
                if self.random.uniform() < 0.5:
                    r, c = corner
                    enclosure[r, c] = 0

        return enclosure

    def sample(self):
        floor_length = self.random.randint(self.min_width, self.max_width+1)
        wall_height = self.random.randint(self.min_height, self.max_height+1)
        return self._create_enclosure(floor_length, wall_height)



class VLineGenerator(PatternGenerator):
    """
    A vertical line is 1 pixel wide and `max_height` pixels tall.
    """
    def __init__(self, max_width, max_height, seed=42, min_height=2):
        self.min_height = min_height
        super(VLineGenerator, self).__init__(max_width, max_height, seed=seed) 
    
    def sample(self):
        height = self.random.randint(self.min_height, self.max_height+1)
        return np.ones((height, 1), dtype=np.int)




class HLineGenerator(PatternGenerator):
    """
    A horizontal line is 1 pixel tall and `max_width` pixels wide.
    """
    def __init__(self, max_width, max_height, seed=42, min_width=2):
        self.min_width = min_width
        super(HLineGenerator, self).__init__(max_width, max_height, seed=seed) 

    def sample(self):
        width = self.random.randint(self.min_width, self.max_width+1)
        return np.ones((1, width), dtype=np.int)


class LineGenerator(PatternGenerator):
    """
    A line is 1 pixel tall and `max_?` pixels long.
    """
    def __init__(self, max_width, max_height, seed=42, min_length=2):
        self.min_length = min_length
        super(LineGenerator, self).__init__(max_width, max_height, seed=seed) 

    def sample(self):
        if self.random.random_sample() < 0.5:
            width = self.random.randint(self.min_length, self.max_width+1)
            return np.ones((1, width), dtype=np.int)
        else:
            height = self.random.randint(self.min_length, self.max_height+1)
            return np.ones((height, 1), dtype=np.int)            


class CrossGenerator(PatternGenerator):
    """
    A cross had up/down/left/right projections of any length >= 1
    """

    def sample(self):
        width = self.random.randint(3,self.max_width)
        height = self.random.randint(3,self.max_height)
        cross = np.zeros((width,height), dtype=np.int)    
        center = (self.random.randint(1,width-1), self.random.randint(1,height-1))
        cross[center[0],:] = 1
        cross[:,center[1]] = 1
        return cross            


class StandardChairGenerator(PatternGenerator):
    """
    A standard chair is a corner with one (centered) or two legs
    """

    def sample(self):
        back_height = self.random.randint(1,self.max_height/2)
        seat_width = self.random.randint(3,self.max_width)
        leg_height = self.random.randint(1,self.max_height-back_height-1)
        back_x = 0 if self.random.random_sample() < 0.5 else seat_width-1
        seat_y = leg_height
        leg_offset = self.random.randint(seat_width/2)
        chair = np.zeros((back_height+leg_height+1,seat_width), dtype=np.int)
        chair[seat_y,:] = 1
        chair[0:seat_y,back_x] = 1
        chair[seat_y:self.max_height,leg_offset] = 1
        chair[seat_y:self.max_height,seat_width-leg_offset-1] = 1
        return chair                    

class StandardTableGenerator(PatternGenerator):
    """
    A standard table is an hline with two legs (or one centered leg)
    """   

    def sample(self):
        width = self.random.randint(3,self.max_width)
        leg_height = self.random.randint(1,self.max_height-1)
        leg_offset = self.random.randint(width/2)
        table = np.zeros((leg_height+1,width), dtype=np.int) 
        table[0,:] = 1
        table[:,leg_offset]=1
        table[:,width-leg_offset-1] = 1
        return table          



class BowlGenerator(PatternGenerator):
    """
    A bowl is a container with symmetric convex sides. 

    Since the sides must be connected, and the width of one side must be
    `width/2`, and the height must be `height`, a bowl can be
    characterized by a list of nonnegative integer `altitude_changes` 
    such that len(altitude_changes) == width/2 and 
    sum(altitude_changes) == height-1. For instance, with
    height = 4 and width = 4, the half-bowl corresponding to
    [2, 1, 0, 0] would be

        [[1, 0, 0, 0],
         [1, 0, 0, 0],
         [1, 1, 0, 0],
         [0, 1, 1, 1]]

    Parameters
    ----------
    max_width : int
    max_height : int
    seed : int
    symmetric_walls : bool
    """
    ROTATIONS = {"bottom":0,"right":1,"top":2,"left":3}

    def __init__(self, max_width, max_height, seed=42, orientation="bottom"):
        self.rotations = self.ROTATIONS[orientation]
        super(BowlGenerator, self).__init__(max_width, max_height, seed=seed)

    def _create_bowl(self, altitude_changes):
        half_width = len(altitude_changes)
        height = sum(altitude_changes) + 1

        half_bowl = np.zeros((height, half_width))

        row = 0
        for idx, ac in enumerate(altitude_changes):
            half_bowl[row, idx] = 1
            for i in xrange(ac):
                row += 1
                half_bowl[row, idx] = 1

        assert row == height-1

        bowl = np.concatenate([half_bowl, np.fliplr(half_bowl)], axis=1)

        for _ in range(self.rotations):
            bowl = np.rot90(bowl)

        return bowl

    def sample(self):
        half_width = self.random.randint(2, self.max_width/2)
        height = self.random.randint(2, self.max_height+1)

        altitude_changes = np.zeros(half_width, dtype=np.int)
        for _ in xrange(height-1):
            idx = self.random.randint(half_width-1)
            altitude_changes[idx] += 1

        return self._create_bowl(altitude_changes)




### Generator Library ###
GENERATOR_LIBRARY = {
    'all' : PatternGenerator,
    'blob' : BlobGenerator,   
    'convex_blob' : ConvexBlobGenerator,
    'container' : ContainerGenerator,
    'corner': CornerGenerator,       
    'rect_enclosure' : RectangularEnclosureGenerator,
    'line' : LineGenerator,
    'vline' : VLineGenerator,
    'hline' : HLineGenerator,
    'cross': CrossGenerator,
    'chair': StandardChairGenerator,
    'table': StandardTableGenerator,
    'bowl' : BowlGenerator
}


### Helper methods ###
def crop(arr):
    mask = arr > 0
    return arr[np.ix_(mask.any(1), mask.any(0))]

def visualize_samples(samples):
    from matplotlib import pyplot as plt

    num_samples = len(samples)

    num_rows = num_cols = np.ceil(np.sqrt(num_samples))

    fig = plt.figure(facecolor='darkslategray')

    for idx in xrange(num_samples):
        plt.subplot(num_rows, num_cols, idx+1)
        plt.axis('off')
        plt.imshow(samples[idx], cmap='binary_r', vmin=0, vmax=1, interpolation='nearest')

    plt.show()

def is_unique(arr, arr_lst):
    for other_arr in arr_lst:
        if np.array_equal(arr, other_arr):
            return False
    return True

def check_sample_validity(sample):
    touching = sample[0].sum() > 0 and \
               sample[-1].sum() >0 and \
               sample[0,:].sum() > 0 and \
               sample[-1,:].sum() > 0
    if not touching:
        sample = crop(sample)
        print("Warning: Sample invalid: not touching boundary.")
        #raise Exception("Sample invalid: not touching boundary.")

def generate_patterns(generator_name, max_width=None, max_height=None, seed=42, num_samples=1, require_unique=True, **kwargs):
    generator_class = GENERATOR_LIBRARY[generator_name]

    if isinstance(generator_class, list):
         generator_class, lib_kwargs = generator_class
         kwargs.update(lib_kwargs)
    assert isinstance(generator_class, PatternGenerator.__class__)

    generator = generator_class(max_width=max_width, max_height=max_height, seed=seed, **kwargs)

    samples = []
    itr = 0
    while True:
        if len(samples) >= num_samples:
            break
        sample = generator.sample()
        itr += 1

        try:
            check_sample_validity(sample)
        except Exception:
            continue    

        if not require_unique or is_unique(sample, samples):
            samples.append(sample)

        stdout.write("\r%s: Generated %i samples, %i/%i acceptable." % (generator_name,itr,len(samples),num_samples))
        stdout.flush()   

        if itr > 10000:
            raise Exception("Failed to generate %i samples for %s with width=%i, height=%i." \
                             % (num_samples,generator_name,max_width,max_height))
    
    stdout.write("\n"); stdout.flush()
    return samples

if __name__ == '__main__':
    seed = 42

    patterns = {}
    patterns['all'] = generate_patterns('all', max_width=5, max_height=4, num_samples=12, seed=seed)
    patterns['sym_containers'] = generate_patterns('container', symmetric_walls=True, max_width=8, max_height=8, num_samples=16,  seed=seed)
    patterns['containers'] = generate_patterns('container', clip_corners=True, max_width=8, max_height=8, num_samples=16, seed=seed)
    patterns['noncontainers'] = generate_patterns('container', clip_corners=True, has_hole=True, max_width=8, max_height=8, num_samples=16, seed=seed)
    patterns['left_containers'] = generate_patterns('container', orientation='left', max_width=8, max_height=8, num_samples=16, seed=seed)
    patterns['right_containers'] = generate_patterns('container', orientation='right', max_width=8, max_height=8, num_samples=16, seed=seed)
    patterns['top_containers'] = generate_patterns('container', orientation='top', max_width=8, max_height=8, num_samples=16, seed=seed) 
    patterns['left_noncontainers'] = generate_patterns('container', orientation='left', has_hole=True, max_width=8, max_height=8, num_samples=16, seed=seed)
    patterns['right_noncontainers'] = generate_patterns('container', orientation='right', has_hole=True, max_width=8, max_height=8, num_samples=16, seed=seed)
    patterns['top_noncontainers'] = generate_patterns('container', orientation='top', has_hole=True, max_width=8, max_height=8, num_samples=16, seed=seed) 
    patterns['blobs'] = generate_patterns('blob', max_width=4, max_height=3, num_samples=20, seed=seed)
    patterns['convex_blobs'] = generate_patterns('convex_blob', max_width=7, max_height=7, num_samples=50, seed=seed)
    patterns['rectangles'] = generate_patterns('rect_enclosure', max_width=9, max_height=9, num_samples=12, clip_corners=False, seed=seed)
    patterns['enclosures'] = generate_patterns('rect_enclosure', max_width=9, max_height=9, min_width=4, min_height=4, num_samples=16, clip_corners=True, seed=seed)
    patterns['nonenclosures'] = generate_patterns('rect_enclosure', has_hole=True, max_width=9, max_height=9, min_width=4, min_height=4, num_samples=16, clip_corners=True, seed=seed)
    patterns['hlines'] = generate_patterns('hline', max_width=10, min_width=2, num_samples=4, seed=seed)
    patterns['vlines'] = generate_patterns('vline', max_height=10, min_height=2, num_samples=4, seed=seed)
    patterns['crosses'] = generate_patterns('cross', max_width=7, max_height=7, num_samples=12, seed=seed)   
    patterns['chairs'] = generate_patterns('chair', max_width=4, max_height=7, num_samples=12, seed=seed)  
    patterns['tables'] = generate_patterns('table', max_width=7, max_height=4, num_samples=12, seed=seed)  
    patterns['lower_left_corners'] = generate_patterns('corner', max_width=5, max_height=5, num_samples=6, orientation="lower_left", seed=seed)
    patterns['lower_right_corners'] = generate_patterns('corner', max_width=5, max_height=5, num_samples=6, orientation="lower_right", seed=seed)
    patterns['upper_left_corners'] = generate_patterns('corner', max_width=5, max_height=5, num_samples=6, orientation="upper_left", seed=seed)
    patterns['upper_right_corners'] = generate_patterns('corner', max_width=5, max_height=5, num_samples=6, orientation="upper_right", seed=seed)
    patterns['bowl'] = generate_patterns('bowl', max_width=10, max_height=6, num_samples=9, seed=seed)
    

    while True:
        label = None
        while not (label in patterns.keys() or label == 'q'):
            print( "Available patterns: " + str(patterns.keys()))
            label = raw_input("Visualize which pattern? (or 'q' to quit): " )
        if label == 'q':
            break
        samples = patterns[label]
        visualize_samples(samples)



