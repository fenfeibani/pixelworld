import networkx as nx

import pixelworld.envs.pixelworld as pw
from ..helpers import h, L


class WinEvent(pw.core.Event):
    """The event that we solved the maze"""
    _reward = 1000
    _terminates = True

class GoalInteracts(pw.core.InteractsObjectAttribute):
    _step_after = ['pushes']

    def _interact(self, obj1, obj2):
        """Override this function in subclasses.
                   
        Parameters
        ----------
        obj1 : Object
            The first object (the goal).

        obj2 : Object
            The second object (presumably the player).
        """
        assert isinstance(obj2, pw.objects.SelfObject)
        event = WinEvent(self.world)


class Goal(pw.objects.BasicObject):
    """Where you want to go"""
    _attributes = ['goal_interacts']
    _defaults = {'mass': 0, 'zorder': -1}


class MazeWorld(pw.core.PixelWorld):
    """World that auto-generates a random, solvable maze.
    """
    def __init__(self, objects=None, width=30, height=30, method='subdivide', **kwargs):
        """
        Parameters
        ----------
        objects : list of object specifications (optional)
            Objects to create in addition to the maze, the goal, and the self
        width : int
        height : int
        method : string
            Method used for generating the maze. Current methods are
            'subdivide' and 'squares'.
        """
        self._method = method

        self._legend = {'W': ['wall', dict(color=1)], 'G': ['goal', dict(color=3)]}

        super(MazeWorld, self).__init__(objects=objects, 
                                        width=width, height=height, **kwargs)

    def add_frame(self, screen, width, height):
        """Add a frame around the maze.
        """
        screen[0][:] = ['W'] * width
        screen[-1][:] = ['W'] * width
        for i in xrange(0, height):
            screen[i][0] = screen[i][-1] = 'W'

    def ensure_solvable(self, screen):
        """Delete wall pixels until there is a path from start to finish.
        """
        height, width = len(screen), len(screen[0])
        gf = nx.Graph()
        gf.add_node((1, 1))
        gf.add_node((height - 2, width - 2))
        for r, row in enumerate(screen):
            for c, x in enumerate(row):
                if c < width - 1 and screen[r][c] == ' ' and screen[r][c + 1] == ' ':
                    gf.add_edge((r, c), (r, c + 1))
                if r < height - 1 and screen[r][c] == ' ' and screen[r + 1][c] == ' ':
                    gf.add_edge((r, c), (r + 1, c))

        while not nx.has_path(gf, (1, 1), (height - 2, width - 2)):
            r, c = self.rng.randint(1, height - 1), self.rng.randint(1, width - 1)
            while screen[r][c] != 'W':
                r, c = self.rng.randint(1, height - 1), self.rng.randint(1, width - 1)

            screen[r][c] = ' '
            if c > 0 and screen[r][c - 1] == ' ':
                gf.add_edge((r, c), (r, c - 1))
            if r > 0 and screen[r - 1][c] == ' ':
                gf.add_edge((r, c), (r - 1, c))
            if c < width - 1 and screen[r][c + 1] == ' ':
                gf.add_edge((r, c), (r, c + 1))
            if r < height - 1 and screen[r + 1][c] == ' ':
                gf.add_edge((r, c), (r + 1, c))

    def make_harder(self, screen):
        """Insert wall pixels until there is no path from start to finish, then remove
        the last one we added.
        """
        height, width = len(screen), len(screen[0])
        gf = nx.Graph()
        gf.add_node((1, 1))
        gf.add_node((height - 2, width - 2))
        for r, row in enumerate(screen):
            for c, x in enumerate(row):
                if c < width - 1 and screen[r][c] == ' ' and screen[r][c + 1] == ' ':
                    gf.add_edge((r, c), (r, c + 1))
                if r < height - 1 and screen[r][c] == ' ' and screen[r + 1][c] == ' ':
                    gf.add_edge((r, c), (r + 1, c))

        while nx.has_path(gf, (1, 1), (height - 2, width - 2)):
            path = nx.shortest_path(gf, source=(1, 1), target=(height-2, width-2))
            i = self.rng.randint(1, len(path) - 1)
            r, c = path[i]

            screen[r][c] = 'W'
            for edge in gf.edges((r, c)):
                gf.remove_edge(*edge)

        # the last one made the maze unsolvable 
        screen[r][c] = ' '

    def method_longest(self, width, height):
        """Create a maze by calling make_harder repeatedly. This makes mazes with
        pretty long shortest paths from start to goal.
        """
        screen = [[" "] * width for i in xrange(height)]
        self.add_frame(screen, width, height)
        for _ in xrange(200):
            self.make_harder(screen)
        screen = '\n'.join([''.join(x for x in line) for line in screen])
        return screen

    def method_squares(self, width, height):
        """Create a maze by dropping squares of various sizes at random locations, and
        deleting pixels at the end until the maze is solvable.
        """
        screen = [[" "] * width for i in xrange(height)]
        for size, num in [(8, 1), (4, 4), (2, 48), (1, 48)]:
            for _ in xrange(num):
                posn = self.rng.randint(1, height - size), self.rng.randint(1, width - size)
                for i in xrange(posn[0], posn[0] + size):
                    screen[i][posn[1]:posn[1] + size] = ['W'] * size
        self.add_frame(screen, width, height)
        screen[height - 2][width - 2] = ' '
        self.ensure_solvable(screen)
        screen = '\n'.join([''.join(x for x in line) for line in screen])
        return screen

    def method_kruskal(self, width, height):
        """Generate a random maze using Kruskal's algorithm"""
        screen = [["W"] * width for i in xrange(height)]
        screen[1][1] = screen[height - 2][width - 2] = " "

        parent = dict()
        parent[1, 1] = (1, 1)
        parent[height - 2, width - 2] = (height - 2, width - 2)

        def lookup((r, c)):
            while parent[r, c] != (r, c):
                r, c = parent[r, c]
            return r, c

        def merge_components((r1, c1), (r2, c2)):
            r1, c1 = lookup((r1, c1))
            r2, c2 = lookup((r2, c2))
            parent[r2, c2] = (r1, c1)

        while lookup((1, 1)) != lookup((height-2, width-2)):
            r, c = self.rng.randint(1, height - 1), self.rng.randint(1, width - 1)
            if screen[r][c] == ' ':
                continue
            screen[r][c] = ' '
            parent[r, c] = r, c
            if r > 1 and screen[r - 1][c] == ' ':
                merge_components((r, c), (r - 1, c))
            if r < height - 2 and screen[r + 1][c] == ' ':
                merge_components((r, c), (r + 1, c))
            if c > 1 and screen[r][c - 1] == ' ':
                merge_components((r, c), (r, c - 1))
            if c < width - 2 and screen[r][c + 1] == ' ':
                merge_components((r, c), (r, c + 1))

        self.add_frame(screen, width, height)
        print  '\n'.join([''.join(x for x in line) for line in screen])
        self.make_harder(screen)
        print  '\n'.join([''.join(x for x in line) for line in screen])
        screen = '\n'.join([''.join(x for x in line) for line in screen])

        return screen

    def method_subdivide(self, width, height):
        """Create a maze by recursively splitting the maze into two halves, and then
        dividing the two halves with a wall that has exactly one hole through it.
        """
        screen = [[" "] * width for i in xrange(height)]
        def construct_maze((r1, c1), (r2, c2), (ravoid, cavoid), split=None):
            if r2 - r1 < 4 and c2 - c1 < 4:
                return

            if r2 - r1 < 4 and split == 1:
                construct_maze((r1, c1), (r2, c2), (ravoid, cavoid), split=0)
                return

            if c2 - c1 < 4 and split == 0:
                construct_maze((r1, c1), (r2, c2), (ravoid, cavoid), split=1)
                return        

            if split==0 or (split is None and self.rng.rand() < 0.5):
                c = self.rng.randint(c1 + 1, c2)
                while c == cavoid:
                    c = self.rng.randint(c1 + 1, c2)
                idxs = range(r1, r2 + 1)
                assert len(idxs) > 0
                r = idxs.pop(self.rng.randint(len(idxs)))
                for i in idxs:
                    screen[i][c] = 'W'

                construct_maze((r1, c1), (r2, c-1), (r, c), 1)
                construct_maze((r1, c + 1), (r2, c2), (r, c), 1)
            else:
                r = self.rng.randint(r1 + 1, r2)
                while r == ravoid:
                    r = self.rng.randint(r1 + 1, r2)
                idxs = range(c1, c2 + 1)
                assert len(idxs) > 0
                c = idxs.pop(self.rng.randint(len(idxs)))
                for i in idxs:
                    screen[r][i] = 'W'

                construct_maze((r1, c1), (r - 1, c2), (r, c), 0)
                construct_maze((r + 1, c1), (r2, c2), (r, c), 0)

        self.add_frame(screen, width, height)
        construct_maze((1, 1), (height - 2, width - 2), (height // 2, width // 2))
        self.ensure_solvable(screen)
        screen = '\n'.join([''.join(x for x in line) for line in screen])

        return screen


class MazeRandomizer(pw.core.Randomizer):
    """Rerandomize the maze using the original method"""
    def _randomize(self):
        if self.world._method == 'squares':
            screen = self.world.method_squares(self.world.width, self.world.height)
        elif self.world._method == 'subdivide':
            screen = self.world.method_subdivide(self.world.width, self.world.height)
        elif self.world._method == 'kruskal':
            screen = self.world.method_kruskal(self.world.width, self.world.height)
        elif self.world._method == 'longest':
            screen = self.world.method_longest(self.world.width, self.world.height)
        else:
            assert False, 'unrecognized method: %s' % self.world._method

        objects, _, _ = h.world.screen(screen, self.world._legend)
        objects = objects + [['self', dict(color=2, position=(1,1))], 
                             ['goal', dict(color=3, position=(self.world.height-2, self.world.width-2))]]
        self.world.remove_objects(self.world.objects)
        self.world.create_objects(objects)

randomizer = MazeRandomizer
world = MazeWorld
