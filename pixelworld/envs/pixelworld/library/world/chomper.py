"""
Chomp the pellets; avoid the ghosts.

Ghost behavior based on description here:
http://gameinternals.com/post/2072558330/understanding-pac-man-ghost-behavior
"""
import numpy as np
import networkx as nx

from ..helpers import h, L
from pixelworld.envs.pixelworld import core, objects, object_attributes as oa, events


class EatDotEvent(core.Event):
    """Chomper ate a dot"""
    _reward = 10


class EatPelletEvent(core.Event):
    """Chomper ate a power pellet"""
    _reward = 50


class EatGhostEvent(core.Event):
    """Chomper ate a ghost"""
    pass


class WinEvent(core.Event):
    """Chomper ate all the dots and pellets"""
    _terminates = True


class DeathEvent(core.Event):
    """The ghosts caught Chomper"""
    _reward = -1000
    _terminates = True


class GhostMode(core.StringObjectAttribute, core.SteppingObjectAttribute):
    """Either 'SCATTER' or 'CHASE': in SCATTER mode, ghosts (except Blinky) go to
    their favorite corner. In CHASE mode, they chase the player in their own
    particular way.

    When the mode changes, all the ghosts do a 180.
    """

    _default_value = 'SCATTER'

    def _step_object(self, obj, t, dt, agent_id, action):
        """Decide mode based on current time, assuming 4 simulation steps per second.
        
        Parameters
        ----------
        obj : Object
            The ghost object
        t : number
            The simulation time
        dt : number
            Time since last step
        agent_id : number
            Id of currently stepping agent.
        action : string
            Last executed action
        """
        timer = t / 4.0

        old_value = self.get(obj)

        if timer <= 7:
            self.set(obj, 'SCATTER')
        elif timer <= 27:
            self.set(obj, 'CHASE')
        elif timer <= 34:
            self.set(obj, 'SCATTER')
        elif timer <= 54:
            self.set(obj, 'CHASE')
        elif timer <= 59:
            self.set(obj, 'SCATTER')
        elif timer <= 79:
            self.set(obj, 'CHASE')
        elif timer <= 84:
            self.set(obj, 'SCATTER')
        else:
            self.set(obj, 'CHASE')

        if old_value != self.get(obj):
            obj.orientation += 2


class Powerup(core.SteppingObjectAttribute, core.NonNegativeIntegerObjectAttribute):
    """When Chomper eats a power pellet, he is stronger than the ghosts for a short
    time. This attribute tracks how many steps are left; when this attribute is
    zero, Chomper is not powered up."""

    def _step_object(self, obj, t, dt, agent_id, action):
        """Decrement the powerup timer if we have powerup. Also change ghost color
        based on whether we're powered up.
        
        Parameters
        ----------
        obj : Object
            The ghost object
        t : number
            The simulation time
        dt : number
            Time since last step
        agent_id : number
            Id of currently stepping agent.
        action : string
            Last executed action
        """
        old_value = self.get(obj)
        new_value = max(0, old_value - 1)
        self.set(obj, new_value)
        ghost_mode = self.world.object_attributes.get('ghost_mode', None)
        if new_value != old_value:
            if new_value > 0:
                if ghost_mode is not None:
                    for obj2 in ghost_mode.objects:
                        obj2.color = 7
            else:
                obj.ghosts_eaten = 0
                if ghost_mode is not None:
                    for obj2 in ghost_mode.objects:
                        obj2.color = obj2.initial_color

class Frightened(core.BooleanObjectAttribute, core.DerivedObjectAttribute):
    def _get_data_object(self, obj):
        player = self.world.objects['chomper_player']
        return (player.powerup > 0)


class GhostsEaten(core.NonNegativeIntegerObjectAttribute):
    """Number of ghosts Chomper has eaten since he became powered up."""
    _default_value = 0


class PlayerInteracts(core.InteractsObjectAttribute):
    """Defines the various interactions between Chomper and other objects."""

    _step_after = ['pushes']

    def _test_for_done(self):
        dots = self.world.objects.find(name='dot', visible=True)
        pellets = self.world.objects.find(name='pellet', visible=True)
        if len(dots) == 0 and len(pellets) == 0:
            event = WinEvent(self.world)

    def _interact(self, obj1, obj2):
        """Handle the various interactions between Chomper and food and Chomper and ghosts.
        
        Parameters
        ----------
        obj1 : Object
            The chomper object
        obj2 : Object
            The other object
        """

        # eat a dot: the dot disappears, we get points, maybe the game is over
        # and we won
        if isinstance(obj2, Dot):
            obj2.visible = False
            event = EatDotEvent(self.world)
            self._test_for_done()

        # eat a pellet: the pellet disappears, we get points, all ghosts
        # reverse direction, maybe the game is over and we won
        elif isinstance(obj2, Pellet):
            obj2.visible = False
            event = EatPelletEvent(self.world)
            obj1.powerup = 20
            for obj in self.world.objects:
                if isinstance(obj, Ghost):
                    obj.orientation += 2
            self._test_for_done()

        elif isinstance(obj2, Ghost):
            # eat a ghost: we get points, the ghost goes back to the ghost
            # house
            if obj1.powerup > 0:
                event = EatGhostEvent(self.world)
                obj2.position = obj2.initial_position
                obj2.orientation = 1

            # the ghost eats us and we die
            else:
                event = DeathEvent(self.world, reason='ghost')


class Teleports(core.InteractsObjectAttribute):
    _step_after = ['pushes', 'position']

    def _interact(self, obj1, obj2):
        """Cause Chomper and the ghosts to teleport to the other teleport location.
        
        Parameters
        ----------
        obj1 : Object
            The teleport object
        obj2 : Object
            The other object
        """
        # if we've already teleported this turn, do nothing
        if obj2.has_teleported:
            return

        objs = self.world.objects['teleport']
        for obj3 in objs:
            if obj3 is not obj1:
                obj2.position = obj3.position
                obj2.has_teleported = True


class HasTeleported(core.BooleanObjectAttribute, core.SteppingObjectAttribute):
    """Keep track of whether we've already teleported this step.
    """
    _step_after = ['teleports']
    _default_value = False

    def _step_object(self, obj, t, dt, agent_id, action):
        self.set(obj, False)


class GhostBehaviorObjectAttribute(core.BooleanObjectAttribute, core.SteppingObjectAttribute):
    """Defines the common ghost behavior that is specialized by the subclasses.
    
    At each step, each ghost picks a target square and tries to go towards
    there. Ghosts are not allowed to reverse direction unless they arrive at a
    dead end.

    When ghosts are frightened of Chomper because he is powered up, they move
    randomly, except that they do not reverse direction.

    If this attribute is set to False, the ghost will not move.
    """

    _depends_on = ['position']

    _direction = {
        0: np.array((0,-1)), # left
        1: np.array((-1,0)), # up
        2: np.array((0,1)),  # right
        3: np.array((1,0)),  # down
        }

    _step_after = ['pushes', 'orients', 'ghost_mode', 'powerup']
    _step_before = ['player_interacts', 'teleports']

    def _step_object(self, obj, t, dt, agent_id, action):
        if self.get(obj):
            position = obj.position
            player = self.world.objects['chomper_player']

            # don't move if we already are on top of the player
            if (player.position == obj.position).all():
                return

            # figure out what color the walls are
            wall_color = self.world.objects.get(name='blue_wall').color

            # pick the next possible squares as those which are open space and
            # which don't cause us to do a 180
            next_position_choices = [(dir, position + self._direction[dir]) for dir in self._direction
                                     if (dir + 2) % 4 != obj.orientation]
            next_position_choices = [(dir, pos) for (dir, pos) in next_position_choices 
                                     if 0 <= pos[0] < self.world.height and 0 <= pos[1] < self.world.width and
                                     self.world.state[tuple(pos.astype(int).tolist())] != wall_color]

            # ghosts are frightened
            if obj.frightened:
                # with probability 0.2, stay put (ghosts are slower when Chomper is
                # powered up)
                if obj.rng.rand() < 0.8 and len(next_position_choices) > 0:
                    dir, pos = next_position_choices[obj.rng.randint(len(next_position_choices))]
                else:
                    return

            # ghosts are not frightened
            else:
                target = self._choose_target(obj, player)
                if next_position_choices:
                    # choose next position which is closest to target
                    i = np.argmin([np.linalg.norm(pos - target) for (dir, pos) in next_position_choices])
                    dir, pos = next_position_choices[i]
                else:
                    # no next_position_choices, means we're in a dead end and
                    # should reverse
                    dir = (obj.orientation + 2) % 4
                    pos = obj.position + self._direction[dir]

            # actually move the ghost if there's no wall in the way
            if 0 <= pos[0] < self.world.height and 0 <= pos[1] < self.world.width and \
                    self.world.state[tuple(pos.astype(int).tolist())] != wall_color:
                obj.orientation = dir
                obj.position = pos

class IsInHouse(core.BooleanObjectAttribute, core.DerivedObjectAttribute):
    def _get_data_object(self, obj):
        return (9 <= obj.position[0] <= 11 and 10 <= obj.position[1] <= 16)


class HouseExit(core.FloatWorldAttribute):
    _ndim = 2

    _default_value = np.array([8, 12])


class BlinkyBehavior(GhostBehaviorObjectAttribute):
    """Blinky chases Chomper and doesn't scatter"""

    _step_before = ['pinky_behavior', 'inky_behavior', 'clyde_behavior']

    def _choose_target(self, obj, player):
        # get out of the ghost house if we're in it
        if obj.is_in_house:
            return self.world.house_exit

        return player.position


class PinkyBehavior(GhostBehaviorObjectAttribute):
    """Pinky goes to a point four pixels in front of Chomper"""

    _step_before = ['inky_behavior', 'clyde_behavior']

    def _choose_target(self, obj, player):
        # get out of the ghost house if we're in it
        if obj.is_in_house:
            return self.world.house_exit

        if obj.ghost_mode == 'SCATTER':
            return (0, 0)
        else:
            return player.position + 4 * self._direction[player.orientation]


class InkyBehavior(GhostBehaviorObjectAttribute):
    """Inky goes to a point based on the position of Chomper and Blinky.
    """

    _step_before = ['clyde_behavior']

    def _choose_target(self, obj, player):
        # get out of the ghost house if we're in it
        if obj.is_in_house:
            return self.world.house_exit

        if obj.ghost_mode == 'SCATTER':
            return (height, width)
        else:
            midpoint = player.position + 2 * self._direction[player.orientation]
            return midpoint + (midpoint - self.world.objects['blinky'].position)


class ClydeBehavior(GhostBehaviorObjectAttribute):
    """Clyde chases Chomper if Chomper is far away, otherwise he just goes to his
    corner.
    """
    def _choose_target(self, obj, player):
        # get out of the ghost house if we're in it
        if obj.is_in_house:
            return self.world.house_exit

        if obj.ghost_mode == 'SCATTER':
            return (height, 0)
        else:
            dist = np.linalg.norm(obj.position - player.position)
            if dist > 8:
                return player.position
            else:
                return (height, 0)


class InitialColorObjectAttribute(oa.ColorObjectAttribute):
    """An attribute that remembers the initial color of the object."""

    _initialize_after = ['color']
    _depends_on = ['color']
    
    def prepare(self):
        self.set(None, self._other_attr['color'].get(self.objects))


class BasicNoPhysicsObject(objects.BasicObject):
    """Object with zero mass, so they can intersect other objects."""
    _defaults = {'mass': 0, 'zorder': 0}


class BlueWall(objects.WallObject):
    """The walls of the maze"""
    _defaults = {'color': 3}


class Dot(BasicNoPhysicsObject):
    """Chomper's basic food"""
    _defaults = {'color': 1}


class Pellet(BasicNoPhysicsObject):
    """Food that powers up Chomper"""
    _defaults = {'color': 4}


class ChomperPlayer(objects.BasicSelfObject):
    """The Chomper"""
    _defaults = {'color': 5, 'zorder': 2}
    _attributes = ['player_interacts', 'powerup', 'orientation', 'orients', 'ghosts_eaten', 'has_teleported']


class Ghost(BasicNoPhysicsObject):
    """The ghosts"""
    _attributes = ['orientation', 'initial_position', 'ghost_mode', 'initial_color', 'has_teleported', 'frightened',
                   'is_in_house']
    _defaults = {'zorder': 1, 'orientation': 1}


class Blinky(Ghost):
    _defaults = {'color': 2}
    _attributes = ['blinky_behavior']


class Pinky(Ghost):
    _defaults = {'color': 6}
    _attributes = ['pinky_behavior']


class Inky(Ghost):
    _defaults = {'color': 11}
    _attributes = ['inky_behavior']


class Clyde(Ghost):
    _defaults = {'color': 8}
    _attributes = ['clyde_behavior']


class Teleport(BasicNoPhysicsObject):
    """Two special points in the maze that allow Chomper and the ghosts to teleport
    across."""
    _defaults = {'color': 0}
    _attributes = ['teleports']


class ChomperJudge(core.Judge):
    """Judge that computes the correct reward for eating ghosts. Other rewards are
    event-intrinsic.
    """
    def _calculate_reward(self, goals, evts):
        tot = 0
        player = self.world.objects['chomper_player']
        for event in evts:
            if isinstance(event, EatGhostEvent):
               tot += 100 * (2 ** (1 +  player.ghosts_eaten))
               player.ghosts_eaten += 1
        return tot

screen = """
XXXXXXXXXXXXXXXXXXXXXXXXXXX
X............X............X
XOXXXX.XXXXX.X.XXXXX.XXXXOX
X.........................X
X.XXXX.XX.XXXXXXX.XX.XXXX.X
X......XX....X....XX......X
XXXXXX.XXXXX X XXXXX.XXXXXX
XXXXXX.XXXXX X XXXXX.XXXXXX
XXXXXX.XX         XX.XXXXXX
T     .   XXXBXXX   .     T
XXXXXX.XX X IPC X XX.XXXXXX
XXXXXX.XX XXXXXXX XX.XXXXXX
XXXXXX.XX         XX.XXXXXX
XXXXXX.XX XXXXXXX XX.XXXXXX
X............X............X
X.XXXX.XXXXX.X.XXXXX.XXXX.X
XO...X.......*.......X...OX
XXXX.X.XX.XXXXXXX.XX.X.XXXX
XXXX.X.XX.XXXXXXX.XX.X.XXXX
X......XX....X....XX......X
X.XXXXXXXXXX.X.XXXXXXXXXX.X
X.........................X
XXXXXXXXXXXXXXXXXXXXXXXXXXX
"""

legend = {
    'X': 'blue_wall',
    '.': 'dot',
    'O': 'pellet',
    '*': 'chomper_player',
    'B': 'blinky',
    'I': 'inky',
    'P': 'pinky',
    'C': 'clyde',
    'T': 'teleport',
    }
objects, height, width = h.world.screen(screen, legend)

judge = ChomperJudge

house_exit = np.array([8, 12])
