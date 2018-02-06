import numpy as np
import itertools as it

import pixelworld.envs.pixelworld as pw


class CheatEvent(pw.core.Event):
    """Event that player tried to cheat, either by moving a larger disk onto a
    smaller disk, or by moving a non-top disk through some trickery."""
    _reward = -100


class WinEvent(pw.core.Event):
    """Event that the player won the game"""
    _reward = 1000
    _terminates = True


class PlaysHanoiClickObjectAttribute(pw.core.AbilityObjectAttribute):
    """Attribute that lets the player use LEFT, DOWN, RIGHT actions to play
    Hanoi."""
    _actions = ['LEFT', 'DOWN', 'RIGHT']

    # the tower that player has selected to move from
    _active_tower = None

    def _execute_action(self, obj, t, dt, agent_id, action):
        """Listen for LEFT/DOWN/RIGHT commands and manipulate the disks.

        Parameters
        ----------
        obj : Object
            The object that has the 'gripper' attribute.
        t : number
            The simulation time.
        dt : number
            The time since the last step.
        agent_id : int
            The id of the agent that is currently stepping.
        action : string
            The most recent action executed.
        """
        if action == 'LEFT':
            tower_chosen = self.world.objects.find(name='tower', label=0)[0]
        elif action == 'DOWN':
            tower_chosen = self.world.objects.find(name='tower', label=1)[0]
        elif action == 'RIGHT':
            tower_chosen = self.world.objects.find(name='tower', label=2)[0]
        else:
            return

        if self._active_tower is tower_chosen:
            # clicked twice on the same tower, do nothing and unselect the
            # tower
            tower_chosen.color = 1
            self._active_tower = None
        elif self._active_tower is None:
            # first click, highlight the clicked tower and select it
            tower_chosen.color = 2
            self._active_tower = tower_chosen
        else:
            # second click, move disk from previously clicked tower to newly
            # clicked tower, and unselect the previous tower
            self._move_disk(self._active_tower, tower_chosen)
            self._active_tower.color = 1
            self._active_tower = None

    def _move_disk(self, tower_from, tower_chosen):
        """Move the top disk from one tower to another. Also check to see if we've won.

        Parameters
        ----------
        tower_from : Object
            Tower from which to move the top disk

        tower_chosen : Object
            Tower to move the disk to
        """

        disk = tower_from.top_disk
        if disk is not None:
            disk.location = tower_chosen.label

        # check to see if we've won
        done = True
        for disk in self.world.objects['disk']:
            if disk.location != 2:
                done = False
                break

        # generate a WinEvent if we've won
        if done:
            event = WinEvent(self.world)


class PlaysHanoiSingleObjectAttribute(PlaysHanoiClickObjectAttribute):
    """Attribute that lets the player use LEFT_TO_CENTER and analogous actions to
    play Hanoi."""

    _actions = ['%s_TO_%s' % (a, b) for a, b in it.permutations(['LEFT', 'CENTER', 'RIGHT'], 2)]

    def _execute_action(self, obj, t, dt, agent_id, action):
        """Listen for LEFT_TO_CENTER and analogous commands and manipulate the disks.

        Parameters
        ----------
        obj : Object
            The object that has the 'gripper' attribute.
        t : number
            The simulation time.
        dt : number
            The time since the last step.
        agent_id : int
            The id of the agent that is currently stepping.
        action : string
            The most recent action executed.
        """
        towers = [self.world.objects.find(name='tower', label=i)[0] for i in xrange(3)]

        if action == 'LEFT_TO_CENTER':
            self._move_disk(towers[0], towers[1])
        elif action == 'LEFT_TO_RIGHT':
            self._move_disk(towers[0], towers[2])
        elif action == 'CENTER_TO_LEFT':
            self._move_disk(towers[1], towers[0])
        elif action == 'CENTER_TO_RIGHT':
            self._move_disk(towers[1], towers[2])
        elif action == 'RIGHT_TO_LEFT':
            self._move_disk(towers[2], towers[0])
        elif action == 'RIGHT_TO_CENTER':
            self._move_disk(towers[2], towers[1])


class Label(pw.core.NonNegativeIntegerObjectAttribute):
    """Label which tells us which tower is which. Values are 0, 1, 2"""
    pass


class TopDisk(pw.core.ObjectObjectAttribute):
    """The top disk of a tower, or None if the tower is empty."""

    _default_value = None

    _read_only = True

    _depends_on = ['label']

    def _get_value(self, idx):
        """get the attribute value corresponding to an Object index or array of
        indices.
        
        Parameters
        ----------
        idx : int | ndarray
            an Object index or array of indices
        
        Returns
        -------
        value : Object | ndarray
            The top disk or an array of top disks
        """
        if pw.utils.is_iterable(idx):
            return np.array([self._get_value(id) for id in idx], dtype=object)
        else:
            label = self._other_attr['label']
            label = label.get(idx)
            min_size = np.inf
            min_disk = None
            for disk in self.world.objects.find(name='disk'):
                if disk.location == label:
                    if disk.size < min_size:
                        min_size = disk.size
                        min_disk = disk
                
            return min_disk


class Size(pw.core.NonNegativeIntegerObjectAttribute):
    """The size of a disk, i.e., its number of children."""
    def _get_value(self, idx):
        """get the attribute value corresponding to an Object index or array of
        indices.
        
        Parameters
        ----------
        idx : int | ndarray
            an Object index or array of indices
        
        Returns
        -------
        value : int | ndarray
            The size or an array of sizes.
        """
        children = self.world.object_attributes['children']

        if pw.utils.is_iterable(idx):
            rv = np.array([len(children.get(id)) for id in idx])
        else:
            rv = len(children.get(idx))

        return rv

class Location(pw.core.NonNegativeIntegerObjectAttribute):
    """Which tower a disk is on. Corresponds to the 'label' attribute of the tower."""

    _initialize_after = ['position', 'label']
    def _set_value(self, idx, value):
        """set the attribute value corresponding to an Object index or array of
        indices. Sets the tower and checks for cheating.
        
        Parameters
        ----------
        idx : int | ndarray
            an Object index or array of indices
        value : int | ndarray
            the values to assign
        """
        # break up multiple sets, although this is basically guaranteed to
        # generate a cheat event
        if pw.utils.is_iterable(idx):
            if pw.utils.is_iterable(value):
                for id, val in zip(idx, value):
                    self._set_value(id, val)
            else:
                for id in idx:
                    self._set_value(id, value)

        # single value
        else:
            # get the disk
            disk = self.world.objects[idx]

            # new position x coordinate is that of the new tower
            new_position = np.zeros(2)
            new_position[1] = self.world.objects.find(name='tower', label=value)[0].position[1]
            
            # if they tried moving a non-top disk, that's a CheatEvent
            if disk is not self.world.objects.get(name='tower', label=disk.location).top_disk:
                event = CheatEvent(self.world)
                return

            # look for other disks on the new tower. our new y position will be
            # one above the smallest disk
            found_other_disk = False
            other_disk_size = np.inf
            for other_disk in self.world.objects.find(name='disk'):
                if other_disk.location == value:
                    found_other_disk = True
                    if other_disk.size < other_disk_size:
                        new_position[0] = other_disk.position[0] - 1
                        other_disk_size = other_disk.size

            # if there's no other disk, our new y position is at the bottom of
            # the tower, and the other_disk_size is infinite so that we never
            # get accused of cheating
            if not found_other_disk:
                new_position[0] = self.world.height - 1
                other_disk_size = np.inf

            if disk.size > other_disk_size:
                event = CheatEvent(self.world)
            else:
                self.world.object_attributes['position'].set(disk, new_position)
                self._set_data(idx, value)


class Tower(pw.objects.ComplexObject):
    """The towers that hold the disks"""
    _attributes = ['top_disk']


class Disk(pw.objects.ComplexObject):
    """The disks that are moved between towers"""
    _attributes = ['location', 'size']
    _defaults = {'color': 2}


class HanoiPlayer(pw.core.Object):
    """A non-physical object that exists solely to play Hanoi through actions"""
    _attributes = ['plays_hanoi_click', 'plays_hanoi_single']


class HanoiWorld(pw.core.PixelWorld):
    def __init__(self, objects=None, num_disks=3, **kwargs):
        assert num_disks <= 15

        # create the towers
        towers = [['tower', dict(position = (height - 15, 20*(i + 1)), 
                                 label=i,
                                 shape=[(j, 0) for j in xrange(15)])]
                  for i in xrange(3)]
        
        # create the disks
        disks = [['disk', dict(color=i + 2, position = (height - num_disks + i + 1, 0), 
                               location = 0,
                               shape = "X" * (i + 1) + " " + "X" * (i + 1))] 
                 for i in xrange(num_disks)][::-1]

        objects = ['hanoi_player'] + towers + disks

        super(HanoiWorld, self).__init__(objects=objects, **kwargs)

world = HanoiWorld
render_size = (200, 800)
height = 20
width = 80
