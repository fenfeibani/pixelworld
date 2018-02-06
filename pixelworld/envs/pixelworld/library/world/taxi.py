import numpy as np
import pixelworld.envs.pixelworld.core as core
import pixelworld.envs.pixelworld.objects as objects
import pixelworld.envs.pixelworld.agents as agents
import pixelworld.envs.pixelworld.object_attributes as oa
from ..helpers import h, L


class ColorAssigner:
    """Callable object that assigns the next available color to locations"""
    _loc_colors = [3, 4, 6, 7]
    _next = 0
    def __call__(self):
        self._next += 1
        return self._loc_colors[(self._next - 1) % 4]
color_assigner = ColorAssigner()
    

class Locations(core.ListObjectAttribute):
    """List of locations available to the passenger spawner.
    """
    def prepare(self):
        for obj in self.objects:
            self.set(obj, self.world.objects['location'])


class SpawnsPassengers(core.BooleanObjectAttribute,
                       core.SteppingObjectAttribute):
    """Object attribute that spawns passengers at random locations whenever no
    visible passengers exist. If an invisible passenger exists, it will reuse
    that passenger.

    The value is boolean, and the behavior is disabled if it is set to False.
    """
    def _step_object(self, obj, t, dt, agent_id, action):
        """See class docstring.

        Parameters
        ----------
        t : number
            Simulation time
        dt : number
            Time since last step
        agent_id : int
            Id of agent the is currently acting
        action : string
            Last selected action.
        """
        if self.get(obj):
            passengers = self.world.objects.find(name='passenger', visible=True)
            if len(passengers) == 0:
                passengers = self.world.objects.find(name='passenger')
                assert len(passengers) <= 1
                loc = obj.locations[self.rng.randint(len(obj.locations))]
                if len(passengers) == 0:
                    self.world.create_object(['passenger', dict(position=loc.position)])
                else:
                    passengers[0].visible = True
                    passengers[0].color = 2
                    passengers[0].position = loc.position
                    passengers[0].destination = self.world.object_attributes['destination']._default_value(
                        passengers[0])

class IndicatesDestination(core.BooleanObjectAttribute, core.SteppingObjectAttribute):
    """Attribute that makes the indicator light show the color of the passenger's
    destination. Set to False to disable this behavior.
    """
    _default_value = True

    _step_after = ['spawns_passengers']

    def _step_object(self, obj, t, dt, agent_id, action):
        """Find the passenger if they exist and are visible, and use their
        destination's color as our color.

        Parameters
        ----------
        t : number
            Simulation time
        dt : number
            Time since last step
        agent_id : int
            Id of agent the is currently acting
        action : string
            Last selected action.
        """
        if self.get(obj):
            passengers = self.world.objects.find(name='passenger', visible=True)
            assert len(passengers) <= 1
            if len(passengers) > 0:
                obj.color = passengers[0].destination.color


class Destination(core.ObjectObjectAttribute):
    """The destination the passenger wants to go to.
    """
    _initialize_after = ['position']
    def _default_value(self, obj):
        """The default value is a random choice among the locations that the passenger
        is not currently at.

        We also take care of setting the color of the indicator here.
        """
        locations = self.world.objects.find(name='location')
        cur = self.world.objects.get(name='location', position=obj.position)
        rv = cur
        while (rv.position == cur.position).all():
            rv = locations[self.rng.randint(len(locations))]
        assert not (rv.position == obj.position).all()
        return rv


class PickedUp(oa.PushesObjectAttribute):
    """Attribute that passengers have once they have been picked up. It causes them
    to move with the taxi and respond to DROPOFF actions.
    """
    _actions = ['DROPOFF']
    _step_before = ['color', 'spawns_passengers']
    def _step(self, t, dt, agent_id, action):
        """Check for DROPOFF actions, and check legality of them when they occur. If
        legal, make the pasenger invisible and not picked-up.

        Parameters
        ----------
        t : number
            Simulation time
        dt : number
            Time since last step
        agent_id : int
            Id of agent the is currently acting
        action : string
            Last selected action.
        """
        if len(self.objects) == 0:
            if action == 'DROPOFF':
                event = BadDropoff(self.world)
            return

        obj = self.objects[0]
        if action == 'DROPOFF':
            locations = self.world.objects['location']
            if (obj.position == obj.destination.position).all():
                obj.visible = False
                obj.remove_attribute(self._name)
                event = GoodDropoff(self.world)
                return
            else:
                event = BadDropoff(self.world)

        super(PickedUp, self)._step(t, dt, agent_id, action)

class BadPickup(core.Event):
    """Event that the taxi tried to pick up a non-existent passenger"""
    _reward = -100

class BadDropoff(core.Event):
    """Event that the taxi tried to drop a passenger off when no passenger was
    aboard or at a location other than the passenger's destination."""
    _reward = -100

class GoodDropoff(core.Event):
    """Event that a passenger was dropped off at their destination"""
    _reward = 10

class GoodPickup(core.Event):
    """Event that a passenger was picked up successfully"""
    _reward = 10

class Taxiing(core.AbilityObjectAttribute):
    """Attribute that the taxi has to make it able to pick up passengers.
    """
    _actions = ['PICKUP', 'DROPOFF']
    def _execute_action(self, obj, t, dt, agent_id, action):
        if action == 'PICKUP':
            passengers = self.world.objects.find(name='passenger', position=obj.position, visible=True)
            if len(passengers) == 0:
                event = BadPickup(self.world)
            elif len(passengers) == 1:
                if not hasattr(passengers[0], 'picked_up') or not passengers[0].picked_up:
                    passengers[0].add_attribute('picked_up')
                    passengers[0].color = 8
                    event = GoodPickup(self.world)
                else:
                    event = BadPickup(self.world)
            else:
                assert False, 'should only be one passenger at most'

class PassengerSpawner(core.Object):
    """Non-physical object that spawns passengers"""
    _attributes = ['spawns_passengers', 'locations']

class Passenger(objects.BasicObject):
    """A passenger"""
    _attributes = ['destination']
    _defaults = {'mass': 0, 'zorder': 2, 'color': 2}

class Location(objects.BasicObject):
    """One of the special locations from which a passenger arrives and to which a
    passenger might want to go"""
    _defaults = {'mass': 0, 'color': lambda x: color_assigner(), 'zorder': 0}

class Taxi(objects.BasicSelfObject):
    """The taxi. It's hard to break even in this business."""
    _attributes = ['taxiing']
    _defaults = {'color': 5, 'zorder': 1, 'mass': 0}

class Indicator(objects.BasicObject):
    """An indicator that shows where the current passenger wants to go."""
    _attributes = ['indicates_destination']
    _defaults = {'color': 0}

screen = """
XXXXXXXXXX
X.  X   .XI
X*  X    X
X        X
X X   X  X
X.X   X. X
XXXXXXXXXX
"""

legend = {
    '*': 'taxi',
    '.': 'location',
    'X': 'wall',
    'I': 'indicator',
}

objects, height, width = h.world.screen(screen, legend)
objects.append('passenger_spawner')

