import numpy as np

from pixelworld.envs.pixelworld import core, objects as objects_module, events
import pixelworld.envs.pixelworld.object_attributes as oa


width = 20
height = 12


class NeedsAirObjectAttribute(core.FloatObjectAttribute, core.InteractsObjectAttribute):
    _default_value = 200

    def _step_object(self, obj, t, dt, agent_id, action):
        """Decrement the air by the amount of time since the last step, and kill the
        player if it drops to zero or below. Also take care of coloring the
        oxygen level indicator bar.

        Parameters
        ----------
        obj : Object
            The object that has the InteractsObjectAttributes
        t : number
            The simulation time.
        dt : number
            The time since the last step.
        agent_id : int
            Id of the agent currently moving.
        action : string
            The most recent action executed.
        """
        air = self.get(obj)
        if air <= 0:
            obj.add_attribute('killed_by')
            obj.killed_by = 'asphyxiation'
            obj.alive = False
            
        self.set(obj, air - dt)

        super(NeedsAirObjectAttribute, self)._step_object(obj, t, dt, agent_id, action)

        self._set_indicator_colors(obj)

    def prepare(self):
        """Prepare by setting the oxygen indicator bar colors.
        """
        for obj in self.objects:
            self._set_indicator_colors(obj)
    
    def _get_normalized_x(self, obj):
        """Return the normalized x value, so that something at position x position 0
        has normalize x value 0, and something all the way to the right has
        normalized x value 1.

        obj : Object
            Object to calculate this for.
        """
        y, x = obj.position
        w = self.world.width
        return x/(w - 2)
    
    def _set_indicator_colors(self, obj):
        """Set the colors of the oxygen indicator bar based on the current oxygen
        level.

        obj : Object
            Object with the property (the sub).
        """
        air = self.get(obj)
        
        bar = self.world.objects['oxygen_indicator']
        for child in bar._children:
            x = self._get_normalized_x(child)
            child.color = 1 if x <= (air/self._default_value) else 2
        
    def _interact(self, obj1, obj2):
        """When the sub surfaces, reset the air, increment the number of surfacings,
        and either rescue the divers (when there are six of them), kill one
        diver (when there are fewer than six), or kill the sub (when there are
        no divers aboard).
                   
        Parameters
        ----------
        obj1 : Object
            The first object (the sub).

        obj2 : Object
            The second object.
        """
        if isinstance(obj2, Surface) and not obj1.surfacing_processed:
            obj1.surfacing_processed = True
            self.set(obj1, self._default_value)
            obj1.num_surfacings += 1
            if obj1.num_divers == 6:
                event = RescueEvent(self.world, num_divers=obj1.num_divers)
                obj1.num_divers = 0
                obj1.num_rescues += 1
            elif obj1.num_divers > 0:
                obj1.num_divers -= 1
            else:
                obj1.add_attribute('killed_by')
                obj1.killed_by = 'technicality'
                obj1.alive = False


class SeachaseOrients(core.AbilityObjectAttribute):
    """Attribute that causes the object's orientation to be set whenever either of
    the LEFT/RIGHT actions are used.
    """
    _depends_on = ['orientation']

    def _execute_action(self, obj, t, dt, agent_id, action):
        if action == 'LEFT':
            obj.orientation = 0
        elif action == 'RIGHT':
            obj.orientation = 2


class SubmarineDeadly(oa.DeadlyObjectAttribute):
    """Makes submarines deadly, but not to the bullets they fire."""
    def _should_kill_other(self, obj, obj2):
        if isinstance(obj2, objects_module.Bullet):
            return False
        return True


class Rescuable(core.InteractsObjectAttribute):
    """Makes it so that you can rescue divers if you have fewer than six already.
    """
    _step_after = ['position', 'velocity']
    def _interact(self, obj1, obj2):
        if isinstance(obj2, SeachasePlayer):
            if obj2.num_divers < 6:
                obj1.visible = False
                obj2.num_divers += 1


class RescueEvent(core.Event):
    """Event that you have rescued six divers."""
    pass


class NumDivers(core.NonNegativeIntegerObjectAttribute, core.SteppingObjectAttribute):
    """Number of divers currently aboard the sub."""
    _default_value = 0
    _step_after = ['needs_air']
    def _step_object(self, obj, t, dt, agent_id, action):
        num = self.get(obj)

        bar = self.world.objects['diver_indicator']
        for child in bar._children:
            child.color = 7 if child.position[1] <= num else 0
        


class NumRescues(core.NonNegativeIntegerObjectAttribute):
    """Number of times you have surfaced with six divers."""
    _default_value = 0


class NumSurfacings(core.NonNegativeIntegerObjectAttribute):
    """Number of times your sub has surfaced."""
    _default_value = 0


class SurfacingProcessed(core.BooleanObjectAttribute, core.SteppingObjectAttribute):
    """Have we processed the latest time the sub surfaced? Resets to False whenever
    you go below the surface."""
    _default_value = 0
    def _step_object(self, obj, t, dt, agent_id, action):
        if obj.position[0] > 1:
            self.set(obj, False)


class SeachaseSpawns(core.SteppingObjectAttribute):
    def _step_object(self, obj, t, dt, agent_id, action):
        """Spawn enemies. There are four horizontal "lanes" that the enemies are
        spawned in. Each lane spawns an enemy whenever it is empty, on either
        the left or right with equal probability.
        
        Parameters
        ----------
        obj : Object
            the Object to step
        t : float
            the current world time
        dt : float
            the amount of time by which the step should advance the world
        agent_id : int
            the id of the agent performing the action
        action : string
            the name of the action that is being performed
        """
        players = self.world.objects.find(name='seachase_player')
        assert len(players) == 1
        player = players[0]

        stuff = np.hstack([self.world.objects.find(name=name) for name in ['fish', 'submarine', 'diver']])

        # There are four horizontal "lanes" that the enemies are spawned
        # in. Each lane spawns an enemy whenever it is empty, on either the
        # left or right with equal probability.
        lanes = [2, 4, 6, 8]
        lane_stuff = [[x for x in stuff if x.position[0] == l] for l in lanes]
        for things, l in zip(lane_stuff, lanes):
            if len(things) == 0:
                u = self.rng.rand()
                # with probability 1/3, spawn a fish
                if u < 0.3333:
                    if self.rng.rand() < 0.5:
                        self.world.create_object([['fish'], dict(position=(l, 0), velocity=(0, 0.75))])
                    else:
                        self.world.create_object([['fish'], dict(position=(l, self.world.width - 1), 
                                                                 velocity=(0, -0.75))])

                # with probability 1/3, spawn a fish chasing a diver
                elif u < 0.6666:
                    if self.rng.rand() < 0.5:
                        self.world.create_object([['fish'], dict(position=(l, 0), velocity=(0, 0.75))])
                        self.world.create_object([['diver'], dict(position=(l, 0.5), velocity=(0, 0.75))])
                    else:
                        self.world.create_object([['fish'], dict(position=(l, self.world.width - 1), 
                                                                 velocity=(0, -0.75))])
                        self.world.create_object([['diver'], dict(position=(l, self.world.width - 1.5), 
                                                                  velocity=(0, -0.75))])

                # with probability 1/3, spawn an enemy sub
                else:
                    if self.rng.rand() < 0.5:
                        self.world.create_object([['submarine'], dict(position=(l, 0), velocity=(0, .5), 
                                                                      orientation=2)])
                    else:
                        posn = (self.rng.randint(3, 18), 19)
                        self.world.create_object([['submarine'], dict(position=(l, self.world.width - 1), 
                                                                      velocity=(0, -.5), 
                                                                      orientation=0)])

        # if we have surfaced at least twice, spawn a sub on the surface
        # whenever there isn't one.
        surface_subs = self.world.objects.find(name='surface_submarine')
        if len(surface_subs) == 0 and player.num_surfacings >= 2:
            if self.rng.rand() < 0.5:
                posn = (1, 0)
                self.world.create_object([['surface_submarine'], dict(position=posn, velocity=(0, .5))])
            else:
                posn = (1, self.world.width - 1)
                self.world.create_object([['surface_submarine'], dict(position=posn, velocity=(0, -.5))])


class SeachasePlayer(objects_module.BasicSelfObject):
    """A yellow submarine."""
    _attributes = ['self_shoots', 'needs_air', 'alive', 'orientation', 'seachase_orients', 'num_divers', 
                   'num_rescues', 'surfacing_processed', 'num_surfacings']
    _defaults = {'color': 5, 'zorder': 1}


class Surface(objects_module.BasicObject):
    """The line at the top where you can surface to get oxygen and drop off rescued
    divers."""
    _defaults = {'mass': 0, 'color': 8, 'zorder': 0}


class Fish(objects_module.BasicObject):
    """Why would you shoot a fish?"""
    _defaults = {'mass': 0, 'color': 4, 'zorder': 1}
    _attributes = ['alive', 'deadly']


class SurfaceSubmarine(objects_module.BasicObject):
    """Submarine on the surface that can kill you while you get oxygen."""
    _defaults = {'mass': 0, 'color': 1, 'zorder': 1}
    _attributes = ['deadly']


class Submarine(objects_module.BasicObject):
    """An enemy submarine that shoots bullets."""
    _attributes = ['randomly_shoots', 'submarine_deadly', 'alive']
    _defaults = {'randomly_shoots': 0.2, 'color': 1, 'mass': 0, 'zorder': 1}


class Diver(objects_module.BasicObject):
    """A diver for you to rescue."""
    _attributes = ['rescuable']
    _defaults = {'color': 7, 'mass': 0, 'zorder': 0.5}


class SeachaseSpawner(core.Object):
    """A non-physical object that exists to spawn enemies."""
    _attributes = ['seachase_spawns']


class OxygenIndicator(core.CompoundObject):
    """Indicates oxygen level with a color bar"""
    def __init__(self, world, *args, **kwargs):
        kwargs['children'] = [ ['immoveable', {'position': (height - 1, x)}] 
                               for x in xrange(1, width - 1)]
        super(OxygenIndicator, self).__init__(world, *args, **kwargs)


class DiverIndicator(core.CompoundObject):
    """Indicates number of divers aboard the sub"""
    _defaults = {'mass': 0}

    def __init__(self, world, *args, **kwargs):
        kwargs['children'] = [ ['immoveable', {'position': (height - 2, x), 'color': 0}] 
                               for x in xrange(1, 7)]
        super(DiverIndicator, self).__init__(world, *args, **kwargs)


class SeachaseJudge(core.Judge):
    """Implements one interpretation of the complicated Seachase scoring rules (see
    https://atariage.com/manual_html_page.html?SoftwareLabelID=424)"""

    def _calculate_reward(self, goals, evts):
        """Calculate rewards for killing things and rescuing divers.

        Parameters
        ----------
        goals : list[Goal]
            a list of goals that were achieved during the step
        events : list[Event]
            a list of Events that occurred during the step
        
        Returns
        -------
        reward : Number
            the reward to associate with the step
        """
        tot = 0
        for evt in evts:
            if isinstance(evt, events.KillEvent):
                assert not self.world.killing_deletes
                if isinstance(self.world.objects[evt.victim], Fish) or \
                        isinstance(self.world.objects[evt.victim], Submarine):
                    tot += min(90, 20 + 10 * self.world.objects['seachase_player'].num_rescues)

                if isinstance(self.world.objects[evt.victim], SeachasePlayer):
                    tot -= 1000
            if isinstance(evt, RescueEvent):
                tot += min(6 * 1000, 6 * 50 * self.world.objects['seachase_player'].num_rescues)

        to_remove = []
        for evt in evts:
            if isinstance(evt, events.LeaveScreenEvent):
                to_remove.extend(evt.indices)

        self.world.remove_objects([self.world.objects[id] for id in to_remove 
                                   if not isinstance(self.world.objects[id], SeachasePlayer)])
        return tot

    def _is_done(self, goals, evts):
        """Return True if the player died.

        Parameters
        ----------
        goals : list[Goal]
            a list of goals that were achieved during the step
        events : list[Event]
            a list of Events that occurred during the step
        
        Returns
        -------
        done : bool
            True if the episode has finished
        """
        for evt in evts:
            if isinstance(evt, events.KillEvent):
                if evt.victim == self.world.objects.get(name='seachase_player').id:
                    return True
        return False


surface = [[['surface'], dict(position=(1, i))] for i in xrange(1, 19)]

objects = ['seachase_player', ['frame', dict(zorder=2, sides=['left', 'right', 'top'])], 'oxygen_indicator', 
           'diver_indicator', 'seachase_spawner'] + surface
judge = SeachaseJudge
