from pixelworld.envs.pixelworld import core, objects as objects_module, events, object_attributes as oa


class RobotDeadlyObjectAttribute(oa.DeadlyObjectAttribute):
    """Makes robots deadly: robots kill the player, but also kill one another when
    they crash into each another or into previous crashes.

    Setting this to False will make the robot not deadly.
    """
    #make sure that robots will kill you immediately if they catch you or if
    #you teleport into them
    _step_after = ['chases_player', 'teleports']

    def _should_kill_self(self, obj, obj2):
        if isinstance(obj2, Robot):
            if len(self.world.objects.find(name='crash', position=obj.position)) == 0:
                self.world.create_object(['crash', dict(position=obj.position)])
            obj.remove_attribute('chases_player')
            obj2.remove_attribute('chases_player')
            return True
        elif isinstance(obj2, Crash) or isinstance(obj2, objects_module.WallObject):
            obj.remove_attribute('chases_player')
            return True
        else:
            assert isinstance(obj2, Player)
            return False

class ChasesPlayerObjectAttribute(core.BooleanObjectAttribute, core.SteppingObjectAttribute):
    """Attribute that causes robots to chase the player.

    Setting this to False will make the robot not chase the player.
    """
    #make the robots chase your last position when you teleport
    _step_before = ['teleports']

    #make sure that the robots move after you do
    _step_after = ['position', 'pushes']

    def _step_object(self, obj, t, dt, agent_id, action):
        """Move towards the player's position.

        Parameters
        ----------
        obj : Object
            The object with this attribute.
        t : number
            Simulation time
        dt : number
            Time since last simulation step.
        agent_id : int
            Id of the agent that is currently moving
        action : string
            Last selected action
        """
        if self.get(obj):
            position = obj.position

            #player has already moved, so figure out where the player is and go
            #towards there
            player_position = self.world.objects['player'].position

            if player_position[0] < position[0]:
                obj.position += (-1, 0)
            elif player_position[0] > position[0]:
                obj.position += (1, 0)
            if player_position[1] < position[1]:
                obj.position += (0, -1)
            elif player_position[1] > position[1]:
                obj.position += (0, 1)


class TeleportsObjectAttribute(core.AbilityObjectAttribute):
    """Attribute that allows the player to teleport to a random location.

    Setting this to False will disable teleportation.
    """
    _actions = ['TELEPORT']
    _depends_on = ['position']
    def _execute_action(self, obj, t, dt, agent_id, action):
        """Teleport if that action was chosen.

        Parameters
        ----------
        obj : Object
            The object with this attribute.
        t : number
            Simulation time
        dt : number
            Time since last simulation step.
        agent_id : int
            Id of the agent that is currently moving
        action : string
            Last selected action
        """
        if action == 'TELEPORT':
            position = self._other_attr['position']
            obj.position = position._default_value(obj)


class Robot(objects_module.BasicObject):
    """A killer robot."""
    _attributes = ['chases_player', 'robot_deadly', 'alive']
    _defaults = {'color': 3, 'mass': 0}


class Crash(objects_module.ImmoveableObject):
    """The remains of a killer robot."""
    _defaults = {'color': 5, 'mass': 0}


class Player(objects_module.BasicSelfObject):
    """Our plucky protagonist."""
    _attributes = ['teleports', 'alive']
    _defaults = {'color': 2}


objects = ['frame', 'player'] + ['robot'] * 10


class RobotsJudge(core.Judge):
    """A judge that rewards you for killing robots and penalizes you for dying. The
    judge ends the episode when you die or when there are no more robots.
    """

    # tracks if player has ever died
    _player_dead = False
    
    def prepare(self):
        try:
            self.world.object_attributes['robot_deadly']._step(0, 0, 0, 'NOOP')
        except KeyError:
            # no robots exist
            pass

    def _calculate_reward(self, goals, evts):
        """Calculate a reward.

        Parameter
        ---------
        evts : list of Events
            Events that took place this step.
        """
        tot = 0
        for event in evts:
            if isinstance(event, events.KillEvent):
                if self.world.objects[event.victim].name == 'player':
                    tot += -1000
                else:
                    tot += 100
        return tot

    def _is_done(self, goals, evts):
        """Decide whether to terminate the episode.

        Parameter
        ---------
        evts : list of Events
            Events that took place this step.
        """
        if self._player_dead:
            return True

        for event in evts:
            if isinstance(event, events.KillEvent):
                if self.world.objects[event.victim].name == 'player':
                    self._player_dead = True
                    return True

        if len(self.world.objects.find(name='robot', alive=True)) == 0:
            return True

        return False

judge = RobotsJudge
