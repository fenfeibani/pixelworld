import numpy as np

from pixelworld.envs.pixelworld import core, objects as objects_module, agents, events


class EatFoodEvent(core.Event):
    """Event that the snake ate a piece of food.

    Parameters
    ----------
    position : numpy array
        Location of the food that was eaten
    food_id : int
        Id of the food object that was eaten
    """
    _reward = 10
    _parameters = ['position', 'food_id']

    def _get_description(self):
        return '%s was eaten at position %s' % (self.world.objects[self.food_id].unique_name, self.position)


class WinEvent(core.Event):
    """Event that the snake grew to full size.
    """
    _reward = 1000

    def _get_description(self):
        return 'snake ate the world'


class HeadBehaviorObjectAttribute(core.AbilityObjectAttribute):
    """Attribute that controls the behavior of the head. 

    Causes the snake to respond to LEFT/UP/RIGHT/DOWN commands.

    Creates a KillEvent if the head tries to move into the wall or a body
    segment. Grow the snake if the head moves into a piece of food. Otherwise,
    take the oldest body segment and move it to the front of the snake, just
    behind the head.

    Setting this attribute to False will disable the head behavior and freeze
    the snake.
    """
    _move_direction = [
        np.array((0,-1)), # left
        np.array((-1,0)), # up
        np.array((0,1)),  # right
        np.array((1,0)),  # down
        ]
    _actions = ['LEFT', 'UP', 'RIGHT', 'DOWN']

    def _execute_action(self, obj, t, dt, agent_id, action):
        """Head behavior.

        Parameters
        ----------
        obj : Object
            The head object.
        t : number
            The simulation time.
        dt : number
            Time since last step.
        agent_id : number
            Id of currently stepping agent.
        action : string
            Last selected action
        """
        # change orientation based on LEFT/UP/RIGHT/DOWN commands
        if action in ['LEFT', 'UP', 'RIGHT', 'DOWN']:
            obj.orientation = ['LEFT', 'UP', 'RIGHT', 'DOWN'].index(action)

        # move the head
        old_position = obj.position
        obj.position += self._move_direction[obj.orientation]

        # eat food, if any
        food = self.world.objects.find(name='food', position=obj.position)
        fed = False
        for x in food:
            event = EatFoodEvent(self.world, food_id=x.id, position=x.position)
            self.world.remove_objects([x])
            fed = True

        if fed:
            # grow the snake by creating a new body segment
            self.world.create_object([['body'], dict(position=old_position, birth=t)])
        else:
            # find the oldest body segment and move it to the front, just
            # behind the head.
            body = self.world.objects.find(name='body')
            body = sorted(body, key=lambda x: x.birth)
            body[0].position = old_position
            body[0].birth = t

        # check for collisions with body or wall
        body_collisions = self.world.objects.find(name='body', position=obj.position)
        if len(body_collisions) > 0:
            event = events.KillEvent(self.world, victim=obj.id, reason='Ouroborous')
        frame_collisions = self.world.objects.find(name='wall', position=obj.position)
        if len(frame_collisions) > 0:
            event = events.KillEvent(self.world, victim=obj.id, reason='wall')

        # check if we grew to full size
        num_body = len(self.world.objects['body'])
        if num_body == (self.world.height - 2) * (self.world.width - 2) - 1:
            event = WinEvent(self.world)


class SpawnsFood(core.FloatObjectAttribute, core.SteppingObjectAttribute):
    """Attribute that causes food to be spawned.

    Value of the attribute is the probability that we will spawn a piece of
    food on each step.
    """
    _default_value = 0.1

    def _step_object(self, obj, t, dt, agent_id, action):
        """Spawn food in an unoccupied position with probability equal to the value of
        the attribute.

        Parameters
        ----------
        obj : Object
            The spawning object.
        t : number
            The simulation time.
        dt : number
            Time since last step.
        agent_id : number
            Id of currently stepping agent.
        action : string
            Last selected action
        """
        if self.rng.rand() < self.get(obj):
            # find an unoccupied position
            position = self.world.rng.randint(1, self.world.height - 1), \
                self.world.rng.randint(1, self.world.width - 1)
            while len(self.world.objects.find(position=position)) > 0:
                position = self.world.rng.randint(1, self.world.height - 1), \
                    self.world.rng.randint(1, self.world.width - 1)
                if self.world.rng.rand() > 0.99:
                    break

            if len(self.world.objects.find(position=position)) == 0:
                self.world.create_object(['food', dict(position=position)])


class BirthObjectAttribute(core.IntegerObjectAttribute):
    """The point in time at which a body segment was born. Used to figure out which
    body segment is oldest and should move."""
    pass


class Head(objects_module.BasicObject):
    """The head of the snake"""
    _attributes = ['head_behavior', 'orientation']
    _defaults = {'mass': 0, 'color': 4, 'zorder': 1}


class Body(objects_module.BasicObject):
    """The body segments of the snake"""
    _attributes = ['birth']
    _defaults = {'color': 4}


class Food(objects_module.BasicObject):
    """Snake food"""
    _defaults = {'color': 2, 'zorder': 0}


class FoodSpawner(core.Object):
    """A non-physical object that spawns the food"""
    _attributes = ['spawns_food']


class SnakeJudge(core.Judge):
    """A judge that waits for the snake to die."""
    def _calculate_reward(self, goals, evts):
        tot = 0
        for evt in evts:
            if isinstance(evt, events.KillEvent):
                tot -= 1000
            if isinstance(evt, WinEvent):
                tot += 1000
        return tot

    def _is_done(self, goals, evts):
        for evt in evts:
            if isinstance(evt, events.KillEvent):
                return True
            if isinstance(evt, WinEvent):
                return True
        return False


class SnakeWorld(core.PixelWorld):
    def __init__(self, objects=None, width=20, height=20, judge=None, **kwargs):
        if objects is None:
            objects = []

        head = [['head', {'position': (height // 2, width // 2)}]]
        body = [['body', {'position': (height // 2, width // 2 + 1 + i), 
                          'birth':-i - 1}] for i in xrange(5)]

        objects += ['frame'] + head + body + ['food'] * 5 + ['food_spawner']

        if judge is None:
            judge = SnakeJudge
        
        super(SnakeWorld, self).__init__(objects=objects, judge=judge, width=width, height=height,
                                         **kwargs)

world = SnakeWorld

agent = ['human', {'rate': 3}]

