import pixelworld.envs.pixelworld.core as core
import pixelworld.envs.pixelworld.objects as objects
import pixelworld.envs.pixelworld.object_attributes as oa
from ..helpers import h


class CollisionListeningObjectAttribute(core.ListeningObjectAttribute):
    """Object attribute that makes object listen for collision events involving
    it. Note that the callback gets called exactly once for each collision that
    involves an object with a CollideCallbackObjectAttribute-derived attribute.

    Subclasses should override _process_collision().

    Note that this does not inherit from ListeningObjectAttribute and has
    slightly different semantics.

    The value of this attribute is boolean; _process_collision will not be
    called for collisions that involve only objects without this attribute or
    which have this attribute set to False.
    """

    _step_after = ['position', 'velocity', 'acceleration']

    _selected_events = ['collision']
    
    def _process_event(self, evt, t, dt, agent_id, action):
        """Step the object attribute by looking for collisions and checking to see if
        any of the objects involved have this attribute.

        Parameters
        ----------
        evt : Event
            The event to process.
        t : number
            The simulation time.
        dt : number
            Time since last simulation step
        agent_id : int
            The id of the currently stepping agent.
        action : string
            The last selected action
        """
        if evt.name is 'collision':
            if any(self.get(i) for i in evt.indices):
                self._process_collision(evt, t, dt, agent_id, action)

    def _process_collision(self, evt, t, dt, agent_id, action):
        """Override this method in subclasses to respond to collision events correctly.

        Parameters
        ----------
        evt : Event
            The collision event
        t : number
            The simulation time.
        dt : number
            Time since last simulation step
        agent_id : int
            The id of the currently stepping agent.
        action : string
            The last selected action
        """
        raise NotImplementedError


class BrickClearEvent(core.Event):
    """Event that we destroyed a block with the ball"""
    _reward = 100


class BallCollidesObjectAttribute(CollisionListeningObjectAttribute):
    """Attribute that makes bricks die when the ball hits them."""
    _step_after = ['position', 'velocity']

    def _process_collision(self, evt, t, dt, agent_id, action):
        """Get rid of the brick and generate a BrickClearEvent whenever the ball hits a
        brick.

        Parameters
        ----------
        evt : Event
            The collision event
        t : number
            The simulation time.
        dt : number
            Time since last simulation step
        agent_id : int
            The id of the currently stepping agent.
        action : string
            The last selected action
        """
        for i in evt.indices:
            obj = self.world.objects[i]
            if isinstance(obj, Brick):
                obj.visible = False
                event = BrickClearEvent(self.world)


class Brick(objects.ImmoveableObject):
    """The bricks we are trying to destroy"""
    pass


class BricksDestroyedGoal(core.Goal):
    """True when all bricks have been destroyed"""
    
    def _is_achieved(self):
        return len(self.world.objects.find(name='brick', visible=True)) == 0


class BlockBreakerJudge(core.Judge):
    """Judge that makes you lose and get a penalty when the ball or paddle escapes the screen"""
    _reward_events = [{'event': 'leave_screen', 'reward': -1000}]
    _termination_events = [{'event': 'leave_screen'}]

class Ball(objects.BasicObject):
    _attributes = ['ball_collides']


paddle_color = 5
ball_color = 3
brick_color = 4

screen = """
WWWWWWWWWWWWWWWWWWWW
W                  W
W   BBBBBBBBBBBB   W
W   BBBBBBBBBBBB   W
W   BBBBBBBBBBBB   W
W   BBBBBBBBBBBB   W
W   BBBBBBBBBBBB   W
W   BBBBBBBBBBBB   W
W                  W
W                  W
W                  W
W                  W
W                  W
W       *          W
W                  W
W                  W
W                  W
W                  W
W                  W
W                  W
"""

legend = {
    'W': 'wall',
    'B': ['brick', {'color': brick_color}],
    'P': ['self', {'name': 'paddle'}],
    '*': ['ball', {
            'color': ball_color,
            'velocity': (1,1.01),}
          ],
    }
objects, height, width = h.world.screen(screen, legend)

paddle_shape = '''....'''
paddle = [['self_big', {
            'name': 'paddle',
            'shape': paddle_shape,
            'position': (19, 10),
            'color': paddle_color,
            }]]

objects += paddle

agent = ['human', {'rate': 3}]
goals = ['bricks_destroyed']
judge = BlockBreakerJudge
