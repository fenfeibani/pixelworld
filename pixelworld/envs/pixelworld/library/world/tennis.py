import numpy as np

from pixelworld.envs.pixelworld import core, objects, events
from ..helpers import L

paddle_color = 5
enemy_color = 6
ball_color = 3


class TracksBallObjectAttribute(core.AbilityObjectAttribute):
    """Attribute that makes enemy track the ball with 90% probability.

    If this attribute is set to False, the enemy will not track the ball.
    """
    def _execute_action(self, obj, t, dt, agent_id, action):
        """Track the ball.

        Parameters
        ----------
        obj : Object
            The enemy paddle.
        t : number
            The simulation time.
        dt : number
            Time since last step.
        agent_id : number
            Id of currently stepping agent.
        action : string
            Last selected action.
        """
        ball = self.world.objects['ball']
        
        if obj.rng.rand() <= self.get(obj):  # track the ball
            diff = None
            if obj.position[0] > ball.position[0]:
                diff = (-1, 0)
            elif obj.position[0] < ball.position[0]:
                diff = (1, 0)

            if diff is not None:
                blocked = False
                for idx in obj.children:
                    new_posn = self.world.objects[idx].position + diff
                    if len(self.world.objects.find(name='wall', position=new_posn)) > 0:
                        blocked = True
                if not blocked:
                    obj.position += diff


class Enemy(objects.UnpushableBigObject):
    _attributes = ['tracks_ball']


class TennisJudge(core.Judge):
    """A judge that rewards player for scoring points and penalizes player for
    being scored upon.
    """
    _step_penalty = 0

    # the ball
    ball = None

    def _calculate_reward(self, goals, evts):
        tot = 0
        for evt in evts:
            if isinstance(evt, events.LeaveScreenEvent):
                if self.ball.id in evt.indices:
                    if self.ball.position[1] <= 0:
                        tot -= 1000
                    else:
                        tot += 1000
                    self.ball.position = (self.world.height // 2, 1)
                    self.ball.velocity = (1, 1.01)

        return tot
    
    def prepare(self):
        self.ball = self.world.objects['ball']


#two-sided wall
frame = [['frame', {'sides': ['top', 'bottom']}]]

#paddle
paddle_shape = '''
.
.
.
.
.
'''
height = L('height', 20)
width = L('width', 20)
paddle = [['self_big', {
            'name': 'paddle',
            'shape': paddle_shape,
            'position': (height // 2, 0),
            'color': paddle_color,
            }]]

#enemy paddle
enemy = [['enemy', {
            'shape': paddle_shape,
            'position': (height // 2, width - 1),
            'color': enemy_color,
            }]]

#bouncing ball
ball = [['basic', {
        'name': 'ball',
        'color': ball_color,
        'position': (height // 2, 1),
        'velocity': (1, 1.01),}
        ]]

objects = frame + paddle + enemy + ball
judge = 'tennis'
agent = ['human', {'rate': 3}]
