import numpy as np

from pixelworld.envs.pixelworld.core import Judge, Agent
from pixelworld.envs.pixelworld.events import CollisionEvent, PushEvent, LeaveScreenEvent

paddle_color = 5
ball_color = 3

#an agent that imperfectly tracks the ball
class SoloTennisAgent(Agent):
    def _get_action(self, obs, tools):
        ball_pos = np.array(np.where(obs==ball_color)).flatten()
        
        paddle_pos = np.mean(np.array(np.where(obs==paddle_color)), axis=1)
        
        if ball_pos.size == 0:  # couldn't find the ball
            return 'NOOP'
        elif tools['rng'].rand() > 0.1:  # track the ball
            if paddle_pos[0] > ball_pos[0]:
                return 'UP'
            elif paddle_pos[0] < ball_pos[0]:
                return 'DOWN'
            else:
                return 'NOOP'
        else:  # random action
            return tools['rng'].choice(['UP', 'DOWN', 'NOOP'])

#a judge that rewards successful paddle movements and collisions between the
#paddle and the ball, and ends the game once the ball leaves the screen
class SoloTennisJudge(Judge):
    _name = 'solo_tennis'
    _step_penalty = 0
    
    _reward_events = [
        {'event': 'collision',
            'params': {'indices': lambda judge, indices: \
                len(set(judge.paddle.children) & set(indices)) != 0 and \
                judge.ball.id in indices},
            'reward': 100,
        },
        {'event': 'push',
            'params': {'success': True},
            'reward': 1,
        },
        {'event': 'leave_screen',
            'params': {'indices': lambda judge, indices: \
                judge.ball.id in indices},
            'reward': -1000,
        },
    ]
    
    _termination_events = [
        {'event': 'leave_screen',
            'params': {'indices': lambda judge, indices: \
                judge.ball.id in indices},
        },
    ]
    
    paddle = None
    ball = None
    
    def prepare(self):
        self.paddle = self.world.objects['paddle']
        self.ball = self.world.objects['ball']

#three-sided wall
frame = [['frame', {'sides': ['top', 'right', 'bottom']}]]

#paddle
paddle_shape = '''
.
.
.
.
'''
paddle = [['self_big', {
            'name': 'paddle',
            'shape': paddle_shape,
            'position': (10, 0),
            'color': paddle_color,
            }]]

#bouncing ball
ball = [['basic', {
        'name': 'ball',
        'color': ball_color,
        'position': (10,1),
        'velocity': (1,1),}
        ]]

objects = frame + paddle + ball
judge = 'solo_tennis'
agent = 'solo_tennis'

