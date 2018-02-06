"""
Multi-agent demo. There are four agents: one human, one random, and two
FollowingAgents. There are five self objects, one controlled by the human
agent, one controlled by a random agent, two controlled by agents that try to
follow the human self, and one controlled by all four agents.

    Reward
    ------
    step: -1
    successful push event: 1
    
    Termination
    -----------
    Never    
"""

import itertools as it
from copy import copy

import numpy as np

import pixelworld.envs.pixelworld as pw


class FollowingAgent(pw.core.Agent):
    """An agent that follows the red player to the best of its ability.

    http://www.imdb.com/title/tt3235888/"""
    def __init__(self, world, agent_idx=None, *args, **kwargs):
        super(FollowingAgent, self).__init__(world, *args, **kwargs)
        self.agent_idx = agent_idx

    def _get_action(self, obs, tools):
        """Look at the observation to see where we should go next. We assume that the
        player's pixel is red, and that our pixel has color self.agent_idx + 2.

        With probability 25%, do nothing, so that the red player can get ahead
        of us eventually.
        """
        me = zip(*(obs == self.agent_idx + 2).nonzero())[0]
        human = zip(*(obs == 2).nonzero())[0]
        if tools['rng'].rand() < 0.25:
            return 'NOOP'

        if me[0] > human[0] and 'UP' in tools['actions']:
            return 'UP'
        elif me[0] < human[0] and 'DOWN' in tools['actions']:
            return 'DOWN'
        elif me[1] > human[1] and 'LEFT' in tools['actions']:
            return 'LEFT'
        elif me[1] < human[1] and 'RIGHT' in tools['actions']:
            return 'RIGHT'
        else:
            return 'NOOP'


class PushesJudge(pw.core.Judge):
    """A judge that rewards successful push events. With the step penalty, this
    means that agents get a reward of 0 for moving and a reward of -1 for not
    moving.
    """
    _reward_events = [dict(event='push',
                           params=dict(success=True), reward=1)]


# five self objects: one for the player, one for the random agent, two for the
# following agents, and one controlled by everyone
selves = [['self', dict(color=2, controller=0)], 
          ['self', dict(color=3, controller=1)],
          ['self', dict(color=4, controller=2)],
          ['self', dict(color=5, controller=3)],
          ['self', dict(color=6)], # controlled by everyone!
          ]
objects = ['frame'] + ['basic'] * 10 + selves
agent = ['human', 'random', 
         ['following', dict(agent_idx=2)],
         # this agent can only go left and right, so its following ability is limited
         ['following', dict(agent_idx=3, allowed_actions=['NOOP', 'LEFT', 'RIGHT'])]]
judge = 'pushes'
