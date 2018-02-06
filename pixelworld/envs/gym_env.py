from __future__ import print_function
from __future__ import absolute_import

import os
import os.path as osp

import gym
import gym.envs
from gym.envs.registration import EnvSpec
from gym.monitoring import monitor
import rllab.envs
from rllab.envs.base import Env, Step
from rllab.envs.gym_env import CappedCubicVideoSchedule
from rllab.envs.gym_env import FixedIntervalVideoSchedule
from rllab.envs.gym_env import NoVideoSchedule
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.misc import logger
import logging

from pixelworld.spaces import to_rllab_space 
from pixelworld.spaces_rllab import NamedBox, NamedDiscrete


# Modified from RLLab GymEnv
class GymEnv(Env, Serializable):
    """
    Like rllab's GymEnv, but expects a gym environment as a parameter,
    not the name of one.
    """

    def __init__(self, env, record_video=True, video_schedule=None,
            log_dir=None, timestep_limit=9999):
        # Ensure the version saved to disk doesn't monitor into our log_dir
        locals_no_monitor = dict(locals())
        locals_no_monitor['log_dir'] = None
        locals_no_monitor['record_video'] = False
        locals_no_monitor['video_schedule'] = None
        Serializable.quick_init(self, locals_no_monitor)

        self.env = env
        self._observation_space = to_rllab_space(env.observation_space)
        self._action_space = to_rllab_space(env.action_space)        
        self.env.spec = EnvSpec('GymEnv-v0')

        monitor.logger.setLevel(logging.WARNING)
        if not record_video:
            self.video_schedule = NoVideoSchedule()
        else:
            if video_schedule is None:
                self.video_schedule = CappedCubicVideoSchedule()
            else:
                self.video_schedule = video_schedule
        self.set_log_dir(log_dir)

        self._horizon = timestep_limit

    def set_log_dir(self, log_dir):
        if self.env.monitor.enabled:
            self.env.monitor.close()

        if log_dir is not None:
            self.env.monitor.start(log_dir, self.video_schedule)
            self.monitoring = True
        else:
            self.monitoring = False

    def __del__(self):
        # We started the monitor, so we have to make sure it gets closed when
        # we're deleted.
        if self.env.monitor.enabled:
            self.env.monitor.close()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()

    def describe(self):
        if hasattr(self.env, 'describe'):
            return self.env.describe()
        else:
            return {}

    def terminate(self):
        if self.monitoring:
            self.env.monitor.close()
        self.env.close()  # Prevents a memory leak  

    # NB: adding a close(self) routine that also closes the env does not have any effect


    def get_stats(self):
        if hasattr(self.env, 'get_stats'):
            return self.env.get_stats()
        else:
            return None
