import numpy as np
import pixelworld.envs.pixelworld.core as core
import pixelworld.envs.pixelworld.objects as objects


class Target(objects.BasicObject):
    """A target to shoot at."""
    _attributes = ['alive']


class Shooter(objects.SelfObject):
    """Shooter that shoots in the direction you last moved"""
    _attributes = ['self_shoots', 'orientation', 'orients', 'alive']

objects = ['frame'] + [[['shooter'], {'color': 2}]] + ['target'] * 10


class TargetsShotGoal(core.Goal):
    """True when all targets have been shot"""
    
    def _is_achieved(self):
        return len(self.world.objects.find(name='target', visible=True)) == 0


class ShootingJudge(core.Judge):
    """Judge that rewards you for shooting things, penalizes you for wasting
    bullets, and ends the episode when you run out of targets."""
    _step_penalty = 0
    _reward_events = [
        {'event': 'kill',
         'reward': 100},
        {'event': 'bullet_fire',
         'reward': -1}
        ]

goals = ['targets_shot']
judge = ShootingJudge
