import copy

import pixelworld.envs.pixelworld as pw
import pixelworld.envs.pixelworld.object_attributes as oa
from ..helpers import h, L

height = 100
width = 200

render_size = (200, 400)


class UnknownEnemyState(Exception):
    pass


class AlienMarchingStateObjectAttribute(pw.core.StateMachineObjectAttribute):
    """Causes the aliens to march back and forth, stepping down one pixel each time
    they reach the end."""
    _depends_on = ['initial_position']

    def _execute(self, obj, state):
        # moving right, and then down when we reach the end of our marching distance
        if state == 0:
            obj.position += (0, 1)
            if obj.position[1] > self._other_attr['initial_position'].get(obj)[1] + 10:
                obj.position += (1, 0)
                return 1
            return 0
        
        # moving left, and then down when we reach the end of our marching distance
        elif state == 1:
            obj.position += (0, -1)
            if obj.position[1] < self._other_attr['initial_position'].get(obj)[1]:
                obj.position += (1, 0)
                return 0
            return 1

        else:
            raise UnknownEnemyState()


class InvaderObject(pw.objects.ComplexSpriteObject):
    """The alien invaders"""
    _attributes = ['alien_marching_state', 'randomly_shoots', 'initial_position',
                   'alive', 'orientation']
    _defaults = {'animated': True,
                 'orientation': 3, 'randomly_shoots': 0.01
                 }


class AvatarObject(pw.objects.SelfSpriteObject):
    """The little spaceship that shoots"""
    _attributes = ['self_shoots', 'alive']


class AliensJudge(pw.core.Judge):
    """A judge that rewards you for killing aliens. How xenophobic.

    The episode terminates when all aliens are dead, or when the avatar dies.
    """
    _step_penalty = 0

    def _calculate_reward(self, goals, events):
        tot = 0
        for event in events:
            if isinstance(event, pw.events.KillEvent):
                if event.victim == self.world.objects['avatar'].id:
                    tot -= 1000
                elif isinstance(self.world.objects[event.victim], InvaderObject):
                    tot += 100

            elif isinstance(event, pw.events.BulletFireEvent):
                if isinstance(self.world.objects[event.bullet_id].bullet_shooter, AvatarObject):
                    tot -= 1

        return tot

    def _is_done(self, goals, events):
        aliens = self.world.objects.find(name='invader', visible=True)
        avatar = self.world.objects.find(name='avatar', visible=True)
        if len(aliens) == 0 or len(avatar) == 0:
            return True
        else:
            return False


invader = ['invader', {'sprites': h.sprite.load('aliens', 'alien')}]
invaders = []
for i in xrange(7):
    invader = copy.deepcopy(invader)
    invader[1]['position'] = (30, (i + 1) * 25)
    invaders.append(invader)

avatar = [['avatar', {'orientation': 1, 'position': (90, 100), 
                      'sprites': h.sprite.load('aliens', 'spaceship')}]]

objects = ['frame'] + invaders + avatar

agent = ['human', {'rate': 3}]
judge = AliensJudge
