"""
A version of aliens where each object is a single pixel.
"""
import copy

import pixelworld.envs.pixelworld as pw
import pixelworld.envs.pixelworld.object_attributes as oa
from ..helpers import h, L

sis = pw.library.import_item('world', 'aliens_sprites')
AlienMarchingStateObjectAttribute = sis.AlienMarchingStateObjectAttribute
AliensJudge = sis.AliensJudge


class AlienDeadly(oa.DeadlyObjectAttribute):
    """Modify DeadlyObjectAttribute so that aliens die when they crash into
    things other than alien bullets and other aliens."""
    def _should_kill_self(self, obj, obj2):
        if isinstance(obj2, PixelInvaderObject) or isinstance(obj2, AlienBullet):
            return False
        return True

class AlienBulletDeadly(oa.BulletDeadlyObjectAttribute):
    def _should_kill_other(self, obj, obj2):
        if isinstance(obj2, PixelInvaderObject):
            return False
        return True

class AlienBullet(pw.objects.Bullet):
    _attributes_removed = ['bullet_deadly']
    _attributes = ['alien_bullet_deadly']
    _defaults = {'zorder': -1}

class PixelInvaderObject(pw.objects.BasicObject):
    """The alien invaders"""
    _attributes = ['alien_marching_state', 'randomly_shoots', 'initial_position',
                   'alive', 'orientation', 'alien_deadly', 'ammo']
    _defaults = {'animated': True,
                 'orientation': 3, 'randomly_shoots': 0.01, 'mass': 0,
                 'color': 1, 'ammo': 'alien_bullet',
                 }


class PixelAvatarObject(pw.objects.SelfObject):
    """The little spaceship that shoots"""
    _attributes = ['self_shoots', 'alive']
    _attributes_removed = ['orients']
    _defaults = {'orientation': 1, 'color': 2}


class Barricade(pw.objects.BasicObject):
    """A barricade that can be destroyed by bullets"""
    _attributes = ['alive']
    _defaults = {'color': 4}


screen = """
-------------------------------------
- A A A A A A A A A A               -
-                                   -
- A A A A A A A A A A               -
-                                   -
- A A A A A A A A A A               -
-                                   -
- A A A A A A A A A A               -
-                                   -
-                                   -
-                                   -
-                                   -
-                                   -
-                                   -
-                                   -
-                                   -
-   BBBB    BBBB     BBBB    BBBB   -
-   B  B    B  B     B  B    B  B   -
-             P                     -
-------------------------------------
"""

legend = {
    '-': 'wall',
    'A': 'pixel_invader',
    'P': 'pixel_avatar',
    'B': 'barricade',
    'b': 'bullet',
    'a': 'alien_bullet',
    }
objects, height, width = h.world.screen(screen, legend)

class PixelAliensJudge(pw.core.Judge):
    """A judge that rewards you for killing aliens. How xenophobic.

    The episode terminates when all aliens are dead, or when the avatar dies.
    """
    _step_penalty = 0

    def _calculate_reward(self, goals, events):
        tot = 0
        for event in events:
            if isinstance(event, pw.events.KillEvent):
                if event.victim == self.world.objects['pixel_avatar'].id:
                    tot -= 1000
                elif isinstance(self.world.objects[event.victim], PixelInvaderObject):
                    tot += 100

            elif isinstance(event, pw.events.BulletFireEvent):
                if isinstance(self.world.objects[event.bullet_id].bullet_shooter, PixelAvatarObject):
                    tot -= 1

        return tot

    def _is_done(self, goals, events):
        aliens = self.world.objects.find(name='pixel_invader', visible=True)
        avatar = self.world.objects.find(name='pixel_avatar', visible=True)
        if len(aliens) == 0 or len(avatar) == 0:
            return True
        else:
            return False



agent = ['human', {'rate': 3}]
judge = PixelAliensJudge
