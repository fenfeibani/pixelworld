"""
Pixelzumer PixelWorld environment
"""
from ..helpers import h, L

from pixelworld.envs.pixelworld import core, randomizers

#------------------------------------------------------------------------------
# Events
#------------------------------------------------------------------------------

class TypedEvent(core.Event):
    _parameters = ['type']
    
    _class_tree_attributes = ['_rewards']
    
    _rewards = {}
    
    @property
    def _reward(self):
        return self._rewards[self.type]

class DeathEvent(core.Event):
    _parameters = ['reason']
    _reward = -1000
    
    def _get_description(self):
        return 'Death by %s' % (self.reason)

class CollectEvent(TypedEvent):
    _rewards = {
        'COIN':     1000,
        'KEY':      100,
        'SWORD':    100,
        'AMULET':   200,
        'TORCH':    3000,
        'WIN':      10000,
    }
    
    @property
    def _reward(self):
        return self._rewards[self.type]
    
    def _get_description(self):
        return 'Collected %s' % (self.type)

class DoorEvent(core.Event):
    _reward = 300
    
    def _get_description(self):
        return 'Opened door'

class KillEvent(TypedEvent):
    _rewards = {
        'SKULL':    2000,
        'SPIDER':   3000,
    }
    
    def _get_description(self):
        return 'Killed %s' % (self.type)


#------------------------------------------------------------------------------
# Judge
#------------------------------------------------------------------------------

class PixelzumaJudge(core.Judge):
    _step_penalty = 0
    
    _termination_events = [
        {'event': 'collect', 'params': {'type': 'WIN'}},
    ]


#------------------------------------------------------------------------------
# Attributes
#------------------------------------------------------------------------------

class BlockingObjectAttribute(core.BooleanObjectAttribute):
    """ does this object block movement """
    pass

class MonsterBlockingObjectAttribute(core.BooleanObjectAttribute):
    """ does this object block monster movement """
    pass

class CollectibleObjectAttribute(core.BooleanObjectAttribute):
    """ is this a collectible object """
    pass

class EffectObjectAttribute(core.StringObjectAttribute):
    """ effect triggered by touching this object """
    pass

class ItemObjectAttribute(core.StringObjectAttribute):
    """ inventory item added by touching this object """
    pass

class LockedObjectAttribute(core.BooleanObjectAttribute):
    """ is this a locked door """
    pass

class InventoryObjectAttribute(core.ListObjectAttribute):
    """ inventory """
    pass

class SupportObjectAttribute(core.BooleanObjectAttribute):
    """ can Pedro stand on this object """
    pass

class ConveyObjectAttribute(core.PointObjectAttribute):
    """ movement direction of a conveyor """
    pass

class PixelzumaDeadlyObjectAttribute(core.BooleanObjectAttribute):
    """ is this a deadly thing (monster) """
    pass

class KillableObjectAttribute(core.BooleanObjectAttribute):
    """ can a monster be killed by sword """
    pass

class KamikaziableObjectAttribute(core.BooleanObjectAttribute):
    """ can a monster be killed by running into it and dying"""
    pass

class FallObjectAttribute(core.IntegerObjectAttribute):
    """ length of current fall - if too long Pedro is no more """
    pass

class SimpleVelocityObjectAttribute(core.PointObjectAttribute):
    """ velocity that is not connected to the physics engine """
    pass

class AnchorObjectAttribute(core.PointObjectAttribute):
    """ original position of moving objects """
    _depends_on = ['position']

    def prepare(self):
        for obj in self.objects:
            obj.anchor = obj.position

class RangeObjectAttribute(core.IntegerObjectAttribute, core.SteppingObjectAttribute):
    """
    Movement range for moving monsters
    This is where monster movement is calculated
    """
    _depends_on = ['position', 'anchor', 'simple_velocity']
    
    _step_after = ['pedro']
    
    def _step_object(self, obj, t, dt, agent_id, action):
        # switch direction if reached end of range or blocked
        next_pos = obj.position + obj.simple_velocity
        end_of_range = max(abs(next_pos - obj.anchor)) > obj.range
        blocked = self.world.objects.get(position=next_pos, monster_blocking=True) is not None
        if end_of_range or blocked:
            obj.simple_velocity = -obj.simple_velocity

        # move object
        obj.position += obj.simple_velocity


class BouncingObjectAttribute(core.SteppingObjectAttribute):
    """
    Bouncing monsters.
    This is where bouncing monster movement is calculated
    """
    _depends_on = ['position', 'simple_velocity']
    
    _step_after = ['pedro']
    
    def _step_object(self, obj, t, dt, agent_id, action):
        # switch direction if reached end of range or blocked
        next_pos = obj.position + obj.simple_velocity
        blocked = self.world.objects.get(position=next_pos, monster_blocking=True) is not None
        if blocked:
            new_velo = obj.simple_velocity
            new_velo[0] = -obj.simple_velocity[0]
            obj.simple_velocity = new_velo
            next_pos = obj.position + obj.simple_velocity
            blocked = self.world.objects.get(position=next_pos, monster_blocking=True) is not None
            if blocked:
                new_velo = obj.simple_velocity
                new_velo[1] = -obj.simple_velocity[1]
                obj.simple_velocity = new_velo
                next_pos = obj.position + obj.simple_velocity
                blocked = self.world.objects.get(position=next_pos, monster_blocking=True) is not None
                assert not blocked

        # move object
        obj.position += obj.simple_velocity


class FlickerDeadlyObjectAttribute(core.SteppingObjectAttribute):
    """This makes the traps flicker in and out of existence by toggling obj.visible
    and obj.deadly."""
    def _step_object(self, obj, t, dt, agent_id, action):
        if t % 4 == 0:
            obj.visible = not obj.visible
            obj.pixelzuma_deadly = not obj.pixelzuma_deadly

class FlickerSupportObjectAttribute(core.SteppingObjectAttribute):
    """This makes the floor flicker in and out of existence by toggling obj.visible
    and obj.support."""
    def _step_object(self, obj, t, dt, agent_id, action):
        if t % 8 == 0:
            obj.visible = not obj.visible
            obj.support = not obj.support

class PedroObjectAttribute(core.BooleanObjectAttribute, core.SteppingObjectAttribute):
    """
    This attribute identifies Pedro.
    All Pedro related dynamics are computed here.
    """
    _depends_on = ['position', 'simple_velocity']
    _actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'JUMP', 'JUMPLEFT', 'JUMPRIGHT', 'NOOP']

    _action_position = {
        'UP':        (-1,  0),
        'DOWN':      ( 1,  0),
        'LEFT':      ( 0, -1),
        'RIGHT':     ( 0,  1),
    }
    _action_velocity = {
        'JUMP':      (-1,  0),
        'JUMPLEFT':  (-1, -1),
        'JUMPRIGHT': (-1,  1),
    }

    MAX_FALL = 2

    def _get_object_at(self, pos):
        """
        Return the first non-Pedro object at specified position, or None if none.
        """
        obj_list = self.world.objects.find(position=pos)
        if len(obj_list) == 0:
            return None
        if getattr(obj_list[0], 'pedro', False):
            if len(obj_list) > 1:
                assert not getattr(obj_list[1], 'pedro', False), "Two Pedros at %s" % pos
                return obj_list[1]
            else:
                return None
        else:
            return obj_list[0]

    def _step(self, t, dt, agent_id, action):
        remove_list = []
        reset = False

        # get pedro object
        obj_list = self.world.objects.find(pedro=True)
        assert len(obj_list) == 1, "Must have exactly one pedro"
        pedro = obj_list[0]
        position = pedro.position

        # get object at and below Pedro's position (or None)
        obj_at = self._get_object_at(position)
        obj_under = self._get_object_at(position + (1,0))

        # find support
        supported_at = obj_at is not None and getattr(obj_at, 'support', False)
        supported_under = obj_under is not None and getattr(obj_under, 'support', False)
        any_support = supported_under or supported_at

        # find velocity and position change
        velocity = (0,0)
        movement = (0,0)
        if any_support:
            # stop falling
            if pedro.fall > self.MAX_FALL and not supported_at:
                event = DeathEvent(self._world, reason='fall')
                reset = True
            pedro.fall = 0
            # conveyor
            if supported_under:
                movement = getattr(obj_under, 'convey', (0,0))
            # actions (override conveyor)
            if action in self._action_position and (action != 'UP' or supported_at):   # up requires support at, other actions can do with support under
                movement = self._action_position[action]
            if action in self._action_velocity:
                velocity = self._action_velocity[action]
                movement = velocity
        else:
            # fall
            velocity = pedro.simple_velocity
            if velocity[0] < 1:
                velocity[0] = 1
            pedro.fall += 1
            movement = velocity
        next_pos = position + movement

        # collect items: add to inventory (if item has collectible string), add effect, remove item
        if obj_at is not None and getattr(obj_at, 'collectible', False):
            item = getattr(obj_at, 'item', None)
            if item is not None:
                pedro.inventory.append(item)
            effect = getattr(obj_at, 'effect', None)
            if effect is not None and effect != '':
                event = CollectEvent(self._world, type=effect)
            obj_at.collectible = False
            remove_list.append(obj_at)

        # open doors
        obj_at_target = self._get_object_at(next_pos)
        if obj_at_target is not None and getattr(obj_at_target, 'locked', False):
            if 'KEY' in pedro.inventory:
                pedro.inventory.remove('KEY')
                obj_at_target.blocking = False
                obj_at_target.locked = False
                remove_list.append(obj_at_target)
                event = DoorEvent(self._world)

        # check for blocking
        if obj_at_target is not None and getattr(obj_at_target, 'blocking', False):
            next_pos = position
            velocity = (0,0)

        # fight monsters
        if not reset:
            for target in [obj_at, obj_at_target]:
                if target is not None and getattr(target, 'pixelzuma_deadly', False):
                    if getattr(target, 'killable', False) and 'SWORD' in pedro.inventory:
                        # kill it!
                        pedro.inventory.remove('SWORD')
                        target.pixelzuma_deadly = False
                        remove_list.append(target)
                        event = KillEvent(self._world, type=target.effect)
                    elif 'AMULET' in pedro.inventory:
                        # stun it
                        pedro.inventory.remove('AMULET')
                        target.pixelzuma_deadly = False
                    else:
                        # die
                        reset = True
                        event = DeathEvent(self._world, reason=target.effect)
                        # running into monsters kills them too
                        if getattr(target, 'killable', False) or getattr(target, 'kamikaziable', False):
                            target.pixelzuma_deadly = False
                            remove_list.append(target)

        if reset:
            # reset (death)
            pedro.position = pedro.anchor
            pedro.simple_velocity = (0,0)
            pedro.fall = 0
        else:
            # update velocity and position
            pedro.simple_velocity = velocity
            pedro.position = next_pos

        # remove any objects marked for removal
        for obj in remove_list:
            # obj.remove()
            obj.visible = False


#------------------------------------------------------------------------------
# Objects
#------------------------------------------------------------------------------

class BasicNoPhysicsObject(core.Object):
    _attributes = [
        'color',
        'position',
        'mass',
        'visible',
        'zorder',
    ]
    _defaults = {'mass': 0}   # mass 0 ignores physics

class CollectibleObject(BasicNoPhysicsObject):
    _attributes = ['collectible', 'effect']

class CollectibleItemObject(CollectibleObject):
    _attributes = ['item']

class PlatformObject(BasicNoPhysicsObject):
    _attributes = ['support', 'blocking', 'monster_blocking']

class FlickerPlatformObject(BasicNoPhysicsObject):
    _attributes = ['support', 'flicker_support']

class FlickerPlatformBlockingObject(BasicNoPhysicsObject):
    _attributes = ['support', 'flicker_support', 'blocking', 'monster_blocking']

class ConveyorObject(PlatformObject):
    _attributes = ['convey']

class RightConveyorObject(ConveyorObject):
    _defaults = {'color': 2, 'convey': (0,1)}

class LeftConveyorObject(ConveyorObject):
    _defaults = {'color': 3, 'convey': (0,-1)}

class LadderObject(BasicNoPhysicsObject):
    _defaults = {'color': 4}
    _attributes = ['support']

class RopeObject(BasicNoPhysicsObject):
    _defaults = {'color': 5}
    _attributes = ['support']

class MonsterObject(BasicNoPhysicsObject):
    _attributes = ['pixelzuma_deadly', 'killable', 'effect']

class MovingMonsterObject(MonsterObject):
    _attributes = ['simple_velocity', 'anchor', 'range']
    _defaults = {'simple_velocity': (0,1), 'range': 3}

class SkullObject(MovingMonsterObject):
    _defaults = {'color': 6, 'effect': 'SKULL'}

class BouncingSkullObject(MonsterObject):
    _attributes = ['simple_velocity', 'bouncing']
    _defaults = {'simple_velocity': (-1, -1), 'effect': 'BOUNCING_SKULL'}

class SpiderObject(MovingMonsterObject):
    _defaults = {'color': 13, 'effect': 'SPIDER', 'range': 10}

class SnakeObject(MonsterObject):
    _attributes = ['kamikaziable']
    _defaults = {'color': 14, 'effect': 'SNAKE', 'killable': False, 'kamikaziable': True}

class TrapObject(BasicNoPhysicsObject):
    _attributes = ['pixelzuma_deadly', 'flicker_deadly', 'effect']
    _defaults = {'effect': 'TRAP'}

class LavaObject(BasicNoPhysicsObject):
    _attributes = ['pixelzuma_deadly', 'effect']
    _defaults = {'color': 5, 'effect': 'LAVA'}

class ExitObject(BasicNoPhysicsObject):
    _attributes = ['monster_blocking']
    _defaults = {'color': 7}

class WinObject(CollectibleObject):
    _defaults = {'color': 7, 'effect': 'WIN'}

class KeyObject(CollectibleItemObject):
    _defaults = {'color': 8, 'item': 'KEY', 'effect': 'KEY'}

class SwordObject(CollectibleItemObject):
    _defaults = {'color': 9, 'item': 'SWORD', 'effect': 'SWORD'}

class AmuletObject(CollectibleItemObject):
    _defaults = {'color': 9, 'item': 'AMULET', 'effect': 'AMULET'}

class TorchObject(CollectibleItemObject):
    _defaults = {'color': 9, 'item': 'TORCH', 'effect': 'TORCH'}

class CoinObject(CollectibleObject):
    _defaults = {'color': 10, 'effect': 'COIN'}

class DoorObject(PlatformObject):
    _attributes = ['locked']
    _defaults = {'color': 11}

class PedroObject(BasicNoPhysicsObject):
    _defaults = {'color': 12, 'zorder': 9999}
    _attributes = ['pedro', 'anchor', 'simple_velocity', 'fall', 'inventory']


#------------------------------------------------------------------------------
# Pixelzuma World
#------------------------------------------------------------------------------

# Screen layout, see legend
screen = """
------------------
-  -          -  -
W  D    *     D  W
----- --H-- ------
- K     H   |    -
-       H   |    -
--H-  <<<<< | -H--
- H            H -
- H     @      H -
------------------
"""

legend = {
    '-': 'platform',
    '~': 'flicker_platform',
    '_': 'flicker_platform_blocking',
    'D': 'door',
    'H': 'ladder',
    '|': 'rope',
    '>': 'right_conveyor',
    '<': 'left_conveyor',
    '*': 'pedro',
    '@': 'skull',
    'B': 'bouncing_skull',
    'S': 'spider',
    'N': 'snake',
    'K': 'key',
    '!': 'sword',
    '^': 'torch',
    '?': 'amulet',
    '$': 'coin',
    'X': 'exit',
    'T': 'trap',
    'L': 'lava',
    'W': 'win',
    }

objects, height, width = h.world.screen(screen, legend)

rate = L('rate', 3)
judge = PixelzumaJudge
agent = ['human', {'rate': rate}]

