import numpy as np

import pixelworld.envs.pixelworld as pw
import pixelworld.envs.pixelworld.core as core
import pixelworld.envs.pixelworld.objects as objects_mod
import pixelworld.envs.pixelworld.events as events
import pixelworld.envs.pixelworld.object_attributes as oa
from ..helpers import h, L


class GiveUpAction(core.AbilityObjectAttribute):
    """Adds the ability for the player to give up."""
    _actions = ['GIVE_UP']
    def _execute_action(self, obj, t, dt, agent_id, action):
        if action == 'GIVE_UP':
            obj.add_attribute('killed_by', 'forfeit')
            obj.alive = False

    
class HoleInteractionObjectAttribute(core.InteractsObjectAttribute):
    """Controls the interaction between the hole and the player (death) and the
    hole and the boulder (they eliminate one another)."""
    _interacts_types = ['boulder', 'sokoban_player']

    _step_after = ['pushes']

    def _interact(self, obj1, obj2):
        if isinstance(obj2, Boulder):
            obj1.visible = False
            obj2.visible = False
            event = HoleFillEvent(self.world)
        elif isinstance(obj2, SokobanPlayer):
            obj2.add_attribute('killed_by', 'falling into hole')
            obj2.alive = False
        else:
            assert False, 'unknown object type: %s' % type(obj2)


class ExitInteractionObjectAttribute(core.InteractsObjectAttribute):
    """Controls the interaction between the player and the exit - either we change
    screens to the next level or we win if there are no more levels."""
    _interacts_types = ['sokoban_player']

    _step_after = ['pushes']

    def _interact(self, obj1, obj2):
        if isinstance(obj2, SokobanPlayer):
            if self.world._chosen_screen is not None or self.world.screen == len(self.world._screens) - 1:
                event = WinEvent(self.world)
            else:
                event = LevelClearEvent(self.world, new_screen=self.world.screen + 1)

class HoleFillEvent(core.Event):
    """Event that a boulder was pushed into a hole, filling it."""
    _reward = 100


class WinEvent(core.Event):
    """Event that the player won the game"""
    _reward = 1000
    _terminates = True


class LevelClearEvent(events.ChangeScreenEvent):
    """Event that the player solved a level"""
    _reward = 1000


class Hole(objects_mod.BasicObject):
    """Holes: lethal to step in, they disappear if you push a boulder into them."""
    _attributes = ['hole_interaction']
    _defaults = {'mass': 0, 'color': 4}


class SokobanExit(objects_mod.BasicObject):
    """Exit: you win if you reach it."""
    _attributes = ['exit_interaction']
    _defaults = {'mass': 0, 'color': 5}


class Boulder(objects_mod.BasicObject):
    """Boulder: you can push them around"""
#    _attributes = ['boulder_pushable']
    _defaults = {'color': 3, 'mass': 1}


class SokobanPlayer(objects_mod.BasicSelfObject):
    """The player avatar"""
    _attributes = ['give_up_action', 'strength', 'alive']
    _defaults = {'color': 2, 'strength': 1, 'mass': 1}


class SokobanJudge(core.Judge):
    _reward_events = [{'event': 'kill', 'reward': -1000}]
    _termination_events = [{'event': 'kill'}]

judge = SokobanJudge

# courtesy of NetHack
screens = ["""
-------- ------
|X|*   ---    |
|^|- 00    0  |
|^||  00| 0 0 |
|^||    |     |
|^|------0----|
|^|    |      |
|^------      |
|  ^^^^0000   |
|  -----      |
----   --------
""",

"""
------  ----- 
|    |  |   | 
| 0  ---- 0 | 
| 0      0  | 
|  ---*---0 | 
|--------- ---
|  ^^^X|     |
|  ----|0    |
--^|   | 0   |
 |^----- 0   |
 |  ^^^^0 0  |
 |  ----------
 ----         
""",

"""
 ----          -----------
-- *--------   |         |
|          |   |         |
| 0-----0- |   |         |
|  |   | 0 |   |    X    |
| 0 0    0-|   |         |
| 0  0  |  |   |         |
| ----0 -- |   |         |
|  0   0 | --  |         |
| ---0-   0 ------------ |
|   |  0- 0 ^^^^^^^^^^^^ |
|  0      ----------------
-----  |  |               
    -------               
""",

"""
-----------       -----------
|    |    ---     |         |
|  00|00   *|     |         |
|     0   ---     |         |
|    |    |       |    X    |
|- ---------      |         |
|  0 |     |      |         |
| 00 |0 0 0|      |         |
|  0     0 |      |         |
| 000|0  0 ---------------- |
|    |  0 0 ^^^^^^^^^^^^^^^ |
-----------------------------
""",

"""
   --------          
--- |    |          
|   0    |----------
| - 00-00| |       |
| 00-      |       |
| -  0 |   |       |
|    -0--0-|   X   |
|  00  0   |       |
| --   |   |       |
|    -0|---|       |
---  0 ----------- |
  |  0*^^^^^^^^^^^ |
  ------------------
""",

"""
--------------------
|        |   |     |
| 00  -00| - |     |
|  | 0 0 |00 |     |
|- |  -  | - |  X  |
|   --       |     |
|   | 0 -   -|     |
| 0 |0 |   --|     |
|-0 |  ----------- |
|  0    ^^^^^^^^^^ |
|   | *-------------
--------            
""",

"""
--------------------------
|*      ^^^^^^^^^^^^^^^^ |
|       ---------------- |
------- ------         | |
 |           |         | |
 | 0 0 0 0 0 |         | |
-------- ----|         | |
|   0 0  0 0 |         | |
|   0        |         | |
----- --------   ------| |
 |  0 0 0   |  --|     | |
 |     0    |  |X      | |
 | 0 0   0 --  |-|     | |
------- ----   |         |
|  0     |     |-|     |--
|        |     |       |  
|   ------     --|     |  
-----            -------  
""",

"""
  ------------------------
  |  ^^^^^^^^^^^^^^^^^^  |
  |  ------------------- |
---- |    -----        | |
|  |0--  --   |        | |
|     |--| 0  |        | |
| 00  |  |  0 |        | |
--  00|   00 --        | |
 |0  0   |0  |   ------| |
 | 00 |  |  0| --|     | |
 | 0 0---| 0 | |X      | |
 |       |  -- |-|     | |
 ---- 0  | --  |         |
    --- -- |   |-|     |--
     | 0   |   |       |  
     |* |  |   --|     |  
     -------     -------  
""",
]

legend = {
    '-': 'wall',
    '|': 'wall',
    '^': 'hole',
    '0': 'boulder',
    'X': 'sokoban_exit',
    '*': 'sokoban_player',
}

world = pw.pixel_worlds.ScreenBasedPixelWorld

