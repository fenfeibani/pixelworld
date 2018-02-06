"""
Pixelzuma PixelWorld environment
"""
from ..helpers import h, L

from pixelworld.envs.pixelworld import core, randomizers, object_attributes, library

pixelzuma = library.import_item('world', 'pixelzuma')

#------------------------------------------------------------------------------
# Pixelzuma World
#------------------------------------------------------------------------------

screen_map = """
   ABC
  DEFGH
 IJKLMNO
PQRSTUVWX
"""

# Screen layout, see legend
screens = {
    'A':"""
------------------
- T$TT     TT T  -
- T TT     TT T  X
--------H---------
-       H        -
-       H        -
-       H        -
-       H        -
-       H        -
--------H---------
""",

    'B':"""
------------------
-  -          -  -
X  D    *     D  X
----- --H-- ------
- K     H   |    -
-       H   |    -
--H-  <<<<< | -H--
- H            H -
- H     @      H -
------------------
""",

    'C':"""
------------------
-                -
X         B B    -
--------H---------
-       H        -
-       H        -
-       H        -
-       H        -
-       H        -
--------H---------
""",

    'D':"""
------------------
-                -
-         B B    X
--------H---------
-       H        -
-       H        -
-       H        -
-       H        -
-       H        -
--------H---------
""",

    'E':"""
--------H---------
-       H        -
X       HS       X
--------H---------
-       H        -
-       H        -
-       H        -
-       H        -
-       H        -
--------H---------
""",

    'F':"""
------------------
- -     ^      - -
X -            - X
----H <<<< -------
-   |        |   -
-   |        |   -
-       @    |   -
-    ------- |   -
-    D  H  D     -
--------H---------
""",
           
    'G':"""
--------H---------
-!      H        -
X       H        X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
""",

    'H':"""
------------------
- T TT     TTKT  -
X T TT     TT T  -
--------H---------
-       H        -
-       H        -
-       H        -
-       H        -
-       H        -
--------H---------
""",

    'I':"""
------------------
-       K        -
-                X
----   -H-   ---H-
-~      |       ~-
-~      |       ~-
-~      |       ~-
-H      |       H-
-~              ~-
------------------
""",

    'J':"""
--------H---------
-       H        -
X N N   H        -
------------------
-                -
-                -
-                -
-                -
-                -
------------------
""",

    'K':"""
--------H---------
- $     H        -
-       H        X
----_________-----
-   LLLLLLLLL    -
-   LLLLLLLLL    -
-     LLLLL      -
-     LLLLL      -
-      LLL       -
------------------
""",

    'L':"""
--------H---------
-       H        -
X    N  H  N     X
--------H---------
-       H        -
-       H        -
-       H        -
-       H        -
-       H        -
--------H---------
""",

    'M':"""
------------------
-  TT TT  TT TT  -
X  TT TT  TT TT  X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
""",
           
    'N':"""
--------H---------
-       H        -
X       HS       X
--------H---------
-       H        -
-       H        -
-       H        -
-       H        -
-       H        -
--------H---------
""",

    'O':"""
------------------
-                -
X                X
----   H-H   -----
-      |K|       -
-      | |       -
-      |         -
-      |         -
-                -
--------H---------
""",

    'P':"""
------------------
-  $             -
-W               X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
""",

    'Q':"""
------------------
-                -
X                X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
""",

    'R':"""
------------------
-   -        -   -
X   D        D   X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
""",

    'S':"""
------------------
-                -
X      @         X
----_________-----
-   LLLLLLLLL    -
-   LLLLLLLLL    -
-     LLLLL      -
-     LLLLL      -
-      LLL       -
------------------
""",

    'T':"""
--------H---------
- ?     H        -
X       H        X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
""",

    'U':"""
------------------
-       $      $ -
X                -
----_________-----
-   LLLLLLLLL    -
-   LLLLLLLLL    -
-     LLLLL      -
-     LLLLL      -
-      LLL       -
------------------
""",

    'V':"""
--------H---------
-       H        -
-       HS       X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
""",

    'W':"""
--------H---------
-       H        -
X  N    H        X
----_________-----
-   LLLLLLLLL    -
-   LLLLLLLLL    -
-     LLLLL      -
-     LLLLL      -
-      LLL       -
------------------
""",

    'X':"""
------------------
-          $ $ $ -
X                -
------------------
-                -
-                -
-                -
-                -
-                -
------------------
""",

}           

class ScreenSwitchingObjectAttribute(core.ListeningObjectAttribute):
    _name = 'screen_switching'
    _selected_events = ['leave_screen']

    _step_after = ['pedro']

    def _process_event_object(self, evt, obj, t, dt, agent_id, action):
        obj_list = self.world.objects.find(pedro=True)
        assert len(obj_list) == 1, 'there can be only one pedro'
        pedro = obj_list[0]
        if pedro.id not in evt.indices:
            return

        pedro_position = pedro.position

        for object in self.world.objects:
            if hasattr(object, 'position'):
                if pedro_position[1] >= self.world.width:
                    object.position -= (0, self.world.width)
                elif pedro_position[1] <= 0:
                    object.position += (0, self.world.width)
                elif pedro_position[0] >= self.world.height:
                    object.position -= (self.world.height, 0)
                elif pedro_position[0] <= 0:
                    object.position += (self.world.height, 0)
                else:
                    print 'shenanigans'

            if hasattr(object, 'anchor'):
                if pedro_position[1] >= self.world.width:
                    object.anchor -= (0, self.world.width)
                elif pedro_position[1] <= 0:
                    object.anchor += (0, self.world.width)
                elif pedro_position[0] >= self.world.height:
                    object.anchor -= (self.world.height, 0)
                elif pedro_position[0] <= 0:
                    object.anchor += (self.world.height, 0)
                else:
                    print 'shenanigans'

        # move pedro's anchor to his new position, so that we can't make
        # progress by dying
        pedro.anchor = pedro.position

class ScreenSwitcherObject(core.Object):
    _attributes = ['screen_switching']
    _defaults = {'screen_switching': True}

def multi_screen(screens, screen_map, legend, initial):
    things, positions = h.world.layout(screen_map)
    all_objects = []
    height = width = None
    
    initial_found = False
    for k, posn in zip(things, positions):
        if k == initial:
            initial_found = True
            center_posn = posn
            break
    assert initial_found, 'unrecognized initial screen'

    for k, posn in zip(things, positions):
        if k not in screens: 
            continue
        screen_objects, screen_height, screen_width = h.world.screen(screens[k], legend)

        assert height is None or height == screen_height, "All screens must be same shape"
        assert width is None or width == screen_width, "All screens must be same shape"
        height = screen_height
        width = screen_width

        for obj in screen_objects:
            r, c = obj[1]['position']
            r += height * (posn[0] - center_posn[0])
            c += width * (posn[1] - center_posn[1])
            obj[1]['position'] = (r, c)

        all_objects.extend(screen_objects)
    return all_objects, height, width

class PixelzumaWorld(core.PixelWorld):
    def __init__(self, objects=None, agent=None, judge=None, rate=3, **kwargs):
        objects, height, width = multi_screen(screens, screen_map, pixelzuma.legend, 'B')
        if judge is None:
            judge = pixelzuma.PixelzumaJudge

        if agent is None:
            agent = ['human', {'rate': rate}]

        objects.append('screen_switcher')

        if 'height' in kwargs:
            kwargs.pop('height')
        if 'width' in kwargs:
            kwargs.pop('width')

        super(PixelzumaWorld, self).__init__(objects=objects, judge=judge, height=height, width=width,
                                             agent=agent, **kwargs)

world = PixelzumaWorld

agent = ['human', {'rate': 3}]
