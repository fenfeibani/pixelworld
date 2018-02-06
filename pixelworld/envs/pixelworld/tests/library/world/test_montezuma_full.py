import unittest

import numpy as np

from pixelworld.envs.pixelworld import universe, library
import pixelworld.envs.pixelworld.library.helpers as h
from pixelworld.envs.pixelworld.tests import test_core

pixelzuma = library.import_item('world', 'pixelzuma')
pixelzuma_full = library.import_item('world', 'pixelzuma_full')

class TestPixelzuma(test_core.TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'pixelzuma_full')
        
        super(TestPixelzuma, self).__init__(*args, **kwargs)
    
    def tearDown(self):
        self.world.end()

    def _test_walkthrough(self):
        self.world = universe.create_world('pixelzuma_full')
    
        walkthrough = """
#down_the_ladder
DOWN DOWN DOWN
#over_to_the_rope_and_down_the_ladder
RIGHT RIGHT JUMPRIGHT NOOP JUMPRIGHT NOOP RIGHT DOWN DOWN DOWN
#jump_over_the_skull
LEFT LEFT LEFT NOOP LEFT LEFT LEFT LEFT LEFT JUMPLEFT NOOP
#get_the_key
LEFT LEFT LEFT UP UP UP JUMP SCORE100 INVENTORY_PLUS_KEY NOOP DOWN DOWN DOWN
#jump_over_the_skull
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT JUMPRIGHT NOOP RIGHT RIGHT RIGHT RIGHT RIGHT
#up_the_ladder_and_up_to_the_door
UP UP UP LEFT JUMPLEFT NOOP JUMPLEFT NOOP NOOP UP UP UP UP LEFT LEFT JUMPLEFT LEFT SCORE300 INVENTORY_MINUS_KEY
#to_the_exit
LEFT LEFT LEFT LEFT LEFT NOOP

#past_the_traps
LEFT LEFT LEFT LEFT NOOP NOOP NOOP NOOP NOOP NOOP LEFT LEFT LEFT
#get_the_coin
LEFT LEFT LEFT LEFT NOOP LEFT LEFT LEFT JUMP SCORE1000 NOOP NOOP NOOP NOOP
#to_the_ladder
RIGHT RIGHT RIGHT RIGHT RIGHT
#down_the_ladder
DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN 

#down_the_ladder
DOWN DOWN
#jump_over_the_spider
LEFT LEFT LEFT JUMPLEFT NOOP LEFT LEFT LEFT LEFT

#avoid_the_skulls
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
#down_the_ladder
DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN 

#down_the_ladder
DOWN DOWN
#jump_over_the_snakes
LEFT LEFT LEFT JUMPLEFT NOOP JUMPLEFT NOOP LEFT LEFT

#jump_to_the_rope
LEFT LEFT LEFT LEFT JUMPLEFT NOOP NOOP NOOP NOOP 
#get_the_key
UP UP UP JUMP SCORE100 INVENTORY_PLUS_KEY NOOP
#down_the_rope
DOWN DOWN DOWN DOWN DOWN DOWN 
#over_to_the_things
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 
#climb_the_platforms
JUMP JUMP NOOP NOOP NOOP NOOP NOOP NOOP JUMP JUMP JUMP JUMP 
#exit
RIGHT RIGHT

#jump_over_the_snakes
RIGHT JUMPRIGHT NOOP JUMPRIGHT NOOP RIGHT RIGHT RIGHT
#up_the_ladder
UP UP UP

#up_the_ladder
UP UP UP UP UP UP UP
#avoid_the_skulls
RIGHT NOOP NOOP NOOP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#avoid_the_spider
RIGHT RIGHT RIGHT RIGHT RIGHT JUMPRIGHT NOOP RIGHT
#down_the_ladder
DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN 

#down_the_ladder
DOWN DOWN
#wait_for_the_platform
NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP 
#get_the_coin
LEFT LEFT LEFT LEFT LEFT LEFT JUMP SCORE1000 NOOP
#wait_again
RIGHT NOOP NOOP NOOP NOOP NOOP NOOP NOOP 
RIGHT RIGHT RIGHT RIGHT RIGHT 
#wait_again
NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP 
#go_right
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#avoid_the_snake
RIGHT RIGHT RIGHT RIGHT JUMPRIGHT NOOP RIGHT RIGHT
#up_the_ladder
UP UP UP

#up_the_ladder
UP
#through_the_door
RIGHT RIGHT SCORE300 INVENTORY_MINUS_KEY RIGHT RIGHT RIGHT
#up_the_rope
JUMP UP JUMPLEFT NOOP
#avoid_the_skull
LEFT JUMPLEFT NOOP LEFT LEFT LEFT JUMPLEFT
#up_the_rope
UP UP UP
#get_the_torch
JUMPRIGHT NOOP RIGHT RIGHT JUMP SCORE3000 INVENTORY_PLUS_TORCH NOOP NOOP NOOP
#back_down_the_rope
JUMPLEFT NOOP DOWN DOWN DOWN RIGHT NOOP
#avoid_the_skull
RIGHT RIGHT RIGHT NOOP JUMPRIGHT NOOP RIGHT
#back_down_the_rope
JUMPRIGHT NOOP DOWN DOWN
#back_out
LEFT LEFT LEFT LEFT LEFT DOWN DOWN

#down_the_ladder
DOWN DOWN
#avoid_the_snake
RIGHT RIGHT JUMPRIGHT NOOP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#avoid_the_traps
RIGHT RIGHT NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT NOOP NOOP NOOP NOOP NOOP 
RIGHT RIGHT RIGHT RIGHT NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT NOOP NOOP NOOP NOOP NOOP
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#avoid_the_spider
RIGHT RIGHT NOOP JUMPRIGHT NOOP RIGHT RIGHT RIGHT RIGHT
#up_the_ladder
UP UP UP

#up_the_ladder
UP UP UP UP UP UP UP
#get_the_key
RIGHT RIGHT NOOP NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT JUMP SCORE100 INVENTORY_PLUS_KEY
NOOP NOOP NOOP NOOP LEFT LEFT LEFT LEFT LEFT 
#go_left
LEFT LEFT NOOP LEFT LEFT LEFT NOOP NOOP NOOP NOOP NOOP LEFT LEFT LEFT LEFT

#get_the_sword
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
JUMP SCORE100 INVENTORY_PLUS_SWORD NOOP
#go_back
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT

#avoid_the_traps
RIGHT RIGHT RIGHT NOOP NOOP NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT RIGHT RIGHT
#down_the_ladder
DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN 

#down_the_ladder
DOWN DOWN
#avoid_the_spider
RIGHT RIGHT JUMPRIGHT NOOP RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#jump_to_the_rope
RIGHT RIGHT RIGHT JUMPRIGHT NOOP NOOP NOOP
#get_the_key
DOWN JUMPRIGHT SCORE100 INVENTORY_PLUS_KEY NOOP JUMPLEFT NOOP
#go_down
DOWN DOWN DOWN RIGHT DOWN DOWN

#go_down
DOWN DOWN
#wait_for_the_platform
NOOP NOOP NOOP
#go_right
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#get_the_coins
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 
JUMPRIGHT SCORE1000 NOOP JUMPRIGHT SCORE1000 NOOP JUMPRIGHT SCORE1000 NOOP 
#go_back
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT

#go_to_edge_and_wait
LEFT LEFT LEFT LEFT NOOP
#go_to_ladder
LEFT LEFT LEFT LEFT LEFT
#wait
NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP
#avoid_snake
LEFT LEFT LEFT LEFT JUMPLEFT NOOP LEFT LEFT LEFT

#kill_the_spider
LEFT LEFT LEFT LEFT LEFT LEFT SCORE3000 INVENTORY_MINUS_SWORD
#up_the_ladder
LEFT LEFT LEFT UP UP UP

#up_the_ladder
UP UP UP UP UP UP NOOP NOOP UP
#avoid_the_spider
LEFT LEFT LEFT LEFT LEFT NOOP JUMPLEFT NOOP LEFT LEFT

#avoid_the_traps
LEFT LEFT NOOP NOOP
NOOP NOOP NOOP NOOP 
LEFT LEFT LEFT NOOP
NOOP NOOP NOOP NOOP 
LEFT LEFT LEFT LEFT
NOOP NOOP NOOP NOOP
LEFT LEFT LEFT NOOP 
NOOP NOOP NOOP NOOP
LEFT LEFT LEFT LEFT LEFT LEFT

#jump_over_the_snake
LEFT LEFT LEFT LEFT LEFT JUMPLEFT NOOP LEFT LEFT
#down_the_ladder
DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN 

#down_the_ladder
DOWN DOWN
#get_the_amulet
LEFT LEFT LEFT LEFT LEFT LEFT JUMP SCORE200 INVENTORY_PLUS_AMULET NOOP
#go_right
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 
RIGHT RIGHT RIGHT  

#get_the_coins
NOOP JUMPRIGHT NOOP NOOP JUMPRIGHT NOOP JUMPRIGHT SCORE1000 NOOP JUMPRIGHT NOOP JUMPRIGHT NOOP
RIGHT RIGHT JUMP SCORE1000 NOOP LEFT LEFT 
#go_back
JUMPLEFT NOOP JUMPLEFT NOOP JUMPLEFT NOOP JUMPLEFT NOOP JUMPLEFT NOOP LEFT LEFT LEFT LEFT

#cross_the_room
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 

#go_to_edge
LEFT LEFT LEFT LEFT
#wait
NOOP NOOP NOOP NOOP NOOP NOOP 
NOOP NOOP NOOP NOOP NOOP NOOP 
#cross_the_pit
JUMPLEFT NOOP JUMPLEFT NOOP JUMPLEFT NOOP JUMPLEFT INVENTORY_MINUS_AMULET NOOP JUMPLEFT NOOP
LEFT LEFT LEFT LEFT

#open_the_doors
LEFT LEFT LEFT SCORE300 INVENTORY_MINUS_KEY LEFT
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT SCORE300 INVENTORY_MINUS_KEY LEFT LEFT LEFT LEFT LEFT LEFT 

#cross_the_room
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 

#get_the_coin
LEFT LEFT LEFT LEFT LEFT LEFT LEFT
LEFT LEFT LEFT LEFT LEFT LEFT LEFT
JUMP SCORE1000 NOOP
#get_the_win
LEFT LEFT SCORE10000 NOOP
""".split()

        foo = """
""".split()
        self.check_walkthrough(walkthrough)

        self.assertTrue(not any(isinstance(x, pixelzuma.DeathEvent) for x in self.world.events))

    def _test_walkthrough_of_death(self):
        walkthrough = """
#walk_off_the_platform
LEFT LEFT LEFT
NOOP NOOP NOOP NOOP NOOP NOOP SCORE-1000 NOOP
#down_the_ladder
DOWN DOWN DOWN
#fall_off_conveyor
NOOP NOOP NOOP NOOP NOOP NOOP SCORE-1000 NOOP
#down_the_ladder
DOWN DOWN DOWN
#over_to_the_rope_and_down_the_ladder
RIGHT RIGHT JUMPRIGHT NOOP JUMPRIGHT NOOP RIGHT DOWN DOWN DOWN
#move_into_skull
LEFT LEFT LEFT LEFT LEFT SCORE-1000 NOOP
#down_the_ladder
DOWN DOWN DOWN
#over_to_the_rope_and_down_the_ladder
RIGHT RIGHT JUMPRIGHT NOOP JUMPRIGHT NOOP RIGHT DOWN DOWN DOWN
#move_left
LEFT LEFT LEFT LEFT LEFT 
LEFT LEFT LEFT LEFT 
LEFT LEFT LEFT LEFT 
#get_the_key
UP UP UP JUMP SCORE100 INVENTORY_PLUS_KEY NOOP DOWN DOWN DOWN
#move_right
RIGHT RIGHT RIGHT RIGHT RIGHT 
RIGHT RIGHT RIGHT RIGHT 
RIGHT RIGHT RIGHT RIGHT 
#up_the_ladder_and_up_to_the_door
UP UP UP LEFT JUMPLEFT NOOP JUMPLEFT NOOP NOOP UP UP UP UP LEFT LEFT JUMPLEFT LEFT SCORE300 INVENTORY_MINUS_KEY
#to_the_exit
LEFT LEFT LEFT LEFT LEFT NOOP

#move_into_trap
LEFT LEFT SCORE-1000 LEFT
#past_the_traps
LEFT LEFT NOOP NOOP NOOP NOOP NOOP NOOP NOOP LEFT LEFT
NOOP NOOP NOOP NOOP NOOP NOOP 
LEFT LEFT LEFT LEFT LEFT
#down_the_ladder
DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN 

#down_the_ladder
DOWN DOWN
#run_into_spider
LEFT LEFT SCORE-1000 LEFT
#down_the_ladder
DOWN DOWN
#left
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 

#run_into_skull
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT SCORE-1000 LEFT 
#run_into_skull
LEFT LEFT LEFT LEFT LEFT SCORE-1000 NOOP
#left
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
#down_ladder
DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN 

#down_ladder
DOWN DOWN
#run_into_snake
LEFT LEFT LEFT SCORE-1000 LEFT
#down_ladder
DOWN DOWN
#run_into_snake
LEFT LEFT LEFT LEFT LEFT SCORE-1000 LEFT
#down_ladder
DOWN DOWN
#left
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 

#walk_off_platform
LEFT LEFT LEFT LEFT LEFT 
NOOP NOOP NOOP NOOP NOOP NOOP SCORE-1000 NOOP
#jump_to_the_rope
LEFT LEFT LEFT LEFT JUMPLEFT NOOP NOOP NOOP NOOP 
#get_the_key
UP UP UP JUMP SCORE100 INVENTORY_PLUS_KEY NOOP
#down_the_rope
DOWN DOWN DOWN DOWN DOWN DOWN 
#over_to_the_things
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 
#climb_the_platforms
JUMP JUMP NOOP NOOP NOOP NOOP NOOP NOOP JUMP JUMP JUMP JUMP 
#exit
RIGHT RIGHT

#right
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT
#up_the_ladder
UP UP UP

#up_the_ladder
UP UP UP UP UP UP UP
#right
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#right
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT
#down_the_ladder
DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN 

#jump_into_lava
DOWN DOWN RIGHT
NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP SCORE-1000 NOOP 
#avoid_the_lava
DOWN DOWN
NOOP NOOP NOOP NOOP NOOP NOOP NOOP 
NOOP NOOP NOOP NOOP NOOP NOOP NOOP 
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#run_into_snake
RIGHT RIGHT RIGHT RIGHT SCORE-1000 RIGHT
#run_into_snake
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT SCORE-1000 RIGHT
#right
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 
#up_the_ladder
UP UP UP

#up_the_ladder
UP
#through_the_door
RIGHT RIGHT SCORE300 INVENTORY_MINUS_KEY RIGHT RIGHT RIGHT
#up_the_rope
JUMP UP JUMPLEFT NOOP
#run_into_skull
LEFT LEFT LEFT SCORE-1000 LEFT
#up_the_ladder
UP
#through_the_door
RIGHT RIGHT RIGHT RIGHT RIGHT
#up_the_rope
JUMP UP JUMPLEFT NOOP
#left
LEFT LEFT LEFT LEFT LEFT LEFT 
#up_the_rope
JUMPLEFT UP UP UP
#walk_off_the_platform
RIGHT NOOP NOOP NOOP NOOP SCORE-1000 NOOP 
#screw_the_torch
DOWN

#down_the_ladder
DOWN DOWN
#right
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#run_into_trap
RIGHT RIGHT SCORE-1000 RIGHT
#avoid_the_traps
RIGHT RIGHT NOOP RIGHT RIGHT RIGHT NOOP NOOP NOOP NOOP NOOP 
RIGHT RIGHT RIGHT RIGHT NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT NOOP NOOP NOOP NOOP NOOP
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#run_into_spider
RIGHT SCORE-1000 NOOP
#avoid_the_spider
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT
#up_the_ladder
UP UP UP

#up_the_ladder
UP UP UP UP UP UP UP
#get_the_key
RIGHT RIGHT NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT JUMP SCORE100 INVENTORY_PLUS_KEY NOOP
#walk_into_trap
SCORE-1000 LEFT
#up_the_ladder
UP UP UP UP UP UP UP
#avoid_the_traps
LEFT LEFT NOOP LEFT LEFT LEFT NOOP NOOP NOOP NOOP NOOP LEFT LEFT LEFT LEFT

#left
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
#up_the_ladder
UP UP UP

#up_the_ladder_and_into_the_skull
UP UP UP UP UP UP UP SCORE-1000 NOOP
#up_the_ladder
UP UP UP UP UP UP UP 
#wait_for_death
NOOP NOOP NOOP NOOP SCORE-1000 NOOP
#back_down
DOWN

#down_the_ladder
DOWN DOWN
#right
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#avoid_the_traps
RIGHT NOOP NOOP NOOP NOOP NOOP NOOP RIGHT RIGHT NOOP NOOP
NOOP NOOP NOOP NOOP RIGHT RIGHT RIGHT RIGHT RIGHT
#down_the_ladder
DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN 

#down_the_ladder
DOWN DOWN
#right
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#walk_off_platform
RIGHT RIGHT RIGHT RIGHT 
NOOP NOOP NOOP NOOP NOOP NOOP SCORE-1000 NOOP
#jump_to_the_rope
RIGHT RIGHT RIGHT JUMPRIGHT NOOP NOOP NOOP
#get_the_key
DOWN JUMPRIGHT SCORE100 INVENTORY_PLUS_KEY NOOP JUMPLEFT NOOP
#go_down
DOWN DOWN DOWN RIGHT DOWN DOWN

#jump_into_lava
DOWN DOWN RIGHT NOOP NOOP NOOP NOOP NOOP NOOP SCORE-1000 NOOP 
#down_the_ladder
DOWN DOWN
#run_into_snake
LEFT LEFT LEFT LEFT SCORE-1000 LEFT
#down_the_ladder
DOWN DOWN
#wait_for_platform
NOOP NOOP NOOP NOOP NOOP NOOP NOOP 
#left
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 

#run_into_spider
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT SCORE-1000 NOOP
#left
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
#up_the_ladder
UP UP UP

#up_the_ladder
UP UP UP UP UP UP UP 
#left
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 

#avoid_the_traps
LEFT LEFT
NOOP NOOP NOOP NOOP 
LEFT LEFT LEFT NOOP
NOOP NOOP NOOP NOOP 
LEFT LEFT LEFT LEFT
NOOP NOOP NOOP NOOP
LEFT LEFT LEFT NOOP 
NOOP NOOP NOOP NOOP
LEFT LEFT LEFT LEFT LEFT LEFT

#left
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
#down_the_ladder
DOWN DOWN DOWN DOWN DOWN DOWN DOWN DOWN 

#down_the_ladder
DOWN DOWN
#right
RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT RIGHT 

#jump_into_lava
RIGHT RIGHT RIGHT RIGHT NOOP NOOP NOOP NOOP NOOP NOOP NOOP NOOP SCORE-1000 NOOP LEFT

#left
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 

#jump_into_lava
LEFT LEFT LEFT LEFT LEFT
NOOP NOOP NOOP NOOP NOOP NOOP NOOP SCORE-1000 NOOP 
#run_into_skull
LEFT LEFT LEFT LEFT
LEFT LEFT SCORE-1000 LEFT
#jump_over_lava_pit
LEFT LEFT LEFT LEFT

NOOP NOOP NOOP
JUMPLEFT NOOP JUMPLEFT NOOP JUMPLEFT NOOP JUMPLEFT NOOP JUMPLEFT NOOP
LEFT LEFT LEFT LEFT

#open_the_doors
LEFT LEFT LEFT SCORE300 INVENTORY_MINUS_KEY LEFT
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT SCORE300 INVENTORY_MINUS_KEY LEFT LEFT LEFT LEFT LEFT LEFT 

#cross_the_room
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 
LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT LEFT 

#get_the_coin
LEFT LEFT LEFT LEFT LEFT LEFT LEFT
LEFT LEFT LEFT LEFT LEFT LEFT LEFT
#get_the_win
LEFT LEFT SCORE10000 NOOP

""".split()
        self.world = universe.create_world('pixelzuma_full')
        self.check_walkthrough(walkthrough)    

    def check_walkthrough(self, walkthrough):
        expected_score = score = 0
        expected_inventory = []

        # a walkthrough item is either a comment, a score change, an inventory
        # change, or an action
        for item in walkthrough:
            if item.startswith('#'):
                continue
            elif item.startswith('SCORE'):
                expected_score += int(item.replace('SCORE', ''))
            elif item.startswith('INVENTORY_PLUS_'):
                expected_inventory.append(item.replace('INVENTORY_PLUS_', ''))
            elif item.startswith('INVENTORY_MINUS_'):
                expected_inventory.remove(item.replace('INVENTORY_MINUS_', ''))
            else:
                # check that action is legal
                assert item in self.world.actions
                obs, reward, done, info = self.world.step(item)
                score += reward

                # check that total score is what we expect
                self.assertEqual(score, expected_score, '%s %s %s' % (item, score, expected_score))

                # check that inventory is what we expect
                inventory = self.world.objects['pedro'].inventory
                self.assertItemsEqual(inventory, expected_inventory, repr(inventory) + repr(expected_inventory))

                #self.world.render()

        self.assertTrue(done)


    def test_amulet_usable(self):
        # create the objects we want
        screen = """
------------------
-                -
X   N*           X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
"""
        objects, width, height = h.world.screen(screen, pixelzuma.legend)
        self.world = universe.create_world('pixelzuma', objects=objects, width=width, height=height)

        # give pedro the amulet
        self.world.objects['pedro'].inventory = ['AMULET']

        # check that the amulet protects us so we don't get a penalty
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)

        # check that we didn't die
        self.assertFalse(any(isinstance(event, pixelzuma.DeathEvent) for event in self.world.events))

    def test_key_usable(self):
        # create the objects we want
        screen = """
------------------
-                -
X  D *           X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
"""
        objects, width, height = h.world.screen(screen, pixelzuma.legend)
        self.world = universe.create_world('pixelzuma', objects=objects, width=width, height=height)

        # give pedro a key
        self.world.objects['pedro'].inventory = ['KEY']
        
        # check that we get a reward and that we can move through the door
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 300)
        self.assertFalse(done)
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)


    def test_sword_usable(self):
        # create the objects we want
        screen = """
------------------
-                -
X   S*           X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
"""
        objects, width, height = h.world.screen(screen, pixelzuma.legend)
        self.world = universe.create_world('pixelzuma', objects=objects, width=width, height=height)

        # give pedro the sword
        self.world.objects['pedro'].inventory = ['SWORD']

        # check that we get the rewards we expect and episode doesn't end
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 3000)
        self.assertFalse(done)
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        obs, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)

        # check that we didn't die
        self.assertFalse(any(isinstance(event, pixelzuma.DeathEvent) for event in self.world.events))

        # check that we killed the spider
        kill_events = [event for event in self.world.events if isinstance(event, pixelzuma.KillEvent)]
        self.assertEqual(len(kill_events), 1)
        self.assertEqual(kill_events[0].type, 'SPIDER')

        # check that we no longer have the sword
        self.assertEqual(self.world.objects['pedro'].inventory, [])

    def test_collectible_items(self):
        # there's a slight weirdness here, which is that collection is delayed
        # one turn from when one might expect it

        # create the objects we want
        screen = """
------------------
-                -
X WK!^?$*        X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
"""
        objects, width, height = h.world.screen(screen, pixelzuma.legend)
        self.world = universe.create_world('pixelzuma', objects=objects, width=width, height=height)

        # get the coin and check that reward and inventory is what we expect
        obj, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        obj, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 1000)
        self.assertFalse(done)
        self.assertItemsEqual(self.world.objects['pedro'].inventory, [])

        # get the amulet and check that reward and inventory is what we expect
        obj, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        obj, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 200)
        self.assertFalse(done)
        self.assertItemsEqual(self.world.objects['pedro'].inventory, ['AMULET'])

        # get the torch and check that reward and inventory is what we expect
        obj, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        obj, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 3000)
        self.assertFalse(done)
        self.assertItemsEqual(self.world.objects['pedro'].inventory, ['AMULET', 'TORCH'])

        # get the sword and check that reward and inventory is what we expect
        obj, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        obj, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 100)
        self.assertFalse(done)
        self.assertItemsEqual(self.world.objects['pedro'].inventory, ['AMULET', 'TORCH', 'SWORD'])

        # get the key and check that reward and inventory is what we expect
        obj, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        obj, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 100)
        self.assertFalse(done)
        self.assertItemsEqual(self.world.objects['pedro'].inventory, ['AMULET', 'TORCH', 'SWORD', 'KEY'])

        # get the win and check that reward and inventory is what we expect
        obj, reward, done, info = self.world.step('LEFT')
        self.assertEqual(reward, 0)
        self.assertFalse(done)
        obj, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 10000)
        self.assertTrue(done)
        self.assertItemsEqual(self.world.objects['pedro'].inventory, ['AMULET', 'TORCH', 'SWORD', 'KEY'])

    def test_trap_behavior(self):
        # create the objects we want
        screen = """
------------------
-                -
X        T     * X
-----------_------
-                -
-                -
-                -
-                -
-                -
------------------
"""
        objects, width, height = h.world.screen(screen, pixelzuma.legend)
        self.world = universe.create_world('pixelzuma', objects=objects, width=width, height=height)

        trap = self.world.objects['trap']
        platform = self.world.objects['flicker_platform_blocking']

        # check that stuff is there
        self.assertTrue(trap.visible)
        self.assertTrue(trap.pixelzuma_deadly)
        self.assertTrue(platform.visible)
        self.assertTrue(platform.support)
        
        # t = 0
        self.world.step('NOOP')

        # check that stuff isn't there
        self.assertTrue(not trap.visible)
        self.assertTrue(not trap.pixelzuma_deadly)
        self.assertTrue(not platform.visible)
        self.assertTrue(not platform.support)

        # t = 1
        self.world.step('NOOP')

        # check that stuff isn't there
        self.assertTrue(not trap.visible)
        self.assertTrue(not trap.pixelzuma_deadly)
        self.assertTrue(not platform.visible)
        self.assertTrue(not platform.support)

        # t = 2
        self.world.step('NOOP')

        # check that stuff isn't there
        self.assertTrue(not trap.visible)
        self.assertTrue(not trap.pixelzuma_deadly)
        self.assertTrue(not platform.visible)
        self.assertTrue(not platform.support)

        # t = 3
        self.world.step('NOOP')

        # check that stuff isn't there
        self.assertTrue(not trap.visible)
        self.assertTrue(not trap.pixelzuma_deadly)
        self.assertTrue(not platform.visible)
        self.assertTrue(not platform.support)

        # t = 4
        self.world.step('NOOP')

        # check that trap is there
        self.assertTrue(trap.visible)
        self.assertTrue(trap.pixelzuma_deadly)
        # check that platform isn't there
        self.assertTrue(not platform.visible)
        self.assertTrue(not platform.support)

        # t = 5
        self.world.step('NOOP')

        # check that trap is there
        self.assertTrue(trap.visible)
        self.assertTrue(trap.pixelzuma_deadly)
        # check that platform isn't there
        self.assertTrue(not platform.visible)
        self.assertTrue(not platform.support)

        # t = 6
        self.world.step('NOOP')

        # check that trap is there
        self.assertTrue(trap.visible)
        self.assertTrue(trap.pixelzuma_deadly)
        # check that platform isn't there
        self.assertTrue(not platform.visible)
        self.assertTrue(not platform.support)

        # t = 7
        self.world.step('NOOP')

        # check that trap is there
        self.assertTrue(trap.visible)
        self.assertTrue(trap.pixelzuma_deadly)
        # check that platform isn't there
        self.assertTrue(not platform.visible)
        self.assertTrue(not platform.support)

        # t = 8
        self.world.step('NOOP')

        # check that trap isn't there
        self.assertTrue(not trap.visible)
        self.assertTrue(not trap.pixelzuma_deadly)
        # check that platform is there
        self.assertTrue(platform.visible)
        self.assertTrue(platform.support)

        # t = 9
        self.world.step('NOOP')

        # check that trap isn't there
        self.assertTrue(not trap.visible)
        self.assertTrue(not trap.pixelzuma_deadly)
        # check that platform is there
        self.assertTrue(platform.visible)
        self.assertTrue(platform.support)

        # t = 10
        self.world.step('NOOP')

        # check that trap isn't there
        self.assertTrue(not trap.visible)
        self.assertTrue(not trap.pixelzuma_deadly)
        # check that platform is there
        self.assertTrue(platform.visible)
        self.assertTrue(platform.support)

        # t = 11
        self.world.step('NOOP')

        # check that trap isn't there
        self.assertTrue(not trap.visible)
        self.assertTrue(not trap.pixelzuma_deadly)
        # check that platform is there
        self.assertTrue(platform.visible)
        self.assertTrue(platform.support)

        # t = 12
        self.world.step('NOOP')

        # check that stuff is there
        self.assertTrue(trap.visible)
        self.assertTrue(trap.pixelzuma_deadly)
        self.assertTrue(platform.visible)
        self.assertTrue(platform.support)

        # t = 13
        self.world.step('NOOP')

        # check that stuff is there
        self.assertTrue(trap.visible)
        self.assertTrue(trap.pixelzuma_deadly)
        self.assertTrue(platform.visible)
        self.assertTrue(platform.support)

        # t = 14
        self.world.step('NOOP')

        # check that stuff is there
        self.assertTrue(trap.visible)
        self.assertTrue(trap.pixelzuma_deadly)
        self.assertTrue(platform.visible)
        self.assertTrue(platform.support)

        # t = 15
        self.world.step('NOOP')

        # check that stuff is there
        self.assertTrue(trap.visible)
        self.assertTrue(trap.pixelzuma_deadly)
        self.assertTrue(platform.visible)
        self.assertTrue(platform.support)

        # t = 16
        self.world.step('NOOP')

        # check that stuff isn't there
        self.assertTrue(not trap.visible)
        self.assertTrue(not trap.pixelzuma_deadly)
        self.assertTrue(not platform.visible)
        self.assertTrue(not platform.support)

    def test_skull_behavior(self):
        # create the objects we want
        screen = """
------------------
-                -
X      @       * X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
"""
        objects, width, height = h.world.screen(screen, pixelzuma.legend)
        self.world = universe.create_world('pixelzuma', objects=objects, width=width, height=height)

        # remember skull's initial position
        skull = self.world.objects['skull']
        anchor = skull.position

        # check skull's movement
        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, 1)))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, 2)))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, 3)))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, 2)))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, 1)))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, 0)))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, -1)))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, -2)))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, -3)))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, -2)))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, -1)))

        self.world.step('NOOP')
        self.assertTrue(np.array_equal(skull.position, anchor + (0, 0)))

    def test_spider_behavior(self):
        # create the objects we want
        screen = """
--------*---------
-       H        -
X      SH        X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
"""
        objects, width, height = h.world.screen(screen, pixelzuma.legend)
        self.world = universe.create_world('pixelzuma', objects=objects, width=width, height=height)

        # remember spider's original position
        spider = self.world.objects['spider']
        anchor = spider.position

        # check spider's movement
        for i in xrange(1, 10):
            self.world.step('NOOP')
            self.assertTrue(np.array_equal(spider.position, anchor + (0, i)))

        for i in xrange(8, -7, -1):
            self.world.step('NOOP')
            self.assertTrue(np.array_equal(spider.position, anchor + (0, i)))

        for i in xrange(-5, 1):
            self.world.step('NOOP')
            self.assertTrue(np.array_equal(spider.position, anchor + (0, i)))

    def test_respawn(self):
        # create the objects we want
        screen = """
------------------
-                -
X     S  *       X
------------------
-                -
-                -
-                -
-                -
-                -
------------------
"""
        objects, width, height = h.world.screen(screen, pixelzuma.legend)
        self.world = universe.create_world('pixelzuma', objects=objects, width=width, height=height)
        
        pedro = self.world.objects['pedro']
        posn = pedro.position

        self.world.step('LEFT')
        self.world.step('LEFT')
        self.world.step('NOOP')

        # check that we respawn in our original position
        self.assertTrue(np.array_equal(pedro.position, posn))
        

if __name__ == '__main__':
    unittest.main()
