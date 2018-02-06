import unittest

import numpy as np

from pixelworld.envs.pixelworld import agents, events, universe, library
from pixelworld.envs.pixelworld.tests import test_core
import pixelworld.envs.pixelworld.library.helpers as h

sip = library.import_item('world', 'aliens_pixels')

class TestAliensPixels(test_core.TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'aliens_pixels')
        
        super(TestAliensPixels, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()

    def test_barricades_can_be_destroyed(self):
        # create the objects we want
        objects, _, _ = h.world.screen("""
B
P
""", sip.legend)

        self.world.remove_objects(self.world.objects)
        self.world.create_objects(objects)

        # get objects
        player = self.world.objects['pixel_avatar']
        barricade = self.world.objects['barricade']

        # shoot
        self.world.step('SHOOT')

        # check that barricade was destroyed
        self.assertTrue(not barricade.alive)
        self.assertTrue(not barricade.visible)

    def test_barricades_kill_aliens(self):
        # create the objects we want
        objects, _, _ = h.world.screen("""
AB 
  P
""", sip.legend)

        self.world.remove_objects(self.world.objects)
        self.world.create_objects(objects)

        # get objects
        player = self.world.objects['pixel_avatar']
        alien = self.world.objects['pixel_invader']
        barricade = self.world.objects['barricade']

        # step
        self.world.step('NOOP')

        # check that alien was destroyed
        self.assertTrue(not alien.alive)
        self.assertTrue(not alien.visible)

        # check that barricade was destroyed
        self.assertTrue(not barricade.alive)
        self.assertTrue(not barricade.visible)

    def test_bullets_kill_aliens(self):
        # create the objects we want
        objects, _, _ = h.world.screen("""
A 
 P
""", sip.legend)

        self.world.remove_objects(self.world.objects)
        self.world.create_objects(objects)

        # get objects
        player = self.world.objects['pixel_avatar']
        alien = self.world.objects['pixel_invader']

        # step
        obs, reward, done, info = self.world.step('SHOOT')

        # -1 for the bullet, 100 for the alien
        self.assertEqual(reward, 99)
        # all aliens dead
        self.assertTrue(done)

        # check that alien was destroyed
        self.assertTrue(not alien.alive)
        self.assertTrue(not alien.visible)

    def test_aliens_cannot_kill_aliens(self):
        # create the objects we want
        objects, _, _ = h.world.screen("""
 a
A 
 P
""", sip.legend)

        self.world.remove_objects(self.world.objects)
        self.world.create_objects(objects)

        # get objects
        player = self.world.objects['pixel_avatar']
        alien = self.world.objects['pixel_invader']
        bullet = self.world.objects['alien_bullet']

        bullet.velocity = (1, 0)

        # step
        obs, reward, done, info = self.world.step('NOOP')

        # check that bullet hit alien
        self.assertTrue(np.array_equal(bullet.position, alien.position))

        # check reward and termination
        self.assertEqual(reward, 0)
        self.assertFalse(done)

        # check that alien was not destroyed
        self.assertTrue(alien.alive)
        self.assertTrue(alien.visible)

    def test_bullets_kill_player(self):
        # create the objects we want
        objects, _, _ = h.world.screen("""
A
b 
P
""", sip.legend)

        self.world.remove_objects(self.world.objects)
        self.world.create_objects(objects)

        # get objects
        player = self.world.objects['pixel_avatar']
        alien = self.world.objects['pixel_invader']
        bullet = self.world.objects['bullet']

        bullet.velocity = (1, 0)
        bullet.bullet_shooter = alien

        # step
        obs, reward, done, info = self.world.step('NOOP')

        # check that bullet hit player
        self.assertTrue(np.array_equal(player.position, bullet.position))

        # check reward and termination
        self.assertEqual(reward, -1000)
        self.assertTrue(done)

        # check that player was destroyed
        self.assertTrue(not player.alive)
        self.assertTrue(not player.visible)

    def test_aliens_kill_player(self):
        # create the objects we want
        objects, _, _ = h.world.screen("""
AP
""", sip.legend)

        self.world.remove_objects(self.world.objects)
        self.world.create_objects(objects)

        # get objects
        player = self.world.objects['pixel_avatar']
        alien = self.world.objects['pixel_invader']

        # step
        obs, reward, done, info = self.world.step('NOOP')

        # check that alien hit player
        self.assertTrue(np.array_equal(player.position, alien.position))

        # check reward and termination
        self.assertEqual(reward, -900) # -1000 for dying, +100 for killing alien
        self.assertTrue(done)

        # check that player was destroyed
        self.assertTrue(not player.alive)
        self.assertTrue(not player.visible)

        # check that alien was destroyed
        self.assertTrue(not alien.alive)
        self.assertTrue(not alien.visible)

    def test_noop(self):
        self.world.seed(0)

        player = self.world.objects['pixel_avatar']
        
        # do NOOP forever
        done = False
        while not done:
            obs, reward, done, info = self.world.step('NOOP')
            # self.world.render()            

        # check that player was destroyed
        self.assertTrue(not player.alive)
        self.assertTrue(not player.visible)
    
        for event in self.world.events:
            if isinstance(event, events.KillEvent):
                if isinstance(self.world.objects[event.victim], sip.PixelInvaderObject):
                    self.assertTrue(event.reason in ['barricade', 'pixel_avatar'], event.reason)

    def test_actions(self):
        # create the objects we want
        objects, _, _ = h.world.screen("""
  
 P
""", sip.legend)

        self.world.remove_objects(self.world.objects)
        self.world.create_objects(objects)

        player = self.world.objects['pixel_avatar']
        posn = player.position

        self.world.step('RIGHT')
        self.assertTrue(np.array_equal(player.position, posn + (0, 1)))

        self.world.step('UP')
        self.assertTrue(np.array_equal(player.position, posn + (-1, 1)))

        self.world.step('LEFT')
        self.assertTrue(np.array_equal(player.position, posn + (-1, 0)))

        self.world.step('DOWN')
        self.assertTrue(np.array_equal(player.position, posn))

    def test_energy(self):
        self.world.agent = agents.RandomAgent(self.world)
        energy = self.world.energy
        for _ in xrange(100):
            self.world.step()
            self.assertEqual(self.world.energy, energy)

if __name__ == '__main__':
    unittest.main()
