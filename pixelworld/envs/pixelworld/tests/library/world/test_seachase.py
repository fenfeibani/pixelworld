import unittest

import numpy as np

from pixelworld.envs.pixelworld import universe, events, objects as objects_mod, agents
from pixelworld.envs.pixelworld.tests import test_core
import pixelworld.envs.pixelworld.library.helpers as h

legend = {'*': 'seachase_player',
          'F': 'fish',
          'S': 'submarine',
          's': 'surface_submarine',
          'B': 'bullet',
          'D': 'diver',
          }


class TestSeachase(test_core.TestCase):
    world = None
    
    def __init__(self, *args, **kwargs):
        self.world = kwargs.pop('world', 'seachase')
        
        super(TestSeachase, self).__init__(*args, **kwargs)
    
    def setUp(self):
        if isinstance(self.world, basestring):
            self.world = universe.create_world(self.world)
    
    def tearDown(self):
        self.world.end()

    def test_fish_deadly(self):
        # make sure there aren't two players
        self.world.remove_objects([self.world.objects['seachase_player']])

        # create the objects we want
        objects, _, _ = h.world.screen("""



F*
""", legend)
        self.world.create_objects(objects)

        player = self.world.objects['seachase_player']
        fish = self.world.objects['fish']

        fish.velocity = (0, 1)

        # check that we get a penalty and the episode terminates
        obs, reward, done, info = self.world.step('NOOP')
        self.assertTrue(np.array_equal(player.position, fish.position))
        self.assertEqual(reward, -1001)
        self.assertTrue(done)

        # check that we died
        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 1)
        self.assertEqual(kill_events[0].victim, self.world.objects['seachase_player']._id)
        self.assertEqual(kill_events[0].reason, 'fish')

    def test_submarines_deadly(self):
        # make sure there aren't two players
        self.world.remove_objects([self.world.objects['seachase_player']])

        # create the objects we want
        objects, _, _ = h.world.screen("""



S*
""", legend)
        self.world.create_objects(objects)

        player = self.world.objects['seachase_player']
        sub = self.world.objects['submarine']

        sub.velocity = (0, 1)

        # check that we get a penalty and the episode terminates
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1001)
        self.assertTrue(done)
        self.assertTrue(np.array_equal(player.position, sub.position))

        # check that we died
        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 1)
        self.assertEqual(kill_events[0].victim, self.world.objects['seachase_player']._id)
        self.assertEqual(kill_events[0].reason, 'submarine')

    def test_surface_submarines_deadly(self):
        # make sure there aren't two players
        self.world.remove_objects([self.world.objects['seachase_player']])

        # create the objects we want
        objects, _, _ = h.world.screen("""

s*
""", legend)
        self.world.create_objects(objects)

        player = self.world.objects['seachase_player']
        sub = self.world.objects['surface_submarine']

        # so we don't die from technicality
        player.alive = True
        player.num_divers = 1
        
        sub.velocity = (0, 1)

        # check that we get a penalty and the episode terminates
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1001)
        self.assertTrue(done)
        self.assertTrue(np.array_equal(player.position, sub.position), repr(player.position) + repr(sub.position))

        # check that we died
        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 1)
        self.assertEqual(kill_events[0].victim, self.world.objects['seachase_player']._id)
        self.assertEqual(kill_events[0].reason, 'surface_submarine')

    def test_submarines_cannot_shoot_themselves(self):
        # create the objects we want
        objects, _, _ = h.world.screen("""



S B
""", legend)
        self.world.create_objects(objects)

        sub = self.world.objects['submarine']
        bullet = self.world.objects['bullet']

        # make the sub the shooter of the bullet
        bullet.bullet_shooter = sub

        sub.velocity = (0, 1)
        bullet.velocity = (0, -1)
        self.world.step('NOOP')
        self.assertTrue(np.array_equal(sub.position, bullet.position))

        # check that the sub is still alive
        self.assertTrue(sub.alive)

    def test_submarines_bullets_deadly(self):
        # make sure there aren't two players
        self.world.remove_objects([self.world.objects['seachase_player']])

        # create the objects we want
        objects, _, _ = h.world.screen("""



S B*
""", legend)
        self.world.create_objects(objects)

        player = self.world.objects['seachase_player']
        sub = self.world.objects['submarine']
        bullet = self.world.objects['bullet']

        # make the sub the shooter of the bullet
        bullet.bullet_shooter = sub

        sub.velocity = (0, 1)
        bullet.velocity = (0, 1)

        # check that we get a penalty and the episode terminates
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1001)
        self.assertTrue(done)
        self.assertTrue(np.array_equal(player.position, bullet.position))

        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 2)

        # check that the bullet died
        self.assertTrue(isinstance(self.world.events[-2], events.KillEvent))
        self.assertEqual(kill_events[-2].victim, bullet._id)
        self.assertEqual(kill_events[-2].reason, 'sudden stop')

        # check that we died
        self.assertTrue(isinstance(self.world.events[-1], events.KillEvent))
        self.assertEqual(kill_events[-1].victim, player._id)
        self.assertEqual(kill_events[-1].reason, 'bullet')

    def test_asphyxiation_deadly(self):
        player = self.world.objects['seachase_player']

        # set oxygen level to zero
        player.needs_air = 0

        # check that we get a penalty and the episode terminates
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1001)
        self.assertTrue(done)

        # check that we died
        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 1)
        self.assertEqual(kill_events[0].victim, player._id)
        self.assertEqual(kill_events[0].reason, 'asphyxiation')

    def test_surfacing_behavior0(self):
        # in seachase, player dies if they surface without rescuing any divers

        player = self.world.objects['seachase_player']

        # surface the sub
        player.position = (1, 5)

        # check that we get a penalty and the episode terminates
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1001)
        self.assertTrue(done)

        # check that we died
        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 1)
        self.assertEqual(kill_events[0].victim, player._id)
        self.assertEqual(kill_events[0].reason, 'technicality')

    def test_surfacing_behavior1_5(self):
        # in seachase, player loses a diver if they surface with fewer than six divers

        player = self.world.objects['seachase_player']
        for num_divers in xrange(1, 6):
            # surface the sub
            player.position = (1, 5)

            # set the number of divers
            player.num_divers = num_divers

            # check that we get a step penalty and the episode doesn't terminate
            obs, reward, done, info = self.world.step('NOOP')
            self.assertEqual(reward, -1)
            self.assertFalse(done)

            # check that we lost a diver
            self.assertEqual(player.num_divers, num_divers - 1)

            # go back under so that we can surface again
            player.position = (2, 5)

            # check that we get a step penalty and the episode doesn't terminate
            obs, reward, done, info = self.world.step('NOOP')
            self.assertEqual(reward, -1)
            self.assertFalse(done)

            # get rid of enemies while we conduct our experiments
            self.world.remove_objects(self.world.objects.find(name='fish'))
            self.world.remove_objects(self.world.objects.find(name='submarine'))
            self.world.remove_objects(self.world.objects.find(name='surface_submarine'))
            self.world.remove_objects(self.world.objects.find(name='bullet'))

    def test_surfacing_behavior6(self):
        # in seachase, you get a bunch of points for surfacing with six divers

        player = self.world.objects['seachase_player']
        
        # set the number of divers
        player.num_divers = 6

        # surface the sub
        player.position = (1, 5)

        # check that we get a reward and the episode doesn't terminate
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 299)
        self.assertFalse(done)

    def test_max_divers(self):
        # in seachase, you cannot have more than six divers in the sub

        # make sure there aren't two players
        self.world.remove_objects([self.world.objects['seachase_player']])

        # create the objects we want
        objects, _, _ = h.world.screen("""


D*
""", legend)
        self.world.create_objects(objects)

        player = self.world.objects['seachase_player']
        diver = self.world.objects['diver']
        diver.velocity = (0, 1)

        # set the number of divers to max
        player.num_divers = 6
        
        # check that we get a step penalty and the episode doesn't terminate
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1)
        self.assertFalse(done)

        # check that we are on top of the diver but did not pick it up
        self.assertTrue(np.array_equal(player.position, diver.position))        
        self.assertTrue(diver.visible)
        self.assertEqual(player.num_divers, 6)

    def test_diver_pickup(self):
        # make sure there aren't two players
        self.world.remove_objects([self.world.objects['seachase_player']])

        # create the objects we want
        objects, _, _ = h.world.screen("""


D*
""", legend)
        self.world.create_objects(objects)

        player = self.world.objects['seachase_player']
        diver = self.world.objects['diver']
        diver.velocity = (0, 1)

        # check that we get a step penalty and the episode doesn't terminate
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1)
        self.assertFalse(done)

        # check that we picked up the diver
        self.assertTrue(np.array_equal(player.position, diver.position))        
        self.assertEqual(player.num_divers, 1)
        self.assertFalse(diver.visible)

    def test_bullets_kill_submarines(self):
        # make sure there aren't two players
        self.world.remove_objects([self.world.objects['seachase_player']])

        # create the objects we want
        objects, _, _ = h.world.screen("""



SB *
""", legend)
        self.world.create_objects(objects)

        player = self.world.objects['seachase_player']
        sub = self.world.objects['submarine']
        bullet = self.world.objects['bullet']

        # make the player the shooter of the bullet
        bullet.bullet_shooter = player

        bullet.velocity = (0, -1)

        # check that we get a reward and the episode doesn't terminates
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 19)
        self.assertFalse(done)

        self.assertTrue(np.array_equal(sub.position, bullet.position))

        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 2)

        # check that the bullet died
        self.assertEqual(kill_events[-2].victim, bullet._id)
        self.assertEqual(kill_events[-2].reason, 'sudden stop')

        # check that the sub died
        self.assertEqual(kill_events[-1].victim, sub._id)
        self.assertEqual(kill_events[-1].reason, 'bullet')

    def test_bullets_kill_fish(self):
        # make sure there aren't two players
        self.world.remove_objects([self.world.objects['seachase_player']])

        # create the objects we want
        objects, _, _ = h.world.screen("""



FB *
""", legend)
        self.world.create_objects(objects)

        player = self.world.objects['seachase_player']
        fish = self.world.objects['fish']
        bullet = self.world.objects['bullet']

        # make the player the shooter of the bullet
        bullet.bullet_shooter = player

        bullet.velocity = (0, -1)

        # check that we get a reward and the episode doesn't terminate
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, 19)
        self.assertFalse(done)

        self.assertTrue(np.array_equal(fish.position, bullet.position))

        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 2)

        # check that the bullet died
        self.assertEqual(kill_events[-2].victim, bullet._id)
        self.assertEqual(kill_events[-2].reason, 'sudden stop')

        # check that the fish died :(
        self.assertEqual(kill_events[-1].victim, fish._id)
        self.assertEqual(kill_events[-1].reason, 'bullet')

    def test_bullets_spare_divers(self):
        # make sure there aren't two players
        self.world.remove_objects([self.world.objects['seachase_player']])

        # create the objects we want
        objects, _, _ = h.world.screen("""



 DB *
""", legend)
        self.world.create_objects(objects)

        player = self.world.objects['seachase_player']
        diver = self.world.objects['diver']
        bullet = self.world.objects['bullet']

        # make the player the shooter of the bullet
        bullet.bullet_shooter = player

        bullet.velocity = (0, -1)

        # check that we get a step penalty and the episode doesn't terminate
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1)
        self.assertFalse(done)

        # check that the bullet hit the diver
        self.assertTrue(np.array_equal(diver.position, bullet.position))

        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 1)

        # check that the bullet died
        self.assertEqual(kill_events[-1].victim, bullet._id)
        self.assertEqual(kill_events[-1].reason, 'sudden stop')

    def test_bullets_spare_surface_submarines(self):
        # make sure there aren't two players
        self.world.remove_objects([self.world.objects['seachase_player']])

        # create the objects we want
        objects, _, _ = h.world.screen("""

 sB *
""", legend)
        self.world.create_objects(objects)

        player = self.world.objects['seachase_player']
        sub = self.world.objects['surface_submarine']
        bullet = self.world.objects['bullet']

        # make the player the shooter of the bullet
        bullet.bullet_shooter = player

        # set num_divers to 1 so we can surface without dying
        player.num_divers = 1

        bullet.velocity = (0, -1)

        # check that we get a step penalty and the episode doesn't terminate
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1)
        self.assertFalse(done)

        # check that the bullet hit the sub
        self.assertTrue(np.array_equal(sub.position, bullet.position))

        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 1)

        # check that the bullet died
        self.assertEqual(kill_events[-1].victim, bullet._id)
        self.assertEqual(kill_events[-1].reason, 'sudden stop')

    def test_frame_absorbs_bullets(self):
        # make sure there aren't two players
        self.world.remove_objects([self.world.objects['seachase_player']])

        # create the objects we want
        objects, _, _ = h.world.screen("""


 B *
 B
 B
 B
 B
 B
 B
 B
""", legend)
        self.world.create_objects(objects)

        player = self.world.objects['seachase_player']
        sub = self.world.objects['surface_submarine']
        bullets = self.world.objects['bullet']

        for bullet in bullets:
            # make the player the shooter of the bullet
            bullet.bullet_shooter = player
            bullet.velocity = (0, -1)

        # check that we get a step penalty and the episode doesn't terminate
        obs, reward, done, info = self.world.step('NOOP')
        self.assertEqual(reward, -1)
        self.assertFalse(done)

        kill_events = [x for x in self.world.events if isinstance(x, events.KillEvent)]
        self.assertEqual(len(kill_events), 8)

        # check that the bullets died
        for event in kill_events:
            self.assertTrue(isinstance(self.world.objects[event.victim], objects_mod.Bullet))
            self.assertEqual(event.reason, 'sudden stop')

    def test_actions_and_orientation(self):
        # check that we can only be oriented LEFT or RIGHT, and that the
        # orientation switches when we move LEFT or RIGHT, respectively

        player = self.world.objects['seachase_player']
        posn = player.position

        self.world.step('LEFT')
        self.assertEqual(player.orientation, 0)
        self.assertTrue(np.array_equal(player.position, posn + (0, -1)))

        self.world.step('UP')
        self.assertEqual(player.orientation, 0)
        self.assertTrue(np.array_equal(player.position, posn + (-1, -1)))

        self.world.step('RIGHT')
        self.assertEqual(player.orientation, 2)
        self.assertTrue(np.array_equal(player.position, posn + (-1, 0)))

        self.world.step('DOWN')
        self.assertEqual(player.orientation, 2)
        self.assertTrue(np.array_equal(player.position, posn))

    def test_energy(self):
        self.world.agent = agents.RandomAgent(self.world)
        energy = self.world.energy
        for _ in xrange(100):
            self.world.step()
            self.assertEqual(self.world.energy, energy)

if __name__ == '__main__':
    unittest.main()
