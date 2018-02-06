import numpy as np
from pixelworld.envs.pixelworld import core, object_attributes as oa, library
from ..helpers import h, L

switch = library.import_item('world', 'switch')
Switch, SwitchHitEvent = switch.Switch, switch.SwitchHitEvent

class MultiSwitchJudge(core.Judge):
    """A judge that rewards you for flipping all the switches in the correct order,
    and penalizes you for flipping a switch out of order. The episode ends when
    you have flipped all of the switches.
    """
    # id of the next switch to flip
    _next_switch = 0

    # ignore the switch-hitting event reward
    _use_event_reward = False

    def prepare(self):
        """Prepare the judge by finding the switch with the lowest id and making that
        the next switch to flip.
        """
        self._next_switch = self._world.objects.find(name='switch')[0].id

    def _calculate_reward(self, goals, events):
        """Calculate the reward. In this case it is +100 for flipping the right switch
        and -100 for flipping the wrong switch.

        Parameters
        ----------
        goals : list[Goal]
            Goals that were achieved during the step
        events : list of Events
            Events that happened this step.
        """
        tot = 0
        for event in events:
            if isinstance(event, SwitchHitEvent):
                if event.switch_id == self._next_switch:
                    tot += 100
                    next_found = False
                    for switch in self._world.objects.find(name='switch'):
                        if switch.id > self._next_switch:
                            # guaranteed to be the switch with the next lowest
                            # id, since we're going through in order.
                            self._next_switch = switch.id
                            next_found = True
                            break
                    if not next_found:
                        self._next_switch = None
                else:
                    tot -= 100

        return tot

    def _is_done(self, goals, events):
        """Decide whether to end the episode. In this case, we end the episode if all
        switches have been flipped in correct order. (With wrong flips ignored;
        you just need to flip each switch correctly once.)

        Parameters
        ----------
        goals : list[Goal]
            Goals that were achieved during the step
        events : list of Events
            Events that happened this step.
        """
        if self._next_switch is None:
            return True
        return False


class MultiSwitchWorld(core.PixelWorld):
    """A world full of delightful switches to flip.

    Parameters
    ----------
    objects : list (optional)
        Objects to create beside self and switches
    num_switches : int
        Number of switches to create.
    small_switches : bool, defaults to False
        Create single-pixel switches instead of two-pixel switches.
    """
    def __init__(self, objects, num_switches=8, small_switches=False, **kwargs):
        assert num_switches > 0

        if small_switches:
            switches = [['switch', dict(sprites=[np.array([[i]], dtype=int), np.array([[i]], dtype=int)],
                                        switch_style=0, mass=0, zorder=-1)]
                        for i in xrange(2, 2 + num_switches)]
        else:
            switches = [['switch', dict(sprites=[np.array([[i, 1]], dtype=int), np.array([[1, i]], dtype=int)])]
                        for i in xrange(2, 2 + num_switches)]

        if objects is None:
            objects = []
        objects = objects + ['self'] + switches

        super(MultiSwitchWorld, self).__init__(objects=objects, **kwargs)
                            

judge = MultiSwitchJudge
world = MultiSwitchWorld
