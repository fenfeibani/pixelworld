import numpy as np
from pixelworld.envs.pixelworld import core, objects, object_attributes as oa, utils
from ..helpers import h, L

class SwitchHitEvent(core.Event):
    """Event that a switch was hit"""

    # on: boolean, was switch turned on or off
    # switch_id: the id of the switch object 
    _parameters = ['on', 'switch_id']

    _reward = 100

    def _get_description(self):
        if self._parameters['on']:
            return '%s was switched on' % self.world.objects[self.switch_id].unique_name
        else:
            return '%s was switched off' % self.world.objects[self.switch_id].unique_name

class SwitchStyle(core.NonNegativeIntegerObjectAttribute):
    """1 (proximity switch) or 0 (floor switch)"""
    _default_value = 1

    def _coerce_single_value(self, value):
        if value > 1:
            return None, TypeError('must be either 0 or 1')
        else:
            return value, False


class SwitchStateObjectAttribute(core.BooleanObjectAttribute, 
                                 core.SteppingObjectAttribute, 
                                 core.LinkObjectAttribute):
    """Which state a switch is on. Setting this variable will cause an event if you
    set it to something it wasn't already.

    Also watches for pushers coming within one pixel of the non-white pixel
    (for switch style 1, proximity switches) or within half a pixel of the
    non-white pixel (for switch style 0, floor switches), and switches itself
    when that happens.
    """

    _linked_attribute = 'sprite'

    #lookup table for what vector each direction represents
    _direction = [
        np.array((0,-1)), # left
        np.array((-1,0)), # up
        np.array((0,1)),  # right
        np.array((1,0)),  # down
        ]

    _step_after = ['pushes', 'position']
    _initialize_after = ['sprite']
    def _step_object(self, obj, t, dt, agent_id, action):
        """Watch for pushers coming within one pixel of the non-white pixel (for
        proximity switches) or within half a pixel (for floor switches), and
        switch when that happens.

        Parameters
        ----------
        obj : Object
            The switch object
        t : number
            The simulation time
        dt : number
            Steps since the last time this was called.
        action : string
            Action (ignored)
        """

        # find the position of the switch's active pixel
        posn = None
        for idx in obj.children:
            if self.world.objects[idx].color != 1:
                posn = self.world.objects[idx].position
        assert posn is not None

        pushers = self.world.object_attributes['pushes'].objects
        hit = False
        for pusher in pushers:
            # if switch style is 1 (proximity switch), check that the pusher is
            # within one pixel and that the orientation is consistent with the
            # switch's orientation if the switch has one
            if obj.switch_style == 1 and np.linalg.norm(pusher.position - posn) <= 1:
                if (not hasattr(obj, 'orientation') or 
                    np.dot(self._direction[obj.orientation], pusher.position - posn) > 0):
                    hit = True

            # if switch style is 0 (floor switch), check that the pusher is
            # within sqrt(2)/2 pixels, so that it is closer to this pixel than
            # any other pixel
            elif obj.switch_style == 0 and np.linalg.norm(pusher.position - posn) <= 0.70710678118:
                hit = True

        if hit:
            self.set(obj.id, not self.get(obj.id))

    def _set_data_object(self, obj, value):
        """Set the value of this attribute, and generate a switch hit event if it
        changed.

        Parameters
        ----------
        obj : Object
            switch object
        value : bool
            New value
        """
        if value != self.get(obj):
            event = SwitchHitEvent(self.world, on=value, switch_id=obj.id)        
        super(SwitchStateObjectAttribute, self)._set_data_object(obj, value)

    def _get_data_object(self, obj):
        """Get the value of this attribute.

        Parameters
        ----------
        obj : Object
            switch object
        """
        return bool(super(SwitchStateObjectAttribute, self)._get_data_object(obj))
        

class SwitchListeningObjectAttribute(core.ListeningObjectAttribute):
    """Attribute that makes objects listen for switch hit events and switch their
    sprite depending on whether the switch was turned on or off.
    """
    _step_after = ['switch_state']
    _selected_events = ['switch_hit']
    def _process_event_object(self, evt, obj, t, dt, agent_id, action):
        obj.sprite = int(evt.on)

class SwitchListener(objects.ComplexSpriteObject):
    """A sprite object to demonstrate switch-listening.
    """
    _attributes = ['switch_listening']
listener_sprites = [
h.sprite.from_string("""
111
121
111"""), 
h.sprite.from_string("""
111
131
111""")]

class Switch(objects.UnpushableObject,
             objects.ComplexSpriteObject, 
             ):
    """A switch that you can turn on or off by walking next to it."""
    _attributes = ['switch_state', 'switch_style']

switch_sprites = [h.sprite.from_string("21"), h.sprite.from_string("12")]

objects = ['self', 
           [['switch'], {'sprites': switch_sprites}],
           [['switch_listener'], {'sprites': listener_sprites}]
           ]
