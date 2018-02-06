'''
    basic set of Events for PixelWorld
'''
import core

class CollisionEvent(core.Event):
    """an event that occurs whenever two Object collide.
    
    records the indices of the colliding Objects.
    
    PositionObjectAttribute handles this event.
    """
    _name = 'collision'
    
    _parameters = ['indices']
    
    def _get_description(self):
        assert len(self.indices) == 2
        obj1 = self._world._objects[self.indices[0]]
        obj2 = self._world._objects[self.indices[1]]
        
        return 'collision between %s and %s' % (obj1.unique_name, obj2.unique_name)


class LeaveScreenEvent(core.Event):
    """an event that occurs whenever Objects leave the visible part of the
    world.
    
    records the indices of the Objects that left the screen.
    
    PositionObjectAttribute handles this event.
    """
    _name = 'leave_screen'
    
    _parameters = ['indices']
    
    def _get_description(self):
        names = [self._world._objects[idx].unique_name for idx in self.indices]
        names = ', '.join(names)
        
        return 'object(s) left the screen: %s' % (names)


class PushEvent(core.Event):
    """an event that occurs whenever a push is attempted.
    
    records the index of the Object that attempted the pushed, and whether the
    push was successful
    
    PushesObjectAttribute handles this event.
    """
    _name = 'push'
    
    _parameters = ['idx', 'success']
    
    def _get_description(self):
        push_type = 'successful' if self.success else 'failed'
        obj = self._world._objects[self.idx]
        return '%s push by %s' % (push_type, obj.unique_name)


class BulletFireEvent(core.Event):
    """Event class for when a shooter fires a bullet."""
    #position: position the shooter was in when the bullet was fired
    #dir: which cardinal direction the bullet was fired in (LEFT=0, UP=1,
    #     RIGHT=2, DOWN=3)
    #bullet_id: the id of the bullet object
    _parameters = ['position', 'dir', 'bullet_id']

    def _get_description(self):
        return 'bullet fired by %s' % \
            self.world.objects[self._parameters['bullet_id']].bullet_shooter.unique_name


class KillEvent(core.Event):
    """Event class for when an object gets killed."""

    #victim: the id of the object that got killed
    #reason: the cause of death
    _parameters = ['victim', 'reason']

    def _get_description(self):
        return '%s died by %s' % (self.world.objects[self.victim].unique_name, self.reason)


class GripEvent(core.Event):
    """an event that occurs whenever a grip is attempted.
    
    records the index of the Object that attempted the grip, the object that
    was to be gripped (if any), and whether the grip was successful
    
    GripsObjectAttribute handles this event.

    Parameters
    ----------
    gripper : index of the gripping object
    grippee : index of the object being gripped, or None if no object
    success : bool (did the grip succeed?)
    """
    _parameters = ['gripper', 'grippee', 'success']
    
    def _get_description(self):
        grip_type = 'successful' if self.success else 'failed'
        gripper = self._world._objects[self.gripper]
        grippee = self._world._objects[self.grippee]
        rv = '%s grip by %s' % (grip_type, gripper.unique_name)
        if grippee is not None:
            rv = rv + ' of %s' % grippee.unique_name
        return rv


class UngripEvent(core.Event):
    """an event that occurs whenever a ungrip is attempted.
    
    records the index of the Object that attempted the grip, the object that
    was being gripped (if any), and whether the grip was successful
    
    GripsObjectAttribute handles this event.

    Parameters
    ----------
    gripper : index of the gripping object
    grippee : index of the object being gripped, or None if no object
    success : bool (did the grip succeed?)
    """
    _parameters = ['gripper', 'grippee', 'success']
    
    def _get_description(self):
        grip_type = 'successful' if self.success else 'failed'
        gripper = self._world._objects[self.gripper]
        rv = '%s ungrip by %s' % (grip_type, gripper.unique_name)
        if self.grippee is not None:
            grippee = self._world._objects[self.grippee]
            rv = rv + ' of %s' % grippee.unique_name
        return rv


class ChangeScreenEvent(core.Event):
    """Event signaling that a ScreenBasedPixelWorld should change screens. 

    Parameters
    ----------
    new_screen : integer, the id of the new screen
    """
    _parameters = ['new_screen']

    def _get_description(self):
        return 'changed screen to %s' % self.new_screen


class CorrectClassificationEvent(core.Event):
    """Event that we responded correctly to a classification problem.

    Parameters
    ----------
    truth : string
        True class
    guess : string
        Guessed class (which in this case will equal truth)
    """
    _reward = 100
    _terminates = True
    _parameters = ['truth', 'guess']


class IncorrectClassificationEvent(core.Event):
    """Event that we responded incorrectly to a classification problem.

    Parameters
    ----------
    truth : string
        True class
    guess : string
        Guessed class
    """
    _reward = -100
    _terminates = True
    _parameters = ['truth', 'guess']


