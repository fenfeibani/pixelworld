from core import Entity, ObjectAttribute, Object, Event, Goal, Judge, Variant, \
    Randomizer, Agent, WorldAttribute, PixelWorld

#construct a dict mapping all base Entity names to their classes
base_entities = {name:family.values()[0] for name,family in Entity.__metaclass__._all_class_families.iteritems()
                    if issubclass(family.values()[0], Entity)}

from universe import create_world

import core
import library

#make sure the Entities defined in these modules are accessible
import agents
import events
import goals
import judges
import object_attributes
import objects
import pixel_worlds
import randomizers
import variants
import world_attributes
