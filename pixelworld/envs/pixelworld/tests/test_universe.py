"""make sure all the preset environments in universe can run
"""
import unittest
import sys

from pixelworld.envs.pixelworld import core, universe, library
from pixelworld.envs.pixelworld.agents import HumanAgent
from pixelworld.envs.pixelworld.datatypes import TypeFamily
from pixelworld.envs.pixelworld.tests import test_core as tc

def test_get_pixelworld_defaults():
    """Check that the defaults have the things we expect"""
    defaults = universe.get_pixelworld_defaults()
    assert sorted(defaults.keys()) == sorted([
            'capture', 'populate', 'objects', 'randomizer', 'observed_objects',
            'time', 'render_type', 'agent', 'height', 'debug', 'seed', 'goals',
            'render_size', 'judge', 'variants', 'observed_attributes',
            'observed_window', 'name','width', 'obs_type'])


def test_worlds():
    for env in library.menu('world'):
        yield run_world, env


def run_world(name):
    """Check that we can run a world and that it behaves as expected"""
    #save the initial class-hierarchy state so we don't pollute the namespace
    TypeFamily._save_class_hierarchy_state()
    
    params = {}
    
    #make sure we don't get prompted for actions
    world_params = universe.load_parameters(name, reload_module=True)
    if issubclass(core.Agent.get_class(world_params['agent']), HumanAgent):
        params['agent'] = 'random'
    
    #create the world
    world = universe.create_world(name, reload_module=True, **params)

    #test the world
    tc.TestWorld('test_world', world=world).test_world()
    
    #run the simulation for a bit
    tc.TestWorld('test_run', world=world).test_run()
    
    #examine the world again
    tc.TestWorld('test_world', world=world).test_world()
    
    #make sure we can rewind the world
    tc.TestWorld('test_state', world=world).test_state()
    
    #restore the initial class-hierarchy state
    TypeFamily._restore_class_hierarchy_state()

def run_all(*names):
    for run, name in test_worlds():
        if len(names) == 0 or name in names:
            sys.stdout.write(name + ' ')
            sys.stdout.flush()
            run(name)

if __name__ == '__main__':
#     import cProfile
#     cProfile.run('run_all()', 'stats')
    run_all(*sys.argv[1:])
    test_get_pixelworld_defaults()

