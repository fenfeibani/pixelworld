import os, sys, re
from numbers import Number
from collections import OrderedDict, defaultdict
import time

from pixelworld.envs.pixelworld import universe, library, core, agents
from pixelworld.envs.pixelworld.core import PixelWorld
from pixelworld.envs.pixelworld.utils import ask, askyn


class Test(object):
    world = None
    
    _defaults = {
#        'seed': 101181,
#        'agent': 'human',
    }
    
    _name = None
    
    def __init__(self, name, run=True, end_prompt=True, **kwargs):
        self._name = name
        
        #create the environment
        self.world = universe.create_world(name,
                        defaults=self._defaults,
                        **kwargs)

def run_tests(ls):
    if len(ls) == 0:
        ls = library.menu()
        ls.remove('blank')
    for name in ls:
        for i in xrange(10):
            d = Test(name=name, agent='random')
            for j in xrange(2):
                for t in xrange(50):
                    obs, reward, done, info = d.world.step()
                    d.world.render()
                d.world.reset()

if __name__ == '__main__':
    run_tests(sys.argv[1:])
