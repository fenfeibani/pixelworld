import os, sys, re
from numbers import Number
from collections import OrderedDict, defaultdict
import time

from pixelworld.envs.pixelworld import universe, library, core
from pixelworld.envs.pixelworld.core import PixelWorld
from pixelworld.envs.pixelworld.utils import ask, askyn


class Test(object):
    world = None
    
    _defaults = {
        'seed': 101181,
        'agent': 'human',
    }
    
    _name = None
    
    _end_prompt = None
    
    def __init__(self, name, run=True, end_prompt=True, **kwargs):
        kwargs['agent'] = core.Agent
        
        print 'ran init'
        self._name = name
        
        self._end_prompt = end_prompt
        
        #create the environment
        start = time.clock()
        self.world = universe.create_world(name,
                        defaults=self._defaults,
                        ignore_world_agent=True,
                        **kwargs)
        end = time.clock()
        print name, end - start, '???'

        self.lag = end-start


def run_tests_create():
    lags = defaultdict(lambda: 0)
    for name in library.menu():
        for i in xrange(100):
            d = Test(name=name, end_prompt=False)
            lags[name] += d.lag

    print
    for name in lags:
        print name, lags[name]

    return lags

def run_tests_randomize():
    lags = defaultdict(lambda: 0)
    for name in library.menu():
        begin = time.clock()
        d = Test(name=name, end_prompt=False)
        for i in xrange(99):
            d.world._reset()
        end = time.clock()
        lags[name] = end - begin

    print
    for name in lags:
        print name, lags[name]

    return lags


if __name__ == '__main__':
    import cProfile
    #cProfile.run('run_tests_reset()', 'stats')
    lags1 = run_tests_create()
    lags2 = run_tests_randomize()
    print '\nRatios\n'
    for k in lags1:
        print k, lags1[k] / lags2[k]
