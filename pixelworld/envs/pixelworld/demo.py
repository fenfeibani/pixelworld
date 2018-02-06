'''
    enables interactive demos of library worlds. see
    pixelworld.envs.pixelworld.library.menu() for options.
'''
import os, sys, re
from numbers import Number
from collections import OrderedDict

from core import PixelWorld
from utils import ask, askyn
import universe


class Demo(object):
    """runs a demo of a library world"""
    #the PixelWorld
    world = None
    
    #default PixelWorld constructor parameters
    _defaults = {
        'agent': 'human',
    }
    
    #the demo name
    _name = None
    
    #True to prompt the user to end the world when the demo ends
    _end_prompt = None
    
    def __init__(self, world, run=True, end_prompt=True, **kwargs):
        """
        Parameters
        ----------
        world : string
            the name of the library world to demo (see
            pixelworld.envs.pixelworld.library.menu()).
        run : bool, optional
            True to run the simulation
        end_prompt : bool, optional
            True to prompt the user to end the world when the demo is over.
        **kwargs
            extra keyword arguments to pass to the PixelWorld constructo.
        """
        self._name = kwargs.get('name', world)
        
        self._end_prompt = end_prompt
        
        #create the environment
        self.world = universe.create_world(world,
                        defaults=self._defaults,
                        ignore_world_agent=False,
                        **kwargs)
        
        #run the simulation
        if run:
            self.run()
    
    def run(self):
        """run the demo"""
        #render the initial scene
        self.world.render()
        
        #run the simulation
        done = False
        while not done:
            if (self.world.multi_agent_mode and 'human' in [agent.name for agent in self.world.agent]) or \
                    (not self.world.multi_agent_mode and self.world.agent.name == 'human'):
                # single step at a time
                done = self.step()
            else:  # run some number of steps
                num_steps = ask('maximum number of steps to execute, or q to quit:',
                                default=1, choices=[Number, 'q'], num_type=int)
                
                if num_steps == 'q':
                    done = True
                else:
                    for _ in xrange(num_steps):
                        done = self.step()
                        if done:
                            break
        
        #display the result and end the simulation
        print '%s finished at time %d' % (self._name, self.world.time)
        print 'total reward: %s' % (self.world.total_reward)
        
        #end the world, possibly after confirmation
        if not self._end_prompt or askyn('end the world?', default=True):
            self.end()
    
    def step(self):
        """advance the simulation forward by one time step
        
        Returns
        -------
        done : bool
            True if the simulation is finished
        """
        #step the world
        obs, reward, done, info = self.world.step()
        
        #render to the screen
        self.world.render()
        
        return done
    
    def end(self):
        """end the demo"""
        #end the world
        output_path = self.world.end()
        
        #print the output gif path
        if output_path:
            print 'gif saved to: %s' % (output_path)


def parse_args(**kwargs):
    """parse the command line, which should take the following form:
        demo.py <world_name> <param1>=<val1> ... <paramN>=<valN>
    
    Parameters
    ----------
    **kwargs
        default parameters that can be overridden
    
    Returns
    -------
    world_name : string
        the name of the world to run
    params : dict
        a dict of specified command line parameters
    """
    assert len(sys.argv) > 1, 'must provide a world name'
    
    world_name = sys.argv[1]
    
    for param in sys.argv[2:]:
        parts = re.match('^([^=]+)=(.+)$', param)
        assert parts is not None, '"%s" is not a valid argument' % (param,)
        
        key = parts.group(1)
        value = parts.group(2)
        
        #try to eval, otherwise just take the string
        try:
            value = eval(value)
        except NameError:
            pass
        
        kwargs[key] = value
    
    return world_name, kwargs


if __name__ == '__main__':
    #this is probably weird and not a good idea, but i need classes to be the
    #same here as they are when defined in the demo itself
    from pixelworld.envs.pixelworld.demo import Demo
    
    name, params = parse_args(end_prompt=False)
    
    d = Demo(world=name, **params)
