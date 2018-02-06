"""Demos illustrating PixelWorld."""

from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import copy
import time
import re

import joblib
import numpy as np
import rllab
from rllab.algos.npo import NPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy

from pixelworld.envs import modules
from pixelworld.envs.pixelworld.utils import KeyboardController
from pixelworld.envs.modular_env import ModularEnv
from pixelworld.load_aaai18_dataset import (get_dataset, get_envs, list_datasets,
                                            make_env, make_dataset)


default_dataset = 'cl_self-in-container_vs_noncontainer'

# Custom dataset, equivalent to cl_self-in-container_vs_noncontainer as defined
# in experiments/aaai18/DATA/bfaw-containment-cl.datasets and files referenced
# therein. Feel free to experiment with modifications!

# Whether self is contained and at bottom of container (self-in-container)
custom_concept = "?x container(x) & inside_supported(self,x)"

custom_generators = [
    # Self is in a container (bfaw-self-in-container-on-floor)
    "?f ?x bfaw-container-on-floor(f,x) & inside_supported(self,x)",

    # Self is in a noncontainer (bfaw-self-in-noncontainer-on-floor)
    "?f ?x bfaw-noncontainer-on-floor(f,x) & inside_supported(self,x)",
]

custom_macros = {
    # There is a blue floor
    "blue-floor": "?f floor(f) & blue(f)",

    # Blue floor one white object
    "bfw1": "?f ?x blue-floor(f) & white(x)",

    # One white object on a blue floor 
    "wobf1": "?f ?x bfw1(f,x) & on_top(x,f)",

    # There is a container on the floor
    "bfaw-container-on-floor": "?f ?x wobf1(f,x) & container(x)",

    # There is a noncontainer on the floor
    "bfaw-noncontainer-on-floor": "?f ?x wobf1(f,x) & noncontainer(x)",
}

# experiments/aaai18 uses datasets of size 50, but 10 generates faster
custom_size = 10 




def do_print_aaai18_datasets():
    """Print the available AAAI18 datasets."""
    print('Datasets:')
    for idx, raw_dataset_name in enumerate(list_datasets()):
        print('  %d. %s' % (idx, raw_dataset_name))


def do_pixelworld_interactive(dataset=default_dataset, seed=None, num_episodes=1,
                              use_minimal_pw=True):
    """Run an example pixelworld environment in interactive mode using the
    keyboard.
    
    Parameters
    ----------
    dataset : str or dataset, optional
        the name of the dataset from which to generate the environment, or an
        actual dataset
    seed : int, optional
        a seed for the random number generator
    num_episodes : int, optional
        number of episodes to run before returning 
    use_minimal_pw : bool, optional
        whether to use minimal pixelworld rather than full pixelworld
    """
    #reseed the rng
    if seed is not None:
        np.random.seed(seed=seed)
    
    # Create an environment using the full dataset
    if isinstance(dataset, basestring):
        dataset = get_dataset(dataset)
    env = make_env(dataset, seed=seed)

    env.reset()
    env.render()
    
    #create the keyboard controller
    controller = KeyboardController(
        window=env.viewer.window,
        actions=env.action_space.names,
    )
    controller.register_key('ESCAPE', 'QUIT')
    controller.register_action('QUIT')
    
    #run the environment in interactive mode
    controller.print_legend()
    abort = False
    for i in range(num_episodes):
        signal = None
        while True:
            action = controller.get_action()
            
            if action == 'QUIT':
                abort = True
                break
            else:
                if action.startswith('SIG'):
                    signal = action
                
                idx = env.action_space.names.index(action)
                
                _, _, done, _ = env.step(idx)
                env.render()
            
            if done:
                print('RESULTS')
                print('-------')
                print(' label: %d' % (env.state['label'],))
                print('signal: %s' % (signal,))
                print('reward: %.3f' % (env.total_reward,))
                print()
                
                break
        
        if abort:
            break
        
        if i < num_episodes - 1:
            env.reset()
            env.render()
        
    env.close()


def do_pixelworld_experiment(dataset=default_dataset, visualize_policy=True,
                             load_policy=False, policy_filename='policy100.pkl'):
    """Train by reinforcement learning in a pixelworld training environment,
    optionally visualizing the policy in test and training environment.

    Parameters
    ----------
    dataset : str, optional
        the name of the dataset from which to generate the environment, or an
        actual dataset
    visualize_policy : bool, optional
        whether to vizualize the policy acting within test/train environments
    load_policy : bool, optional
        whether to load a saved policy from disk
    policy_filename : str, optional
        filename to store/load policy to/from
    """
    train_env, test_env = get_envs(dataset, rllab_env=True)

    if load_policy:
        policy = joblib.load(policy_filename)
    else:
        # Train policy in train_env
        policy = CategoricalGRUPolicy(
            hidden_sizes=[32],
            env_spec=train_env.spec)
        baseline = LinearFeatureBaseline(
            env_spec=train_env.spec)
        algo = NPO(
            env=train_env,
            policy=policy,
            baseline=baseline,
            max_path_length=10000,
            whole_paths=True,
            n_itr=100, # 100
            batch_size=2000)

        np.random.seed(137)
        algo.train()
        joblib.dump(policy, policy_filename)

    # Visual trained policy by rollouts in test and training environments
    if visualize_policy:
        delay = 0.5 # 0.1 2
        num_envs = 10 # 100
        for env_name, env in [('train', train_env), ('test', test_env)]:
            print()
            num_positive = 0
            tot_tot_r = 0.0
            for seed in range(num_envs):
                print(env_name, 'rollout', seed)
                np.random.seed(seed)
                policy.reset()
                o = env.reset()
                d = False
                tot_r = 0.0

                env.render()
                time.sleep(delay)
                while not d:
                    a, info = policy.get_action(o)
                    o, r, d, env_info = env.step(a)
                    tot_r += r
                    print('  step(%s) -> reward %s' % (a, r))
                    env.render()
                    time.sleep(delay)
                
                if a == 1: # SIG1
                    positive = r > 0
                else:
                    positive = r < 0
                num_positive += positive
                print(env_name, 'rollout done (%s, %s)' % (tot_r, positive))
                tot_tot_r += tot_r
        print(env_name, 'avg tot r', tot_tot_r/num_envs)
        print(env_name, 'avg positive', num_positive/float(num_envs))


def do_custom_pixelworld_interactive(concept=custom_concept, 
                                     generators=custom_generators, 
                                     macros=custom_macros, use_cache=True,
                                     num_samples=custom_size,
                                     **kwargs):
    """do_pixelworld_interactive using custom dataset."""
    dataset = make_dataset(concept=concept, generators=generators, macros=macros,
                           use_cache=use_cache, num_samples=num_samples)
    do_pixelworld_interactive(dataset=dataset, **kwargs)


def do_custom_pixelworld_experiment(concept=custom_concept, 
                                    generators=custom_generators, 
                                    macros=custom_macros, use_cache=True,
                                    num_samples=custom_size,
                                    **kwargs):
    """do_pixelworld_experiment using custom dataset."""
    dataset = make_dataset(concept=concept, generators=generators, macros=macros,
                           use_cache=use_cache, num_samples=num_samples)
    do_pixelworld_experiment(dataset, **kwargs)



def parse_args(**kwargs):
    """Parse the command line, which should take the following form:
        demo.py <param1>=<val1> ... <paramN>=<valN>
    
    Parameters
    ----------
    **kwargs
        default parameters that can be overridden
    
    Returns
    -------
    params : dict
        a dict of specified command line parameters
    """
    for param in sys.argv[1:]:
        parts = re.match('^([^=]+)=(.+)$', param)
        assert parts is not None, '"%s" is not a valid argument' % (param,)
        
        key = parts.group(1)
        value = parts.group(2)
        
        #try to eval, otherwise just take the string
        try:
            value = eval(value)
        except Exception:
            pass
        
        kwargs[key] = value
    
    return kwargs

if __name__ == "__main__":
    params = parse_args(
        demo='interactive',
        dataset=default_dataset,
        )
    
    demo = params.pop('demo')
    dataset = params.pop('dataset')
    
    if demo == 'interactive':
        do_pixelworld_interactive(dataset=dataset, **params)
    elif demo == 'experiment':
        do_pixelworld_experiment(dataset=dataset, **params)
    elif demo == 'custom_interactive':
        do_custom_pixelworld_interactive(**params)
    elif demo == 'custom_experiment':
        do_custom_pixelworld_experiment(**params)
    elif demo == 'print_datasets':
        do_print_aaai18_datasets(**params)
    else:
        demos = ['interactive', 'experiment', 'custom_interactive',
                 'custom_experiment', 'print_datasets']
        raise ValueError('"%s" is not a valid demo; try one of %s' 
                         % (demo, ', '.join(demos)))
