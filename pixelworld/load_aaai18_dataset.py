"""Load AAAI18 PixelWorld datasets defined within experiments/aaai18"""

import argparse
import copy
import numpy as np
import os
from pprint import pprint
import random
import time
import traceback

import rllab

from pixelworld import run_policy
from pixelworld.concept_csp import (CSPClassificationConcept, 
                                    CSPBringAboutConcept, generate_dataset)
from pixelworld.envs import modules
from pixelworld.envs.gym_env import GymEnv
from pixelworld.envs.modular_env import ModularEnv
from pixelworld.expcfg import load_raw_cfg, process_cfg, process_metacfg
from pixelworld.run_policy import split_dataset, parse_video_schedule


EXPERIMENT_DIR = '../experiments/aaai18/'
EXPERIMENT_FILENAMES = ['con1-class', 'obj-con1-class', 'objects1-class', 
                        'con1-sig', 'objects1-sig']
RAW_DATASETS = None


def load_datasets():
    """Loads specifications for AAAI18 datasets from metaexperiment files on disk."""
    global RAW_DATASETS

    if RAW_DATASETS is not None:
        return RAW_DATASETS

    RAW_DATASETS = {}
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), EXPERIMENT_DIR)

    for filename in EXPERIMENT_FILENAMES:
        filename = os.path.join(root_dir, filename)
        raw_cfg = load_raw_cfg(filename)
        assert raw_cfg.get('mode', 'meta') == 'meta'

        cfgs, depends = process_metacfg(raw_cfg, filename, only_executable=False)

        for k, v in cfgs.items():
            if k[:5] == 'data-' and k[-4:] == '-big':
                name = k[5:-4]
                assert name not in RAW_DATASETS
                RAW_DATASETS[name] = v

    return RAW_DATASETS


def list_datasets():
    """Return list of names of possible AAAI18 datasets."""
    return sorted(load_datasets().keys())


def get_dataset(dataset_name):
    """Get a AAAI18 dataset by name.

    A dataset is a dict containing two keys:
        'spec': list of pixelworld specifications for environments
        'labels': list of ints
            For a classification dataset, 1 if the environment satisfies the
            concept, 0 otherwise. For a bringabout dataset, always 1 since we
            generate environments assuming the concept can be brought about.
    """
    raw_datasets = load_datasets()
    if dataset_name not in raw_datasets:
        raise Exception('Dataset %s not found. Use list_datasets().' 
                        % (dataset_name,))
    dataset = process_cfg(raw_datasets[dataset_name], quiet=True)['dataset']
    return dataset


def train_test_split(dataset):
    """Divide dataset into training and testing datasets.

    Uses parameters combined from run_policy.defaults and files in 
    experiments/aaai18.
    """
    test_size = 0.1
    test_train_seed = 521

    split = split_dataset(dataset=dataset, test_size=test_size,
                          test_train_seed=test_train_seed)

    return split['train_dataset'], split['test_dataset']


def make_env(dataset, use_minimal_pw=True, seed=0, video_schedule='none',
             rllab_env=False):
    """Create an environment from a dataset.

    Parameters
    ----------
    dataset : dict
        Dataset of environment specs and labels.
    use_minimal_pw : bool, optional
        Whether to use minimal pixelworld instead of full pixelworld.
    seed : int, optional
        a seed for the random number generator used for selecting from dataset
    video_schedule : str, optional
        string specifying which episodes to record videos, one of:
            none:      no video recording
            cubic:     capped cubic (record episodes whose numbers are
                         are cubes less than 1000, then every 1000 episodes)
            fixed:     every episode
            fixed-<n>: every <n> episodes
    rllab_env : bool, optional
        whether to return an rllab Env rather than an OpenAI gym Env

    Returns
    -------
    env : Gym or rllab Env
        environment sampling from the dataset
    """
    specs = dataset['specs']
    labels = dataset['labels']
    concept_object = specs[0]['concept']

    module_specs = [
        {},
        {
            'class': modules.DatasetModule,
            'kwargs': {'seed': seed},
        },
        {
            'class': modules.MinimalPixelWorldModule if use_minimal_pw else modules.PixelWorldModule,
            'kwargs': {'remove_noop': True},
        },
        {
            'class': modules.HandObservationModule,
        },
    ]
    if isinstance(concept_object, CSPClassificationConcept):
        module_specs[0]['class'] = modules.ClassificationSignalModule
    elif isinstance(concept_object, CSPBringAboutConcept):
        module_specs[0]['class'] = modules.BringAboutShapingModule
        module_specs[0]['kwargs'] = {'reward_mistake': 0}
    else:
        raise Exception('')

    # Create the environment
    env = ModularEnv(
        module_specs=module_specs,
        specs=specs,
        labels=labels,
    )

    # Optionally wrap the Gym environment in a rllab environment
    if rllab_env:
        env = GymEnv(env, video_schedule=parse_video_schedule(video_schedule))
    return env


def get_envs(dataset, rllab_env=False):
    """Return training and testing environments for a given dataset.

    Parameters
    ----------
    dataset : str or dataset, optional
        the name of the dataset from which to generate the environment, or an
        actual dataset    
    rllab_env : bool, optional
        whether to return an rllab Env rather than an OpenAI gym Env.

    Returns
    -------
    train_env : gym or rllab Env
        environment sampling from the training dataset
    test_env : gym or rllab Env
        environment sampling from the test dataset
    """
    if isinstance(dataset, basestring):
        dataset = get_dataset(dataset)
    train_dataset, test_dataset = train_test_split(dataset)
    train_env = make_env(train_dataset, rllab_env=rllab_env)
    test_env = make_env(test_dataset, rllab_env=rllab_env)
    return train_env, test_env


def generate_datasets():
    """Generate and cache all AAAI18 datasets.

    The cache is maintained by the diskcache package in the 'cache' directory.
    To clear the cache, delete this directory.
    """
    dataset_names = list_datasets()
    for idx, dataset_name in enumerate(dataset_names):
        print('*** Generating (%s/%s) %s' % (idx+1, len(dataset_names), dataset_name))
        dataset = get_dataset(dataset_name)


def make_dataset(concept, generators, concept_type='classification', 
                 use_cache=False, num_samples=50, macros={}):
    """Make a dataset from its logical description.

    Parameters
    ----------
    concept : str
        logical expression to test whether a concept holds of the environment
    generators : [str]
        list of logical expressions used to generate candidate environments
    concept_type : str
        'classification' or 'bringabout'
    use_cache : bool, optional
        whether to cache the generated dataset to disk
    num_samples : int
        how many samples to generate
    macros : {str : concept}
        dictionary of macros for parsing concepts and generators

    Returns
    -------
    dataset : dataset
        generated dataset
    """
    seed = 0
    settings = {'default_num_samples': 20,
                'fixed_floor': True,
                'floor_height': 3,
                'frame': True,
                'height': 18,
                'padding': 2,
                'width': 35}
    dataset = generate_dataset(concept, generators, num_samples, seed,
                               settings=settings, concept_type=concept_type, 
                               use_cache=use_cache, concept_macros=macros,
                               generator_macros=macros)
    return dataset


if __name__ == '__main__':
    generate_datasets()
