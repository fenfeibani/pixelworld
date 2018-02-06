"""Train and evaluate policies given an experiment configuration."""

from __future__ import print_function
from __future__ import absolute_import

import collections
from collections import defaultdict
import copy
import numpy as np
import os
import re
import shutil
import time

from gym.envs.registration import EnvSpec
import joblib
from rllab.sampler.utils import rollout
from rllab.misc import logger
from rllab.misc import tensor_utils
from sklearn.model_selection import train_test_split

from pixelworld.envs.gym_env import GymEnv
from pixelworld.expcfg import load_raw_cfg
from pixelworld.misc import recursive_merge_into, safe_annotate
from pixelworld.output import (setup_output, shutdown_output, 
    get_output_dir, write_object, read_object, exists_object, set_delete_all)


# Policy evaluation 
# -----------------

def average_images(tensor):
    """Computes the average RGB image from a tensor of RGB images."""

    rs = tensor[:, :, :, 0]
    gs = tensor[:, :, :, 1]
    bs = tensor[:, :, :, 2]

    r = np.average(rs, axis=0)
    g = np.average(gs, axis=0)
    b = np.average(bs, axis=0)

    res = np.stack((r, g, b), axis=2)

    return np.array(np.round(res),dtype=np.uint8)


def max_images(tensor):
    """Computes the maximum RGB image from a tensor of RGB images."""

    rs = tensor[:, :, :, 0]
    gs = tensor[:, :, :, 1]
    bs = tensor[:, :, :, 2]

    r = np.amax(rs, axis=0)
    g = np.amax(gs, axis=0)
    b = np.amax(bs, axis=0)

    res = np.stack((r, g, b), axis=2)

    return np.array(np.round(res),dtype=np.uint8)


def rollout_record(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()

    # environment descriptions
    descs = defaultdict(list)
    for key, val in env.describe().items():
        descs[key].append(val)
    
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)

        # NOTE: this runs describe after the environment is done, so describe
        #       must be happy with this.
        # NOTE: this appending is dangerous! it makes the assumption that vals of
        #       different keys must all be simulatenously present, which might not be true;
        #       the result is a misalignment of vals of different keys further down!
        for key, val in env.describe().items():
            descs[key].append(val)

        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
    results =  dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos)
    )   
    for key, val in descs.items():
        # Keys ending with with _varlist should be concatenated not stacked
        if len(key) > 8 and key[-8:] == '_varlist':
            results[key[:-8]] = np.concatenate(val, axis=0)
        else:
            results[key] = tensor_utils.stack_tensor_list(val)

    return results


def seeded_rollout(env, instance_index, policy, seed, max_path_length=np.inf):
    """Deterministic evaluation running policy in the given environment
    initialized to the given instance. Seed is used to sample from
    the agent's action distribution as well as, potentially, the environment.

    Assumes it is passed a GymEnv which wraps a Gym environment.
    """
    env.env.set_subenv(instance_index)

    # Rollout itself does not invoke any random functions,
    # but agents and environments might. The policy classes used in
    # RLLab use np.random for their randomness, so the following
    # is sufficient to make that reproducible.
    np.random.seed(seed)

    #return rollout(env, policy, max_path_length=max_path_length)
    return rollout_record(env, policy, max_path_length=max_path_length)


def test_policy(policy, test_env, num_repeats):
    #num_specs = len(test_env.env.specs)
    num_specs = test_env.env.num_subenvs
    final_rewards = np.zeros((num_specs,num_repeats))
    correct_class = np.zeros((num_specs,num_repeats))
    steps = np.zeros((num_specs,num_repeats))
    for instance_index in range(num_specs):
        for repeat_index in range(num_repeats):
            out = seeded_rollout(test_env, instance_index, policy, 512+repeat_index)
            final_rewards[instance_index, repeat_index] = out["rewards"][-1]
            correct_class[instance_index, repeat_index] = out["rewards"][-1] > 0
            steps[instance_index, repeat_index] = len(out["rewards"])-1

    return {"correct_class": correct_class, "steps": steps, "final_rewards": final_rewards}


from sys import stdout

def test_policy_record(policy, test_env, num_repeats, record, output_dir, 
    output_prefix='', bundle_trajectories=False):
    """Runs the specified policy on the given environment, thereby recording
    information about the performed trajectories."""

    trajs_to_write = []

    # TODO: make this invariant to test data set (dirs etc.)!
    num_specs = test_env.env.num_subenvs
    final_rewards = np.zeros((num_specs,num_repeats))
    correct_class = np.zeros((num_specs,num_repeats))
    steps = np.zeros((num_specs,num_repeats))
    gt_labels = np.zeros(num_specs)
    for instance_index in range(num_specs):
        stdout.write("\rtest_policy_record testing %s of %s" % (instance_index+1, num_specs))
        stdout.flush()
        env_renders_instance = []
        self_poss = []
        eye_poss = []
        actions = []
        probs = []
        
        for repeat_index in range(num_repeats):
            out = seeded_rollout(test_env, instance_index, policy, 512+repeat_index)

            final_rewards[instance_index, repeat_index] = out["rewards"][-1]
            correct_class[instance_index, repeat_index] = out["rewards"][-1] > 0
            steps[instance_index, repeat_index] = len(out["rewards"])-1

            # FIXME: get rid of repeated overwriting
            if out.has_key("gt_label"):
                gt_labels[instance_index] = out["gt_label"][-1]

            # max/average image trace and agent's (self's) trajectory (positions)
            # except 'actions', the other fields are the result of calling describe(.)
            # on the environment both initially and after each step(.), so there cardinality
            # of each field should be one more than that of 'actions'
            if record:
                has_rgb = out.has_key("rgb")
                has_self_pos = out.has_key("self_pos")
                has_actions = out.has_key("actions")
                has_keys = [has_rgb, has_self_pos, has_actions]
                if all(has_keys):
                    # sanity checks
                    assert out["rgb"].shape[0] == len(out["actions"]) + 1
                    assert len(out["self_pos"]) == len(out["actions"]) + 1
                    env_renders_instance.append(max_images(out["rgb"]))
                    self_poss.append(out["self_pos"])
                    actions.append(out["actions"])

                if out["agent_infos"].has_key("prob"):    
                    probs.append(out["agent_infos"]["prob"])

                # optional
                if out.has_key("eye_pos"):
                    eye_poss.append(out["eye_pos"])


        # memorize per instance
        if record:
            avg_image = average_images(tensor_utils.stack_tensor_list(env_renders_instance))
            actionss = tensor_utils.stack_tensor_list(actions)
            probss = tensor_utils.stack_tensor_list(probs)

            if hasattr(test_env, 'get_stats'):
                stats = test_env.get_stats()
            else:
                stats = {}

            trajs_to_write.append(
                {"self_positions": self_poss, "eye_positions": eye_poss,
                "avg_image": avg_image, "actions": actionss, "probs": probss,
                "stats": stats})

    if bundle_trajectories:
        bundle_fname = output_prefix + 'traj_bundle.pkl'
        write_object(bundle_fname, trajs_to_write)
    else:
        for instance_index, traj in enumerate(trajs_to_write):
            traj_fname = output_prefix + 'traj_instance_' + str(instance_index) + '.pkl'
            write_object(traj_fname, traj)

    print()
    return {"correct_class": correct_class, "steps": steps, "final_rewards": final_rewards, "gt_labels": gt_labels}



# Experiment setup and execution
# ------------------------------

from rllab.envs.gym_env import CappedCubicVideoSchedule
from rllab.envs.gym_env import FixedIntervalVideoSchedule
from rllab.envs.gym_env import NoVideoSchedule

def parse_video_schedule(video_schedule):
    if video_schedule == 'none':
        return NoVideoSchedule()
    elif video_schedule == 'cubic':
        return CappedCubicVideoSchedule()
    elif video_schedule == 'fixed':
        return FixedIntervalVideoSchedule(1)
    elif video_schedule[:6] == 'fixed-':
        return FixedIntervalVideoSchedule(int(video_schedule[6:]))
    else:
        print("Valid video schedules:",
              "  none:      no video recording",
              "  cubic:     capped cubic (record episodes whose number are"
              "               cubes less than 1000, then every 1000 episodes)",
              "  fixed:     every episode",
              "  fixed-<n>: every <n> episodes",
              sep='\n')
        raise Exception("Invalid video schedule: " + str(video_schedule))

def before_experiment(cfg, results):
    #initialize the parallel workers
    if cfg['num_cores'] > 1:
        from rllab.sampler import parallel_sampler
        parallel_sampler.initialize(cfg['num_cores'])

def cfg_pre_resolution_hook(orig_cfg):
    cfg = load_raw_cfg(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'run_policy.defaults'))
    new_keys = recursive_merge_into(cfg, orig_cfg, list_new_keys=True)

    undeclared_new_keys = []
    for k in new_keys:
        if k in cfg['user_defined']:
            continue
        if len(k)>=2 and k[:2] == "__":
            continue
        undeclared_new_keys.append(k)
    if len(undeclared_new_keys) > 0:
        raise Exception("Keys %s are neither declared in run_policy.defaults or user_defined!" % (undeclared_new_keys,))

    # If env is present, set {train, test, test_on_train}_env's specification from it.
    if 'env' in cfg and cfg['env'] is not None:
        if 'init' in cfg['env']:
            cfg['train_env']['init'] = cfg['env']['init']
            cfg['test_env']['init'] = cfg['env']['init']
            cfg['test_on_train_env']['init'] = cfg['env']['init']
            if 'kwargs' in cfg['env']:
                recursive_merge_into(cfg['train_env']['kwargs'], cfg['env']['kwargs'])
                recursive_merge_into(cfg['test_env']['kwargs'], cfg['env']['kwargs'])
                recursive_merge_into(cfg['test_on_train_env']['kwargs'], cfg['env']['kwargs'])
        else:
            print("WARNING: env does not have an init. If you have problems later, maybe that's why.")
            cfg['train_env'] = cfg['env']
            cfg['test_env'] = cfg['env']
            cfg['test_on_train_env'] = cfg['env']
        del cfg['env']

    if not cfg['train_trajectories']:
        cfg['test_on_train_env'] = None

    required = {'baseline', 'policy', 'algo', 'dataset', 'train_env', 'test_env'}
    for name in required:
        if name not in cfg:
            raise Exception("Configuration must have {}.".format(name))
        elif not isinstance(cfg[name], collections.Mapping) or 'init' not in cfg[name]:
            # experiments/run_policy.defaults fills in various kwargs which it
            # cannot do if the object is not an init.  
            print("WARNING: %s does not have an init. If you have problems later, maybe that's why." % (name,))

    # FIXME: where is the right place to seed?
    np.random.seed(cfg["setup_np_random_seed"])

    if cfg['force_delete_all']:
        set_delete_all(True)

    return cfg


# TODO: cross-validation by using StratifiedShuffleSplit directly?
def split_dataset(dataset, test_size, test_train_seed, reduce_training_size=None):
    if dataset is None:
        return {'train_dataset': None, 'test_dataset': None}
    train_specs, test_specs, train_labels, test_labels = train_test_split(
        dataset["specs"], dataset["labels"], stratify=dataset["labels"],
        test_size=test_size, random_state=test_train_seed)

    # Reduce the training set size without affecting the test/train split.
    if reduce_training_size is not None:
        if reduce_training_size > len(train_specs):
            raise Exception("Cannot reduce training set size from %s to %s"
                % (len(train_specs), reduce_training_size))

        _, train_specs, _, train_labels = train_test_split(
            train_specs, train_labels, stratify=train_labels,
            test_size=reduce_training_size, random_state=test_train_seed)    

    split = {
        'train_dataset': {'specs': train_specs, 'labels': train_labels},
        'test_dataset': {'specs': test_specs, 'labels': test_labels}
    }
    uv_train, uc_train = np.unique(train_labels, return_counts=True)
    uv_test, uc_test = np.unique(test_labels, return_counts=True)
    print("training examples", len(train_specs))
    print("training labels", uv_train)
    print("training counts", uc_train)
    print("test examples", len(test_specs))
    print("test labels", uv_test)
    print("test counts", uc_test)
    return split


def cfg_post_object_hook(name, obj, cfg):
    if name == 'train_env':
        if obj is None:
            return None
        obj = GymEnv(obj, video_schedule=cfg['train_video'])
    elif name == 'test_env':
        if obj is None:
            return None
        obj = GymEnv(obj, video_schedule=cfg['test_video'])
    elif name == 'test_on_train_env':
        if obj is None:
            return None
        obj = GymEnv(obj, video_schedule=cfg['test_video'])
    elif name == 'test_video' or name == 'train_video':
        if obj is None:
            return None
        obj = parse_video_schedule(obj)
    elif name == 'stage_setup_fn':
        # Fold kwargs into the stage_init function, if there is one
        if obj is None:
            obj = lambda setup, stage_idx: None
        else:
            stage_init_kwargs = cfg["stage_init_kwargs"]
            obj = lambda setup, stage_idx: \
                obj(setup, stage_idx, **stage_init_kwargs)

    return obj


def execute_experiment_iteration(cfg, results):
    for stage_idx in range(cfg["num_stages"]):
        print("stage %s of %s" % (stage_idx, cfg["num_stages"]))
        output_dir = cfg.get('output_dir', None)
        if output_dir is not None:
            output_dir = os.path.join(output_dir, str(stage_idx))
        if '__final_output_dir_idx' in cfg:
            cfg['__final_output_dir_idx_stage'] = os.path.join(
                cfg['__final_output_dir_idx'], str(stage_idx))

        # HACK: annotate policy with its name for proper loading later
        output_prefix = os.path.join(output_dir, "policy.pkl")
        safe_annotate(cfg["policy"], _concepts101_ref="+%s" % (output_prefix,))
        
        # If no_rerun, skip if this iteration is already completed
        any_incomplete = False
        if cfg['train']:
            any_incomplete |= not exists_object('policy.pkl', output_dir)
        if cfg['test']:
            any_incomplete |= not exists_object('test_results.pkl', output_dir)
        
        if cfg.get('__no_rerun', False):
            if not any_incomplete:
                print("*** Already complete, will not re-run ***")
                continue

        resumed_previous_run = False
        if cfg.get('resume_on_restart', False): # True
            if '__final_output_dir' in cfg:
                resume_dir = cfg['__final_output_dir_idx_stage']
                print("*** Looking for partial run to resume (in final output dir %s) ***" % (resume_dir,))
            else:
                resume_dir = output_dir
                print("*** Looking for partial run to resume (in output dir %s) ***" % (resume_dir,))
            
            # Check whether there's an incomplete run in resume_dir.
            itr_file_pat = re.compile(r'itr_([0-9]*)\.pkl')
            max_itr = -1
            if os.path.exists(resume_dir):
                for f in os.listdir(resume_dir):
                    m = itr_file_pat.match(f)
                    if m is not None:
                        max_itr = max(max_itr, int(m.groups()[0]))
                if max_itr >= 0:
                    if resume_dir != output_dir:
                        # When resuming into different dir, copy over previous data
                        print("*** Copying over previous data from previous run (from %s to %s)"
                                    % (resume_dir, output_dir))

                        if os.path.exists(output_dir):
                            print("*** Clearing existing dir")
                            shutil.rmtree(output_dir)
                        os.makedirs(output_dir)
                        for f in sorted(os.listdir(resume_dir)):
                            print("*** Copying %s" % (f,))
                            shutil.copyfile(os.path.join(resume_dir, f),
                                            os.path.join(output_dir, f))

                    # Only resume if rllab.txt and rllab.csv and can be copied,
                    # otherwise the training information is inaccurate.
                    continue_resume = True
                    for f in ['rllab.txt', 'rllab.csv']:
                        fpath = os.path.join(output_dir, f)
                        if not os.path.exists(fpath):
                            print("*** Error: %s not found; canceling resume, beginning from start" % (fpath,))
                            continue_resume = False

                    if continue_resume:
                        itr_file = 'itr_%d.pkl' % (max_itr,)
                        print("*** Resuming from iteration found in %s ***" % (itr_file,))
                        
                        # Load stored policy 
                        snapshot = joblib.load(os.path.join(resume_dir, itr_file))
                        if snapshot['itr'] != max_itr:
                            print("*** Snapshot invalid (wrong itr number)! Starting from scratch. ***")
                        else:
                            cfg['policy'] = snapshot['policy']
                            cfg['baseline'] = snapshot['baseline']
                            new_algo = cfg['algo']
                            cfg['algo'] = snapshot['algo']  
                            cfg['algo'].n_itr = new_algo.n_itr  # only read changes to n_itr
                            cfg['algo'].current_itr = max_itr+1
                            # train env may not be pickleable, so don't reload it
                            resumed_previous_run = True

                            cfg['algo'].policy = cfg['policy']
                            cfg['algo'].baseline = cfg['baseline']
                            cfg['algo'].env = cfg['train_env']
                else:
                    print("*** Nothing to resume, training from scratch (no intermediate iterations found) ***")
            else:
                print("*** Nothing to resume, training from scratch (output directory not created yet) ***")

            # TODO: persist and resume random state

        if output_dir is not None:
            final_output_dir = cfg.get('__final_output_dir_idx_stage', None)
            setup_output(output_dir, clean=not resumed_previous_run, 
                final_output_dir=final_output_dir)

        if cfg['gym_logging']:
            cfg["train_env"].set_log_dir(os.path.join(get_output_dir(), "gym_train"))
            cfg["test_env"].set_log_dir(os.path.join(get_output_dir(), "gym_test"))

        cfg["stage_setup_fn"](cfg, stage_idx)

        init_policy = copy.deepcopy(cfg['policy'])

        start_time = time.time()
        if cfg['train']:
            if cfg["train_np_random_seed"] is not None:
                np.random.seed(cfg["train_np_random_seed"]) # FIXME: is this the right thing to do?

            if cfg["algo"].n_itr > 0:
                cfg["algo"].train()

            write_object("env.pkl", cfg["train_env"])
            write_object("baseline.pkl", cfg["baseline"])
            write_object("policy.pkl", cfg["policy"])

            if hasattr(cfg["train_env"], 'get_stats'):
                write_object("train_env_stats.pkl", cfg["train_env"].get_stats())
        time_train = time.time() - start_time
        print("Train time: %ds" % (time_train,))

        
        # Collect policies to test on training & testing datasets, if not already
        # specified in the configuration file
        if cfg['to_test_on_test'] is not None:
            to_test_on_test = cfg['to_test_on_test']
            to_test_on_test_decided = True
        else:
            to_test_on_test = []
            to_test_on_test_decided = False
        if cfg['to_test_on_train'] is not None:
            to_test_on_train = cfg['to_test_on_train']
            to_test_on_train_decided = True
        else:
            to_test_on_train = []
            to_test_on_train_decided = False

        if cfg['test'] and not to_test_on_test_decided:
            if cfg["train"]:
                policy = cfg["policy"]
            else:
                policy = read_object("policy.pkl")
            to_test_on_test.append(('final', policy))

        if cfg['train'] and cfg['select_policies']:
            train_csv_filepath = os.path.join(output_dir, 'rllab.csv')
            train_csv_data = np.genfromtxt(train_csv_filepath, delimiter=',', names=True)
            train_csv_data = train_csv_data.reshape((-1,)) # Needed if there's only one iteration

            argmin_perplexity = np.argmin(train_csv_data['Perplexity'])
            min_perplexity = train_csv_data['Perplexity'][argmin_perplexity]
            argmax_return = np.argmax(train_csv_data['AverageReturn'])
            max_return = train_csv_data['AverageReturn'][argmax_return]

            # Save other distinguished policies to their own files
            itrs_to_save = [
                ("max_return", argmax_return,
                    "it has maximum return of %.4f" % (max_return,)),
                ("min_perplexity", argmin_perplexity,
                    "it has minimum perplexity of %.4f" % (min_perplexity,)),
            ]
            for name, itr, reason in itrs_to_save:
                print('*** Saving iteration %d into %s because %s' 
                            % (itr, name, reason))
                itr_policy = read_object("itr_" + str(itr) + ".pkl")['policy']
                write_object("policy_" + name + ".pkl", itr_policy)
                if not to_test_on_test_decided:
                    to_test_on_test.append((name, itr_policy))
                if cfg['train_trajectories'] and not to_test_on_train_decided:
                    to_test_on_train.append((name, itr_policy))
     
        if cfg['train_trajectories'] and not to_test_on_train_decided:
            # load previously stored training information (for number of iterations)
            train_csv_filepath = os.path.join(output_dir, 'rllab.csv')
            train_csv_data = np.genfromtxt(train_csv_filepath, delimiter=',', names=True)
            train_csv_data = train_csv_data.reshape((-1,)) # Needed if there's only one iteration
            n_itr = train_csv_data.shape[0]

            if cfg['num_train_trajectories'] == 0:
                itrs_to_record = range(n_itr)
            elif cfg['num_train_trajectories'] == 1:
                itrs_to_record = [n_itr-1]
            else:
                # n equally spaced points in (0, 1)
                n = float(cfg['num_train_trajectories']-1)
                pts = (0.5 + np.arange(n))/n

                # n equally spaced within the interval [0, ..., n_itr-2]
                itrs_to_record = [int(x) for x in (n_itr-1)*pts] + [n_itr-1]
                itrs_to_record = sorted(set(itrs_to_record))

            to_test_on_train.append(('init', init_policy))
            for itr in itrs_to_record:
                policy = read_object("itr_" + str(itr) + ".pkl")['policy']
                to_test_on_train.append(('itr_' + str(itr), policy))


        # Test policies on training & testing datasets
        print('\n*** Testing %s policies: %s' % 
                (len(to_test_on_test), 
                 ", ".join([name for name, _ in to_test_on_test])))
        start_time = time.time()
        overall_means = {}
        for policy_idx, (name, policy) in enumerate(to_test_on_test):
            start_time2 = time.time()
            print('\n*** [%d of %d] Testing %s policy on test data ***\n'
                % (policy_idx+1, len(to_test_on_test), name))
            
            # TODO: what is the right way to render complex policies in complex environments? 
            if cfg["render_policy"]:
                cfg["test_env"].env.render_policy(policy)

            if name == 'final':
                output_prefix = ''
                result_file = 'test_results.pkl'
                bundle_trajectories = False
            else:
                output_prefix = name + "_test_"
                result_file = name + '_test_results.pkl'
                bundle_trajectories = True

            test_results = test_policy_record(policy, cfg["test_env"], 
                cfg["test_repeats"], cfg["test_trajectories"], 
                cfg["output_dir"], output_prefix=output_prefix,
                bundle_trajectories=bundle_trajectories)

            write_object(result_file, test_results)

            test_size = len(cfg['test_dataset']['specs'])
            train_size = len(cfg['train_dataset']['specs'])
            print("\nTest size %s (%.3f) train size %s (%.3f)\n" 
                    % (test_size, test_size/float(test_size+train_size), 
                       train_size, train_size/float(test_size+train_size)))
            
            print("Test instances:")
            for idx in range(test_size):
                print("%s\t%s\t%s" % (idx, cfg['test_dataset']['labels'][idx],
                                           cfg['test_dataset']['specs'][idx]))

            correct_class = test_results["correct_class"]
            overall_means[name] = correct_class.mean()

            print("\nTest classification results (all):\n", correct_class)
            print("\nTest classification results (avg):\n", correct_class.mean(axis=1, keepdims=True))
            print("\nTest classification results (overall):", correct_class.mean(), "\n\n")
            print('*** [%d of %d] Test time: %ds'
                % (policy_idx+1, len(to_test_on_test), time.time() - start_time2))
        time_test = time.time() - start_time
        print("*** Total test time: %ds\n" % (time_test,))

        for name in overall_means.keys():
            print("\nTest classification results(%s): %0.2f" % (name, overall_means[name]) )


        print('\n*** Testing %s policies on training set: %s' % 
                (len(to_test_on_train), 
                    ", ".join([name for name, _ in to_test_on_train])))
        start_time = time.time()
        for policy_idx, (name, policy) in enumerate(to_test_on_train):
            start_time2 = time.time()
            print('*** [%d of %d] Testing %s policy on train data ***'
                % (policy_idx+1, len(to_test_on_train), name))

            output_prefix = name + "_"
            train_results = test_policy_record(policy, cfg["test_on_train_env"],
                cfg["test_repeats"], cfg["train_trajectories"], 
                cfg["output_dir"], output_prefix=output_prefix, 
                bundle_trajectories=True) 
            write_object(output_prefix + "train_results.pkl", train_results)
            
            print("*** [%d of %d] Test on training time: %ds" % 
                (policy_idx+1, len(to_test_on_train), time.time() - start_time2,))
        time_test_on_train = time.time() - start_time
        print("*** Total test on training time: %ds\n" % (time_test_on_train,))

        # Record timing information
        write_object('timing.pkl', {
            'process_cfg': cfg['__time_process_cfg'],
            'train': time_train,
            'test': time_test,
            'test_on_train': time_test_on_train,
        })

        if output_dir is not None:
            shutdown_output()
