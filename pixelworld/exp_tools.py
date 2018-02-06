"""Various code used to manipulate experimental configurations or used within 
metaexperiment files."""

from __future__ import print_function
from __future__ import absolute_import

import collections
from collections import defaultdict
import copy
import glob
from itertools import combinations, permutations, product
import os

import joblib
import numpy as np
from sklearn.feature_selection import mutual_info_regression

from pixelworld.misc import recursive_merge_into
from pixelworld.expcfg import (recursively_resolve_refs, safe_annotate, 
    create_objects, make_file_ref)
from pixelworld import logger

def merge_cfgs(*cfgs):
    """Merges the given cfgs, the keys in later cfgs overriding those in earlier
    ones."""
    out_cfg = {}
    for cfg in cfgs:
        recursive_merge_into(out_cfg, cfg)
    return out_cfg


def generate_combinations(cfgs, *component_name_lists):
    """Generates all possible combinations of configurations, marking them
    as executable."""
    for lst in component_name_lists:
        for name in lst:
            if name not in cfgs:
                raise Exception("Configuration %s not found." % (name,))

    for component_names in product(*component_name_lists):
        cfg = merge_cfgs(*[cfgs[name] for name in component_names])
        cfg['__execute__'] = True
        # # The following created a sub-directory structure:
        # name = "/".join(component_names)
        name = "_".join(component_names)
        yield name, cfg



def recursively_format_strings(x, bindings):
    if isinstance(x, basestring):
        return x.format(**bindings)
    elif isinstance(x, collections.Mapping):
        return {k: recursively_format_strings(v, bindings) for k, v in x.items()}
    elif isinstance(x, collections.Sequence):
        return [recursively_format_strings(xi, bindings) for xi in x]
    else:
        return x


def destructuring_bind(form, value, bindings=None, debug=False):
    """Bind form to value, returning a dict of bindings."""
    bindings = {} if bindings is None else bindings
    if debug:
        print('BINDING: %s <- %s' % (form, value))

    if isinstance(form, basestring):
        bindings[form] = value
    elif isinstance(form, collections.Mapping):
        if len(form) != len(value):
            raise Exception("Form must have the same number of items as value.")
        else:
            for (key_form, val_form), (key_value, val_value) in zip(form.items(), value.items()):
                bindings = destructuring_bind(key_form, key_value, bindings)
                bindings = destructuring_bind(val_form, val_value, bindings)
    elif isinstance(form, collections.Sequence):
        if len(form) < len(value):
            raise Exception("Form must be at least as long as value.")
        else:
            for idx, sub_form in enumerate(form):
                sub_value = value[idx] if idx < len(value) else None
                bindings = destructuring_bind(sub_form, sub_value, bindings)
    else:
        raise Exception("Invalid form.")
    return bindings


def generate_pattern(cfgs, vals=None, bindings={}, key=None, keys=None,
    **kwargs):
    """Generates configurations following a pattern.

    Creates a pattern for each assignment of a sequence of values to either a
    single key or a nested structure of keys. Additional kwargs of the form
    loop<suffix> themselves contains key, keys, and/or vals. Assignments are
    formed from all possible combinations.

    Each pair of keys 'name<suffix>' and 'cfg<suffix>' are used to generate an
    experiment, independently.

    Vals must either be a list of a dictionary.

    When processing, first name and all the strings of cfg are formatted by 
    format, so that {key} is mapped to val. Next the cfg has any internal 
    references of the form +@key resolved to val. These two steps allow both 
    macro-like string substitution and the insertion of general values. Note 
    that any references which should remain after the pattern has been generated
    need to be escaped, e.g. use ++blah instead +blah.

    If supplied, 'bindings' is used to initialize the bindings, and is resolved
    using cfgs.
    If supplied, 'let<suffix>' is added to the bindings after each loop assignment is
    folded into the bindings as described previously. Let is resolved using
    bindings.

    If supplied, 'if<suffix>' keys are evaluated to determine whether an
    experiment should be generated. Each should be a dictionary containing 
    'func' containing a function and (optionally) 'kwargs' to be passed in.
    This is evaluated in same binding context as experiment cfg file (i.e., 
    after loops and let have be processed). All such keys must succeed.

    Both vals and bindings have their references resolved in the metaexperiment
    configuration before substitution.
    """
    if (key is not None) and (keys is not None):
        raise Exception("Cannot specify both key or keys.")

    if vals is not None:
        if (key is not None) and (keys is None):
            raise Exception("If you specify vals you must specify exactly one of key or keys.")

    if (key is not None) or (keys is not None):
        if vals is None:
            raise Exception("If you specify key or keys you must specify vals.")

    loop_forms = []
    loop_vals = []
    if_forms = []
    name_forms = {}
    cfg_forms = {}
    lets = []

    # Parse all keys with suffixes: loops, ifs, names, cfgs
    if vals is not None:
        loop_forms.append(key if keys is None else keys)
        loop_vals.append(recursively_resolve_refs(vals, cfgs))
    for k in sorted(kwargs):
        v = kwargs[k]
        if len(k) >= 4 and k[:4] == 'loop':            
            key = v.get('key', None)
            keys = v.get('keys', None)
            vals = v.get('vals', None)
            if not (key is None) ^ (keys is None):
                raise Exception("Loop %s must specify exactly one of key or keys." %
                    (k,))
            if vals is None:
                raise Exception("Loop %s must specify vals." %(k,))
            loop_forms.append(key if keys is None else keys)
            loop_vals.append(recursively_resolve_refs(vals, cfgs))
        elif len(k) >= 2 and k[:2] == 'if':
            if_forms.append(v)
            if 'func' not in v or not set(v.keys()) <= set(['func', 'args', 'kwargs']):
                raise Exception("Invalid if form: " + str(v))
        elif len(k) >= 4 and k[:4] == 'name':
            name_forms[k[4:]] = v
        elif len(k) >= 3 and k[:3] == 'cfg':
            cfg_forms[k[3:]] = v
        elif len(k) >= 3 and k[:3] == 'let':
            lets.append(v)
        else:
            raise Exception("Invalid generate_pattern kwarg " + k)

    if set(name_forms.keys()) != set(cfg_forms.keys()):
        raise Exception("names (%s) and cfgs (%s) don't align!" %
            (", ".join(sorted(name_forms.keys())),
             ", ".join(sorted(cfg_forms.keys()))))

    for idx in range(len(loop_vals)):
        if isinstance(loop_vals[idx], collections.Mapping):
            loop_vals[idx] = [{k: v} for k, v in loop_vals[idx].items()]
        elif not isinstance(loop_vals[idx], collections.Sequence):
            raise Exception("vals must be either a dict or a list.")

    base_bindings = recursively_resolve_refs(bindings, cfgs)

    for all_vals in product(*loop_vals):
        bindings = copy.deepcopy(base_bindings)
        for form, val in zip(loop_forms, all_vals):
            # print('%s = %s' % (form, val))
            bindings = destructuring_bind(form, val, bindings)

        # Process and add let bindings
        for let in lets:
            this_let = recursively_format_strings(let, bindings)
            this_let = recursively_resolve_refs(this_let, bindings)
            this_let = create_objects(this_let)
            for k,v in this_let.items():
                bindings[k] = v

        # Check ifs; skip if any fail
        skip = False
        for if_form in if_forms:
            this_if = recursively_format_strings(if_form, bindings)
            this_if = recursively_resolve_refs(this_if, bindings)
            if not this_if['func'](*this_if.get('args', []), **this_if.get('kwargs', {})):
                skip = True
                break
        if skip:
            continue

        for exp_k in name_forms.keys():
            name = name_forms[exp_k]
            cfg = cfg_forms[exp_k]

            this_name = name.format(**bindings)
            this_cfg = recursively_format_strings(cfg, bindings)
            this_cfg = recursively_resolve_refs(this_cfg, bindings)
            for k2, v2 in this_cfg.items():
                if isinstance(v2, collections.Mapping) and '__init__' in v2:
                    v2 = v2['__init__'](*v2.get('__args__',[]), **v2.get('__kwargs__',{}))
                    this_cfg[k2] = v2

            yield this_name, this_cfg


def load_policy(path, experiment, index, env_spec=None, name=None):
    filename = "policy.pkl" if name is None else "policy_" + name + ".pkl" 
    filepath = os.path.join(path, experiment, str(index), "0", filename)

    policy_ref = make_file_ref(filepath)
    print("load_policy:", policy_ref)    
    policy = recursively_resolve_refs(policy_ref)
    print("   policy:", policy)
    return policy

def load_policies(path, names, name_prefix=None, name_prefixes=None):
    if name_prefix is not None:
        if name_prefixes is not None:
            raise Exception("Cannot specify both name_prefix and name_prefixes")
        exp_paths = [os.path.join(path, name_prefix + name) for name in names]
    elif name_prefixes is not None:
        exp_paths = [os.path.join(path, prefix + name) 
                        for prefix, name in zip(name_prefixes, names)]
    else:
        exp_paths = [os.path.join(path, name) for name in names]

    policy_filenames = [os.path.join(exppath, "*", "0", "policy.pkl") for exppath in exp_paths]
    policy_ref_list = [make_file_ref(filename) for filename in policy_filenames]
    print("load_policies:", policy_ref_list)
    policies = recursively_resolve_refs(policy_ref_list)
    print("   policies:", policies)
    return policies


def load_best_policies(path, names, name_prefix=None, name_prefixes=None):
    import joblib
    import numpy as np

    if name_prefix is not None:
        if name_prefixes is not None:
            raise Exception("Cannot specify both name_prefix and name_prefixes")
        exp_paths = [os.path.join(path, name_prefix + name) for name in names]
    elif name_prefixes is not None:
        exp_paths = [os.path.join(path, prefix + name) 
                        for prefix, name in zip(name_prefixes, names)]
    else:
        exp_paths = [os.path.join(path, name) for name in names]


    out = []
    for path in exp_paths:
        policy_ref = "+%s/*/0/policy.pkl" % (path,)
        policies = recursively_resolve_refs([policy_ref])[0]
        candidates = []
        for policy in policies:
            path2 = policy._concepts101_filepath
            parts = path2.split('/')
            assert(parts[-1] == 'policy.pkl')
            assert(parts[-2] == '0')
            f = ('/'.join(parts[:-1] + ['test_results.pkl']))
            print(path2)
            print(f)
            test_results = joblib.load(open(f))
            qual = np.mean(test_results['correct_class'])
            candidates.append((qual, policy))
        candidates.sort()
        out.append(candidates[-1][1])
        
    return out


def filter_max_unary_value(policies, value_fn, filter_fn, k=0, threshold=float('-inf')):
    """Return best k, taking at most one from each index. For k=0, takes exactly
    one from each index. If a threshold is given, only include policies with
    value >= the threshold."""

    # policy names are of the form: (index, stage, iteration_name)
    indexes = sorted(set([index for index, _, _ in policies]))
    all_name_values = []
    for index in indexes:
        value_names = [(value_fn(props), name) 
                       for name, props in policies.items() 
                       if name[0] == index and filter_fn(props)]
        value_names = [(v, n) for v,n in value_names if v >= threshold]
        value_names.sort(key=lambda (v,n): -v)
        if len(value_names) > 0:
            all_name_values.append(value_names[0])
    all_name_values.sort()

    if k == 0:
        k = len(all_name_values)
    k = min(k, len(all_name_values))
    out = [name for _, name in all_name_values[:k]]
    return out


def filter_max_training_reward(policies, **kwargs):
    filter_fn = lambda props: 'train_data' in props
    value_fn = lambda props: props['train_data']['AverageReturn']
    return filter_max_unary_value(policies, value_fn, filter_fn, **kwargs)

def filter_min_perplexity(policies, **kwargs):
    filter_fn = lambda props: 'train_data' in props
    value_fn = lambda props: -props['train_data']['Perplexity']
    return filter_max_unary_value(policies, value_fn, filter_fn, **kwargs)



FILTERS = {
    'none': lambda policies: policies.keys(),
    None: lambda policies: policies.keys(),
    'final': lambda policies: [p for p in policies if p[2] == 'final'],
    'max_training_reward': filter_max_training_reward,
    'min_perplexity': filter_min_perplexity,
}

def load_policies_filtered(home_dir, experiment=None, experiments=None, **kwargs):
    assert (experiment is None) ^ (experiments is None), "Must specify experiment or experiments"
    if experiments is None:
        return load_policies_filtered_one_experiment(home_dir, experiment, **kwargs)
    else:
        if len(experiments) == 0:
            return []
        policies = []
        for experiment in experiments:
            policies += load_policies_filtered_one_experiment(home_dir, experiment, **kwargs)
        if len(policies) == 0:
            logger.log('load_policies_filtered: no policies selected! (home_dir %s experiment %s experiments %s kwargs %s)'
                % (home_dir, experiment, experiments, kwargs))
        return policies

def load_policies_filtered_one_experiment(home_dir, experiment, filter='none', 
    filter_args=[], filter_kwargs={}, quiet=False, 
    exception_if_no_policies_found=True, exception_if_run_not_complete=False,
    exception_if_all_policies_filtered=False):
    """Load policies within (recursively) the directory 

        <home_dir>/<experiment>     if home_dir is not None
        <experiment>                otherwise

    selecting only the subset of names returned by the given filter. The policies
    found under this directory are named:

        (index, stage, iteration_name)

    and are mapped to a dictionary of properties of that policy. A filter is
    a function mapping this dictionary to a subsets of names.

    The policy's property dictionary may include:
    
    'train_data': a 1-d numpy array originating from 'rllab.csv', with one row 
        per training iteration, containing a structured datatype with, at least,
        the fields (accessible by array[idx][field] for fields found in 
        array.dtype.names):
            'MaxReturn'
            'LossAfter'
            'TimeOptimizer'
            'MeanKL'
            'AverageLength'
            'TimeIter'
            'AverageReturn'
            'StdLength'
            'AverageDiscountedReturn'
            'TimeSampler'
            'NumTrajs'
            'Iteration'
            'MinReturn'
            'StdReturn'
            'ExplainedVariance'
            'MinLength'
            'TimeOther'
            'dLoss'
            'LossBefore'
            'Entropy'
            'MaxLength'
            'Perplexity'

    'train_results', 'test_results': (optional) information about testing on
        the training set or testing set, respectively, loading from a file
        <name>_train_results.pkl or <name>_test_results.pkl.
        This contains a dictionary with (at least) the keys:
            'gt_labels' : array(num_test_examples,)
                0 or 1 for the ground truth class of the example
            
            'steps' : array(num_test_examples, num_test_repeats)
                number of steps taken in episode
            
            'correct_class' : array(num_test_examples, num_test_repeats)
                0 or 1 whether classification was correct
            
            'final_rewards' : array(num_test_examples, num_test_repeats)
                the reward of the final step of each episode (typically +-1)
    """
    if not quiet:
        print('load_policies_filtered  home_dir: %s experiment: %s filter: %s' 
            % (home_dir, experiment, filter))
    policies = defaultdict(dict)
    if home_dir is None or len(experiment) >= 4 and experiment[:4] == 'out/':
        output_dir = experiment
    else:
        output_dir = os.path.join(home_dir, experiment)

    if not os.path.exists(output_dir):
        msg = ("load_policies_filtered_one_experiment: output dir %s doesn't exist!" \
               " (home_dir %s experiment %s filter %s filter_args %s filter_kwargs %s)"
               % (output_dir, home_dir, experiment, filter, filter_args, filter_kwargs))
        if exception_if_no_policies_found:
            raise Exception(msg)
        else:
            logger.log(msg)
            return []

    if filter in FILTERS:
        filter = FILTERS[filter]
        if not quiet:
            print('+++ recognized filter %s from bank' % (filter,))

    # Load policy features
    # Loop over indices 
    run_complete = False
    for idx_str in os.listdir(output_dir):
        #make sure we got an index
        if not idx_str.isdigit(): continue
        
        idx_dir = os.path.join(output_dir, idx_str)
        if not os.path.isdir(idx_dir):
            continue
        idx = int(idx_str)
        if os.path.isdir(idx_dir):
            # Loop over stages
            for stage_str in os.listdir(idx_dir):
                #make sure we got a stage
                if not stage_str.isdigit(): continue
                
                stage_dir = os.path.join(idx_dir, stage_str)
                stage = int(stage_str)

                policy_file = os.path.join(stage_dir, 'policy.pkl')
                if not os.path.exists(policy_file):
                    msg = ("load_policies_filtered_one_experiment: policy %s doesn't exist! is run complete?" \
                           " (home_dir %s experiment %s filter %s filter_args %s filter_kwargs %s)"
                           % (policy_file, home_dir, experiment, filter, filter_args, filter_kwargs))
                    if exception_if_run_not_complete:
                        raise Exception(msg)
                    else:
                        logger.log(msg)
                        continue

                train_csv_data = []
                train_csv_filepath = os.path.join(stage_dir, 'rllab.csv')
                if os.path.getsize(train_csv_filepath) > 0:
                    train_csv_data = np.genfromtxt(train_csv_filepath, delimiter=',', names=True)
                    train_csv_data = train_csv_data.reshape((-1,))

                for itr_index, train_data in enumerate(train_csv_data):
                    # make sure to get the iteration from the file in order to handle
                    # resumed runs correctly
                    itr = int(train_data['Iteration'])
                    name = 'itr_' + str(itr) # + str(itr)
                    policies[(idx, stage, name)]['train_data'] = train_data

                for f in os.listdir(stage_dir):
                    if len(f) > 18 and f[-18:] == '_train_results.pkl':
                        name = f[:-18]
                        if name == 'init':
                            continue
                        results = joblib.load(os.path.join(stage_dir, f))
                        policies[(idx, stage, name)]['train_results'] = results

                    if len(f) > 17 and f[-17:] == '_test_results.pkl':
                        name = f[:-17]
                        if name == 'init':
                            continue
                        results = joblib.load(os.path.join(stage_dir, f))
                        policies[(idx, stage, name)]['test_results'] = results

                # Load final policy's test results into policy 'final'
                # make sure to get the iteration from the file in order to handle
                # resumed runs correctly
                max_itr = int(max(train_csv_data['Iteration']))
                final_itr = "itr_" + str(max_itr) # str(len(train_csv_data)-1)
                policies[(idx, stage, 'final')] = policies[(idx, stage, final_itr)]
                del policies[(idx, stage, final_itr)]

                test_results_filepath = os.path.join(stage_dir, 'test_results.pkl')
                if os.path.exists(test_results_filepath):
                    results = joblib.load(test_results_filepath)
                    policies[(idx, stage, 'final')]['test_results'] = results

    if len(policies) == 0:
        msg = ("load_policies_filtered_one_experiment: no policies found!" \
               " (home_dir %s experiment %s filter %s filter_args %s filter_kwargs %s)"
               % (home_dir, experiment, filter, filter_args, filter_kwargs))
        if exception_if_no_policies_found:
            raise Exception(msg)
        else:
            logger.log(msg)
            return []

    filtered_policies = filter(dict(policies), *filter_args, **filter_kwargs)
    if not quiet:
        print("+++ choosing %s of %s policies" % (len(filtered_policies), len(policies)))

    if len(filtered_policies) == 0:
        msg = ("load_policies_filtered_one_experiment: no policies selected!" 
               " (home_dir %s experiment %s filter %s filter_args %s filter_kwargs %s)"
               % (home_dir, experiment, filter, filter_args, filter_kwargs))
        if exception_if_all_policies_filtered:
            raise Exception(msg)
        else:
            logger.log(msg)
            return []

    # Load policies selected by the filter
    loaded_policies = []
    for idx, stage, name in filtered_policies:
        assert (idx, stage, name) in policies
        path = os.path.join(output_dir, str(idx), str(stage))
        if len(name) > 4 and name[:4] == 'itr_':
            filename = os.path.join(path, name + '.pkl')
            policy = joblib.load(filename)['policy']
        elif name == 'final':
            filename = os.path.join(path, 'policy.pkl')
            policy = joblib.load(filename)
        else:
            filename = os.path.join(path, 'policy_' + name + '.pkl')
            policy = joblib.load(filename)
        object_ref = '+' + filename
        print('+++ loaded policy (%s, %s, %s) from %s' % (idx, stage, name, object_ref))
        safe_annotate(policy, _concepts101_ref=object_ref)
        loaded_policies.append(policy)

    return loaded_policies


def write_to_file(obj, filename, path=None, overwrite=False):
    if path is not None:
        filename = os.path.join(path, filename)
    filename = os.path.abspath(filename)
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not overwrite and os.path.exists(filename):
        print("WARNING: file already exists %s; not overwriting." % (filename,))
        pass
        # Check to see whether same as one on disk?
        # When to overwrite?
    else:
        print("Writing to %s" % (filename,))
        joblib.dump(obj, filename)


# Special-case stuff
# ------------------

def circumfix_strings(vals, prefix="", postfix=""):
    return ["%s%s%s" % (prefix, val, postfix) for val in vals]

def apply_pattern(pattern, vals):
    return [pattern % args for args in vals]



def curr_is_not_redundant(curr_kind, prior_exp_names):
    """Compute whether a setting is redundant in curriculum learning."""
    if curr_kind in ['nocurr', 'test']:
        return len(prior_exp_names) > 0
    else:
        return True


def curr_compute_dependents(curr_kind, exp_name, prior_exp_names, curriculum, pattern):
    """Compute the dependent experiment names in curriculum learning."""
    option_deps = []
    test_deps = []
    if curr_kind == 'nocurr':
        pass
    elif curr_kind == 'curr':
        option_deps = [(exp, 'curr') for exp in prior_exp_names]
    elif curr_kind == 'nocurr2':
        if len(curriculum[exp_name][2]) == 0:
            #print("ZERO!", exp_name)
            option_deps = [(exp_name, 'curr')]
        else:
            #print("NONZERO!", exp_name)
            option_deps = [(exp_name, 'nocurr')]
    elif curr_kind == 'test':
        test_deps = [(exp, 'curr') for exp in prior_exp_names]
    else:
        raise Exception("Invalid curr_kind: %s" % (curr_kind,))        

    deps = {
        'all': [pattern % vals for vals in sorted(set(option_deps + test_deps))],
        'option': [pattern % vals for vals in option_deps],
        'test': [pattern % vals for vals in test_deps]
    }
    return deps


def list_exp_files(home_dir=None, experiment=None, output_dir=None, idx=None, stage=None):
    if output_dir is None:
        assert home_dir is not None and experiment is not None, "Specify either output_dir or home_dir and experiment"
        output_dir = os.path.join(home_dir, experiment)
    if home_dir is None or experiment is None:
        assert output_dir is not None, "Specify either output_dir or home_dir and experiment"

    if idx is None:
        idxes = [int(s) for s in os.listdir(output_dir) if s.isdigit()]
    else:
        idxes = [idx]

    for _idx in idxes:
        idx_dir = os.path.join(output_dir, str(_idx))
        if not os.path.isdir(idx_dir):
            logger.log("File %s in an experiment directory isn't a directory!" % (idx_dir,))
            continue

        if stage is None:
            stages = [int(s) for s in os.listdir(idx_dir) if s.isdigit()]
        else:
            stages = [stage]

        for _stage in stages:          
            stage_dir = os.path.join(idx_dir, str(_stage))
            if not os.path.isdir(stage_dir):
                logger.log("File %s in an experiment/index directory isn't a directory!" % (stage_dir,))
                continue
            for f in os.listdir(stage_dir):
                yield stage_dir, f


def get_dependent_experiments(prior_exp_names, this_variant, quiet=True):
    """Compute facts about dependent experiments.

    Each prior experiment name can take three forms:
        out/.*     
            external reference to the output directory of an experiment from
            another metaexperiment file
        .*/.*
            an internal experiment with variant specified
        [^/]*
            an internal experiment without variant specified; postfix with
            this_variant

    Returns a dictionary containing:
        'paths': 
            list of paths of directories corresponding to each prior_exp
            directories are relative to home_dir (i.e. the meta-experiment
            output dir) unless they being with 'out/', in which case they are
            relative to the concepts101 directory.
        
        'internal': 
            the full experiment name (with variant) of all experiments contained
            in this meta-experiment
    """
    paths = []
    internal = []

    for exp_name in prior_exp_names:
        if len(exp_name) >= 4 and exp_name[:4] == 'out/':
            paths.append(exp_name)
        elif '/' in exp_name:
            internal.append(exp_name)
            paths.append(exp_name)
        else:
            full_exp_name = '%s/%s' % (exp_name, this_variant)
            internal.append(full_exp_name)
            paths.append(full_exp_name)

    if not quiet:
        print('&&& get_dependent_experiments')
        print('    prior_exp_names:')
        for exp_name in prior_exp_names:
            tag = '[ext]' if len(exp_name) >= 4 and exp_name[:4] == 'out/' else '[int]'
            print('    %s %s' % (tag, exp_name))
        print('    internal:')
        for exp_name in internal:
            print('        %s' % (exp_name,))
        print('    paths:')
        for path in paths:
            print('        %s' % (path,))
            if not os.path.exists(path):
                print("            !!! directory doesn't exist!")
                continue
            if not os.path.isdir(path):
                print("            !!! exists but isn't a directory!")
                continue
            has_policy = False
            for _, filename in list_exp_files(output_dir=path):
                if filename == 'policy.pkl':
                    has_policy = True
            if not has_policy:
                print('            !!! contains no policy.pkl!')

    return {'paths': paths, 'internal': internal}
