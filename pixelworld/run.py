"""Run experiments defined within a metaexperiment file."""

from __future__ import print_function
from __future__ import absolute_import

import argparse
import copy
import os
from pprint import pprint
import random
import subprocess
import time
import traceback

from toposort import toposort_flatten

from pixelworld import logger
from pixelworld.expcfg import (load_raw_cfg, process_cfg, process_metacfg, 
                               instantiate_loops, apply_override_and_default)
from pixelworld.misc import IntervalTimer
import pixelworld.run_policy

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir')
    parser.add_argument('--force-delete-all', action='store_true')
    parser.add_argument('--test', action='store_true', help="Don't train, only test.")
    parser.add_argument('--expname', '-e')
    parser.add_argument('--no-rerun', '-n', action='store_true', 
        help="Don't rerun completed experiments.")
    parser.add_argument('--run-dependents', '-d', action='store_true', 
        help="Run all experiments the given experiments depend on")
    parser.add_argument('--only-new', action='store_true',
        help="Only run experiments whose directories don't yet exist")
    parser.add_argument('--resume', action='store_true',
        help="Resume incomplete experiments")

    parser.add_argument('--final-output-dir', 
        help="Final destination to copy files to.")
    parser.add_argument('--final-output-period', type=int, default=0,
        help="Period (in seconds) at which to sync files into final output dir."
        " Zero means don't sync until complete.")

    args, rest = parser.parse_known_args()

    if len(rest) == 0:
        print("Usage: python run.py <experiment_filepath> [<experiment_index>]"
              "  the optional <experiment_index> is either a single integer"
              "  or a range <start>-<end> (start inclusive, end exclusive)", sep='\n')
        import sys
        sys.exit(1)

    filename = rest[0]

    override_cfg = {}
    default_cfg = {
        'force_delete_all': args.force_delete_all,
        '__no_rerun': args.no_rerun,
        '__run_dependents': args.run_dependents,
        '__only_new': args.only_new,
        'resume_on_restart': args.resume,
    }

    if filename[:12] == 'experiments/':
        default_cfg['home_dir'] = os.path.join("out", filename[12:])

    if args.output_dir is not None:
        override_cfg['output_dir'] = args.output_dir    
    elif filename[:12] == 'experiments/':
        default_cfg['output_dir'] = os.path.join("out", filename[12:])
    else:
        raise Exception("Cannot determine output directory from experiment filename %s. Specify manually with --output_dir" % (filename,))

    if args.test:
        override_cfg['test'] = True
        override_cfg['train'] = False

    if args.expname is not None:
        default_cfg['__expname'] = args.expname

    if args.final_output_dir is not None:
        override_cfg['__final_output_dir'] = args.final_output_dir
        override_cfg['__final_output_period'] = args.final_output_period

    if len(rest) >= 2:
        if '-' in rest[1]:
            if len(rest[1].split('-')) != 2:
                print("Invalid experiment_index specification: too many -'s.")
                print("Experiment index must be of the form <num> or <num>-<num>.") 
                print("   experiment_index:", rest[1])
                raise Exception()
            index_start, index_end = rest[1].split('-')
            index_start = int(index_start)
            index_end = int(index_end)
        else:
            index_start = int(rest[1])
            index_end = index_start + 1
    else:
        index_start = 0
        index_end = None
    
    return filename, override_cfg, default_cfg, index_start, index_end  



def match_expnames(expnames, cfgs):
    expanded_expnames = []
    for index, expname in enumerate(expnames):
        if expname in cfgs:
            expanded_expnames.append(expname)    
        else:
            pattern = expname
            add_expnames = []
            for ename in cfgs.keys():
                if re.match(pattern, ename): 
                    add_expnames.append(ename)
            if len(add_expnames) == 0:
                raise Exception("Experiment %s (or matching pattern) not defined in meta-experiment." % (pattern,)) 
            expanded_expnames += add_expnames
    return expanded_expnames


def get_expnames_to_perform(expnames, output_dir, cfgs, depends=False, 
    run_dependents=False, no_rerun=False, only_new=False):

    if expnames is None or len(expnames) == 0:
        expnames = cfgs.keys()

    # Allow regular expression matching
    expnames = match_expnames(expnames, cfgs)

    # If dependents, add all dependents to the list of expnames
    if run_dependents:
        fringe = expnames
        expnames = []
        while len(fringe) > 0:
            expname = fringe.pop()
            expnames.append(expname)

            if expname not in depends:
                print("WARNING: depends on experiment %s which is not included"
                            % (expname,))
            else:
                for parent in depends[expname]:
                    if parent in fringe or parent in expnames:
                        continue
                    else:
                        fringe.append(parent)

    expnames = [x for x in toposort_flatten(depends) if x in expnames]

    # TODO: should this set defaults using run_policy code? e.g. # of stages?
    instantiated_cfgs = {}
    for expname, cfg in cfgs.items():
        instantiated_cfgs[expname] = instantiate_loops(cfg, 0, quiet=True)

    # Determine which experiments to run
    run_to_run = set()
    new_expnames = set()
    for expname in expnames:
        if only_new and os.path.exists(os.path.join(output_dir, expname)):
            continue
        if no_rerun:
            any_to_run = False
            for index in range(instantiated_cfgs[expname]['__loop_total_size']):
                cfg = instantiate_loops(cfgs[expname], index, quiet=True)
                for stage_idx in range(cfg.get('num_stages',1)):
                    path = os.path.join(output_dir, expname, str(index), str(stage_idx))
                    if cfg.get('train', True):
                        filepath = os.path.join(path, 'policy.pkl')
                        exists = os.path.isfile(filepath)
                        any_to_run |= not exists
                    if cfg.get('test', True):
                        filepath = os.path.join(path, 'test_results.pkl')
                        exists = os.path.isfile(filepath)
                        any_to_run |= not exists
                    if any_to_run:
                        break
                if any_to_run:
                    break
            if any_to_run:
                run_to_run.add(expname)
        else:
            run_to_run.add(expname)

    not_run_deps = set()
    print("experiments to perform (those marked * are already complete):")
    idx = 1
    for expname in expnames:
        if len(depends[expname]) > 0:
            deps = "    (depends on %s)" % (", ".join(depends[expname]))
            not_run_deps.update(depends[expname] - set(expnames))
        else:
            deps = ""
        tasks = []
        if expname in run_to_run:
            tasks.append('run')
        if len(tasks) == 0:
            bullet = ' '*len(str(idx)) + '*'
            tasks = ['done']
        else:
            bullet = str(idx) + '.'
            idx += 1
        loop_size = instantiated_cfgs[expname]['__loop_total_size']
        print("  %s [%s] %s %s%s" % (bullet, " ".join(tasks), loop_size, expname, deps))   
    print()

    if len(not_run_deps) > 0:
        print("WARNING: The above experiments depend on but do not include the following:")
        for x in sorted(not_run_deps):
            print("   " + x)
        print()            

    return expnames, run_to_run, instantiated_cfgs


def sync_final_dir(output_dir, final_dir):
    cmd = ['rsync', '-a', output_dir+'/', final_dir]
    print("\n%%% Syncing final dir; executing: " + " ".join(cmd))
    subprocess.call(cmd)
    print("%%% Syncing complete")


def run_experiment(results, raw_cfg, run_module, index_start, index_end):
    index = index_start
    raw_cfg['base_output_dir'] = raw_cfg['output_dir'] # TODO: replace output_dir with base_output_dir in config
    quiet = raw_cfg.get('__quiet', False)

    while index_end is None or index < index_end:
        if '__final_base_dir' in raw_cfg:
            log_dir = raw_cfg['__final_base_dir']
        else:
            log_dir = raw_cfg['home_dir']
        logger.open_log(log_dir, 'important.log', 
                        prefix='%s %s ' % (raw_cfg['__expname'], index))
        try:        
            # Set elements in raw_cfg if the functions called in process_cfg need to see them
            raw_cfg['__index'] = index        
            raw_cfg['output_dir'] = os.path.abspath(os.path.join(raw_cfg['base_output_dir'], str(index)))

            start_time = time.time()
            cfg = process_cfg(raw_cfg, index=index,
                    pre_resolution_hook=run_module.cfg_pre_resolution_hook,
                    post_object_hook=run_module.cfg_post_object_hook,
                    quiet=quiet)
            cfg['__time_process_cfg'] = time.time() - start_time

            if index == index_start:
                if hasattr(run_module, 'before_experiment'):
                    run_module.before_experiment(cfg, results)

            if '__final_output_dir' in cfg:
                output_dir = raw_cfg['output_dir']
                final_dir = os.path.abspath(os.path.join(cfg['__final_output_dir'], str(index)))
                cfg['__final_output_dir_idx'] = final_dir
                if not os.path.exists(final_dir):
                    os.makedirs(final_dir)

                period = cfg.get('__final_output_period', 0)
                if period > 0:
                    jitter = random.randint(0, period)
                    print("*** Starting rsync process period %ss jitter %ss ***" % (period, jitter))
                    timer = IntervalTimer(period, sync_final_dir, args=[output_dir, final_dir])
                    timer.start(jitter)

            run_module.execute_experiment_iteration(cfg, results)

            if '__final_output_dir' in cfg:
                if period > 0:
                    timer.stop(block=True)

                sync_final_dir(output_dir, final_dir)

            index += 1
            if index_end is None:
                index_end = cfg['__loop_total_size']
        except:
            tb = traceback.format_exc()
            logger.log(tb)
            raise
        finally:
            logger.close_log()

    if hasattr(run_module, 'after_experiment'):
        run_module.after_experiment(cfg, results)


# HACK
def strip_expname(expname):
    res = expname
    if len(expname) > 11 and expname[0:11] == "blue-floor-":
        res = expname[11:]
    if len(res) > 10 and res[0:10] == "all-white-":
        res = res[10:]

    # introduce line breaks
    ind = res.find("-vs-")
    if ind > -1:
        prefix = res[0:ind]
        suffix = res[ind+4:]
        res = prefix + "\n-vs-\n" + suffix

    return res


def run_all_experiments(filename, override_cfg={}, default_cfg={}, 
                        index_start=0, index_end=0):
    results = {}
    raw_cfg = load_raw_cfg(filename)

    defaulted_overridden_cfg = copy.deepcopy(raw_cfg)
    apply_override_and_default(defaulted_overridden_cfg, 
        override_cfg=override_cfg, default_cfg=default_cfg)

    mode = default_cfg.get('mode', 'policy')
    file_mode = mode = raw_cfg.get('mode', mode)
    mode = override_cfg.get('mode', mode)
    if mode == 'policy':
        run_module = pixelworld.run_policy
    elif mode == 'meta':
        run_module = pixelworld.run_policy
    else:
        raise Exception("Invalid mode: " + mode)

    if hasattr(run_module, 'before_all_experiments'):
        run_module.before_all_experiments(raw_cfg, results)

    expname = default_cfg.get('__expname', None)
    expname = raw_cfg.get('__expname', expname)
    expname = override_cfg.get('__expname', expname)

    if file_mode == 'meta':
        no_rerun = defaulted_overridden_cfg.get('__no_rerun', False)
        run_dependents = defaulted_overridden_cfg.get('__run_dependents', False)
        only_new = defaulted_overridden_cfg.get('__only_new', False)
        output_dir = defaulted_overridden_cfg['output_dir']

        cfgs, depends = process_metacfg(raw_cfg, filename)
        
        expnames = [expname] if expname is not None else []
        expnames, run_to_run, instantiated_cfgs = get_expnames_to_perform(expnames, 
            output_dir, cfgs, depends, run_dependents, no_rerun, only_new)

        if mode == 'meta':
            to_run = run_to_run
        elif mode == 'parse':
            to_run = expnames
        else:
            raise Exception('Mode %s not supported here' % (mode,))

        for expname in expnames:            
            if expname in to_run:
                print("\n*** Experiment %s ***\n" % (expname,))
                raw_cfg = cfgs[expname]
                apply_override_and_default(raw_cfg, override_cfg=override_cfg, default_cfg=default_cfg)
                raw_cfg['output_dir'] = os.path.join(raw_cfg['output_dir'], expname)
                if '__final_output_dir' in raw_cfg:
                    raw_cfg['__final_base_dir'] = raw_cfg['__final_output_dir']
                    raw_cfg['__final_output_dir'] = os.path.join(raw_cfg['__final_output_dir'], expname)
                raw_cfg['__expname'] = expname
                run_experiment(results, raw_cfg, run_module, index_start, index_end)
    else:
        print("\n*** Experiment %s ***\n" % (expname,))
        apply_override_and_default(raw_cfg, override_cfg=override_cfg, default_cfg=default_cfg)
        if '__final_output_dir' in raw_cfg:
            raw_cfg['__final_base_dir'] = raw_cfg['__final_output_dir']
        raw_cfg['__expname'] = filename
        run_experiment(results, raw_cfg, run_module, index_start, index_end)


    if hasattr(run_module, 'after_all_experiments'):
        run_module.after_all_experiments(raw_cfg, results)

    return results


if __name__ == "__main__":
    filename, override_cfg, default_cfg, index_start, index_end \
        = parse_arguments()

    run_all_experiments(filename, override_cfg, default_cfg, index_start, index_end)
