"""Code to parse and interpret experiment and metaexperiment files."""

from __future__ import print_function
from __future__ import absolute_import

from collections import OrderedDict
import collections
import copy
import glob
import joblib
import os
import pkg_resources
from pprint import pprint
import re

from toposort import toposort_flatten
import yaml

from pixelworld.misc import print_nested_dict, set_nested, safe_annotate, recursive_merge_into


# References
# ----------

def make_file_ref(filename):
    """Creates a file reference, appropriately escaping any '+'s."""
    return '+' + '++'.join(filename.split('+'))

def resolve_object_ref(object_ref, bindings={}):
    """Resolve an object reference. These references are either to internal
    objects defined in the bindings dictionary:
        +@name
    to objects imported from python packages:
        +module1.module2:identifier
    or to pickled object stored in a file (technically loaded with joblib.load),
    which has three forms:
        +/dir1/dir2/filename
        +dir1/dir2/filename
        +~/dir1/dir2/filename
    The first gives an absolute path, which is unambiguous. The second is a
    path relative to the current working directory. The third is relative to
    the home directory, found in bindings['home_dir'].

    Unescapes filenames (i.e., pairs of '++'s are converted to single +'s).

    A reference to a file may contain wildcards (*). If so, all files matching
    this pattern (as interpreted by glob.glob) will be read, and their 
    corresponding objects will be returned as a list.

    External refs are annotated, if they are objects, with attributes:
        _concepts101_ref          object_ref
        _concepts101_filepath     absolute path to file
    If the reference is to a list, then all the elements of the list are
    annotated similarly, but with a _concepts101_ref sufficient to resolve
    that element in particular. Internal refs are not annotated because the same
    object may be shared among multiple refs.
    """
    if len(object_ref) < 2 or object_ref[0] != '+':
        raise Exception("Object reference must begin with +.")

    ref_is_internal = object_ref[1] == '@'
    ref_is_module = ":" in object_ref
    ref_has_wildcard = "*" in object_ref
    relative_to_base_output = object_ref[1] == '~'

    if ref_is_internal:
        if object_ref[2:] not in bindings:
            raise Exception("Name '{}' cannot be found in bindings.".format(
                object_ref[2:]))
        else:
            return bindings[object_ref[2:]]
    elif ref_is_module:
        entry_point = pkg_resources.EntryPoint.parse('x={}'.format(
            object_ref[1:]))
        obj = entry_point.load(require=False)
        safe_annotate(obj, _concepts101_ref=object_ref)
        return obj
    else:
        if relative_to_base_output:
            if 'home_dir' not in bindings:
                raise Exception("Reference relative to home dir without bindings['home_dir'] being set.")
            if bindings['home_dir'] is None:
                raise Exception("Reference relative to home dir while bindings['home_dir'] is None.")
            if len(object_ref) < 4 or object_ref[2] != '/':
                raise Exception("Invalid reference.")
            path = os.path.join(bindings['home_dir'], object_ref[3:])
        else:
            path = object_ref[1:]

        # Unescape path: s/++/+/
        path = '+'.join(path.split('++'))

        files = glob.glob(path) if ref_has_wildcard else [path]
        if len(files) == 0:
            raise Exception("Reference %s matches no files!" % (object_ref,))
        objs = []
        for f in files:
            filepath = os.path.abspath(f)
            obj = joblib.load(filepath)
            # FIXME: this should be the following if we want to be able to resolve
            # references whose paths include +'s (we don't require this yet), 
            # but it will make inconsistent changes to SMCBankEnv action/observation
            # spaces (so all policies we use need to be retrained):
            #
            # safe_annotate(obj, make_file_ref
            #     _concepts101_ref=make_file_ref(f),
            #     _concepts101_filepath=filepath)
            safe_annotate(obj,
                _concepts101_ref="+%s" % (f,),
                _concepts101_filepath=filepath)
            objs.append(obj)        
        if ref_has_wildcard:
            safe_annotate(objs, _concepts101_ref=object_ref)
            return objs
        else:
            return objs[0]


def modify_object(obj, modifier, bindings):
    """Modify an object according to a modifier string. The modifier string
    is a sequence of modifiers. There are three modifiers:
        [key]       obj <- obj[key]            if key is an integer
                    obj <- obj["key"]          otherwise
        [+@key]     obj <- obj[bindings["key"]]  
        [++key]     obj <- obj["+key"]]          
        .attr       obj <- obj.attr
        *           map the subsequent modifier string over the object

    The map modifier when applied to a sequence maps over its values, and
    when applied to a dictionary maps over its values.
    """
    while len(modifier) > 0:
        if modifier[0] == '[':
            end_pos = modifier.find(']')
            key = modifier[1:end_pos]
            modifier = modifier[end_pos+1:]
            try:
                key = int(key)
            except ValueError:
                pass
            if not isinstance(key, int) and key[0] == '+':
                if key[1] == '+':
                    key = '+' + key[2:]
                else:
                    if key[2:] not in bindings:
                        raise Exception("Bindings does not have key " + key[2:])
                    key = bindings[key[2:]]
            try:
                obj = obj[key]
            except ValueError:
                raise Exception("Object does not have key %s" % (key,))
        elif modifier[0] == '.':
            m = re.match(r'[\[\*.]', modifier[1])
            end_pos = m.end()-1 if m else len(modifier)
            attr = modifier[1:end_pos]
            modifier = modifier[end_pos:]
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                raise Exception("Object does not have attribute " + attr)
        elif modifier[0] == '*':
            if isinstance(obj, collections.Mapping):
                return dict([(k, modify_object(v, modifier[1:], bindings)) 
                                for k, v in obj.items()])
            elif isinstance(obj, collections.Sequence): 
                return [modify_object(x, modifier[1:], bindings) for x in obj]
            else:
                raise Exception("Can only map over mappings and sequences.")
        else:
            raise Exception("Invalid modifier: {}".format(modifier))
    return obj


def resolve_ref(ref, bindings={}, resolve_internal=True, resolve_external=True):
    """Resolves a reference, consisting of an object reference optionally 
    followed by a modifier string:
        +object_ref
        +object_ref+modifier
    For the object reference format see resolve_object_ref, for the modifier
    format see modify_object. Will not split the modifier string at pairs of ++'s
    """
    if len(ref) < 2 or ref[0] != '+':
        raise Exception("Object reference must begin with +.")

    object_ref = ref
    modifier = ""
    if '+' in ref[1:]:
        pos = 1
        while pos < len(ref):
            if ref[pos] == '+':
                if pos+1 < len(ref) and ref[pos+1] == '+':
                    pos += 2
                    continue
                else:
                    object_ref = ref[:pos]
                    modifier = ref[pos+1:]
                    break
            pos += 1

    ref_is_internal = ref[1] == '@'
    if ref_is_internal and not resolve_internal:
        return ref
    if not ref_is_internal and not resolve_external:
        return ref

    return modify_object(resolve_object_ref(object_ref, bindings), modifier, bindings)


def recursively_resolve_refs(x, bindings={}, resolve_internal=True, 
        resolve_external=True):
    if isinstance(x, basestring):
        if x[0] == '+':
            if x[1] == '+':
                return x[1:]
            else:
                return resolve_ref(x, bindings, 
                    resolve_internal=resolve_internal, 
                    resolve_external=resolve_external)
        else:
            return x
    elif isinstance(x, collections.Mapping):
        return dict([(k, recursively_resolve_refs(v, bindings=bindings, 
                    resolve_internal=resolve_internal, 
                    resolve_external=resolve_external)) for k, v in x.items()])
    elif isinstance(x, collections.Sequence):
        return [recursively_resolve_refs(xi, bindings=bindings, 
                    resolve_internal=resolve_internal, 
                    resolve_external=resolve_external) for xi in x]
    else:
        return x


def find_internally_referenced_objects(x):
    if isinstance(x, basestring):
        if len(x)>2 and x[:2] == '+@':
            name = x[2:]
            if '+' in name:
                name = name[:name.find('+')]
            return {name}
        else:
            return set()
    elif isinstance(x, collections.Mapping):
        out = set()
        for k in x:
            out.update(find_internally_referenced_objects(x[k]))
        return out
    elif isinstance(x, collections.Sequence):
        out = set()
        for v in x:
            out.update(find_internally_referenced_objects(v))
        return out
    else:
        return set()



# Configuration file processing
# -----------------------------
# See experiments/run_policy.defaults for overall description.

def instantiate_loops(raw_cfg, index, quiet=False):
    loop_raw_keys = sorted([k for k in raw_cfg.keys() if k[:4] == 'loop'])
    nonloop_raw_keys = sorted([k for k in raw_cfg.keys() if k[:4] != 'loop'])

    raw_cfg = copy.deepcopy(raw_cfg)
    for loop_key in loop_raw_keys:
        raw_val = raw_cfg[loop_key]
        if not ('key' in raw_val) ^ ('keys' in raw_val):
            raise Exception("Loops must specify either 'key' or 'keys'."
                " Loop key: " + loop_key)
        if not ('vals' in raw_val) ^ ('range' in raw_val):
            raise Exception("Loops must specify either 'vals' or 'range'."
                " Loop key: " + loop_key)

        if 'range' in raw_val:
            if not 'high' in raw_val['range']:
                raise Exception("Range loops must set 'high'."
                    " Loop key: " + loop_key)

            vals = range(raw_val['range'].get('low', 0),
                         raw_val['range']['high'],
                         raw_val['range'].get('step', 1))

            if 'key' in raw_val:
                raw_val['vals'] = vals
            elif 'keys' in raw_val:
                raw_val['vals'] = [[v]*len(raw_val['keys']) for v in vals]

        if 'keys' in raw_val:
            for val in raw_val['vals']:
                if not isinstance(val, collections.Sequence):
                    raise Exception("Loop vals with keys must have lists as vals elements.")
                if len(val) != len(raw_val['keys']):
                    raise Exception("Loop vals must be the same size as keys:"
                        "keys %s vals element %s" % (raw_val['keys'], val))

    loop_sizes = [len(raw_cfg[k]['vals']) for k in loop_raw_keys]
    loop_total_size = reduce(lambda x,y: x*y, loop_sizes, 1)
    if len(loop_raw_keys) > 0:
        if not quiet:
            print("instantiate_loops: index %i of %i" % (index, loop_total_size))

        assert index is not None        
        loop_indices = collections.OrderedDict()       
        for loop_key, loop_size in reversed(zip(loop_raw_keys, loop_sizes)):
            loop_indices[loop_key] = index % loop_size
            index //= loop_size

        if not quiet and len(loop_raw_keys) > 1:
            print("  loop sizes:", loop_sizes)
            print("  loop indices:", loop_indices)
        
        if index != 0:
            raise Exception("Index exceeds loop size!")

    # Set non-loop keys
    cfg = dict([(k,raw_cfg[k]) for k in nonloop_raw_keys])
    if not quiet:
        print("  params:")
        print_nested_dict(cfg, "    ", max_keys=0, next_max_keys=20)

    # Set loop keys
    if not quiet:
        print("  loop params:")
    for loop_key in loop_raw_keys:
        raw_val = raw_cfg[loop_key]

        if 'key' in raw_val:
            keys = [raw_val['key']]
            vals = [raw_val['vals'][loop_indices[loop_key]]]
        else:
            keys = raw_val['keys']
            vals = raw_val['vals'][loop_indices[loop_key]]
        
        for key, val in zip(keys, vals):
            if not quiet:
                print("    %s: %s" % (key, val))
            set_nested(cfg, key.split('.'), val)

    cfg['__loop_total_size'] = loop_total_size
    cfg['__loop_sizes'] = loop_sizes
    cfg['__loop_raw_keys'] = loop_raw_keys
    return cfg


def create_objects(cfg, post_object_hook=None, debug=False):
    """Create objects from the raw values of each top-level key of the configuration.

    Creating an object processes the raw value in three steps:
      * Resolving any references to other objects, either internal or external.
      * If the *raw* value contains an 'init' key, call the init function, 
        optionally passing in 'args' or 'kwargs'.
      * Post-process the object by passing it through post_object_hook.

    Objects are be created after objects they have internal references to. If
    this causes a cycle, an exception is raised. Any dependencies recorded
    under the key __additional_dependencies are added to the inferred 
    dependencies.
    """
    if debug:
        print ("create_objects")
    
    init_names = set([k for k in cfg if isinstance(cfg[k], collections.Mapping) 
                                        and 'init' in cfg[k]])

    dependencies = {name : find_internally_referenced_objects(raw_val) 
                        for name, raw_val in cfg.items()}
    for name, deps in cfg.get('__additional_dependencies', {}).items():
        if name in dependencies:
            dependencies[name].update(deps)
        else:
            dependencies[name] = set(deps)
    if debug:
        print("dependencies")
        pprint(dependencies)

    order = toposort_flatten(dependencies)
    defined_objects = set(cfg.keys())
    if not set(order) <= defined_objects:
        raise Exception("Objects which are depended on but not defined: %s"\
            % (set(order) - defined_objects))
    if debug:
        print("order")
        pprint(order)

    raw_cfg = cfg
    cfg = {'__object_creation_order': order}
    # If present, pass home_dir for use in resolving relative refs (+~/...)    
    if 'home_dir' in raw_cfg:
        cfg['home_dir'] = raw_cfg['home_dir']

    for name in order:
        if debug:
            print("creating", name, "from raw value:")
            pprint(raw_cfg[name])
        obj = recursively_resolve_refs(raw_cfg[name], bindings=cfg)
        if debug:
            print("resolved raw value:")
            pprint(obj)

        if name in init_names:
            if not set(obj.keys()) <= {'init', 'kwargs', 'args', 'np_random_seed'}:
                raise Exception("Object {}'s spec has unsupported keys {}".format(
                    name, set(obj.keys()) - {'init', 'kwargs', 'args', 'np_random_seed'}))
            if 'np_random_seed' in obj:
                np.random.seed(obj['np_random_seed'])
            obj = obj['init'](*obj.get('args',[]), **obj.get('kwargs',{}))
            if debug:
                print("after init:")
                pprint(obj)

        if post_object_hook:
            obj = post_object_hook(name, obj, cfg)
            if debug:
                print("afer post_object_hook:")
                pprint(obj)

        cfg[name] = obj

    return cfg


def process_cfg(raw_cfg, index=0, quiet=False, debug=False,
                pre_resolution_hook=None, post_object_hook=None):
    """Process a raw configuration file to a usable one.

    Processing stages can add side information under keys which begin with two 
    underscores. The given hooks may add information under arbitrary keys.
    """
    cfg = copy.deepcopy(raw_cfg)
    if debug:
        print("initial cfg")
        pprint(cfg)

    cfg = instantiate_loops(cfg, index, quiet=quiet)
    if debug:
        print("post instantiate loops")
        pprint(cfg)
    
    if pre_resolution_hook is not None:
        cfg = pre_resolution_hook(cfg)
        if debug:
            print("post pre_resolution_hook")
            pprint(cfg)

    cfg = create_objects(cfg, post_object_hook=post_object_hook, debug=debug)
    if debug:
        print("post create objects")
        pprint(cfg)

    return cfg


def process_includes(raw_cfg, filename, nested_includes=None):
    """Process any files included by raw_cfg. This will (recursively) load the
    raw configuration files listed in __includes__ and merge them, in order,
    into an empty configuration file before merging in raw_cfg. Nested
    dictionaries under the same key will be merged together. Makes a copy of 
    raw_cfg before modifying it. Keys that begin with _ are private and will not
    be included.

    raw_cfg:
        Raw configuration file to process for includes.
    filename:
        Name of file loaded in raw_cfg.
    nested_includes:
        When includes are nested, this is a list of the sequence of files
        the include was nested in. This is used to check that there are no
        cyclic includes.
    """
    # Implementation note: There's some redundant effort in including a file 
    # multiple times, but it does yield consistent results when includes are 
    # nested, and we don't have side effects to worry about.
    nested_includes = [] if nested_includes is None else list(nested_includes)
    nested_includes.append(os.path.abspath(filename))

    # Start with an empty raw_cfg then merge in the includes in sequence,
    # then the existing raw_cfg
    if '__include__' in raw_cfg:
        includes = raw_cfg['__include__']
        if isinstance(includes, basestring):
            includes = [includes]

        out_cfg = {}
        for include in includes:
            if not os.path.isabs(include):
                dirname = os.path.dirname(filename)
                include = os.path.abspath(os.path.join(dirname, include))

            if include in nested_includes:
                raise Exception("Cyclic includes: %s given ancestors %s"
                                    % (include, nested_includes))

            include_raw_cfg = load_raw_cfg(include)
            include_raw_cfg = process_includes(
                include_raw_cfg, include, nested_includes=nested_includes)

            if '__namespace__' in include_raw_cfg:
                namespace = include_raw_cfg['__namespace__']
                del include_raw_cfg['__namespace__']
                new_include_raw_cfg = {}
                for k,v in include_raw_cfg.items():
                    if isinstance(k, basestring) and k[0] == '_':
                        continue
                    new_include_raw_cfg['%s.%s' % (namespace, k)] = v
                include_raw_cfg = new_include_raw_cfg

            recursive_merge_into(out_cfg, include_raw_cfg)

        recursive_merge_into(out_cfg, raw_cfg)
        del out_cfg['__include__']
    else:
        out_cfg = raw_cfg

    return out_cfg


def process_metacfg(raw_metacfg, filename, only_executable=True):
    """Process a raw metacfg.

    The processed metacfg contains full raw_cfgs for each experiment described
    in the metacfg, along with a __depends__ digraph."""

    raw_metacfg = process_includes(raw_metacfg, filename)

    generator_keys = []
    executable_keys = set()
    inherits = {}
    inherits_sets = {}
    depends = {}
    fragments = {}

    cfgs = {}
    for k, v in raw_metacfg.items():
        v = copy.deepcopy(v)
        if isinstance(v, collections.Mapping) and v.get('__ignore__', False):
            continue

        if isinstance(v, collections.Mapping):
            inherits[k] = v.get('__inherits__', set())
            inherits_sets[k] = set(inherits[k])
            if v.get('__execute__', False):
                executable_keys.add(k)
                depends[k] = set(v.get('__depends__', set()))
            for key in v.keys():
                if len(key)>=4 and key[:2] == "__" and key[-2:] == "__":
                    del v[key]
                    if key[2:-2] not in set(['depends', 'inherits', 'execute']):
                        raise Exception("Invalid special key " + key)
            fragments[k] = v
        else:
            cfgs[k] = v

    # Resolve fragments' inheritance
    for name in toposort_flatten(inherits_sets):
        frag = {}
        for parent_cfg in [cfgs[parent_name] for parent_name in inherits[name]]:
            recursive_merge_into(frag, parent_cfg)
        recursive_merge_into(frag, fragments[name])
        cfgs[name] = frag

    # Extract generator keys
    for name in cfgs:
        if len(name) >= 5 and name[:5] == "__gen":
            if isinstance(v, collections.Mapping):
                generator_keys.append(name)
            else:
                raise Exception("Generator key %s has a non-dict value %s" % (name,v))

    # Extract generator dependencies and check validity
    generator_depends = {}
    for gen_key in generator_keys:
        gen_spec = cfgs[gen_key]
        for key in gen_spec:
            if key not in {'call', 'args', 'kwargs', 'inherits', '__depends__'}:
                raise Exception("Generator %s has invalid key %s" % (gen_key, key))
        if 'call' not in gen_spec:
            raise Exception("Generator %s has no call key." % (gen_key,))
        generator_depends[gen_key] = set(gen_spec.get('__depends__', set()))

    # Call generators in dependency order
    generator_order = toposort_flatten(generator_depends)
    for gen_key in generator_order:
        gen_spec = cfgs[gen_key]

        call_fn = resolve_ref(gen_spec['call'])
        args = [cfgs] + gen_spec.get('args', [])
        kwargs = gen_spec.get('kwargs', {})
        
        inherits = gen_spec.get('inherits', [])
        base_cfg = {}
        for parent_cfg in [cfgs[parent_name] for parent_name in inherits]:
            recursive_merge_into(base_cfg, parent_cfg)

        new_cfgs = {}
        for name, cfg in call_fn(*args, **kwargs):
            if name in cfgs or name in new_cfgs:
                raise Exception("Generating an already-existing configuration " + name)
            
            new_cfg = {}
            recursive_merge_into(new_cfg, base_cfg)
            parents = cfg.get('__inherits__', [])
            for parent_cfg in [cfgs[parent_name] for parent_name in parents]:
                recursive_merge_into(new_cfg, parent_cfg)
            recursive_merge_into(new_cfg, cfg)

            if new_cfg.get('__execute__', False):
                executable_keys.add(name)
                depends[name] = set(new_cfg.get('__depends__', set())) 
            
            for key in new_cfg.keys():
                if len(key)>=4 and key[:2] == "__" and key[-2:] == "__":                    
                    del new_cfg[key]
                    if key[2:-2] not in set(['depends', 'execute', 'inherits']):
                        raise Exception("Invalid special key " + key)
            new_cfgs[name] = new_cfg

        cfgs.update(new_cfgs)

    # Remove non-executable cfgs
    for gen_key in generator_keys:
        del cfgs[gen_key]

    if only_executable:
        for name in cfgs.keys():
            if name not in executable_keys:
                del cfgs[name]

        # Check for coherence in the dependency graph by default
        if raw_metacfg.get('__check_dependency_graph', True):
            for name in cfgs:
                for depends_name in depends[name]:
                    if depends_name not in cfgs:
                        raise Exception("%s depends on the non-existent %s (turn off this check with __check_dependency_graph: False)."
                            % (name, depends_name))

    return cfgs, depends


def apply_override_and_default(cfg, override_cfg={}, default_cfg={}):
    for key, val in default_cfg.items():
        if key not in cfg:
            cfg[key] = val
    for key, val in override_cfg.items():
        cfg[key] = val


def load_raw_cfg(filename, override_cfg={}, default_cfg={}):
    with open(filename) as f:
        raw_cfg = yaml.safe_load(f)
    return raw_cfg


def load_cfg(filename, index=0, quiet=False, 
             pre_resolution_hook=None, post_object_hook=None, home_dir=None):
    raw_cfg = load_raw_cfg(filename)
    # If present, store home_dir for use in resolving relative refs (+~/...)
    if home_dir is not None:
        raw_cfg['home_dir'] = home_dir
    return process_cfg(raw_cfg, index=index, quiet=quiet,
        pre_resolution_hook=pre_resolution_hook,
        post_object_hook=post_object_hook)
