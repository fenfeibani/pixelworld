"""
Modular environment based on pixelworld with the ability to easily mix'n'match
modules for sensors, actuators and rewards.
"""
from __future__ import print_function
from __future__ import absolute_import

from pprint import pprint

from collections import defaultdict
from gym import Env
import numpy as np
from rllab.core.serializable import Serializable
from scipy.misc import imresize
from toposort import toposort_flatten

from pixelworld.misc import flatten, compute_descendants
from pixelworld.spaces import flatcat_spaces
from pixelworld.spaces_gym import NamedDiscrete, NamedBox


# Light-weight efficient way to build up an environment out of modules.
# Flexible observation and action spaces.
# Modules communicate through a shared state dictionary (if needed) and they
#    can see the global kwargs used to init all modules.

class ModularEnv(Env, Serializable):
    """A gym environment made up of a series of modules. See module.py:Module
    for the module class.

    Modules can perform actions, supply observations, and compute rewards.
    The environment's state is a dictionary passed to the modules.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}
    reward_range = (-np.inf, np.inf)

    def __init__(self, module_specs=[], max_path_length=100, debug=False, 
        enable_log=True, emit_log=False, **kwargs):
        """
        module_specs : [dict]
            This may consist of nested lists, and are flattened before use.

            a module_spec must provide keys 'class', and optionally 'kwargs'.
            the value of 'kwargs' is merged into __init__'s kwargs and used
            to instantiate class.

        max_path_length : int
            Maximum number of actions to be performed before terminating.

        Other kwargs are passed into modules.
        """
        Serializable.quick_init(self, locals())
        module_specs = flatten(module_specs)
        self.viewer = None
        self.max_path_length = max_path_length

        self.enable_log = enable_log
        self.emit_log = emit_log
        self._log = []        

        # Create modules and record what they depend on and provide, also
        # indexing modules by various search criteria.
        dependency_graph = {}
        all_provides = set()
        all_depends = set()
        unsorted_modules = []
        self.name_to_module = {}
        self.klass_to_modules = defaultdict(list)
        self.provides_to_module = {}
        for idx, module_spec in enumerate(module_specs):
            if 'class' not in module_spec:
                raise Exception("module_spec must specify class.")
            for key in module_spec:
                if key not in {'class', 'kwargs', 'name'}:
                    raise Exception("Invalid module spec key %s" % (key,))
            module_kwargs = {}
            module_kwargs.update(kwargs)
            module_kwargs.update(module_spec.get('kwargs', {}))
            if 'name' in module_spec:
                if 'name' in module_kwargs:
                    raise Exception("Cannot have name in both module_spec and module_kwargs")
                module_kwargs['name'] = module_spec['name']
            if 'modular_env' in module_spec:
                raise Exception("Cannot set modular_env in module_spec")
            module_kwargs['modular_env'] = self

            print("creating module %s" % (module_spec['class'],))
            # pprint(kwargs)
            klass = module_spec['class']
            module = klass(**module_kwargs)

            unsorted_modules.append(module)
            self.klass_to_modules[klass].append(module)
            if 'name' in module_spec:
                name = module_spec['name']
                if name in self.name_to_module:
                    raise Exception("Duplicate module name %s!" % (name,))
                self.name_to_module[name] = module

            depends = set(module.depends)
            provides = set(module.provides)
            for key in depends:
                if len(key) >= 2 and key[:2] == "__":
                    raise Exception("Cannot depend on a key beginning with __")
            for key in provides:
                if len(key) >= 2 and key[:2] == "__":
                    raise Exception("Cannot provide a key beginning with __")
                self.provides_to_module[key] = module
            if debug:
                print("  depends:", depends)
                print("  provides:", provides)
                print()
            if len(provides & all_provides) != 0:
                raise Exception("Duplicated provides!")
            all_depends.update(depends)
            all_provides.update(provides)

            dependency_graph[idx] = depends
            for k in provides:
                if k in dependency_graph:
                    raise Exception("Key %s is provided twice!" % (k,))
                dependency_graph[k] = {idx}

        if debug:
            print("Dependency graph:")
            pprint(dependency_graph)
            print("Topological sort:")
            pprint(toposort_flatten(dependency_graph))

        if not (all_depends <= all_provides):
            print("Missing dependencies:", all_depends - all_provides)
            raise Exception("Keys are dependend on but not provided!")
        if not all([not isinstance(k, int) for k in all_provides]):
            raise Exception("Keys must be not be integers!")

        # Sort keys and modules by dependencies then filter out modules. Compute 
        # also modules downstream of any given module. When one module's step
        # is called all modules downstream of its update method are called.
        self.modules = []
        self.downstream_modules = []
        sorted_nodes = toposort_flatten(dependency_graph)
        for node in sorted_nodes:
            if isinstance(node, int):                
                self.modules.append(unsorted_modules[node])
                descendant_modules = [] 
                descendants = compute_descendants(dependency_graph, node)
                if debug:
                    print("Module (node %i) has descendants %s" % (node, descendants))
                for descendant_node in sorted_nodes:
                    if descendant_node in descendants and isinstance(descendant_node, int):
                        descendant_modules.append(unsorted_modules[descendant_node])
                self.downstream_modules.append(descendant_modules)
                if debug:
                    print("  downstream_modules %s" % (descendant_modules,))


        # Create the global action_space and observation space based on original 
        # (unsorted) module list. This allows for action and observation order
        # to be determined by input module order. Unique names are enforced here.
        action_spaces = []
        observation_spaces = []
        for module in unsorted_modules:
            if module.action_space is not None:
                action_spaces.append(module.action_space)
            if module.observation_space is not None:
                observation_spaces.append(module.observation_space)   
        self.action_space = flatcat_spaces(*action_spaces)               
        self.observation_space = flatcat_spaces(*observation_spaces)


        # Filter out subclasses of modules and compute action to module {idx,action} maps.
        self.reward_modules = []
        self.termination_modules = []
        self.observation_modules = []  
        self.action_to_module_idx = [None for _ in range(self.action_space.n)]
        self.action_to_module_action = [None for _ in range(self.action_space.n)]            
        for idx, module in enumerate(self.modules):
            if module.is_reward_module:
                self.reward_modules.append(module)
            if module.is_termination_module:
                self.termination_modules.append(module)                
            if module.observation_space is not None:
                self.observation_modules.append(module)
            if module.action_space is not None:
                for module_action_index, action_name in enumerate(module.action_space.names):
                    global_action_index = self.action_space.names.index(action_name)
                    self.action_to_module_idx[global_action_index] = idx
                    self.action_to_module_action[global_action_index] = module_action_index

        if debug:
            print("modules:", self.modules)
            print("reward_modules:", self.reward_modules)
            print("observation_modules:", self.observation_modules)
            print("observation_spaces:", observation_spaces)
            print("observation_space:", self.observation_space)
            print("action_spaces:", action_spaces)
            print("action_space:", self.action_space)
            print("action_to_module_idx:", self.action_to_module_idx)
            print("action_to_module_action:", self.action_to_module_action)

        # Initialize empty render layout cache
        self.render_last_sizes = None
        self.render_last_image_locs = None
        self.render_last_canvas_size = None

        self.episode_num = -1
        self.total_reward = 0.0

    def reset_log(self):
        if len(self._log) > 0:
            self._log = []

    def get_log(self):
        return np.array(self._log).reshape((-1,3))

    def get_stats(self):
        log = self.get_log()
        self.reset_log()
        return {'log': log}

    def log(self, source, kind, details):
        """Log information.

        source : str
            identity of the source module
        kind : str
            kind of event
        params : dict
            event details
        """
        if self.enable_log:
            self._log.append((source, kind, details))
            if self.emit_log:
                print("MLOG %s: %s %s" % (source, kind, 
                    " ".join(["%s=%s" % (k,details[k]) for k in sorted(details)])))

    def make_observation(self):
        # TODO: Rather than concatenating, pre-allocate the obs array.
        return np.concatenate([module.make_observation(self.state).flat 
                               for module in self.observation_modules])

    def _reset(self):
        self.episode_num += 1
        self.log("ModularEnv", "reset", {"episode_num": self.episode_num})
        self.total_reward = 0.0
        self.num_steps = 0
        self.terminated = False
        self.state = {'__last_action_name': None}
        modular_env_provides = {}
        for module in self.modules:
            module.during_reset(self.state)
        for module in self.modules:
            module.after_reset(self.state)
        return self.make_observation()

    def _step(self, action):
        self.num_steps += 1
        if self.terminated:
            raise Exception("Trying to act after termination!")
        if not (isinstance(action, int) and 0 <= action < self.action_space.n):
            raise Exception("Invalid action %s" % (action,))

        module_idx = self.action_to_module_idx[action]
        module_action = self.action_to_module_action[action]
        acting_module = self.modules[module_idx]

        reward, done, info = acting_module.execute_action(self.state, module_action)

        self.state['__last_action_name'] = self.action_space.names[action]

        # Update all modules that might be affected by state changes from this
        # module.
        for module in self.downstream_modules[module_idx]:
            module.update_state(self.state)

        # Decide whether to terminate the episode
        if self.num_steps >= self.max_path_length:
            done = True
        for module in self.termination_modules:
            if done:
                break
            done = done or module.decide_termination(self.state)

        self.terminated = done

        # Evaluate any additional rewards
        for module in self.reward_modules:
            reward += module.compute_reward(self.state)

        self.total_reward += reward

        if done:
            self.log("ModularEnv", "done", 
                {"episode_num": self.episode_num,
                 "steps": self.num_steps,
                 "total_reward": self.total_reward})

        return self.make_observation(), reward, done, info

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None            
            return

        # Get images from modules
        rgbs = flatten([module.make_rgb_arrays(self.state) for module in self.modules])

        # How much space between adjacent rgbs
        internal_padding = 2
        
        # Layout images
        #   TODO: user control over this?
        rgbs.sort(key=lambda rgb: -sum(rgb.shape[:2]))
        sizes = [sum(rgb.shape[:2]) for rgb in rgbs]
        if sizes == self.render_last_sizes:
            image_locs = self.render_last_image_locs
            canvas_size = self.render_last_canvas_size
        else:
            # Layout biggest first
            rgb = rgbs[0]
            height, width = rgb.shape[:2]
            height += internal_padding
            width += internal_padding
            image_locs = [(0,0)]
            widths = np.array([width] * height)
            heights = np.array([height] * width)

            # Layout rest to greedily minimize the sum of max height and max width       
            for rgb in rgbs[1:]:
                canvas_height = len(widths)
                canvas_width = len(heights)
                height, width = rgb.shape[:2]
                height += internal_padding
                width += internal_padding
                best_loc = None
                best_score = None
                best_canvas_size = None

                # Search places to add from the right
                for y in range(canvas_height - height + 1):
                    x = max(widths[y:y+height])
                    new_canvas_width = max(x + width, canvas_width)
                    score = new_canvas_width + canvas_height
                    if best_score is None or score < best_score:
                        best_score = score
                        best_loc = (y, x)
                        best_canvas_size = (canvas_height, new_canvas_width)                    

                # Search places to add from the bottom
                for x in range(canvas_width - width + 1):
                    y = max(heights[x:x+width])
                    new_canvas_height = max(y + height, canvas_height)
                    score = new_canvas_height + canvas_width
                    if best_score is None or score < best_score:
                        best_score = score
                        best_loc = (y, x)
                        best_canvas_size = (new_canvas_height, canvas_width)

                assert best_loc is not None, "Layout bug!"                

                # Update layout heights and widths arrays
                y, x = best_loc
                new_canvas_width = max(x + width, canvas_width)
                new_canvas_height = max(y + height, canvas_height)
                if new_canvas_height > canvas_height:
                    padding = np.zeros(new_canvas_height - canvas_height, dtype=np.int)
                    widths = np.concatenate([widths, padding])
                if new_canvas_width > canvas_width:
                    padding = np.zeros(new_canvas_width - canvas_width, dtype=np.int)
                    heights = np.concatenate([heights, padding])                
                widths[y:y+height] = x + width
                heights[x:x+width] = y + height
                image_locs.append(best_loc)

        # Render image into canvas
        #    Q: how to set background color?
        canvas_height = max([y + rgb.shape[0] for rgb, (y, x) in zip(rgbs, image_locs)])
        canvas_width = max([x + rgb.shape[1] for rgb, (y, x) in zip(rgbs, image_locs)])
        canvas = 64*np.ones([canvas_height, canvas_width] + [3], dtype=np.uint8)
        assert len(rgbs) == len(image_locs)
        for rgb, (y, x) in zip(rgbs, image_locs):
            height, width = rgb.shape[:2]
            canvas[y:y+height, x:x+width] = rgb

        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(canvas)
        elif mode == 'rgb_array':
            return canvas
        else:
            super(ModularEnv, self).render(mode=mode)

    def find_modules(self, name=None, klass=None, provides=None):
        if name is not None:
            if klass is not None or provides is not None:
                raise Exception("Only one search criterion allowed.")
            maybe_module = self.name_to_module.get(name, None)
            return [maybe_module] if maybe_module is not None else []
        elif klass is not None:
            if name is not None and provides is not None:
                raise Exception("Only one search criterion allowed.")
            return self.klass_to_modules[klass]
        elif provides is not None:
            if name is not None and klass is not None:
                raise Exception("Only one search criterion allowed.")
            maybe_module = self.provides_to_module.get(provides, None)
            return [maybe_module] if maybe_module is not None else []
        else:
            raise Exception("Must supply exactly one search criterion.")
    
    def clone(self, out=None):
        """Clones state of this environment, optionally into an existing one."""
        if out is None:
            # Reconstruct using the state saved by Serializable.quick_init
            out = type(self)(*self.__args, **self.__kwargs)
        
        if type(out) != type(self):
            raise Exception("out has the wrong type")
        if out._Serializable__args != self._Serializable__args or out._Serializable__kwargs != self._Serializable__kwargs:
            raise Exception("out was constructed with the wrong arguments")

        out.num_steps = self.num_steps
        out.terminated = self.terminated
        out.state['__last_action_name'] = self.state['__last_action_name']
        
        for module, out_module in zip(self.modules, out.modules):
            module.clone(self.state, out_module, out.state)
        return out

    def describe(self):   
        """Returns a multi-faceted description of this environment, as a 
        dictionary of tensors.

        NOTE: This should not be called before reset(), but may be called
        after the environment is done. Describe can return different sets of
        keys at different invocations. After a run, all descriptions are 
        combined by combining the values which have the same key. For details
        of how descriptions are combined, see run_policy.py:rollout_record.
        """
        description = {}
        description['rgb'] = self.render(mode='rgb_array')
        for module in self.modules:
            module.make_description(self.state, description)
        return description


    # HACK: to work with current run_policy evaluation code
    def set_subenv(self, instance_index):
        modules = self.find_modules(provides='data_index')
        if len(modules) != 1:
            raise Exception("Cannot find unique DatasetModule!")
        modules[0].set_data_index(instance_index, 'fixed')

    @property
    def current_subenv(self):
        return self.state['pw']

    @property
    def current_spec(self):
        return self.state['spec']

    # HACK: to work with current run_policy evaluation code
    @property
    def num_subenvs(self):
        modules = self.find_modules(provides='data_index')
        if len(modules) != 1:
            raise Exception("Cannot find unique DatasetModule!")
        return len(modules[0].specs)

    @property
    def subenv_label(self):
        return self.state['label']
        
