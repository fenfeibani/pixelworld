"""
Modules for ModularEnv.
"""
from __future__ import print_function
from __future__ import absolute_import

from pprint import pprint
from copy import copy, deepcopy

from gym import Env
import numpy as np
from rllab.core.serializable import Serializable
from toposort import toposort_flatten
import weakref

from pixelworld.misc import better_imresize
from pixelworld.spaces import inject_discrete, project_box
from pixelworld.spaces_gym import NamedDiscrete, NamedBox
from pixelworld.envs.minimal_pixelworld import MinimalPixelWorld
from pixelworld.envs.pixelworld.core import PixelWorld
from pixelworld.envs.pixelworld.utils import ind_to_rgb, generate_color_palette
import pixelworld.envs.pixelworld as pw

LUT = generate_color_palette(12)


class Module(object):
    """Modules are fragments of environments. They can create observations,
    execute actions, supply rewards, progressively define state, and terminate
    the episode.
    """
    # Which keys of the state does this module depend on?
    depends = []
    # Which keys of the state does this module provide?
    provides = []
    # Does this module compute additional rewards after step?
    is_reward_module = False
    # Does this module help decide whether the episode has terminated?
    is_termination_module = False

    action_space = None
    observation_space = None
    
    def __init__(self, modular_env=None, **kwargs):
        """Called by ModularEnv.__init__ with its kwargs."""
        if modular_env is None:
            self._modular_env = None
        else:    
            self._modular_env = weakref.ref(modular_env)

    def close(self): 
        pass       


    @property    
    def modular_env(self):  
        if self._modular_env is None:
            return None
        else:      
            me = self._modular_env()
            assert not me is None, "ModularEnv weakref has been deleted."
            return me

    def log(self,*args):
        if not self._modular_env is None:
            self.modular_env.log(*args)
      

    def during_reset(self, state):
        """Called by ModularEnv.reset().
        This may modify the value of state only at keys this modules provides.
        """
        pass

    def after_reset(self, state):
        """Called by ModularEnv.reset() after all modules' reset()s have been 
        called.

        This must not modify state.
        """
        pass

    def clone(self, state, out_module, out_state):
        """Clone the state keys this module provides from state to out_state,
        and any internal module state from self to out_module. Assume that the
        other module has been created by calling __init__ with the same kwargs.
        """
        raise NotImplementedError

    def make_observation(self, state, observation=None):
        """Return an observation. Only called if observation_space is not None.
        Called after executing any actions and computing any rewards.

        This must not modify state.
        """
        raise NotImplementedError

    def execute_action(self, state, action):
        """Return a tuple of reward, done, info. Only called if one of this 
        modules' actions is taken.

        This may modify the value of state only at keys this modules provides.
        """
        raise NotImplementedError

    def update_state(self, state):
        """Called when one of the keys this depends on may have changed. 

        This may modify the value of state only at keys this modules provides.
        """
        pass

    def compute_reward(self, state):
        """Return a reward. Called after executing the action and determining 
        whether the episode has terminated, and only if self.is_reward_module is
        True. next_state is None if the episode has just terminated.

        This must not modify state.
        """
        pass

    def decide_termination(self, state):
        """Return whether to terminate the episode. Called after executing the 
        action, and only if self.is_termination_module is True and the episode
        hasn't yet terminated.

        This must not modify state.
        """
        raise NotImplementedError

    def make_description(self, state, description):
        """Add entries the description of the current state

        This must not modify state.
        """
        pass

    def make_rgb_arrays(self, state):
        """Return a list of rgb arrays to be combined into the rendered image.

        This must not modify state.
        """
        return []


class RewardShapingModule(Module):
    """Potential-based shaping reward. This is guaranteed not to change the
    optimal policy (Ng, Harada and Russell, 1999).
    (See also BringAboutShapingModule for simple implemented potential function)
    """
    depends = '__all__'
    is_reward_module = True

    def __init__(self, shaping_func, modular_env=None, **kwargs):
        super(RewardShapingModule,self).__init__(modular_env, **kwargs)        
        self.shaping_func = shaping_func
        self.last_pot = None

    def after_reset(self, state):
        self.last_pot = self.shaping_func(state)

    def clone(self, state, out_module, out_state):
        out_module.last_pot = self.last_pot

    def compute_reward(self, state):
        # The potential in terminal states must be constant for the set of 
        # optimal policies not to change.
        if next_state is None:
            pot = 0
        else:
            pot = self.shaping_func(state)
        r = pot - self.last_pot
        self.last_pot = pot
        return r


class DatasetModule(Module):
    """
    Dataset.
    """
    depends = []
    # see __init__ for provides

    def __init__(self, specs=None, labels=None, reset_mode='random', seed=0, 
        modular_env=None, **kwargs):
        """
        specs : [spec]
            List of specs. A spec determines an environment, in a format to be
            determined by subclasses.
        labels : [int]
            (optional) List of classification labels.
        seed : int
            Seed for randomly selecting which subenvironment to run upon reset
            in the reset mode 'random'
        reset_mode : 'random', 'fixed', 'sequential'
            Mode to select specs upon reset.
        """
        super(DatasetModule,self).__init__(modular_env, **kwargs)
        self.provides = ['data_index', 'spec']
        if labels:
            self.provides += ['label']
        self.specs = specs
        self.labels = labels
        self.reset_mode = reset_mode
        self.random = np.random.RandomState(seed)
        self.data_index = 0
        self.log("DatasetModule", "init", {"size": len(self.specs)})

    def during_reset(self, state):
        if self.reset_mode == "random":
            self.data_index = self.random.randint(len(self.specs))
        elif self.reset_mode == "sequential":
            self.data_index = (self.data_index + 1) % len(self.specs)
        state['data_index'] = self.data_index
        state['spec'] = self.specs[self.data_index]
        if self.labels:
            state['label'] = self.labels[self.data_index]
        self.log("DatasetModule", "reset", {"data_index": state['data_index']})

    def clone(self, state, out_module, out_state):
        out_module.random.set_state(self.random.get_state())
        out_module.data_index = self.data_index
        out_state['data_index'] = state['data_index']
        out_state['spec'] = state['spec']
        if self.labels:
            out_state['label'] = state['label']

    def make_description(self, state, description):
        if 'label' in state:
            description["gt_label"] = state['label']

    def set_data_index(self, data_index, reset_mode):
        self.data_index = data_index
        self.reset_mode = reset_mode


class PixelWorldModule(Module):
    """Wraps a pixel world."""
    depends = ['spec', 'data_index']
    provides = ['pw', 'pw_obs', 'height', 'width']

    observed_attributes = ['name','id','head_of_family','position','color','meta']

    def __init__(self, cache=True, include_actions=True, remove_noop=False,
        cache_limit=1000, modular_env=None, **kwargs):
        """
        cache : bool
            Whether to cache pixelworld instances for a given spec.

        include_actions : bool
            Whether to include the pixelworld's actions.

        remove_noop : bool
            Remove NOOP action.
        """
        super(PixelWorldModule,self).__init__(modular_env, **kwargs)

        # HACK to determine pixelworld action space
        if 'spec' in kwargs:
            spec = kwargs['spec']
        elif 'specs' in kwargs:
            spec = kwargs['specs'][0]
        else:
            raise Exception("Cannot find spec to determine pw action space!")
  
        pw = PixelWorld(obs_type='state', objects=spec['objects'],
                        height=spec['height'], width=spec['width'],
                        observed_attributes=self.observed_attributes)         
        if include_actions:
            self.pw_actions = list(pw.actions)
            self.actions = list(self.pw_actions)
            if remove_noop:
                self.actions.remove('NOOP')
            self.action_space = NamedDiscrete(len(self.actions), names=self.actions)
        self.pw = None
        self.include_actions = include_actions
        self.spec = None
        self.cache = cache
        self.cache_map = {}
        self.cache_limit = cache_limit

    def during_reset(self, state):
        self.spec = spec = state['spec']
        data_index = state['data_index']        
        if data_index in self.cache_map:
            self.pw = self.cache_map[data_index]
        else:
            self.pw = PixelWorld(obs_type='state', objects=spec['objects'],
                                 height=spec['height'], width=spec['width'],
                                 observed_attributes=self.observed_attributes)
            if self.include_actions and (self.pw.actions != self.pw_actions):
                raise Exception("Pixelworlds must have the same action sets!")
            if self.cache:
                if len(self.cache_map) > self.cache_limit:
                    keys = list(self.cache_map.keys())                    
                    keys = keys[:len(keys)/2]
                    for k in keys:
                        del self.cache_map[k]
                self.cache_map[data_index] = self.pw

        state['pw'] = self.pw
        state['pw_obs'] = self.pw.reset()
        state['height'] = spec['height']
        state['width'] = spec['width']


    def update_state(self, state):
        raise Exception("spec should not be modified during execution!")

    def clone(self, state, out_module, out_state):
        if self.spec != out_module.spec:
            # Pixelworld has changed, so we must update its instance
            assert state['spec'] == out_state['spec'], "DatasetModule must be cloned first"
            out_module.during_reset(out_state)
        else:
            src_pw = state['pw']
            dest_pw = out_state['pw']
            
            #HACK: this deals with situations in which a GRIP action has been
            #called (without a subsequent UNGRIP) in one world but not in the
            #other. it will only work if there is only one object that grips and
            #no other changes to the object structure have occurred between the
            #two worlds, so that the grip object is the last object in the
            #master data array. NOTE: this does not reproduce the corresponding
            #grip-related events.
            if src_pw.num_objects != dest_pw.num_objects:
                if src_pw.num_objects == dest_pw.num_objects + 1 and \
                src_pw.objects[-1].name == 'grip':
                    #add a grip object to dest and make it grip the same objects
                    #as src
                    src_grip = src_pw.objects[-1]
                    src_gripper = src_grip.grip
                    gripper_id = src_gripper.id
                    dest_gripper = dest_pw.objects[gripper_id]
                    
                    dest_grip = dest_pw.create_object('grip')
                    dest_grip.grip = dest_gripper
                    for idx in src_grip.children:
                        dest_grippee = dest_pw.objects[idx]
                        dest_grip.add_child(dest_grippee)
                        if dest_grippee != dest_gripper:
                            dest_grippee.add_attribute('gripped', dest_gripper)
                elif dest_pw.num_objects == src_pw.num_objects + 1 and \
                     dest_pw.objects[-1].name == 'grip':
                    #remove all traces of the grip from dest
                    dest_pw.objects[-1].remove()
                    dest_pw.object_attributes['grip'].remove()
                    dest_pw.object_attributes['gripped'].remove()
                else:
                    raise RuntimeError('non-matching object structure!')
            
            #copy the object positions from src to dest.
            #HACK: this assumes that corresponding objects are "the same"
            dest_pw.object_attributes['position'].set(None, src_pw.object_attributes['position'].get())

    def execute_action(self, state, action):
        action_name = self.actions[action]
        obs, reward, done, info = self.pw.step(action_name)
        state['pw_obs'] = obs
        return 0, done, {}

    def make_rgb_arrays(self, state):
        obs = state['pw_obs']

        rgb = ind_to_rgb(obs, LUT)
        # max_dim = float(max(rgb.shape)) #get the resize factor
        # #f = max(self._render_size)/max_dim   #resize for viewing 
        # f = max([x/float(y) for x,y in zip(self._render_size, rgb.shape)])
        # size = [2*int(rgb.shape[0]*f/2.), 2*int(rgb.shape[1]*f/2.)]

        # TODO: better size control
        size = [10 * rgb.shape[0], 10 * rgb.shape[1]]
        rgb = better_imresize(rgb, size, interp='nearest')
        return [rgb]

    def make_description(self, state, description):
        self_obj = self.pw.objects.get(name='self')
        if self_obj is not None:
            row, col = self_obj.position.astype(int)
            # FIXME: correspond to above size control
            # FIXME: need to translate into current render layout location?
            scale_factor = 10.0 
            row = scale_factor * (row + .5)
            col = scale_factor * (col + .5)
            description["self_pos"] = (row, col)


class MinimalPixelWorldModule(Module):
    """Wraps a barebones version of PixelWorld"""
    depends = ['spec', 'data_index']
    provides = ['pw', 'pw_obs', 'height', 'width']
    
    observed_attributes = ['name','id','head_of_family','position','color','meta']
    
    def __init__(self, cache=True, include_actions=True, remove_noop=False,
        do_aaai18_reset_behavior=False, cache_limit=1000, modular_env=None, 
        **kwargs):
        """
        cache : bool
            Whether to cache pixelworld instances for a given spec.
        
        include_actions : bool
            Whether to include the pixelworld's actions.
        
        remove_noop : bool
            Remove NOOP action.
        
        do_aaai18_reset_behavior : bool, optional
            see MinimalPixelWorld
        """
        super(MinimalPixelWorldModule,self).__init__(modular_env, **kwargs)
        
        if include_actions:
            self.pw_actions = list(MinimalPixelWorld._actions)
            self.actions = list(self.pw_actions)
            
            if remove_noop: self.actions.remove('NOOP')
            
            self.action_space = NamedDiscrete(len(self.actions), names=self.actions)
        
        self.do_aaai18_reset_behavior = do_aaai18_reset_behavior
        
        self.pw = None
        self.include_actions = include_actions
        self.spec = None
        self.cache = cache
        self.cache_map = {}
        self.cache_limit = cache_limit
    
    def during_reset(self, state):
        self.spec = spec = state['spec']
        data_index = state['data_index']        
        if data_index in self.cache_map:
            self.pw = self.cache_map[data_index]
        else:
            self.pw = MinimalPixelWorld(
                        objects=spec['objects'],
                        height=spec['height'],
                        width=spec['width'],
                        do_aaai18_reset_behavior=self.do_aaai18_reset_behavior,
                        )
            
            if self.include_actions and (self.pw.actions != self.pw_actions):
                raise Exception("Pixelworlds must have the same action sets!")
            
            if self.cache:
                if len(self.cache_map) > self.cache_limit:
                    keys = list(self.cache_map.keys())                    
                    keys = keys[:len(keys)/2]
                    for k in keys:
                        del self.cache_map[k]
                self.cache_map[data_index] = self.pw
        
        state['pw'] = self.pw
        state['pw_obs'] = self.pw.reset()
        state['height'] = spec['height']
        state['width'] = spec['width']
    
    def update_state(self, state):
        raise Exception("spec should not be modified during execution!")
    
    def clone(self, state, out_module, out_state):
        if self.spec != out_module.spec:
            # Pixelworld has changed, so we must update its instance
            assert state['spec'] == out_state['spec'], "DatasetModule must be cloned first"
            out_module.during_reset(out_state)
        else:
            src_pw = state['pw']
            dest_pw = out_state['pw']
            
            #copy the object positions from src to dest.
            #HACK: this assumes that corresponding objects are "the same"
            dest_pw.objects.positions = src_pw.objects.positions
    
    def execute_action(self, state, action):
        action_name = self.actions[action]
        obs, reward, done, info = self.pw.step(action_name)
        state['pw_obs'] = obs
        return 0, done, {}
    
    def make_rgb_arrays(self, state):
        obs = state['pw_obs']
        
        rgb = ind_to_rgb(obs, LUT)
        
        # TODO: better size control
        size = [10 * rgb.shape[0], 10 * rgb.shape[1]]
        rgb = better_imresize(rgb, size, interp='nearest')
        return [rgb]

    def make_description(self, state, description):
        self_obj = self.pw.self_obj
        
        if self_obj is not None:
            row, col = self_obj.position.astype(int)
            # FIXME: correspond to above size control
            # FIXME: need to translate into current render layout location?
            scale_factor = 10.0 
            row = scale_factor * (row + .5)
            col = scale_factor * (col + .5)
            description["self_pos"] = (row, col)


class ClassificationSignalModule(Module):
    """
    Terminate the episode with a signal result.
    """
    is_termination_module = True
    is_reward_module = True

    depends = ['label']
    provides = ['signal']

    actions = ['SIG0','SIG1']


    def __init__(self, reward=1.0, signals=[-1,1], step_cost=0.0, modular_env=None, **kwargs):
        super(ClassificationSignalModule,self).__init__(modular_env, **kwargs)        
        self.step_cost = step_cost
        self.action_space = NamedDiscrete(len(self.actions), self.actions)
        self.reward = reward
        self.signals = signals     

    @property
    def positive_signal(self):
        return self.signals[1]

    @property
    def negative_signal(self):
        return self.signals[0]    

    def during_reset(self, state):
        state['signal'] = None
        #self.correct = False

    def clone(self, state, out_module, out_state):
        out_state['signal'] = state['signal']
        #out_module.correct = self.correct

    def execute_action(self, state, action):
        """Store signal result and terminate episode."""
        state['signal'] = self.signals[action]
        self.log("ClassificationSignalModule", "done", {"signal": state['signal'],
                    "label": state['label']})
        #return reward, True, {}
        return 0, False, {}  # dummy return, reward and done handled separately


    def decide_termination(self,state):
        return not state['signal'] is None

    def compute_reward(self,state):    
        assert state['label'] == 0 or state['label'] == 1
        #self.correct = (state['label'] == state['signal'])    
        signal = state['signal']
        if signal is None: reward = 0
        elif signal == self.positive_signal:
            reward = self.reward if state['label'] else -self.reward
        elif signal == self.negative_signal:
            reward = -self.reward if state['label'] else self.reward
        else:
            raise Exception("Invalid signal value") 
        return reward - self.step_cost 
 

    def make_rgb_arrays(self, state):
        if state['signal'] == self.positive_signal:
            color = 9
        elif state['signal'] == self.negative_signal:
            color = 2
        else:
            color = 7
        ind = np.array([[color]])
        rgb = ind_to_rgb(ind, LUT)
        size = [50, 50]
        rgb = better_imresize(rgb, size, interp='nearest')
        return [rgb]


class BringAboutDirectModule(Module):
    """
    Terminate the episode and provide reward when the concept is achieved.
    """
    depends = ['spec', 'pw']
    is_termination_module = True
    is_reward_module = True

    def __init__(self, reward=1.0, step_cost=0.0, modular_env=None, **kwargs):
        super(BringAboutDirectModule,self).__init__(modular_env,**kwargs)        
        self.reward = reward
        self.step_cost = step_cost
        self.recent_trajectory = []
        self.initial_object_state = []

    def during_reset(self, state):
        spec = state['spec']
        self.concept = spec['concept']
        self.order = spec['order']  
        self.recent_trajectory = []
        self.initial_object_state = self._get_object_state(state)       

    def clone(self, state, out_module, out_state):
        out_module.concept = self.concept
        out_module.order = self.order

    def update_state(self,state):
        """ Update the local state trajectory with current state """
        action_name = state['__last_action_name']
        self._update_trajectory(state,action_name)

    def decide_termination(self, state):
        done = self._get_concept_status(state)
        if done:
            self.log("BringAboutDirectModule", "done", {})
        return done

    def compute_reward(self,state):
        success = self._get_concept_status(state) 
        reward = self.reward if success else 0  
        return reward - self.step_cost         

    def _get_object_state(self,state):
        object_state = [obj.state for obj in state['pw'].objects if obj.head_of_family == obj.id]
        return deepcopy(object_state)

    def _get_concept_status(self,state):
        trajectory = [(self.initial_object_state, None)] + self.recent_trajectory  
        return self.concept.concept_is_present(trajectory)        

    def _update_trajectory(self,state,action_name):
        object_state = self._get_object_state(state)
        self.recent_trajectory.append((object_state,action_name))
        if len(self.recent_trajectory) > self.order:
            self.recent_trajectory.pop(0)




class BringAboutSignalModule(BringAboutDirectModule):
    """
    Terminate the episode with a signal result.
    """
    provides = ['signal']
    signals = [1]

    def __init__(self, reward=1.0, step_cost=0.0, reward_mistake=-1.0, modular_env=None, **kwargs):
        super(BringAboutSignalModule,self).__init__(
            reward=reward, step_cost=step_cost, modular_env=modular_env, **kwargs)
        self.reward_mistake = reward_mistake
        self.actions = ['SIG1']
        self.action_space = NamedDiscrete(len(self.actions), self.actions)

    def during_reset(self, state):
        super(BringAboutSignalModule,self).during_reset(state)      
        state['signal'] = None       

    def clone(self, state, out_module, out_state):
        super(BringAboutSignalModule,self).clone(state,out_module,out_state)
        out_state['signal'] = state['signal']

    def execute_action(self, state, action):
        """Store signal result and terminate episode."""
        state['signal'] = self.signals[action]
        self.log("BringAboutSignalModule", "done", {})
        return 0, True, {}  # reward via separate call to compute reward

    def decide_termination(self, state):        
        return not state['signal'] is None    

    def compute_reward(self,state):      
        if state['signal'] == 1:
            success = self._get_concept_status(state)
            reward = self.reward if success else self.reward_mistake
        else:
            reward = 0
        return reward - self.step_cost         



class BringAboutShapingModule(BringAboutSignalModule):
    """Potential-based shaping reward. This is guaranteed not to change the
    optimal policy (Ng, Harada and Russell, 1999).

    State potentials:

            0.5              1                     0
        concept_true -----------------> terminated_concept_true
            | ^
          0 | | 0
            V |
        concept_false ----------------> terminated_concept_false
             0              -1                     0

    Edges give intrinsic rewards, nodes potentials.
    Effective reward transitioning along an edge is intrinsic reward plus
    difference in potential:

                            0.5                     
        concept_true -----------------> terminated_concept_true
            | ^
       -0.5 | | 0.5
            V |
        concept_false ----------------> terminated_concept_false
                            -1
    """

    def __init__(self, reward=1.0, step_cost=0.0, reward_mistake=-1.0, max_pot=0.5, modular_env=None, **kwargs):
        super(BringAboutShapingModule,self).__init__(
            reward=reward, step_cost=step_cost, reward_mistake=reward_mistake, 
            modular_env=modular_env, **kwargs)
        self.last_pot = None
        self.max_pot = max_pot

    def clone(self, state, out_module, out_state):
        super(BringAboutShapingModule, self).clone(state, out_module, out_state)
        out_module.last_pot = self.last_pot

    def _compute_shaping_potential(self, state):
        if state['signal'] == 1:
            # The potential of terminal states must be constant for the set of 
            # optimal policies not to change.
            pot = 0
        else:
            pot = self.reward*self.max_pot if self._get_concept_status(state) else 0
        return pot

    def after_reset(self, state):
        self.last_pot = self._compute_shaping_potential(state)
        super(BringAboutShapingModule, self).after_reset(state)

    def compute_reward(self, state):
        # Compute shaping reward gained by transition from last state to this state
        pot = self._compute_shaping_potential(state)
        shaping_reward = pot - self.last_pot
        self.last_pot = pot  # this must only be called once per step
        return shaping_reward + super(BringAboutShapingModule, self).compute_reward(state)



class LocalObservationModule(Module):
    """
    Simple vision around a focus. The focus may follow the self, or it may be
    independently controllable.
    """
    depends = ['spec', 'pw', 'pw_obs', 'height', 'width']
    # see __init__ for provides

    def __init__(self, name="eye", radius=1, x_radius=None, y_radius=None, 
                controllable=False, include_return_to_self=False,
                track_self=False, modular_env=None, **kwargs):
        super(LocalObservationModule,self).__init__(modular_env,**kwargs)     
        self.x_radius = x_radius if x_radius is not None else radius
        self.y_radius = y_radius if y_radius is not None else radius
        self.controllable = controllable
        self.track_self = track_self
        self.include_return_to_self = include_return_to_self
        if controllable:
            uname = str(name).upper()
            self.actions = [uname + '_LEFT', uname + '_RIGHT', 
                            uname + '_UP', uname + '_DOWN']
            if include_return_to_self:
                self.actions += [uname + '_RETURN']
            self.action_space = NamedDiscrete(len(self.actions), names=self.actions)

        name = str(name)
        if len(name) == 0:
            raise Exception("Name must be non-empty.")
        self.eye_pos_key = '%s_pos' % (name,)
        self.eye_obs_key = '%s_obs' % (name,)
        self.provides = [self.eye_pos_key, self.eye_obs_key]

        # TODO: should the following names be a string?
        obs_names = [(name,(y,x)) for y in range(-self.y_radius, self.y_radius+1) 
                                  for x in range(-self.x_radius, self.x_radius+1)] 
        shape = [2*self.y_radius+1, 2*self.x_radius+1]
        self.observation_space = NamedBox(low=0, high=np.iinfo(int).max, 
                                          shape=shape, names=obs_names)

    def during_reset(self, state):
        self.self_obj = state['pw'].objects.get(name='self')
        if self.eye_pos_key in state['spec']:
            eye_pos = state['spec'][self.eye_pos_key]
        elif self.self_obj is not None:
            eye_pos = self.self_obj.position.astype(int)
        else:
            eye_pos = [state['height']//2, state['width']//2]
        state[self.eye_pos_key] = np.array(eye_pos, dtype=int)
        self.update_observation(state)

    def clone(self, state, out_module, out_state):
        out_module.self_obj = out_state['pw'].objects.get(name='self')
        out_module.update_state(out_state)

    def execute_action(self, state, action):
        eye_pos = state[self.eye_pos_key]
        # LEFT RIGHT UP DOWN RETURN
        # 0    1     2  3    4
        if action == 0 and eye_pos[1] > 0:   
            eye_pos[1] -= 1
        elif action == 1 and eye_pos[1] < state['width']:
            eye_pos[1] += 1
        elif action == 2 and eye_pos[0] > 0:
            eye_pos[0] -= 1
        elif action == 3 and eye_pos[0] < state['height']:
            eye_pos[0] += 1 
        elif action == 4 and self.include_return_to_self:
            eye_pos[:] = self.self_obj.position.astype(int)
        self.update_observation(state)
        return 0.0, False, {}

    def update_state(self, state):
        if self.track_self:
            state[self.eye_pos_key][:] = self.self_obj.position.astype(int)
        self.update_observation(state)

    def make_observation(self, state):
        return state[self.eye_obs_key]

    def make_rgb_arrays(self, state):
        obs = self.make_observation(state)
        rgb = ind_to_rgb(obs, LUT)
        # max_dim = float(max(rgb.shape)) #get the resize factor
        # #f = max(self._render_size)/max_dim   #resize for viewing 
        # f = max([x/float(y) for x,y in zip(self._render_size, rgb.shape)])
        # size = [2*int(rgb.shape[0]*f/2.), 2*int(rgb.shape[1]*f/2.)]

        # TODO: better size control
        size = [10 * rgb.shape[0], 10 * rgb.shape[1]]
        rgb = better_imresize(rgb, size, interp='nearest')
        return [rgb]

    def update_observation(self, state):
        """Observe a 2-d local 'state patch' around the focus (or given location). 
        Size is determined by self.radius.
        """
        obs = state['pw_obs']
        obs_height = obs.shape[0]
        obs_width = obs.shape[1]

        frow, fcol = state[self.eye_pos_key]

        local_height = 2*self.y_radius + 1
        local_width = 2*self.x_radius + 1
        local_half_height = self.y_radius
        local_half_width = self.x_radius
        
        # extract local patch around the hand
        local_obs = np.zeros([local_height, local_width], dtype=np.uint32)

        # determine effective region (i.e., within observation boundaries; 
        # rest will be padded with zeros)
        obs_min_row = max(0, frow - local_half_height)
        obs_max_row = min(obs_height, frow + local_half_height)
        obs_min_col = max(0, fcol - local_half_width)
        obs_max_col = min(obs_width, fcol + local_half_width)

        if obs_min_row <= obs_max_row and obs_min_col <= obs_max_col:
            local_obs_min_row = max(0, local_half_height - frow)
            local_obs_max_row = min(local_height, local_height - ((frow + local_half_height + 2) - obs_height))
            local_obs_min_col = max(0, local_half_width - fcol)
            local_obs_max_col = min(local_width, local_width - ((fcol + local_half_width + 2) - obs_width))

            local_obs[local_obs_min_row : local_obs_max_row + 1, local_obs_min_col : local_obs_max_col + 1] = \
                obs[obs_min_row : obs_max_row + 1, obs_min_col : obs_max_col + 1]

        state[self.eye_obs_key] = local_obs

    def make_description(self, state, description):
        frow, fcol = state[self.eye_pos_key]
        # FIXME: correspond to above size control
        # FIXME: need to translate into current render layout location?
        scale_factor = 10.0 
        frow = scale_factor * (frow + .5)
        fcol = scale_factor * (fcol + .5)
        description["eye_pos"] = (frow, fcol)



class HandObservationModule(LocalObservationModule):
    """ Fixed settings for the default local hand/self observation """

    def __init__(self, radius=1, **kwargs):
        super(HandObservationModule,self).__init__(name="hand", track_self=True, 
           radius=radius, controllable=False, include_return_to_self=False, **kwargs)


class SimpleEyeModule(LocalObservationModule):
    """ Fixed settings for the simple movable eye """

    def __init__(self, radius=1, **kwargs):
        super(SimpleEyeModule,self).__init__(name="eye", track_self=False, 
           radius=radius, controllable=True, include_return_to_self=True, **kwargs)    





class FoveaLocalObservationModule(LocalObservationModule):
    """


    """
    depends = ['spec', 'pw', 'pw_obs', 'height', 'width']
    # see __init__ for provides

    def __init__(self, etc, **kwargs):
        raise NotImplementedError

    def update_state(self, state):
        raise NotImplementedError


class ImageModule(Module):
    """Provides a simple image environment instead of a PixelWorld"""
    depends = ['spec']
    provides = ['pw', 'pw_obs', 'width', 'height']

    action_space = None
    observation_space = None
    
    def __init__(self, specs, labels, modular_env=None,**kwargs):
        self.specs = specs
        self.labels = labels
        super(ImageModule,self).__init__(modular_env,**kwargs)

    def during_reset(self, state):
        spec = state['spec']
        self.im = spec
        state['height'], state['width'] = spec.shape
        state['pw_obs'] = spec

    def clone(self, state, out_module, out_state):
        out_state['pw_obs'] = self.im
        out_state['height'], out_state['width'] = self.im.shape
        out_module.im = self.im

    def update_state(self, state):
        raise Exception("spec should not be modified during execution!")

    def make_description(self, state, description):
        pass

    def make_rgb_arrays(self, state):
        rgb = np.dstack([self.im] * 3)
        size = [10 * rgb.shape[0], 10 * rgb.shape[1]]
        rgb = better_imresize(rgb, size, interp='nearest')
        return [rgb]
