from __future__ import print_function
from __future__ import absolute_import

import collections
import numbers

import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init as LI
import numpy as np
from rllab.misc import ext, special
from rllab.misc.overrides import overrides
from rllab.core.lasagne_layers import OpLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import GRUNetwork, GRULayer, MLP
from rllab.core.serializable import Serializable
from rllab.distributions.recurrent_categorical import RecurrentCategorical
from rllab.policies.base import StochasticPolicy
from rllab.policies.categorical_gru_policy import CategoricalGRUPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.spaces import Discrete, Box
import theano.tensor as TT
import theano

from pixelworld.spaces_rllab import NamedBox, NamedDiscrete

debug = False

def compute_output_b_init(action_names, output_b_init=None, weight_signal=1.0,
                          weight_nonsignal=1.0, weight_smc=1.0):
    signals = set()
    nonsignals = set()
    smcs = set()
    for name in action_names:
        if len(name) > 1 and name[0] == '+':
            smcs.add(name)
        elif len(name) > 3 and name[:3] == 'SIG':
            signals.add(name)
        else:
            nonsignals.add(name)

    sum_weights = float(weight_signal + weight_nonsignal + weight_smc)
    prob_signal = weight_signal / sum_weights

    # Do this to ensure that the expected episode length is always 
    #    1/prob_signal = sum_weights / weight_signal
    if len(smcs) == 0:
        prob_nonsignal = (weight_nonsignal + weight_smc) / sum_weights
        prob_smc = "N/A"
    else:
        prob_nonsignal = weight_nonsignal / sum_weights
        prob_smc = weight_smc / sum_weights

    if output_b_init is None:
        if debug: print("  prob_signal %s prob_nonsignal %s prob_smc %s expected length %s" % 
            (prob_signal, prob_nonsignal, prob_smc, 1/prob_signal))

        output_b_init = np.zeros(len(action_names))
        probs = []
        for name in action_names:
            if name in smcs:
                probs.append(float(prob_smc)/len(smcs))
            elif name in signals:
                probs.append(float(prob_signal)/len(signals))
            else:
                probs.append(float(prob_nonsignal)/len(nonsignals))
        output_b_init = np.log(np.array(probs, dtype=np.float) / sum(probs))
    elif isinstance(output_b_init, numbers.Number):
        output_b_init = LI.Constant(output_b_init)
    elif isinstance(output_b_init, collections.Sequence):
        output_b_init = np.asarray(output_b_init, dtype=np.float)

    if debug:
        print("compute_output_b_init:")
        if isinstance(output_b_init, np.ndarray):
            print("         b       prob   name")
            norm = np.exp(output_b_init).sum()
            for name, b in zip(action_names, output_b_init):
                if name in smcs:
                    print("   [smc] %.4f %.4f %s" % (b, np.exp(b)/norm, name))
                elif name in signals:
                    print("   [sig] %.4f %.4f %s" % (b, np.exp(b)/norm, name))
                else:
                    print("  [~sig] %.4f %.4f %s" % (b, np.exp(b)/norm, name))
        else:
            print("   names:", action_names)
            print("   binit:", output_b_init)
    
    return output_b_init

# Modified from RLLab MLPPolicy
class InitCategoricalMLPPolicy(CategoricalMLPPolicy):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.tanh,
            output_b_init=None,
            weight_signal=1.0,
            weight_nonsignal=1.0, 
            weight_smc=1.0):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Discrete)
        output_b_init = compute_output_b_init(env_spec.action_space.names,
            output_b_init, weight_signal, weight_nonsignal, weight_smc)

        prob_network = MLP(
            input_shape=(env_spec.observation_space.flat_dim,),
            output_dim=env_spec.action_space.n,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=NL.softmax,
            output_b_init=output_b_init
        )
        super(InitCategoricalMLPPolicy, self).__init__(env_spec, hidden_sizes,
            hidden_nonlinearity, prob_network)


# Modified from RLLab GRUNetwork
class InitGRUNetwork(object):
    def __init__(self, input_shape, output_dim, hidden_dim, hidden_nonlinearity=NL.rectify,
                 output_nonlinearity=None, name=None, input_var=None,
                 output_b_init=LI.Constant(0.)):
        l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var)
        l_step_input = L.InputLayer(shape=(None,) + input_shape)
        l_step_prev_hidden = L.InputLayer(shape=(None, hidden_dim))
        l_gru = GRULayer(l_in, num_units=hidden_dim, hidden_nonlinearity=hidden_nonlinearity,
                         hidden_init_trainable=False)
        l_gru_flat = L.ReshapeLayer(
            l_gru, shape=(-1, hidden_dim)
        )
        l_output_flat = L.DenseLayer(
            l_gru_flat,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            b=output_b_init
        )
        l_output = OpLayer(
            l_output_flat,
            op=lambda flat_output, l_input:
            flat_output.reshape((l_input.shape[0], l_input.shape[1], -1)),
            shape_op=lambda flat_output_shape, l_input_shape:
            (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
            extras=[l_in]
        )
        l_step_hidden = l_gru.get_step_layer(l_step_input, l_step_prev_hidden)
        l_step_output = L.DenseLayer(
            l_step_hidden,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            W=l_output_flat.W,
            b=l_output_flat.b,            
        )

        self._l_in = l_in
        self._hid_init_param = l_gru.h0
        self._l_gru = l_gru
        self._l_out = l_output
        self._l_step_input = l_step_input
        self._l_step_prev_hidden = l_step_prev_hidden
        self._l_step_hidden = l_step_hidden
        self._l_step_output = l_step_output

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_hidden_layer(self):
        return self._l_step_prev_hidden

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

# Modified from RLLab GRUPolicy
class InitCategoricalGRUPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32,),
            state_include_action=True,
            hidden_nonlinearity=NL.tanh,
            output_b_init=None,
            weight_signal=1.0,
            weight_nonsignal=1.0, 
            weight_smc=1.0):
        """
        :param env_spec: A spec for the env.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        assert isinstance(env_spec.action_space, Discrete)
        Serializable.quick_init(self, locals())
        super(InitCategoricalGRUPolicy, self).__init__(env_spec)

        assert len(hidden_sizes) == 1

        output_b_init = compute_output_b_init(env_spec.action_space.names,
            output_b_init, weight_signal, weight_nonsignal, weight_smc)

        if state_include_action:
            input_shape = (env_spec.observation_space.flat_dim + env_spec.action_space.flat_dim,)
        else:
            input_shape = (env_spec.observation_space.flat_dim,)

        prob_network = InitGRUNetwork(
            input_shape=input_shape,
            output_dim=env_spec.action_space.n,
            hidden_dim=hidden_sizes[0],
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=NL.softmax,
            output_b_init=output_b_init
        )

        self._prob_network = prob_network
        self._state_include_action = state_include_action

        self._f_step_prob = ext.compile_function(
            [
                prob_network.step_input_layer.input_var,
                prob_network.step_prev_hidden_layer.input_var
            ],
            L.get_output([
                prob_network.step_output_layer,
                prob_network.step_hidden_layer
            ])
        )

        self._prev_action = None
        self._prev_hidden = None
        self._hidden_sizes = hidden_sizes
        self._dist = RecurrentCategorical(env_spec.action_space.n)

        self.reset()

        LasagnePowered.__init__(self, [prob_network.output_layer])

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        n_batches, n_steps = obs_var.shape[:2]
        obs_var = obs_var.reshape((n_batches, n_steps, -1))
        if self._state_include_action:
            prev_action_var = state_info_vars["prev_action"]
            all_input_var = TT.concatenate(
                [obs_var, prev_action_var],
                axis=2
            )
        else:
            all_input_var = obs_var
        return dict(
            prob=L.get_output(
                self._prob_network.output_layer,
                {self._prob_network.input_layer: all_input_var}
            )
        )

    def reset(self):
        self._prev_action = None
        self._prev_hidden = self._prob_network.hid_init_param.get_value()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        if self._state_include_action:
            if self._prev_action is None:
                prev_action = np.zeros((self.action_space.flat_dim,))
            else:
                prev_action = self.action_space.flatten(self._prev_action)
            all_input = np.concatenate([
                self.observation_space.flatten(observation),
                prev_action
            ])
        else:
            all_input = self.observation_space.flatten(observation)
            # should not be used
            prev_action = np.nan
        probs, hidden_vec = [x[0] for x in self._f_step_prob([all_input], [self._prev_hidden])]
        action = special.weighted_sample(probs, xrange(self.action_space.n))
        self._prev_action = action
        self._prev_hidden = hidden_vec
        agent_info = dict(prob=probs)
        if self._state_include_action:
            agent_info["prev_action"] = prev_action
        return action, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self._dist

    @property
    def state_info_keys(self):
        if self._state_include_action:
            return ["prev_action"]
        else:
            return []
