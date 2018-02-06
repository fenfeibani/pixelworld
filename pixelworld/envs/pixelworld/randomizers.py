'''
    basic set of Randomizers for PixelWorld
'''
from copy import copy
import inspect
import itertools
import random

import numpy as np

import core, objects as objects_mod


class ReseedingRandomizer(core.Randomizer):
    """Randomizer that reseeds the rng before randomizing"""
    
    def generate_seed(self):
        """generate a new rng seed value"""
        int_info = np.iinfo(np.int64)
        
        return self.rng.randint(int_info.max)
    
    def seed(self, seed=None):
        """seed the rng using a generated seed value. we seed the rng twice in
        case we first need to restore a previous rng state.
        
        Parameters
        ----------
        seed : int, optional
            the seed value for restoring a previous rng state before doing the
            real reseed
        """
        #restore a previous state
        if seed is not None: self._seed(seed)
        
        #now generate a new seed and reseed
        seed = self.generate_seed()
        self._seed(seed)
    
    def pre_randomize(self, seed):
        """reseed the rng before randomizing"""
        super(ReseedingRandomizer, self).pre_randomize(seed)
        self.seed(seed=seed)
    
    def _seed(self, seed):
        """private version of seed()
        
        Parameters
        ----------
        seed : int
            the new seed value
        """
        self.world.seed(seed)


class BatchRandomizer(core.Randomizer):
    """a Randomizer that accepts or generates randomization parameters in
    batches"""
    #the number of randomizations in each batch
    _batch_size = 100
    
    #the following store lists of randomization parameters (one element for each
    #call to randomize())
    #list of dict mapping Variant names to states
    _variant_states = None
    #list of lists of Object specs
    _object_specs = None
    #list of dicts mapping ObjectAttribute names to arrays of attribute values,
    #organized by Object index
    _object_attribute_values = None
    
    #this will store the current index in the batch
    _batch_idx = 0
    
    _auto_state = False
    
    def __init__(self, world, batch_size=None, variant_states=None,
                    object_specs=None, object_attribute_values=None, **kwargs):
        """
        Parameters
        ----------
        batch_size : int, optional
            the number of randomizations in each batch. this should be specified
            unless variant_states, object_specs, and/or object_attribute_values
            are passed explicitly rather than generated.
        variant_states : list, optional
            override the class-defined or generated variant states
        object_specs : list, optional
            override the class-defined or generated object specs
        object_attribute_values : list, optional
            override the class-defined or generated object attribute values
        """
        self.batch_size = batch_size or self._batch_size
        self.variant_states = variant_states or self._variant_states
        self.object_specs = object_specs or self._object_specs
        self.object_attribute_values = object_attribute_values or self._object_attribute_values
        
        super(BatchRandomizer, self).__init__(world, **kwargs)
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, batch_size):
        if not isinstance(batch_size, int):
            raise TypeError('batch_size must be an integer')
        
        self._batch_size = batch_size
        
        #delete the old generated/assigned batches
        self.clear_batch()
    
    @property
    def variant_states(self):
        """return (possibly after generating) the current batch of Variant
        states"""
        if self._variant_states is None:
            self.variant_states = self.generate_variant_states()
        
        return self._variant_states
    
    @variant_states.setter
    def variant_states(self, states):
        """set the current batch of Variant states. states should be a list of
        dicts, with one dict per batch index. each dict should map each active
        Variant to its state at that batch position."""
        if states is not None:
            if not isinstance(states, list):
                raise TypeError('variant_states must be a list')
            
            self._batch_size = len(states)
        
        self._variant_states = states
    
    @property
    def object_specs(self):
        """return (possibly after generating) the current batch of Object
        specs"""
        if self._object_specs is None:
            self.object_specs = self.generate_object_specs()
        
        return self._object_specs
    
    @object_specs.setter
    def object_specs(self, specs):
        """set the current batch of Object specs. specs should be a list of
        lists of Object specs, with one list per batch index."""
        if specs is not None:
            if not isinstance(specs, list):
                raise TypeError('object_specs must be a list')
            
            self._batch_size = len(specs)
        
        self._object_specs = specs
    
    @property
    def object_attribute_values(self):
        """return (possibly after generating) the current batch of
        ObjectAtttribute values"""
        if self._object_attribute_values is None:
            self.object_attribute_values = self.generate_object_attribute_values()
        
        return self._object_attribute_values
    
    @object_attribute_values.setter
    def object_attribute_values(self, values):
        """set the current batch of ObjectAttribute values. values should be a
        list of dicts, with one dict per batch index. each dict should map each
        RandomizingObjectAttribute included in self.randomized_attributes to an
        array of attribute values. each element of that array maps the Object
        with that index to its new attribute value."""
        if values is not None:
            if not isinstance(values, list):
                raise TypeError('object_attribute_values must be a list')
            
            self._batch_size = len(values)
        
        self._object_attribute_values = values
    
    @property
    def batch_idx(self):
        """the current index in the batch. a new batch is generated if this is
        >= the batch size."""
        if self._batch_idx >= self.batch_size:
            self.clear_batch()
        
        return self._batch_idx
    
    @batch_idx.setter
    def batch_idx(self, idx):
        self._batch_idx = idx
    
    def post_randomize(self):
        """increment the batch index"""
        super(BatchRandomizer, self).post_randomize()
        self.batch_idx += 1
    
    def clear_batch(self):
        """clear the batch so that a new set of parameters will be generated"""
        self._batch_idx = 0
        self.variant_states = None
        self.object_specs = None
        self.object_attribute_values = None
    
    def generate_variant_states(self):
        """generate a randomized set of Variant states. we generate every
        possible combination of Variant states and then randomly sample
        batch_size of them.
        
        Returns
        -------
        states : list
            a list of dicts of Variant state sets
        """
        
        #generate the product of all Variant states
        all_states = self._generate_variant_state_product()
        
        batch_size = self.batch_size or len(all_states)
        
        if len(all_states) == 0:  # no Variants
            return [{} for _ in xrange(batch_size)]
        
        #generate the random sample indices
        indices = range(len(all_states.values()[0]))
        self.world.rng.shuffle(indices)
        indices = indices[:batch_size]
        
        #construct the list of Variant state sets
        states = [{name:all_states[name][idx] for name in all_states} for idx in indices]
        
        #replicate if we have fewer state sets than the batch size (i.e. we
        #didn't have enough Variant state combinations to satisfy the batch
        #size)
        num_states = len(states)
        if num_states < batch_size:
            indices = range(num_states)
            
            while len(states) < batch_size:
                self.world.rng.shuffle(indices)
                states.extend([states[idx] for idx in indices[:batch_size - len(states)]])
        
        return states
    
    def generate_object_specs(self):
        """generate a randomized set of Object specification lists.
        
        Returns
        -------
        specs : list
            a list of lists of Object specs (one list for each batch index)
        """
        return [[] for _ in xrange(self.batch_size)]
    
    def generate_object_attribute_values(self):
        """generate a randomized set of sets of ObjectAttribute values.
        
        Returns
        -------
        values : list
            a list of dicts of arrays of attribute values
        """
        values = [{} for _ in xrange(self.batch_size)]
        for name in self.randomized_attributes:
            #get the Object indices
            attr = self.world.object_attributes[name]
            indices = self._get_object_attribute_indices(attr)
            
            num_objects = np.max(indices) + 1
            
            #construct the base value array
            attr_values = np.empty((num_objects, attr.ndim), dtype=attr.dtype)
            
            for value in values:
                attr_values[indices, :] = attr._get_random_values(indices)
                
                value[name] = copy(attr_values)
        
        return values
    
    def _get_variant_state(self, variant):
        return self.variant_states[self.batch_idx][variant.name]
    
    def _get_object_specs(self):
        return self.object_specs[self.batch_idx]
    
    def _get_object_attribute_values(self, attr, indices):
        try:
            values = self.object_attribute_values[self.batch_idx][attr.name][indices, :]
            
            if attr.ndim == 1:
                values = values[:, 0]
        except IndexError:  # Object has changed, revert to the old method
            return super(BatchRandomizer, self)._get_object_attribute_values(attr, indices)
        
        return values
    
    def _generate_variant_state_product(self, variants=None, fixed_states=None):
        """generate every possible combination of Variant states
        
        Parameters
        ----------
        variants : list, optional
            a list of Variants for which to generate states. defaults to the
            current set of active Variants.
        fixed_states : dict, optional
            a dict of Variant states to hold fixed. this should only be
            specified by recursive calls.
        
        Returns
        -------
        states : dict
            a dict of lists specifying every possible combination of Variant
            states
        """
        #get the Variants to generate parameters for
        if variants is None:
            variants = self.world.active_variants.values()
        else:
            variants = copy(variants)
        
        if fixed_states is None:
            fixed_states = {}
        
        if len(variants) == 0:
            return {}
        
        #initialize the dict of Variant state lists
        states = {other.name:[] for other in variants}
        
        #iterate over the first Variant's states
        variant = variants.pop(0)
        variant_states = variant.get_conditional_states(fixed_states)
        
        if len(variants) == 0:  #only one Variant
            states = {variant.name: variant_states}
        else:
            #generate the product of the other Variants' with the first
            #Variant's states
            for state in variant_states:
                #generate the sub-state product with the first Variant's state
                #fixed.
                new_fixed_states = copy(fixed_states)
                new_fixed_states[variant.name] = state
                sub_states = self._generate_variant_state_product(
                                variants=variants, fixed_states=new_fixed_states)
                
                #append the sub-states to the list
                for name,values in sub_states.iteritems():
                    states[name].extend(values)
                
                #tile the fixed state of the first Variant
                sub_length = len(sub_states.values()[0])
                states[variant.name].extend([state] * sub_length)
        
        return states


class RandomPositionsRandomizer(ReseedingRandomizer):
    _name = 'random_positions'
    _randomized_attributes = ['position']



