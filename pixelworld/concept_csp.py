"""Generate/test environments using logically expressed concepts"""

from __future__ import print_function
from __future__ import absolute_import

from collections import defaultdict
import copy
from itertools import product
from pprint import pprint
import random
from sys import stdout

from constraint import (Problem, Solver, InSetConstraint, FunctionConstraint, 
                        AllEqualConstraint, AllDifferentConstraint)
import diskcache
import numpy as np
import os

from pixelworld.concept_parser import parse_concept
from pixelworld.csp import (generate_random_solutions, is_consistent, print_csp, 
          InfiniteIntegralDomain, NoOverlapConstraint, PairwiseSpatialConstraint, 
          LogPolarUnarySpatialConstraint)

from pixelworld.misc import make_hashable
from pixelworld.pattern_generator import generate_patterns
from pixelworld.scene_annotation import generate_annotations

# Converting relation clauses into constraints
# --------------------------------------------

# The CSP representation has a set of variables for each term in the concept.
# Specifically, for a term x:
#   x.is_self       whether this is a self
#                       this is constrained to true for the term self
#                       and constrained to false otherwise
#   x.color
#   x.shape         pixel, blob, horz_wall, vert_wall, container, ....
#   x.shape_idx     0...MAX_NUM_SHAPES
#   x.x
#   x.y
#   x.pushable      start with set to false?
#   x.is_target     whether object is a bringabout target 
# See clause_to_csp below for more details.


#
# Colors
#
WHITE = 1
RED = 2
BLUE = 3
GREEN = 4
YELLOW = 5
MAGENTA = 6
PURPLE = 7
ORANGE = 8


# Available object colors   
COLORS = {
   'white': WHITE, 
   'red': RED,
   'blue': BLUE,
   'green': GREEN,
   'yellow': YELLOW,
   #'magenta': MAGENTA,
   #'purple': PURPLE,
   'orange': ORANGE
}

SPATIAL_RELATIONS = [
    'adjacent',
    'touches',
    'near',   # <= 2 pixels apart
    'far',    # > 4 pixels apart
    'left_of', # coarse spatial relations
    'right_of',
    'above',
    'below',  
    'dir_left_of', # fine spatial relations
    'dir_right_of',
    'dir_above',
    'dir_below',  
    'h_aligned',
    'v_aligned',
    'bottom_aligned',
    'top_aligned',
    'left_aligned',
    'right_aligned',
    'inside',
    'inside_supported',
    'taller',
    'wider',
    'same_height',
    'same_width'
]

LOCATIONS = [
    # Locations of object center.
    # Using log-polar coordinates from the center of the PixelWorld.
    # Caveat: it is user's responsibility to specify valid location of a particular object.
    #   E.g., There's no way to put center of large objects close to the boundary of PW.

    # radial distance: if settings['exclusive_rbins'] is False, the ranges become inclusive.
    'almost_center',        # By default, [0, 2)
    'near_center',          # [2, 4)
    'middle_from_center',   # [4, 8)
    'far_from_center',      # [8, 16). See csp.py to add more.

    # orientation: nearest neighbor angular bins; center of bins are 45-degree apart, clockwise starting from 3'o clock.
    'right_from_center',
    'lower_right_from_center',
    'down_from_center',
    'lower_left_from_center',
    'left_from_center',
    'upper_left_from_center',
    'up_from_center',
    'upper_right_from_center',
]


def make_default_settings():
    """Creates a default settings data structure."""
    default_settings = {
        'height': 24, 
        'width': 24, 
        'max_box_height': 7,
        'max_box_width': 7,
        'max_container_height': 5,
        'max_container_width': 9,
        'default_num_samples': 20,
        'fixed_floor': False,
        'floor_height': 3,
        'infinite_position_domain': False,
        'frame': False, # indicates presence of PixelWorld frame
        'frame_color': PURPLE,
        'padding': 0, # padding around outside edge
        'colors': COLORS.values(),  
        'check_overlap': True,
        'allow_pushable': False, # Whether to allow objects the option of being pushable
        'allow_targets': False, # Whether to allow use of the is_target attribute
        'add_self': True,
        'make_self_red_pixel': True,
        'self_color_is_unique': False,
        'objects_are_white': False,
        'objects_are_small_blobs': False,
        'self_grips': False,  # True if the self can grip/ungrip other objects
        }
    return default_settings




#
# Shapes
#

def make_shapes(settings):
    """Create a dictionary of shapes.
    default_num_samples: int (number of samples used for
    most shapes that do not have special restrictions on that number)
    """

    framesize = 2 if settings['frame'] else 0
    maxH = settings['height'] - framesize
    maxW = settings['width'] - framesize
    boxH = settings['max_box_height']  
    boxW = settings['max_box_width']  
    conH = settings['max_container_height']
    conW = settings['max_container_width']

    default_num_samples = settings['default_num_samples']

    shapes = {
        'pixel': [np.array([[1]], dtype=np.int)],
        'small_blob': generate_patterns('blob', max_width=3, max_height=3, num_samples=20),
        'blob': generate_patterns('blob', max_width=5, max_height=5, num_samples=30),
        'd_slob': generate_patterns('blob', max_width=3, max_height=3, num_samples=20), # distractor small blob
        'large_blob': generate_patterns('blob', max_width=7, max_height=7, num_samples=50),
        'convex_blob' : generate_patterns('convex_blob', max_width=7, max_height=7, num_samples=50),
        
        'line': generate_patterns('line', max_width=boxW, max_height=boxH, min_length=2, num_samples=boxW-2 + boxH-2),
        'hline': generate_patterns('hline', max_width=boxW, min_width=2, num_samples=boxW-2),
        'vline': generate_patterns('vline', max_height=boxH, min_height=2, num_samples=boxH-2),
        'floor_shape': generate_patterns('hline',max_width=maxW,min_width=maxW,num_samples=1),
        'wall': generate_patterns('vline',max_height=maxH,min_height=maxH,num_samples=1),
        
        'sym_container': generate_patterns('container', max_width=conW, max_height=conH,
                                           num_samples=default_num_samples, clip_corners=False, symmetric_walls=True),
        'rect_container': generate_patterns('container', max_width=conW, max_height=conH,
                                            num_samples=default_num_samples, clip_corners=False),
        'container': generate_patterns('container', max_width=conW, max_height=conH,
                                       num_samples=default_num_samples, clip_corners=True),
        'noncontainer': generate_patterns('container', has_hole=True, max_width=conW, max_height=conH,
                                          num_samples=default_num_samples, clip_corners=True),
        'left_container': generate_patterns('container', max_width=conW, max_height=conH,
                                            num_samples=default_num_samples, clip_corners=True, orientation="left"),
        'right_container': generate_patterns('container', max_width=conW, max_height=conH,
                                             num_samples=default_num_samples, clip_corners=True, orientation="right"),
        'top_container': generate_patterns('container', max_width=conW, max_height=conH,
                                           num_samples=default_num_samples, clip_corners=True, orientation="top"),
        'left_noncontainer': generate_patterns('container', has_hole=True, max_width=conW, max_height=conH,
                                               num_samples=default_num_samples, clip_corners=True, orientation="left"),
        'right_noncontainer': generate_patterns('container', has_hole=True, max_width=conW, max_height=conH,
                                                num_samples=default_num_samples, clip_corners=True, orientation="right"),
        'top_noncontainer': generate_patterns('container', has_hole=True, max_width=conW, max_height=conH,
                                              num_samples=default_num_samples, clip_corners=True, orientation="top"),
        
        'small_container': [np.array([[1,0,0,1],
                                      [1,1,1,1]])],
        
        'small_table':     [np.array([[1,1,1,1],
                                      [1,0,0,1]])],
               
        'lower_left_corner': generate_patterns('corner', orientation='lower_left', 
                                               num_samples=20, max_width=conW, max_height=conH,clip_corners=True),
        'lower_right_corner': generate_patterns('corner', orientation='lower_right', 
                                                num_samples=20, max_width=conW, max_height=conH,clip_corners=True),
        'upper_left_corner': generate_patterns('corner', orientation='upper_left', 
                                               num_samples=20, max_width=conW, max_height=conH,clip_corners=True),
        'upper_right_corner': generate_patterns('corner', orientation='upper_right', 
                                                num_samples=20, max_width=conW, max_height=conH,clip_corners=True), 
        
        'box': generate_patterns('rect_enclosure', max_width=boxW, max_height=boxH,
                                 num_samples=10, clip_corners=False),
        'enclosure': generate_patterns('rect_enclosure', max_width=boxW, max_height=boxH,
                                       min_width=4, min_height=4, num_samples=20, clip_corners=True),
        'nonenclosure': generate_patterns('rect_enclosure', has_hole=True, max_width=boxW, max_height=boxH,
                                        min_width=4, min_height=4, num_samples=20, clip_corners=True), 
        
        'cross': generate_patterns('cross', max_width=boxW, max_height=boxH, num_samples=20),
        'chair': generate_patterns('chair', max_width=5, max_height=8, num_samples=20),
        'table': generate_patterns('table', max_width=8, max_height=5, num_samples=20),
        }

    return shapes


# Add annotations for selected shapes
def make_annotations(shapes):
    """Creates annotations for a selection of the specified shapes."""

    annotations = {
        'box' : generate_annotations(shapes['box'], 'enclosure'),    
        'enclosure' : generate_annotations(shapes['enclosure'], 'enclosure'),
        'nonenclosure' : generate_annotations(shapes['nonenclosure'], 'holey_enclosure'),  
        'sym_container' : generate_annotations(shapes['sym_container'], 'containment'),
        'rect_container' : generate_annotations(shapes['rect_container'], 'containment'),
        'container' : generate_annotations(shapes['container'], 'containment'),
        'noncontainer' : generate_annotations(shapes['noncontainer'], 'holey_containment'),
        'left_container' : generate_annotations(shapes['left_container'], 'left_containment'),
        'right_container' : generate_annotations(shapes['right_container'], 'right_containment'),
        'top_container' : generate_annotations(shapes['top_container'], 'top_containment'),
        'left_noncontainer' : generate_annotations(shapes['left_noncontainer'], 'left_noncontainment'),  
        'right_noncontainer' : generate_annotations(shapes['right_noncontainer'], 'right_noncontainment'),
        'top_noncontainer' : generate_annotations(shapes['top_noncontainer'], 'top_noncontainment'),
        'lower_left_corner': generate_annotations(shapes['lower_left_corner'], 'holey_containment'),
        'lower_right_corner': generate_annotations(shapes['lower_right_corner'], 'holey_containment')
        }

    return annotations


def make_relation_processors(settings): 
    """Creates a dictionary of relation processors."""
    shapes = settings['shape_templates']
    fixed_floor = settings['fixed_floor']
    floor_height = settings['floor_height']
    height = settings['height']

    # Dictionary of the form
    #   relation: [arity, add_constraint_function, kwargs]
    relation_processors = {
        'pushable': [1, add_in_set, {'key': 'pushable', 'vals': [True]}],
        'target': [1, add_in_set, {'key':'is_target', 'vals':[True]}],
        'whiteblob': [1, add_in_sets, {'keys':['shape','color'], 'vals':[['small_blob'],[COLORS['white']]]}],
        
        'same_color': [None, add_all_equal, {'keys': ['color']}],
        'diff_color': [None, add_all_different, {'keys': ['color']}],
        'same_kind': [None, add_all_equal, {'keys': ['shape']}],
        'diff_kind': [None, add_all_different, {'keys': ['shape']}], 
        'same_shape': [None, add_all_equal, {'keys': ['shape','shape_idx']}],
        'h_between': [3, add_between, {'orientation': 'horizontal'}],
        'v_between': [3, add_between, {'orientation': 'vertical'}],
        # diff_shape: TODO - this is a bit trickier
        'on_top': [2, add_on, {'relation': 'above'}],
        'on_bottom': [2, add_on, {'relation':'below'}],   
        'on_left': [2, add_on, {'relation':'left_of'}],
        'on_right': [2, add_on, {'relation':'right_of'}],
        }

    for shape in shapes.keys():
        relation_processors[shape] = [1, add_in_set, {'key': 'shape', 'vals': [shape]}]

    for color in COLORS.keys():
        relation_processors[color] = [1, add_in_set, {'key': 'color', 'vals': [COLORS[color]]}]

    for rel in SPATIAL_RELATIONS:
        relation_processors[rel] = [2, add_pairwise_spatial, {'relation': rel}]

    for rel in LOCATIONS:
        relation_processors[rel] = [1, add_logpolar_unary_spatial, {'relation': rel}]

    if fixed_floor:
        relation_processors['floor'] = [1, add_in_sets, {'keys':['shape', 'y'], 'vals':[['floor_shape'],[height - floor_height]]}]
    else:
        relation_processors['floor'] = [1, add_in_set, {'key': 'shape', 'vals': ['floor_shape']}]

    # an object is the union over various shapes
    relation_processors['object'] = \
        [1, add_in_set, {'key': 'shape', 'vals': ['blob', 'small_blob', 'convex_blob', 'container', 'noncontainer', 'enclosure', 'nonenclosure']}]

    # a potential container is either a container, noncontainer, or lower_corner (supports "inside" relation)
    relation_processors['potential_container'] = \
        [1, add_in_set, {'key': 'shape', 'vals': ['container','noncontainer','lower_left_corner','lower_right_corner']}]

    relation_processors['corner'] = \
        [1, add_in_set, {'key': 'shape', 'vals': ['lower_left_corner','lower_right_corner','upper_left_corner','upper_right_corner']}]

    relation_processors['lower_corner'] = \
        [1, add_in_set, {'key': 'shape', 'vals': ['lower_left_corner','lower_right_corner']}]

    return relation_processors



def populate_settings(settings):
    """Populates the given settings data structure with derived fields."""

    shapes = make_shapes(settings) #, default_num_samples=settings['default_num_samples'])
    settings['shape_templates'] = shapes
    settings['shape_annotations'] = make_annotations(shapes)

    return settings


#
# Relation definitions
#

def add_in_set(problem, arguments, key, vals, settings):
    var = "%s.%s" % (arguments[0], key)
    problem.addConstraint(InSetConstraint(vals), [var])


def add_in_sets(problem, arguments, keys, vals, settings):
    for (key,val) in zip(keys,vals):
        var = "%s.%s" % (arguments[0], key)
        problem.addConstraint(InSetConstraint(val), [var])

def add_func(problem, arguments, func, keys, settings):
    variables = ['%s.%s' % (arg, key) for arg in arguments for key in keys]
    problem.addConstraint(FunctionConstraint(func), variables)

def add_pairwise_spatial(problem, arguments, relation, settings):
    keys = ['x', 'y', 'shape', 'shape_idx']
    variables = ['%s.%s' % (arg, key) for arg in arguments for key in keys]
    problem.addConstraint(PairwiseSpatialConstraint(settings, mode=relation), variables)    

def add_logpolar_unary_spatial(problem, arguments, relation, settings):
    keys = ['x', 'y', 'shape', 'shape_idx']
    variables = ['%s.%s' % (arguments[0], key) for key in keys]
    problem.addConstraint(LogPolarUnarySpatialConstraint(settings, mode=relation), variables)    

def add_all_equal(problem, arguments, keys, settings):
    for key in keys:
        variables = ['%s.%s' % (arg, key) for arg in arguments]
        problem.addConstraint(AllEqualConstraint(), variables)

def add_all_different(problem, arguments, keys, settings):
    """ If multiple keys, this requires all keys to differ """
    for key in keys:
        variables = ['%s.%s' % (arg, key) for arg in arguments]
        problem.addConstraint(AllDifferentConstraint(), variables )


def add_between(problem, arguments, settings, orientation="horizontal"):
    keys = ['x', 'y', 'shape', 'shape_idx']
    assert len(arguments) == 3        
    if orientation == "horizontal": 
        rel1 = "left_of"
        rel2 = "h_aligned"
    elif orientation == "vertical":
        rel1 = "above"
        rel2 = "v_aligned"
    else:
        raise Exception("Invalid orientation " + orientation)
    center = arguments[0]
    obj1 = arguments[1]
    obj2 = arguments[2]
    variables1 = ['%s.%s' % (arg, key) for arg in [obj1,center] for key in keys]    
    problem.addConstraint(PairwiseSpatialConstraint(settings,mode=rel1), variables1)
    problem.addConstraint(PairwiseSpatialConstraint(settings,mode=rel2), variables1)       
    variables2 = ['%s.%s' % (arg, key) for arg in [center,obj2] for key in keys]    
    problem.addConstraint(PairwiseSpatialConstraint(settings,mode=rel1), variables2)
    problem.addConstraint(PairwiseSpatialConstraint(settings,mode=rel2), variables2)


def add_on(problem, arguments, settings, relation="above"):
    keys = ['x', 'y', 'shape', 'shape_idx']  
    assert len(arguments) == 2  
    variables = ['%s.%s' % (arg, key) for arg in arguments for key in keys]  
    problem.addConstraint(PairwiseSpatialConstraint(settings,mode="adjacent"),variables)
    problem.addConstraint(PairwiseSpatialConstraint(settings,mode=relation),variables)        


# Override these settings (for generation only) by passing settings dict to generate_dataset
DEFAULT_SETTINGS = make_default_settings()

# Objects in PixelWorld which should be ignored by the csp logic
IGNORED_OBJECTS = ['frame','grip']


def clause_to_csp(terms, clauses, settings):
    """
    Convert a clause into a CSP which tests whether the clause is satisfied.

    terms: [str]
    clauses: [clause]
        clause: (relation, arguments)
        relation: str
        arguments: (str)

    See also concept_parser.py:parse_concept.
    """
    problem = Problem()

    # Check consistency of settings
    if settings['width'] <= 0:
        raise Exception("settings['width'] must be a positive integer.")
    if settings['height'] <= 0:
        raise Exception("settings['height'] must be a positive integer.")

    # Identify position variable domains
    if settings['infinite_position_domain']:
        x_domain = InfiniteIntegralDomain()
        y_domain = InfiniteIntegralDomain()
    else:
        x_domain = range(settings['width'])
        y_domain = range(settings['height'])        

    # Create variables for each term
    for term in terms:
        problem.addVariable(term + '.is_self', [False, True])
        problem.addVariable(term + '.x', x_domain)
        problem.addVariable(term + '.y', y_domain)
        shapes = settings['shape_templates'].keys()
        max_shape_idx = max([len(x) for x in settings['shape_templates'].values()])
        problem.addVariable(term + '.shape', shapes)
        problem.addVariable(term + '.shape_idx', range(max_shape_idx))
        problem.addVariable(term + '.color', settings['colors'])
        problem.addVariable(term + '.pushable',
            [True, False] if settings['allow_pushable'] else [False])
        problem.addVariable(term+'.is_target',
            [True,False] if settings['allow_targets'] else [False])
        # TODO: remove variables for special-case terms?

    # Create constraints for each clause

    relation_processors = make_relation_processors(settings) 
    for relation, arguments in clauses:
        arity, fn, kwargs = relation_processors[relation]
        if arity is not None and len(arguments) != arity:
            raise Exception("Relation %s has arity %s but used with %s arguments" % 
                (relation, arity, len(terms)))
        kwargs['problem'] = problem
        kwargs['arguments'] = arguments
        kwargs['settings'] = settings
        fn(**kwargs)

    # Extra constraints
    if settings['add_self']:
        problem.addConstraint(InSetConstraint([True]), ['self.is_self'])
        problem.addConstraint(InSetConstraint([False]), ['self.is_target'])
    for term in terms:
        if term != 'self':
            problem.addConstraint(InSetConstraint([False]), [term+'.is_self'])
            if settings['objects_are_small_blobs']:
                problem.addConstraint(InSetConstraint(['small_blob']), [term+'.shape'])
            if settings['objects_are_white']:
                problem.addConstraint(InSetConstraint({WHITE}), [term+'.color'])            
            if settings['self_color_is_unique']:
                problem.addConstraint(AllDifferentConstraint(), ['self.color',term+'.color'])


    if settings['add_self'] and settings['make_self_red_pixel']:
        problem.addConstraint(InSetConstraint({RED}), ['self.color'])
        problem.addConstraint(InSetConstraint(['pixel']), ['self.shape'])
        problem.addConstraint(InSetConstraint({0}), ['self.shape_idx'])

    if settings['check_overlap']:
        frame_size = 1 if settings['frame'] else 0
        constraint = NoOverlapConstraint(height=settings['height'], width=settings['width'],
                                         frame=frame_size, padding=settings['padding'], 
                                         templates=settings['shape_templates'])
        variables = ["%s.%s" % (term, key) for term in terms 
                     for key in ['x','y','shape','shape_idx']]
        problem.addConstraint(constraint, variables)

    return problem


def expand_macro_clause(clause, settings):
    """
    If clause is a macro, expand it making the appropriate substitutions.
    Otherwise returns this as a sole positive clause.
    """
    macros = settings.get('macros', {})
    relation, arguments = clause
    if relation in macros:
        m_terms, m_positive_clauses, m_negative_clause_lists, _ = \
            parse_concept(macros[relation], settings=settings)

        # Go through +ve/-ve clauses and rename vars.
        substitution_map = dict(zip(m_terms, arguments))
        substitution_map['self'] = 'self'
        positive_clauses = []
        negative_clause_lists = []
        for clause in m_positive_clauses:
            rel, args = clause
            args = tuple([substitution_map[arg] for arg in args])
            positive_clauses.append((rel, args))

        for negative_clause_list in m_negative_clause_lists:
            new_negative_clause_list = []
            for clause in negative_clause_list:
                rel, args = clause
                args = tuple([substitution_map[arg] for arg in args])
                new_negative_clause_list.append((rel, args))
            negative_clause_lists.append(new_negative_clause_list)
    else:
        positive_clauses = [clause]
        negative_clause_lists = []

    return positive_clauses, negative_clause_lists


def expand_macros1(positive_clauses, negative_clause_lists, settings):
    """
    Expand macros in parsed concept once. Returns new 
        positive_clauses, negative_clause_lists
    and whether all macros have been fully expanded.
    """
    new_positive_clauses = []
    new_negative_clause_lists = []
    macros = settings.get('macros', {})

    for clause in positive_clauses:
        m_positive_clauses, m_negative_clause_lists = expand_macro_clause(clause, settings)
        new_positive_clauses.extend(m_positive_clauses)
        new_negative_clause_lists.extend(m_negative_clause_lists)

    for negative_clause_list in negative_clause_lists:
        new_negative_clause_list = []
        for clause in negative_clause_list:
            m_positive_clauses, m_negative_clause_lists = expand_macro_clause(clause, settings)
            new_negative_clause_list.extend(m_positive_clauses)
            if len(m_negative_clause_lists) > 0:
                raise Exception("Cannot negate a macro with negated clauses")
        new_negative_clause_lists.append(new_negative_clause_list)

    for clause in new_positive_clauses:
        if clause[0] in macros:
            return new_positive_clauses, new_negative_clause_lists, False

    for clause_list in new_negative_clause_lists:
        for clause in clause_list:
            if clause[0] in macros:
                return new_positive_clauses, new_negative_clause_lists, False

    return new_positive_clauses, new_negative_clause_lists, True


def expand_macros(positive_clauses, negative_clause_lists, settings):
    """Repeatedly expand macros until they're all expanded, throwing an
    exception if this requires too many iterations."""
    for i in range(100):
        positive_clauses, negative_clause_lists, fully_expanded =\
            expand_macros1(positive_clauses, negative_clause_lists, settings)
        if fully_expanded:
            return positive_clauses, negative_clause_lists
    raise Exception("Macro expansion took too many iterations! Perhaps there's a loop?")


def concept_to_csps(concept, settings):
    """
    A concept consistent of a positive clause and a list of negated clauses.
    See also clause_to_csp.

    Returns a dictionary:
        'positive': csp corresponding to positive clause
        'negative': list of csps corresponding to negative clauses
        'terms': terms in concept (variables quantified over)
        'logic': given concept's logical expression
    and any extra keys included in concept if its a dictionary (see parse_concept).
    """
    terms, positive_clauses, negative_clause_lists, meta = parse_concept(concept, settings=settings)


    positive_clauses, negative_clause_lists = expand_macros(
        positive_clauses, negative_clause_lists, settings)

    positive_csp = clause_to_csp(terms, positive_clauses, settings)

    # Negative CSPs don't need extra constraints required for generation
    settings = copy.deepcopy(settings)
    settings['check_overlap'] = False
    negative_csps = [clause_to_csp(terms, clauses, settings) 
                        for clauses in negative_clause_lists]
    csps = meta
    csps['positive'] = positive_csp 
    csps['negatives'] = negative_csps
    csps['terms'] = terms
    return csps


# Interconverting CSP assignments and PixelWorld objects
# ------------------------------------------------------

def assignments_to_terms(assignments):
    # Infer terms from variables of the form '<term>.x'.
    terms = [k[:-2] for k in assignments.keys() if k[-2:] == '.x']
    return terms

def assignments_to_objects(assignments, settings, terms=None):
    if terms is None:
        terms = assignments_to_terms(assignments)
    objects = []
    bindings = {}
    for term in sorted(terms):
        is_self = assignments[term + '.is_self']
        shape = assignments[term + '.shape']
        shape_idx = assignments[term + '.shape_idx']
        pushable = assignments[term + '.pushable']
        is_target = assignments[term + '.is_target']
        pw_attributes = {
            'color': assignments[term + '.color'],
            'position': (assignments[term + '.y'], assignments[term + '.x']),
            'meta': {
                'is_self': is_self,
                'shape': shape,
                'shape_idx': shape_idx, 
                'pushable': pushable,
                'is_target': is_target
            },
        }
        
        compound_shape = shape != 'pixel'
        
        if is_self:
            pw_kind = 'self_big' if compound_shape else 'self'
            
            if settings['self_grips']:
                pw_attributes['grips'] = True
        elif pushable:
            pw_kind = 'complex' if compound_shape else 'basic'
        else:
            pw_kind = 'immoveable_big' if compound_shape else 'immoveable'
        
        if compound_shape:
            shape_idx = shape_idx % len(settings['shape_templates'][shape])
            template = settings['shape_templates'][shape][shape_idx]

            # convert template into list of (y,x) coordinates of non-zero elems
            pw_attributes['shape'] = zip(*np.where(template!=0))

        bindings[term] = len(objects)
        objects.append([pw_kind, pw_attributes])
    return objects, bindings

def object_state_to_assignments(object_state, bindings, assignments=None):
    if assignments is None:
        assignments = {}
    for term in sorted(bindings):
        props = object_state[bindings[term]]
        meta = props['meta']
        # TODO: better way to detect self object?
        assignments[term + '.is_self'] = meta['is_self']
        assignments[term + '.y'], assignments[term + '.x'] = props['position']
        assignments[term + '.color'] = props['color']
        assignments[term + '.shape'] = meta['shape']
        assignments[term + '.shape_idx'] = meta['shape_idx']
        # TODO: better way to detect pushable objects?
        assignments[term + '.pushable'] = meta['pushable']
        assignments[term + '.is_target'] = meta['is_target']
    return assignments

def objects_to_object_state(objects):
    object_state = []
    for name, object_attributes in objects:
        object_attributes['name'] = name
        object_state.append(object_attributes)
    return object_state


# Generating pixel world objects from concepts
# --------------------------------------------

def concept_to_objects(concept, seed, settings, min_n=None, max_n=None, use_iter=False,
                    yield_intermediate=False):
    """Generates pixelworld objects which satisfy a concept.

    seed: if None generation is non-random, otherwise random with this seed
    min_n: if not None, the minimum number generate_dataset
    max_n: if not None, the minimum number generate_dataset
    use_iter: whether to use problem.solutionIter instead of repeatedly calling
       problem.solution (see csp.py:generate_random_solutions).
    yield_intermediate: whether to yield intermediate results.
        if True, yields  objects, assignments, bindings
        if False, yields objects
    """
    csps = concept_to_csps(concept, settings=settings)
    n = 0
    for assignments in generate_random_solutions(csps['positive'], seed, use_iter):
        good = True
        for csp in csps['negatives']:
            if is_consistent(csp, assignments):
                good = False
                break
        if good:
            objects, bindings = assignments_to_objects(assignments, settings=settings, terms=csps['terms'])
            if yield_intermediate:
                yield objects, assignments, bindings
            else:
                yield objects
            n += 1
            if max_n is not None and n >= max_n: 
                break
    if min_n is not None and n < min_n:
        raise Exception("Requested at least %s solutions, but can only generate %s." % (min_n, n))


# TODO: can the following be better implemented as a modified Solver?
# TODO: is this fast enough for bring-about concepts?
#   Speed up with an iterative version? A lazy version?
def concept_holds(concept, object_state, bindings=None, csps=None, settings={}):
    """Verifies whether set of pixelworld objects satisfies a concept.
    If bindings is non-empty, only searches those extending it."""
    if csps is None:
        csps = concept_to_csps(concept, settings=settings)
    
    terms = csps['terms']

    # Filter out any sub-objects pixelworld creates
    object_state2 = []
    for obj in object_state:
        if 'head_of_family' not in obj or obj['head_of_family'] == obj['id']:
            object_state2.append(obj)
    object_state = object_state2


    # Filter out any object that should be ignored (e.g. frame)
    object_state2 = [obj for obj in object_state if not obj['name'] in IGNORED_OBJECTS]
    object_state = object_state2


    object_indices = set(range(len(object_state)))

    # Terms must be distinct, so check there are enough objects first
    if len(terms) > len(object_indices):
        return False

    if bindings is None:
        bindings = {}
        assignments = {}
    else:
        assignments = object_state_to_assignments(object_state, bindings)
        terms = [term for term in terms if term not in bindings]
        object_indices.difference_update(bindings.values())
        if len(terms) == 0:
            if not is_consistent(csps['positive'], assignments):
                return False
            for csp in csps['negatives']:
                if is_consistent(csp, assignments):
                    return False
            return True                

    def verify_concept_recursive(term_index, object_indices_left):
        # Uses terms, bindings, assignments from outer scope
        term = terms[term_index]
        for object_index in object_indices_left:
            # print("%sbinding %s to %s" % ("   "*term_index, term, object_index))
            bindings[term] = object_index
            object_state_to_assignments(object_state, {term: object_index}, assignments)

            # TODO: check only constraints involving newly assigned variables
            if is_consistent(csps['positive'], assignments):
                if term_index+1 == len(terms):
                    good = True 
                    for csp in csps['negatives']:
                        if is_consistent(csp, assignments):
                            good = False
                            break
                    if not good:
                        continue
                    else:
                        return True
                else:
                    next_object_indices_left = list(object_indices_left)
                    next_object_indices_left.remove(object_index)
                    solved = verify_concept_recursive(
                                term_index+1, next_object_indices_left)
                    if solved:
                        return True
        del bindings[term]
        del assignments[term + '.is_self']
        del assignments[term + '.x']
        del assignments[term + '.y']
        del assignments[term + '.color']
        del assignments[term + '.shape']
        del assignments[term + '.shape_idx']
        del assignments[term + '.pushable']
        del assignments[term + '.is_target']
        return False
    return verify_concept_recursive(0, list(object_indices))



### API ###
class Concept(object):
    """
    Defines a boolean function of trajectories.

    Parameters
    ----------
    order : int
        The number of most recent state-actions that the concept depends on.
        Note that order=0 indicates that the concept depends only on the initial
        state.
    """
    def __init__(self, order=0):
        self.order = order

    def concept_is_present(self, trajectory):
        """
        Parameters
        ----------
        trajectory : [(Any, Any)]
                A chronological list of (state, action) pairs.
                See PixelWorld.object_state for state representation.
     
        Returns
        -------
        is_present : bool
        """
        raise NotImplementedError()


class ClassificationConcept(Concept):
    """
    Classification depends only on the first state of the trajectory.
    """
    def _concept_is_initially_present(self, initial_state):
        """
        Parameters
        ----------
        initial_state : Any
            See PixelWorld.object_state for state representation.

        Returns
        -------
        is_present : bool
        """
        raise NotImplementedError()

    def concept_is_present(self, trajectory):
        """
        Parameters
        ----------
        trajectory : [(Any, Any)]
                A chronological list of (state, action) pairs.
                See PixelWorld.object_state for state representation.
     
        Returns
        -------
        is_present : bool
        """
        return self._concept_is_initially_present(trajectory[0][0])


class BringAboutConcept(Concept):
    """
    Bring-about depends only on the last `self.order` state-actions of the trajectory.
    """
    def __init__(self, order=1):
        super(BringAboutConcept, self).__init__(order=order)

    def _concept_is_finally_present(self, last_state_actions):
        """
        Parameters
        ----------
        last_state_actions : [(Any, Any)]

        Returns
        -------
        is_present : bool
        """
        raise NotImplementedError()

    def concept_is_present(self, trajectory):
        """
        Parameters
        ----------
        trajectory : [(Any, Any)]
                A chronological list of (state, action) pairs.
                See PixelWorld.object_state for state representation.
     
        Returns
        -------
        is_present : bool
        """
        if len(trajectory) < self.order:
            return False
        return self._concept_is_finally_present(trajectory[-self.order:])


class CSPClassificationConcept(ClassificationConcept):
    """
    Classification concept detected with a CSP.
    """
    def __init__(self, concept, settings=None):
        if settings is None:   
            settings = {} # Not using user settings for concept testing 
        else:
            settings = copy.deepcopy(settings)           
        settings['infinite_position_domain'] = True
        settings['check_overlap'] =  False       
        self.csps = concept_to_csps(concept, settings=settings)

    def _concept_is_initially_present(self, initial_state):
        """
        Parameters
        ----------
        initial_state : Any
            See PixelWorld.object_state for state representation.

        Returns
        -------
        is_present : bool
        """
        return concept_holds(None, initial_state, csps=self.csps)

    def __eq__(self, other):
        return isinstance(other, CSPClassificationConcept) and self.csps['logic'] == other.csps['logic']

    def __ne__(self, other):
        return self != other


class CSPBringAboutConcept(BringAboutConcept):
    """
    Bringabout concept detected with a CSP.
    """
    def __init__(self, concept, order=1, settings=None):
        assert order == 1, 'CSPBringAboutConcept order = %i not yet implemented.' % order
        if settings is None:   
            settings = {} # Not using user settings for concept testing 
        else:
            settings = copy.deepcopy(settings) 
        settings['infinite_position_domain'] = True
        settings['check_overlap'] =  False  
        self.csps = concept_to_csps(concept, settings=settings)
        super(CSPBringAboutConcept, self).__init__(order=order)

    def _concept_is_finally_present(self, last_state_actions):
        last_state = last_state_actions[-1][0]
        return concept_holds(None, last_state, csps=self.csps)

    def __eq__(self, other):
        return isinstance(other, CSPBringAboutConcept) and self.csps['logic'] == other.csps['logic']

    def __ne__(self, other):
        return self != other



_CACHE = {}
_CACHE_FILENAME = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '../cache/')

def generate_datasets(concepts, concept_types, concepts_to_balance,
                      generators, dataset_size, seed,
                      concept_macros={}, generator_macros={},
                      settings={}, debug=False, use_cache=False,
                      bringabout_must_start_false=True):
    # ********************************************************
    # WARNING: Add any new function parameters to 'cache_key'!
    # ********************************************************
    """
    concepts : {name : concept}
        concepts to annotate dataset with
    concept_types : {name : str}
        map from concept name to string of concept type, which may be:
            'classification'
            'bringabout'
    concepts_to_balance : [name]
        names of concepts to balance
        dataset will contain equal numbers of each possible value of the concept
        NOTE: you can only balance classification concepts
    generators : [concept]
        list of concepts to generate examples
    dataset_size : int
        dataset size. must be divisible by the number of buckets required for 
        balancing
    seed : int
        seed for random number generator
    concept_macros : {str : concept}
        dictionary of macros for parsing concepts
    generator_macros : {str : concept}
        dictionary of macros for parsing generators
    settings : dict
        additional settings to add/override DEFAULT_SETTINGS
    debug : bool
        whether to print debug output
    use_cache : bool
        whether to use an on-disk cache to store datasets
    bringabout_must_start_false : bool
        whether all the bringabout concepts must be false in the initial state
    """
    # Validate input
    assert set(concepts_to_balance) <= set(concepts.keys())
    assert set(concept_types.keys()) == set(concepts.keys())
    assert all([typ in {'classification', 'bringabout'} for typ in concept_types.values()]) 
    for name in concepts_to_balance:
        assert concept_types[name] == 'classification'

    print("generate_dataset:")

    concepts_to_balance = sorted(concepts_to_balance)

    # Try to read from cache: first in memory then (optionally) on disk
    # NOTE: delete the file or use cache.clear() to clear the on disk cache
    global _CACHE
    cache_key = make_hashable({
        'concepts': concepts,
        'concepts_to_balance': concepts_to_balance,
        'concept_types': concept_types,
        'generators': generators,
        'dataset_size': dataset_size, 
        'seed': seed,
        'concept_macros': concept_macros,
        'generator_macros': generator_macros,
        'settings': settings,
        'bringabout_must_start_false': bringabout_must_start_false})
    print("Checking cache.")
    cache = None
    if cache_key in _CACHE:
        print("Dataset found in memory!")
        return _CACHE[cache_key]
    elif use_cache:
        try:
            cache = diskcache.Cache(_CACHE_FILENAME)
            print("Checking dataset cache (cache_size %d)." % (len(cache),))
            if cache_key in cache:
                out = cache[cache_key]
                cache.close()
                print("Found on disk!")
                return out
            else:
                print("Not found.")
        except Exception as e:
            print("Exception in trying to access cache, skipping: " + str(e))
            use_cache = False

    # FIXME: this really should be above the caching step, but then settings
    # has to only contain hashable things
    settings = dict(settings)
    for key, val in DEFAULT_SETTINGS.items():
        if key not in settings:
            settings[key] = copy.deepcopy(val)
    settings = populate_settings(settings)

    concept_settings = copy.deepcopy(settings)
    generator_settings = copy.deepcopy(settings)
    concept_settings['macros'] = concept_macros
    generator_settings['macros'] = generator_macros

    num_concepts = len(concepts)
    num_bconcepts = len(concepts_to_balance)
    sorted_concept_names = list(concepts_to_balance)
    for name in sorted(concepts.keys()):
        if name not in sorted_concept_names:
            sorted_concept_names.append(name)
    assert len(sorted_concept_names) == len(concepts)

    all_concept_csps = {}
    concept_objects = {}
    orders = {}
    for name, concept in concepts.items():
        csps = concept_to_csps(concept, settings=concept_settings)
        if concept_types[name] == 'classification':
            concept_object = CSPClassificationConcept(concept, settings=concept_settings)
            order = 0
        else:
            assert concept_types[name] == 'bringabout'
            concept_object = CSPBringAboutConcept(concept, settings=concept_settings)
            order = 1
        all_concept_csps[name] = csps
        concept_objects[name] = concept_object
        orders[name] = order

    assert dataset_size % 2**num_bconcepts == 0, 'Dataset size must be divisible by %s to achieve balance' % (2**n,)


    # Generate printable strings for labels and blabels
    label_val_strs = []
    for name in sorted_concept_names:
        if concept_types[name] == 'classification':
            label_val_strs.append([(1, '+'), (0, '-')])
        else:        
            label_val_strs.append([(1, 'b')])

    all_labels = []
    labels_to_str = {}     # b|rest
    for val_strs in product(*label_val_strs):
        s = "".join([s for _, s in val_strs])
        if num_bconcepts < num_concepts:
            labels_str = s[:num_bconcepts] + "|" + s[num_bconcepts:]
        else:
            labels_str = s
        labels = tuple([v for v, _ in val_strs])
        all_labels.append(labels)
        labels_to_str[labels] = labels_str

    all_blabels = []
    blabels_to_str = {}     # b
    if num_bconcepts > 0:
        for val_strs in product(*label_val_strs[:num_bconcepts]):
            blabels_str = "".join([s for _, s in val_strs])
            blabels = tuple([v for v, _ in val_strs])
            all_blabels.append(blabels)
            blabels_to_str[blabels] = blabels_str
  
    print(concepts_to_balance)
    for name in sorted_concept_names:
        concept = concepts[name]
        print('%s: balance %s type %s' % (name, name in concepts_to_balance, concept_types[name]))
        pprint(concept)
        if debug:
            parse_concept(concept, settings=concept_settings, debug=True)
            print()
            csps = concept_to_csps(concept, settings=concept_settings)
            pprint(csps)
            print()
            print("POSITIVE")
            print_csp(csps['positive'])
            print()    
            for idx, csp in enumerate(csps['negatives']):
                print("NEGATIVE", idx)
                print_csp(csp)
                print()


    all_objects = []
    all_objects_labels = []
    labels_to_specs = defaultdict(list)

    num_left = dataset_size
    num_needed_by_blabels = ((dataset_size // 2**num_bconcepts) * np.ones((2,) 
                             * num_bconcepts, dtype=np.int))
    num_left_by_blabels = ((dataset_size // 2**num_bconcepts) * np.ones((2,) 
                           * num_bconcepts, dtype=np.int))
    num_generated = 0   
    num_generated_by_gen = [0] * len(generators)
    num_generated_by_gen_labels = defaultdict(int)
    num_generated_by_blabels = np.zeros((2,) * num_bconcepts, dtype=np.int)
    num_generated_by_labels = np.zeros((2,) * num_concepts, dtype=np.int)
    num_duplicates = 0
    num_duplicates_by_gen = [0] * len(generators)
    num_duplicates_by_gen_labels = defaultdict(int)
    num_duplicates_by_labels = np.zeros((2,) * num_concepts, dtype=np.int)

    gen_iters = [concept_to_objects(x, seed, settings=generator_settings) for x in generators]
    rand = random.Random(seed)

    # Wrap in a try-finally block so that e.g. you can cancel this with ^C if
    # it blocks and still see some statistics
    try:
        while num_left > 0:
            it_idx = rand.randint(0, len(gen_iters)-1)
            it = gen_iters[it_idx]
            objects = it.next()     

            if settings['frame']:
                # Add frame, increase world size, and adjust all positions
                color = settings['frame_color']
                meta = {'is_self': False, 'pushable': False, 'is_target': False} 
                objects.append(['frame', {'color': color, 'meta': meta}])

            # Check for duplicates; this is O(n^2) which may eventually be a problem
            try:
                obj_idx = all_objects.index(objects)
            except ValueError:
                obj_idx = None

            if obj_idx is None:
                object_state = objects_to_object_state(objects)            
                all_objects.append(objects)
                num_generated += 1
                num_generated_by_gen[it_idx] += 1

                spec = {
                    'objects': objects,
                    'width': settings['width'],
                    'height': settings['height'],
                    # these will be added later:
                    'concept': None,
                    'order': None,
                }

                labels = []
                done = False
                for name in sorted_concept_names:
                    satisfies_concept = concept_holds(None, object_state, 
                                                      csps=all_concept_csps[name])
                    if concept_types[name] == 'bringabout':
                        # positive on assumption that concept CAN be achieved
                        labels.append(1)
                        # throw away cases where concept is already satisfied
                        if bringabout_must_start_false and satisfies_concept:
                            done = True
                            break
                    else:
                        labels.append(1 if satisfies_concept else 0)
                if done:
                    continue
                labels = tuple(labels)
                all_objects_labels.append(labels)

                blabels = labels[:num_bconcepts]

                num_generated_by_blabels[blabels] += 1
                num_generated_by_labels[labels] += 1
                num_generated_by_gen_labels[(it_idx, labels)] += 1

                if len(blabels) > 0:
                    if num_left_by_blabels[blabels] > 0:
                        num_left_by_blabels[blabels] -= 1
                    else:
                        continue
                num_left -= 1            
                labels_to_specs[labels].append(spec)         
            else:
                labels = all_objects_labels[obj_idx]
                num_duplicates += 1
                num_duplicates_by_gen[it_idx] += 1
                num_duplicates_by_gen_labels[(it_idx, labels)] += 1
                num_duplicates_by_labels[labels] += 1            


            counts_by_blabels = []
            for blabels in all_blabels:
                got = num_generated_by_blabels[blabels]
                needed = num_needed_by_blabels[blabels]
                s = "%s %i/%i" % (blabels_to_str[blabels], got, needed)
                counts_by_blabels.append(s)

            stdout.write("\rGenerated %i (%i dups) %s" % (num_generated, 
                            num_duplicates, " ".join(counts_by_blabels)))
            stdout.flush()
    except:
        print("\n*** Exception raised! Printing out statistics: ***")
        raise
    finally:
        stdout.write("\n")
        print("Dataset distribution:")
        for labels in all_labels:
            print("   %s: %s specs %s generated %s duplicate"
                    % (labels_to_str[labels], len(labels_to_specs[labels]),
                       num_generated_by_labels[labels], 
                       num_duplicates_by_labels[labels]))
        print("Per generator:")
        for it_idx in range(len(generators)):
            print("   %s: %s generated %s duplicate"
                    % (it_idx, num_generated_by_gen[it_idx], 
                       num_duplicates_by_gen[it_idx]))
            for labels in all_labels:
                print("       %s: %s generated %s duplicate"
                        % (labels_to_str[labels], 
                           num_generated_by_gen_labels[(it_idx, labels)],
                           num_duplicates_by_gen_labels[(it_idx, labels)]))
        print("Per labels:")
        for labels in all_labels:
            print("   %s: %s generated %s duplicate"
                    % (labels_to_str[labels], num_generated_by_labels[labels], 
                       num_duplicates_by_labels[labels]))
            for it_idx in range(len(generators)):
                print("       %s: %s generated %s duplicate"
                        % (it_idx, 
                           num_generated_by_gen_labels[(it_idx, labels)],
                           num_duplicates_by_gen_labels[(it_idx, labels)]))

    all_spec_labels = []
    for labels, specs in labels_to_specs.items():
        all_spec_labels += [(spec, labels) for spec in specs]
    rand.shuffle(all_spec_labels)

    if debug:
        num_by_blabels = np.zeros((2,) * num_bconcepts, dtype=np.int)
        for idx, (spec, labels) in enumerate(all_spec_labels):
            blabels = labels[:num_bconcepts]
            num_by_blabels[blabels] += 1
            print("--- datum %i (%s %i/%i)---" 
                % (idx, labels_to_str[labels], num_by_blabels[blabels], 
                   num_needed_by_blabels[blabels]))
            pprint(spec)
            print()

    out = {}
    for idx, name in enumerate(sorted_concept_names):
        labels_val = [labels[idx] for spec, labels in all_spec_labels]
        specs_val = []
        for spec, _ in all_spec_labels:
            this_spec = copy.deepcopy(spec)
            spec['concept'] = concept_objects[name]
            spec['order'] = orders[name]
            specs_val.append(spec)
        out[name] = {'labels': labels_val, 'specs': specs_val}

    # Optionally store in cache
    _CACHE[cache_key] = out
    if use_cache:
        print("Storing in dataset cache.")        
        try:
            cache[cache_key] = out
            cache.close()
        except Exception as e:
            print("Exception in trying to access disk cache, skipping: " + str(e))

    return out


def generate_dataset(concept, generators, num_samples, seed, debug=False,
                     settings={}, concept_type='classification', use_cache=False,
                     concept_macros={}, generator_macros={}):
    # ********************************************************
    # WARNING: Add any new function parameters to 'cache_key'!
    # ********************************************************
    """
    concept : concept
        concept for classification
    generators : [concept]
        list of concepts to generate examples
    num_samples : int
        number of samples of each of positive and negative concepts
    seed : int
        seed for random number generator
    debug : bool
        whether to print debug output
    settings : dict
        additional settings to add/override DEFAULT_SETTINGS
    concept_type : str
        'classification' or 'bringabout'.
    use_cache : bool
        whether to use an on-disk cache to store datasets
    concept_macros : {str : concept}
        dictionary of macros for parsing concepts
    generator_macros : {str : concept}
        dictionary of macros for parsing generators
    """
    global _CACHE
    assert concept_type == 'classification' or concept_type == 'bringabout'

    # Try to read from cache: first in memory then (optionally) on disk
    # NOTE: delete the file or use cache.clear() to clear the on disk cache
    cache_key = make_hashable({
        'concept': concept,
        'generators': generators,
        'num_samples': num_samples, 
        'seed': seed,
        'settings': settings,
        'concept_type': concept_type,
        'concept_macros': concept_macros,
        'generator_macros': generator_macros})
    print("Checking cache.")
    cache = None
    if cache_key in _CACHE:
        print("Dataset found in memory!")
        return _CACHE[cache_key]
    elif use_cache:
        try:
            cache = diskcache.Cache(_CACHE_FILENAME)
            print("Checking dataset cache (cache_size %d)." % (len(cache),))
            if cache_key in cache:
                out = cache[cache_key]
                cache.close()
                print("Found on disk!")
                return out
            else:
                print("Not found.")
        except Exception as e:
            print("Exception in trying to access cache, skipping: " + str(e))
            use_cache = False

    # FIXME: this really should be above the caching step, but then settings
    # has to only contain hashable things
    settings = dict(settings)
    for key, val in DEFAULT_SETTINGS.items():
        if key not in settings:
            settings[key] = copy.deepcopy(val)
    settings = populate_settings(settings)

    concept_settings = copy.deepcopy(settings)
    generator_settings = copy.deepcopy(settings)
    concept_settings['macros'] = concept_macros
    generator_settings['macros'] = generator_macros

    if concept_type == 'classification':
        concept_object = CSPClassificationConcept(concept, settings=concept_settings)
        order = 0
    else:
        assert concept_type == 'bringabout'
        concept_object = CSPBringAboutConcept(concept, settings=concept_settings)
        order = 1

    print("generate_dataset: concept", concept)

    if debug:
        parse_concept(concept, settings=concept_settings, debug=True)
        print()
        csps = concept_to_csps(concept, settings=concept_settings)
        pprint(csps)
        print()
        print("POSITIVE")
        print_csp(csps['positive'])
        print()    
        for idx, csp in enumerate(csps['negatives']):
            print("NEGATIVE", idx)
            print_csp(csp)
            print()

    rand = random.Random(seed)

    if concept_type == 'bringabout':
        num_positive_samples = num_samples
        num_negative_samples = 0
    else:
        num_positive_samples = num_samples
        num_negative_samples = num_samples

    iters = [concept_to_objects(x, seed, settings=generator_settings) for x in generators]
    concept_csps = concept_to_csps(concept, settings=concept_settings)

    all_objects = []
    positive_specs = []
    negative_specs = []
    num_generated = 0
    num_generated_iter = [0] * len(iters)
    num_generated_label = [0, 0]
    num_generated_iter_label = defaultdict(int)
    num_duplicates = 0
    num_duplicates_iter = [0] * len(iters)
    while True:
        it_idx = rand.randint(0, len(iters)-1)
        it = iters[it_idx]
        objects = it.next()     

        if settings['frame']:
            # Add frame, increase world size, and adjust all positions
            color = settings['frame_color']
            meta = {'is_self':False,'pushable':False,'is_target':False} 
            objects.append(['frame',{'color':color,'meta':meta}])

        # Check for duplicates; this is O(n^2) which may eventually be a problem
        if objects not in all_objects:
            all_objects.append(objects)
            num_generated += 1
            num_generated_iter[it_idx] += 1

            spec = {
                'objects': objects,
                'order': order,
                'width': settings['width'],
                'height': settings['height'],
                'concept': concept_object
            }

            satisfies_concept = concept_holds(None, objects_to_object_state(objects), 
                                    csps=concept_csps)
            if concept_type == 'bringabout':
                # positive on assumption that concept CAN be achieved
                label = 1
                # throw away cases where concept is already satisfied
                if satisfies_concept:
                    continue
            else:
                label = 1 if satisfies_concept else 0
            num_generated_label[label] += 1
            num_generated_iter_label[(it_idx, label)] += 1

            if label == 1:
                if len(positive_specs) < num_positive_samples:
                    positive_specs.append(spec)
            else:
                if len(negative_specs) < num_negative_samples:
                    negative_specs.append(spec)
        
            done = len(positive_specs) == num_positive_samples \
                   and len(negative_specs) == num_negative_samples            
        else:
            num_duplicates += 1
            num_duplicates_iter[it_idx] += 1
        
        stdout.write("\rGenerated %i samples (%i duplicates), %i/%i positive %i/%i negative."
                        % (num_generated, num_duplicates, 
                           num_generated_label[1], num_positive_samples,
                           num_generated_label[0], num_negative_samples))
        stdout.flush()

        if done:
            break

    stdout.write("\n")
    print("Per generator:")
    for it_idx in range(len(generators)):
        print("   %s: %s generated %s duplicate %s positive %s negative"
                % (it_idx, num_generated_iter[it_idx], 
                    num_duplicates_iter[it_idx],
                    num_generated_iter_label[it_idx,1],
                    num_generated_iter_label[it_idx,0]))    

    spec_labels = [(spec, 0) for spec in negative_specs]
    spec_labels += [(spec, 1) for spec in positive_specs]
    rand.shuffle(spec_labels)

    if debug:
        num = {0: 0, 1: 0}
        for idx, (spec, label) in enumerate(spec_labels):
            num[label] += 1
            print("--- datum %i (%s %i/%i)---" 
                % (idx, "pos" if label==1 else "neg", num[label], num_samples))
            pprint(spec)
            print()

    out = {
        'specs': [spec for spec, label in spec_labels],
        'labels': [label for spec, label in spec_labels]
    }

    # Optionally store in cache
    _CACHE[cache_key] = out
    if use_cache:
        print("Storing in dataset cache.")        
        try:
            cache[cache_key] = out
            cache.close()
        except Exception as e:
            print("Exception in trying to access disk cache, skipping: " + str(e))

    return out


def generate_dataset_by_name(concept_dict, generator_dict, concept_name, 
    generator_names, num_samples, seed, use_macros=False, merge_macros=False, 
    settings={}, additional_settings={}, **kwargs):
    """A version of generate_dataset convenient to use with large metaexperiments.

    use_macros : bool
        Whether to use macro expansion when parsing concepts and generators.
        If so, concept_dict will be used for concept macros, generator_dict
        for generator macros.

    merge_macros :  bool
        Whether to merge generators and concepts in the same macro set.
        Raises an exception if names overlap.

    additional_settings : dict
        Overrides settings; useful for experiment files.
    """
    
    concept = concept_dict[concept_name]
    generators = [generator_dict[name] for name in generator_names]

    print("generate_dataset_by_name:")
    print("  concept_name:", concept_name)
    print("        -> concept:", concept)
    print("  generator_names:", generator_names)
    for generator in generators:
        print("        -> generator:", generator)

    if use_macros:
        if merge_macros:
            macros = dict(concept_dict)
            for k, v in generator_dict.items():
                if k in macros:
                    raise Exception("%s is defined in concepts_dict and generator_dict" % (k,))
                macros[k] = v
            concept_macros = macros
            generator_macros = macros
        else:
            concept_macros = concept_dict
            generator_macros = generator_dict
    else:
        concept_macros = {}
        generator_macros = {}

    settings = copy.deepcopy(settings)
    if additional_settings is not None:
        for k, v in additional_settings.items():
            settings[k] = v

    return generate_dataset(concept, generators, num_samples, seed, 
            concept_macros=concept_macros, generator_macros=generator_macros, 
            settings=settings, **kwargs)


# Tests
# -----

def test_infinite_boundaries():
    """
    Unit test for infinite position values during verification.
    """
    concept = "red(self)"
    object_state = [{'name' : 'self', 'position' : (7, 1), 'color' : 2,
        'meta': {'is_self': True, 'shape': 'pixel', 'shape_idx': 0, 
                 'pushable': False, 'is_target':False}}]
    assignments = object_state_to_assignments(object_state, {'self' : 0})

    settings = dict(DEFAULT_SETTINGS)
    settings['width'] = 8
    settings['height'] = 8
    settings['infinite_position_domain'] = False
    settings = populate_settings(settings)

    # Should be bounded by width=8, height=8
    bounded_csps = concept_to_csps(concept, settings=settings)
    bounded_positive_csp = bounded_csps['positive']

    # Should be consistent to start
    assert is_consistent(bounded_positive_csp, assignments, debug=True)

    # Now self moves offscreen
    object_state[0]['position'] = (8, 1)
    assignments = object_state_to_assignments(object_state, {'self' : 0})

    try:
        is_consistent(bounded_positive_csp, assignments)
        raise Exception('Bounded CSP should fail consistency check')
    except Exception:
        pass

    # This is the main test

    # Should have infinite width and height
    settings['infinite_position_domain'] = True
    infinite_csps = concept_to_csps(concept, settings=settings)
    infinite_positive_csp = infinite_csps['positive']

    assert is_consistent(infinite_positive_csp, assignments, debug=True)

    print("Test infinite boundaries passed.")

  

def main():
    concepts = []
    concepts.append("?x ?y ?z floor(z) & adjacent(self,z) & adjacent(x,z) & adjacent(y,z) & adjacent(self,x) & right_of(y,x)")
    concepts.append("?x red(x) & adjacent(x, self)")
    concepts.append("?x red(x) & adjacent(x, self) & ~above(x, self)")

    settings = populate_settings(DEFAULT_SETTINGS)

    concept = raw_input("Enter concept (or 0-2 for predefined): ")
    if concept in ["0", "1", "2"]:
        concept = concepts[int(concept)]
    print("concept:", concept)
    print()
    quantified_terms, positive_clauses, negative_clause_lists, meta = parse_concept(concept, settings=settings, debug=True)
    print()    
    print('quantified_terms:', quantified_terms)
    print('positive_clauses:', positive_clauses)
    print('negative_clause_lists:', negative_clause_lists)
    print('meta:', meta)
    print()
    csps = concept_to_csps(concept, settings)
    pprint(csps)
    print()
    print("POSITIVE")
    print_csp(csps['positive'])
    print()    
    for idx, csp in enumerate(csps['negatives']):
        print("NEGATIVE", idx)
        print_csp(csp)
        print()
    
    settings.update({'frame': True, 'padding': 2, 'allow_targets': True, 'allow_pushable': True})

    for idx, objects in enumerate(concept_to_objects(concept, 1, settings, min_n=10, max_n=10)):
        print("\n*** Solution %s ***" % (idx,))
        pprint(objects)
        if not concept_holds(concept, objects_to_object_state(objects), settings=settings):
            raise Exception("Concept doesn't hold!")


def check_valid_datasets(datasets, dataset_size):
    for name in datasets.keys():
        assert set(datasets[name].keys()) == {'labels', 'specs'}
        assert len(datasets[name]['labels']) == dataset_size
        assert len(datasets[name]['specs']) == dataset_size

    # Check the specs for the two concepts are equal and aligned
    for name1 in datasets.keys():
        for name2 in datasets.keys():
            if name1 < name2:
                for idx in range(dataset_size):
                    spec1 = dict(datasets[name1]['specs'][idx])
                    spec2 = dict(datasets[name2]['specs'][idx])
                    for spec in [spec1, spec2]:
                        assert set(spec.keys()) == {'objects', 'height', 'width', 'concept', 'order'}
                        del spec['concept']
                    assert spec1 == spec2

def test_generate_datasets():
    debug = False
    seed = 0
    dataset_size = 128
    num_samples = dataset_size//2


    concepts = {'': "?x red(x) & adjacent(x, self)"}
    concept_types = {'': 'classification'}
    concepts_to_balance = ['']
    generators = ["?x red(x)", "?x green(x)"]

    datasets = generate_datasets(concepts, concept_types, concepts_to_balance,
                generators, dataset_size, seed, debug=debug)
    dataset2 = generate_dataset(concepts[''], generators, num_samples, seed,
                debug=debug, concept_type=concept_types[''])
    check_valid_datasets(datasets, dataset_size)
    assert datasets[''] == dataset2
    assert set(dataset2.keys()) == {'labels', 'specs'}


    concept_types = {'': 'bringabout'}
    concepts_to_balance = []
    num_samples = dataset_size

    datasets = generate_datasets(concepts, concept_types, concepts_to_balance,
                generators, dataset_size, seed, debug=debug)
    dataset2 = generate_dataset(concepts[''], generators, num_samples, seed,
                debug=debug, concept_type=concept_types[''])
    check_valid_datasets(datasets, dataset_size)
    assert datasets[''] == dataset2
    assert set(dataset2.keys()) == {'labels', 'specs'}


    concepts = {
        'adjacent': "?x adjacent(x, self)",
        'red': "?x red(x)"
    }
    concept_types = {'adjacent': 'classification', 'red': 'classification'}
    concepts_to_balance = ['adjacent']

    datasets = generate_datasets(concepts, concept_types, concepts_to_balance,
                generators, dataset_size, seed, debug=debug)    
    check_valid_datasets(datasets, dataset_size)
    assert set(datasets.keys()) == {'adjacent', 'red'}
    for name in ['adjacent']:
        assert sum(datasets[name]['labels']) == dataset_size//2, name + " not balanced!"


    concepts_to_balance = ['adjacent', 'red']

    datasets = generate_datasets(concepts, concept_types, concepts_to_balance,
                generators, dataset_size, seed, debug=debug)
    check_valid_datasets(datasets, dataset_size)
    assert set(datasets.keys()) == {'adjacent', 'red'}
    for name in ['adjacent', 'red']:
        assert sum(datasets[name]['labels']) == dataset_size//2, name + " not balanced!"




if __name__ == "__main__":
    #main()
    #test_infinite_boundaries()
    test_generate_datasets()


# TODO: 
#    * Negative individual relations

