"""Extensions to python-constraint needed to create Constraint Satisfaction Problems
(CSPs) corresponding to logically expressed concepts."""

from __future__ import print_function
from __future__ import absolute_import

from collections import defaultdict
import copy
import numpy as np
from pprint import pprint
import random
import re

from scipy.spatial.distance import cdist

from constraint import (Problem, InSetConstraint, FunctionConstraint, 
    BacktrackingSolver, Domain, Constraint, Unassigned)

# Randomized constraint satisfaction solver
# -----------------------------------------  

# TODO: smarter proposing:
#   eliminate knowably inconsistent assignments:
#       implement constraints with special-case code:
#           forwardCheck -- called during solution by Constraint.__call__
#           preProcess -- called by the problem before solution
#       special-case domains
#   special-case variable_proposer and value_sorter

def random_variable_ordering(random, domains, constraints, vconstraints, assignments):
    lst = [(variable,) for variable in domains]
    random.shuffle(lst)
    return lst

def degree_mrv_variable_ordering(random, domains, constraints, vconstraints, assignments):
    """Use the max degree and minimum remaining values heuristics."""
    lst = [(-len(vconstraints[variable]),
            len(domains[variable]), variable) for variable in domains]
    lst.sort()
    return lst

def alphabetical_variable_ordering(random, domains, constraints, vconstraints, assignments):
    """Sort variables alphabetically, presumably works because it groups variables
    corresponding to the same object."""
    lst = [(1, variable) for variable in domains]
    lst.sort()
    return lst

def value_uniform_sorter(random, domains, constraints, vconstraints, assignments, variable, values):
    """Randomize the order that variable values are explored. This won't affect 
    exhaustiveness, and each time a variable is selected its domain will be 
    shuffled differently. A RandomBacktrackingSolver using this will uniformly 
    sample from the set of consistent assignments, because the generation of 
    each assignment samples every variable exactly once. Successive iterations
    from getSolutionIter will not be dependent, however.
    """
    random.shuffle(values)
    return values

def value_non_sorter(random, domains, constraints, vconstraints, assignments, variable, values):
    return values


class InfiniteIntegralDomain(Domain):
    """Represents an infinite integral domain for a variable in the CSP. 
    Obviously should not be used for generation; only should be used for verification.

    Parameters
    ----------
    min_int : int or None
        Defaults to np.NINF. Inclusive.
    max_int : int
        Defaults to np.inf. Exclusive.
    """
    def __init__(self, min_int=None, max_int=None):
        self.min_int = min_int or np.NINF
        self.max_int = max_int or np.inf

    def __contains__(self, i):
        """
        """
        if not (isinstance(i, int) or (i).is_integer()):
            return False
            
        return self.min_int <= i < self.max_int

    def __len__(self):
        # Hack to avoid error in python-constraint addVariable method
        return 1

    def resetState(self):
        """
        """
        raise Exception('This method should not be called; the domain is infinite.')

    def pushState(self):
        """
        """
        raise Exception('This method should not be called; the domain is infinite.')

    def popState(self):
        """
        """
        raise Exception('This method should not be called; the domain is infinite.')

    def hideValue(self, value):
        """
        """
        raise Exception('This method should not be called; the domain is infinite.')


# Modified from python-constraint module, available:
#   https://github.com/python-constraint/python-constraint
class RandomBacktrackingSolver(BacktrackingSolver):
    """
    Problem solver with backtracking capabilities, allowing randomized variable
    proposers and value sorters. Aside from factoring out proposers and sorters,
    this is identical to BacktrackingSolver.
    """

    def __init__(self, forwardcheck=True, seed=0, variable_proposer=None, 
        value_sorter=None, random_restart=True, 
        random_restart_max_iters=1000, random_restart_max_tries=10):
        """
        forwardcheck: If false forward checking will not be requested to 
        constraints while looking for solutions (default is true).
        
        seed: seed for generating the random object used (potentially) by the
        variable proposer and the value sorter. 

        variable_proposer: yields variables to try assigning, given the current
        state of the solving process.

        value_sorter: sorts (potentially with mutation) the set of remaining
        values of a variable in the order they could be tried.

        random_restart: will randomly restart if more than random_restart_iters
        assignments are tried. random_restart_iters is initialized to
        random_restart_max_iters and increased by 2x every random_restart_max_tries
        restarts.
        """
        self._forwardcheck = forwardcheck
        self._random = random.Random(seed)
        self._random_restart = random_restart
        self._random_restart_max_iters = random_restart_max_iters
        self._random_restart_max_tries = random_restart_max_tries

        if variable_proposer is None:
            self._variable_proposer = alphabetical_variable_ordering
        else:
            self._variable_proposer = variable_proposer

        if value_sorter is None:
            self._value_sorter = value_uniform_sorter
        else:
            self._value_sorter = value_sorter

    def getSolutionIter(self, domains, constraints, vconstraints):
        if not self._random_restart:
            for assignments in self.getBoundedSolutionIter(domains, constraints,
                                vconstraints, bound=np.inf):
                yield assignments
            return

        random_restart_max_iters = self._random_restart_max_iters
        random_restart_tries = 0
        while True:
            for assignments in self.getBoundedSolutionIter(domains, constraints, 
                                vconstraints, bound=random_restart_max_iters):
                yield assignments
            random_restart_tries += 1
            if random_restart_tries >= self._random_restart_max_tries:
                random_restart_tries = 0
                random_restart_max_iters *= 2

    def getBoundedSolutionIter(self, domains, constraints, vconstraints, bound=np.inf):
        forwardcheck = self._forwardcheck
        assignments = {}
        queue = []
        idx = 0

        while True:
            # Select an unassigned variable
            for item in self._variable_proposer(self._random, domains, constraints, vconstraints, assignments):
                if item[-1] not in assignments:
                    # Found unassigned variable
                    variable = item[-1]
                    values = list(domains[variable])

                    values = self._value_sorter(self._random, domains, constraints, vconstraints, assignments, variable, values)

                    if forwardcheck:
                        pushdomains = [domains[x] for x in domains
                                       if x not in assignments and x != variable]
                    else:
                        pushdomains = None
                    break
            else:
                # No unassigned variables. We've got a solution. Go back
                # to last variable, if there's one.
                yield assignments.copy()
                if not queue:
                    return
                variable, values, pushdomains = queue.pop()
                if pushdomains:
                    for domain in pushdomains:
                        domain.popState()

            while True:
                # We have a variable. Do we have any values left?
                if not values:
                    # No. Go back to last variable, if there's one.
                    del assignments[variable]
                    # variable_assignment_order.pop()
                    while queue:
                        variable, values, pushdomains = queue.pop()
                        if pushdomains:
                            for domain in pushdomains:
                                domain.popState()
                        if values:
                            break
                        del assignments[variable]
                        # variable_assignment_order.pop()
                    else:
                        return

                # Got a value. Check it.
                assignments[variable] = values.pop()

                # Bound
                idx += 1
                if idx > bound:
                    return

                if pushdomains:
                    for domain in pushdomains:
                        domain.pushState()

                for constraint, variables in vconstraints[variable]:
                    if not constraint(variables, domains, assignments,
                                      pushdomains):
                        # Value is not good.
                        break
                else:
                    break

                if pushdomains:
                    for domain in pushdomains:
                        domain.popState()

            # Push state before looking for next variable.
            queue.append((variable, values, pushdomains))

        raise RuntimeError("Can't happen")


def generate_random_solutions(problem, seed, min_n=None, max_n=None, use_iter=False):
    """

    Note: if use_iter is true the sequence of solutions is correlated, but all
    solutions will be eventually yielded without repeats."""
    solver = problem.getSolver() # Save original solver
    if seed is None:
        problem.setSolver(RandomBacktrackingSolver(
            variable_proposer=alphabetical_variable_ordering,
            #variable_proposer=degree_mrv_variable_ordering,
            value_sorter=value_non_sorter))
    else:
        problem.setSolver(RandomBacktrackingSolver(seed=seed))

    n = 0
    if use_iter:
        for assignments in problem.getSolutionIter():
            yield assignments
            n += 1
            if max_n is not None and n >= max_n: 
                break
    else:
        while True:
            assignments = problem.getSolution()
            if assignments is None:
                raise Exception("No solutions to this CSP!")
            yield assignments
            n += 1
            if max_n is not None and n >= max_n: 
                break
    
    problem.setSolver(solver) # Restore original solver
    if min_n is not None and n < min_n:
        raise Exception("Requested at least %s solutions, but can only generate %s." % (min_n, n))


def is_consistent(problem, assignments, debug=False):
    for variable, value in assignments.items():
        domain = problem._variables[variable]
        if debug:
            print("Variable: %s (value %s) is in %s?" % (variable, value, domain))
        if not value in domain:
            raise Exception("Invalid assignment to variable %s of %s" % (variable, value))
    
    domains = problem._variables
    for constraint, variables in problem._constraints:
        if debug:
            values = [assignments[v] for v in variables] 
            print("Constraint: %s (variables %s values %s) satisfied?" % (constraint, variables, values))

        # Needed because InSetConstraint assumes it is only called while solving
        if isinstance(constraint, InSetConstraint):
            for variable in variables:
                if variable in assignments and assignments[variable] not in constraint._set:
                    if debug:
                        print("   No.")
                    return False
            if debug:
                print("   Yes.")
        elif not constraint(variables, domains, assignments):
            if debug:
                print("   No.")
            return False
        elif debug:
            print("   Yes.")
    if debug:
        print("   All satisfied.")
    return True


def get_unassigned_variables(problem, assignment):
    return [variable for variable in problem._variables if variable not in assignments]


def is_solution(problem, assignments, debug=False):
    # An assignment is a solution if all variables are assigned and it is
    # consistent. We check these in turn:
    if all([variable in assignments for variable in problem._variables]):
        return is_consistent(problem, assignments, debug)
    else:
        if debug:
            unassigned = get_unassigned_variables(problem, assignment)
            print("Unassigned variables:", unassigned)
        return False


def print_csp(csp):
    for variable in sorted(csp._variables):
        print("variable %s domain %s" % (variable, csp._variables[variable]))

    for constraint, variables in csp._constraints:
        if isinstance(constraint, FunctionConstraint):
            constraint = 'FunctionConstraint(%s)' % (constraint._func)
        elif isinstance(constraint, InSetConstraint):
            constraint = 'InSetConstraint(%s)' % (constraint._set)

        print("constraint %s %s" % (variables, constraint))




# Special purpose constraints
# ---------------------------

def spatial_relation_holds(x1, y1, template1, height1, width1, x2, y2, template2, height2, width2):
    """Computes a spatial relation defaulting to 'shape1 right_of shape2'
    between the two specified shapes.

    Implements a strict notion of 'right_of' that requires the two shapes
    to have non-zero overlap in the vertical direction.

    NOTE: this is different to an earlier version that was much more permissive
          and only tested on the level of half-spaces.
    """
    
    # determine common y range
    # NOTE: assuming (width, height) = (1, 1) for a single pixel here!
    common_min_y = int(np.maximum(y1, y2))
    common_max_y = int(np.minimum(y1 + height1 - 1, y2 + height2 - 1))

    # non-empty common y range
    if common_min_y <=  common_max_y:

        # quick check based on bounding box
        if x2 + width2 - 1 < x1:
            return True
        elif x1 + width1 -1 < x2:
            return False

        # detailed check
        else:
            ys1, xs1 = np.where(template1 > 0)
            ys2, xs2 = np.where(template2 > 0)
                 
            ys1 = ys1 + y1
            xs1 = xs1 + x1
            ys2 = ys2 + y2
            xs2 = xs2 + x2

            # check for all y in the common range whether the ordering of
            # x's adheres to the relation
            d1 = defaultdict(list)
            d2 = defaultdict(list)
            for y1,x1 in zip(ys1,xs1):
                d1[y1].append(x1)
            for y2,x2 in zip(ys2,xs2):
                d2[y2].append(x2)

            relation_holds = False
            opposite_holds = False
            for y in range(common_min_y, common_max_y + 1):
                for x1 in d1[y]:
                    for x2 in d2[y]:
                        if x1 > x2:
                            relation_holds = True
                        else:
                            opposite_holds = True

            return relation_holds and not opposite_holds

    return False


class PairwiseSpatialConstraint(Constraint):
    """
    Require two objects to be in a specific spatial arrangement (as specified by mode).
    """
    def __init__(self, settings, mode='adjacent'):
        self.mode = mode
        self.terms = None
        self.templates = settings['shape_templates']
        self.annotations = settings['shape_annotations']
        
    def __call__(self, variables, domains, assignments, forwardcheck=False,
                 _unassigned=Unassigned):
        self.terms = self.terms = [k[:-2] for k in variables if k[-2:] == '.x']

        assert len(variables) == len(self.terms) * 4
        assert len(self.terms) == 2


        term1 = self.terms[0]
        x1 = assignments.get(term1 + '.x', _unassigned)
        if x1 == _unassigned:
            return True
        y1 = assignments.get(term1 + '.y', _unassigned)
        if y1 == _unassigned:
            return True
        shape1 = assignments.get(term1 + '.shape', _unassigned)
        if shape1 == _unassigned:
            return True
        shape_idx1 = assignments.get(term1 + '.shape_idx', _unassigned)
        if shape_idx1 == _unassigned:
            return True
        shape_idx1 = shape_idx1 % len(self.templates[shape1])

        term2 = self.terms[1]
        x2 = assignments.get(term2 + '.x', _unassigned)
        if x2 == _unassigned:
            return True
        y2 = assignments.get(term2 + '.y', _unassigned)
        if y2 == _unassigned:
            return True
        shape2 = assignments.get(term2 + '.shape', _unassigned)
        if shape2 == _unassigned:
            return True
        shape_idx2 = assignments.get(term2 + '.shape_idx', _unassigned)
        if shape_idx2 == _unassigned:
            return True
        shape_idx2 = shape_idx2 % len(self.templates[shape2])


        template1 = self.templates[shape1][shape_idx1]
        height1, width1 = template1.shape

        template2 = self.templates[shape2][shape_idx2]
        height2, width2 = template2.shape

        if self.mode == 'adjacent':
            ys1, xs1 = np.where(template1 > 0)
            ys2, xs2 = np.where(template2 > 0)
            pts1 = np.array([xs1 + x1, ys1 + y1]).T
            pts2 = np.array([xs2 + x2, ys2 + y2]).T
            dists = cdist(pts1, pts2)
            return np.min(dists.flatten()) <= 1.0

        if self.mode == 'touches':
            ys1, xs1 = np.where(template1 > 0)
            ys2, xs2 = np.where(template2 > 0)
            pts1 = np.array([xs1 + x1, ys1 + y1]).T
            pts2 = np.array([xs2 + x2, ys2 + y2]).T
            dists = cdist(pts1, pts2, 'chebyshev')
            return np.min(dists.flatten()) <= 1.0

        if self.mode == 'near':
            ys1, xs1 = np.where(template1 > 0)
            ys2, xs2 = np.where(template2 > 0)
            pts1 = np.array([xs1 + x1, ys1 + y1]).T
            pts2 = np.array([xs2 + x2, ys2 + y2]).T
            dists = cdist(pts1, pts2)
            return np.min(dists.flatten()) <= 2.0

        if self.mode == 'far':
            ys1, xs1 = np.where(template1 > 0)
            ys2, xs2 = np.where(template2 > 0)
            pts1 = np.array([xs1 + x1, ys1 + y1]).T
            pts2 = np.array([xs2 + x2, ys2 + y2]).T
            dists = cdist(pts1, pts2)
            return np.min(dists.flatten()) > 4.0       


        if self.mode == 'inside':
            if shape2 not in self.annotations:
                # TODO: return False once 'inside' is annotated for
                # all relevant shapes
                ys2, xs2 = np.where(template2 <= 0)            
            else:
                annotation = self.annotations[shape2][shape_idx2]
                interior_points = annotation.get_points('interior')
                inside_supported_points = annotation.get_points('inside_supported')
                ys2, xs2 = np.concatenate([interior_points, inside_supported_points]).T

            ys1, xs1 = np.where(template1 > 0)

            pts1 = np.array([xs1 + x1, ys1 + y1]).T
            pts2 = np.array([xs2 + x2, ys2 + y2]).T
     
            # at least one pixel has distance zero to a pixel 'inside'
            dists = cdist(pts1, pts2)
            return np.min(dists.flatten()) == 0.0

        elif self.mode == 'inside_supported':
            if shape2 not in self.annotations:
                # TODO: return False once 'inside_supported' is annotated for
                # all relevant shapes
                ys2, xs2 = np.where(template2 <= 0)            
            else:
                annotation = self.annotations[shape2][shape_idx2]
                ys2, xs2 = annotation.get_points('inside_supported').T

            ys1, xs1 = np.where(template1 > 0)

            pts1 = np.array([xs1 + x1, ys1 + y1]).T
            pts2 = np.array([xs2 + x2, ys2 + y2]).T
     
            dists = cdist(pts1, pts2)
            return np.min(dists.flatten()) == 0.0

        # coarse spatial relations
        elif self.mode == 'left_of':
            max_x1 = x1 + width1
            return max_x1 <= x2

        elif self.mode == 'right_of':
            max_x2 = x2 + width2
            return max_x2 <= x1

        elif self.mode == 'above':
            max_y1 = y1 + height1
            return max_y1 <= y2 

        elif self.mode == 'below':
            max_y2 = y2 + height2
            return max_y2 <= y1

        # fine spatial relations
        elif self.mode == 'dir_left_of':
            return spatial_relation_holds(x2, y2, template2, height2, width2, x1, y1, template1, height1, width1)

        elif self.mode == 'dir_right_of':
            return spatial_relation_holds(x1, y1, template1, height1, width1, x2, y2, template2, height2, width2)

        elif self.mode == 'dir_above':
            return  spatial_relation_holds(y2, x2, template2, width2, height2, y1, x1, template1, width1, height1)

        elif self.mode == 'dir_below':
            return spatial_relation_holds(y1, x1, template1, width1, height1, y2, x2, template2, width2, height2)

        elif self.mode == 'h_aligned':
            max_y1 = y1 + height1
            max_y2 = y2 + height2            
            return (y1 < max_y2) and (y2 < max_y1) 

        elif self.mode == 'v_aligned':
            max_x1 = x1 + width1
            max_x2 = x2 + width2
            return (x1 < max_x2) and (x2 < max_x1)         

        elif self.mode == 'bottom_aligned':    
            return y1 == y2 

        elif self.mode == 'top_aligned':
            return y1 + height1 == y2 + height2

        elif self.mode == 'left_aligned':
            return x1 == x2

        elif self.mode == 'right_aligned':
            return x1 + width1 == x2 + width2

        elif self.mode == 'taller':
            return height1 > height2

        elif self.mode == 'wider':
            return width1 > width2

        elif self.mode == 'same_height':
            return height2 == height1
  
        elif self.mode == 'same_width':
            return width2 == width1

        else:
            raise Exception("unknown mode " + self.mode)
        

class NoOverlapConstraint(Constraint):
    """
    Require objects not to overlap. Also requires that objects not extend beyond
    the outside border (possibly padded by frame)
    """
    def __init__(self, height, width, templates, frame=0, padding=0, lazy=False):
        self.height = height
        self.width = width
        self.frame = frame # uniform frame size
        self.padding = padding
        self.templates = templates
        self.terms = None

        assert not lazy, "Implement me"
        self.lazy = lazy

    def __call__(self, variables, domains, assignments, forwardcheck=False,
                 _unassigned=Unassigned):
        if self.terms is None:
            self.terms = self.terms = [k[:-2] for k in variables if k[-2:] == '.x']
            assert len(variables) == len(self.terms) * 4
        
        occupancy = np.zeros((self.height, self.width), dtype=np.int)

        # Find object whose pixel occupancy is determined
        # i.e., values are assigned to x.x, x.y, x.shape, and x.shape_idx
        for term in self.terms:            
            x = assignments.get(term + '.x', _unassigned)
            if x == _unassigned: continue
            y = assignments.get(term + '.y', _unassigned)
            if y == _unassigned: continue
            shape = assignments.get(term + '.shape', _unassigned)
            if shape == _unassigned: continue
            shape_idx = assignments.get(term + '.shape_idx', _unassigned)
            if shape_idx == _unassigned: continue                       
            shape_idx = shape_idx % len(self.templates[shape])

            template = self.templates[shape][shape_idx]
            height, width = template.shape
            
            # Constraint violated if shapes bbox falls outside padding (excludes walls and floors)
            if self.padding > 0 and not (shape in ["wall","floor_shape"]):
                if x < self.padding or y < self.padding:
                    return False
                if x+width > self.width-self.padding or y+height > self.height-self.padding:
                    return False    

            # Constraint violated if the shape's bbox falls on or outside the frame
            #   TODO: This could be extracted to a constraint only on this term's variables.
            if x < self.frame or y < self.frame:
                return False
            if x + width > self.width-self.frame or y + height > self.height-self.frame:
                return False
                   
            # Check not overlapping
            #    TODO: faster
            #    TODO: correct to allow objects to go out of bounds if there there is no frame
            if np.sum(occupancy[y:y+height,x:x+width] * template) > 0:
                return False
            occupancy[y:y+height,x:x+width] += template

        return True



PI = np.pi
class LogPolarUnarySpatialConstraint(Constraint):
    """
    Require one object to be in a specific spatial arrangement (as specified by mode).
    """
    def __init__(self, settings, mode='almost_center'):
        self.mode = mode
        self.terms = None
        self.templates = settings['shape_templates']
        self.annotations = settings['shape_annotations']
        # Assume square 
        assert settings['height'] == settings['width']
        self.pw_siz = settings['height']

        self.exclusive_rbins = settings.get('exclusive_rbins', True)

        # Fixed values for now.
        self.n_rbins = 4
        self.n_obins = 8
        self.min_rbin = 4

        self.distance_names = [
            'almost_center',
            'near_center',
            'middle_from_center',
            'far_from_center'
        ]

        self.orientation_names = [
            'right_from_center',
            'lower_right_from_center',
            'down_from_center',
            'lower_left_from_center',
            'left_from_center',
            'upper_left_from_center',
            'up_from_center',
            'upper_right_from_center',
        ]

        self._precompute_parameters()

    def _precompute_parameters(self):
        """
        Pre-computes parameters used in constraints evaluation.
        """

        # coordinate of points on unit circle, evenly spaced angularly.
        n_obins = self.n_obins
        thetas = 1./n_obins * np.arange(n_obins) * 2*PI
        self.unit_xy = np.array( [np.cos(thetas), np.sin(thetas)] )

        # boundaries for radial bins: 
        # starting from min_rbin, bin size exponentially grows w.r.t. the distance to center.
        self.rbins = self.min_rbin * 2**np.arange(np.log2(self.pw_siz)-1) * .5

    def __call__(self, variables, domains, assignments, forwardcheck=False,
                 _unassigned=Unassigned):

        self.terms = self.terms = [k[:-2] for k in variables if k[-2:] == '.x']

        assert len(variables) == len(self.terms) * 4
        assert len(self.terms) == 1

        term = self.terms[0]
        x = assignments.get(term + '.x', _unassigned)
        if x == _unassigned:
            return True
        y = assignments.get(term + '.y', _unassigned)
        if y == _unassigned:
            return True
        shape = assignments.get(term + '.shape', _unassigned)
        if shape == _unassigned:
            return True
        shape_idx = assignments.get(term + '.shape_idx', _unassigned)
        if shape_idx == _unassigned:
            return True
        shape_idx = shape_idx % len(self.templates[shape])

        template = self.templates[shape][shape_idx]
        height, width = template.shape

        center = np.array([width, height]) *.5 + np.array([x,y])
        v = center - 0.5 * np.array( [self.pw_siz, self.pw_siz] )


        if self.mode in self.distance_names:
            rbin = self.distance_names.index(self.mode)
            if self.exclusive_rbins:
                return rbin == self._compute_rbin(v)
            else:
                return rbin >= self._compute_rbin(v)
        else:
            assert self.mode in self.orientation_names

            obin = self.orientation_names.index(self.mode)
            return obin == self._compute_obin(v)


    def _compute_rbin(self, pnt):
        return np.digitize(np.abs(pnt), self.rbins).max()

    def _compute_obin(self, pnt):
        return np.dot(pnt, self.unit_xy).argmax()




