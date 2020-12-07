#!/usr/bin/python3

"""A customized implementation of the pymanopt SteepestDescent solver.
Specifically, we have added functionality for keeping track of:

i) Error/cost terms for the current estimate
ii) The current iteration number
iii) The current estimate of the quantity to be optimized.
"""

# Native Python imports
import time
from copy import deepcopy

# Maniold optimization
from pymanopt.solvers.steepest_descent import SteepestDescent
from pymanopt.solvers.linesearch import LineSearchBackTracking

class CustomSteepestDescent(SteepestDescent):
    """
    Steepest descent (gradient descent) algorithm based on
    steepestdescent.m from the manopt MATLAB package.

    Subclasses the SteepestDescent solver.
    """

    def __init__(self, linesearch=None, *args, **kwargs):

        # Subclass the solver
        super().__init__(*args, **kwargs)

        # Store the intermediate results and associated errors
        self.estimates = []
        self.errors = []
        self.iterations = []

        # Line Search
        if linesearch is None:
            self._linesearch = LineSearchBackTracking()
        else:
            self._linesearch = linesearch
        self.linesearch = None

    # Function to solve optimisation problem using steepest descent.
    def solve(self, problem, x=None, reuselinesearch=False):
        """
        Perform optimization using gradient descent with linesearch.
        This method first computes the gradient (derivative) of obj
        w.r.t. arg, and then optimizes by moving in the direction of
        steepest descent (which is the opposite direction to the gradient).
        Arguments:
            - problem
                Pymanopt problem setup using the Problem class, this must
                have a .manifold attribute specifying the manifold to optimize
                over, as well as a cost and enough information to compute
                the gradient of that cost.
            - x=None
                Optional parameter. Starting point on the manifold. If none
                then a starting point will be randomly generated.
            - reuselinesearch=False
                Whether to reuse the previous linesearch object. Allows to
                use information from a previous solve run.
        Returns:
            - x
                Local minimum of obj, or if algorithm terminated before
                convergence x will be the point at which it terminated.
        """
        man = problem.manifold
        verbosity = problem.verbosity
        objective = problem.cost
        gradient = problem.grad

        if not reuselinesearch or self.linesearch is None:
            self.linesearch = deepcopy(self._linesearch)
        linesearch = self.linesearch

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        # Initialize iteration counter and timer
        iter = 0
        time0 = time.time()

        if verbosity >= 2:
            print(" iter\t\t   cost val\t    grad. norm")

        self._start_optlog(extraiterfields=['gradnorm'],
                           solverparams={'linesearcher': linesearch})

        # Reset intermediate results and associated errors
        self.estimates = []
        self.errors = []
        self.iterations = []

        while True:
            # Calculate new cost, grad and gradnorm
            cost = objective(x)
            grad = gradient(x)
            gradnorm = man.norm(x, grad)
            iter = iter + 1

            if verbosity >= 2:
                print("%5d\t%+.16e\t%.8e" % (iter, cost, gradnorm))

            if self._logverbosity >= 2:
                self._append_optlog(iter, x, cost, gradnorm=gradnorm)

            # Descent direction is minus the gradient
            desc_dir = -grad

            # Perform line-search
            stepsize, x = linesearch.search(objective, man, x, desc_dir,
                                            cost, -gradnorm**2)

            # Take intermediate results for later plotting
            self.estimates.append(x)
            self.errors.append(cost)
            self.iterations.append(iter)

            stop_reason = self._check_stopping_criterion(
                time0, stepsize=stepsize, gradnorm=gradnorm, iter=iter)

            if stop_reason:
                if verbosity >= 1:
                    print(stop_reason)
                    print('')
                break

        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(x, objective(x), stop_reason, time0,
                              stepsize=stepsize, gradnorm=gradnorm,
                              iter=iter)
            return x, self._optlog