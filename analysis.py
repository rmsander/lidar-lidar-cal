#!/usr/bin/python3

import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R

import autograd.numpy as np

from pymanopt.manifolds import Euclidean, Rotations, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, TrustRegions

import relative_pose_processing

# def cost_full(X, A, B):
def cost_full(X):

    # Convert lists to autograd arrays
    A = np.array(main_rel_poses)
    B = np.array(front_rel_poses)

    # Initialize cost
    cost = 0

    # Estimates for rotation (R) and translation (t)
    R, t = X

    # Tab = np.zeros((4, 4))
    # Tab[:3, :3] = R
    # Tab[:3, 3] = t
    # Tab[3,3] = 1

    # Construct 4 x 4 pose manually
    R_t = np.hstack((R, t.reshape((3, 1))))  # 3 x 4 upper block
    T4 = np.array([0, 0, 0, 1])  # Use 1 in last entry for homogeneous coordinates
    Tab = np.vstack((R_t, T4))  # Merge 3 x 4 with 1 x 4 to get pose

    # Sum the cost over all poses
    for ix in range(len(A)):
        a = A[ix]
        b = B[ix]
        cost += np.square(np.linalg.norm(b @ Tab @ np.linalg.inv(a) - Tab))

    return cost

main_odometry = relative_pose_processing.process_df('main_odometry_clean.csv')
front_odometry = relative_pose_processing.process_df('front_odometry_clean.csv')
rear_odometry = relative_pose_processing.process_df('rear_odometry_clean.csv')

(main_aligned, front_aligned, real_aligned) = relative_pose_processing.align_df([main_odometry, front_odometry, rear_odometry])

main_rel_poses = relative_pose_processing.calc_rel_poses(main_aligned)
front_rel_poses = relative_pose_processing.calc_rel_poses(front_aligned)

# (1) Instantiate a manifold
translation_manifold = Euclidean(3)
so3 = Rotations(3)
manifold = Product((so3, translation_manifold))
# (2) Define the cost function (here using autograd.numpy)
# cost = lambda X: cost_full(X, main_rel_poses, front_rel_poses)

problem = Problem(manifold=manifold, cost=cost_full)

# (3) Instantiate a Pymanopt solver
solver = SteepestDescent()
#solver = TrustRegions()

# let Pymanopt do the rest
# To provide initial guess, do something like
# Xopt = solver.solve(problem, x=initial_guess)
Xopt = solver.solve(problem)
print(Xopt)

plt.show()

main_odometry.x.plot()
front_odometry.x.plot()
rear_odometry.x.plot()
plt.show()
