#!/usr/bin/python3

import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
import scipy.signal
from scipy.spatial.transform import Rotation as R

import autograd.numpy as np

from pymanopt.manifolds import Euclidean, Rotations, Product
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

#def cost_full(X, A, B):
def cost_full(X):
    A = main_rel_poses
    B = front_rel_poses
    cost = 0
    R, t = X
    Tab = np.zeros((4,4))
    Tab[:3,:3] = R
    Tab[:3,3] = t
    for ix in range(len(A)):
        a = A[ix]
        b = B[ix]
        cost += np.linalg.norm(b @ Tab @ np.linalg.inv(a) - Tab)**2

def retime_data(df):
    df['time'] = pd.to_datetime(df['time'], unit='ns')
    df = df.set_index('time')
    df = df.resample('.1S').mean()
    df = df.interpolate()
    return df

def quatmul(q1, q2):
    a,b,c,d = q1
    e,f,g,h = q2
    
    w = a*e - b*f - c*g - d*h
    x = b*e + a*f + c*h - d*g
    y = a*g - b*h + c*e + d*f
    z = a*h + b*g - c*f + d*e
    return [w,x,y,z]

def quatmul_delta_pandas(df, df_shift):
    # [dw, dx, dy, dz] * df = df_shift
    # q = df_shift * df^{-1}

    a = df_shift.qw
    b = df_shift.qx
    c = df_shift.qy
    d = df_shift.qz

    e = -df.qw # This inverts q1
    f = df.qx
    g = df.qy
    h = df.qz

    dw = (a*e - b*f - c*g - d*h).clip(-1,1)
    dx = b*e + a*f + c*h - d*g
    dy = a*g - b*h + c*e + d*f
    dz = a*h + b*g - c*f + d*e
    return [dw, dx, dy, dz]

def calc_rel_poses(df):
    x = df.x.values
    y = df.y.values
    z = df.z.values
    dqw =df.dqw.values
    dqx =df.dqx.values
    dqy =df.dqy.values
    dqz =df.dqz.values
    T_homg_list = []
    for ix in range(len(x)-1):
        T_homg = np.zeros((4,4))
        #rot = quaternion_matrix([dqw, dqx, dqy, dqz])
        rot = R.from_quat([dqx[ix], dqy[ix], dqz[ix], dqw[ix]]).as_matrix()
        dx = x[ix+1] - x[ix]
        dy = y[ix+1] - y[ix]
        dz = z[ix+1] - z[ix]
        T_homg[:3,:3] = rot
        T_homg[0,3] = dx
        T_homg[1,3] = dy
        T_homg[2,3] = dz
        T_homg[3,3] = 1
        T_homg_list.append(T_homg)

main_odometry = pd.read_csv('main_odometry_clean.csv')
front_odometry = pd.read_csv('front_odometry_clean.csv')
rear_odometry = pd.read_csv('rear_odometry_clean.csv')

main_odometry = retime_data(main_odometry)
front_odometry = retime_data(front_odometry)
rear_odometry = retime_data(rear_odometry)


[dqw,dqx,dqy,dqz] = quatmul_delta_pandas(main_odometry, main_odometry.shift())
main_odometry['dqw'] = dqw
main_odometry['dqx'] = dqx
main_odometry['dqy'] = dqy
main_odometry['dqz'] = dqz

[dqw,dqx,dqy,dqz] = quatmul_delta_pandas(front_odometry, front_odometry.shift())
front_odometry['dqw'] = dqw
front_odometry['dqx'] = dqx
front_odometry['dqy'] = dqy
front_odometry['dqz'] = dqz

[dqw,dqx,dqy,dqz] = quatmul_delta_pandas(rear_odometry, rear_odometry.shift())
rear_odometry['dqw'] = dqw
rear_odometry['dqx'] = dqx
rear_odometry['dqy'] = dqy
rear_odometry['dqz'] = dqz

main_odometry['dtheta'] = 2*np.arccos(main_odometry.dqw) - 2*np.pi
main_odometry['dtheta'][main_odometry['dtheta'] < -np.pi] += 2*np.pi
main_odometry['omega'] = main_odometry.dtheta / .1
main_odometry.omega.plot()
plt.title('main_odometry')
plt.ylabel('rad/s')

plt.figure()
front_odometry['dtheta'] = 2*np.arccos(front_odometry.dqw) - 2*np.pi
front_odometry['dtheta'][front_odometry['dtheta'] < -np.pi] += 2*np.pi
front_odometry['omega'] = front_odometry.dtheta / .1
front_odometry.omega.plot()
plt.title('front_odometry')
plt.ylabel('rad/s')


(main_aligned, front_aligned) = main_odometry.align(front_odometry, join='inner')
print(main_aligned)
main_aligned = main_aligned.fillna(0)
front_aligned = front_aligned.fillna(0)
xcorr = scipy.signal.correlate(main_aligned.omega, front_aligned.omega)
print('Time delay:', np.argmax(xcorr) - len(main_aligned.omega))


main_aligned = main_aligned.drop(main_aligned.index[0])
main_aligned = main_aligned.drop(main_aligned.index[-1])
front_aligned = front_aligned.drop(front_aligned.index[0])
front_aligned = front_aligned.drop(front_aligned.index[-1])

#plt.figure()
#main_aligned.dqw.plot()
#main_aligned.dqx.plot()
#main_aligned.dqy.plot()
#main_aligned.dqz.plot()
#plt.plot(main_aligned.dqw**2 + main_aligned.dqx**2 + main_aligned.dqy**2 + main_aligned.dqz**2)
#plt.legend(['w','x','y','z', 'norm'])
#plt.show()




main_rel_poses = calc_rel_poses(main_aligned)
front_rel_poses = calc_rel_poses(front_aligned)



plt.figure()
plt.plot(xcorr)
plt.show()
plt.figure()
main_rel_poses = calc_rel_poses(main_aligned)
print(main_rel_poses)
front_rel_poses = calc_rel_poses(front_aligned)



# (1) Instantiate a manifold
translation_manifold = Euclidean(3)
so3 = Rotations(3)
manifold = Product((so3, translation_manifold))
# (2) Define the cost function (here using autograd.numpy)
#cost = lambda X: cost_full(X, main_rel_poses, front_rel_poses)

problem = Problem(manifold=manifold, cost=cost_full)

# (3) Instantiate a Pymanopt solver
solver = SteepestDescent()

# let Pymanopt do the rest
Xopt = solver.solve(problem)
print(Xopt)




plt.figure()
plt.plot(xcorr)
plt.figure()
main_aligned.omega.plot()
front_aligned.omega.plot()


#plt.show()

#main_odometry.x.plot()
#front_odometry.x.plot()
#rear_odometry.x.plot()
#plt.show()




