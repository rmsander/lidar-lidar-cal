#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def clean_csv(fn):
    data = pd.read_csv(fn)
    data_new = pd.DataFrame()

    data_new['time'] = data['%time']
    #data_new.index = data_new.time

    data_new['x'] = data['field.pose.pose.position.x']
    data_new['y'] = data['field.pose.pose.position.y']
    data_new['z'] = data['field.pose.pose.position.z']

    data_new['qx'] = data['field.pose.pose.orientation.x']
    data_new['qy'] = data['field.pose.pose.orientation.y']
    data_new['qz'] = data['field.pose.pose.orientation.z']
    data_new['qw'] = data['field.pose.pose.orientation.w']

    for ix in range(36):
        data_new['cov_pose_%d' % ix] = data['field.pose.covariance%d' % ix]

    return data_new


if __name__ == '__main__':
    main_data = clean_csv('main_odometry.csv')
    main_data.to_csv('main_odometry_clean.csv')

    front_data = clean_csv('front_odometry.csv')
    front_data.to_csv('front_odometry_clean.csv')

    rear_data = clean_csv('rear_odometry.csv')
    rear_data.to_csv('rear_odometry_clean.csv')

    main_data.cov_pose_0.plot()
    plt.show()

    #front_data.x.plot()
    #front_data.y.plot()
    #front_data.z.plot()
    #plt.legend(['x','y','z'])

    #plt.figure()
    #front_data.qx.plot()
    #front_data.qy.plot()
    #front_data.qz.plot()
    #plt.plot(front_data.qx**2 + front_data.qy**2 + front_data.qz**2 + front_data.qw**2)
    #plt.legend(['qx','qy','qz','norm'])

    #plt.show()
