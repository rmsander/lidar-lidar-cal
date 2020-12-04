#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

import relative_pose_processing


main_odometry = relative_pose_processing.process_df('main_odometry_clean.csv')
front_odometry = relative_pose_processing.process_df('front_odometry_clean.csv')
rear_odometry = relative_pose_processing.process_df('rear_odometry_clean.csv')

(main_aligned, front_aligned, real_aligned) = relative_pose_processing.align_df([main_odometry, front_odometry, rear_odometry])

xcorr = scipy.signal.correlate(main_aligned.omega, front_aligned.omega)
print('Time delay:', np.argmax(xcorr) - len(main_aligned.omega))


plt.plot(xcorr)
plt.figure()
main_aligned.omega.plot()
front_aligned.omega.plot()

plt.show()


