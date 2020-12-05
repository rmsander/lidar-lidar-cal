#!/usr/bin/python

import rospy

# Because of transformations
import tf_conversions

import tf2_ros
from geometry_msgs.msg import Pose, TransformStamped
from nav_msgs.msg import Odometry

def pose_to_tf(odom_msg, lidar_name):
    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()

    pos = odom_msg.pose.pose.position   
    quat = odom_msg.pose.pose.orientation

    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "l1_map"
    t.child_frame_id = lidar_name
    t.transform.translation.x = pos.x
    t.transform.translation.y = pos.y
    t.transform.translation.z = pos.z
    t.transform.rotation.x = quat.x
    t.transform.rotation.y = quat.y
    t.transform.rotation.z = quat.z
    t.transform.rotation.w = quat.w

    br.sendTransform(t)

if __name__ == '__main__':
    rospy.init_node('lidar_odom_tf_broadcaster')

    #rospy.Subscriber('/husky4/lo_frontend_front/odometry', Odometry, pose_to_tf, 'husky4/velodyne_front')
    rospy.Subscriber('/husky4/lo_frontend_main/odometry', Odometry, pose_to_tf, 'husky4/velodyne')
    #rospy.Subscriber('/husky4/lo_frontend_rear/odometry', Odometry, pose_to_tf, 'husky4/velodyne_rear')
    rospy.spin()
