#!/usr/bin/env python3

"""
rviz 配置：
1. 代码发布话题
2. 如何保存rviz配置： 保存rviz的配置
3. 如何在python中启动rviz节点： 在launch文件中写入与rviz的相关配置信息，需要加入一个TF节点
4. 如何保持记录轨迹并显示和保存
5. 如何应用到多个机器人的话题中
"""


import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion
import tf
import math

# 起始运动状态
x, y, th = 0, 0, 0


def DataUpdating(path_pub, path_record):
    """
    数据更新函数
    """
    global x, y, th

    # 时间戳
    current_time = rospy.Time.now()

    # 发布tf
    br = tf.TransformBroadcaster()
    br.sendTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                     rospy.Time.now(), "odom", "map")

    # 配置运动
    dt = 1 / 50
    vx = 0.25
    vy = 0.25
    vth = 0.2
    delta_x = (vx * math.cos(th) - vy * math.sin(th)) * dt
    delta_y = (vx * math.sin(th) + vy * math.cos(th)) * dt
    delta_th = vth * dt
    x += delta_x
    y += delta_y
    th += delta_th

    # 四元素转换
    quat = tf.transformations.quaternion_from_euler(0, 0, th)

    # 配置姿态
    pose = PoseStamped()
    pose.header.stamp = current_time
    pose.header.frame_id = 'odom'
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]

    # 配置路径
    path_record.header.stamp = current_time
    path_record.header.frame_id = 'odom'
    path_record.poses.append(pose)

    # 路径数量限制
    if len(path_record.poses) > 1000:
        path_record.poses.pop(0)
    # 发布路径
    path_pub.publish(path_record)


def node():
    """
    节点启动函数
    """
    try:

        # 初始化节点path
        rospy.init_node('PathRecord')

        # 定义发布器 path_pub 发布 trajectory
        path_pub = rospy.Publisher('trajectory', Path, queue_size=50)

        # 初始化循环频率
        rate = rospy.Rate(50)

        # 定义路径记录
        path_record = Path()

        # 在程序没退出的情况下
        while not rospy.is_shutdown():
            # 数据更新函数
            DataUpdating(path_pub, path_record)

            # 休眠
            rate.sleep()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    node()
