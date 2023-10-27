#!/usr/bin/env python
# coding=utf-8

import turtle
import rospy
from geometry_msgs.msg import Twist, Point, Quaternion  # 从 geometry_nsg 中导入Twist消息类型
from sensor_msgs.msg import LaserScan
# from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from math import pi, radians, copysign, sqrt, pow
from easydict import EasyDict
import math
import time
import numpy as np
import multiprocessing
from envAgent import agent


# 多线程解决rospy.spin阻塞程序执行的问题
def thread_job():
    rospy.spin()


class ROSAgent(agent.EnvAgent):
    def __init__(self, robots, config):
        super(ROSAgent, self).__init__(robots)
        self.max_detect = config.max_detect
        self.com_radius = config.com_radius
        # self.robots = robots
        self.ctrl_pubs = []
        self.spin_process = None
        self.init_nodes()
        # self.p = multiprocessing.Process(target=self.init_nodes)
        # self.p.start()

    def init_nodes(self):
        try:
            rospy.init_node('ros_agent', anonymous=True)
            # self.start_spin()
            for i in range(len(self.robots)):
                control_pub = rospy.Publisher('/car{id}/cmd_vel'.format(id=i+1), Twist, queue_size=10)   # 发布速度控制话题
                self.ctrl_pubs.append(control_pub)
                rospy.Subscriber("/car{id}/scan".format(id=i+1), LaserScan, self.laser_callback, i, queue_size=1)
                rospy.Subscriber("/car{id}/odom".format(id=i+1), Odometry, self.odom_callback, i, queue_size=1)

        except rospy.ROSInterruptException:
            pass
        time.sleep(2)

    def print_shutdown(self):
        print("Shutdown rosagent Node")

    # def clear_nodes(self):
    #     # self.end_spin()
    #     rospy.signal_shutdown("closed!")
    #     self.ctrl_pubs = []

    # def start_spin(self):
    #     self.spin_process = multiprocessing.Process(target=rospy.spin)
    #     self.spin_process.start()
    #     time.sleep(2)

    # def end_spin(self):
    #     # self.shutdown()
    #     self.spin_process.terminate()
    #     print("Process terminated!")
    #     self.spin_process = None

    # # 控制指令发布
    # def control_publisher(self, vx, vy, av_z):
    #     # 对节点进行初始化，命名一个叫control_publisher的节点
    #     # 实例化一个发布者对象，发布的话题名为/cnd_ vel，消息类型为 Twist,队列长度为10
    #     rate = rospy.Rate(10)
    #     # 设置循环的频率
    #     while not rospy.is_shutdown():
    #         # 装载消息
    #         cmd_msg = Twist()
    #         cmd_msg.linear.x = vx
    #         cmd_msg.linear.y = vy
    #         cmd_msg.angular.z = av_z
    #         self.ctrl_pubs[0].publish(cmd_msg)
    #         rospy.loginfo("message have published")
    #         rate.sleep()

    # 控制小车运动
    def control(self, robot_id, velocity, angular_velocity):
        cmd_msg = Twist()
        cmd_msg.linear.x = velocity
        cmd_msg.linear.y = 0
        cmd_msg.linear.z = 0
        cmd_msg.angular.x = 0
        cmd_msg.angular.y = 0
        cmd_msg.angular.z = angular_velocity * math.pi / 180
        self.ctrl_pubs[robot_id].publish(cmd_msg)
        # print("linear velocity x {vx} y {vy} vaz {vaz}".format(
        #     vx=self.robots[robot_id].velocity[0], vy=self.robots[robot_id].velocity[1],
        #     vaz=self.robots[robot_id].angular_v))
        #
        # print("position x {x} y {y} yaw {yaw}".format(
        #     x=self.robots[robot_id].position[0], y=self.robots[robot_id].position[1], yaw=self.robots[robot_id].yaw))
        # print("control command have published ： " +
        #       "vx - {vx}, vy - {vy}, r_z - {vrz}".format(
        #           vx=cmd_msg.linear.x, vy=cmd_msg.linear.y, vrz=cmd_msg.angular.z))

    # 雷达call_back函数
    def laser_callback(self, data, agent_id):
        self.robots[agent_id].extract_laser_data(data)
        # print("laser data updated")
        # self.robots[agent_id].auto_execute()

    # 获取机器人的姿态
    def odom_callback(self, data, agent_id):
        self.robots[agent_id].extract_odom_data(data)    # 获取里程计感知信息
        self.simu_com(agent_id)   # 将绝对位置转化为相对位置从而模拟机器人之间的相对位置变化与更新
        # print("odom data updated")

    # 位置更新时模拟一次机器人间的通信
    def simu_com(self, agent_id):
        msg_id = self.robots[agent_id].send_simu_msg()
        for i in range(len(self.robots)):
            if i == agent_id:
                continue
            msg_i = self.robots[i].send_simu_msg()
            self.robots[agent_id].process_simu_msg(msg_i)
            self.robots[i].process_simu_msg(msg_id)

    # 停止运行
    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        for ctrl_pub in self.ctrl_pubs:
            ctrl_pub.publish(Twist(0))
        rospy.sleep(1)


if __name__ == "__main__":
    robots = []
    #
    # a = ROSAgent(robots,config)
    # a.control_publisher(1, 0, 0)


