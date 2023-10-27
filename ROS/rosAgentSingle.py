#!/usr/bin/env python
# coding=utf-8

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
from gpkg.msg import Task,Teammate,RobotState


# 多线程解决rospy.spin阻塞程序执行的问题
def thread_job():
    rospy.spin()


class RosSingleAgent():
    def __init__(self, robot, config):
        self.robot = robot
        self.max_detect = config.max_detect
        self.com_radius = config.com_radius

        self.spin_process = None

        try:
            # 节点
            rospy.init_node('ros_agent_{}'.format(self.robot.id+1), anonymous=True)
            # 机器人与Gazebo之间 
            self.control_pub = rospy.Publisher('/car{id}/cmd_vel'.format(id=self.robot.id+1), Twist, queue_size=10)  # 发布速度控制话题
            rospy.Subscriber("/car{id}/scan".format(id=self.robot.id+1), LaserScan, self.laser_callback, queue_size=1)
            rospy.Subscriber("/car{id}/odom".format(id=self.robot.id+1), Odometry, self.odom_callback, queue_size=1)
            # 机器人之间
            self.team_pub = rospy.Publisher("/robot{}_team".format(self.robot.id+1), Teammate, queue_size=10 )
            for i in range(config.num_agents):
                if i == self.robot.id:
                    continue
                rospy.Subscriber("/robot{}_team".format(i+1),Teammate, self.teammate_callback,queue_size=1)
            # 机器人与控制中心之间
            rospy.Subscriber("/robot{}_task".format(self.robot.id+1), Task, self.task_callback, queue_size=1)
            self.state_pub = rospy.Publisher("/robot{}_state".format(self.robot.id+1), RobotState, queue_size=10 )
        # rospy.spin()
        except rospy.ROSInterruptException:
            pass
        time.sleep(2)

    def print_shutdown(self):
        print("Shutdown rosagent Node")

    def task_callback(self,data):
        # 接收到任务的回调函数
        print("data.goal :{}".format(data.goal))
        print("data.start_run :{}".format(data.start_run))
        self.robot.fire = data.fire  # 开机/关机
        if data.fire:
            if data.start_run:   # 执行任务
                self.robot.set_goal(data.goal)
                self.robot.task_execute = True
                self.robot.end_run = False
                print("New task detected ")
            # self.robot.has_goal = True
            # self.robot.end_run = data.end_run # 启动/暂停
                self.robot.planner.use_opt = data.use_opt  # 使用优化器的类型 0- 固定左转 1- GNN 2- 专家数据
                print("optimizer : {}".format(data.use_opt))
                self.pub_robot_state()

                # for test ===========================================================================
                if self.robot.config.test_only_one:
                    # 只运行0号机器人 其他机器人不动
                    if self.robot.id != 0:
                        time.sleep(0.5)
                        self.robot.task_execute = False  # 不执行
                        self.robot.end_run = True  # 停止运行
                        self.robot.stop()  # 停止运行
                        self.pub_robot_state()  # 发布自身状态s
                # for test ===========================================================================

            else:
                if data.has_goal: # 发布任务
                    self.robot.set_goal(data.goal)
                    self.robot.task_execute = False  # 不执行
                    self.robot.end_run = True # 停止运行
                    print("robot load map : {}".format(data.map_id))
                    self.robot.load_global_map(data.map_id)
                    self.pub_robot_state()
                else:  # 停止运行
                    self.robot.task_execute = False  # 不执行
                    self.robot.end_run = True # 停止运行
                    self.robot.stop()  # 停止运行
                    self.pub_robot_state() # 发布自身状态
                    self.robot.reset_record()  # 清除记录
                    print("robot has benn reset")
                    self.pub_robot_state()

    def pub_robot_state(self, turn_dir=0):
        """向中心节点传送当前状态"""
        state = RobotState()
        state.turn_dir = turn_dir
        state.task_receive = self.robot.has_goal
        state.end_run = self.robot.end_run
        state.task_execute = self.robot.task_execute
        state.journey = self.robot.journey
        state.nn_output = self.robot.nn_output
        self.state_pub.publish(state)
        print("robot_{}'s state: has_goal : {} , task_execute : {}, end_run : {}, journey :{:.2f} , turn : {}"
              .format(self.robot.id, state.task_receive,state.task_execute,state.end_run, state.journey, state.turn_dir))

    # 控制小车运动
    def control(self, velocity, angular_velocity):
        cmd_msg = Twist()
        cmd_msg.linear.x = velocity
        cmd_msg.angular.z = angular_velocity * math.pi / 180
        self.control_pub.publish(cmd_msg)
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
    def laser_callback(self, data):
        self.robot.extract_laser_data(data)
        # self.simu_com(agent_id)   # 将绝对位置转化为相对位置从而模拟机器人之间的相对位置变化与更新

        # print("laser data updated")

    # 获取机器人的姿态
    def odom_callback(self, data):
        self.robot.extract_odom_data(data)    # 获取里程计感知信息
        self.teammate_pub()
        # print("odom data updated")
        # self.robot.auto_execute()

    # 获取队友的信息并存储到领进节点
    def teammate_callback(self,msg):
        # print("message from {}, \
        # position: ({},{}), yaw: {}".format(msg.sender_id, msg.x, msg.y, msg.yaw))
        # print("feature 0 : {}, feature 1 : {}".format(msg.private_feature[0:5], msg.first_order_feature[0:5]))
        # print("================================================================")
        if self.robot.position is None or self.robot.yaw is None or msg.x==-1:
            return
        if self.robot.use_expert:
            self.robot.all_robots_position[msg.sender_id,:] = np.array([msg.x, msg.y])   # save all robots‘ position for global expert
        rel_dist, rel_angle = self.robot.global_to_relative((msg.x,msg.y))
        rel_angle_index = self.robot.find_laser_angle_index(rel_angle)
        if rel_dist <= min(self.robot.com_radius, self.robot.scan_distance[rel_angle_index]+self.robot.radius+0.05):   # 其他智能体在通信范围内且在可感知范围内
            features = [np.array(msg.private_feature), np.array(msg.first_order_feature)]
            self.robot.neighborhood[msg.sender_id].set_value(rel_dist, rel_angle, features, True, msg.task_execute, msg.yaw)
        else:
            self.robot.neighborhood[msg.sender_id].reset()

    # 发送机器人信息
    def teammate_pub(self):
        msg = Teammate()
        msg.sender_id = self.robot.id
        msg.task_execute = self.robot.task_execute
        if self.robot.position is not None and self.robot.yaw is not None:  # ros话题订阅正常
            msg.x, msg.y = self.robot.position
            msg.yaw = self.robot.yaw
        else:
            msg.x, msg.y, msg.yaw = [-1 for i in range(3)]
        # print("robot {}'s position: {}".format(self.robot.id, self.robot.position))
        # print(self.robot.features[0].shape)
        # print(self.robot.features[1].shape)
        # print("=============================================================")
        msg.private_feature = self.robot.features[0].flatten().cpu().detach().numpy().tolist()
        msg.first_order_feature = self.robot.features[1].flatten().cpu().detach().numpy().tolist()
        self.team_pub.publish(msg)

    # 停止运行
    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        self.control_pub.publish(Twist(0))
        rospy.sleep(1)


# class TestAgent():
#     def __init__(self, robot,config):
#         self.robot = robot
#         self.config = config 
#         self.team_pub = rospy.Publisher("robot{}".format(self.robot.id), Teammate, queue_size=10 )
#         rospy.init_node("agent{}".format(self.robot.id), anonymous=True)
#         rospy.Subscriber("master", Goal, self.goal_callback, queue_size=1)
#         for i in range(config.num_agents):
#             if i == self.robot.id:
#                 continue
#             rospy.Subscriber("robot{}".format(i),Teammate, self.teammate_callback,queue_size=1)
#         # rospy.spin()
#         rate = rospy.Rate(10)
#         while not rospy.is_shutdown():
#             # str = "snake"
#             msg = Teammate()
#             msg.sender_id = self.robot.id
#             msg.x, msg.y = self.robot.position
#             # rospy.loginfo(msg.goal_dis)
#             self.team_pub.publish(msg)
#             print("publish personal state successfully!")
#             rate.sleep()

#     def goal_callback(self, data):
#         self.robot.goal_dis = data.goal_dis
#         self.robot.goal_angle = data.goal_angle
#         print("goal updated")

#     def teammate_callback(self,data):
#         print("get teammate data!")


