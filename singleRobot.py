#!/usr/bin/env python

import time
import sys
import getopt
import random

import configure
import numpy as np
import time
import math
from utils import angle_transform, vector_angle
from easydict import EasyDict
# from GNN.GraphSAGE import GraphsageOptimizerNet
import torch
from tf.transformations import euler_from_quaternion
import dhbug
import multiprocessing
import threading
import rosAgentSingle
# multiprocessing.set_start_method('spawn')
import dhbug
from gpkg.msg import Task, RobotState
import os
import pynvml   # pip install nvidia-ml-py3
from multiRobot import Robot
import configure
import map
import utils

def most_free_gpu():
    """返回最空闲的GPU编号"""
    pynvml.nvmlInit()
    max_memory = 0
    most_free_device = 0
    num_device = pynvml.nvmlDeviceGetCount()
    for i in range(num_device):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)# 这里的0是GPU id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)  
        if meminfo.free / 1024 /1024 > max_memory:
            most_free_device = i
            max_memory = meminfo.free / 1024 /1024
    return most_free_device

# 机器人类: 用于分布式执行
class DistributedRobot(Robot):
    def __init__(self, robot_id, config, run_agent=None, planner=None, task=None,):
        super(DistributedRobot,self).__init__(robot_id,config)
        self.fire = True # 是否启动
        self.nn_output = [1,1]
        self.use_expert = config.use_expert
        if self.use_expert:  # 如果使用专家进行测试
            self.expert = map.load_A_star_expert()   # A star algorithm, need global map
            self.map_id = -1 # 用于加载专家数据需要的地图
            self.cur_map = None  # global map， used in cases where expert data are used to guide direction
            self.all_robots_position = np.zeros([config.num_agents, 2])   # num_robots * 2
            self.num_directions = config.num_directions
            self.special_points = []
            self.dis_predict = [sys.maxsize for i in range(self.num_directions)]

    def reset_record(self):
        super(DistributedRobot,self).reset_record()
        self.nn_output = [1,1]

    def load_global_map(self, map_id):
        """加载全局地图"""
        self.map_id = map_id
        self.cur_map = None
        if self.map_id > 0:
            npy_file_path = "numpyMapFolder/" + "map{:0>4d}.npy".format(self.map_id)
            self.cur_map = map.load_npy_map(npy_file_path, self.config.map_resolution)
            print("successfully load global map {}".format(self.map_id))

    def set_goal(self, goal):
        """设定任务"""
        self.goal = np.array(goal)
        self.check_goal()

        self.has_goal = True
        self.reach = False

    def adjust_angle(self, target_angle):
        """调整机器人自身角到给定的角度 """
        if abs(self.yaw-target_angle) > 10:
            self.run_agent.control(0, 0.1*(target_angle-self.yaw))
        else:
            self.run_agent.control(0, 0)

    def auto_execute(self):
        if not self.end_run:
            self.run_agent.control(self.plan_speed, self.plan_angular)
        else:
            return

    def execute_plan(self, velocity, angular_velocity):
        assert self.run_agent is not None, "Robot doesn't have execute agent"
        # self.run_agent.control(self.id, velocity/5, angular_velocity/5)    # for test
        self.run_agent.control(velocity, angular_velocity)

    def dispose_turn_point(self, turn_dir):
        # 记录由模态0切换到模态1、-1时的数据
        if self.need_record_nn and self.planner.use_opt==1:      # 当邻居智能体个数大于1时，使用图神经网络决策并记录数据
            self.run_agent.pub_robot_state(turn_dir=turn_dir)
            print("robot {} has neighbor {}, use optimizer {}--{} and record infomation" \
                  .format(self.id, self.check_neighbor(), self.nn_output, turn_dir))

    def require_optimizer_output(self):
        self.nn_output = super().require_optimizer_output()
        return self.nn_output

    def get_turn_dir(self, use_opt = 0):
        if use_opt == 1:  # 使用 GNN
            return self.advice_left_direction()
        elif use_opt == 2:   # 使用专家数据, 矫正运行
            return self.get_expert0_advice(pose_correct=True)
        elif self.use_opt == 3:   # 使用专家数据, 不矫正
            return  self.get_expert0_advice(pose_correct=False)
        else:
            return 1

    def get_expert0_advice(self, pose_correct=True):
        """获取专家数据生成方案的建议转向, 其中使用六个方向的打分，并用邻居在这六个方向的延长线上的交点来拓展打分方式"""
        since = time.time()
        self.special_points = [self.position]
        self.dis_predict = [sys.maxsize for i in range(self.num_directions)]
        goal_i_mx = self.cur_map.matrix_idx(self.goal[0], self.goal[1])  # matrix index of goal point on the global map
        position_i_mx = self.cur_map.get_safe_index(self.position)

        nghb_list = self.check_neighbor()
        jobs = []
        for k in range(self.num_directions):  
                p = threading.Thread(target=self._compute_own_score,args=(k, goal_i_mx, position_i_mx, pose_correct))
                p.start()
                jobs.append(p)
             # 计算相邻智能体中辅助感知位置的预估距离
        for j in nghb_list: 
            for k in range(self.num_directions):
                p = threading.Thread(target=self._compute_nghb_score,args=(j,k,goal_i_mx, position_i_mx, pose_correct))
                p.start()
                jobs.append(p)

        for job in jobs:
            job.join()
        print("compute expert score computation in {:.2f} seconds".format(time.time() - since))
        print("expert data : {}".format(self.dis_predict))
        case = self.cur_map.cases_generator(1, self.config.num_agents, self.radius)
        case[0] = [self.position, self.goal, self.yaw]
        self.cur_map.paint_paths([self.special_points], "result/"+"map{:0>4d}robot{:0>2d}expert_correct{}.png".format(self.map_id, self.id, int(pose_correct)), case)
        if pose_correct:
            score = np.array(self.dis_predict)
            fl_index = math.ceil((self.yaw % 360 - 30) / 60)  # first left index
            left_index = np.array([fl_index, fl_index+1, fl_index+2]) % 6
            right_index = np.array([fl_index-3, fl_index-2, fl_index-1]) % 6
            return np.min(score[left_index]) <= np.min(score[right_index])
        else:
            return self.dis_predict[1] <= self.dis_predict[4]

    def expert_global_dis(self, point1_mx, point2_mx):
        """计算全局路径
        point1_mx : 起始点在全局地图矩阵的索引
        point2_mx : 终止点在全局地图矩阵的索引
        """
        dis = map.compute_expert_dis(self.expert, self.cur_map.safe_map, point1_mx, point2_mx)
        if dis == -1:
            dis = sys.maxsize // 10
        return dis

    def straight_dis(self, point1_mx, point2_mx):
        """返回两个矩阵索引点之间的直线距离"""
        return np.linalg.norm(np.asarray(point1_mx) - np.asarray(point2_mx), 2)

    def _compute_own_score(self, k, goal_i_mx, position_i_mx, pose_correct=True):
        """
        Args:
            k:  direction id need to be computed
            pose_correction:  whether use yaw of the robot to correct its laser data
        Returns:
        """
        assert self.cur_map is not None, "need global map information"
        if pose_correct:
            dir = angle_transform(360 / self.num_directions * k + 30)  # 固定方向
        else:
            dir = angle_transform(self.yaw + 360 / self.num_directions * k + 30)  # 全局坐标系中机器人的参考方向对应的角度

        _, line_seg = utils.intersect_circle_ray(self.position, dir,
                                           self.position, self.com_radius)  # 全局坐标系中的方向线段
        reach_point = self.cur_map.find_point(line_seg)  # 最远能走到且无障碍的地图矩阵下标
        if reach_point is not None:
            # self.special_points.append(self.cur_map.abs_cor(reach_point[0], reach_point[1]))
            # dis = self.expert_global_dis(reach_point, goal_i_mx) + \
            #       self.straight_dis(position_i_mx, reach_point)
            # if dis < self.dis_predict[k]:
            #     self.dis_predict[k] = dis
            mid_points = self.cur_map.find_midpoint(position_i_mx, reach_point)  # 中间点 用于增加计算信息
            mid_points.append(reach_point)
            for cpu_point in mid_points:
                dis = self.expert_global_dis(cpu_point, goal_i_mx) + \
                                    self.straight_dis(position_i_mx, cpu_point)
                if dis < self.dis_predict[k]:
                    self.dis_predict[k] = dis
                self.special_points.append(self.cur_map.abs_cor(cpu_point[0], cpu_point[1]))

    def _compute_nghb_score(self,j,k,goal_i_mx,position_i_mx,pose_correct):
        # 计算位于position_i_mx的第i个智能体的第k个方向上与邻居j的交点处到矩阵索引goal_i_mx的专家距离
        if pose_correct:
            dir = angle_transform(360 / self.num_directions * k + 30)  # 固定方向
        else:
            dir = angle_transform(self.yaw + 360 / self.num_directions * k + 30)  # 全局坐标系中机器人的参考方向对应的角度

        # dir = angle_transform(robot.yaw + 360 / self.num_directions * k +30)
        has_inter, line_seg = utils.intersect_circle_ray(self.position, dir,
                                                    self.all_robots_position[j], self.com_radius)
        if has_inter:  # 若存在交点，通过交点计算估计距离
            reach_point = self.cur_map.find_point(line_seg)
            if reach_point is not None:
                mid_points = self.cur_map.find_midpoint(position_i_mx, reach_point)  # 中间点 用于增加计算信息
                mid_points.append(reach_point)
                for cpu_point in mid_points:
                    dis = self.expert_global_dis(cpu_point, goal_i_mx) + \
                          self.straight_dis(position_i_mx, cpu_point)
                    if dis < self.dis_predict[k]:
                        self.dis_predict[k] = dis
                    self.special_points.append(self.cur_map.abs_cor(cpu_point[0], cpu_point[1]))


def main(argv):

    try:
        opts, args = getopt.getopt(argv, "hr:", ["help", "robotid="])
    except getopt.GetoptError:
        print('Error: robot.py -r <robotid> -h <help>')
        print('   or: robot.py --robotid=<robotid> --help')
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('robot.py -r <robotid> -h <help>')
            print('or: robot.py --robotid=<robotid> --help')
            sys.exit()
        elif opt in ("-r", "--robotid"):
            robot_id = int(arg)


    gpu_id = most_free_gpu()
    # if robot_id < 11:
    #     gpu_id = 0
    # else:
    #     gpu_id = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id) 


    config = configure.get_config()

    robot = DistributedRobot(robot_id, config)
    robot.planner = dhbug.DhPathPlanner_HPoint(robot, config)
    robot.run_agent = rosAgentSingle.RosSingleAgent(robot, config)
    print("robot fire successfully!    robot state : end_run: {}".format(robot.end_run))

    while robot.fire:  # 启动后
        if robot.has_goal and not robot.end_run: # 有目标，并开始运行
            # robot.set_goal([18,2])  # for test
            # robot.get_expert_advice() # for test
            robot.run(max_time=config.max_run_time)
            robot.run_agent.pub_robot_state()
            # robot.reset_record()
        time.sleep(1)

if __name__ == "__main__":
    main(sys.argv[1:])
