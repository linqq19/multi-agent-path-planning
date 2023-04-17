import sys
import random
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
# multiprocessing.set_start_method('spawn')
# from keras.models import Model
# from keras.layers import Dense, Input
# from keras.models import load_model
import cmath


class GlobalMessage:
    def __init__(self, robot_id, position, yaw, features,task_execute):
        self.robot_id = robot_id
        self.global_position = position
        self.yaw = yaw
        self.features = features
        self.task_execute = task_execute


class NeighborInfo:
    """机器人存储的邻居信息的结构"""
    def __init__(self, sender=None):
        self.sender_id = sender
        self.task_execute = False
        self.features = [np.array([]) for i in range(3)]
        self.rel_dist = sys.maxsize
        self.rel_angle = 360
        self.connect = False
        self.RID = 0    # 仅添加了此变量未具体应用
        self.yaw = 0

    def set_value(self, dist, angle, features, connect, task_execute, yaw):
        self.rel_dist = dist
        self.rel_angle = angle
        self.features = features
        self.connect = connect
        self.task_execute = task_execute
        self.yaw = yaw

    def reset(self):
        self.features = [np.array([]) for i in range(3)]
        self.rel_dist = sys.maxsize
        self.rel_angle = 360
        self.connect = False
        self.task_execute = False
        self.yaw = 0


class Robot:
    def __init__(self, robot_id, config, run_agent=None, planner=None):
        # 机器人的环境和任务信息
        self.config = config
        self.dim_nn_output = config.dim_nn_output
        self.resolution = config.map_resolution # 矩阵相邻格点间的实际距离
        self.id = robot_id
        self.run_env_type = config.run_env_type
        self.has_goal = False
        self.task_execute = False # 是否开始执行任务
        self.goal = None     # 目标点 (全局坐標系)
        self.goal_dis = None  # 目标点距离 （机器人坐标系）
        self.goal_angle = None  # 目标点角度  （机器人坐标系）
        self.reach = False    # 是否到达目的地
        self.end_run = True  # 是否结束执行任务，结束时的两种情况：1. 到达目标点 2. 目标点判定为不可达
        # 机器人的位置姿态信息
        self.position = None   # 机器人的位置
        self.yaw = None    # 机器人的姿态角,角度制
        # self.velocity = np.array([0, 0])  # 机器人速度
        # self.angular_v = 0  # 机器人z轴角速度
        # 机器人的感知信息
        self.scan_distance = np.array([])  # 雷达扫出的距离
        self.scan_angles = np.array([])   # 雷达扫的角度
        # 机器人自身参数
        self.decel = config.agent_decel    # 机器人加速度
        self.radius = config.robotRadius   # 机器人半径
        self.constant_speed = config.constant_speed  # 设定的机器人最大行进速度
        self.com_radius = config.com_radius   # 机器人的通信半径
        self.max_detect = config.max_detect  # 机器人的最大感知距离
        self.dim_feature = config.dim_feature
        # 机器人交互
        self.num_hops = config.num_hops
        self.features = [torch.tensor([]) for i in range(self.num_hops+1)]  # 感知地图特征
        self.neighborhood = [NeighborInfo(sender=i) for i in range(config.num_agents)]   # 存储其他机器人的相关信息，自身的为空字典
        # 机器人实际实行代理
        self.run_agent = run_agent   # 算法中的机器人类与实际执行环境中机器人信息交互的代理
        self.is_agent_connect = True  # 代理是否链接上
        self.planner = planner   # 路径规划器
        # 优化器
        self.use_optimizer = config.use_optimizer
        if self.use_optimizer == 1:  # 神经网络优化器
            sys.path.append("./GNN")
            device = "cuda" if torch.cuda.is_available() else "cpu"

            if config.pose_correction:
                self.nn_optimizer = torch.load(config.absolute_nn_path, map_location=device)
                print("load model: "+config.absolute_nn_path)
            else:
                self.nn_optimizer = torch.load(config.relative_nn_path, map_location=device)
                print("load model: "+config.relative_nn_path)

            self.nn_optimizer.eval()
            # self.nn_output_dim = 2
            self.opt_output = None
            self.advice_left_direction = self.advice_left_direction_nn     # 优化器输出函数
        elif self.use_optimizer == 2:  # 概率图优化器
            pass
            # self.opt_output = None
            # self.encoder = load_model('encoder.h5', compile=False)  # 自编码器
            # self.advice_left_direction = self.advice_left_direction_bn  # 优化器输出函数
            #
            # self.FOV = config.max_detect  # 单位m
            # self.FOVshape = 203  # FOV矩阵的维度
            # self.FOVcentre = int((self.FOVshape - 1) / 2)  # FOV矩阵中心下标
            # self.grid = self.FOV / (self.FOVshape - 3)  # FOV矩阵中每一小格的长度
            #
            # self.input = np.zeros([1, 203, 203])
            # self.x0 = np.zeros(3)
            # self.r0 = 0
            # self.sita0 = 0

        if self.run_env_type in [0, 1]:  # ros仿真环境
            self.check_goal = self.check_goal_simu
        else:
            self.check_goal = self.check_goal_real

        # 评价指标
        self.test = True
        if self.run_env_type == 0:
            # self.velocity = 0  # 速度
            self.journey = 0  # 路程
            self.path = []

        self.plan_speed = 0
        self.plan_angular = 0
        self.record_time = 0

        self.need_record_nn = False

    def reset_record(self):
        self.stop()
        self.planner.reset()
        self.has_goal = False
        self.goal = None     # 目标点 (全局坐標系)
        self.goal_dis = None  # 目标点距离 （机器人坐标系）
        self.goal_angle = None  # 目标点角度  （机器人坐标系）
        self.reach = False    # 是否到达目的地
        self.end_run = True  # 是否结束执行任务，结束时的两种情况：1. 到达目标点 2. 目标点判定为不可达
        self.position = None   # 机器人的位置
        self.yaw = None    # 机器人的姿态角
        self.scan_distance = np.array([])  # 雷达扫出的距离
        self.scan_angles = np.array([])   # 雷达扫的角度
        self.features = [torch.tensor([]) for i in range(self.num_hops+1)]  # 感知地图特征
        self.neighborhood = [NeighborInfo(sender=i) for i in range(self.config.num_agents)]   # 存储其他机器人的相关信息，自身的为空字典
        if self.run_env_type == 0:
            # self.velocity = 0  # 速度
            self.journey = 0  # 路程
            self.path = []

        self.plan_speed = 0
        self.plan_angular = 0
        self.need_record_nn = False
        self.is_agent_connect = False  # 代理是否连接上
        self.task_execute = False # 是否开始执行任务

    def stop(self):
        """停止运行"""
        self.execute_plan(0,0)

    def set_goal(self, goal):
        """设定任务"""
        self.goal = np.array(goal)
        self.check_goal()

        self.reach = False
        self.end_run = False
        self.has_goal = True
        self.task_execute = True

    def has_reach_goal(self):
        """判断时候到达目标位置，到达返回True，否则返回False"""
        self.check_goal()
        goal_errors = self.scan_angles - self.goal_angle
        goal_index = np.where(goal_errors == np.min(goal_errors))[0][0]
        if self.goal_dis < min(self.scan_distance[goal_index],0.7):  # 目标位于感知范围内，且已到达目标点0.1米范围内
            if abs(self.goal_angle) < 10: # 正对目标
                return 1
            else: # 尚未正对目标
                return 0
        else:  # 未到达目标范围内或者看不到目标
            return -1

    def check_goal_simu(self):
        """仿真环境中根据全局坐标系计算目标相对于机器人的方向和距离"""
        # assert self.run_env_type == 0, "Not ROS simulation environment!"  # ROS-Gazebo仿真环境
        assert self.goal is not None, "No goal info ！ Check whether robot has task to do"
        # 计算目标在本体坐标系中的角度
        if self.position is not None:
            self.goal_angle = vector_angle(self.goal - self.position, self.yaw)
            self.goal_dis = np.linalg.norm(self.goal - self.position, 2)
            # self.has_goal = True

    def check_goal_real(self):
        """实际环境中用于设置目标的函数"""
        pass


    def extract_laser_data(self, data):
        """提取激光雷达获取的信息并保存，同时更新根据环境信息提取出的特征变量"""
        self.scan_angles = np.array([data.angle_min + i * data.angle_increment for i in range(len(data.ranges))])\
                           * 180 / math.pi  # 弧度转化为角度
        scan_dist = np.array(data.ranges)
        scan_dist[scan_dist > self.max_detect] = self.max_detect
        self.scan_distance = scan_dist
        self.update_optimizer()    # 对于有实时更新需求的优化器，在雷达的回调函数中同时更新
        # print("robot {} 's  laser data updated !, use time {}".format(self.id, time.time() - self.record_time))
        self.record_time = time.time()

    def update_optimizer(self):
        """更新优化器输出结果"""
        # if not (self.has_goal and self.task_execute):  # 执行任务过程中更新
        #     self.update_own_feature()
        self.update_own_feature()   # 对于GNN来说，只要与周围机器人发生交互，周围机器人的特征更新就需要一直保持，原因：邻居的特征融合了自身的信息



    def extract_odom_data(self, data):
        """提取里程计获取的位置和姿态信息并保存"""
        posi = data.pose.pose.position
        # if self.id == 0:
        #     print("odom updated, cur path points num {}".format(len(self.path)))
        if self.has_goal and self.task_execute:
            if len(self.path) == 0:
                self.path.append(np.array([posi.x, posi.y]))
            else:
                self.journey += np.linalg.norm(np.array([posi.x, posi.y])-self.position,2)
                # if self.id == 0:
                #     print("current journey {}".format(self.journey))
        self.position = np.array([posi.x, posi.y])

        direc = data.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([direc.x, direc.y, direc.z, direc.w])
        self.yaw = yaw * 180 / math.pi

        if self.run_env_type==0 and not self.end_run and self.has_goal and len(self.path)>0:
            if np.linalg.norm(self.path[-1]-self.position,2) > 0.02:
                self.path.append(self.position)


    def update_own_feature(self):
        """更新机器人自身感知信息获取的特征"""
        if not self.has_goal:
            return
        self.check_goal()
        if self.use_optimizer == 1:
            with torch.no_grad():
                # self.features = [torch.tensor([]) for i in range(self.num_hops+1)]
                device = "cuda" if torch.cuda.is_available() else "cpu"   # 根据环境确定cpu或gpu
                percept_info = torch.tensor(self.get_map_info(), device=device, dtype=torch.float).unsqueeze(0)  # 获取三通道感知信息
                feature = self.nn_optimizer.cnn_feature_extract(percept_info)  # 由CNN压缩特征
                assert min(feature.shape) != 0, "Error! perception info don't get"
                # features = []
                self.features[0] = feature
                neighbor_list = self.check_neighbor()
                for hop in range(self.num_hops):
                    raw_features = []
                    for idx in neighbor_list:
                        neighbor_feature = self.neighborhood[idx].features[hop].astype(float)
                        if min(neighbor_feature.shape) > 0:
                            raw_features.append(neighbor_feature)
                    if len(raw_features) > 0:
                        neighbor_features = torch.tensor(np.vstack(raw_features), device=device, dtype=torch.float)
                        # print("feature fusion process: hop {}, {},{}".format(hop, feature.shape, neighbor_features.shape))
                        feature = self.nn_optimizer.feature_fusion(hop, feature, neighbor_features)
                        feature = torch.squeeze(feature, dim=0)
                        # print("dim after fusion: {}".format(feature.shape))
                    else:
                        neighbor_features = torch.tensor([])
                        # print("feature fusion process: hop {}, {},{}".format(hop, feature.shape, neighbor_features.shape))
                        feature = self.nn_optimizer.feature_fusion(hop, feature, neighbor_features)
                        # print("dim after fusion: {}".format(feature.shape))

                    self.features[hop+1] = feature

        # elif self.use_optimizer == 2:
        #     channels = self.get_map_info()
        #
        #     # 提取自身感知通道
        #     self.input = channels[0]
        #     self.input = self.input.reshape(-1)
        #     self.input = np.expand_dims(self.input, axis=0)
        #     self.x0 = self.encoder(self.input, training=False)[0]
        #
        #     # 提取目标位置通道 channel2
        #     channel2 = channels[2]  # 提取目标位置通道
        #     tmp_p0 = np.argwhere(channel2 == 1)
        #     tmp_p0 = tmp_p0.reshape(2)
        #     if tmp_p0.shape[0] == 0:  # 十分接近目标导致目标丢失
        #         tmp_p0 = np.array([0, 0])
        #
        #     tmp_p0 -= self.FOVcentre
        #     tmp = complex(tmp_p0[0], tmp_p0[1])  # 创建一个复数来进行坐标转换
        #     r, sita = cmath.polar(tmp)  # 计算长度和角度
        #     self.r0 = r * self.grid
        #     self.sita0 = sita
        #
        # else:
        #     pass

    # def require_optimizer_output(self):
    #     """获取神经网络的输出值"""
    #     if not self.has_goal:
    #         return
    #     assert self.use_optimizer, "No optimizer find"
    #     neighbor_list = self.check_neighbor()
    #     if len(neighbor_list) == 0 or self.features is None or min(self.features[-1].shape) == 0:
    #         # print("robot {} current features: {}".format(self.id, self.features))
    #         return [1, 1, 1, 1, 1, 1]
    #     return self.nn_optimizer.feature_to_action(self.features[-1]).cpu().detach().numpy()

    def require_optimizer_output(self):
        """获取神经网络的输出值"""
        nn_output = np.ones(self.dim_nn_output)
        if not self.has_goal:
            return nn_output
        assert self.use_optimizer==1, "Not gnn optimizer!"
        neighbor_list = self.check_neighbor()
        if  self.features is None or min(self.features[-1].shape) == 0 or len(neighbor_list) == 0:
            print("robot {} has {} neighbor".format(self.id,len(neighbor_list)))
            return nn_output
        self.need_record_nn = True # 要记录数据
        print("robot {} has neighbor {}, need record data, current position :{} ".format(self.id, neighbor_list, self.position))
        # print("neighbor position : {} ".format(np.vstack([self.master.robots.robots[i].position for i in neighbor_list])))
        # for feature in self.features:
        #     print(feature)
        with torch.no_grad():
            nn_output = self.nn_optimizer.feature_to_action(self.features[-1]).cpu().detach().numpy().flatten()
            # print(nn_output)
            return nn_output

    def advice_left_direction_nn(self):
        """根据神经网络的输出对左右两个方向进行打分
        :return: True:左边更优， False： 右边更优"""
        if not self.has_goal:
            return
        self.opt_output = self.require_optimizer_output()
        if self.opt_output is None:
            return True   # 默认左边更好
        # scores = self.require_optimizer_output()
        # scores = self.opt_output
        print("robot {} has neighbor {}, need record data, current position :{} ".format(self.id, self.check_neighbor(), self.position))
        print("robot {} optimizer output {}".format(self.id, self.opt_output))

        # print(self.opt_output)
        if self.dim_nn_output == 2:
            left_far_prob, right_far_prob = self.opt_output
            return left_far_prob <= right_far_prob
        elif self.dim_nn_output == 6 and self.config.pose_correction:
            fl_index = math.ceil((self.yaw % 360 - 30) / 60)  # first left index
            left_index = np.array([fl_index, fl_index+1, fl_index+2]) % 6
            right_index = np.array([fl_index-3, fl_index-2, fl_index-1]) % 6
            return np.max(self.opt_output[left_index]) >= np.max(self.opt_output[right_index])


    def advice_left_direction_bn(self):
        """根据概率图模型的输出对左右两个方向进行打分
        :return: True:左边更优， False： 右边更优"""
        if not self.has_goal:
            return

        x0 = self.x0
        r0 = self.r0
        sita0 = self.sita0

        w = np.random.normal(-0.433, 0.714)
        w0 = np.random.normal([-0.132, -0.158, -0.014], 0.009)

        k = np.random.normal(0.075, 0.031)
        b = np.random.normal(-0.032, 0.996)
        c = np.random.normal(-0.432, 0.714)
        d = np.random.normal(2.328, 0.095)

        z = w + np.dot(w0, x0) + k * (r0 + b) + (c + d * math.sin(sita0))
        t = 1 / (1 + math.e ** (-z))  # sigmoid

        if t < 0.5:
            return False
        else:
            return True

    def check_neighbor(self):
        """
        :return: 返回当前通信范围内的邻居节点的id列表
        """
        connect_list = []
        for i in range(len(self.neighborhood)):
            if self.neighborhood[i].connect:
                connect_list.append(i)
        return connect_list

    def global_to_relative(self, global_position):
        """
         计算一个全局坐标相对于自身的方位
        :param global_position:  地图坐标系下的位置坐标
        :return: 机器人坐标系中的距离和角度
        """
        vec = global_position-self.position
        dist = np.linalg.norm(vec, 2)
        angle = angle_transform(np.arctan2(vec[1], vec[0]) * 180 / np.pi - self.yaw)   # 相对于机器人坐标系的方位和角度
        return dist, angle

    def get_laser_info(self):
        """获取激光雷达的感知角度信息、感知距离信息"""
        # self.scan_distance[self.scan_distance > self.max_detect] = self.max_detect  # 对雷达感知数据进行模拟的处理
        return self.scan_angles, self.scan_distance

    def local_matrix_shape(self):
        num_detect_point = math.floor(self.max_detect / self.resolution)
        return 2 * num_detect_point + 1 + 2  # 边缘增加投影关系的部分

    def get_local_map(self, post_correct=None):
        """
        返回智能体的感知地图矩阵，矩阵维度由最大感知距离和感知分辨率确定
        :return: 感知地图矩阵
        """
        if post_correct is None:
            post_correct = self.config.pose_correction

        if len(self.scan_angles) == 0:
            raise Exception("Robot not Connected")
        num_detect = math.floor(self.max_detect / self.resolution)  # 最远可探测的矩阵格点数
        n = 2 * num_detect + 1 + 2
        matrix = np.zeros([n, n])
        for i in range(len(self.scan_angles)):
            rho = self.scan_distance[i] / self.resolution
            if post_correct:
                theta = (self.scan_angles[i] + self.yaw) * math.pi / 180  # 朝北校准
            else:
                theta = self.scan_angles[i] * math.pi / 180  # 取消朝北校准
            obs_x = num_detect + 1 - math.floor(rho * math.cos(theta))  # 障碍物位置索引
            obs_y = num_detect + 1 - math.floor(rho * math.sin(theta))
            matrix[obs_x, obs_y] = 1
        return matrix

    def get_neighbour_map(self, post_correct=None):
        """
        返回智能体感知范围内的邻结点地图0-1矩阵，其中1表示其他智能体所在位置，矩阵维度与感知地图维度相同
        :return:
        """
        if post_correct is None:  # 默认使用configure配置的是否进行矫正
            post_correct = self.config.pose_correction

        num_detect = math.floor(self.max_detect / self.resolution)  # 最远可探测的矩阵格点数
        n = 2 * num_detect + 1 + 2
        matrix = np.zeros([n, n])
        for agent in self.neighborhood:
            if agent.connect:
                dist = agent.rel_dist / self.resolution
                if post_correct:
                    angle = (agent.rel_angle + self.yaw) * math.pi / 180
                else:
                    angle = agent.rel_angle* math.pi / 180
                neighbor_x = num_detect + 1 - math.floor(dist * math.cos(angle))
                neighbor_y = num_detect + 1 - math.floor(dist * math.sin(angle))
                matrix[neighbor_x, neighbor_y] = 1
        # matrix[num_detect + 1, num_detect + 1] = 1
        return matrix

    def get_neighbor_id_map(self, post_correct=None):
        """
        返回智能体感知范围内的邻结点地图0-1矩阵，其中1表示其他智能体所在位置，矩阵维度与感知地图维度相同
        :return:
        """
        if post_correct is None:  # 默认使用configure配置的是否进行矫正
            post_correct = self.config.pose_correction

        num_detect = math.floor(self.max_detect / self.resolution)  # 最远可探测的矩阵格点数
        n = 2 * num_detect + 1 + 2
        matrix = np.zeros([n, n])
        for agent in self.neighborhood:
            if agent.connect:
                dist = agent.rel_dist / self.resolution
                if post_correct:
                    angle = (agent.rel_angle + self.yaw) * math.pi / 180
                else:
                    angle = agent.rel_angle* math.pi / 180
                neighbor_x = num_detect + 1 - math.floor(dist * math.cos(angle))
                neighbor_y = num_detect + 1 - math.floor(dist * math.sin(angle))
                save_id = agent.sender_id
                if save_id == 0:
                    save_id = -1
                matrix[neighbor_x, neighbor_y] = save_id  # id 保存
        save_id = self.id
        if save_id == 0:
            save_id = -1
        matrix[num_detect + 1, num_detect + 1] = save_id
        return matrix


    def get_neighbor_yaw_map(self, post_correct=None):
        """
        返回智能体感知范围内的邻结点地图0-1矩阵，其中1表示其他智能体所在位置，矩阵维度与感知地图维度相同
        :return:
        """
        if post_correct is None:  # 默认使用configure配置的是否进行矫正
            post_correct = self.config.pose_correction

        num_detect = math.floor(self.max_detect / self.resolution)  # 最远可探测的矩阵格点数
        n = 2 * num_detect + 1 + 2
        matrix = np.zeros([n, n])
        for agent in self.neighborhood:
            if agent.connect:
                dist = agent.rel_dist / self.resolution
                if post_correct:
                    angle = (agent.rel_angle + self.yaw) * math.pi / 180
                else:
                    angle = agent.rel_angle* math.pi / 180
                neighbor_x = num_detect + 1 - math.floor(dist * math.cos(angle))
                neighbor_y = num_detect + 1 - math.floor(dist * math.sin(angle))
                matrix[neighbor_x, neighbor_y] = agent.yaw  # 角度制保存
        matrix[num_detect + 1, num_detect + 1] = self.yaw
        return matrix


    def get_goal_map(self, post_correct=None):
        """
        返回智能体感知范围内的目标0-1矩阵，其中1表示目标位置，当目标位于智能体感知范围外时，
        1表示目标在智能体感知边界上的投影位置
        :return:
        """
        if post_correct is None:
            post_correct = self.config.pose_correction
        num_detect = math.floor(self.max_detect / self.resolution)  # 最远可探测的矩阵格点数
        n = 2 * num_detect + 1 + 2
        matrix = np.zeros([n, n])
        rho = min(self.goal_dis, self.max_detect + self.resolution) / self.resolution  # 对目标进行投影
        if post_correct:
            angle = (self.goal_angle + self.yaw) * math.pi / 180
        else:
            angle = self.goal_angle * math.pi / 180 
        goal_x = num_detect + 1 - math.floor(rho * math.cos(angle))  # 目标位置索引
        goal_y = num_detect + 1 - math.floor(rho * math.sin(angle))
        matrix[goal_x, goal_y] = 1
        return matrix

    def get_map_info(self,  post_correct=None, whole_info = False):
        """
        返回智能体的三通道信息
        post_correct : 是否对邻居和自身的姿态进行矫正
        whole_info : 是否加载全部信息，在生成数据时设定为True
        :return:
        """
        if post_correct is None:
            post_correct = self.config.pose_correction
        map_local = self.get_local_map(post_correct)
        position_local = self.get_neighbour_map(post_correct)
        goal_local = self.get_goal_map(post_correct)
        if whole_info:   # 生成专家数据时用于存储全部信息
            yaw_local = self.get_neighbor_yaw_map(post_correct)
            id_local = self.get_neighbor_id_map(post_correct=post_correct)
            local_info = np.stack((map_local, position_local, goal_local, yaw_local, id_local), axis=0)
        else:
            local_info = np.stack((map_local, position_local, goal_local), axis=0)
        # local_info = np.concatenate((map_local, state_local, goal_local),axis=1)
        return local_info  # 3 * n * n

    def plan(self):
        """使用路径规划器规划速度和角速度"""
        assert self.planner is not None, "Robot doesn't have path planner!"
        try:
            v, v_az  = self.planner.run()
        except:
            v, v_az = 0, 0
        return v, v_az

    def auto_plan(self):
        """自动进行规划"""
        while not self.end_run:
            self.plan_speed, self.plan_angular = self.planner.run()


    def execute_plan(self, velocity, angular_velocity):
        """执行控制值令，因实际指令传达方式的不同，需以不同的形式实现"""
        pass
        # print("No control command dirtributed , please check your execute agent realization!")

    def run(self, max_time=180):
        """单个机器人执行dh-bug路径规划"""
        self.execute_plan(0,0)
        time.sleep(3)
        start_time = time.time()
        last_mode = 0
        cur_mode = 0
        while not self.end_run and time.time()-start_time<max_time:
            v, v_az = self.plan()
            cur_mode = self.planner.get_mode()
            if time.time() - self.record_time > 0.4:   # or self.id == 0:
                self.execute_plan(0, 0)
                # time.sleep(0.2)
            elif last_mode == 0 and abs(cur_mode)==1:
                self.execute_plan(0, 0)
                time.sleep(0.4)
            else:
                self.execute_plan(v, v_az)
            last_mode = cur_mode
            # time.sleep(0.2)
            # self.execute_plan(0,0)
            # time.sleep(0.5)
            # if self.id == 0:
            #     print("robot {}, pos: {}, opt: {}, neighbor:{}, act_mode : {}".format(self.id, self.position, self.require_optimizer_output(),self.check_neighbor(), self.planner.act_mode))
        self.stop()
        self.end_run = True

    def dispose_turn_point(self, turn_dir):
        # 记录由模态0切换到模态1、-1时的数据
        pass

    def find_laser_angle_index(self, angle):
        if np.min(self.scan_angles.shape) == 0:
            return None
        return np.argmin(np.abs(self.scan_angles - angle))

    def is_insight(self, rel_dist, rel_angle):
        rel_angle_index = self.find_laser_angle_index(rel_angle)
        if rel_dist < min(self.com_radius, self.scan_distance[rel_angle_index]+self.radius+0.05):
            return True
        else:
            return  False

    def is_insight_global(self, global_position):
        # 全局坐标点是否在自身视野范围内且可通信
        rel_dist, rel_angle = self.global_to_relative(global_position)
        return self.is_insight(rel_dist, rel_angle)

    def get_turn_dir(self, use_opt = 0):
        return self.advice_left_direction()


class CentralizedOneRobot(Robot):
    def __init__(self, robot_id, config, run_agent=None, planner=None):
        super(CentralizedOneRobot,self).__init__(robot_id,config)
        # 记录转向时的相关数据
        self.opt_turn_odom = []
        self.opt_turn_laser = []
        self.opt_turn_dirs = []
        self.master = None

        self.dcs_pos = None
        self.nn_output = None

    def reset_record(self):
        super(CentralizedOneRobot,self).reset_record()
        self.opt_turn_odom = []
        self.opt_turn_laser = []
        self.opt_turn_dirs = []
        self.master = None
        self.dcs_pos = None
        self.nn_output = None

    def process_simu_msg(self, msg):
        """根据从代理处获得的全局信息转化为相对的局部信息"""
        # if not self.has_goal:
        #     self.neighborhood[msg.robot_id].set_value(sys.maxsize, sys.maxsize, None, None, True)
        #     return
        if self.position is None or self.yaw is None or msg.global_position is None:
            return
        try:
            rel_dist, rel_angle = self.global_to_relative(msg.global_position)
            rel_angle_index = self.find_laser_angle_index(rel_angle)
            if rel_dist <= min(self.com_radius, self.scan_distance[rel_angle_index]+self.config.robotRadius+0.05):   # 邻居自身会对激光雷达的测距产生影响
                    self.neighborhood[msg.robot_id].set_value(rel_dist, rel_angle, msg.features, True, msg.task_execute, msg.yaw)
            else:
                self.neighborhood[msg.robot_id].reset()
        except IndexError:
            print("lack personal perception data ! ")


    def send_simu_msg(self):
        """仿真环境下模拟信息传递，地处线传递绝对坐标，后使用距离模拟通信能力"""
        # neighbor_features = self.get_neighbor_features()
        if self.features is None:
            self_feature = None
        else:
            self_feature = [self.features[i].cpu().detach().numpy() for i in range(len(self.features))]
        msg = GlobalMessage(self.id, self.position,self.yaw, self_feature,self.task_execute)
        return msg

    def adjust_angle(self, target_angle):
        """调整机器人自身角到给定的角度 """
        if abs(self.yaw-target_angle) > 10:
            self.run_agent.control(self.id, 0, 0.1*(target_angle-self.yaw))
        else:
            self.run_agent.control(self.id, 0, 0)

    def auto_execute(self):
        if not self.end_run:
            self.run_agent.control(self.id, self.plan_speed, self.plan_angular)
        else:
            return

    def execute_plan(self, velocity, angular_velocity):
        assert self.run_agent is not None, "Robot doesn't have execute agent"
        # self.run_agent.control(self.id, velocity/5, angular_velocity/5)    # for test
        self.run_agent.control(self.id, velocity, angular_velocity)

    def dispose_turn_point(self, turn_dir):
        # 记录由模态0切换到模态1、-1时的数据
        if self.need_record_nn:      # 当邻居智能体个数大于1时，使用图神经网络决策并记录数据
            print("robot {} has neighbor, use optimizer and record infomation".format(self.id))
            # self.opt_turn_dirs.append((turn_dir))
            # self.opt_turn_odom.append([self.position[0], self.position[1], self.yaw])
            # self.opt_turn_laser.append(self.get_map_info())
            self.master.record_decision(turn_dir)

    def require_optimizer_output(self):
        self.dcs_pos = self.position
        self.nn_output = super(CentralizedOneRobot, self).require_optimizer_output()
        return self.nn_output

    # def extract_odom_data(self, data):
    #     if not self.end_run:
    #         super(CentralizedOneRobot, self).extract_odom_data(data)


class Robots:
    def __init__(self, config, dh_planner=True,):
        self.num_agents = config.num_agents
        self.robots = [CentralizedOneRobot(i, config) for i in range(self.num_agents)]
        for cur_robot in self.robots:
            cur_robot.planner = dhbug.DhPathPlanner_HPoint(cur_robot, config)
        self.com_radius = config.com_radius


    def set_run_agent(self, run_agent):
        for cur_robot in self.robots:
            cur_robot.run_agent = run_agent

    def set_master(self, master):
        for robot in self.robots:
            robot.master = master

    def set_task(self, case):
        """
        给机器人设定任务
        :param case:
        :return:
        """
        for i in range(len(self.robots)):
            self.robots[i].set_goal(case[i][1])
            self.robots[i].planner.reset()

    def check_run_state(self):
        """
        检查当前任务运行状态
        :return: 只要存在一个机器人尚未到达目的地，返回False, 否则返回True
        """
        for i in range(self.num_agents):
            if not self.robots[i].end_run:
                return False
        return True

    def run_dh_bug(self, max_time=180):
        """
        设定任务，并执行DH-bug算法
        :return:
        """
        # self.set_task(case)s
        # self.init_topic_and_optimizer()
        # 运行DH-bug算法
        run_pool = list(range(self.num_agents))
        start_time = time.time()
        # # for test
        # self.robots[0].planner.act_mode = 2
        # self.robots[0].planner.minD = 0.1
        # self.robots[0].planner.bypass_points.append(self.robots[0].position)
        while len(run_pool):
            for i in run_pool:
                if self.robots[i].end_run:  # 若第i个agent
                    run_pool.remove(i)
                    break
                since = time.time()


                v, v_az = self.robots[i].plan()


                print("time used for planning of robot {} is {:.2f},  plam speed {} and {}, act_mode : {}".format(i, time.time()-since, v, v_az, self.robots[i].planner.act_mode))
                if time.time() - self.robots[i].record_time > 0.4:
                    self.robots[i].execute_plan(0, 0)
                    continue
                else:
                    self.robots[i].execute_plan(v, v_az)
                    time.sleep(0.2)
                    self.robots[i].execute_plan(0, 0)
                # time.sleep(0.1)
                print("robots {}'s position {}, optimizer output {}".format(i, self.robots[i].position, self.robots[i].require_optimizer_output()))
            run_time = time.time() - start_time
            if run_time > max_time:
                self.stop()
                print(" dh-bug algorithm executes overtime ")
                break
        for robot in self.robots:
            robot.end_run = True
        path_length = [robot.journey for robot in self.robots]
        return np.array(path_length)

    def run_dh_bug_4(self, max_time=180):
        """
        设定任务，并执行DH-bug算法
        :return:
        """
        self.init_topic_and_optimizer()
        # 运行DH-bug算法
        jobs = []
        for i in range(self.num_agents):
            p = threading.Thread(target=self.robots[i].auto_plan)
            p.start()
            jobs.append(p)
        for job in jobs:
            job.join(max_time)
        path_length = [robot.journey for robot in self.robots]
        return np.array(path_length)

    def init_topic_and_optimizer(self):
        # 初始化运行规划器及优化器
        for robot in self.robots:
            robot.plan()
            robot.execute_plan(0,0)
            robot.update_own_feature()
        time.sleep(2)

    def run_dh_single(self, case, max_time=180):
        # 只运行一个机器人的DH-bug算法，用于调试
        case[1][1] = np.array([8.5, 2])  # test
        self.set_task(case)
        self.init_topic_and_optimizer()
        num = 0
        # max_time = 10000
        start_time = time.time()
        while not self.robots[1].end_run and time.time() - start_time < max_time:
            # self.robots[1].update_own_feature()
            # start_time = time.time()
            v, v_az = self.robots[1].plan()
            # end_time = time.time()
            # print("time used for planning {}".format(end_time-start_time))
            self.robots[1].execute_plan(v, v_az)
            time.sleep(0.3)   # for test
            self.robots[1].execute_plan(0, 0)
            # if num >= 1:
            #     self.robots[1].update_own_feature()
            #     feature = self.robots[1].feature
            #     score = self.robots[1].require_optimizer_output()
            #     print(score)
            num += 1
        print("total journey is {}".format(self.robots[1].journey))

    def stop(self):
        """终止所有机器人的运行"""
        for robot in self.robots:
            robot.execute_plan(0, 0)

    def reset(self):
        for robot in self.robots:
            robot.reset_record()

    def compute_adjacency(self):
        """
        计算邻接矩阵
        :return: 二维numpy数组，维度n*n，n为智能体个数
        """
        adj_matrix = np.zeros([self.num_agents, self.num_agents])
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents, 1):
                if self.robots[i].is_insight_global(self.robots[j].position):
                    adj_matrix[i][j], adj_matrix[j][i] = 1, 1
        return adj_matrix

    def wait_for_connect(self):
        """等待机器人连接"""
        since = time.time()
        while True:
            all_connect = True
            for robot in self.robots:
                if robot.position is None or robot.scan_distance.shape[0] == 0:
                    time.sleep(0.1)
                    all_connect = False
                    print("waiting for connection!")
            if all_connect:
                break
            if time.time() - since > 60 * 3:
                raise Exception("connect to run agent overtime!")

