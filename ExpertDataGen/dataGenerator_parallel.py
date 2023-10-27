import time

import scipy.io as scio
# from astar import *
import map
import sys
from utils import *
import multiRobot
from envAgent import rosAgent
import rosUtils
import re
import threading
import configure
# import Astar
import numpy as np
import os
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# 与C语言动态链接库的接口
import ctypes
from numpy.ctypeslib import ndpointer
class point(ctypes.Structure):    # C语言定义的计算类
    _fields_ = [('x',ctypes.c_int),
              ('y',ctypes.c_int),]

class pair(ctypes.Structure):
    _fields_ = [('F',ctypes.c_float),
                ('P',point)]

"""
    地图矩阵与地图坐标系： 以地图的左下角表示地图坐标系的原点，以正前方为y轴正方向，以向右为x轴正方向
"""


def dis_to_score(distance):
    """
    将预估的距离转化为分值
    :param distance: 预估距离
    :return: 打分值
    """
    return distance


# 用于训练数据生成的类
class DataGenerator:
    def __init__(self, config):
        self.config = config
        # self.robots = robots  # 智能体
        self.launch_folder_path = config.launch_folder

        self.num_agents = config.num_agents  # 智能体个数
        self.max_detect = config.max_detect  # 激光雷达最大感知距离,单位：米
        self.com_radius = config.com_radius  # 通信半径
        self.num_directions = config.num_directions  # 预选方向数
        self.num_hops = config.num_hops  # 通信跳数
        self._resolution = config.map_resolution # 有矩阵表示地图时，矩阵相邻两个坐标点之间对应的实际距离
        self._map_size = config.map_size

        self.expert_data_root = "expertData/"
        self.create_save_data_folders()

        self.map_id = 0  # 地图id
        self.case_id = 0  # 生成的样本id

        self.robots = multiRobot.Robots(config)  # 无实际执行代理
        self.run_agent = rosAgent.ROSAgent(self.robots.robots, self.config)
        self.robots.set_run_agent(self.run_agent)
        # self.run_agent = None
        self.global_map = None  # 当前的全局地图

        # for computation:
        self.dis_predict = None
        self.special_points = None

        # A star runner in C language
        self.c_astar = self.load_C_astar_so()

        self.data_modes = ["dense", "adjust"]  # 矫正数据 vs 密集生成的数据

    def create_save_data_folders(self):
        self.pose_correct_dataFolder = self.expert_data_root + "poseCorrect/"
        self.relative_dataFolder = self.expert_data_root + "noCorrect/"
        if not os.path.exists(self.expert_data_root):
            os.mkdir(self.expert_data_root)
        if not os.path.exists(self.pose_correct_dataFolder):
            os.makedirs(self.pose_correct_dataFolder)
        if not os.path.exists(self.relative_dataFolder):
            os.makedirs(self.relative_dataFolder)

    def load_C_astar_so(self):
        c_runner = ctypes.cdll.LoadLibrary("./libastar.so")
        c_runner.getshortest.restype = pair
        c_runner.getshortest.argtypes = [ndpointer(ctypes.c_int),point,point,ctypes.c_float]
        return c_runner

    def generate_ros_files(self, num_files):
        rosUtils.generate_gazebo_world(world_number=num_files,
                                       resolution=self._resolution,
                                       map_shape= int(self._map_size/self._resolution))
        rosUtils.launch_generator("worldFolder/", "numpyMapFolder/", "launchFolder/",
                                  self.config.map_resolution, self.config.num_agents, self.config.robotRadius)

    def reconstruct_launch(self, result_folder, target_folder="reconstruct_folder"):
        # 根据分布式运行结果记录数据重构决策时的lanuch文件
        # 查看目的文件夹是否存在
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)

        # 遍历结果文件
        result_files = []
        for _, _, filenames in os.walk(result_folder):
            for filename in filenames:
                if filename.endswith(".npz"):
                    result_files.append(filename)

        # 转向智能体的信息
        turn_robots_dict = {}

        # 对每个文件进行处理
        num_valid_data = 0
        for result_file in result_files:
            # result_file = "map0095robots15_test_result.npz"
            data = np.load(result_folder + result_file)
            if data["opt_turn_dir_robots"].shape[0] != data["opt_turn_dir_odoms"].shape[0]:
                print("record data abnormal of "+result_file)
                continue
            world_id = int(result_file.split("robots")[0].split("map")[-1])
            turn_robots_dict[str(world_id)] = data["opt_turn_dir_robots"]
            num_valid_data += len(data["opt_turn_dir_robots"])
            world_file = "map{:0>4d}.world".format(world_id)
            for case_idx, robot_id in enumerate(data["opt_turn_dir_robots"]):
                print(result_file)
                num_robots = data["opt_turn_dir_odoms"][case_idx].shape[0]
                case = []
                for i in range(num_robots):
                    case.append([data["opt_turn_dir_odoms"][case_idx][i][0:2],
                                data["goal_points"][i],
                                data["opt_turn_dir_odoms"][case_idx][i][2]])
                rosUtils.generate_given_map_launch(world_map_folder_path="worldFolder/", world_file=world_file, launch_folder_path=target_folder, cases=[case], launch_id=case_idx)
                np.save(target_folder+"/map{:0>4d}case_data{:0>2d}.npy".format(world_id,case_idx),case)
        np.savez(target_folder+"/ros_test_data_info.npz",turn_robots_dict=turn_robots_dict,num_valid_data=num_valid_data)    # 保存转向地图中的相关数据和总的有效数目

    def generate_reconstruct_samples(self, launch_folder="reconstruct_folder/"):
        """根据分布式测试程序的测试数据重新生成专家数据"""
        target_folder = "reconstruct_expert_data/"
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        
        if not os.path.exists(launch_folder):
            return

        uuid = rosUtils.create_ros_pid()  # 创建ros节点
        launch_files = rosUtils.search_target_launch_files(target_folder=launch_folder)  # 找到所有的launch文件
        launch_files.sort()
        for launch_file in launch_files:
            # 判断当前环境下生成的样本数
            # launch_file = "map0017robots15case04.launch"
            map_id = int(launch_file.split("robots")[0].split("map")[-1])  # 更新当前地图id
            case_id = int(launch_file.split(".")[0].split("case")[-1])  # 结果数据中的case——id
            self.map_id, self.case_id = map_id, case_id
            # 加载环境
            print(launch_file)
            num_robots = int(launch_file.split("case")[0].split("robots")[-1])
            if os.path.exists(target_folder+\
                              "map{:0>4d}robots{:0>2d}case{:0>3d}.mat".format(map_id, num_robots, case_id)):
                continue
            launch_file_path = launch_folder + launch_file
            launch = rosUtils.launch_ros_map(uuid, launch_file_path)  # 加载当前的launch文件
            npy_file_path = "numpyMapFolder/" + launch_file.split(".launch")[0].split("robots")[0] + ".npy"
            self.global_map = map.load_npy_map(npy_file_path, self._resolution)

            case = np.load(launch_folder+"map{:0>4d}case_data{:0>2d}.npy".format(map_id, case_id),allow_pickle=True)
            
            self.generate_one_sample_parallel(case,save_folder=target_folder)

            for robot in self.robots.robots:
                robot.reset_record()
            # 关闭环境
            rosUtils.shutdown_ros_map(launch)  # 关闭当前地图

    def generate_samples(self, num_samples_per_map=2):
        """生成任意张地图下的任意个样本"""
        uuid = rosUtils.create_ros_pid()  # 创建ros节点
        launch_files = rosUtils.search_target_launch_files()  # 找到所有的launch文件
        launch_files.sort()
        for launch_file in launch_files:
            # 判断当前环境下生成的样本数
            self.map_id = int(launch_file.split("robots")[0].split("map")[-1])  # 更新当前地图id
            total_relative_cases,  relative_case_id = self.search_case_id(target_folder=self.relative_dataFolder)   # 未矫正数据
            total_pose_correct_cases, posecor_case_id = self.search_case_id(target_folder=self.pose_correct_dataFolder)  # 矫正数据
            if total_relative_cases >= num_samples_per_map and total_pose_correct_cases >=num_samples_per_map:
                continue
            # 加载环境
            print("current map id : {}".format(self.map_id))
            launch_file_path = self.launch_folder_path + launch_file
            launch = rosUtils.launch_ros_map(uuid, launch_file_path)  # 加载当前的launch文件
            npy_file_path = "numpyMapFolder/" + launch_file.split(".launch")[0].split("robots")[0] + ".npy"
            self.global_map = map.load_npy_map(npy_file_path, self._resolution)
            # self.run_agent = rosAgent.ROSAgent(self.robots.robots, self.config)
            # self.robots.set_run_agent(self.run_agent)
            cases = self.global_map.cases_generator(num_samples_per_map,
                                                    self.num_agents, self.config.robotRadius)

            # 生成相对位置数据样本
            run_agent_flag = True  # 默认能连上
            if total_relative_cases < num_samples_per_map:
                self.case_id = relative_case_id
                for case in cases[0:num_samples_per_map-total_relative_cases]:
                    try:
                        self.generate_one_sample_parallel(case, pose_correct=False, save_folder=self.relative_dataFolder)
                    except Exception:
                        run_agent_flag = False
                        print("ros cannot connect")
                        break
                    else:
                        self.case_id += 1
                    # self.robots.reset()
                if run_agent_flag:
                    print("relative samples complete!")

            # 生成矫正后的数据样本
            if total_pose_correct_cases < num_samples_per_map and run_agent_flag:
                self.case_id = posecor_case_id
                for case in cases[0:num_samples_per_map-total_pose_correct_cases]:
                    try:
                        self.generate_one_sample_parallel(case, pose_correct=True, save_folder=self.pose_correct_dataFolder)
                    except Exception:
                        run_agent_flag = False
                        print("ros cannot connect")
                        break
                    else:
                        self.case_id += 1
                    # self.robots.reset()
                if run_agent_flag:
                    print("absolute samples complete!")

            rosUtils.shutdown_ros_map(launch)  # 关闭当前地图
            self.robots.reset()

    def generate_given_map_samples(self, n):
        # 在当前地图下通过不断运行机器人生成n个数据样本
        assert self.global_map is not None, "Np global map found for data Generator"
        total_cases, self.case_id = self.search_case_id()   # 更新当前的case序号
        if total_cases >= n:    # 若该地图下已经包含足够的样本，则停止生成
            return
        cases = self.global_map.cases_generator(n, self.num_agents, self.config.robotRadius)
        for i in range(n):
            self.generate_one_sample(cases[i])

            self.case_id += 1
            self.robots.set_task(cases[i])
            self.robots.run_dh_bug(max_time=90)

    def search_case_id(self, target_folder=None):
        """寻找当前地图下已经生成的case中最大的id 和 case的个数"""
        if target_folder is None:
            target_folder = self.expert_data_root   # 默认路径
        max_id = -1
        total_cases = 0
        for _, _, cases in os.walk(target_folder):
            for case in cases:
                ret = re.match("^map\d{4}robots\d{2}case\d{3}.mat", case)
                if ret:
                    name, _ = case.split(".")
                    map_id = int(name.split("robots")[0].split("map")[-1])
                    num_robots = int(name.split("robots")[-1].split("case")[0])
                    if map_id == self.map_id and num_robots == self.num_agents:
                        case_id = int(name.split("case")[-1])
                        total_cases += 1
                        if case_id > max_id:
                            max_id = case_id
            # break
        return total_cases, max_id+1

    def compute_case_adjacency(self, case):
        """
        计算邻接矩阵
        :return: 二维numpy数组，维度n*n，n为智能体个数
        """
        adj_matrix = np.zeros([self.num_agents, self.num_agents])
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents, 1):
                if np.linalg.norm(case[j][0] - case[i][0], 2) < self.com_radius:
                    adj_matrix[i][j], adj_matrix[j][i] = 1, 1
        return adj_matrix

    def expert_global_dis(self, start_mx, end_mx):
        """
        使用全局信息计算路径长度的专家算法
        :return:
        """
        since = time.time()
        astar_len = self.c_astar.getshortest(self.global_map.safe_map.astype(np.int32),
            point(x=start_mx[0],y=start_mx[1]),point(x=end_mx[0],y=end_mx[1]),1.0)
        # print("C based Astar , begin ({}, {})- end ({},{})- length:{:.2f}, use time : {:.2f}\n".format(start_mx[0],start_mx[1],end_mx[0], end_mx[1],astar_len.F,time.time() - since))

        # a_star = Astar.AStar(self.global_map, start_mx, end_mx, "euclidean")
        # since = time.time()
        # distance = a_star.run()
        # print("python based Astar - length:{:.2f}, use time : {:.2f}".format(distance,time.time() - since))
        if astar_len.F == -1:
            return sys.maxsize/10
        else:
            return astar_len.F

    def straight_dis(self, start_point, end_point):
        """
        智能体由起点前往当前目标点的直线距离
        :param start_point: 智能体所在位置,由地图矩阵的坐标点表示
        :param end_point: 智能体直接前往的位置，由地图矩阵的坐标点表示
        :return: 直线距离 （坐标点定义）
        """
        return np.linalg.norm(np.asarray(end_point) - np.asarray(start_point), 2)

    def generate_one_sample_parallel(self, case, pose_correct=None, save_folder=None):  # 并行生成一个样本
        since = time.time()

        if save_folder is None:
            save_folder = self.expert_data_root


        self.robots.wait_for_connect()

        self.robots.set_task(case)  # 设定机器人执行任务
        self.robots.stop()
        adj_matrix = self.robots.compute_adjacency()

        self.dis_predict = np.ones([self.num_agents, self.num_directions])*sys.maxsize  # 各个智能体在所有方向上的打分信息
        robots_list = self.robots.robots

        all_agents_percept = []  # 各个智能体的三通道感知信息
        self.special_points = [[] for i in range(self.num_agents)]

        jobs = []
        for i in range(self.num_agents):
            perception_i = robots_list[i].get_map_info(post_correct=pose_correct, whole_info=True)
            all_agents_percept.append(perception_i)

            neighbor_idx = np.where(adj_matrix[i] == 1)[0]
            goal_i_mx = self.global_map.matrix_idx(case[i][1][0], case[i][1][1])
            position_i_mx = self.global_map.matrix_idx(robots_list[i].position[0], robots_list[i].position[1])
            self.special_points[i].append(robots_list[i].position)
            # 计算感知范围内各个方向上的预估距离
            for k in range(self.num_directions):  
                p = threading.Thread(target=self.compute_own_dis_full,args=(i,k,goal_i_mx,position_i_mx, pose_correct))
                p.start()
                jobs.append(p)
             # 计算相邻智能体中辅助感知位置的预估距离
            for j in neighbor_idx: 
                for k in range(self.num_directions):
                    p = threading.Thread(target=self.compute_neighbor_dis_full,args=(i,j,k,goal_i_mx,position_i_mx,pose_correct))
                    p.start()
                    jobs.append(p)

        for job in jobs:
            job.join()

        task = [[cur_robot.position, case[i][1], cur_robot.yaw] for i, cur_robot in enumerate(robots_list)]
        laser_raw_distance_info = [robot.scan_distance  for robot in self.robots.robots]
        laser_raw_angles_info = self.robots.robots[0].scan_angles
        # 保存结果文件
        file_path = save_folder + \
                    "map{:0>4d}robots{:0>2d}case{:0>3d}.mat".format(self.map_id, adj_matrix.shape[0], self.case_id)
        scio.savemat(file_path, {'global_map': self.global_map.map, 'case': task,
                                 'adj_matrix': adj_matrix, 'percept_info': all_agents_percept,
                                 'score': self.dis_predict, 'raw_laser_angles': laser_raw_angles_info,
                                 'raw_laser_dis': laser_raw_distance_info})
        # self.save_sample(self.global_map.map, task, adj_matrix, all_agents_percept, self.dis_predict, save_folder)

        # 保存中间计算点数据文件
        self.global_map.paint_paths(self.special_points,save_folder+"map{:0>4d}robots{:0>2d}case{:0>3d}data.png".format(self.map_id, self.num_agents, self.case_id) ,case)
        print(" complete all computation for sample {}, use time : {:.2f}".format(self.case_id, time.time() - since))
        self.robots.reset()

    def compute_own_dis(self,i,k,goal_i_mx,position_i_mx,pose_correct):
        # 计算第i个智能体自身的第k个方向上从矩阵索引position_i_mx到矩阵索引goal_i_mx的专家距离
        robot = self.robots.robots[i]
        if pose_correct:
            dir = angle_transform(360 / self.num_directions * k + 30)  # 固定方向
        else:
            dir = angle_transform(robot.yaw + 360 / self.num_directions * k + 30)  # 全局坐标系中机器人的参考方向对应的角度

        # dir = angle_transform(robot.yaw + 360 / self.num_directions * k + 30)  # 全局坐标系中机器人的参考方向对应的角度
        _, line_seg = intersect_circle_ray(robot.position, dir,
                                            robot.position, self.com_radius)  # 全局坐标系中的方向线段
        reach_point = self.global_map.find_point(line_seg)  # 最远能走到且无障碍的地图矩阵下标
        if reach_point is not None:
            self.special_points[i].append(self.global_map.abs_cor(reach_point[0], reach_point[1]))
            dis = self.expert_global_dis(reach_point, goal_i_mx) + \
                                self.straight_dis(position_i_mx, reach_point)
            if dis < self.dis_predict[i,k]:
                self.dis_predict[i,k] = dis

    def compute_own_dis_full(self,i,k,goal_i_mx,position_i_mx,pose_correct, block=True):
        # 计算第i个智能体自身的第k个方向上从矩阵索引position_i_mx到矩阵索引goal_i_mx的专家距离, 但计算过程中同一方向上计算多个点，以提升邻居的观测效果
        robot = self.robots.robots[i]
        if pose_correct:
            dir = angle_transform(360 / self.num_directions * k + 30)  # 固定方向
        else:
            dir = angle_transform(robot.yaw + 360 / self.num_directions * k + 30)  # 全局坐标系中机器人的参考方向对应的角度

        # dir = angle_transform(robot.yaw + 360 / self.num_directions * k + 30)  # 全局坐标系中机器人的参考方向对应的角度
        _, line_seg = intersect_circle_ray(robot.position, dir,
                                            robot.position, self.com_radius)  # 全局坐标系中的方向线段
        reach_point = self.global_map.find_point(line_seg)  # 最远能走到且无障碍的地图矩阵下
        if reach_point is not None:
            mid_points = self.global_map.find_midpoint(position_i_mx, reach_point, block=block)  # 中间点 用于增加计算信息
            mid_points.append(reach_point)
            for cpu_point in mid_points:
                dis = self.expert_global_dis(cpu_point, goal_i_mx) + \
                                    self.straight_dis(position_i_mx, cpu_point)
                if dis < self.dis_predict[i,k]:
                    self.dis_predict[i,k] = dis
                if dis == sys.maxsize/10:
                    print("robots {} direction {} from ({}, {}) to ({}, {}) cannot be computed" \
                          .format(i, k, cpu_point[0], cpu_point[1], goal_i_mx[0], goal_i_mx[1]) )

                self.special_points[i].append(self.global_map.abs_cor(cpu_point[0], cpu_point[1]))


    def compute_neighbor_dis_full(self,i,j,k,goal_i_mx,position_i_mx,pose_correct, block=True):
        # 计算位于position_i_mx的第i个智能体的第k个方向上与邻居j的交点处到矩阵索引goal_i_mx的专家距离,
        robot = self.robots.robots[i]
        neighbor = self.robots.robots[j]

        if pose_correct:
            dir = angle_transform(360 / self.num_directions * k + 30)  # 固定方向
        else:
            dir = angle_transform(robot.yaw + 360 / self.num_directions * k + 30)  # 全局坐标系中机器人的参考方向对应的角度

        # dir = angle_transform(robot.yaw + 360 / self.num_directions * k +30)
        has_inter, line_seg = intersect_circle_ray(robot.position, dir,
                                                    neighbor.position, self.com_radius)
        if has_inter:  # 若存在交点，通过交点计算估计距离
            reach_point = self.global_map.find_point(line_seg)
            if reach_point is not None:
                mid_points = self.global_map.find_midpoint(position_i_mx, reach_point, block=block)  # 中间点 用于增加计算信息
                mid_points.append(reach_point)
                for cpu_point in mid_points:
                    dis = self.expert_global_dis(cpu_point, goal_i_mx) + \
                          self.straight_dis(position_i_mx, cpu_point)
                    if dis < self.dis_predict[i, k]:
                        self.dis_predict[i, k] = dis
                    self.special_points[i].append(self.global_map.abs_cor(cpu_point[0], cpu_point[1]))


    def compute_neighbor_dis(self,i,j,k,goal_i_mx,position_i_mx, pose_correct):
        # 计算位于position_i_mx的第i个智能体的第k个方向上与邻居的交点处到矩阵索引goal_i_mx的专家距离
        robot = self.robots.robots[i]
        neighbor = self.robots.robots[j]

        if pose_correct:
            dir = angle_transform(360 / self.num_directions * k + 30)  # 固定方向
        else:
            dir = angle_transform(robot.yaw + 360 / self.num_directions * k + 30)  # 全局坐标系中机器人的参考方向对应的角度

        # dir = angle_transform(robot.yaw + 360 / self.num_directions * k +30)
        has_inter, line_seg = intersect_circle_ray(robot.position, dir,
                                                    neighbor.position, self.com_radius)
        if has_inter:  # 若存在交点，通过交点计算估计距离
            reach_point = self.global_map.find_point(line_seg)
            if reach_point is not None:
                self.special_points[i].append(self.global_map.abs_cor(reach_point[0], reach_point[1]))
                dis = self.expert_global_dis(reach_point, goal_i_mx) + \
                        self.straight_dis(position_i_mx, reach_point)
                if dis < self.dis_predict[i,k]:
                    self.dis_predict[i,k] = dis


    def generate_one_sample(self, case):  # 串行生成一个样本
        """
        在给定的地图下，生成一个数据样本
        :return:
        """
        self.robots.set_task(case)  # 设定机器人执行任务
        self.robots.stop()
        self.robots.wait_for_connect()
        adj_matrix = self.robots.compute_adjacency()
        robots_list = self.robots.robots

        all_agents_percept = []  # 各个智能体的三通道感知信息
        all_agents_dis = []  # 各个智能体在所有方向上的打分信息
        special_points = [[] for i in range(self.num_agents)]
        for i in range(self.num_agents):
            start_time = time.time()
            perception_i = robots_list[i].get_map_info()
            all_agents_percept.append(perception_i)

            neighbor_idx = np.where(adj_matrix[i] == 1)[0]
            dis_predict = np.ones(self.num_directions) * sys.maxsize  # 初始化各个评估方向上的预测距离
            goal_i_mx = self.global_map.matrix_idx(case[i][1][0], case[i][1][1])
            position_i_mx = self.global_map.matrix_idx(robots_list[i].position[0], robots_list[i].position[1])
            special_points[i].append(robots_list[i].position)
            for k in range(self.num_directions):  # 计算感知范围内各个方向上的预估距离
                dir = angle_transform(robots_list[i].yaw + 360 / self.num_directions * k + 30)  # 全局坐标系中机器人的参考方向对应的角度
                _, line_seg = intersect_circle_ray(robots_list[i].position, dir,
                                                   robots_list[i].position, self.com_radius)  # 全局坐标系中的方向线段
                reach_point = self.global_map.find_point(line_seg)  # 最远能走到且无障碍的地图矩阵下标
                special_points[i].append(self.global_map.abs_cor(reach_point[0],reach_point[1]))
                if reach_point is not None:
                    dis_predict[k] = self.expert_global_dis(reach_point, goal_i_mx) + \
                                     self.straight_dis(position_i_mx, reach_point)
            print("----complete agent {i}'s own computation! use time: {t} seconds---".format(i=i, t=time.time()-start_time))
            for j in neighbor_idx:  # 计算相邻智能体中辅助感知位置的预估距离
                for k in range(self.num_directions):
                    dir = angle_transform(robots_list[i].yaw + 360 / self.num_directions * k +30)
                    has_inter, line_seg = intersect_circle_ray(robots_list[i].position, dir,
                                                               robots_list[j].position, self.com_radius)
                    if has_inter:  # 若存在交点，通过交点计算估计距离
                        reach_point = self.global_map.find_point(line_seg)
                        special_points[i].append(self.global_map.abs_cor(reach_point[0], reach_point[1]))
                        if reach_point is not None:
                            dis = self.expert_global_dis(reach_point, goal_i_mx) + \
                                  self.straight_dis(position_i_mx, reach_point)
                            if dis < dis_predict[k]:
                                dis_predict[k] = dis
                    else:  # 若无交点，计算下一个位置
                        continue
            print("----------complete agent {i}'s others computation! ---------------------".format(i=i))

            all_agents_dis.append(dis_predict)

        all_agents_score = dis_to_score(all_agents_dis)
        task = [[cur_robot.position, case[i][1], cur_robot.yaw] for i, cur_robot in enumerate(robots_list)]
        self.save_sample(self.global_map.map, task, adj_matrix, all_agents_percept, all_agents_score)
        # self.global_map.paint_paths(special_points,"expertData/map_{}_data.png".format(self.map_id) ,case)
        print(" complete all computation for sample {}".format(self.case_id))
        self.robots.reset()


    def save_sample(self, cur_map, cur_case, adj_matrix, sample_x, sample_y,save_folder=None):
        """
        :param cur_map: 当前地图
        :param cur_case: 当前任务
        :param adj_matrix: 邻接矩阵
        :param sample_x:  样本点 数据
        :param sample_y: 样本点标签
        :param file_name:  样本点 数据
        :return: 是否保存成功
        """
        if save_folder is None:
            save_folder = self.expert_data_root
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        file_path = save_folder + \
                    "map{:0>4d}robots{:0>2d}case{:0>3d}.mat".format(self.map_id, adj_matrix.shape[0], self.case_id)
        scio.savemat(file_path, {'global_map': cur_map, 'case': cur_case,
                                 'adj_matrix': adj_matrix, 'percept_info': sample_x, 'score': sample_y})
        return True

    def run(self):
        case = self.global_map.cases_generator(1, self.num_agents, self.config.robotRadius)
        self.generate_one_sample(case[0])


if __name__ == "__main__":
    config = configure.get_config()
    config.use_optimizer = 0
    data_gen = DataGenerator(config)

    # 产生新的地图文件
    # data_gen.generate_ros_files(1050)
    # 每张地图生成1个数据样本文件
    data_gen.generate_samples(num_samples_per_map=4)

    # 重构ros中测试时决策时刻的launch文件
    # data_gen.reconstruct_launch("result/lower_n20/")

    # 根据决策时刻的文件生成专家数据
    # data_gen.generate_reconstruct_samples()
