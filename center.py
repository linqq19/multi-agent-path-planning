#!/usr/bin/env python
#coding=utf-8

import math
import map
import rosUtils
# import robot
import numpy as np
import os
import time
from dataGenerator import DataGenerator
from astar import *
import rospy
from gpkg.msg import Task, RobotState
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
import configure
from sensor_msgs.msg import LaserScan
import scipy.io as sio



class Center:
    def __init__(self, config):
        self.config = config
        self.num_agents = config.num_agents  # 机器人个数
        ## self.robots = robot.Robots(config)

        if config.run_env_type == 0:   # ROS-Gazebo仿真环境
            self.launch = None
            self.map = None
            self.uuid = rosUtils.create_ros_pid()  # 创建ros节点
            ## self.run_agent = rosAgent.ROSAgent(self.robots.robots, self.config)
            ## self.robots.set_run_agent(self.run_agent)
            self.launch_folder = config.launch_folder
            self.task_pubs = []
            self.robot_paths = [[] for i in range(self.num_agents)]
            self.task_receive = [False for i in range(self.num_agents)]
            self.task_execute = [False for i in range(self.num_agents)]
            self.end_run = [True for i in range(self.num_agents)]
            self.path_lengths = [0 for i in range(self.num_agents)]
            self.create_center_node()  # 分布式版本，创建中心节点
            self.run_state = 0 # 所有机器人的任务执行状态

            # 决策时刻相关数据记录
            self.turn_dir_state = 0 # 记录机器人转向的状态
            self.turn_dir_agent = -1 # 记录转向的机器人ID
            self.robots_record_state = np.zeros([2,self.num_agents])  # 是否对当前地图下的所有机器人的当前状态进行了记录， 第一行为雷达数据，第二行为里程计数据
            self.laser_angles = None    # 雷达的各个角度
            self.num_laser_data = None
            self.nn_outputs = []
            # self.laser_rho_group = None
            # self.robots_record_state = np.zeros(self.num_agents)  # 是否对当前地图下的所有机器人的当前状态进行了记录， 第一行为雷达数据，第二行为里程计数据
            self.odom_group = None
            self.laser_group = None
            # self.turn_dir_laser_rhos = []
            self.turn_dir_odoms = []
            self.turn_dir_robots = []
            self.turn_dirs = []
            self.turn_dir_lasers = []
            self.odom_record_flag = False
            self.laser_record_flag = False
            self.max_detect = config.max_detect
            # ------------------------------------------
            # 用于专家数据生成
            self.use_expert = True  # 是否使用专家数据进行对比
            self.map_id = -1
            # ------------------------------------------
        else:
            self.run_agent = None

    def create_center_node(self):
        """创建分布式版本的总控节点"""
        # 所有机器人的任务发布话题
        for i in range(self.num_agents):
            task_pub = rospy.Publisher("robot{}_task".format(i+1), Task, queue_size=10)
            self.task_pubs.append(task_pub)
        rospy.init_node("center", anonymous=True)
        # 所有机器人的位置接收话题,用于绘制轨迹
        for i in range(self.num_agents):
            rospy.Subscriber("/car{}/odom".format(i+1), Odometry, self.odom_callback, i, queue_size=1)
            rospy.Subscriber("robot{}_state".format(i+1), RobotState, self.robostate_callback,i, queue_size=1)
            rospy.Subscriber("/car{}/scan".format(i+1), LaserScan, self.laser_callback, i, queue_size=1)

    def laser_callback(self,data,robot_id):
        # 接收机器人的scan话题，用于记录机器人决策时的感知数据
        self.laser_angles = np.array([data.angle_min + i * data.angle_increment for i in range(len(data.ranges))])\
                           * 180 / math.pi  # 弧度转化为角度
        self.num_laser_data  = len(self.laser_angles)
        if self.run_state != 1:
            return
        # 处理转向时的数据记录
        if self.turn_dir_state != 0:
            if self.robots_record_state[1, robot_id] == 0:  # 若当前机器人的位置尚未记录
                scan_dist = np.array(data.ranges)
                scan_dist[scan_dist > self.max_detect] = self.max_detect
                self.laser_group[robot_id,:] = scan_dist
                self.robots_record_state[1, robot_id] = 1
            if np.min(self.robots_record_state[1,:]) == 1:  # 若所有机器人的位置都记录了
                if not self.laser_record_flag:
                    self.turn_dir_lasers.append(self.laser_group)
                    self.laser_record_flag = True
                if self.odom_record_flag and self.laser_record_flag:
                    self.turn_dir_state = 0

    def odom_callback(self,data,robot_id):
        # 接受机器人的odom话题，用于记录机器人轨迹
        if self.run_state !=1:
            return
        # 记录机器人位置
        posi = data.pose.pose.position
        if len(self.robot_paths[robot_id])==0 :  # 初始位置记录
            self.robot_paths[robot_id].append((posi.x,posi.y))
        else:
            last_posi = self.robot_paths[robot_id][-1]
            if np.linalg.norm(np.array(last_posi)-np.array([posi.x,posi.y]),2)>0.02:  # 记录与山之前位置变化较大的值
                self.robot_paths[robot_id].append((posi.x, posi.y))
        # 处理转向时的数据记录
        if self.turn_dir_state != 0:
            if self.robots_record_state[0,robot_id] == 0:  # 若当前机器人的位置尚未记录
                direc = data.pose.pose.orientation
                _, _, yaw = euler_from_quaternion([direc.x, direc.y, direc.z, direc.w])
                self.odom_group[robot_id,:] = [posi.x, posi.y, yaw * 180 / math.pi]
                self.robots_record_state[0,robot_id] = 1
            if np.min(self.robots_record_state[0,:]) == 1:  # 若所有机器人的位置都记录了
                if not self.odom_record_flag:
                    self.turn_dir_odoms.append(self.odom_group)
                    self.odom_record_flag = True
                if self.laser_record_flag and self.odom_record_flag:  # 若同时所有机器人的感知数据也被记录下来了
                    self.turn_dir_state = 0


    # def conpute_expert_data(self, pose_correct):
    #     assert self.odom_group == np.zeros([self.num_agents, 3]), "all robots positions and yaw are not get in the test center"
    #     all_robots_position = self.odom_group  # 所有机器人的位置和姿态
    #     adj = compute_adj()
    #     pass

    # def compute_adj_matrix(positions):
    #     """计算一个n*2维度数组对应的邻接矩阵"""
    #     assert positions.shape[1] == 2, " the inputs of all robots position are not standard ndarrays!"
    #     assert positions.shape[0] != 0, "can not compute adj matrix for None type !"
    #     n = positions.shape[1]
    #     adj_matrix = np.zeros([n, n])
    #     for i in range(n):
    #         for j in range(i + 1, n, 1):
    #             if self.robots[i].is_insight_global(self.robots[j].position):
    #                 adj_matrix[i][j], adj_matrix[j][i] = 1, 1
    #     return adj_matrix


    def robostate_callback(self, data, robot_id):
        # 接收机器人发布的状态话题，用于了解各个机器人的运行状态
        if data.turn_dir == 0:
            self.task_receive[robot_id] = data.task_receive # 是否收到任务
            self.task_execute[robot_id] = data.task_execute  # 是否开始执行任务
            self.end_run[robot_id] = data.end_run # 是否处于任务运行之中
            if data.end_run and data.task_execute:
                self.path_lengths[robot_id] = data.journey # 路程
        elif abs(data.turn_dir) == 1:   # 机器人转向的数据处理s
            while self.turn_dir_state != 0:   # 等待上次数据记录完成
                time.sleep(0.1)
            self.robots_record_state = np.zeros([2, self.num_agents])
            self.odom_group = np.zeros([self.num_agents,3])
            self.laser_group = np.zeros([self.num_agents, self.num_laser_data])
            self.turn_dir_state = data.turn_dir
            self.turn_dir_agent = robot_id
            self.turn_dir_robots.append(self.turn_dir_agent)   # 转向id组
            self.turn_dirs.append(self.turn_dir_state)  # 转向方向组
            self.odom_record_flag = False
            self.laser_record_flag = False
            self.nn_outputs.append(data.nn_output)


    def init_ros_env(self, launch_file_path):
        # 加载launch文件和npy文件
        self.launch = rosUtils.launch_ros_map(self.uuid, launch_file_path)  # 加载当前的launch文件
        npy_file_path = "numpyMapFolder/" + launch_file_path.split("/")[-1].split(".launch")[0].split("robots")[0] + ".npy"
        self.map = map.load_npy_map(npy_file_path, self.config.map_resolution)

    def pub_task(self,case,use_opt, start_run=False,fire=True):
        """ 向所有机器人发布运行指令：
        case: 根据地图随机生成的任务 
        use_opt: 使用优化器的类型 0 - 固定左转 1 - 使用gnn优化器 2 - 使用专家数据生成方案
        start_run: 开始执行, True/False
        fire: 启动或停止 
        """
        for i in range(self.num_agents):
            msg = Task()
            if fire:  # 若已启动
                if case is not None:
                    msg.has_goal = 1
                    msg.goal = case[i][1] # 任务
                else:
                    msg.has_goal = 0
                msg.use_opt = use_opt # 使用优化器
                msg.map_id = self.map_id       ##########################################
                msg.start_run = start_run # 启动
            msg.fire = fire
            self.task_pubs[i].publish(msg)

    def reset_map_related_record(self):
        # 重置与单张地图相关的测试信息
        self.map_id = -1
        self.map = None

    def reset_task_related_record(self):
        """重置机器人相关记录"""
        self.path_lengths = [0 for _ in range(self.num_agents)]
        self.robot_paths = [[] for _ in range(self.num_agents)]
        self.task_receive = [False for i in range(self.num_agents)]
        self.task_execute =  [False for i in range(self.num_agents)]
        self.end_run = [True for i in range(self.num_agents)]
        # 与转向的记录相关
        self.turn_dir_state = 0  # 记录机器人转向的状态
        self.turn_dir_agent = -1  # 记录转向的机器人ID
        self.robots_record_state = np.zeros([2,self.num_agents])  # 是否对当前地图下的所有机器人的当前状态进行了记录， 第一行为雷达数据，第二行为里程计数据
        self.odom_group = None
        self.turn_dir_odoms = []
        self.turn_dir_robots = []
        self.turn_dirs = []
        self.laser_group = None
        self.turn_dir_lasers = []
        self.odom_record_flag = False
        self.laser_record_flag = False
        self.nn_outputs = []

    def test_one_case_d(self, launch_file_path, case, map_id, use_opt=1, max_time=360, load_map_time=10):
        """分布式环境下测试指定文件"""
        self.reset_task_related_record()
        # 加载测试地图
        launch = rosUtils.launch_ros_map(self.uuid, launch_file_path)  # 加载当前的launch文件
        time.sleep(len(case)*0.3)

        # 发布任务直至所有的机器人接收到任务
        self.pub_task(case, use_opt, start_run=False, fire=True)  # 发布任务
        time.sleep(load_map_time)  # 等待
        while not all(self.task_receive):  # 对应robot的has_goal
            self.pub_task(case, use_opt, start_run=False, fire=True)  # 发布任务
            time.sleep(load_map_time)

        # 开始执行
        self.run_state = 1
        # 等待自身话题订阅完成
        since = time.time()
        while min([len(path) for path in self.robot_paths]) == 0:
            time.sleep(0.1)
            if time.time() - since > 120:
                rosUtils.shutdown_ros_map(launch)  # 关闭当前地图，退出
                return None, None, None

        # 发布开始执行指令
        self.pub_task(case,use_opt,start_run=True, fire=True) # 发布任务完成，开始执行
        while not all(self.task_execute):
            self.pub_task(case, use_opt, start_run=True, fire=True)  # 发布任务完成，开始执行
            time.sleep(0.5)

        # 等待所有机器人执行完成
        while not all(self.end_run): # 对应robot的end_run
            time.sleep(0.1)
        print("all robots have accomplish their goal {}==================".format(np.sum(np.array(self.end_run))))
        self.run_state = 0
        self.pub_task(None, use_opt, start_run=False, fire=True)  # 停止运行
        path_lengths = self.path_lengths
        while self.turn_dir_state != 0:   # 等待最后临近结束时的数据记录完成
            time.sleep(0.1)
        case_name = ["pure", "opt", "expert_absolute","expert_relative"]
        result_name = "result/{}_{}_result.png".format(launch_file_path.split("/")[-1].split(".")[0],
                                                       case_name[use_opt])

        self.map.paint_paths(self.robot_paths, result_name, case)  # 绘制路径
        
        start_points = np.vstack([path[0] for path in self.robot_paths])
        end_state = np.vstack([path[-1] for path in self.robot_paths])
        rosUtils.shutdown_ros_map(launch)  # 关闭当前地图
        time.sleep(10)  # 测试能否解决问题 sigtem
        return path_lengths, end_state, start_points

    def find_valid_case(self):
        """寻找robot 0决策时存在周边邻居的场景， 需将config中的 test_one_robot 选项设置为 True"""
        assert self.config.run_env_type == 0, "not ros environment, please check your settings"
        dirs = "turnHasNghbCase/"
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        launch_files = rosUtils.search_target_launch_files()  # 找到所有的launch文件
        launch_files.sort()

        for launch_file in launch_files:
            # 相同测试环境
            case_file_name = dirs + "{}_valid_case".format(launch_file.split("case")[0])
            if os.path.exists(case_file_name + ".npz"):
                continue

            num = 0 # 尝试获取能进行转向的次数
            while num < 5:
                launch_file_path = self.launch_folder + launch_file
                print("current map id : {}".format(launch_file.split("robots")[0].split("map")[1]))
                self.map_id = int(launch_file.split("robots")[0].split("map")[1])  #################################
                npy_file_path = "numpyMapFolder/" + launch_file_path.split("/")[-1].split(".launch")[0].split("robots")[
                    0] + ".npy"
                self.map = map.load_npy_map(npy_file_path, self.config.map_resolution)
                case = self.map.cases_generator(1, self.num_agents, self.config.robotRadius)[0]

                # 使用GNN优化
                opt_path_length, opt_end_position, start_points = self.test_one_case_d(launch_file_path, case, self.map_id,
                                                                                       use_opt=1)
                if len(self.turn_dir_robots) != 0:
                    np.save(case_file_name, case)
                    break

                self.reset_map_related_record()
                num += 1
        self.pub_task(None, 0, False, fire=False)  # 停止机器人的运行


    def test_expert(self):
        """测试各种专家方案是否有效， 在此模式下需将config中的test_one_robot选项设置为true"""
        assert self.config.run_env_type == 0, "not ros environment, please check your settings"
        dirs = "expert_result"  # 专家测试方案的默认文件夹
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        launch_files = rosUtils.search_target_launch_files()  # 找到所有的launch文件
        launch_files.sort()

        for launch_file in launch_files:
            # 相同测试环境
            result_file_name = "result/{}_expert_result".format(launch_file.split(".")[0])
            case_file_name =  "{}_valid_case".format(launch_file.split("case")[0])+".npy"
            if os.path.exists(result_file_name + ".npz"):
                continue
            if not os.path.exists("turnHasNghbCase/"+ case_file_name):
                continue

            launch_file_path = self.launch_folder + launch_file
            print("current map id : {}".format(launch_file.split("robots")[0].split("map")[1]))
            self.map_id = int(launch_file.split("robots")[0].split("map")[1])  #################################
            npy_file_path = "numpyMapFolder/" + launch_file_path.split("/")[-1].split(".launch")[0].split("robots")[
                0] + ".npy"
            self.map = map.load_npy_map(npy_file_path, self.config.map_resolution)

            case = np.load("turnHasNghbCase/"+ case_file_name, allow_pickle=True)

            # # for test ===========================================================================
            # print("load expert case !!!!")     # load case that has turn point from expert data ( temporarily used)
            # exper_file = launch_file.split("case")[0] + "case001.mat"
            # case_load = sio.loadmat("expertData/" + exper_file)["case"]
            # case = [[data[0].flatten(), data[1].flatten(), data[2].flatten().item()] for data in case_load]
            # # for test ===========================================================================

            # 不使用优化器
            pure_path_length, pure_end_position, start_points = self.test_one_case_d(launch_file_path, case, self.map_id,
                                                                          use_opt=0)
            if pure_path_length is None:
                continue

            # 使用专家数据进行导航， 进行位姿矫正 expert-absolute , 专家数据生成方式： 六个方向- 直接拓展延伸
            epabs_path_length, epabs_end_position, _ = self.test_one_case_d(launch_file_path, case, self.map_id,
                                                                            use_opt=2)
            # 使用专家数据进行导航， 不进行位姿矫正  expert - relative ，  专家数据生成方式： 六个方向- 直接拓展延伸
            eprel_path_length, eprel_end_position, _ = self.test_one_case_d(launch_file_path, case, self.map_id,
                                                                            use_opt=3)

            goal_points = np.vstack([single[1] for single in case])

            np.savez(result_file_name,
                     start_points=start_points, goal_points=goal_points,
                     pure_path_length=pure_path_length,   pure_end_position=pure_end_position,
                     epabs_path_length=epabs_path_length, epabs_end_position=epabs_end_position,
                     eprel_path_length=eprel_path_length, eprel_end_position=eprel_end_position)
            self.reset_map_related_record()
        self.pub_task(None, 0, False, fire=False)  # 停止机器人的运行


    def test_ros (self):
        """在ros环境下，由已知地图生成case并运行dh-bug算法"""
        assert self.config.run_env_type == 0, "not ros environment, please check your settings"
        dirs = "result"
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        launch_files = rosUtils.search_target_launch_files()  # 找到所有的launch文件
        launch_files.sort()

        for launch_file in launch_files:
            # 相同测试环境
            result_file_name = "result/{}_test_result".format(launch_file.split(".")[0])
            if os.path.exists(result_file_name+".npz"):
                continue

            launch_file_path = self.launch_folder + launch_file
            print("current map id : {}".format(launch_file.split("robots")[0].split("map")[1]))
            self.map_id = int(launch_file.split("robots")[0].split("map")[1])                  #################################
            npy_file_path = "numpyMapFolder/" + launch_file_path.split("/")[-1].split(".launch")[0].split("robots")[
                0] + ".npy"
            self.map = map.load_npy_map(npy_file_path, self.config.map_resolution)
            case = self.map.cases_generator(1, self.num_agents, self.config.robotRadius)[0]

            # for test ===========================================================================
            if self.config.test_only_one:  # 仅测试一个智能体
                while np.linalg.norm(case[0][0]-case[0][1],2) < 7:
                    case = self.map.cases_generator(1, self.num_agents, self.config.robotRadius)[0]
            # for test ===========================================================================

            # for test ===========================================================================
            print("load expert case !!!!")
            exper_file = launch_file.split("case")[0] + "case001.mat"
            case_load = sio.loadmat("expertData/" + exper_file)["case"]
            case = [[data[0].flatten(), data[1].flatten(), data[2].flatten().item()] for data in case_load]
            # for test ===========================================================================

            # 使用GNN优化
            opt_path_length, opt_end_position, start_points = self.test_one_case_d(launch_file_path,case,self.map_id,use_opt=1)
            if opt_path_length is None:
                 continue
            if len(self.turn_dir_robots) != 0:
                opt_turn_dir_robots, opt_turn_dirs, opt_turn_dir_odoms = np.array(self.turn_dir_robots), \
                                                                         np.array(self.turn_dirs), \
                                                                         np.stack(self.turn_dir_odoms)
                opt_turn_dir_lasers = np.stack(self.turn_dir_lasers)
                opt_nn_outputs = np.stack(self.nn_outputs)
            else:
                opt_turn_dir_robots , opt_turn_dirs , opt_turn_dir_odoms , opt_turn_dir_lasers,\
                    opt_nn_outputs = [np.array([]) for i in range(5)]

            # 不使用优化器
            pure_path_length, pure_end_position, _ = self.test_one_case_d(launch_file_path,case,self.map_id,use_opt=0)
            if pure_path_length is None:
                continue

            goal_points = np.vstack([single[1] for single in case])
   
            if not self.config.use_expert:
                np.savez(result_file_name, opt_path_length=opt_path_length, pure_path_length=pure_path_length,
                        start_points=start_points, goal_points=goal_points,
                        opt_end_position=opt_end_position, pure_end_position=pure_end_position,
                        opt_turn_dir_robots=opt_turn_dir_robots, opt_turn_dirs=opt_turn_dirs,
                        opt_turn_dir_odoms=opt_turn_dir_odoms,
                        opt_turn_dir_lasers= opt_turn_dir_lasers,
                        laser_angles = self.laser_angles,
                        opt_nn_outputs = opt_nn_outputs)
            else:
                # 使用专家数据进行导航， 进行位姿矫正 expert-absolute
                epabs_path_length, epabs_end_position , _ = self.test_one_case_d(launch_file_path,case,self.map_id,use_opt=2)
                # 使用专家数据进行导航， 不进行位姿矫正  expert - relative
                eprel_path_length, eprel_end_position, _ = self.test_one_case_d(launch_file_path,case,self.map_id,use_opt=3)
                np.savez(result_file_name, opt_path_length=opt_path_length, pure_path_length=pure_path_length,
                        start_points=start_points, goal_points=goal_points,
                        opt_end_position=opt_end_position, pure_end_position=pure_end_position,
                        opt_turn_dir_robots=opt_turn_dir_robots, opt_turn_dirs=opt_turn_dirs,
                        opt_turn_dir_odoms=opt_turn_dir_odoms,
                        opt_turn_dir_lasers= opt_turn_dir_lasers,
                        laser_angles = self.laser_angles,
                        opt_nn_outputs = opt_nn_outputs,
                        epabs_path_length = epabs_path_length,epabs_end_position = epabs_end_position,
                        eprel_path_length = eprel_path_length, eprel_end_position = eprel_end_position)

            self.reset_map_related_record()

        self.pub_task(None, 0, False, fire=False)  # 停止机器人的运行

 


if __name__ == "__main__":
    config = configure.get_config()
    test_center = Center(config)
    test_center.find_valid_case()
    # test_center.test_ros()
    # test_center.manual_test()
    # test_center.test_expert()


