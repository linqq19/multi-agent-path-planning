#!/usr/bin/env python3
# coding=utf-8
import map
from envAgent import rosAgent #, airsimAgent
import rosUtils
import multiRobot
import numpy as np
import os
import time
from dataGenerator import DataGenerator
from astar import *
import configure

class Master:
    def __init__(self, config):
        self.config = config
        self.num_agents = config.num_agents  # 机器人个数
        # self.map = map.load_npy_map(self.config.map_file_path, config.map_resolution)
        self.robots = multiRobot.Robots(config)

        if config.run_env_type == 0:   # ROS-Gazebo仿真环境
            self.launch = None
            self.map = None
            self.uuid = rosUtils.create_ros_pid()  # 创建ros节点
            # time.sleep(3)
            self.run_agent = rosAgent.ROSAgent(self.robots.robots, self.config)
            self.robots.set_run_agent(self.run_agent)
            self.launch_folder = config.launch_folder
            self.turn_dir_percept = []
            self.turn_dir_adj = []
            self.turn_dirs = []
            self.turn_positions = []
            self.turn_nn_output = []
        else:
            self.run_agent = None

    def reset_record(self):
        self.turn_dir_percept = []
        self.turn_dir_adj = []
        self.turn_dirs = []
        self.turn_positions = []
        self.turn_nn_output = []

    def init_ros_env(self, launch_file_path):
        self.launch = rosUtils.launch_ros_map(self.uuid, launch_file_path)  # 加载当前的launch文件
        npy_file_path = "numpyMapFolder/" + launch_file_path.split("/")[-1].split(".launch")[0].split("robots")[0] + ".npy"
        self.map = map.load_npy_map(npy_file_path, self.config.map_resolution)

    def test_one_case(self, launch_file_path, case, use_opt=True):
        for robot in self.robots.robots:
            robot.planner.use_optimizer = use_opt
        launch = rosUtils.launch_ros_map(self.uuid, launch_file_path)  # 加载当前的launch文件
        time.sleep(len(case)*0.3)
        self.robots.set_task(case)   # 设定任务
        path_length = self.robots.run_dh_bug(max_time=self.config.max_run_time)   # 运行任务
        end_state = np.vstack([robot.position for robot in self.robots.robots])

        result_name = "result/{}_{}_result.png".format(launch_file_path.split("/")[-1].split(".")[0],
                                                       "opt" if use_opt else "pure")

        self.map.paint_paths([robot.path for robot in self.robots.robots], result_name, case)  # 绘制路径

        start_points = np.vstack([robot.path[0] for robot in self.robots.robots])

        rosUtils.shutdown_ros_map(launch)  # 关闭当前地图

        for robot in self.robots.robots:
            robot.reset_record()  # 清空记录
        return path_length, end_state, start_points

    def test_ros(self):
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
            npy_file_path = "numpyMapFolder/" + launch_file_path.split("/")[-1].split(".launch")[0].split("robots")[
                0] + ".npy"
            self.map = map.load_npy_map(npy_file_path, self.config.map_resolution)
            case = self.map.cases_generator(1, self.num_agents, self.config.robotRadius)[0]
            # 使用优化器
            opt_path_length, opt_end_position, start_points = self.test_one_case(launch_file_path,case,use_opt=True)
            # 不使用优化器
            pure_path_length, pure_end_position, start_points = self.test_one_case(launch_file_path,case,use_opt=False)
            goal_points = np.vstack([single[1] for single in case])


            np.savez(result_file_name, opt_path_length=opt_path_length, pure_path_length=pure_path_length,
                     start_points=start_points, goal_points=goal_points,
                     opt_end_position=opt_end_position, pure_end_position=pure_end_position)

    def run_ros(self):
        self.uuid = rosUtils.create_ros_pid()  # 创建ros节点
        launch_file = "map0000robots15case01.launch"
        launch_file_path = self.launch_folder + launch_file
        npy_file_path = "numpyMapFolder/" + launch_file_path.split("/")[-1].split(".launch")[0].split("robots")[0] + ".npy"
        self.map = map.load_npy_map(npy_file_path, self.config.map_resolution)
        # self.init_ros_env(launch_file_path)
        for i in range(1):
            case = self.map.cases_generator(1, self.num_agents, self.config.robotRadius)[0]
            # test
            # data = np.load("map000robots04_test_result.npz")
            # for i in range(self.num_agents):
            #     case[i][1] = data["goal_points"][i]
            # case[2][1] = np.array([16.05, 0.65])
            # test
            # 使用优化器
            opt_path_length, opt_end_position, start_points = self.test_one_case(launch_file_path, case, use_opt=True)
            # 不使用优化器
            pure_path_length, pure_end_position, _ = self.test_one_case(launch_file_path, case,use_opt=False)
            # self.robots.set_task(case)
            # self.robots.run_dh_bug()
            # self.robots.run_dh_single(case)

    def test_one_robot(self, launch_file_path, case, use_opt=True,test_index=0):
        self.reset_record()  # 清除自身记录
        self.robots.set_master(self)
        for robot in self.robots.robots:
            robot.planner.use_optimizer = use_opt
        launch = rosUtils.launch_ros_map(self.uuid, launch_file_path)  # 加载当前的launch文件

        # 等待自身话题订阅完成
        since = time.time()
        while True:
            flag = True
            for robot in self.robots.robots:
                if robot.position is None:
                    flag = False
                    break
            if flag:
                break
            time.sleep(0.1)
            if time.time() - since > 120:
                rosUtils.shutdown_ros_map(launch)  # 关闭当前地图，退出
                return None, None, None

        self.robots.set_task(case)   # 设定任务
        time.sleep(3)
        # 设定只让第0个机器人运行
        for i, robot in enumerate(self.robots.robots):
            if i != 0:
                robot.end_run = True

        path_length = self.robots.run_dh_bug(max_time=self.config.max_run_time)   # 运行任务
        end_state = np.vstack([robot.position for robot in self.robots.robots])

        result_name = "result/{}_{}_test_{}_result.png".format(launch_file_path.split("/")[-1].split(".")[0],
                                                       "opt" if use_opt else "pure", test_index)

        self.map.paint_paths([robot.path for robot in self.robots.robots], result_name, case)  # 绘制路径

        start_points = np.vstack([robot.path[0] for robot in self.robots.robots])
        print("robot 0's path length is {}".format(path_length[0]))

        rosUtils.shutdown_ros_map(launch)  # 关闭当前地图

        for robot in self.robots.robots:
            robot.reset_record()  # 清空记录

        return path_length, end_state, start_points

    def record_decision(self, next_mode):
        all_percept = []
        for robot in self.robots.robots:
            all_percept.append(robot.get_map_info())
        self.turn_dir_percept.append(np.stack(all_percept))
        self.turn_dirs.append(next_mode)
        self.turn_dir_adj.append(self.robots.compute_adjacency())
        self.turn_positions.append(self.robots.robots[0].dcs_pos)
        self.turn_nn_output.append(self.robots.robots[0].nn_output)

    def mannual_test(self):
        self.uuid = rosUtils.create_ros_pid()  # 创建ros节点
        launch_file = "map0000robots01case01.launch"    ###############
        launch_file_path = self.launch_folder + launch_file
        npy_file_path = "numpyMapFolder/" + launch_file_path.split("/")[-1].split(".launch")[0].split("robots")[
            0] + ".npy"
        self.map = map.load_npy_map(npy_file_path, self.config.map_resolution)

        case = self.map.cases_generator(1, self.num_agents, self.config.robotRadius)[0]

        # case[0][1] = np.array([6, 15])  ###############

        # test_goal = np.array([  ############### map0018
        #     [5, 13],
        #     [6, 18.05],
        #     [3, 15.05],
        #     [18, 8.05]
        # ])

        # test_goal = np.array([  ############### map0000
        #     [6, 12],
        #     [8, 17.05],
        #     [4, 12.05],
        #     [14.3, 13.45]
        # ])
        #
        # for i in range(4):
        #     case[i][1] = test_goal[i]

        case[0][1] = np.array([10,4])

        for i in range(5):
            # 使用优化器
            opt_path_length, opt_end_position, start_points = self.test_one_robot(launch_file_path, case, use_opt=True,test_index=i)
            if len(self.turn_dirs) > 0:
                decision_pcpts, decisions = np.stack(self.turn_dir_percept), np.array(self.turn_dirs)
                decision_adj = np.stack(self.turn_dir_adj)
                decision_pos = np.stack(self.turn_positions)
                decision_output = np.stack(self.turn_nn_output)
            else:
                decision_pcpts, decisions, decision_adj, decision_pos, decision_output = [np.array([]) for i in range(5)]
            # 不使用优化器
            # pure_path_length, pure_end_position, _ = self.test_one_robot(launch_file_path, case, use_opt=False)
            pure_path_length, pure_end_position = np.array([]),  np.array([])
            goal_points = np.vstack([single[1] for single in case])
            result_file_name = "result/{}_test_{}_result".format(launch_file.split(".")[0],i)
            np.savez(result_file_name, opt_path_length=opt_path_length, pure_path_length=pure_path_length,
                     start_points=start_points, goal_points=goal_points,
                     opt_end_position=opt_end_position, pure_end_position=pure_end_position,
                     decision_pos=decision_pos, decision_output=decision_output,
                     decision_pcpts=decision_pcpts, decisions=decisions,decision_adj=decision_adj)



if __name__ == "__main__":
    config = configure.get_config()
    master = Master(config)
    master.mannual_test()
