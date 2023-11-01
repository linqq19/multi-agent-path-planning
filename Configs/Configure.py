from easydict import EasyDict

def get_config():
    config = EasyDict({'run_env_type': 0,  # 运行环境类型, 0为ROS环境，1为AirSim
                       'alg_mode': 1,  # 0 与网络无关， 1 数据生成模式,  2 ros 地图生成模式
                       'use_optimizer': 1,  # 使用的优化器的种类： 0 : 无优化器， 1：GNN优化 2: 专家数据-矫正 3 专家数据-不矫正
                       "max_run_time": 360,   # 最长运行时间，用于测试时确定时长
                        "pose_correction": True, # 是否进行姿势矫正
                       "use_expert": True, # 分布式测试中是否使用专家数据生成模式直接进行测试
                       "dim_nn_output" : 6,
                       "test_only_one" : True, #  是否只测试第一个机器人，其他机器人不动, center.py
                        "relative_nn_path": "./GNN/relative_robots15_rm0_min_valid_loss_1n4_none.pkl", # GNN加载路径， 相对
                       "absolute_nn_path": "./GNN/absolute_min_valid_loss_cnn_10epoch__dataset_remove_0neighbor_0schedulerclassification.pkl", # GNN加载路径， 绝对
                        # "nn_path": "./GNN/robots15_rm0_min_valid_loss_1n4_none.pkl",  # 训练好的神经网络的存储路径
                       'num_agents': 15,  # 智能体个数
                       'max_detect': 5,  # 激光雷达传感器最大感知范围
                       'map_size': 20,  # 正方形地图尺寸
                       # 'map_scale': 0.03,  # gazebo地图比例尺
                       'num_directions': 6,  # 预选方向个数
                       'num_hops': 2,  # 通信跳数
                       'robotRadius': 0.15,  # 机器人半径
                       'laser_num': 360,  # 激光传感器扫过一圈测得的角度个数
                       'sector_safe_width': 60,  # 扇面安全宽度，以角度计算
                       'rot_speed': 20,  # 原地旋转角速度, 单位度每秒
                       'dis_threthold': 0.6,  # 安全阈值，即以做大最度前进时，刹车停下走过的最短距离
                       'agent_decel': 2,  # 机器人的加速度
                       'constant_speed': 0.5,  # 机器人的最大行进速度 dhbug算法中用于计算速度的v0值
                       'k_rot': 2,  # 运动机器人的角度修正参量
                       'min_detect': 0.15,  # 激光雷达最小阈值
                       "dim_feature": 128,  # 神经网络输出向量维度
                       "launch_folder": "launchFolder/"  # ros工程配置

                       })
    config.com_radius = config.max_detect  # 假设条件： 通信半径等于感知半径
    config.map_resolution = config.max_detect / 100  # 地图比例尺

    config.pose_correction = False    ###############################################################
    config.dim_nn_output = 2          ###############################################################

    return config
