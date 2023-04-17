#!/usr/bin/env python3
# coding=utf-8

from easydict import EasyDict
from center import Center
from master import Master
from dataGenerator import DataGenerator
import configure


def main():
    config = configure.get_config()
    # ROS-Gazebo环境下运行DH-bug算法
    config.alg_mode = 2
    if config.alg_mode == 0 and config.run_env_type == 0:  # 分布式执行
        master = Center(config)
        master.test_ros()
    # ROS-Gazebo环境下生成训练数据
    elif config.alg_mode == 1 and config.run_env_type == 0: # 生成数据
        data_gen = DataGenerator(config)
        # data_gen.generate_ros_files(1)  # ![](result/map027robots04_opt_result.png)
        data_gen.generate_samples()  # 每张地图生成1个数据样本文件
    elif config.alg_mode == 2 and config.run_env_type == 0: # 生成地图
        data_gen = DataGenerator(config)
        data_gen.generate_ros_files(999)
    elif config.alg_mode == 3 and  config.run_env_type == 0: # 集中执行
        master = Master(config)
        master.test_ros()

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    main()
