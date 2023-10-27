#!/usr/bin/env python
# coding=utf-8

# 原先激光雷达的点数为320000个，每秒旋转20次，但这样跑4辆车十分卡顿
# 为使仿真更加流畅，暂时将取点数设为128000个，每秒旋转10次，看一下效果

# import AirsimCarCtrl
# from AirsimCarCtrl import CarCtrl
import rospy
# from geometry_msgs.msg import Twist, Point, Quaternion  # 从 geometry_nsg 中导入Twist消息类型
# from sensor_msgs.msg import LaserScan
# from nav_msgs.msg import Odometry
import math
import cmath
import time
import numpy as np
import multiprocessing
from envAgent import agent

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
# import matplotlib.pyplot as plt
from airsim_ros_pkgs.msg import CarControls
from airsim_ros_pkgs.msg import CarState
# from geometry_msgs.msg import Pose, PoseWithCovariance, TwistWithCovariance
import airsim


# 多线程解决rospy.spin阻塞程序执行的问题
def thread_job():
    rospy.spin()


class AirsimAgent(agent.EnvAgent):
    def __init__(self, robots, config):
        super(AirsimAgent, self).__init__(robots)
        self.max_detect = config.max_detect
        self.com_radius = config.com_radius
        self.MAX_LIN_VEL = 1.0  # 宏定义最大油门
        self.MAX_ANG_VEL = 1.0  # 宏定义最大转角
        self.FOV = config.max_detect  # 60  # 机器人视野半径，单位为m，暂时为int型
        self.HCAR = -1.4  # 车辆高度，LIDAR默认安装在坐标为-1.5m的位置（实测数据为-1.48左右）
        self.ctrl_pubs = []
        self.velocity = np.zeros(len(self.robots), dtype=int)
        self.spin_process = None
        try:
            rospy.init_node('airsim_agent', anonymous=True)

            for i in range(len(self.robots)):
                control_pub = rospy.Publisher('/airsim_node/car{id}/car_cmd'.format(id=i+1), CarControls, queue_size=10)

                self.ctrl_pubs.append(control_pub)
                rospy.Subscriber('/airsim_node/car{id}/lidar/LidarCustom'.format(id=i+1), PointCloud2, self.laser_callback, i, queue_size=1)
                rospy.Subscriber('/airsim_node/car{id}/car_state'.format(id=i+1), CarState, self.odom_callback, i, queue_size=1)

        except rospy.ROSInterruptException:
            print('AirsimAgent Failed')
            pass
        time.sleep(2)

    def start_spin(self):
        self.spin_process = multiprocessing.Process(target=rospy.spin)
        self.spin_process.start()
        time.sleep(2)

    def end_spin(self):
        # self.shutdown()
        self.spin_process.terminate()
        print("Process terminated!")
        self.spin_process = None

    def constrain(self, input, low, high):
        '''
        将输入数据限制在low～high范围内
        :param input: 输入数据
        :param low: 最小值
        :param high: 最大值
        :return: 限幅后的数
        '''
        if input < low:
            input = low
        elif input > high:
            input = high
        return input

    def checkLinearLimitVelocity(self, vel):
        '''
        使用constrain函数对车辆速度进行限幅
        :param vel: 输入速度
        :return: 限幅后的速度
        '''
        vel = self.constrain(vel, -self.MAX_LIN_VEL, self.MAX_LIN_VEL)
        return vel

    def checkAngularLimitVelocity(self, vel):
        '''
        使用constrain函数对车辆转角进行限幅
        :param vel: 输入转角
        :return: 限幅后的转角
        '''
        vel = self.constrain(vel, -self.MAX_ANG_VEL, self.MAX_ANG_VEL)
        return vel

    def orientationToRPY(self, x, y, z, w):
        '''
        将四元数表示的方位转化为欧拉坐标系中的rpy转角
        由于地面均为平坦，因此可以只读取y角(偏航角)
        取值范围：R -180～180， P -90～90， Y -180～180
        :return:仅返回偏航角，汽车位于初始化状态时角度为0，逆时针为正方向
        '''
        # r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        # p = math.asin(2 * (w * y - z * z))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))

        # angleR = r * 180 / math.pi
        # angleP = p * 180 / math.pi
        angleY = -y * 180 / math.pi  # 将逆时针设为正方向

        return angleY

    def control(self, robot_id, velocity, angular_velocity):
        '''
        通过ros发布话题控制小车行动，刹车优先
        :param robot_id: 小车id
        :param velocity: 小车速度
        :param angular_velocity: 小车角加速度
        '''

        car_commands = CarControls()
        car_commands.gear_immediate = True  # 立即上档

        throttle = 0
        steering = 0
        brake = 0
        if velocity > self.velocity[robot_id]:  # 速度未达标，给油门
            throttle = (self.velocity[robot_id] - velocity) / 10.0  # 以差值最大为10m/s给油门
            brake = 0
        elif velocity == self.velocity[robot_id]:
            pass
        else:
            brake = (self.velocity[robot_id] - velocity) / 20.0  # 以差值最大为20m/s给刹车
            throttle = 0

        # 逆时针为正值，角度制
        steering = - angular_velocity / 30  # 以差值最大为30度/s给方向盘，但实际参数肯定和想象的不一致，需要调试
        if abs(steering)> 0.01 and abs(self.velocity[robot_id]) < 0.01:
            throttle = 0.1
        throttle = self.checkLinearLimitVelocity(throttle)  # 油门和转角限幅
        brake = self.checkLinearLimitVelocity(brake)
        steering = self.checkLinearLimitVelocity(steering)

        if brake == 1.0:  # 急停刹车
            car_commands.handbrake = False
            car_commands.brake = 1

        else:  # 前进档
            car_commands.handbrake = False
            car_commands.brake = brake
            car_commands.manual = False
            car_commands.manual_gear = 0  # 不使用手动档，所以默认为0
            car_commands.throttle = throttle
            car_commands.steering = steering

        self.ctrl_pubs[robot_id].publish(car_commands)

    # 雷达call_back函数
    def laser_callback(self, data, agent_id):
        assert isinstance(data, PointCloud2)  # 判定接受数据与PointCloud2数据是否相同
        gen = point_cloud2.read_points(data)  # 读点云数据
        angles = np.zeros(360)  # 储存每个角度对应的距离，精度为1度,-179~180
        dists = np.zeros(360)
        for p in gen:
            tmp = complex(p[0], p[1])  # 创建一个复数来进行坐标转换
            d, a = cmath.polar(tmp)  # 计算长度和角度
            a = round(a * 180 / math.pi)  # 角度制，-179～180
            if a == -180: a = 180
            if -p[2] > self.HCAR:  # 如果障碍物过高则不可通行  # LIDAR读出的z坐标正方向向下，因此取相反数
                if dists[a + 179] > d or dists[a + 179] == 0:  # 取最小值
                    angles[a + 179] = a
                    dists[a + 179] = d
            else: # 此处必须加以限制，否则会出现激光雷达干扰产生的同心圆
                if dists[a + 179] == 0:
                    angles[a + 179] = a
                    dists[a + 179] = self.FOV  # 无障碍物，设置为最大FOV
        idx = np.argwhere(dists == 0)  # 找出激光雷达没看到的位置
        for i in idx:
            angles[i] = i - 179
            dists[i] = self.FOV
        dists[dists > self.max_detect] = self.max_detect
        self.robots[agent_id].scan_angles = angles
        self.robots[agent_id].scan_distance = dists
        self.robots[agent_id].update_own_feature()

    # 获取机器人的姿态
    def odom_callback(self, data, agent_id):
        x = data.pose.pose.orientation.x
        y = data.pose.pose.orientation.y
        z = data.pose.pose.orientation.z
        w = data.pose.pose.orientation.w

        self.robots[agent_id].position = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        self.robots[agent_id].yaw = self.orientationToRPY(x, y, z, w)

        vx = data.twist.twist.linear.x
        vy = data.twist.twist.linear.y
        self.velocity[agent_id] = math.sqrt(vx*vx + vy*vy)

        self.simu_com(agent_id)   # 将绝对位置转化为相对位置从而模拟机器人之间的相对位置变化与更新

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
        car_commands = CarControls()
        car_commands.gear_immediate = True  # 立即上档
        car_commands.handbrake = True
        car_commands.brake = 1
        car_commands.manual = False
        car_commands.manual_gear = 0  # 不使用手动档，所以默认为0
        car_commands.throttle = 0
        car_commands.steering = 0
        for ctrl_pub in self.ctrl_pubs:
            ctrl_pub.publish(car_commands)
        rospy.sleep(1)


if __name__ == "__main__":
    robots = []


