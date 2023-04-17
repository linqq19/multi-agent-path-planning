import numpy as np
import sys
from utils import *


# 安全扇面结构
class SafeSector:
    def __init__(self):
        self.midDirection = None  # 扇面中线角度，与载体坐标系相同，逆时针-90度到90度，正前方为0度
        self.goal = None  # 目标角度
        self.goalErr = sys.maxsize  # 与目标方向的角度偏差

    # 计算误差角度
    def compute_err(self):
        self.goalErr = self.midDirection - self.goal

    # 通过扇面中线角度改变扇面
    def set_mid(self, orient):
        self.midDirection = orient
        self.compute_err()


# 基于DH-BUG的避障行为
class DhPathPlanner:
    def __init__(self, robot, config):
        # 初始化机器人相关参数
        self.target_robot = robot
        self._decel = robot.decel
        self._radius = robot.radius
        self._constant_speed = robot.constant_speed
        self._k_rot = config.k_rot
        self._min_detect = config.min_detect
        self.use_opt = self.target_robot.use_optimizer
        # 机器人状态
        self.goal_dis = sys.maxsize   # 目标的距离
        self.goal_angle = sys.maxsize  # 目标的角度
        # 读取config参数
        self._max_detect = config.max_detect  # 传感器最大检测距离
        self._rot_speed = config.rot_speed  # 原地旋转角速度
        self._sector_safe_width = config.sector_safe_width  # 扇面安全宽度，以角度计算
        self._dis_threthold = config.dis_threthold   # 激光雷达的测量阈值
        self._speed_dist = 1.5  # 计算速度时用于限制的距离
        self._protect_radius = self._radius * 2
        # 初始化获得的激光传感器距离信息
        self.laser_angles, self.laser_row = [], []
        self._laser_num = 360
        self.safe_sector_width_R = math.ceil(self._sector_safe_width / (360 / self._laser_num))  # 安全扇面所需的角度数，转化为激光传感器角度分辨率的个数
        self._avg_angle_speed = 10  # 平均转动角速度
        # 初始化DH-bug 算法参数
        self.act_mode = 0  # 机器人的避障模式
        # 0为避障模式，-1为右转模式，-2为左沿墙模式，1为左转模式，2为右沿墙模式，－3为左转反向模式（由左沿墙转为右沿墙），3为右转反向模式（由右沿墙转为左沿墙），4为朝向目标模式（暂态模式，朝向目标后即开始避障)，-4为躲避运动障碍物模式，5为慎思式解锁模式，-5为等待模式
        self.minD = sys.maxsize/10  # 到目标的最短距离

        self.get_robot_state()

    def turn_dir(self):
        """
        返回绕樯运动模式，其中1为左转后右绕墙， -1为右转左绕墙
        :return:
        """
        if self.use_opt == 0:  # 固定左转右绕障
            return 1
        else:
            left_better = self.target_robot.get_turn_dir(use_opt = self.use_opt)
            return 1 if left_better else -1

    def find_safe_angles(self):
        # 寻找全部的安全角度
        safe_angles = []
        idx = 0
        for angle in self.laser_angles:
            if self.check_safe(angle):
                safe_angles.append(idx)
                idx += 1
        return safe_angles

    def get_mode(self):
        return self.act_mode

    def check_safe(self, angle):
        """
        判断所选方向是否能安全前进，可安全前进区域为矩形+扇面
        :param angle: 所选方向, 机器人坐标系
        :return: 是否安全
        """
        index = self.find_angle_index(self.laser_angles, angle)
        if self.laser_row[index] > self._dis_threthold:
            return True
        else:
            return False

        # for i in range(len(self.laserRow)):
        #     delta = abs(angle_transform(self.laserAngle[i] - angle))
        #     if delta < self._sector_safe_width/2 and self.laserRow[i] < self._dis_threthold:
        #         return False
        #     if self._sector_safe_width/2 < delta < 90 and \
        #             self.laserRow[i] < self._radius / abs(math.cos((90 - delta)/180*math.pi)) - self._radius:
        #         return False
        # return True

    def check_safe_index(self, index):
        if self.laser_row[index] > self._dis_threthold:
            return True
        else:
            return False

    def find_safe_angle_intervals(self):
        # 先粗搜，后细搜
        dist_sign = np.sign(self.laser_row - self._dis_threthold)
        intervals = []
        start = None
        scan = False
        for i, sign in enumerate(dist_sign):
            if not scan:
                if sign == 1:
                    start = i
                    scan = True
                else:
                    continue
            else:
                if sign != 1 or i == len(dist_sign)-1:
                    end = i
                    scan = False
                    if end - start > self.safe_sector_width_R+10:
                        intervals.append([start + int(self.safe_sector_width_R / 2) + 5, end - int(self.safe_sector_width_R / 2) - 5])
                else:
                    continue
        return intervals

    def find_safe_run_index(self):
        dist_sign = np.sign(self.laser_row - self._dis_threthold)
        intervals = []
        start = None
        scan = False
        for i, sign in enumerate(dist_sign):
            if not scan:
                if sign == 1:
                    start = i
                    scan = True
                else:
                    continue
            else:
                if sign != 1 or i == len(dist_sign) - 1:
                    end = i
                    scan = False
                    if end - start > self.safe_sector_width_R + 10:
                        intervals.append([start + int(self.safe_sector_width_R / 2) + 5,
                                          end - int(self.safe_sector_width_R / 2) - 5])
                else:
                    continue
        return intervals


    # # 查找与目标角度最小的激光传感器的扫描角度对应值
    # def find_angle_index(self, angle):
    #     min_angle = sys.maxsize
    #     index = 0
    #     for i in range(len(self.laserAngle)):
    #         if abs(self.laserAngle[i] - angle) < min_angle:
    #             min_angle = abs(self.laserAngle[i] - angle)
    #             index = i
    #     return index

    # 获取机器人状态
    def get_robot_state(self):
        raw_laser_angles, raw_laser_row = self.target_robot.get_laser_info()
        if len(raw_laser_angles) != 0:
            index90n = self.find_angle_index(raw_laser_angles, -90)
            index90p = self.find_angle_index(raw_laser_angles, 90)
            self.laser_row = raw_laser_row[index90n: index90p + 1]
            self.laser_angles = raw_laser_angles[index90n:  index90p + 1]
        # self.laserRow = self.laserRow - (self._radius + 0) * np.ones(self.laserRow.shape)  # 激光传感器获取的相对于机器人坐标的障碍物距离值, 考虑安全半径

    def update_goal(self):
        self.target_robot.check_goal()
        self.goal_dis = self.target_robot.goal_dis
        self.goal_angle = self.target_robot.goal_angle

    def find_angle_index(self, angles, target_angle):
        err = abs(angles - target_angle)
        index = np.where(err == np.min(err))[0][0]
        return index

    # 避障速度计算
    def speed(self, delta):
        if abs(delta) > 90:
            return 0
        min_vmax = sys.maxsize  # 求各个障碍物方向下允许的最大速度的最小值,
        # minTheta = delta  # 允许的最大速度的最小值对应的方向
        # 第一步：计算给定方向上允许的最大线速度
        # 当给定方向theta上的障碍物距离d_theta大于安全距离d_s时，v_max(theta) = sqrt（2*a*(d_theta-d_s)）
        # 否则，v_max(theta) = v0 * (d_theta/d_s)^2
        for i in range(len(self.laser_angles)):
            dtheta = angle_transform(self.laser_angles[i] - delta)
            if abs(dtheta) < self._sector_safe_width / 2:
                if self.laser_row[i] > self._protect_radius:
                    v = math.sqrt(2 * self._decel * (self.laser_row[i] - self._protect_radius)) / math.cos(
                        abs(dtheta) * math.pi / 180)
                # elif self.laserRow[i] > self._radius:
                #     v = self._constant_speed * math.sqrt((self.laserRow[i]) / self._dis_threthold) / math.cos(
                #         abs(dtheta) * math.pi / 180)
                else:
                    v = 0
                if v < min_vmax:
                    min_vmax = v

        # 第二步，加入角度变化因子
        v = min_vmax * pow(math.cos(delta * math.pi / 180), 2)
        return v

    def reset(self):
        self.act_mode = 0
        self.laser_row = []
        self.laser_angles = []

    # 避障行为的控制函数
    def run(self):
        self.get_robot_state()  # 更新机器人当前位置、朝向角、速度及激光传感器检测到的障碍物信息

        if not len(self.laser_row):  # 若尚未获取到感知信息，则停止不动
            print("-------robot {}  hasn't connected to Agent successfully-----------".format(self.target_robot.id))
            return 0, 0

        self.update_goal()  # 机器人的目标位置和距离

        # print("------robot  {id}'s goal------{gx},{gy}".format(id = self.target_robot.id, gx=self.target_robot.goal[0], gy=self.target_robot.goal[1]))
        # print("####  ####  current run mode : {mode} #### ####".format(mode=self.act_mode))
        # print("current position : {}, current angle: {}".format(self.target_robot.position, self.target_robot.yaw))
        delta_heading = 360  # 需计算出的机器人旋转角度
        plan_speed = 0  # 路径规划算法输出的速度
        plan_angular_speed = 0  # 路径规划算法输出的角速度
        opt_sector = SafeSector()  # 当前周围环境下的最优扇面
        opt_sector.goal = self.goal_angle

        goal_index = self.find_angle_index(self.laser_angles, self.goal_angle)
        if self.goal_dis < min(self.laser_row[goal_index], 0.7):  # 目标位于感知范围内，且已到达目标点0.1米范围内
            if abs(self.goal_angle) < 10:  # 正对目标
                print("Robot {}  has reach it's destination successfully !！!".format(self.target_robot.id))
                self.target_robot.reach = True
                self.target_robot.end_run = True
                self.act_mode = 0
                return 0, 0
            else:  # 尚未正对目标
                self.act_mode = 4

        # print("current neighbors : {}".format(self.target_robot.check_neighbor()))
        # print("neural network's output: {}".format(self.target_robot.require_optimizer_output()))
        # print("suggest mode {}".format(self.turn_dir()))

        # 避障模式
        if self.act_mode == 0:
            if abs(self.goal_angle) >= 90:
                plan_speed = 0
                plan_angular_speed = self._k_rot * self.goal_angle
                self.act_mode = 4
            else:
                intervals = self.find_safe_angle_intervals()
                forward_safe_angle = 360
                for interval in intervals:
                    angle_start = self.laser_angles[interval[0]]
                    angle_end = self.laser_angles[interval[1]]
                    if angle_start <= self.goal_angle <= angle_end:
                        opt_sector.set_mid(self.goal_angle)
                    if abs(angle_transform(angle_start-self.goal_angle)) < abs(opt_sector.goalErr):
                        opt_sector.set_mid(angle_start)
                    if abs(angle_transform(angle_end-self.goal_angle)) < abs(opt_sector.goalErr):
                        opt_sector.set_mid(angle_end)
                    if angle_start <= 0 <= angle_end:
                        forward_safe_angle = 0
                    if abs(angle_start) < abs(forward_safe_angle):
                        forward_safe_angle = angle_start
                    if abs(angle_end) < abs(forward_safe_angle):
                        forward_safe_angle = angle_end

                if abs(opt_sector.goalErr) < 90:  # 能够保证到目标距离减小，而且在安全扇面内
                    delta_heading = opt_sector.midDirection  # 旋转角度
                    plan_speed = self.speed(delta_heading)
                elif abs(forward_safe_angle) < 90:     # 否则直奔目标运动，直到找不到可行通道
                    delta_heading = forward_safe_angle
                    plan_speed = self.speed(delta_heading)
                else:  # 最大程度逼近障碍物且前方无可行通道，开始转向绕障
                    self.minD = self.goal_dis
                    self.act_mode = self.turn_dir()
                    self.target_robot.dispose_turn_point(self.act_mode)    # 测试中心记录模态切换到转向时的相关数据                    delta_heading = 0
                    plan_speed = 0
                    print("change to mode {mode}".format(mode=self.act_mode))

                plan_angular_speed = self._k_rot * delta_heading

        # 右转模式
        elif self.act_mode == -1:
            intervals = self.find_safe_angle_intervals()
            plan_speed = 0
            if len(intervals) > 0:
                self.act_mode = -2
                # delta_heading = self.laserAngle[intervals[-1][1]]  # 右转至存在安全扇形区域
                delta_heading = 1 / 3 * self.laser_angles[intervals[-1][0]] + 2 / 3 * self.laser_angles[intervals[-1][1]]
                plan_angular_speed = self._k_rot * delta_heading
            else:  # 右转
                plan_angular_speed = -1 * self._rot_speed

        # 左转模式
        elif self.act_mode == 1:
            intervals = self.find_safe_angle_intervals()  # 找出一个可行的区域
            plan_speed = 0
            if len(intervals) != 0:
                self.act_mode = 2
                # delta_heading = self.laserAngle[intervals[0][0]]  # 最右边扇形的起点
                delta_heading = 2 / 3 * self.laser_angles[intervals[0][0]] + 1 / 3 * self.laser_angles[intervals[0][1]]
                plan_angular_speed = self._k_rot * delta_heading
            else:
                plan_angular_speed = self._rot_speed

        # 左沿墙模式
        elif self.act_mode == -2:
            intervals = self.find_safe_angle_intervals()
            if len(intervals) == 0:  # 找不到符合宽度的扇面
                plan_speed = 0
                plan_angular_speed = -1 * self._rot_speed
                self.act_mode = -1
            else:
                # delta_heading = self.laserAngle[intervals[-1][1]]
                delta_heading = 1 / 3 * self.laser_angles[intervals[-1][0]] + 2 / 3 * self.laser_angles[intervals[-1][1]]
                plan_speed = self.speed(delta_heading)
                plan_angular_speed = self._k_rot * delta_heading

                # 计算比例因子，判断是否在M线上
                if self.goal_dis < self.minD:
                    self.minD = self.goal_dis
                    if -170 < self.goal_angle < -5:  # # 如果目标方向与沿墙壁方向相反，停止沿墙行走，开始避障行为
                        self.act_mode = 4  # 停止沿墙行走，转入朝向目标模式后转入避障行为

        elif self.act_mode == 2:  # 右沿墙模式
            intervals = self.find_safe_angle_intervals()    # 从右侧开始扫，扫描全部扇形

            if len(intervals) == 0:  # 找不到符合宽度的扇面
                plan_speed = 0
                plan_angular_speed = self._rot_speed
                self.act_mode = 1
            else:
                # delta_heading = self.laserAngle[intervals[0][0]]
                delta_heading = 2 / 3 * self.laser_angles[intervals[0][0]] + 1 / 3 * self.laser_angles[intervals[0][1]]
                plan_speed = self.speed(delta_heading)
                plan_angular_speed = self._k_rot * delta_heading

                # 计算比例因子，判断是否在M线上
                if self.goal_dis < self.minD:
                    self.minD = self.goal_dis
                    if 5 < self.goal_angle < 170:  # # 如果目标方向与沿墙壁方向相反，停止沿墙行走，开始避障行为
                        self.act_mode = 4  # 停止沿墙行走，转入朝向目标模式后转入避障行为

        # 朝向目标（暂态模式，朝向目标后即开始避障
        elif self.act_mode == 4:
            if abs(self.goal_angle) > 10:
                plan_speed = 0
                plan_angular_speed = np.sign(self.goal_angle) * min(self._k_rot * abs(self.goal_angle), self._rot_speed)
            else:
                plan_speed = 0
                plan_angular_speed = 0
                self.act_mode = 0

        plan_speed = min(plan_speed, 0.5 * self.goal_dis)  # 接近终点时减速
        plan_speed = min(plan_speed, self._constant_speed)    # 不超过最大限速
        if abs(plan_angular_speed/self._k_rot) > 10:  # 限制转弯大小
            plan_speed = 0

        return plan_speed, plan_angular_speed




# 在DH-bug简单实现的基础上，加入H点轨迹记录以完成， 需要用到自身里程计的信息来判断是否以同一姿态经过相同位置。
class DhPathPlanner_HPoint(DhPathPlanner):
    def __init__(self, robot, config):
        super(DhPathPlanner_HPoint,self).__init__(robot, config)

        self.position = None
        self.yaw = None
        self.bypass_points = []  # 绕障时的经过的点
        self.bypass_angles = [] # 对应绕障经过点的方向

        # 解决原地转圈问题：
        self.rotate_angle = 0
        self.last_yaw = 181

    def update_odom(self):
        self.position = self.target_robot.position
        self.yaw = self.target_robot.yaw

    def reset_bypass(self):
        self.bypass_points = []  # 绕障时的经过的点
        self.bypass_angles = [] # 对应绕障经过点的方向

    def reset(self):
        super(DhPathPlanner_HPoint, self).reset()
        self.position = None
        self.yaw = None
        self.bypass_points = []  # 绕障时的经过的点
        self.bypass_angles = [] # 对应绕障经过点的方向

        # 解决原地转圈问题：
        self.rotate_angle = 0
        self.last_yaw = 181

    # 避障行为的控制函数
    def run(self):
        self.get_robot_state()  # 更新机器人当前位置、朝向角、速度及激光传感器检测到的障碍物信息

        if not len(self.laser_row):  # 若尚未获取到感知信息，则停止不动
            print("-------robot hasn't connected to Agent successfully-----------")
            return 0, 0

        self.update_goal()  # 机器人的目标位置和距离

        self.update_odom()   # 根据机器人自身的里程计获取信息

        # print("------robot  {id}'s goal------{gx},{gy}".format(id = self.target_robot.id, gx=self.target_robot.goal[0], gy=self.target_robot.goal[1]))
        # print("####  ####  current run mode : {mode} #### ####".format(mode=self.act_mode))
        # print("current position : {}, current angle: {}".format(self.target_robot.position, self.target_robot.yaw))
        delta_heading = 360  # 需计算出的机器人旋转角度
        plan_speed = 0  # 路径规划算法输出的速度
        plan_angular_speed = 0  # 路径规划算法输出的角速度
        opt_sector = SafeSector()  # 当前周围环境下的最优扇面
        opt_sector.goal = self.goal_angle

        goal_index = self.find_angle_index(self.laser_angles, self.goal_angle)
        if self.goal_dis < min(self.laser_row[goal_index], 0.7):  # 目标位于感知范围内，且已到达目标点0.1米范围内
            if abs(self.goal_angle) < 10:  # 正对目标
                print("Robot {} has reach it's destination successfully !！!".format(self.target_robot.id))
                self.target_robot.reach = True
                self.target_robot.end_run = True
                self.act_mode = 0
                return 0, 0
            else:  # 尚未正对目标
                self.act_mode = 4

        # print("current neighbors : {}".format(self.target_robot.check_neighbor()))
        # print("neural network's output: {}".format(self.target_robot.require_optimizer_output()))
        # print("suggest mode {}".format(self.turn_dir()))


        # 周围0.6m范围内是否有更高优先级且在执行任务的邻居存在，若存在，则静止
        # for nghb in self.target_robot.neighborhood:
        #     if nghb.connect and nghb.sender_id < self.target_robot.id and nghb.rel_dist < 0.6 and nghb.task_execute:    # 存在问题，
        #         print("robot {} in position {} is making way for superior workmate {}".format(self.target_robot.id, self.target_robot.position,nghb.sender_id))
        #         return 0, 0

        if abs(self.act_mode) == 2:   # 对动态障碍物导致的绕圈现象进行处理
            dis_his = np.linalg.norm(np.stack(self.bypass_points)-self.position,2,axis=1)
            if dis_his.shape[0] > 1:
                min_dis = np.min(dis_his[0:-1])
                min_index = np.argmin(dis_his[0:-1])
                if min_dis < 0.05 and abs(angle_transform(self.bypass_angles[min_index] - self.yaw)) < 20:   # 位置接近，方向接近
                    print("minimum dis of robot {} is {}".format(self.target_robot.id, dis_his))
                    np.savez("decision_record.npz",bypass_points=np.stack(self.bypass_points),position=self.position)
                    # print("route points : {}, position: {}".format(np.stack(self.bypass_points), self.position))
                    self.act_mode = 4
                    self.reset_bypass()
                    return 0, 0
                else:
                    if np.linalg.norm(self.bypass_points[-1] - self.position, 2) > 0.07:
                        self.bypass_points.append(self.position)
                        self.bypass_angles.append(self.yaw)
            else:
                if np.linalg.norm(self.bypass_points[-1] - self.position, 2) > 0.07:   # 初始记录的点
                    self.bypass_points.append(self.position)
                    self.bypass_angles.append(self.yaw)

        # 避障模式
        if self.act_mode == 0:
            if abs(self.goal_angle) >= 90:
                plan_speed = 0
                plan_angular_speed = self._k_rot * self.goal_angle
                self.act_mode = 4
            else:
                intervals = self.find_safe_angle_intervals()
                forward_safe_angle = 360
                for interval in intervals:
                    angle_start = self.laser_angles[interval[0]]
                    angle_end = self.laser_angles[interval[1]]
                    if angle_start <= self.goal_angle <= angle_end:
                        opt_sector.set_mid(self.goal_angle)
                    if abs(angle_transform(angle_start-self.goal_angle)) < abs(opt_sector.goalErr):
                        opt_sector.set_mid(angle_start)
                    if abs(angle_transform(angle_end-self.goal_angle)) < abs(opt_sector.goalErr):
                        opt_sector.set_mid(angle_end)
                    if angle_start <= 0 <= angle_end:
                        forward_safe_angle = 0
                    if abs(angle_start) < abs(forward_safe_angle):
                        forward_safe_angle = angle_start
                    if abs(angle_end) < abs(forward_safe_angle):
                        forward_safe_angle = angle_end

                if abs(opt_sector.goalErr) < 90:  # 能够保证到目标距离减小，而且在安全扇面内
                    delta_heading = opt_sector.midDirection  # 旋转角度
                    plan_speed = self.speed(delta_heading)
                elif abs(forward_safe_angle) < 90:     # 否则直奔目标运动，直到找不到可行通道
                    delta_heading = forward_safe_angle
                    plan_speed = self.speed(delta_heading)
                else:  # 最大程度逼近障碍物且前方无可行通道，开始转向绕障
                    self.minD = self.goal_dis
                    self.act_mode = 5
                    return 0, 0   # 为解决可能存在的时延问题
                    # self.act_mode = self.turn_dir()
                    self.bypass_points.append(self.position)  # ======== 记录绕障点 =========
                    self.bypass_angles.append(self.yaw)
                    self.target_robot.dispose_turn_point(self.act_mode)    # 测试中心记录模态切换到转向时的相关数据
                    delta_heading = 0
                    plan_speed = 0
                    print("robot —{} — change to mode {}， cur position : {}".format(self.target_robot.id, self.act_mode, self.position))

                plan_angular_speed = self._k_rot * delta_heading

        # 右转模式
        elif self.act_mode == -1:
            intervals = self.find_safe_angle_intervals()
            plan_speed = 0
            if len(intervals) > 0:
                self.act_mode = -2
                # delta_heading = self.laserAngle[intervals[-1][1]]  # 右转至存在安全扇形区域
                delta_heading = 1 / 3 * self.laser_angles[intervals[-1][0]] + 2 / 3 * self.laser_angles[intervals[-1][1]]
                plan_angular_speed = self._k_rot * delta_heading
            else:  # 右转
                plan_angular_speed = -1 * self._rot_speed

        # 左转模式
        elif self.act_mode == 1:
            intervals = self.find_safe_angle_intervals()  # 找出一个可行的区域
            plan_speed = 0
            if len(intervals) != 0:
                self.act_mode = 2
                # delta_heading = self.laserAngle[intervals[0][0]]  # 最右边扇形的起点
                delta_heading = 2 / 3 * self.laser_angles[intervals[0][0]] + 1 / 3 * self.laser_angles[intervals[0][1]]
                plan_angular_speed = self._k_rot * delta_heading
            else:
                plan_angular_speed = self._rot_speed

        # 左沿墙模式
        elif self.act_mode == -2:
            intervals = self.find_safe_angle_intervals()
            if len(intervals) == 0:  # 找不到符合宽度的扇面
                plan_speed = 0
                plan_angular_speed = -1 * self._rot_speed
                self.act_mode = -1
            else:
                # delta_heading = self.laserAngle[intervals[-1][1]]
                delta_heading = 1 / 3 * self.laser_angles[intervals[-1][0]] + 2 / 3 * self.laser_angles[intervals[-1][1]]
                plan_speed = self.speed(delta_heading)
                plan_angular_speed = self._k_rot * delta_heading

                # 计算比例因子，判断是否在M线上
                if self.goal_dis < self.minD:
                    self.minD = self.goal_dis
                    if -170 < self.goal_angle < -5:  # # 如果目标方向与沿墙壁方向相反，停止沿墙行走，开始避障行为
                        self.act_mode = 4  # 停止沿墙行走，转入朝向目标模式后转入避障行为
                        self.reset_bypass()  # ======== 清空绕障数据 ========

        elif self.act_mode == 2:  # 右沿墙模式
            intervals = self.find_safe_angle_intervals()    # 从右侧开始扫，扫描全部扇形

            if len(intervals) == 0:  # 找不到符合宽度的扇面
                plan_speed = 0
                plan_angular_speed = self._rot_speed
                self.act_mode = 1
            else:
                # delta_heading = self.laserAngle[intervals[0][0]]
                delta_heading = 2 / 3 * self.laser_angles[intervals[0][0]] + 1 / 3 * self.laser_angles[intervals[0][1]]
                plan_speed = self.speed(delta_heading)
                plan_angular_speed = self._k_rot * delta_heading

                # 计算比例因子，判断是否在M线上
                if self.goal_dis < self.minD:
                    self.minD = self.goal_dis
                    if 5 < self.goal_angle < 170:  # # 如果目标方向与沿墙壁方向相反，停止沿墙行走，开始避障行为
                        self.act_mode = 4  # 停止沿墙行走，转入朝向目标模式后转入避障行为
                        self.reset_bypass()  # ======== 清空绕障数据 ========

        # 朝向目标（暂态模式，朝向目标后即开始避障
        elif self.act_mode == 4:
            if abs(self.goal_angle) > 10:
                plan_speed = 0
                plan_angular_speed = np.sign(self.goal_angle) * min(self._k_rot * abs(self.goal_angle), self._rot_speed)
            else:
                plan_speed = 0
                plan_angular_speed = 0
                self.act_mode = 0

        elif self.act_mode == 5:
            self.act_mode = self.turn_dir()
            self.target_robot.dispose_turn_point(self.act_mode)    # 测试中心记录模态切换到转向时的相关数据
            self.bypass_points.append(self.position)  # ======== 记录绕障点 =========
            self.bypass_angles.append(self.yaw)

            print("robot —{} — change to mode {}， cur position : {}".format(self.target_robot.id, self.act_mode, self.position))
            plan_speed = 0
            plan_angular_speed = 0


        plan_speed = min(plan_speed, 0.5 * self.goal_dis)  # 接近终点时减速
        plan_speed = min(plan_speed, self._constant_speed)    # 不超过最大限速
        if abs(plan_angular_speed/self._k_rot) > 10:  # 限制转弯大小
            plan_speed = 0

        # #  for test
        # if self.act_mode == 2 or self.act_mode == 1:
        #     plan_speed = 0
        #     plan_angular_speed = -30


        # 解决原地转圈问题
        if plan_speed == 0 and plan_angular_speed != 0:
            if self.rotate_angle == 0 and self.last_yaw == 181: # 第一次旋转
                self.last_yaw = self.target_robot.yaw
            else:
                self.rotate_angle += angle_transform(self.target_robot.yaw - self.last_yaw)
                self.last_yaw = self.target_robot.yaw
                if abs(self.rotate_angle) > 360 + 30:
                    self.act_mode = 4
                    self.rotate_angle = 0
                    self.last_yaw = 181
        else:
            self.rotate_angle = 0
            self.last_yaw = 181

        return plan_speed, plan_angular_speed


if __name__ == "__main__":
    import configure
    import multiRobot

    config = configure.get_config()
    robot = multiRobot.CentralizedOneRobot(0, config)
    planner = DhPathPlanner_HPoint(robot, config)
    data = np.load("map0000robots15_test_result.npz")
    planner.laser_row = data["opt_turn_dir_lasers"][0][0]
    planner.laser_angles = data["laser_angles"]
    intervals = planner.find_safe_angle_intervals()
    intervals = planner.find_safe_run_index()
    print(intervals)