# -*- coding: UTF-8 -*-
import copy
import math
import time

import cv2
import numpy as np
import random
from easydict import EasyDict
import os
import threading
import time
import configure
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




class Map:
    def __init__(self, gmap, map_scale, cor_origin=None, map_id=0, safe_dis=0.15):
        self.map = gmap  # numpy 二维数组，0-1矩阵
        self.shape = gmap.shape
        self.id = map_id
        if cor_origin is None:
            self.cor_origin = [self.shape[0] - 1, 0]  # 坐标原点的矩阵索引
        else:
            self.cor_origin = cor_origin

        # self.config = config
        self.resolution = map_scale
        self.safe_dis = safe_dis
        # self.radius = config.robotRadius
        # self.dist_mx = np.ceil(self.radius / self.resolution)
        # self.num_cases = config.num_samples  # 样本数量

        right_top = self.abs_cor(0, self.shape[1] - 1)
        self.max_map_x = right_top[0]
        self.max_map_y = right_top[1]
        left_bottom = self.abs_cor(self.shape[0] - 1, 0)
        self.min_map_x = left_bottom[0]
        self.min_map_y = left_bottom[1]
        self.obstacles = np.argwhere(self.map == 1)  # 障碍物区域
        start_time = time.time()
        self.safe_map = self.get_safe_map()  # safe = 0 , dangerous = 1
        # cv2.imwrite("test_map.png", self.safe_map*255)
        print("time used to ompute safe_matrix is {}".format(time.time()-start_time))

        self.x_range, self.y_range = self.shape
        self.motions = [(-1, 0), (-1, 1),(0, 1), (1, 1),(1, 0), (1, -1),(0, -1),(-1, -1), 
                         ]
        self.obs = self.obs_map()
        self.cases = []

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        xs, ys = np.where(self.safe_map == 1)
        return set(zip(xs,ys))

    def get_safe_map(self):
        safe_map = self.map.copy()

        keep_dis = int(np.ceil(self.safe_dis/self.resolution))

        # 计算周边原型区域keep_dis范围内内的点
        area = np.zeros([2 * keep_dis + 1, 2 * keep_dis + 1])
        for r in range(-1 * keep_dis, keep_dis + 1, 1):
            for t in range(-1 * keep_dis, keep_dis + 1, 1):
                if math.sqrt(r*r+t*t) < keep_dis:
                    area[keep_dis+r, keep_dis+t] = 1

        for i in range(keep_dis,safe_map.shape[0]-keep_dis,1):
            for j in range(keep_dis,safe_map.shape[1]-keep_dis,1):
                obstacle = self.map[i-keep_dis:i+keep_dis+1,j-keep_dis:j+keep_dis+1]
                if np.max(np.multiply(obstacle, area)) == 1:
                    safe_map[i][j] = 1

        # 边界区域需要全部置1
        safe_map[0:keep_dis, :] = 1
        safe_map[safe_map.shape[0]-keep_dis: -1,:] = 1
        safe_map[:, 0:keep_dis] = 1
        safe_map[:, safe_map.shape[0]-keep_dis: -1] = 1
        return safe_map
 
    def is_dangerous(self,i, j):
        return self.safe_map[i][j]

    def cases_generator_0(self, num_cases, num_agents, robot_radius):
        cases = []
        while len(cases) < num_cases:   # 当没有生成足够的case
        # for _ in range(num_cases):
            case = []
            points = []
            success_flag = True
            for i in range(num_agents):
                agents_angle = random.randint(0, 359)  # 初始角度
                start_cor, points, success_flag = self.start_point_generator(points, robot_radius)  # 初始位置
                if not success_flag:   # 当生成点的时间超时
                    break
                end_cor = self.point_generator(robot_radius)  # 终点位置
                case.append([start_cor, end_cor, agents_angle])
            if success_flag:
                cases.append(case)
        return cases

    def cases_generator(self, num_cases, num_agents, robot_radius):  # parallel
        # 并行程序生成多个case
        self.cases = []
        jobs = []
        for i in range(num_cases):
            p = threading.Thread(target = self.generate_one_case, args=(num_agents,robot_radius))
            p.start()
            jobs.append(p)

        for job in jobs:
            job.join()
        return self.cases

    def generate_one_case(self, num_agents, robot_radius):
        # 生成一个case， 且生成过程中不超时太多
        succes_flag = True
        while True:
            case = []
            points = []
            for i in range(num_agents):
                agents_angle = random.randint(0, 359)  # 初始角度
                start_cor, points, success_flag = self.start_point_generator(points, robot_radius)  # 初始位置
                if not success_flag:  # 当生成点的时间超时
                    break
                end_cor = self.point_generator(robot_radius)  # 终点位置
                case.append([start_cor, end_cor, agents_angle])
            if success_flag:
                self.cases.append(case)
                print("generate one case successfully!")
                break

    def test(self):
        start_time = time.time()
        points = []
        self.start_point_generator(points, 0.15)
        end_time = time.time()
        print("time used for generate one start point is {}".format(end_time-start_time))

    def point_generator(self, robot_radius):
        # 随机生成点
        array_freespace = np.argwhere(self.safe_map == 0)  # 可行区域   # 更换为安全区域
        # array_obstaclesspace = np.argwhere(self.map == 1)  # 障碍物区域
        mx = random.choice(array_freespace)
        point = np.array(self.abs_cor(mx[0], mx[1]))
        # 判断点是否属于边界或者与障碍物重合
        while not self.check_robots_xoy_in(point, robot_radius) or\
                self.check_robots_obstacle_in(mx, robot_radius):
            # print("reselect!")
            mx = random.choice(array_freespace)
            point = np.array(self.abs_cor(mx[0], mx[1]))
        return point

    def start_point_generator(self, points, robot_radius):
        # 生成机器人的初始位置，初始位置需要蔓旭（1）不与障碍物重合 (2) 机器人之间不相碰撞
        point = self.point_generator(robot_radius)
        since = time.time()
        generate_flag = True
        while self.is_robots_coincide(point, points, robot_radius):
            point = self.point_generator(robot_radius)
            if time.time() - since > 60:
                generate_flag = False
                break
        points.append(point)
        return point, points, generate_flag

    def check_xoy_in(self, point):
        # 判断全局坐标点是否在地图内
        if self.min_map_x <= point[0] <= self.max_map_x and self.min_map_y <= point[1] <= self.max_map_y:
            return True
        else:
            return False

    def check_robots_xoy_in(self, point, radius):
        assert 2 * radius < min(self.max_map_y-self.min_map_y, self.max_map_x-self.min_map_x), "robot too big"
        # 1、判断全局坐标点机器人半径范围内是否都在地图内
        if self.min_map_x + radius < point[0] < self.max_map_x - radius\
                and self.min_map_y + radius < point[1] < self.max_map_y - radius:
            return True
        else:
            return False

    def check_robots_obstacle_in(self, mx, robot_radius):
        # 2、 判断生成的位置间与障碍物间的距离是否小于半径,如果小于，返回True
        if np.min(np.linalg.norm(mx-self.obstacles, ord=2, axis=1)) < robot_radius/self.resolution+2:
            return True
        else:
            return False

    def is_robots_coincide(self, point, points, robot_radius):
        # 3、判断生成的位置间是否相差2个半径，,如果大于两个机器人半径，不碰撞， 返回False
        for other_point in points:
            if np.linalg.norm(point-other_point, 2) < 2 * robot_radius + 4:
                return True
        return False

    def is_obstacle(self, i, j):
        if self.map[i][j] == 1:
            return True
        else:
            return False

    def abs_cor(self, i, j):
        # 地图坐标系下矩阵格点对应的实际坐标值
        return [self.resolution * (j - self.cor_origin[1]), self.resolution * (self.cor_origin[0] - i)]

    def get_safe_index(self, position):  # need to check 
        """用于获取机器人在实际地图中最近的安全点，以弥补安全距离与实际障碍物的格点表示带来的gap"""
        mxx, mxy = self.matrix_idx(position[0], position[1])   # x * shape + y
        queue = []
        has_visited = []
        queue.append(self._cor2int(mxx,mxy))
        while len(queue) > 0:
            x, y = self._int2cor(queue[0])
            has_visited.append(queue[0])
            nghb_list = self.get_nghb_int_list(queue[0])

            del queue[0]
            if not self.is_dangerous(x, y):
                return x, y

            for nghb in nghb_list:
                if not np.isin(nghb,has_visited):
                    queue.append(nghb)

    def get_nghb_int_list(self, num):
        """获取一个以整数表示的坐标点对应的邻居列表"""
        row, col = self.shape
        x, y = self._int2cor(num)
        nghb = np.array([num-1, num+1, num-col-1, num-col, num-col+1, num+col-1, num+col, num+col+1])

        if 0 < x < row-1 and 0 < y < col-1:
            return nghb
        else:  # need to test 
            nghb = nghb[nghb>=0]
            nghb = nghb[nghb<=(row-1)*(col-1)]
            nums = []
            for n in nghb:
                if abs(n%col - y) != col-1:
                    nums.append(n)
            return nums
        

    def _cor2int(self, x_cor, y_cor):
        """将坐标点映射为整数"""
        return x_cor* self.shape[1] + y_cor

    def _int2cor(self, num):
        """将整数映射为坐标点"""
        assert 0 <= num < self.shape[0]*self.shape[1], "integer out of range"
        return num//self.shape[1], num%self.shape[1]

    def _matrix_idx(self, x_cor, y_cor):
        # 计算绝对坐标对应的矩阵的索引值，返回值范围为R
        # return [self.cor_origin[1] - x_cor/self.resolution, self.cor_origin[0] + y_cor/self.resolution]
        lb_x = self.cor_origin[0] - y_cor / self.resolution
        lb_y = x_cor / self.resolution + self.cor_origin[1]
        return lb_x, lb_y

    def matrix_idx(self, x_cor, y_cor):
        # 计算绝对坐标对应的矩阵的索引值，返回值范围为R
        # return [self.cor_origin[1] - x_cor/self.resolution, self.cor_origin[0] + y_cor/self.resolution]
        lb_x = self.cor_origin[0] - y_cor / self.resolution
        lb_y = x_cor / self.resolution + self.cor_origin[1]
        return self.trans_to_int([lb_x, lb_y])

    def project_line_seg(self, line_seg):
        """
        当线段终点在地图外时，将一个外部点投影到地图边界上, 返回调整后的线段
        :param cur_map:当前地图
        :param line_seg: 有向线段，其中第一个点为线段起点，第二个点为线段终点, 全局坐标系
        :param resolution: 矩阵两个相邻格点表示的实际地图上的距离
        :return: 线段终点的投影, map矩阵索引
        """
        ls_start, ls_end = line_seg
        if not self.check_xoy_in(ls_start):  # 若线段起点不在地图范围内，返回空
            return None

        if self.check_xoy_in(ls_end):  # 若线段都位于地图内，返回线段本身
            new_end = ls_end
        else:
            x1, y1 = ls_start
            x2, y2 = ls_end
            if x1 == x2 and y2 > self.max_map_y:
                new_end = (x2, self.max_map_y)
            elif x1 == x2 and y2 < self.min_map_y:
                new_end = (x2, self.min_map_y)
            else:
                k = (y2 - y1) / (x2 - x1)  # 线段斜率
                slope_map = (self.shape[0] - 1) / (self.shape[1] - 1)  # 地图斜率
                if abs(k) < slope_map:  # 与Y轴的两条平行线段相交的情况
                    if x2 > x1:
                        new_end = (self.max_map_x, y1 + k * (self.max_map_x - x1))
                    else:
                        new_end = (self.min_map_x, y1 + k * (self.min_map_x - x1))
                else:  # 与X轴的两条平行线段相交的情况
                    if y2 > y1:
                        new_end = (x1 + (self.max_map_y - y1) / k, self.max_map_y)
                    else:
                        new_end = (x1 + (self.min_map_y - y1) / k, self.min_map_y)
        return ls_start, new_end

    def find_point(self, line_seg):
        """
        寻找从线段起点到线段终点的最远可行点的索引
        :param line_seg:  绝对坐标系下线段的起始点和终止点
        :return:
        """
        line_seg_project = self.project_line_seg(line_seg)  # 地图范围内全局坐标系下的方向线段
        if line_seg_project is None:
            return None
        max_point = self.find_max_reachable_1(line_seg_project)
        return max_point

    def find_max_reachable(self, projected_line_seg):
        """
        计算从有向线段起点开始走向有向线段终点最远能走到的点
        :param cur_map: numpy数组， 0-1矩阵表示的地图，以全局坐标系表示，左下角为坐标圆点，向右为x正方向，向上为y正方向
        :param projected_line_seg: 全局xoy坐标系下的有向线段表示，如：[(0,0), (4,5)]
        :return: 最远能走到的点的全局xoy坐标
        """
        ls_start, ls_end = projected_line_seg
        start_mx = self._matrix_idx(ls_start[0], ls_start[1])
        end_mx = self._matrix_idx(ls_end[0], ls_end[1])

        x1, y1 = self.trans_to_int(start_mx)  # 起始点
        x2, y2 = self.trans_to_int(end_mx)  # 终止点

        if x1 == x2 and y1 == y2:
            return x1, y1

        if x1 == x2:
            dy = int((y2 - y1) / abs(y1 - y2))
            for i in range(y1 + dy, y2 + dy, dy):
                if self.map[x1][i]:  # 未到达端点已遇到障碍，返回上一个可达点
                    return x1, i - dy
                elif i == y2 and self.map[x2, y2] == 0:  # 到达线段端点，则返回端点本身
                    return x1, y2
        else:
            dx = int((x2 - x1) / abs(x2 - x1))
            k = (y2 - y1) / (x2 - x1)
            for xi in range(x1 + dx, x2 + dx, dx):
                yi = y1 + k * (xi - x1)
                if not self.check_idx_in([xi, int(yi)]) or self.map[xi][int(yi)] \
                        or not self.check_idx_in([xi, int(yi) + 1]) or self.map[xi][int(yi) + 1]:
                    return xi - dx, int(y1 + k * (xi - dx - x1))
                elif xi == x2 and self.map[x2][y2] == 0:
                    return x2, y2

    def find_max_reachable_1(self, projected_line_seg):
        """
        计算从有向线段起点开始走向有向线段终点最远能走到的点
        :param cur_map: numpy数组， 0-1矩阵表示的地图，以全局坐标系表示，左下角为坐标圆点，向右为x正方向，向上为y正方向
        :param projected_line_seg: 全局xoy坐标系下的有向线段表示，如：[(0,0), (4,5)]
        :return: 最远能走到的点的全局xoy坐标
        """
        ls_start, ls_end = projected_line_seg
        start_mx = self._matrix_idx(ls_start[0], ls_start[1])
        end_mx = self._matrix_idx(ls_end[0], ls_end[1])

        x1, y1 = self.trans_to_int(start_mx)  # 起始点
        x2, y2 = self.trans_to_int(end_mx)  # 终止点

        if np.linalg.norm(np.array(ls_start)-np.array(ls_end),2) < self.safe_dis:
            return x1, y1

        if x1 == x2:
            dy = int((y2 - y1) / abs(y1 - y2))
            i = y1
            while i <= y2 and not self.is_obstacle(x1,i):  # 先寻找无障碍的最大点
                i += dy
            i -= dy
            while i >= y1 and self.is_dangerous(x1, i):  # 再退回到安全点
                i -= dy
            return x1, i
            
        else:
            dx = int((x2 - x1) / abs(x2 - x1))
            k = (y2 - y1) / (x2 - x1)
            xi = x1
            yi = y1
            while xi != x2 and not self.is_obstacle(xi,yi) and not self.is_obstacle(xi,yi+1) :
                xi += dx
                yi = int(y1 + k * (xi - x1))
                if not self.check_idx_in([xi, yi]) or not self.check_idx_in([xi, yi+1]) :
                    xi -= dx
                    yi = int(y1 + k * (xi - x1))
                    break
            while xi != x1 and (self.is_dangerous(xi,yi) or self.is_dangerous(xi,yi+1)) :
                xi -= dx
                yi = int(y1 + k * (xi - x1))
            return xi, yi

    def find_midpoint(self, start_mx, end_mx):
        # 寻找起始点与最大可达点之间的间距为1m的中间点
        s_x, s_y = start_mx
        e_x, e_y = end_mx
        midpoints = []
        for i in range(int(np.linalg.norm(np.array(start_mx)-np.array(end_mx),ord=2) * self.resolution / 1)):
            dis = (i + 1) * 1/self.resolution
            if s_x == e_x:
                point = self.trans_to_int((s_x, s_y+dis))
            else:
                dx = int((e_x - s_x) / abs(e_x - s_x))
                k = (s_y - e_y) / (s_x - e_x)
                point = self.trans_to_int((s_x+dx*dis/math.sqrt((k*k+1)),s_y+dx*dis*k/math.sqrt(k*k+1)))
            if not self.is_dangerous(point[0], point[1]):   # 如果中间点安全
                midpoints.append(point)
        return midpoints

    def trans_to_int(self, coor):
        """
        将其他类型的二维元组转化为为整数类型的二维元组
        :param coor: 长度为2的元组
        :return: 整型，长度为 2 的元组
        """
        x, y = np.around(coor)
        if y > self.shape[1]:
            y = self.shape[1] - 1
        if x > self.shape[0]:
            x = self.shape[0] - 1
        return int(x), int(y)

    def check_idx_in(self, point):
        """
        判断一个坐标是否在地图的表示范围内
        :param cur_map: 当前的地图二值矩阵
        :param point: 点的矩阵坐标
        :return: True or False
        """
        if 0 <= point[0] <= self.shape[0] - 1 and 0 <= point[1] <= self.shape[1] - 1:
            return True
        else:
            return False

    def paint_paths_0(self, paths_abs, result_filename):
        result = (1-copy.deepcopy(self.map))*100 + 150
        colors = [i for i in range(3, 255, 1)]
        paint_colors = random.sample(colors, len(paths_abs) + 2)
        for i, path in enumerate(paths_abs):
            for point in path:
                x, y = self.matrix_idx(point[0], point[1])
                result[x][y] = 0

        cv2.imwrite(result_filename, result)

    def paint_paths_0(self, paths_abs, result_filename, case):
        # 随机生成不同颜色
        result = (1-copy.deepcopy(self.map))*100 + 150
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # 将单通道转换成三通道
        for i, path in enumerate(paths_abs):
            paint_colors = np.random.randint(3, 200, 3, dtype=np.int32)  # 对不同机器人的路径生成不同颜色

            start_point = self.matrix_idx(path[0][0], path[0][1])  # 标注起始点位置
            goal_point = self.matrix_idx(case[i][1][0], case[i][1][1])  # 标注目标点位置

            cv2.circle(result, (start_point[1], start_point[0]), 3, (int(paint_colors[2]), int(paint_colors[1]), int(paint_colors[0])), -1) ## 还存在问题
            cv2.circle(result, (goal_point[1], goal_point[0]), 3, (int(paint_colors[2]), int(paint_colors[1]), int(paint_colors[0])), -1)

            # 方块型标注
            # result[start_point[0]-3:start_point[0]+3, start_point[1]-3:start_point[1]+3] = paint_colors
            # result[goal_point[0]-3:goal_point[0]+3, goal_point[1]-3:goal_point[1]+3] = paint_colors
            for point in path:
                x, y = self.matrix_idx(point[0], point[1])
                result[x][y] = paint_colors

        cv2.imwrite(result_filename, result)

    def paint_paths(self, paths_abs, result_filename, case):
        # 固定路径颜色
        result = (1 - copy.deepcopy(self.map)) * 100 + 150
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # 将单通道转换成三通道
        # paint_colors = np.random.randint(151, 249, size=[21, 3])
        # paint_colors[0], paint_colors[1] = [0, 0, 255], [102, 178, 255]  #
        # paint_colors[2], paint_colors[3] = [0, 204, 204], [0, 255, 0]
        # paint_colors[4], paint_colors[5] = [255, 127, 0], [255, 0, 0]
        # paint_colors[6], paint_colors[7] = [255, 0, 139], [0, 0, 0]
        # paint_colors[8],paint_colors[9] = [255,255,0], [255,0,255]
        font = cv2.FONT_HERSHEY_SIMPLEX
        paint_colors = np.array([
            [0, 0, 255],  # 蓝
            [0, 255, 0],  # 绿
            [255, 0, 0],  # 红


            [255,153,18], # 镉黄
            [176,23,31],  #印度红
            [255,215,0], # 金黄
            [85,102,0], # 黑黄
            [0,255,255], #青色

            [34,139,34], # 森林绿
            [3,168,158], # 锰蓝
            [0,199,140], # 土耳其蓝
            [138,43,226], # 紫罗兰
            [160,82,45], #赫色

            [25,25,112], #深蓝
            [199,97,20], #土色
            [255,125,64], #肉黄
            [255,99,71], #番茄红
            [8,46,84], #靛青色

            [128, 138, 135],  # 冷灰
            [192, 192, 192],  # 灰
            # https://www.cnblogs.com/general001/articles/4151861.html
        ],dtype=np.int32)

        for i, path in enumerate(paths_abs):
            start_point = self.matrix_idx(path[0][0], path[0][1])  # 标注起始点位置
            goal_point = self.matrix_idx(case[i][1][0], case[i][1][1])  # 标注目标点位置

            cv2.putText(result, str(i),(start_point[1], start_point[0]),font,0.8,
                        (int(paint_colors[i][0]), int(paint_colors[i][1]), int(paint_colors[i][2])),2)
            cv2.circle(result, (start_point[1], start_point[0]), 3,
                       (int(paint_colors[i][0]), int(paint_colors[i][1]), int(paint_colors[i][2])), -1)
            # cv2.circle(result, (goal_point[1], goal_point[0]), 3,
            #            (int(paint_colors[i][0]), int(paint_colors[i][1]), int(paint_colors[i][2])), -1)
            result[goal_point[0] - 3:goal_point[0] + 3, goal_point[1] - 3:goal_point[1] + 3] = paint_colors[i] # 用方块标注目标点

            for point in path:
                x, y = self.matrix_idx(point[0], point[1])
                result[x][y] = paint_colors[i]


        dx = int(2/self.resolution)
        for i in range(1, int(self.shape[0]/dx), 1):
            cv2.line(img=result, pt1=(dx * i, 0), pt2=(dx*i, self.shape[0] - 1), color=(192,192,192), thickness=1)
            cv2.line(img=result, pt1=(0, dx * i), pt2=(self.shape[0]-1, dx*i), color=(192,192,192), thickness=1)

        cv2.imwrite(result_filename, result)


def load_npy_map(map_npy_filepath, resolution):
    """
    加载实际使用的地图并返回地图对象, 待实际运行时完成
    :return:
    """
    map_matrix = np.load(map_npy_filepath)
    map_id = int(map_npy_filepath.split(".")[0].split("map")[-1])
    map_matrix[map_matrix == 255] = 0
    map_matrix[0][0] = 1
    # map_matrix = np.rot90(map_matrix, k=-1)              # ===================================================

    return Map(map_matrix, resolution, map_id=map_id)


def get_npy_files_list(target_folder="numpyMapFolder/"):
    if not os.path.exists(target_folder):
        return []
    npy_lists = []
    for _, _, filenames in os.walk(target_folder):
        for filename in filenames:
            if filename.endswith(".npy"):
                npy_lists.append(filename)
    return npy_lists


def load_A_star_expert():
    c_runner = ctypes.cdll.LoadLibrary("./libastar.so")
    c_runner.getshortest.restype = pair
    c_runner.getshortest.argtypes = [ndpointer(ctypes.c_int), point, point, ctypes.c_float]
    return c_runner


def compute_expert_dis(c_astar, map_npy, start_mx, end_mx):
    since = time.time()
    astar_len = c_astar.getshortest(map_npy.astype(np.int32),
                                         point(x=start_mx[0], y=start_mx[1]), point(x=end_mx[0], y=end_mx[1]), 1.0)
    print("C based Astar , begin ({}, {})- end ({},{})- length:{:.2f}, use time : {:.2f}\n".format(start_mx[0],
                                                                                                   start_mx[1],
                                                                                                   end_mx[0],
                                                                                                   end_mx[1],
                                                                                                   astar_len.F,
                                                                                                   time.time() - since))
    return astar_len.F    # 无规划路径时返回值为-1



# 计算障碍物系数，方式是用A* 路径长除以直线路径长度
def compute_obstacle_coefficient(map_npy_folder="numpyMapFolder/"):
    config = configure.get_config()
    map_npy_list = get_npy_files_list(map_npy_folder)
    expert = load_A_star_expert()
    total_A_star_length = 0  # A* 距离
    total_line_length = 0  # 绝对坐标下的欧式距离
    total_cor_line_length = 0  # 矩阵坐标下的欧式距离
    total_compute_num = 0 # 计算的总个数
    for npy_file in map_npy_list:
        since = time.time()
        cur_map = load_npy_map(map_npy_folder+npy_file, config.map_resolution)
        cases = cur_map.cases_generator(10,config.num_agents, config.robotRadius)
        for case in cases:
            for robot_task in case:
                start_mx = cur_map.matrix_idx(robot_task[0][0], robot_task[0][1])
                end_mx = cur_map.matrix_idx(robot_task[1][0], robot_task[1][1])
                len = compute_expert_dis(expert, cur_map.map, start_mx, end_mx)
                if len != -1:
                    total_A_star_length += len
                    total_compute_num += 1
                    total_line_length += np.linalg.norm(robot_task[1] - robot_task[0], 2)
                    total_cor_line_length += np.linalg.norm(np.array(end_mx) - np.array(start_mx), 2)
        print("time used for {} is {}".format(npy_file, time.time() - since))
        # break

    print("average A star length is {}".format(total_A_star_length/total_compute_num))
    print("statistics result, A* vs Euclidean vs Cor Enclidean: {}  , {} , {} ".format(total_A_star_length, total_line_length, total_cor_line_length))
    return  total_A_star_length/total_compute_num


if __name__ == "__main__":
    def test1():
        config = EasyDict({'run_env_type': 0,  # 运行环境类型, 0为ROS环境，1为AirSim
                           'alg_mode': 2,  # 0 与网络无关， 1 数据生成模式,  2 ros 地图生成模式

                           'num_agents': 4,  # 智能体个数
                           'max_detect': 5,   # 激光雷达传感器最大感知范围
                           'map_size': 20,    # 正方形地图尺寸
                           # 'map_scale': 0.03,  # gazebo地图比例尺
                           'num_directions': 6,  # 预选方向个数
                           'num_hops': 2,  # 通信跳数
                           'robotRadius': 0.15,  # 机器人半径
                           'laser_num': 360,   # 激光传感器扫过一圈测得的角度个数
                           'sector_safe_width': 60,  # 扇面安全宽度，以角度计算
                           'rot_speed': 20,  # 原地旋转角速度, 单位度每秒
                           'dis_threthold': 0.6,  # 安全阈值，即以做大最度前进时，刹车停下走过的最短距离
                           'agent_decel': 2,  # 机器人的加速度
                           'constant_speed': 0.5,  # 机器人的最大行进速度 dhbug算法中用于计算速度的v0值
                           'k_rot': 2,  # 运动机器人的角度修正参量
                           'min_detect': 0.15,  # 激光雷达最小阈值
                           'use_nn': True,  # 是否使用神经网络作为优化器
                           "dim_feature": 128,    # 神经网络输出向量维度
                            "launch_folder": "/home/ubuntu/Documents/ROS/hello/src/gpkg/scripts/launchFolder/"  # ros工程配置
                           })
        config.com_radius = config.max_detect   # 假设条件： 通信半径等于感知半径
        config.map_resolution = config.max_detect / 100  # 地图比例尺
        npy_map_path = "/home/ubuntu/Documents/ROS/hello/src/gpkg/scripts/numpyMapFolder/"
        npy_map_files_list = [fn for fn in os.listdir(npy_map_path) if fn.endswith("npy")]
        for i in range(len(npy_map_files_list)):
            # if int(npy_map_files_list[i].split(".npy")[0].split("map")[-1]) != 491:
            #     continue
            print("This is", npy_map_files_list[i].split(".npy")[0])
            npy_file = npy_map_path + npy_map_files_list[i]
            map_matrix = np.load(npy_file)
            map_matrix[map_matrix == 255] = 0
            map_matrix[0][0] = 1
            cur_map = Map(map_matrix, config.map_resolution)
            # cur_map.paint_paths((0,1),"bus.jpg",cur_map.cases_generator(1,10,0.3)[0])
            # cur_map.test()
            # start_time = time.time()
            cases = cur_map.cases_generator(3, 15, config.robotRadius)
            print(cases)
            # print("time used for generate a case is {}".format(time.time() - start_time))
            # break

    # compute_obstacle_coefficient()
    test1()
