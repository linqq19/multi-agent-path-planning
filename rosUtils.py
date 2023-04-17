import random
import cv2
import numpy as np
import math
import os
import time
import roslaunch
import re
import map
from easydict import EasyDict

def shutdown_ros_map(launch):
    # 关闭当前launch文件
    time.sleep(1)
    launch.shutdown()  # 关闭launch文件
    time.sleep(5)


def launch_ros_map(uuid, launch_path_file):
    # 运行launch文件
    launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_path_file])
    launch.start()
    time.sleep(10)  # 等待gazebo完全加载
    return launch


def create_ros_pid():
    # 创建启动launch文件的进程
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)
    return uuid


def is_target_launch_file(launch_file_name):
    """检查launch文件名是否符合指定格式"""
    ret = re.match("^map\d{4}robots\d{2}case\d{2}\.launch$", launch_file_name)
    if ret:
        return True
    else:
        return False


def is_target_world_file(world_file_name):
    """检查world文件名是否符合指定格式"""
    ret = re.match("^map\d{4}.world$", world_file_name)
    if ret:
        return True
    else:
        return False


def find_max_world_id(world_dir):
    """寻找指定文件夹下格式为“mapXXX.world”的最大XXX值"""
    if not os.path.exists(world_dir):
        return -1
    max_id = -1
    for _, _, filenames in os.walk(world_dir):
        for filename in filenames:
            if is_target_world_file(filename):
                cur_id = int(filename.split(".")[0].split("map")[-1])
                if cur_id > max_id:
                    max_id = cur_id
    return max_id


def search_target_launch_files(target_folder="launchFolder"):
    # 寻找launch文件夹下的所有launch文件
    if not os.path.exists(target_folder):
      return []
    launch_files = []
    for _, _, filenames in os.walk(target_folder):
        for filename in filenames:
            if is_target_launch_file(filename) and has_correspond_npy_file(filename):
                launch_files.append(filename)
    return launch_files


def has_correspond_npy_file(launch_file_name):
    # 寻找目录中是否有当前launch文件对应的npy地图二进制文件
    map_name = launch_file_name.split(".")[0].split("robots")[0]
    if os.path.exists("numpyMapFolder/" + map_name + ".npy"):
        return True
    else:
        return False


def launch_generator(world_map_folder_path, npy_map_folder_path, launch_folder_path, map_scale, num_robots,
                     robot_radius):
    """
    根据world文件和npy文件生成相应的launch文件
    :param world_map_folder_path: world文件保存位置
    :param npy_map_folder_path:npy文件保存位置
    :param launch_folder_path:生成的launch文件保存位置
    :param map_scale: 矩阵-地图比例尺
    :param num_robots: 机器人个数
    :return:
    """
    if not os.path.exists(launch_folder_path):
        os.mkdir(launch_folder_path)
    assert os.path.exists(world_map_folder_path), "No map files exists"
    map_files_list = [fn for fn in os.listdir(world_map_folder_path) if fn.endswith("world")]
    map_files_list.sort()
    for world_file in map_files_list:
        launch_name = launch_folder_path + world_file.split(".")[0] + 'robots' + str(num_robots).zfill(2)+ "case01.launch"
        if is_target_world_file(world_file) and has_correspond_npy_file(world_file) and not \
                os.path.exists(launch_name):
            print("generate launch file, current map id : {}".format(world_file.split(".")[0].split("map")[-1]))
            npy_file = npy_map_folder_path + world_file.split(".")[0] + '.npy'
            g_map = map.load_npy_map(npy_file, map_scale)
            cases = g_map.cases_generator(1, num_robots, robot_radius)
            # 设置开头
            msg_1 = \
                "<launch>\n"
            for i in range(num_robots):
                car_msg_1 = " \n<group ns="'"car' + str(i + 1) + '"' ">\n\
                <param name=" + '"robot_description"'" command=" + '"$(find xacro)/xacro $(find gpkg)/urdf/xacro/672combine.xacro ns:=car' + str(
                    i + 1) + '"' + " />\n\
                <node name=" + '"joint_state_publisher"'" pkg=" + '"joint_state_publisher"'" type=" + '"joint_state_publisher"' + "/>\n\
                <node name=" + '"robot_state_publisher"'" pkg=" + '"robot_state_publisher"'" type=" + '"robot_state_publisher"' + "/>\n\
                <node name=" + '"spawn_model"'" pkg=" + '"gazebo_ros"'" type=" + '"spawn_model"'" args=" + '"-urdf -model car' + str(
                    i + 1) + ' -param robot_description -x ' + str(cases[0][i][0][0]) + ' -y ' + str(
                    cases[0][i][0][1]) + ' -Y ' + str(cases[0][i][2] * np.pi / 180) + ' "' + "/>\n\
    </group>\n"
                msg_1 = msg_1 + car_msg_1
            # 结束开头
            msg = msg_1 + \
                  "\n\t\t<!-- 启动 gazebo -->\n\
          <include file=" + '"$(find gazebo_ros)/launch/empty_world.launch"'">\n\
            <arg name=" + '"world_name"' " value=" + '"$(find gpkg)/scripts/worldFolder/' + world_file + '"' + " />\n\
        </include>\n"
            msg_0 = "</launch>\n"
            msg = msg + msg_0

            with open(launch_name,'w') as f:
                f.write(msg)

def generate_given_map_launch(world_map_folder_path, world_file, launch_folder_path, cases, launch_id):
    if not os.path.exists(launch_folder_path):
        os.mkdir(launch_folder_path)
    assert os.path.exists(world_map_folder_path+world_file), "No map files exists"
    num_robots = len(cases[0])
    launch_name = launch_folder_path +"/"+ world_file.split(".")[0] + 'robots' + str(num_robots).zfill(2) + "case" + str(launch_id).zfill(2) + ".launch"
    if os.path.exists(launch_name):
        print("launch file already exists")
        return
    # 设置开头
    msg_1 = \
        "<launch>\n"
    for i in range(num_robots):
        car_msg_1 = " \n<group ns="'"car' + str(i + 1) + '"' ">\n\
                    <param name=" + '"robot_description"'" command=" + '"$(find xacro)/xacro $(find gpkg)/urdf/xacro/672combine.xacro ns:=car' + str(
            i + 1) + '"' + " />\n\
                    <node name=" + '"joint_state_publisher"'" pkg=" + '"joint_state_publisher"'" type=" + '"joint_state_publisher"' + "/>\n\
                    <node name=" + '"robot_state_publisher"'" pkg=" + '"robot_state_publisher"'" type=" + '"robot_state_publisher"' + "/>\n\
                    <node name=" + '"spawn_model"'" pkg=" + '"gazebo_ros"'" type=" + '"spawn_model"'" args=" + '"-urdf -model car' + str(
            i + 1) + ' -param robot_description -x ' + str(cases[0][i][0][0]) + ' -y ' + str(
            cases[0][i][0][1]) + ' -Y ' + str(cases[0][i][2] * np.pi / 180) + ' "' + "/>\n\
        </group>\n"
        msg_1 = msg_1 + car_msg_1
    # 结束开头
    msg = msg_1 + \
          "\n\t\t<!-- 启动 gazebo -->\n\
  <include file=" + '"$(find gazebo_ros)/launch/empty_world.launch"'">\n\
                <arg name=" + '"world_name"' " value=" + '"$(find gpkg)/scripts/worldFolder/' + world_file + '"' + " />\n\
            </include>\n"
    msg_0 = "</launch>\n"
    msg = msg + msg_0

    with open(launch_name,'w') as f:
        f.write(msg)


def generate_gazebo_world(resolution=0.05, map_shape=200,world_number=1):
    """
    resolution： 地图分辨率
    ap_shape: 地图二进制文件的大小
    生成gazebo所需的地.world文件以及地图图片、二进制npy文件
    world_num：生成的world的数目
    :return:
    """
    # 横纵坐标的列表
    c_value_x_list = []  # 横坐标
    c_value_y_list = []  # 纵坐标
    # model_radius = 0.2667 #圆柱体模型半径
    ratio = map_shape*resolution/10  # 放大倍率
    map_edge = map_shape * resolution
    cylinder_number = int(8 * ratio) # 单个world里圆柱体数量
    model_radius = np.random.uniform(0.2, 0.5 * ratio, size=cylinder_number)  # 圆柱体模型半径

    # ==========新加入的正方形块===========
    s_value_x_list = []  # 横坐标
    s_value_y_list = []  # 纵坐标
    s_yaw_list = []  # 矩形旋转
    square_number = cylinder_number  # 正方形块个数
    edge1 = np.random.uniform(0.2, 2 * ratio, size=square_number)  # 长方体长
    # edge2 = np.random.uniform(0.2, 1.2, size=square_number)  # 长方体宽
    edge2 = np.random.uniform(0.4, 1 * ratio, size=square_number)  # 长方体宽
    s_value_list = {"s_value_x_list": s_value_x_list, "s_value_y_list": s_value_y_list, "s_yaw": s_yaw_list}
    # ==================================
    c_value_list = {"c_value_x_list": c_value_x_list, "c_value_y_list": c_value_y_list}
    for j in range(world_number):
        c_value_x_list.clear()
        c_value_y_list.clear()
        # 生成圆柱形障碍，同时避免与边界相交
        for i in range(cylinder_number):
            c_value_x = random.uniform(1, map_shape * resolution-1)
            c_value_y = random.uniform(1, map_shape * resolution-1)
            while not (1 + model_radius[i] < c_value_x < map_edge - model_radius[i]
                       and 1 + model_radius[i] < c_value_y < map_edge - model_radius[i]):
                c_value_x = random.uniform(1, map_shape * resolution-1)
                c_value_y = random.uniform(1, map_shape * resolution-1)
            c_value_x_list.append(c_value_x)
            c_value_y_list.append(c_value_y)

        s_value_x_list.clear()
        s_value_y_list.clear()
        # === 新加入生成正方体障碍 =====
        for i in range(square_number):
            s_value_x = random.uniform(1, map_shape * resolution-1)
            s_value_y = random.uniform(1, map_shape * resolution-1)
            s_yaw = random.uniform(0, 6.28)  # 由于使用cv绘制困难，后续再加
            while not (1 + edge1[i] < s_value_x < map_edge - edge1[i]
                       and 1 + edge2[i] < s_value_y < map_edge - edge2[i]):
                s_value_x = random.uniform(1, map_shape * resolution-1)
                s_value_y = random.uniform(1, map_shape * resolution-1)
            s_value_x_list.append(s_value_x)
            s_value_y_list.append(s_value_y)
            s_yaw_list.append(s_yaw)

        # ===========================

        # 使用cv2绘制地图边界
        emptyImage3 = np.zeros((map_shape, map_shape), np.uint8)
        emptyImage3[...] = 255
        cv2.line(img=emptyImage3, pt1=(1, 1), pt2=(1, map_shape-1), color=1, thickness=2)
        cv2.line(img=emptyImage3, pt1=(1, 1), pt2=(map_shape-1, 1), color=1, thickness=2)
        cv2.line(img=emptyImage3, pt1=(map_shape-1, map_shape-1), pt2=(1, map_shape-1), color=1, thickness=2)
        cv2.line(img=emptyImage3, pt1=(map_shape-1, map_shape-1), pt2=(map_shape-1, 1), color=1, thickness=2)

        for i in range(cylinder_number):
            # print(c_value_list['c_value_x_list'][i])
            # print(c_value_list['c_value_y_list'][i])
            cv2.circle(emptyImage3, (round(c_value_list['c_value_x_list'][i] / resolution),
                                     map_shape - (round(c_value_list['c_value_y_list'][i] / resolution))),
                       radius=round(model_radius[i] / resolution), color=1, thickness=-1)
            # print((c_value_list['c_value_x_list'][i]/resolution))
            # print(map_shape-((c_value_list['c_value_y_list'][i]/resolution)))
            # print((round(c_value_list['c_value_x_list'][i]/resolution)))
            # print(map_shape-(round(c_value_list['c_value_y_list'][i]/resolution)))
            # print()
        # print(c_value_list)

        # ======= 新加入使用cv2绘制正方形 ===========
        for i in range(square_number):
            # 矩形中心点
            x0 = s_value_x_list[i]
            y0 = s_value_y_list[i]

            # 先计算未旋转时的矩形顶点坐标，左上，右下
            # 左上
            x1 = x0 - 0.5 * edge1[i]
            y1 = y0 + 0.5 * edge2[i]

            # 右下
            x2 = x0 + 0.5 * edge1[i]
            y2 = y0 - 0.5 * edge2[i]

            # 右上
            x3 = x0 + 0.5 * edge1[i]
            y3 = y0 + 0.5 * edge2[i]

            # 旋转后的顶点坐标，左上，右下
            # 左上
            x1r = (x1 - x0) * math.cos(s_yaw_list[i]) - (y1 - y0) * math.sin(s_yaw_list[i]) + x0
            y1r = (x1 - x0) * math.sin(s_yaw_list[i]) + (y1 - y0) * math.cos(s_yaw_list[i]) + y0

            # 右下
            x2r = (x2 - x0) * math.cos(s_yaw_list[i]) - (y2 - y0) * math.sin(s_yaw_list[i]) + x0
            y2r = (x2 - x0) * math.sin(s_yaw_list[i]) + (y2 - y0) * math.cos(s_yaw_list[i]) + y0

            # 右上
            x3r = (x3 - x0) * math.cos(s_yaw_list[i]) - (y3 - y0) * math.sin(s_yaw_list[i]) + x0
            y3r = (x3 - x0) * math.sin(s_yaw_list[i]) + (y3 - y0) * math.cos(s_yaw_list[i]) + y0

            # 通过划线的方式绘制实心矩形
            a = (y3r - y1r) / (x3r - x1r)  # 左上-右上边的斜率
            # b = (y3 - y1) / (x3 - x1)  # 右上-右下边的斜率

            # # 可用程序备份
            # for x in range(0, int(abs(x1r - x3r)/resolution)):
            #     if x1r < x3r:
            #         cv2.line(emptyImage3, (int((x3r - x*resolution)/resolution), map_shape-(int((y3r - a*x*resolution)/resolution))),
            #                  (int((x2r - x*resolution)/resolution), map_shape-(int((y2r - a*x*resolution)/resolution))), 1, 2)
            #     else:
            #         cv2.line(emptyImage3, (int((x3r + x*resolution)/resolution), map_shape-(int((y3r + a * x*resolution)/resolution))),
            #                  (int((x2r + x*resolution)/resolution), map_shape-(int((y2r + a * x*resolution)/resolution))), 1, 2)

            res = resolution * 0.02  # 使用线段绘制矩形的分辨率

            for x in range(0, int(abs(x1r - x3r) / res)):
                if x1r < x3r:
                    cv2.line(emptyImage3,
                             (int((x3r - x * res) / resolution), map_shape - (int((y3r - a * x * res) / resolution))),
                             (int((x2r - x * res) / resolution), map_shape - (int((y2r - a * x * res) / resolution))),
                             1, 2)
                else:
                    cv2.line(emptyImage3,
                             (int((x3r + x * res) / resolution), map_shape - (int((y3r + a * x * res) / resolution))),
                             (int((x2r + x * res) / resolution), map_shape - (int((y2r + a * x * res) / resolution))),
                             1, 2)

            # pt1 = (round(x1r/resolution), round(y1r/resolution))
            # pt2 = (round(x2r/resolution), round(y2r/resolution))
            # cv2.rectangle(emptyImage3, pt1, pt2, color=1, thickness=2) 该函数不能绘制旋转矩形

        # =======================================

        # 设置开头
        msg_1 = "<sdf version='1.6'>\n\
          <world name='default'>\n\
            <light name='sun' type='directional'>\n\
              <cast_shadows>0</cast_shadows>\n\
              <pose frame=''>0 0 10 0 -0 0</pose>\n\
              <diffuse>0.8 0.8 0.8 1</diffuse>\n\
              <specular>0.1 0.1 0.1 1</specular>\n\
              <attenuation>\n\
                <range>1000</range>\n\
                <constant>0.9</constant>\n\
                <linear>0.01</linear>\n\
                <quadratic>0.001</quadratic>\n\
              </attenuation>\n\
              <direction>-0.5 0.5 -1</direction>\n\
            </light>\n\
            <model name='ground_plane'>\n\
              <static>1</static>\n\
              <link name='link'>\n\
                <collision name='collision'>\n\
                  <geometry>\n\
                    <plane>\n\
                      <normal>0 0 1</normal>\n\
                      <size>100 100</size>\n\
                    </plane>\n\
                  </geometry>\n\
                  <surface>\n\
                    <friction>\n\
                      <ode>\n\
                        <mu>100</mu>\n\
                        <mu2>50</mu2>\n\
                      </ode>\n\
                      <torsional>\n\
                        <ode/>\n\
                      </torsional>\n\
                    </friction>\n\
                    <contact>\n\
                      <ode/>\n\
                    </contact>\n\
                    <bounce/>\n\
                  </surface>\n\
                  <max_contacts>10</max_contacts>\n\
                </collision>\n\
                <visual name='visual'>\n\
                  <cast_shadows>0</cast_shadows>\n\
                  <geometry>\n\
                    <plane>\n\
                      <normal>0 0 1</normal>\n\
                      <size>100 100</size>\n\
                    </plane>\n\
                  </geometry>\n\
                  <material>\n\
                    <script>\n\
                      <uri>file://media/materials/scripts/gazebo.material</uri>\n\
                      <name>Gazebo/Grey</name>\n\
                    </script>\n\
                  </material>\n\
                </visual>\n\
                <self_collide>0</self_collide>\n\
                <kinematic>0</kinematic>\n\
                <gravity>1</gravity>\n\
              </link>\n\
            </model>\n\
            <gravity>0 0 -9.8</gravity>\n\
            <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>\n\
            <atmosphere type='adiabatic'/>\n\
            <physics name='default_physics' default='0' type='ode'>\n\
              <max_step_size>0.001</max_step_size>\n\
              <real_time_factor>1</real_time_factor>\n\
              <real_time_update_rate>1000</real_time_update_rate>\n\
            </physics>\n\
            <scene>\n\
              <ambient>0.4 0.4 0.4 1</ambient>\n\
              <background>0.7 0.7 0.7 1</background>\n\
              <shadows>0</shadows>\n\
            </scene>\n\
            <spherical_coordinates>\n\
              <surface_model>EARTH_WGS84</surface_model>\n\
              <latitude_deg>0</latitude_deg>\n\
              <longitude_deg>0</longitude_deg>\n\
              <elevation>0</elevation>\n\
              <heading_deg>0</heading_deg>\n\
            </spherical_coordinates>\n\
            <model name='my_wall_01'>\n\
              <pose frame=''>4.84148 4.89701 0 0 -0 0</pose>\n\
              <link name='Wall_0'>\n\
                <collision name='Wall_0_Collision'>\n\
                  <geometry>\n\
                    <box>\n\
                      <size>"+str(map_edge) + " 0.15 2.5</size>\n\
                    </box>\n\
                  </geometry>\n\
                  <pose frame=''>0 0 1.25 0 -0 0</pose>\n\
                  <max_contacts>10</max_contacts>\n\
                  <surface>\n\
                    <contact>\n\
                      <ode/>\n\
                    </contact>\n\
                    <bounce/>\n\
                    <friction>\n\
                      <torsional>\n\
                        <ode/>\n\
                      </torsional>\n\
                      <ode/>\n\
                    </friction>\n\
                  </surface>\n\
                </collision>\n\
                <visual name='Wall_0_Visual'>\n\
                  <pose frame=''>0 0 1.25 0 -0 0</pose>\n\
                  <geometry>\n\
                    <box>\n\
                      <size>"+str(map_edge) + " 0.15 2.5</size>\n\
                    </box>\n\
                  </geometry>\n\
                  <material>\n\
                    <script>\n\
                      <uri>file://media/materials/scripts/gazebo.material</uri>\n\
                      <name>Gazebo/Wood</name>\n\
                    </script>\n\
                    <ambient>1 1 1 1</ambient>\n\
                  </material>\n\
                </visual>\n\
                <pose frame=''>0.034695 -4.92526 0 0 0 -3.13923</pose>\n\
                <self_collide>0</self_collide>\n\
                <kinematic>0</kinematic>\n\
                <gravity>1</gravity>\n\
              </link>\n\
              <link name='Wall_1'>\n\
                <collision name='Wall_1_Collision'>\n\
                  <geometry>\n\
                    <box>\n\
                      <size>"+str(map_edge) + " 0.15 2.5</size>\n\
                    </box>\n\
                  </geometry>\n\
                  <pose frame=''>0 0 1.25 0 -0 0</pose>\n\
                  <max_contacts>10</max_contacts>\n\
                  <surface>\n\
                    <contact>\n\
                      <ode/>\n\
                    </contact>\n\
                    <bounce/>\n\
                    <friction>\n\
                      <torsional>\n\
                        <ode/>\n\
                      </torsional>\n\
                      <ode/>\n\
                    </friction>\n\
                  </surface>\n\
                </collision>\n\
                <visual name='Wall_1_Visual'>\n\
                  <pose frame=''>0 0 1.25 0 -0 0</pose>\n\
                  <geometry>\n\
                    <box>\n\
                      <size>"+str(map_edge) + " 0.15 2.5</size>\n\
                    </box>\n\
                  </geometry>\n\
                  <material>\n\
                    <script>\n\
                      <uri>file://media/materials/scripts/gazebo.material</uri>\n\
                      <name>Gazebo/Wood</name>\n\
                    </script>\n\
                    <ambient>1 1 1 1</ambient>\n\
                  </material>\n\
                </visual>\n\
                <pose frame=''>-4.91343 0.011303 0 0 -0 1.58019</pose>\n\
                <self_collide>0</self_collide>\n\
                <kinematic>0</kinematic>\n\
                <gravity>1</gravity>\n\
              </link>\n\
              <link name='Wall_2'>\n\
                <collision name='Wall_2_Collision'>\n\
                  <geometry>\n\
                    <box>\n\
                      <size>"+str(map_edge) + " 0.15 2.5</size>\n\
                    </box>\n\
                  </geometry>\n\
                  <pose frame=''>0 0 1.25 0 -0 0</pose>\n\
                  <max_contacts>10</max_contacts>\n\
                  <surface>\n\
                    <contact>\n\
                      <ode/>\n\
                    </contact>\n\
                    <bounce/>\n\
                    <friction>\n\
                      <torsional>\n\
                        <ode/>\n\
                      </torsional>\n\
                      <ode/>\n\
                    </friction>\n\
                  </surface>\n\
                </collision>\n\
                <visual name='Wall_2_Visual'>\n\
                  <pose frame=''>0 0 1.25 0 -0 0</pose>\n\
                  <geometry>\n\
                    <box>\n\
                      <size>"+str(map_edge) + " 0.15 2.5</size>\n\
                    </box>\n\
                  </geometry>\n\
                  <material>\n\
                    <script>\n\
                      <uri>file://media/materials/scripts/gazebo.material</uri>\n\
                      <name>Gazebo/Wood</name>\n\
                    </script>\n\
                    <ambient>1 1 1 1</ambient>\n\
                  </material>\n\
                </visual>\n\
                <pose frame=''>-0.057827 4.9363 0 0 -0 0</pose>\n\
                <self_collide>0</self_collide>\n\
                <kinematic>0</kinematic>\n\
                <gravity>1</gravity>\n\
              </link>\n\
              <link name='Wall_3'>\n\
                <collision name='Wall_3_Collision'>\n\
                  <geometry>\n\
                    <box>\n\
                      <size>"+str(map_edge) + " 0.15 2.5</size>\n\
                    </box>\n\
                  </geometry>\n\
                  <pose frame=''>0 0 1.25 0 -0 0</pose>\n\
                  <max_contacts>10</max_contacts>\n\
                  <surface>\n\
                    <contact>\n\
                      <ode/>\n\
                    </contact>\n\
                    <bounce/>\n\
                    <friction>\n\
                      <torsional>\n\
                        <ode/>\n\
                      </torsional>\n\
                      <ode/>\n\
                    </friction>\n\
                  </surface>\n\
                </collision>\n\
                <visual name='Wall_3_Visual'>\n\
                  <pose frame=''>0 0 1.25 0 -0 0</pose>\n\
                  <geometry>\n\
                    <box>\n\
                      <size>"+str(map_edge) + " 0.15 2.5</size>\n\
                    </box>\n\
                  </geometry>\n\
                  <material>\n\
                    <script>\n\
                      <uri>file://media/materials/scripts/gazebo.material</uri>\n\
                      <name>Gazebo/Wood</name>\n\
                    </script>\n\
                    <ambient>1 1 1 1</ambient>\n\
                  </material>\n\
                </visual>\n\
                <pose frame=''>4.91343 0.011303 0 0 0 -1.5614</pose>\n\
                <self_collide>0</self_collide>\n\
                <kinematic>0</kinematic>\n\
                <gravity>1</gravity>\n\
              </link>\n\
              <static>1</static>\n\
            </model>\n\
            <state world_name='default'>\n\
              <sim_time>1639 151000000</sim_time>\n\
              <real_time>4470 492795205</real_time>\n\
              <wall_time>1576044024 212025517</wall_time>\n\
              <iterations>1639151</iterations>\n\
              <model name='ground_plane'>\n\
                <pose frame=''>0 0 0 0 -0 0</pose>\n\
                <scale>1 1 1</scale>\n\
                <link name='link'>\n\
                  <pose frame=''>0 0 0 0 -0 0</pose>\n\
                  <velocity>0 0 0 0 -0 0</velocity>\n\
                  <acceleration>0 0 0 0 -0 0</acceleration>\n\
                  <wrench>0 0 0 0 -0 0</wrench>\n\
                </link>\n\
              </model>\n\
              <model name='my_wall_01'>\n\
                <pose frame=''>4.84148 4.89701 0 0 0 -0.007021</pose>\n\
                <scale>1 1 1</scale>\n\
                <link name='Wall_0'>\n\
                  <pose frame=''>" + str(map_edge/2) + " -0.028372 0 0 -0 3.13693</pose>\n\
                  <velocity>0 0 0 0 -0 0</velocity>\n\
                  <acceleration>0 0 0 0 -0 0</acceleration>\n\
                  <wrench>0 0 0 0 -0 0</wrench>\n\
                </link>\n\
                <link name='Wall_1'>\n\
                  <pose frame=''>-0.07175 "+str(map_edge/2) + " 0 0 -0 1.57317</pose>\n\
                  <velocity>0 0 0 0 -0 0</velocity>\n\
                  <acceleration>0 0 0 0 -0 0</acceleration>\n\
                  <wrench>0 0 0 0 -0 0</wrench>\n\
                </link>\n\
                <link name='Wall_2'>\n\
                  <pose frame=''>" + str(map_edge/2) + " " + str(map_edge) + " 0 0 0 -0.007021</pose>\n\
                  <velocity>0 0 0 0 -0 0</velocity>\n\
                  <acceleration>0 0 0 0 -0 0</acceleration>\n\
                  <wrench>0 0 0 0 -0 0</wrench>\n\
                </link>\n\
                <link name='Wall_3'>\n\
                  <pose frame=''>" + str(map_edge) + " " + str(map_edge/2) + " 0 0 0 -1.56842</pose>\n\
                  <velocity>0 0 0 0 -0 0</velocity>\n\
                  <acceleration>0 0 0 0 -0 0</acceleration>\n\
                  <wrench>0 0 0 0 -0 0</wrench>\n\
                </link>\n\
              </model>\n"
        # 开头部分model信息
        for i in range(cylinder_number):
            model_msg_0 = " <model name='c" + str(i) + "'>\n\
              <pose frame=''>" + str(c_value_list['c_value_x_list'][i]) + " " + str(c_value_list['c_value_y_list'][i]) + " 0 0 -0 0</pose>\n\
              <scale>1 1 1</scale>\n\
              <link name='link'>\n\
                <pose frame=''>" + str(c_value_list['c_value_x_list'][i]) + " " + str(
                c_value_list['c_value_y_list'][i]) + " 0 0 -0 0</pose>\n\
                <velocity>0 0 0 0 -0 0</velocity>\n\
                <acceleration>0 0 0 0 -0 0</acceleration>\n\
                <wrench>0 0 0 0 -0 0</wrench>\n\
              </link>\n\
            </model>\n"
            msg_1 = msg_1 + model_msg_0

        # ====== 新家正方体障碍物 =======
        for i in range(square_number):
            model_msg_0 = " <model name='s" + str(i) + "'>\n\
              <pose frame=''>" + str(s_value_list['s_value_x_list'][i]) + " " + str(
                s_value_list['s_value_y_list'][i]) + " 0 0 -0 " + str(s_yaw_list[i]) + "</pose>\n\
              <scale>1 1 1</scale>\n\
              <link name='link'>\n\
                <pose frame=''>" + str(s_value_list['s_value_x_list'][i]) + " " + str(
                s_value_list['s_value_y_list'][i]) + " 0 0 -0 " + str(s_yaw_list[i]) + "</pose>\n\
                <velocity>0 0 0 0 -0 0</velocity>\n\
                <acceleration>0 0 0 0 -0 0</acceleration>\n\
                <wrench>0 0 0 0 -0 0</wrench>\n\
              </link>\n\
            </model>\n"
            msg_1 = msg_1 + model_msg_0

        # ============================

        # 结束开头
        msg_2 = msg_1 + "<light name='sun'>\n\
                <pose frame=''>0 0 10 0 -0 0</pose>\n\
              </light>\n\
            </state>\n"
        # 另外部分model信息
        for i in range(cylinder_number):
            model_msg_1 = "<model name='c" + str(i) + "'>\n\
              <link name='link'>\n\
                <pose frame=''>0 0 0 0 -0 0</pose>\n\
                <inertial>\n\
                  <mass>0.284858</mass>\n\
                  <inertia>\n\
                    <ixx>0.0288096</ixx>\n\
                    <ixy>0</ixy>\n\
                    <ixz>0</ixz>\n\
                    <iyy>0.0288096</iyy>\n\
                    <iyz>0</iyz>\n\
                    <izz>0.010143</izz>\n\
                  </inertia>\n\
                  <pose frame=''>0 0 0 0 -0 0</pose>\n\
                </inertial>\n\
                <self_collide>0</self_collide>\n\
                <kinematic>0</kinematic>\n\
                <gravity>1</gravity>\n\
                <visual name='visual'>\n\
                  <geometry>\n\
                    <cylinder>\n\
                      <radius>" + str(model_radius[i]) + "</radius>\n\
                      <length>1</length>\n\
                    </cylinder>\n\
                  </geometry>\n\
                  <material>\n\
                    <script>\n\
                      <uri>file://media/materials/scripts/gazebo.material</uri>\n\
                      <name>Gazebo/Wood</name>\n\
                    </script>\n\
                    <ambient>1 1 1 1</ambient>\n\
                    <shader type='vertex'>\n\
                      <normal_map>__default__</normal_map>\n\
                    </shader>\n\
                  </material>\n\
                  <pose frame=''>0 0 0 0 -0 0</pose>\n\
                  <cast_shadows>0</cast_shadows>\n\
                  <transparency>0</transparency>\n\
                </visual>\n\
                <collision name='collision'>\n\
                  <laser_retro>0</laser_retro>\n\
                  <max_contacts>10</max_contacts>\n\
                  <pose frame=''>0 0 0 0 -0 0</pose>\n\
                  <geometry>\n\
                    <cylinder>\n\
                      <radius>" + str(model_radius[i]) + "</radius>\n\
                      <length>1</length>\n\
                    </cylinder>\n\
                  </geometry>\n\
                  <surface>\n\
                    <friction>\n\
                      <ode/>\n\
                      <torsional>\n\
                        <ode/>\n\
                      </torsional>\n\
                    </friction>\n\
                    <bounce/>\n\
                    <contact>\n\
                      <ode/>\n\
                    </contact>\n\
                  </surface>\n\
                </collision>\n\
              </link>\n\
              <static>1</static>\n\
              <allow_auto_disable>1</allow_auto_disable>\n\
              <pose frame=''>" + str(c_value_list['c_value_x_list'][i]) + " " + str(c_value_list['c_value_y_list'][i]) + " 0 0 -0 0</pose>\n\
            </model>\n"
            msg_2 = msg_2 + model_msg_1

        # ========== 新加入正方体障碍 ========
        for i in range(square_number):
            model_msg_1 = "<model name='s" + str(i) + "'>\n\
            <!--pose>" + str(s_value_list['s_value_x_list'][i]) + " " + str(s_value_list['s_value_y_list'][i]) + " " + "0 0 -0 0</pose-->\n\
            <pose>" + str(s_value_list['s_value_x_list'][i]) + " " + str(
                s_value_list['s_value_y_list'][i]) + " 0 0 -0 " + str(s_yaw_list[i]) + "</pose>\n\
            <link name='s_0'>\n\
                <collision name='s_0_Collision'>\n\
                    <geometry>\n\
                        <box>\n\
                            <size>" + str(edge1[i]) + " " + str(edge2[i]) + " " + "2.5</size>\n\
                        </box>\n\
                    </geometry>\n\
                    <pose>0 0 1.25 0 -0 0</pose>\n\
                    <max_contacts>10</max_contacts>\n\
                </collision>\n\
                <visual name='s_0_Visual'>\n\
                    <pose>0 0 1.25 0 -0 0</pose>\n\
                    <geometry>\n\
                        <box>\n\
                            <size>" + str(edge1[i]) + " " + str(edge2[i]) + " " + "2.5</size>\n\
                        </box>\n\
                    </geometry>\n\
                    <material>\n\
                        <script>\n\
                            <uri>file://media/materials/scripts/gazebo.material</uri>\n\
                            <name>Gazebo/Wood</name>\n\
                        </script>\n\
                        <ambient>1 1 1 1</ambient>\n\
                    </material>\n\
                    <meta>\n\
                        <layer>0</layer>\n\
                    </meta>\n\
                </visual>\n\
                <pose>0 0 0 0 -0 0</pose>\n\
                <self_collide>0</self_collide>\n\
                <enable_wind>0</enable_wind>\n\
                <kinematic>0</kinematic>\n\
            </link>\n\
            <static>1</static>\n\
        </model>\n"
            msg_2 = msg_2 + model_msg_1
        # =================================

        # 结尾部分
        msg = msg_2 + "<gui fullscreen='0'>\n\
              <camera name='user_camera'>\n\
                <pose frame=''>7.46544 4.75826 14.2026 4e-06 1.5658 1.56416</pose>\n\
                <view_controller>orbit</view_controller>\n\
                <projection_type>perspective</projection_type>\n\
              </camera>\n\
            </gui>\n\
          </world>\n\
        </sdf>\n\
        "
        # 保存文件
        # save_dir = './world_c' + str(cylinder_number) + 's' + str(square_number) + '/'

        save_dir = "worldFolder/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        world_id = find_max_world_id(save_dir) + 1
        print("current map id : {:0>4d}".format(world_id))

        npy_save_dir = "numpyMapFolder/"
        if not os.path.exists(npy_save_dir):
            os.mkdir(npy_save_dir)
        np.save(npy_save_dir + "map{:0>4d}.npy".format(world_id), emptyImage3)

        dx = int(2/resolution)
        for i in range(1, int(map_shape/dx), 1):
            cv2.line(img=emptyImage3, pt1=(dx * i, 0), pt2=(dx*i, map_shape - 1), color=1, thickness=1)
            cv2.line(img=emptyImage3, pt1=(0, dx * i), pt2=(map_shape-1, dx*i), color=1, thickness=1)

        cv2.imwrite(save_dir + "map{:0>4d}.png".format(world_id), emptyImage3)
        with open(save_dir + "map{:0>4d}.world".format(world_id), 'w') as f:
            f.write(msg)



