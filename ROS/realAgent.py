from envAgent import agent
from geometry_msgs.msg import Twist, Point, Quaternion  # 从 geometry_nsg 中导入Twist消息类型
import math


class RealAgent(agent.EnvAgent):
    def __init__(self, robot):
        super(RealAgent, self).__init__([robot])

    # 雷达call_back函数
    def laser_callback(self, data, agent_id):
        self.robots[0].extract_laser_data(data)

    # 获取机器人的姿态
    def odom_callback(self, data, agent_id):
        pass  # 实际机器人无机器人的相对位置信息

    # 控制小车运动
    def control(self, robot_id, velocity, angular_velocity):
        pass
