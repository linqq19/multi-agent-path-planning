class EnvAgent:
    """环境代理，各种机器人环境实代理的父类"""
    def __init__(self, robots):
        self.robots = robots

    # 雷达call_back函数
    def laser_callback(self, data, agent_id):
        pass

    # 获取机器人的姿态
    def odom_callback(self, data, agent_id):
        pass

    # 控制小车运动
    def control(self, robot_id, velocity, angular_velocity):
        pass




