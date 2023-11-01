import math
import collections

# 角度转换函数:将-360到360度的角度转到-180度到180度
def angle_transform(alpha):
    alpha = alpha % 360
    if alpha < -180:
        alpha = 360 + alpha
    if alpha > 180:
        alpha = -360 + alpha
    return alpha


def intersect_circle_ray(ray_start, ray_angle, circle_center, radius):
    """
    计算射线在圆内的线段:
    圆由圆心和半径定义，射线由起点和全局方向角定义
    当有两个交点时，
        返回的第一个点为离射线起点最近的交点，第二个点为离射线起点最远的交点
    当射线起点在圆内时，
        返回的第一个点为射线起点，第二个点为射线与圆的交点
    当射线或其所在的直线与圆相切时，返回None
    当无交点时，返回None

    ray_angle : [0-360)， 全局坐标系, x轴正方向为0°，逆时针
    ray_start ： 射线起点， 全局坐标系定义
    circle_center : 圆心， 全局坐标系定义
    radius ： 半径
    :return: None或者[重合线段起点，重合线段终点]  全局坐标系
    """
    if ray_angle >= 360 or ray_angle < 0:
        ray_angle = ray_angle % 360

    xr, yr = ray_start
    xc, yc = circle_center

    if ray_angle == 90 or ray_angle == 270:  # 射线与y轴平行
        if abs(xc - xr) < radius:  # 射线所在直线与圆有交点
            y_high = yc + math.sqrt(pow(radius, 2) - pow(xr - xc, 2))
            y_low = yc - math.sqrt(pow(radius, 2) - pow(xr - xc, 2))
            if ray_angle == 90 and yr < y_low:
                return True, [(xr, y_low), (xr, y_high)]
            if ray_angle == 90 and y_low < yr < y_high:
                return True, [(xr, yr), (xr, y_high)]
            if ray_angle == 270 and yr > y_high:
                return True, [(xr, y_high), (xr, y_low)]
            if ray_angle == 270 and y_low < yr < y_high:
                return True, [(xr, yr), (xr, y_low)]

    else:  # 其他情况，利用直线和圆的方程联立求解，进而判断交点情况
        # 射线所在直线方程转化为 y = kx + b
        k = math.tan(ray_angle * math.pi / 180)
        b = yr - k * xr
        # 联立直线方程 y = kx + b 和 圆方程 （x-xc）^2 + (y - yc)^2 = radius^2 得一元二次方程ax^2+dx+c=0
        a = pow(k, 2) + 1
        d = 2 * k * (b - yc) - 2 * xc
        c = pow(xc, 2) + pow(b-yc, 2) - pow(radius, 2)
        delta = pow(d, 2) - 4 * a * c
        if delta > 0:  # 若射线对应的直线有交点，则进一步判断射线的交点情况
            x_isec1 = (-1 * d - math.sqrt(delta)) / (2 * a)
            x_isec2 = (-1 * d + math.sqrt(delta)) / (2 * a)
            y_isec1 = k * x_isec1 + b
            y_isec2 = k * x_isec2 + b
            if ray_angle < 90 or ray_angle > 270:  # 射线方向与x轴正半轴夹角小于90°
                if xr < x_isec1:
                    return True, [(x_isec1, y_isec1), (x_isec2, y_isec2)]
                elif x_isec1 < xr < x_isec2:
                    return True, [(xr, yr), (x_isec2, y_isec2)]
            else:
                if xr > x_isec2:
                    return True, [(x_isec2, y_isec2), (x_isec1, y_isec1)]
                elif x_isec1 < xr < x_isec2:
                    return True, [(xr, yr), (x_isec1, y_isec1)]

    return False, None  # 其他情况均无交点


# 向量夹角计算函数: 返回值范围0到180度
def angle_between_vector(x1, y1, x2, y2):
    if math.pow(x1, 2) + math.pow(y1, 2) == 0 or math.pow(x2, 2) + math.pow(y2, 2) == 0:
        return 0
    else:
        return math.acos((x1 * x2 + y1 * y2) / (
                    math.sqrt(pow(x1, 2) + pow(y1, 2)) * math.sqrt(pow(x2, 2) + pow(y2, 2)))) * 180 / math.pi


# 计算机器人坐标系下向量与目标方向之间的夹角， 范围是 -180 到 +180
# angle : 机器人坐标系下的角度
def vector_angle(vector, angle):
    x, y = vector
    # (x,y)为输入向量， angle为机器人坐标系下的方位角
    if x == 0:  # 90度和-90度情况
        if y >= 0:
            result = 0 - angle  # 取值范围 - 180 到 180
        else:
            result = 180 - angle  # 取值范围 0 到 360
    else:
        if x > 0:  # 右半平面情况
            result = math.atan2(y, x) * 180 / math.pi - angle  # 取值范围 - 180 到 360
        else:  # 左半平面情况
            result = math.atan2(y, x) * 180 / math.pi - angle  # 取值范围 -360 到 180

    # 将角度换算到 - 180度到180之间
    if result > 180:
        result -= 360
    elif result <= -180:
        result += 360
    return result


def matrix2table(martrix):
    """输入图的邻接矩阵，输出邻接表"""
    result = collections.defaultdict(list)
    N = len(martrix)
    for i in range(N):
        for j in range(N):
            if martrix[i][j] and i != j:
                result[i].append(j)
    return result
