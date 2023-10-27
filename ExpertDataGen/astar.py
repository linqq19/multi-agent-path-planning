"""
A_star 2D
@author: huiming zhou
"""

import os
import sys
import math
import heapq
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Search_based_Planning/")

# from Search_2D import plotting, env
import plotting
import map
import time


class AStar:
    """AStar set the cost + heuristics as the priority
    """
    def __init__(self,g_map, s_start, s_goal, heuristic_type):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type

        self.Env = g_map  # class Env

        self.u_set = self.Env.motions  # feasible input set
        self.obs = self.Env.obs  # position of obstacles

        self.OPEN = []  # priority queue / OPEN set
        self.CLOSED = []  # CLOSED set / VISITED order
        self.PARENT = dict()  # recorded parent
        self.g = dict()  # cost to come

    def searching(self):
        """
        A_star Searching.
        :return: path, visited order
        """

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        """
        repeated A*.
        :param e: weight of A*
        :return: path and visited order
        """

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        """
        run A* with weight e.
        :param s_start: starting state
        :param s_goal: goal state
        :param e: weight of a*
        :return: path and visited order.
        """

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  # conditions for updating Cost
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        """
        find neighbors of state s that not in obstacles.
        :param s: state
        :return: neighbors
        """

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """
        check if the line segment (s_start, s_end) is collision.
        :param s_start: start node
        :param s_end: end node
        :return: True: is collision / False: not collision
        """

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        """
        f = g + h. (g: Cost to come, h: heuristic value)
        :param s: current state
        :return: f
        """

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        """
        Calculate heuristic.
        :param s: current node (state)
        :return: heuristic function value
        """

        heuristic_type = self.heuristic_type  # heuristic type
        goal = self.s_goal  # goal node

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

    def check_path_len(self, path):  # 计算路径长度
        if len(path) <= 1:
            return 0
        length = 0
        for i in range(len(path)-1):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]), 2)
        return length

    def print_path(self,path):
        for point in path:
            print("({},{}),obs:{} -->".format(point[0],point[1],
                self.Env.map[point[0]][point[1]]))

    def run(self):
        if self.Env.is_dangerous(self.s_start[0],self.s_start[1]) \
            or self.Env.is_dangerous(self.s_goal[0],self.s_goal[1]):
            return -1
        path, _ = self.searching()
        astar_len = self.check_path_len(path)
        return astar_len

def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    astar = AStar(s_start, s_goal, "euclidean")
    plot = plotting.Plotting(s_start, s_goal)

    path, visited = astar.searching()
    plot.animation(path, visited, "A*")  # animation

    # path, visited = astar.searching_repeated_astar(2.5)               # initial weight e = 2.5
    # plot.animation_ara_star(path, visited, "Repeated A*")


def test0():
    npy_files = []
    for _, _, filenames in os.walk("numpyMapFolder"):
        for filename in filenames:
            if filename.endswith(".npy"):
                npy_files.append(filename)
    for map_npy in npy_files:
        map_matrix = np.load('numpyMapFolder/'+map_npy)
        print(map_npy)
        map_matrix[map_matrix == 255] = 0
        map_matrix[0][0] = 1

        # map_matrix = np.rot90(map_matrix, k=-1)
        map_t = map.Map(map_matrix, 0.05)
        for i in range(10):
            # map_t = map.load_npy_map('numpyMapFolder/'+map_npy, 0.04)
            case = map_t.cases_generator(1,1,0.15)[0][0]
            print("start_point: {}".format(case[0]))
            print("end_point: {}".format(case[1]))
            start_mx = map_t.matrix_idx(case[0][0], case[0][1])
            end_mx = map_t.matrix_idx(case[1][0],case[1][1])
            # start_mx = map_t.matrix_idx(6, 7)
            # end_mx = map_t.matrix_idx(9, 5)

            start_mx = (147,186)
            end_mx = (71,380)

            print("start_point_mx: {}".format(start_mx))
            print("end_point_mx: {}".format(end_mx))

            # start = Point(start_mx[0], start_mx[1])
            # end = Point(end_mx[0], end_mx[1])

            a_star = AStar(map_t, start_mx, end_mx,"euclidean")
            plot = plotting.Plotting(start_mx, end_mx, map_t)

            start_time = time.time()
            path, _ = a_star.searching()
            # a_star.print_path(path)
            astar_len = a_star.check_path_len(path)
            time_cost = time.time() - start_time
            print("map： {} ， path length :  {length}  , use time {cost}".format(map_npy,length=astar_len, cost=time_cost))

            # plot.animation(path, visited, "A*")  # animation

            # astar_len = a_star.run()
            break
        break

if __name__ == '__main__':
    # main()
    test0()
