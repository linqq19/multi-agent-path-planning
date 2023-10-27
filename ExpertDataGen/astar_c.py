import os
import numpy as np
import time
import map

import ctypes
from numpy.ctypeslib import ndpointer
class point(ctypes.Structure):
    _fields_ = [('x',ctypes.c_int),
              ('y',ctypes.c_int),]
class pair(ctypes.Structure):
    _fields_ = [('F',ctypes.c_float),
                ('P',point)]
test = ctypes.cdll.LoadLibrary("./libastar.so")
test.getshortest.restype = pair
test.getshortest.argtypes = [ndpointer(ctypes.c_int),point,point,ctypes.c_float ]


npy_files = []
for _, _, filenames in os.walk("numpyMapFolder"):
    for filename in filenames:
        if filename.endswith(".npy"):
            npy_files.append(filename)


print("test")

for map_npy in npy_files:
    map_matrix = np.load('numpyMapFolder/'+map_npy).astype(np.int32)
    print(map_npy)
    map_matrix[map_matrix == 255] = 0
    map_matrix[0][0] = 1

    map_t = map.Map(map_matrix, 0.05)
    np.savetxt("map.txt",map_t.safe_map,fmt="%1d")
    # map_t = map.load_npy_map('numpyMapFolder/'+map_npy, 0.04)
    case = map_t.cases_generator(1,1,0.15)[0][0]
    print("start_point: {}".format(case[0]))
    print("end_point: {}".format(case[1]))
    start_mx = map_t.matrix_idx(case[0][0], case[0][1])
    end_mx = map_t.matrix_idx(case[1][0],case[1][1])
    # start_mx = map_t.matrix_idx(6, 7)
    # end_mx = map_t.matrix_idx(9, 5)

    # start_mx = (147,186)
    # end_mx = (71,380)

    print("start_point_mx: {}".format(start_mx))
    print("end_point_mx: {}".format(end_mx))

    # start = Point(start_mx[0], start_mx[1])
    # end = Point(end_mx[0], end_mx[1])

    # a_star = AStar(map_t, start_mx, end_mx,"euclidean")

    load_time = time.time()
    astar_len = test.getshortest(map_t.safe_map,point(x=start_mx[0],y=start_mx[1]),point(x=end_mx[0],y=end_mx[1]),1.0)

    time_cost = time.time() - load_time
    print("map: {} , path length :  {length}  , use time {cost}".format(map_npy,length=astar_len.F, cost=time_cost))

    # plot.animation(path, visited, "A*")  # animation

    # astar_len = a_star.run()
    # break;

