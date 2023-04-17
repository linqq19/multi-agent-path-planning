#!/bin/bash

# kill ros pid
ps aux | grep ros |  awk '{print $2}' | xargs kill -9; ps aux | grep rviz |  awk '{print $2}' | xargs kill -9


## kill python processes
#ps -ef | grep -v grep | grep python3 | grep -v root | awk '{print $2}' | xargs kill -9
#nvidia-smi | grep -v grep |grep python3 | awk '{print $5}' | xargs kill -9

# 关闭Gazebo服务
killall -9 gzserver
killall -9 gzclient
killall -9 roscore
killall -9 rosmaster

# 运行roscore
gnome-terminal -t "roscore" -x bash -c "roscore;exec bash;"
sleep 2s

# 加载机器人
num_robots=15
for (( i = 0; i < ${num_robots}; i++ ))
do
	  {
        python3 singleRobot.py -r ${i}
    }&
    sleep 3s
done

#sleep 2s
# 运行主程序
#python3 center.py


