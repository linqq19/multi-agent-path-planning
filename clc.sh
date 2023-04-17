# kill ros pid
ps aux | grep ros |  awk '{print $2}' | xargs kill -9; ps aux | grep rviz |  awk '{print $2}' | xargs kill -9


## kill python processes
ps -ef | grep -v grep | grep python3 | grep -v root | awk '{print $2}' | xargs kill -9
nvidia-smi | grep -v grep |grep python3 | awk '{print $5}' | xargs kill -9

# 关闭Gazebo服务
killall -9 gzserver
killall -9 gzclient
killall -9 roscore
killall -9 rosmaster
