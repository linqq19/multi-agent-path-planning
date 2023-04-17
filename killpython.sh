# kill python processes
ps -ef | grep -v grep | grep python | grep -v root | awk '{print $2}' | xargs kill -9
nvidia-smi | grep -v grep |grep python | awk '{print $5}' | xargs kill -9

