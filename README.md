# multi-agent-path-planning

Project Architecture：

![Architecture](https://github.com/linqq19/multi-agent-path-planning/assets/54255402/e698c6f6-a7fd-4176-a499-9117c22da028)

Main Functions:

1. Generate random maps and nessary files for ROS:
   ~~~ python
   python3 main.py
   ~~~
2. Generate Expert Data:
   ~~~ python
   python3 ExpertDataGen/datagenerator.py
   ~~~
3. Distributed Path Planning test  in ROS:
   ~~~ Bash
   ./run.sh
   ~~~
4. close gazebo and clear all ROS　topics:
   ~~~ Bash
   ./clc.sh
   ~~~
