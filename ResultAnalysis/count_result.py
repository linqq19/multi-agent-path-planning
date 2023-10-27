import os
import numpy as np
import torch
import sys
from utils import matrix2table
import multiRobot
import configure
import dhbug

def count_result(num_robots=15, reach_radius=0.76, result_folder_path = "result/", use_expert=True, test_one_robot=True):
    """
    统计所有测试的地图上的结果
    :param num_robots: 机器人数目
    :param reach_radius: 到达目的地的有效半径
    :param result_folder_path: 存放结果文件的文件夹路径
    :return:
    """
    result_files = []
    for _, _, filenames in os.walk(result_folder_path):
        for filename in filenames:
            if filename.endswith(".npz"):
                result_files.append(filename)
    total_opt_length, total_pure_length = 0,0
    total_opt_reach_length, total_pure_reach_length = 0, 0
    opt_reach_number, pure_reach_number = 0, 0
    total_opt_both_reach_length, total_pure_both_reanch_length = 0,0
    valid_num = len(result_files)  # 有效样本数
    num_use_opt = 0
    use_opt_length , use_pure_length = 0.00, 0.00  # 所有使用了优化器的机器人的路径长度两种情况下的对比
    use_opt_reach_length, use_pure_reach_length = 0.0, 0.0
    num_use_opt_reach = 0
    num_use_opt_both_reach = 0
    use_opt_both_reach_length, use_pure_both_reach_length = 0.0, 0.0
    num_over_twenty = 0
    num_lower_f_twenty = 0
    num_over_ten = 0
    num_lower_f_ten = 0
    if use_expert:
        total_epabs_length, total_eprel_length = 0, 0
        total_epabs_reach_length, total_eprel_reach_length = 0, 0 
        epabs_reach_number, eprel_reach_number = 0, 0

    for result_file in result_files:
        print(result_file)
        # result_file = "map0137robots15case01_test_result.npz" # for test
        data = np.load(result_folder_path + result_file, allow_pickle=True)
        # 先判断是否存在ros代理连上的情况 #在目标点与起始点位置很近的情况，可能存在问题。   需要判断每个机器人的代理是否连接上
        line_dist = np.linalg.norm(data["goal_points"]-data["start_points"],ord=2,axis=1)
        opt_journey = np.linalg.norm(data["opt_end_position"] - data["start_points"], ord=2,axis=1)
        pure_journey = np.linalg.norm(data["pure_end_position"] - data["start_points"],ord=2,axis=1)

        if np.linalg.norm(data["opt_end_position"] - data["start_points"]) < 0.1 or \
            np.linalg.norm(data["pure_end_position"] - data["start_points"]) < 0.1:
            valid_num -= 1
            continue

        # 使用优化算法的结果统计
        opt_state = reach_radius - np.linalg.norm(data["opt_end_position"] - data["goal_points"], axis=1)
        opt_state[opt_state >= 0] = 1
        opt_state[opt_state < 0] = 0
        if test_one_robot:
            opt_state[1:-1] = 0
        opt_reach_number += np.sum(opt_state)
        case_opt_length = np.inner(opt_state, data["opt_path_length"])
        total_opt_reach_length += np.inner(opt_state, data["opt_path_length"])
        total_opt_length += np.sum(data["opt_path_length"])
        # 未使用优化算法的结果统计
        pure_state = reach_radius - np.linalg.norm(data["pure_end_position"] - data["goal_points"], axis=1)
        pure_state[pure_state >= 0] = 1
        pure_state[pure_state < 0] = 0
        if test_one_robot:
            pure_state[1:-1] = 0
        case_pure_length =  np.inner(pure_state, data["pure_path_length"])
        total_pure_reach_length += np.inner(pure_state, data["pure_path_length"])
        pure_reach_number += np.sum(pure_state)
        total_pure_length += np.sum(data["pure_path_length"])

        both_reach = np.multiply(opt_state, pure_state)
        # opt_ratio = 1-case_opt_length/case_pure_length
        if np.sum(both_reach) != 0:
            opt_ratio = 1 - np.inner(both_reach, data["opt_path_length"])/np.inner(both_reach, data["pure_path_length"])
        else:
            opt_ratio = 0
            print(result_file+" all not reach in two cases")
        # print(" opt_length = ( 1 - opt_ratio ) * pure_length")
        print("cur_map : {}, opt_ratio: {:.2f}, num_reach: opt : {} vs pur : {}"\
            .format(result_file, opt_ratio, np.sum(opt_state), np.sum(pure_state)))
        total_opt_both_reach_length +=  np.inner(both_reach, data["opt_path_length"])
        total_pure_both_reanch_length += np.inner(both_reach, data["pure_path_length"])

        if opt_ratio >= 0.2 :
            num_over_twenty += 1
        elif 0.2 > opt_ratio >= 0.1 :
            num_over_ten += 1
        elif -0.2 <= opt_ratio < -0.1 :
            num_lower_f_ten += 1
        elif opt_ratio < -0.2 :
            num_lower_f_twenty += 1


        # 只统计使用了优化器的情况
        use_opt_id = set(data["opt_turn_dir_robots"])
        if len(use_opt_id)!= 0:
            use_opt_length += np.sum(np.array([data["opt_path_length"][id] for id in use_opt_id]))
            use_pure_length += np.sum(np.array([data["pure_path_length"][id] for id in use_opt_id]))
            num_use_opt += len(use_opt_id)
        
        # 使用优化器且opt到达
        use_opt_reach_id = []
        use_opt_both_reach_id = []
        for robot_id in use_opt_id:
            if opt_state[robot_id] == 1:
                use_opt_reach_id.append(robot_id)
                if pure_state[robot_id] == 1:
                    use_opt_both_reach_id.append(robot_id)
        if len(use_opt_reach_id)!= 0:
            use_opt_reach_length += np.sum(np.array([data["opt_path_length"][id] for id in use_opt_reach_id]))
            use_pure_reach_length += np.sum(np.array([data["pure_path_length"][id] for id in use_opt_reach_id]))
            num_use_opt_reach += len(use_opt_reach_id)

        if len(use_opt_both_reach_id)!= 0:
            use_opt_both_reach_length += np.sum(np.array([data["opt_path_length"][id] for id in use_opt_both_reach_id]))
            use_pure_both_reach_length += np.sum(np.array([data["pure_path_length"][id] for id in use_opt_both_reach_id]))
            num_use_opt_both_reach += len(use_opt_both_reach_id)


        if use_expert:  # 如果测试中使用了专家相关数据
            epabs_state = reach_radius - np.linalg.norm(data["epabs_end_position"] - data["goal_points"], axis=1)
            epabs_state[epabs_state >= 0] = 1
            epabs_state[epabs_state < 0] = 0
            eprel_state = reach_radius - np.linalg.norm(data["eprel_end_position"] - data["goal_points"], axis=1)
            eprel_state[eprel_state >= 0] = 1
            eprel_state[eprel_state < 0] = 0
            if test_one_robot:
                epabs_state[1:-1] = 0
                eprel_state[1:-1] = 0
            total_epabs_length += np.sum(data["epabs_path_length"])
            total_eprel_length += np.sum(data["eprel_path_length"])
            total_epabs_reach_length += np.inner(epabs_state, data["epabs_path_length"])
            total_eprel_reach_length += np.inner(eprel_state, data["eprel_path_length"])
            epabs_reach_number += np.sum(epabs_state)
            eprel_reach_number += np.sum(eprel_state)
            if np.sum(epabs_state) == 0 or np.sum(eprel_state) == 0:
                print("expert not reach, file : " + result_file)

    opt_reach_prop = opt_reach_number / (num_robots * valid_num)
    pure_reach_prop = pure_reach_number / (num_robots * valid_num)


    print("Total number of valid cases :{}".format(len(result_files)))
    print("           opt            vs              pure")
    print("总路程： {:.2f} vs {:.2f}".format(total_opt_length, total_pure_length))
    print("两者各自到达目标点的总路程： {:.2f} vs {:.2f}".format(total_opt_reach_length, total_pure_reach_length))
    print("到达数： {} vs {}".format(int(opt_reach_number), int(pure_reach_number)))
    print("到达率： {:.2f} vs {:.2f}".format(opt_reach_prop, pure_reach_prop))
    print("平均到达长度： {:.2f} vs {:.2f}".format(total_opt_reach_length/opt_reach_number,total_pure_reach_length/pure_reach_number))
    print("两者同时到达的总路程： {:.2f} vs {:.2f}".format(total_opt_both_reach_length, total_pure_both_reanch_length))
    print("使用到了优化器的路径数: {}".format(num_use_opt))
    print("使用了优化器的路径长度对比 : {:.2f} vs {:.2f}".format(use_opt_length, use_pure_length))
    print("使用优化器且到达的路径数： {}".format(num_use_opt_reach))
    print("使用优化器且到达的路径长度对比 : {:.2f} vs {:.2f}".format(use_opt_reach_length, use_pure_reach_length))
    print("使用优化器且两者均到达的路径数： {}".format(num_use_opt_both_reach))
    print("使用优化器且两者均到达的路径长度对比 : {:.2f} vs {:.2f}".format(use_opt_both_reach_length, use_pure_both_reach_length))
    print("opt ratio distribution -0.2 -0.1 0.1 0.2 : {}, {}, {}, {}".format(num_lower_f_twenty, num_lower_f_ten, num_over_ten, num_over_twenty))
    if use_expert:
        print("专家总路程 矫正vs不矫正 {:.2f} , {:.2f}".format(total_epabs_length, total_eprel_length))
        print("专家到达的总路程 矫正vs不矫正 {:.2f} , {:.2f}".format(total_epabs_reach_length, total_eprel_reach_length))
        print("专家到达数 矫正 vs 不矫正 {} , {}".format(int(epabs_reach_number), int(eprel_reach_number)))


def count_expert_result(num_robots=15, reach_radius=0.76, result_folder_path="expert_result/", test_one_robot=True):
    """
    统计所有测试的地图上的结果
    :param num_robots: 机器人数目
    :param reach_radius: 到达目的地的有效半径
    :param result_folder_path: 存放结果文件的文件夹路径
    :return:
    """

    # 全部结果文件
    result_files = []
    for _, _, filenames in os.walk(result_folder_path):
        for filename in filenames:
            if filename.endswith(".npz"):
                result_files.append(filename)

    # 结果数组初始化
    total_length = np.zeros(8)
    total_reach_length = np.zeros(8)
    reach_num = np.zeros(8)

    for result_file in result_files:
        print(result_file)
        data = np.load(result_folder_path + result_file, allow_pickle=True)
        # 先判断是否存在ros代理连上的情况 #在目标点与起始点位置很近的情况，可能存在问题。   需要判断每个机器人的代理是否连接上
        line_dist = np.linalg.norm(data["goal_points"] - data["start_points"], ord=2, axis=1)
        pure_journey = np.linalg.norm(data["pure_end_position"] - data["start_points"], ord=2, axis=1)

        if np.linalg.norm(data["pure_end_position"] - data["start_points"]) < 0.1:
            valid_num -= 1
            continue

        Flag = True
        for i in range(4,8,1):
            if data["expert{}_end_position".format(i)].tolist() is None:
                Flag = False
        if data["pure_end_position"] is None:
            Flag = False
        if data["eprel_end_position"] is None:
            Flag = False
        if data["epabs_end_position"] is None:
            Flag = False
        if not Flag:
            continue
        # pure
        pure_state = reach_radius - np.linalg.norm(data["pure_end_position"] - data["goal_points"], axis=1)
        pure_state[pure_state >= 0] = 1
        pure_state[pure_state < 0] = 0
        if test_one_robot:
            pure_state[1:-1] = 0
        # case_pure_length = np.inner(pure_state, data["pure_path_length"])
        total_reach_length[0] += np.inner(pure_state, data["pure_path_length"])
        reach_num[0] += np.sum(pure_state)
        total_length[0] += np.sum(data["pure_path_length"])

        # 优化器 2 & 3
        epabs_state = reach_radius - np.linalg.norm(data["epabs_end_position"] - data["goal_points"], axis=1)
        epabs_state[epabs_state >= 0] = 1
        epabs_state[epabs_state < 0] = 0
        eprel_state = reach_radius - np.linalg.norm(data["eprel_end_position"] - data["goal_points"], axis=1)
        eprel_state[eprel_state >= 0] = 1
        eprel_state[eprel_state < 0] = 0
        if test_one_robot:
            epabs_state[1:-1] = 0
            eprel_state[1:-1] = 0
            total_length[2] += np.sum(data["epabs_path_length"])
            total_length[3] += np.sum(data["eprel_path_length"])
            total_reach_length[2] += np.inner(epabs_state, data["epabs_path_length"])
            total_reach_length[3] += np.inner(eprel_state, data["eprel_path_length"])
            reach_num[2] += np.sum(epabs_state)
            reach_num[3] += np.sum(eprel_state)
            if np.sum(epabs_state) == 0 or np.sum(eprel_state) == 0:
                print("expert not reach, file : " + result_file)

        for i in range(4,8,1):
            expert_state =  reach_radius - np.linalg.norm(data["expert{}_end_position".format(i)] - data["goal_points"], axis=1)
            expert_state[expert_state >= 0] = 1
            expert_state[expert_state < 0] = 0
            if test_one_robot:
                expert_state[1:-1] = 0
            # case_pure_length = np.inner(pure_state, data["pure_path_length"])
            total_reach_length[i] += np.inner(expert_state, data["expert{}_path_length".format(i)])
            reach_num[i] += np.sum(expert_state)
            total_length[i] += np.sum(data["expert{}_path_length".format(i)])

    reach_num[1] = 0.1
    average_reach_path_length, average_path_length = np.ones_like(reach_num), np.ones_like(reach_num)
    np.divide(total_reach_length, reach_num, average_reach_path_length)
    np.divide(total_length, reach_num, average_path_length)


    print("Total number of valid cases :{}".format(len(result_files)))
    np.set_printoptions(precision=2, suppress=True)
    print("总   路   程 : ", total_length)
    print("到达的总路程  : ", total_reach_length)
    print("到   达   数 : ",  reach_num)
    print("平均 路径长   :", average_path_length)
    print("平均到达路径长 :", average_reach_path_length)
    print("\n")
    print("\n")

def reconstruct_percept(data):
    config = configure.get_config()
    robot = multiRobot.Robot(0,config)
    robot.planner = dhbug.DhPathPlanner_HPoint(robot, config)
    
    robot.scan_angles = data["laser_angles"].flatten()
    num_robots = data["opt_turn_dir_odoms"].shape[1]
    adjs = []
    inputs = []
    for i in range(data["opt_turn_dirs"].shape[0]):
        one_all_pcpt = []
        adj = np.zeros([num_robots, num_robots],dtype=np.int64)
        for j in range(num_robots):
            # 先恢复机器人的信息
            x, y, yaw = data["opt_turn_dir_odoms"][i,j]  # 存疑： 角度制？
            robot.position = np.array([x,y])
            robot.yaw = yaw
            robot.scan_distance = data["opt_turn_dir_lasers"][i,j].flatten()
            robot.set_goal(data["goal_points"][j])

            # 恢复邻居信息
            for k in range(num_robots):
                if k == j:
                    continue
                nghb_position = data["opt_turn_dir_odoms"][i,k,0:2]
                rel_dist, rel_angle = robot.global_to_relative(nghb_position)
                rel_angle_index = robot.find_laser_angle_index(rel_angle)
                if rel_dist <= min(robot.com_radius, robot.scan_distance[rel_angle_index]+robot.config.robotRadius+0.05):   # 邻居自身会对激光雷达的测距产生影响
                    robot.neighborhood[k].set_value(rel_dist, rel_angle, [], True)
                    adj[j,k] = 1
            # 再通过调用函数获取数据
            robot_pcpt = robot.get_map_info()
            one_all_pcpt.append(robot_pcpt)
            robot.reset_record()
            robot.scan_angles = data["laser_angles"].flatten()
        one_all_pcpt = np.stack(one_all_pcpt)
        inputs.append(one_all_pcpt)
        adjs.append(adj)
    inputs = np.stack(inputs)
    return inputs, adjs

def analysis_result_distributed(num_robots=4, reach_radius=0.76, result_folder_path = "analysis/"):
    result_files = []
    for _, _, filenames in os.walk(result_folder_path):
        for filename in filenames:
            if filename.endswith(".npz"):
                result_files.append(filename)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load("GNN/robots15_rm0_min_valid_loss_1n4_none.pkl", map_location=device)

    for result_file in result_files:
        # result_file = "map0016robots15_test_result.npz" # for test
        data = np.load(result_folder_path + result_file)
        input_npy, adjs = reconstruct_percept(data)
        input = torch.tensor(input_npy).float().to(device)
        decisions = torch.tensor(data["opt_turn_dirs"], dtype=torch.int32).to(device)
        adj_list = [matrix2table(matrix) for matrix in adjs]
        index = data["opt_turn_dir_robots"]
        
        # print("input done")
        # output = torch.tensor(data["decisions"]).float().to(device)

        test_ouput = model(input, adj_list, index)
        adj = [matrix2table(np.zeros([4,4])) for i in range(len(adj_list))]
        single_output = model(input, adj, index)
        print(result_file)
        print("test output:{}".format(data["opt_nn_outputs"]))
        print("record data output : {}".format(test_ouput.cpu().detach().numpy()))
        print("personal data output : {}".format(single_output.cpu().detach().numpy()))
        
        print("decisions : {}".format(data["opt_turn_dirs"]))
        print("path length : {}".format(data["opt_path_length"][0]))
        print("neighbour: {}".format(adjs[0][0]))

        print("turn position: {}".format(data["opt_turn_dir_odoms"][:,0]))
        print("================================================")


if __name__ == "__main__":
    sys.path.append("./GNN")
    # analysis_result_distributed(result_folder_path = "result/")
    # count_result(result_folder_path = "result_experttest/")

    count_expert_result()
