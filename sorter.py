import os
import configure
import scipy.io as sio
import torch
import shutil
import numpy as np

# 将launch文件按各机器人初始位置是否存在邻居进行划分，拷贝之各对应子文件夹
class LaunchSorter():
    def __init__(self):
        self.config = configure.get_config()
        self.num_robots = self.config.num_agents
        self.expert_data_root = "expertData/"
        self.launch_root = "launchFolder/"
        self.new_root = self.create_copy_target_folders()

    def create_copy_target_folders(self):
        new_root = "sorted_launch/"
        if not os.path.exists(new_root):
            os.mkdir(new_root)
        for i in range(self.num_robots):
            child_folder = new_root+"robot_{}_runnable".format(i)
            if not os.path.exists(child_folder):
                os.makedirs(child_folder)
        return new_root

    def search_target_data_files(self):
        # make a list of file name of input yaml
        list_path = []
        assert os.path.isdir(self.expert_data_root), '{} is not a valid directory'.format(dir)

        for root, _, fnames in sorted(os.walk(self.expert_data_root)):
            for fname in fnames:
                if fname.endswith(".mat"):
                    path = os.path.join(root, fname)
                    list_path.append(path)
        return list_path

    def load_data(self, path):
        data_contents = sio.loadmat(path)
        map_channel = data_contents['global_map']  # W x H

        input_tensor = data_contents['percept_info']
        target_sequence = data_contents['score']  # step x num_agent x 5
        input_GSO_sequence = data_contents['adj_matrix']  # Step x num_agent x num_agent
        case = data_contents["case"]
        tensor_map = torch.from_numpy(map_channel).float()
        step_input_tensor = torch.from_numpy(input_tensor[:]).float()
        step_input_GSO = torch.from_numpy(input_GSO_sequence[:, :]).float()
        step_target = torch.from_numpy(target_sequence[:, :]).long()
        # step_input_tensor = self.trans_percept(input_tensor, case)
        # cv2.imwrite("local_map.png",input_tensor[0][0,:]*255)
        return step_input_tensor, step_target, step_input_GSO, tensor_map


    def sort_launch_files(self):
        exp_data_list = self.search_target_data_files()
        for data_name in exp_data_list:
            map_id = int(data_name.split("/")[-1].split("robots")[0].split("map")[-1])
            launch_file = "map{:0>4d}robots{:0>2d}case01.launch".format(map_id, self.num_robots)
            data_path = data_name
            _, _, GSO, _ = self.load_data(data_path)
            num_ngbors = np.sum(GSO.numpy(), axis=1, dtype=np.int32)
            for i in range(self.num_robots):
                if num_ngbors[i] > 0:
                    shutil.copy(self.launch_root+launch_file, self.new_root + "robot_{}_runnable".format(i))


class ResultSorter:
    def __init__(self, result_folder="result/"):
        self.result_folder = result_folder
        self.target_lower_n20 = self.result_folder + "lower_n20/"
        self.target_lower_n10 = self.result_folder + "lower_n10/"
        self.target_higher_p10 = self.result_folder + "higher_p10/"
        self.target_higher_p20 = self.result_folder + "higher_p20/"
        self.target_not_evident = self.result_folder + "not_evident/"
        self.has_turn_launch_folder = "has_turn_launch_folder/"
        self.create_copy_target_folders()


    def create_copy_target_folders(self):
        need_folders = [self.target_lower_n20, self.target_lower_n10, self.target_higher_p10,
                        self.target_higher_p20, self.target_not_evident, self.has_turn_launch_folder]
        for folder in need_folders:
            if os.path.exists(folder):
                 shutil.rmtree(folder)
            os.makedirs(folder)

    def search_target_file(self):
        list_path = []
        assert os.path.isdir(self.result_folder), '{} is not a valid directory'.format(dir)

        for root, _, fnames in sorted(os.walk(self.result_folder)):
            for fname in fnames:
                if fname.endswith(".npz"):
                    path = os.path.join(root, fname)
                    list_path.append(path)
        return list_path

    def move_result_files(self, target_folder, src_file):
        shutil.copy(src_file, target_folder)
        test_name = src_file.split("/")[-1].split("_test")[0]
        shutil.copy(self.result_folder+test_name+"_opt_result.png", target_folder)
        shutil.copy(self.result_folder+test_name+"_pure_result.png", target_folder)

    def copy_launch_folder(self, target_folder, src_file):
        # 根据结果文件 src_file 格式， 将对应的launch文件复制到 target_folder 文件夹
        # src_file format: map0001robots15case01_test_result.npz
        # launch file format : map0001robots15case01.launch
        launch_file = "launchFolder/" + src_file.split("/")[-1].split("_test")[0] + ".launch"
        shutil.copy(launch_file, target_folder)

    def sort_result_folder(self, reach_radius=0.76):
        files = self.search_target_file()
        for result_file in files:
            print(result_file)
            data = np.load(result_file, allow_pickle=True)

            if np.linalg.norm(data["opt_end_position"] - data["start_points"]) < 0.1 or \
                    np.linalg.norm(data["pure_end_position"] - data["start_points"]) < 0.1:
                continue

            # 使用优化算法的结果统计
            opt_state = reach_radius - np.linalg.norm(data["opt_end_position"] - data["goal_points"], axis=1)
            opt_state[opt_state >= 0] = 1
            opt_state[opt_state < 0] = 0

            # 未使用优化算法的结果统计
            pure_state = reach_radius - np.linalg.norm(data["pure_end_position"] - data["goal_points"], axis=1)
            pure_state[pure_state >= 0] = 1
            pure_state[pure_state < 0] = 0

            both_reach = np.multiply(opt_state, pure_state)
            # opt_ratio = 1-case_opt_length/case_pure_length
            if np.sum(both_reach) != 0:
                opt_ratio = 1 - np.inner(both_reach, data["opt_path_length"]) / np.inner(both_reach,
                                                                                         data["pure_path_length"])
            else:
                opt_ratio = 0
                print(result_file + " all not reach in two cases")
            # print(" opt_length = ( 1 - opt_ratio ) * pure_length")
            print("cur_map : {}, opt_ratio: {:.2f}, num_reach: opt : {} vs pur : {}" \
                  .format(result_file, opt_ratio, np.sum(opt_state), np.sum(pure_state)))

            if opt_ratio > 0.2:
                self.move_result_files(self.target_higher_p20, result_file)
            elif opt_ratio > 0.1:
                self.move_result_files(self.target_higher_p10, result_file)
            elif opt_ratio < -0.2:
                self.move_result_files(self.target_lower_n20, result_file)
            elif opt_ratio < -0.1:
                self.move_result_files(self.target_lower_n10, result_file)
            else:
                self.move_result_files(self.target_not_evident, result_file)

            if data["opt_turn_dir_robots"].shape[0] > 0 :   # 将对应的launch 文件拷贝到新的文件夹备用
                self.copy_launch_folder(self.has_turn_launch_folder, result_file)


if __name__ == "__main__":
    def test0():
        sorter = LaunchSorter()
        sorter.sort_launch_files()

    def test1():
        result_sorter = ResultSorter(result_folder="result_singletest/")
        result_sorter.sort_result_folder()

    test1()