import pickle
import torch
import os
import numpy as np
import glob
import h5py
from tqdm import tqdm

dataset_dir = os.path.dirname(os.path.realpath(__file__)) + '/isaac_sim_traj'
dataset_files = sorted(glob.glob(os.path.join(dataset_dir, 'dataset*.pkl')))
env_files = sorted(glob.glob(os.path.join(dataset_dir, 'env*.pkl')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# check if dataset_files and env_files are not empty
if not dataset_files or not env_files:
    raise ValueError("No dataset or environment files found in the specified directory.")
# check if the number of dataset files matches the number of environment files
if len(dataset_files) != len(env_files):
    raise ValueError("The number of dataset files does not match the number of environment files.")

dataset_nums = len(dataset_files)
# each file has 2000 data, but we only choose 1000 samples
# each samples has 50 steps
data_choose_numbers = 200
all_q_feature = []
all_q_next = []
all_label = [] # target joint state
all_robot_feature = []
env_id_list = []


target_env_list = [1,2,3,4,6,7,8,9]


valid_indices = [i for i in range(len(dataset_files)) if (i + 1) in target_env_list]

for i in valid_indices:
    with open(dataset_files[i], 'rb') as f:
        data = pickle.load(f)
        for idx, item in enumerate(tqdm(data[:data_choose_numbers], desc=f"Processing dataset {i+1}")):
            joint_traj = torch.tensor(item['joint_trajectory'])[:, :-2]

            q_cur_all = joint_traj[:-1]
            q_g = joint_traj[-1]
            q_g_all = q_g.expand_as(q_cur_all)

            env_id_list.extend([i+1]*len(q_cur_all))
            q_feature = torch.cat([q_cur_all, q_g_all], dim=1) 
            q_next_all = joint_traj[1:]

            all_q_feature.append(q_feature)
            all_q_next.append(q_next_all)


# process trajectories data   
all_q_feature = torch.cat(all_q_feature, dim=0).reshape(-1, q_feature.shape[-1])
all_q_next = torch.cat(all_q_next, dim=0).reshape(-1, q_next_all.shape[-1])
env_id_array = np.array(env_id_list, dtype=np.int32)

print('env_id_length:', len(env_id_list))
print('q_feature_shape:', all_q_feature.shape)
print('q_next_shape:', all_q_next.shape)
(unique_ids, counts) = np.unique(env_id_array, return_counts=True)
print("env_id counts:", dict(zip(unique_ids, counts)))

traj_output_file = 'diverse_processed_dataset.hdf5'
with h5py.File(traj_output_file, 'w') as f:
    f.create_dataset('env_id', data=env_id_array, compression='gzip')
    f.create_dataset('q_feature', data=all_q_feature, compression='gzip')
    f.create_dataset('q_next', data=all_q_next, compression='gzip')

print(f"Trajectories are saved to {traj_output_file}")

# process environments data
env_output_file = 'diverse_processed_env.hdf5'
with h5py.File(env_output_file, 'w') as f:
    for i in tqdm(valid_indices, desc="Saving env files"):
        env_file = env_files[i]
        with open(env_file, 'rb') as ef:
            env_data = pickle.load(ef)
            for j in range(len(env_data)):
                for k, v in env_data[j].items():
                    # print(k.split('_')[-1])
                    f.create_dataset(f"env_{i+1}_{k.split('_')[-1]}", data=v, compression='gzip')

print(f"Environments are saved to {env_output_file}")


