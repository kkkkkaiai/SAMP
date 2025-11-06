#!/usr/bin/env python3

import torch 
import numpy as np 
import h5py
import time
from torch.utils.data import Dataset


def generate_anchor_points(n_x, n_y, n_z, max_x, max_y, max_z):
    x = np.linspace(-max_x, max_x, n_x)
    y = np.linspace(-max_y, max_y, n_y)
    z = np.linspace(0, max_z, n_z)
    x, y, z = np.meshgrid(x, y, z, indexing="ij")

    position_arr = np.zeros((n_x * n_y * n_z, 3))
    position_arr[:, 0] = x.flatten()
    position_arr[:, 1] = y.flatten() # offset y to be above the ground
    position_arr[:, 2] = z.flatten() + 0.05
    
    return position_arr

class CustomDataset(Dataset):
    def __init__(self, traj_datapath:str = './diverse_processed_dataset.hdf5', 
                 env_datapath:str = './diverse_processed_env.hdf5',
                 anchor_size:int = 100,
                 preload=True,
                 preload_to_gpu=False,
                 seed=42)->None:

        torch.manual_seed(seed)
        
        self.traj_dataset = traj_datapath
        self.env_dataset = env_datapath
        self.anchor_size = anchor_size
        self.device = 'cuda' if preload_to_gpu and torch.cuda.is_available() else 'cpu'
        
        # self.anchor_points = generate_anchor_points(self.anchor_size, self.anchor_size, self.anchor_size, 0.8, 0.8, 1.2)
        # self.anchor_points = torch.tensor(self.anchor_points).float()

        with h5py.File(self.traj_dataset, "r") as f:
            self.dataset_num = f['q_feature'].shape[0]
        
        self.preload = preload
        self.preload_to_gpu = preload_to_gpu
        
        if self.preload:
            start_time = time.time()
            print("Preloading data into memory...")

            self.env_data_cache = {}
            with h5py.File(self.env_dataset, "r") as f:
                max_env_id = 1000
                for env_id in range(max_env_id):
                    try:
                        env_key = f'env_{env_id}_{self.anchor_size}'
                        if env_key in f:
                            print(f"Preloading {env_key}")
                            tensor_data = torch.tensor(f[env_key][:], device=self.device)
                            self.env_data_cache[env_id] = tensor_data
                    except:
                        break

            print("Preloading trajectory data...")
            with h5py.File(self.traj_dataset, "r") as f:
                batch_size = 50000 
                self.q_feature_cache = []
                self.q_next_cache = []
                self.env_id_cache = []
                
                for i in range(0, self.dataset_num, batch_size):
                    end_idx = min(i + batch_size, self.dataset_num)
                    self.q_feature_cache.append(torch.tensor(f['q_feature'][i:end_idx], device=self.device))
                    self.q_next_cache.append(torch.tensor(f['q_next'][i:end_idx], device=self.device))
                    self.env_id_cache.append(torch.tensor(f['env_id'][i:end_idx], device=self.device))

                self.q_feature_cache = torch.cat(self.q_feature_cache)
                self.q_next_cache = torch.cat(self.q_next_cache)
                self.env_id_cache = torch.cat(self.env_id_cache)
                
            load_time = time.time() - start_time
            print(f"Preloaded {len(self.env_data_cache)} environments and {self.dataset_num} trajectory points in {load_time:.2f} seconds")
        else:

            print("Using memory-mapped mode for data access")
            self.env_data_cache = {}

            with h5py.File(self.env_dataset, "r") as f:
                max_env_id = 1000

                for env_id in range(max_env_id):
                    try:
                        env_key = f'env_{env_id}'
                        if env_key in f:
                            self.env_data_cache[env_id] = torch.tensor(f[env_key][:])
                    except:
                        break

            self.traj_file = h5py.File(self.traj_dataset, 'r')
            self.q_feature_dset = self.traj_file['q_feature']
            self.q_next_dset = self.traj_file['q_next']
            self.env_id_dset = self.traj_file['env_id']

    def __del__(self):
        if not self.preload and hasattr(self, 'traj_file'):
            self.traj_file.close()

    def set_joint_limits(self, limits):
        self.joint_limits = limits

    def get_inputs(self, idx, rand_scale=0.015):
        item = {}

        q_feature = self.q_feature_dset[idx].copy()
        q_next = self.q_next_dset[idx]
        env_id = int(self.env_id_dset[idx])

        randomize_q = rand_scale * torch.randn(7) + q_feature[:7]
        limits = torch.as_tensor(self.joint_limits).float()
        randomize_q = torch.clamp(randomize_q, limits[:, 0], limits[:, 1])
        q_feature[:7] = randomize_q

        item["q_feature"] = torch.tensor(q_feature)
        item["q_next"] = torch.tensor(q_next)
        item["env_id"] = torch.tensor(env_id)
        
        return item

    def __len__(self):
        return self.dataset_num

    def __getitem__(self, idx):
        start_time = time.time()
        
        if self.preload:
            q_feature = self.q_feature_cache[idx].clone()
            q_next = self.q_next_cache[idx].clone()
            env_id = self.env_id_cache[idx].item()
            
            randomize_q = 0.03 * torch.randn(7, device=q_feature.device) + q_feature[:7]
            limits = torch.as_tensor(self.joint_limits, device=q_feature.device).float()
            randomize_q = torch.clamp(randomize_q, limits[:, 0], limits[:, 1])
            q_feature[:7] = randomize_q
            
            item = {
                "q_feature": q_feature,
                "q_next": q_next,
                "env_id": torch.tensor(env_id, device=q_feature.device),
                "obs_feature": self.env_data_cache[env_id]
            }
        else:
            item = self.get_inputs(idx)
            env_id = item["env_id"].item()
            
            if env_id in self.env_data_cache:
                item["obs_feature"] = self.env_data_cache[env_id]
            else:
                with h5py.File(self.env_dataset, "r") as f:
                    env_data = f[f'env_{env_id}'][:]
                    item["obs_feature"] = torch.tensor(env_data)
        
        return item
    
    def data_sanity_check(self)->None:
        if self.preload:
            idx = 0
            data_sample = {
                "q_feature": self.q_feature_cache[idx].clone(),
                "q_next": self.q_next_cache[idx].clone(),
                "env_id": self.env_id_cache[idx].item()
            }
        else:
            data_sample = self.get_inputs(0)
        
        for key, item in data_sample.items():
            print(f'{key}: {type(item)}, shape: {item.shape if isinstance(item, torch.Tensor) else "scalar"}')
               

if __name__ == "__main__":
    start = time.time()
    dataset = CustomDataset(preload=True, preload_to_gpu=False)
    print(f"init time: {time.time() - start:.2f}s")
    dataset.data_sanity_check()

    start = time.time()
    num_samples = 1000
    for i in range(num_samples):
        _ = dataset[i % len(dataset)]