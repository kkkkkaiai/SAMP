# -----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# This file is part of the RDF project.
# Copyright (c) 2023 Idiap Research Institute <contact@idiap.ch>
# Contributor: Yimming Li <yiming.li@idiap.ch>
# -----------------------------------------------------------------------------


import os
import numpy as np
import glob
import torch
import trimesh
import torch.nn.functional as F
import skimage
import open3d as o3d
import matplotlib.pyplot as plt
import time
import copy
import h5py
import matplotlib.pyplot as plt

# Multi-res Encoding
import hashencoder.encoding as encoding
import hashencoder.sdf_utils as sdf_utils

from torch import nn
from hashencoder.mlp import MLPRegression
from robot_layer.panda_layer import PandaLayer
from hashencoder.hashgrid import HashEncoder

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

class HashMLPModel(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels = 1, 
                 mlp_units = [128, 128], 
                 enc_method = 'hashgrid',
                 enc_kwargs = {},
                 device = 'cuda'):
        super().__init__()
        if enc_method == 'freq':
            self.encoder = encoding.Frequency(in_channels, **enc_kwargs)
            in_channels = self.encoder.output_dim
        elif enc_method == 'hashgrid':
            self.encoder = HashEncoder(input_dim=3, 
                            num_levels=32, 
                            level_dim=2, 
                            base_resolution=1, 
                            per_level_scale=2, 
                            log2_hashmap_size=19)
            in_channels = self.encoder.output_dim
        else:
            print(f'Disable encoding: {enc_method}')
            self.encoder = None

        self.mlp = MLPRegression(in_channels, out_channels, mlp_units, skips=[], act_fn=torch.nn.ReLU, nerf=False)

        

        if device == 'cuda':
            self.to(device)

    def forward(self, x):
        if self.encoder is not None:
            x = self.encoder(x)
        return self.mlp(x)


class HashNNSDF():
    def __init__(self,robot,lr=0.002, device='cuda'):
        self.device = device    
        self.robot = robot
        self.lr = lr
        self.model_path = os.path.join(CUR_DIR, 'models')
        self.panda_visual_mesh = None
        self.panda_collision_mesh = None
        self.model = HashMLPModel(3, 1, enc_method='hashgrid')

        self.joint_limits = [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973)
        ]

        self.infer_model = None
        self.points_list = None

    def train_hash_nn(self,epoches=2000):
        mesh_path = os.path.dirname(os.path.realpath(__file__)) + "/robot_layer/meshes/panda/*"
        mesh_files = glob.glob(mesh_path)
        mesh_files = sorted(mesh_files) #[1:] #except finger
        mesh_dict = {}

        dataset = {'link0':{}, 
                   'link1':{}, 
                   'link2':{}, 
                   'link3':{}, 
                   'link4':{}, 
                   'link5':{}, 
                   'link6':{}, 
                   'link7':{}, 
                   'link8':{}}
        
        dur = 3.0
        dataset_offset ={'link0':torch.tensor([dur, dur, 0.0]), 'link1':torch.tensor([dur, 0.0, 0.0]), 'link2':torch.tensor([dur, -dur, 0.0]),
                         'link3':torch.tensor([0.0, dur, 0.0]), 'link4':torch.tensor([0.0, 0.0, 0.0]), 'link5':torch.tensor([0.0, -dur, 0.0]),
                         'link6':torch.tensor([-dur, dur, 0.0]),'link7':torch.tensor([-dur, 0.0, 0.0]),'link8':torch.tensor([-dur, -dur, 0.0])}

        ## For sample data display
        point_near_data_set = np.empty((0, 3))
        point_random_data_set = np.empty((0, 3))
        sdf_near_data_set = np.empty((0,))
        sdf_random_data_set = np.empty((0,))

        os.makedirs(self.model_path, exist_ok=True)

        for i,mf in enumerate(mesh_files):
            mesh_name = mf.split('/')[-1].split('.')[0]
            link_name = mesh_name.split('_')[1]
            print('link_name: ',link_name)

            mesh = trimesh.load(mf)
            offset = mesh.bounding_box.centroid
            scale = np.max(np.linalg.norm(mesh.vertices-offset, axis=1))

            mesh_dict[i] = {}
            mesh_dict[i]['mesh_name'] = mesh_name
            mesh_dict[i]['offset'] = torch.from_numpy(np.copy(offset))
            mesh_dict[i]['scale'] = scale
            mesh_dict[i]['outer_offset'] = dataset_offset[link_name]

            # load data
            data = np.load(os.path.dirname(os.path.realpath(__file__))+f'/data/sdf_points/voxel_128_{mesh_name}.npy',allow_pickle=True).item()
            point_near_data = data['near_points']
            sdf_near_data = data['near_sdf']
            point_random_data = data['random_points']
            sdf_random_data = data['random_sdf']
            sdf_random_data[sdf_random_data <-1] = -sdf_random_data[sdf_random_data <-1]

            dataset[link_name]['point_near_data'] = np.array(point_near_data) + dataset_offset[link_name].numpy()
            dataset[link_name]['sdf_near_data'] = np.array(sdf_near_data)
            dataset[link_name]['point_random_data'] = np.array(point_random_data) + dataset_offset[link_name].numpy()
            dataset[link_name]['sdf_random_data'] = np.array(sdf_random_data)

            ## For sample data display
            point_near_data_set = np.concatenate((point_near_data_set, dataset[link_name]['point_near_data']), axis=0)
            point_random_data_set = np.concatenate((point_random_data_set, dataset[link_name]['point_random_data']), axis=0)
            sdf_near_data_set = np.concatenate((sdf_near_data_set, dataset[link_name]['sdf_near_data']), axis=0)
            sdf_random_data_set = np.concatenate((sdf_random_data_set, dataset[link_name]['sdf_random_data']), axis=0)

        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                        threshold=0.01, threshold_mode='rel',
                                                        cooldown=0, min_lr=0, eps=1e-04)
        
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        for iter in range(epoches+1):
            model.train()
            with torch.amp.autocast(device_type='cuda'):
                choice_near = np.random.choice(len(point_near_data),1024*2,replace=False)
                choice_random = np.random.choice(len(point_random_data),256*4,replace=False)
                p_set = torch.empty((0, 3)).to(self.device)  
                sdf_set = torch.empty((0,)).to(self.device)
                for i in dataset.keys():
                    point_near_data = dataset[i]['point_near_data']
                    sdf_near_data = dataset[i]['sdf_near_data']
                    point_random_data = dataset[i]['point_random_data']
                    sdf_random_data = dataset[i]['sdf_random_data']
                    p_near, sdf_near = torch.from_numpy(point_near_data[choice_near]).float().to(self.device), \
                                        torch.from_numpy(sdf_near_data[choice_near]).float().to(self.device)
                    p_random, sdf_random = torch.from_numpy(point_random_data[choice_random]).float().to(self.device), \
                                            torch.from_numpy(sdf_random_data[choice_random]).float().to(self.device)
                    p = torch.cat([p_near,p_random],dim=0)
                    sdf = torch.cat([sdf_near,sdf_random],dim=0)

                    p_set = torch.cat([p_set, p], dim=0) 
                    sdf_set = torch.cat([sdf_set, sdf], dim=0)

                p_set = p_set/4
                sdf_set = sdf_set/4

                sdf_pred = model.forward(p_set)
                
                mse_loss = F.mse_loss(sdf_pred[:,0], sdf_set, reduction='mean')
                loss = mse_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(loss)
                
                if iter % 500 == 0:
                    print(f'Iter:{iter:5d}  Loss:{loss.item():.10f}')
                    os.makedirs(f'{self.model_path}/iters', exist_ok=True)
                    mesh_dict[0]['model'] = model
                    torch.save(mesh_dict,f'{self.model_path}/iters/SAMP_USDF_{iter}.pt') # save nn sdf model

        mesh_dict[0]['model'] = model
        torch.save(mesh_dict,f'{self.model_path}/SAMP_USDF_ALL.pt') # save nn sdf model
        print(f'{self.model_path}/SAMP_USDF_ALL.pt model saved!')

    def sdf_to_mesh(self, model, nbData):
        verts_list, faces_list, mesh_name_list = [], [], []
        mesh_dict = model[0]
        mesh_name = mesh_dict['mesh_name']
        mesh_name_list.append(mesh_name)
        model_k = mesh_dict['model'].to(self.device)
        model_k.eval()

        domain = torch.linspace(-1,1,nbData).to(self.device)
        z_domain = torch.linspace(-1,1,nbData).to(self.device)

        grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, z_domain, indexing='ij')
        grid_x, grid_y, grid_z = grid_x.reshape(-1,1), grid_y.reshape(-1,1), grid_z.reshape(-1,1)
        p = torch.cat([grid_x, grid_y, grid_z],dim=1).float().to(self.device)
        

        p_split = torch.split(p, 100000, dim=0)
        d =[]
        with torch.no_grad():
            for p_s in p_split:
                d_s = model_k.forward(p_s)
                d.append(d_s)
        d = torch.cat(d,dim=0)
        # scene.add_geometry(mesh)

        volume = d.view(nbData, nbData, nbData).detach().cpu().numpy()
        vmin, vmax = volume.min(), volume.max()
        
        spacing = np.array([8/(nbData), 8/(nbData), 8/nbData])  # 修改 spacing
        verts, faces, normals, values = skimage.measure.marching_cubes(
            d.view(nbData, nbData, nbData).detach().cpu().numpy(), level=0.0, spacing=spacing)
        verts = verts - [4,4,4]

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        center = mesh.get_center()
        print("Center of the mesh:", center)

        # o3d.visualization.draw_geometries([mesh], window_name='mesh', width=800, height=600)
        verts_list.append(verts)
        faces_list.append(faces)

        return verts_list, faces_list,mesh_name_list
    

    def create_surface_mesh(self,model, nbData, vis=False, save_mesh_name=None):
        verts_list, faces_list,mesh_name_list = self.sdf_to_mesh(model, nbData)

        for verts, faces, mesh_name in zip(verts_list, faces_list,mesh_name_list):
            rec_mesh = trimesh.Trimesh(verts,faces)
            
            def split_mesh_by_fixed_block_size(mesh, block_size=1.0):
                vertices = mesh.vertices
                vertices = vertices + (block_size * 1.5)
                block_ids = np.floor(vertices / block_size).astype(np.int32)
                block_vertex_indices = {}
                for idx, bid in enumerate(block_ids):
                    bid_tuple = tuple(bid)
                    block_vertex_indices.setdefault(bid_tuple, []).append(idx)

                block_meshes = []
                for bid, indices in block_vertex_indices.items():
                    indices_set = set(indices)
                    face_mask = np.all(np.isin(mesh.faces, list(indices_set)), axis=1)
                    if np.sum(face_mask) > 0:
                        sub_mesh = mesh.submesh([face_mask], append=True)
                        block_meshes.append(sub_mesh)
                return block_meshes

            if vis:
                rec_mesh.show()
            if save_mesh_name !=None:
                save_path = os.path.join(CUR_DIR,"output_meshes")
                if os.path.exists(save_path) is False:
                    os.mkdir(save_path)
                trimesh.exchange.export.export_mesh(rec_mesh, os.path.join(save_path,f"{save_mesh_name}_{mesh_name}.stl"))

            block_meshes = split_mesh_by_fixed_block_size(rec_mesh, block_size=3.0)
            mesh_name_list = [6, 7, 8, 4, 3, 5, 0, 2, 1]
            for i in range(len(block_meshes)):
                print(model[mesh_name_list[i]]['outer_offset'])
                block_meshes[i].vertices = block_meshes[i].vertices - model[mesh_name_list[i]]['outer_offset'].cpu().numpy()
                block_meshes[i].export(f"output_meshes/block_mesh_{mesh_name_list[i]}.stl")
                print(f"panda_block_mesh_{mesh_name_list[i]}.stl saved!")

    def batched_forward(self, model, x, batch_size=2000):
        outs = []
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                outs.append(model(x[i:i+batch_size]))
        return torch.cat(outs, dim=0)

    def add_outer_offset_in_chunks(self, x_bounded, outer_offset, chunk_size=10000):
        # x_bounded: [K, B, N, 3], outer_offset: [K, B, 1, 3]
        K, B, N, _ = x_bounded.shape
        out = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            # [K, B, chunk, 3] + [K, B, 1, 3] -> [K, B, chunk, 3]
            chunk = x_bounded[:, :, start:end, :] + outer_offset
            out.append(chunk)
        return torch.cat(out, dim=2)

    def whole_body_nn_sdf(self, x, pose, theta, model, used_links=[0,1,2,3,4,5,6,7,8]):
        B = len(theta)
        N = len(x)
        K = len(used_links)
        device = self.device

        x = x.float().to(device)
        pose = pose.to(device)
        theta = theta.to(device)

        offset = torch.cat([model[i]['offset'].unsqueeze(0) for i in used_links], dim=0).to(device)
        offset = offset.unsqueeze(0).expand(B, K, 3).reshape(B*K, 3).float()
        scale = torch.tensor([model[i]['scale'] for i in used_links], device=device)
        scale = scale.unsqueeze(0).expand(B, K).reshape(B*K).float()
        outer_offset = torch.cat([model[i]['outer_offset'].unsqueeze(0) for i in used_links], dim=0).to(device)
        outer_offset = outer_offset.unsqueeze(1).unsqueeze(2).expand(K, B, 1, 3)

        trans_list = self.robot.get_transformations_each_link(pose, theta)
        fk_trans = torch.cat([t.unsqueeze(1) for t in trans_list], dim=1)[:, used_links, :, :].reshape(-1, 4, 4)

        x_robot_frame_batch_scaled = sdf_utils.transform_points(
            x, torch.linalg.inv(fk_trans).float(), device=device
        )

        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled / scale.unsqueeze(-1).unsqueeze(-1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled.reshape(B, K, N, 3).transpose(0, 1)

        x_bounded = torch.clamp(x_robot_frame_batch_scaled, -1.0+1e-2, 1.0-1e-2)

        if self.infer_model is None:
            self.infer_model = model[0]['model'].eval().to(device)

        chunk = x_bounded + outer_offset  # [K, B, N, 3]
        chunk = chunk.reshape(K*B*N, 3)

        with torch.cuda.amp.autocast():
            sdf_pred = self.batched_forward(self.infer_model, chunk/4, batch_size=int(N/2)) 
        sdf_pred = sdf_pred.reshape(K, B, N) * 4

        res_x_chunk = x_robot_frame_batch_scaled - x_bounded
        res_x_norm = res_x_chunk.norm(dim=-1)
        sdf_chunk = sdf_pred + res_x_norm

        sdf = sdf_chunk.transpose(0, 1)
        sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
        sdf_value, idx = sdf.min(dim=1)
        return sdf_value

    def whole_body_nn_sdf_batch_points(self, x, pose, theta, model, used_links=[0,1,2,3,4,5,6,7,8]):
        B = len(theta)
        N = len(x[0])
        K = len(used_links)
        device = self.device

        x = x.float().to(device)
        pose = pose.to(device)
        theta = theta.to(device)

        offset = torch.cat([model[i]['offset'].unsqueeze(0) for i in used_links], dim=0).to(device)
        offset = offset.unsqueeze(0).expand(B, K, 3).reshape(B*K, 3).float()
        scale = torch.tensor([model[i]['scale'] for i in used_links], device=device)
        scale = scale.unsqueeze(0).expand(B, K).reshape(B*K).float()
        outer_offset = torch.cat([model[i]['outer_offset'].unsqueeze(0) for i in used_links], dim=0).to(device)
        outer_offset = outer_offset.unsqueeze(1).unsqueeze(2).expand(K, B, 1, 3)

        trans_list = self.robot.get_transformations_each_link(pose, theta)
        fk_trans = torch.cat([t.unsqueeze(1) for t in trans_list], dim=1)[:, used_links, :, :].reshape(-1, 4, 4)
        
        x_robot_frame_batch_scaled = sdf_utils.transform_points_batch(
            x, torch.linalg.inv(fk_trans).float(), device=device, links_num=len(used_links)
        )

        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled - offset.unsqueeze(1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled / scale.unsqueeze(-1).unsqueeze(-1)
        x_robot_frame_batch_scaled = x_robot_frame_batch_scaled.reshape(B, K, N, 3).transpose(0, 1)

        x_bounded = torch.clamp(x_robot_frame_batch_scaled, -1.0+1e-2, 1.0-1e-2)

        if self.infer_model is None:
            self.infer_model = model[0]['model'].eval().to(device)


        chunk = x_bounded + outer_offset  # [K, B, N, 3]
        chunk = chunk.reshape(K*B*N, 3)

        optimal_batch_size = min(N*B, 16384)
        with torch.cuda.amp.autocast():
            sdf_pred = self.batched_forward(self.infer_model, chunk/4, batch_size=optimal_batch_size) 
        sdf_pred = sdf_pred.reshape(K, B, N) * 4

        res_x_chunk = x_robot_frame_batch_scaled - x_bounded
        res_x_norm = res_x_chunk.norm(dim=-1)
        sdf_chunk = sdf_pred + res_x_norm

        sdf = sdf_chunk.transpose(0, 1)
        sdf = sdf * scale.reshape(B, K).unsqueeze(-1)
        sdf_value, idx = sdf.min(dim=1)
        return sdf_value

    def trimesh_to_o3d(self, mesh):
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        if mesh.visual.kind != 'none' and hasattr(mesh.visual, 'vertex_colors'):
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh.visual.vertex_colors[:, :3].astype(np.float64) / 255.0)
        o3d_mesh.compute_vertex_normals()
        return o3d_mesh

    def display_manipulator(self, model, trans_list, collision_mesh=None, raw_points=None, detect_points=None, sdf_color=None, offset=1.2):
        if self.panda_visual_mesh is None:
            self.panda_visual_mesh = []

            mesh_path = os.path.join(CUR_DIR,f"output_meshes/block_mesh_*.stl")
            mesh_files = glob.glob(mesh_path)
            mesh_files.sort()
            for i,mf in enumerate(mesh_files):
                mesh = trimesh.load(mf)
                mesh_dict = model[i]
                in_offset = mesh_dict['offset'].cpu().numpy()
                scale = mesh_dict['scale']
                print(scale)
                mesh.vertices = mesh.vertices*scale + in_offset
                self.panda_visual_mesh.append(mesh)

        display_mesh_list = []
        for i in range(len(trans_list)):
            trans_mat = trans_list[i].squeeze(0).cpu().numpy()
            mesh = copy.copy(self.panda_visual_mesh[i])
            mesh.apply_transform(trans_mat)
            o3d_mesh = self.trimesh_to_o3d(mesh)
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.translate([0, offset, 0])
            display_mesh_list.append(o3d_mesh)
        
        if collision_mesh is not None:
            # scale the collision mesh
            collision_mesh.scale(1/6, center=[0,0,0])
            display_mesh_list.append(collision_mesh)
        if raw_points is not None:
            raw_point_cloud = o3d.geometry.PointCloud()
            raw_point_cloud.points = o3d.utility.Vector3dVector(raw_points)
            if sdf_color is None:
                raw_point_cloud.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.86, 0.62, 0.27]]), (len(raw_points), 1)))
            else:
                raw_point_cloud.colors = o3d.utility.Vector3dVector(sdf_color)
            
            raw_point_cloud.translate([0, offset, 0])
            display_mesh_list.append(raw_point_cloud)

        o3d.visualization.draw_geometries(display_mesh_list, window_name='Cubes', width=800, height=600)

    def display_collision_mesh(self):
        if self.panda_collision_mesh is None:
            mesh_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output_meshes")
            mesh_file = mesh_path + "/hashgrid_panda_link0.stl"
            self.panda_collision_mesh = trimesh.load(mesh_file)
        
        
        o3d_mesh = self.trimesh_to_o3d(self.panda_collision_mesh)
        o3d_mesh.compute_vertex_normals()

        return o3d_mesh

    def whole_body_nn_sdf_with_joints_grad_batch(self,x,pose,theta,model,used_links = [0,1,2,3,4,5,6,7,8]):
        delta = 0.001
        B = theta.shape[0]
        theta = theta.unsqueeze(1)
        d_theta = (theta.expand(B,7,7)+ torch.eye(7,device=self.device).unsqueeze(0).expand(B,7,7)*delta).reshape(B,-1,7)
        theta = torch.cat([theta,d_theta],dim=1).reshape(B*8,7)
        pose = pose.expand(B*8,4,4)
        sdf = self.whole_body_nn_sdf(x,pose,theta,model, used_links = used_links).reshape(B,8,-1)
        d_sdf = (sdf[:,1:,:]-sdf[:,:1,:])/delta
        return sdf[:,0,:],d_sdf.transpose(1,2)
    
    def batch_robot_points(self, thetas, device='cuda'):   
        if self.points_list is None:
            with h5py.File('./panda_point_cloud_1024.hdf5', "r") as f:
                self.points_list = [
                    torch.from_numpy(f[key][:]).float().to(device)
                    for key in sorted(f.keys())
                ]

        num_links = len(self.points_list)
        B = thetas.shape[0]
        pose = torch.eye(4, device=device).unsqueeze(0).expand(B, 4, 4)
        trans_list = self.robot.get_transformations_each_link(pose, thetas)  # list of [B, 4, 4]

        batch_points = []
        for link_idx in range(1, num_links):
            pts = self.points_list[link_idx]
            N = pts.shape[0]
            pts_expand = pts.unsqueeze(0).expand(B, N, 3) # (B, N, 3)
            trans_mats = trans_list[link_idx]  # (B, 4, 4)
            pts_homo = torch.cat([pts_expand, torch.ones(B, N, 1, device=device)], dim=2) # (B, N, 4)
            pts_trans = torch.einsum('bij,bnj->bni', trans_mats, pts_homo)[..., :3]
            batch_points.append(pts_trans)

        batch_points = torch.cat(batch_points, dim=1)  # (B, sum_N, 3)
        
        return batch_points
    
    def normalize_franka_joints(self, batch_trajectory, limits=(-1, 1)):
        franka_limits = torch.as_tensor(self.joint_limits).type_as(batch_trajectory)
        return (batch_trajectory - franka_limits[:, 0]) / (
            franka_limits[:, 1] - franka_limits[:, 0]
        ) * (limits[1] - limits[0]) + limits[0]
    
    def unnormalize_franka_joints(self, batch_trajectory, limits=(-1, 1)):
        franka_limits = torch.as_tensor(self.joint_limits).type_as(batch_trajectory)
        franka_limit_range = franka_limits[:, 1] - franka_limits[:, 0]
        franka_lower_limit = franka_limits[:, 0]
        for _ in range(batch_trajectory.ndim - 1):
            franka_limit_range = franka_limit_range.unsqueeze(0)
            franka_lower_limit = franka_lower_limit.unsqueeze(0)
        return (batch_trajectory - limits[0]) * franka_limit_range / (
            limits[1] - limits[0]
        ) + franka_lower_limit


def sample_panda_points(panda, hash_nn_sdf, model, device='cuda'):
    points_size = 10
    x = torch.meshgrid(torch.linspace(-0.8,0.8,points_size),
                       torch.linspace(-0.8,0.8,points_size),
                       torch.linspace(0.0,1.2,points_size))
    x = torch.cat([x[0].reshape(-1,1),x[1].reshape(-1,1),x[2].reshape(-1,1)],dim=1).to(torch.float16).to(device)

    theta1 = torch.tensor([0, 0, 0.0, 0.0, 0, 0, 0]).float().to(device).reshape(-1,7)
    # theta2 = torch.tensor([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]).float().to(device).reshape(-1,7)
    # theta3 = torch.tensor([0, 0.3, 0, 2.2, 0, -2.0, -np.pi/4]).float().to(device).reshape(-1,7)
    # theta4 = torch.tensor([0, 0.3, 0, -2.2, 0, 2.0, np.pi/4]).float().to(device).reshape(-1,7)
    # theta5 = torch.tensor([0.5, -0.3, 0, 2.2, 0, -2.0, -np.pi/4]).float().to(device).reshape(-1,7)
    theta = torch.cat([theta1], dim=0).to(device).float()

    theta = theta.repeat(10,1).to(device).float()

    pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(device).expand(len(theta),4,4).float()    
    
    print(x.shape, pose.shape, theta.shape)
    # repeat infer 10 times
    for i in range(1):
        start_time = time.time()
        sdf_value = hash_nn_sdf.whole_body_nn_sdf(x, pose, theta, model)
        print("Time for inference:", time.time() - start_time)

    filter_distance = 1.0
    color_distance = 0.1
    index = 0

    all_sdf = sdf_value[index].detach().cpu().numpy()
    min_sdf = all_sdf.min()
    max_sdf = all_sdf.max()
    print("SDF min:", min_sdf, "max:", max_sdf)
    normalized_sdf = (all_sdf - min_sdf) / (color_distance - min_sdf)
    cmap = plt.get_cmap('coolwarm') 
    all_sdf_rgb = cmap(normalized_sdf)[:, :3]  # [N, 3]
    all_sdf_rgb = all_sdf_rgb.astype(np.float64)

    mask = all_sdf < filter_distance
    mask = mask & (x.detach().cpu().numpy()[:, 1] > 0.0)
    filtered_points = x.detach().cpu().numpy()[mask]
    filtered_colors = all_sdf_rgb[mask]

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    trans_list = panda.get_transformations_each_link(pose, theta[index].unsqueeze(0))
    panda_visual_mesh = []
    mesh_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "robot_layer", "meshes", "panda", "*")
    mesh_files = glob.glob(mesh_path)
    mesh_files = sorted(mesh_files)
    
    for i in range(len(mesh_files)):
        mesh = trimesh.load(mesh_files[i])
        panda_visual_mesh.append(mesh)


    display_mesh_list = []
    # point sampling number for each link
    sample_points_num = [10, 70, 120, 100, 100, 100, 200, 200, 100]
    all_extend_points_for_display = []
    all_extend_points_for_save = []
    link_points_num = []

    for i in range(len(trans_list)):
        trans_mat = trans_list[i].squeeze(0).cpu().numpy()
        if len(trans_mat.shape) == 3:
            trans_mat = trans_mat[index]
        mesh = copy.copy(panda_visual_mesh[i])
        surface_points, face_indices = trimesh.sample.sample_surface_even(mesh, count=sample_points_num[i])
       
        normals = mesh.face_normals[face_indices]
        # sample distance along normal direction
        distances = [0.001, 0.03, 0.06]

        extend_surface_points_list = []
        for i in range(len(distances)):
            rand_vec = np.random.randn(*normals.shape)
            rand_vec -= (rand_vec * normals).sum(axis=1, keepdims=True) * normals  # 保证正交
            rand_vec = rand_vec / (np.linalg.norm(rand_vec, axis=1, keepdims=True) + 1e-8)
            tangent_dist = np.random.uniform(0, distances[i]*0.4, size=(surface_points.shape[0], 1))
            surface_point_extend = surface_points + normals * distances[i] + rand_vec * tangent_dist
            extend_surface_points_list.append(surface_point_extend)

        mesh.apply_transform(trans_mat)
        extend_surface_points_all_transform = trimesh.transform_points(np.concatenate(extend_surface_points_list, axis=0), trans_mat)
        o3d_mesh = hash_nn_sdf.trimesh_to_o3d(mesh)
        o3d_mesh.compute_vertex_normals()
        display_mesh_list.append(o3d_mesh)

        link_points_num.append(extend_surface_points_all_transform.shape[0])
        all_extend_points_for_display.append(extend_surface_points_all_transform)
        all_extend_points_for_save.append(np.concatenate(extend_surface_points_list, axis=0))

    all_extend_points_for_display = np.concatenate(all_extend_points_for_display, axis=0)
    all_extend_points_for_save_np = np.concatenate(all_extend_points_for_save, axis=0)
    all_display_points = all_extend_points_for_display
    all_save_points = all_extend_points_for_save_np

    all_points_tensor = torch.from_numpy(all_display_points).float().to(device)
    pose_batch = pose[:1]
    theta_batch = theta[:1]

    sdf_values = hash_nn_sdf.whole_body_nn_sdf(all_points_tensor, pose_batch, theta_batch, model)
    sdf_values = sdf_values.detach().cpu().numpy()[0]

    mask = (sdf_values > 0.0005) & (all_display_points[:, 2] >= 0.01)

    filtered_points_display = all_display_points[mask]
    filtered_points_save = all_save_points[mask]
    num_save = 1024
    if filtered_points_save.shape[0] > num_save:
        idx = np.random.choice(np.arange(link_points_num[0], filtered_points_save.shape[0]), num_save, replace=False)
        idx = np.sort(idx)
        filtered_points_save = filtered_points_save[idx]
        filtered_points_display = filtered_points_display[idx]

    link_points_num = np.array(link_points_num)
    link_start = np.concatenate([[0], np.cumsum(link_points_num)[:-1]])
    link_end = np.cumsum(link_points_num)

    mask_idx = np.where(mask)[0]  
    final_idx = mask_idx[idx]     

    link_ids = np.zeros_like(final_idx)
    for i, (start, end) in enumerate(zip(link_start, link_end)):
        in_link = (final_idx >= start) & (final_idx < end)
        link_ids[in_link] = i

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i)[:3] for i in range(len(link_points_num))]
    points_list = []
    colors_list = []

    with h5py.File("panda_point_cloud_1024.hdf5", "w") as f:
        for i in range(len(sample_points_num)):
            if i == 0:
                link_points = all_extend_points_for_save[0]
            else:
                link_points = filtered_points_save[link_ids == i]
            print("shape of link_points:", link_points.shape)
            if link_points.shape[0] == 0:
                continue
            f.create_dataset(f"link_{i}_points", data=link_points)
            link_points = filtered_points_display[link_ids == i]

            if link_points.shape[0] == 0:
                continue
            points_list.append(link_points)
            color = np.array(colors[i]).reshape(1, 3)
            colors_list.append(np.repeat(color, link_points.shape[0], axis=0))

    all_points_colored = np.concatenate(points_list, axis=0)
    all_colors_colored = np.concatenate(colors_list, axis=0)

    filtered_points_o3d = o3d.geometry.PointCloud()
    filtered_points_o3d.points = o3d.utility.Vector3dVector(all_points_colored)
    filtered_points_o3d.colors = o3d.utility.Vector3dVector(all_colors_colored)
    display_mesh_list.append(filtered_points_o3d)

    o3d.visualization.draw_geometries(display_mesh_list, 
                                      window_name='point_near_data', width=1024, height=768)
    

if  __name__ =='__main__':
    device = 'cuda'
    panda = PandaLayer(device=device)
    panda_hash_nn_sdf = HashNNSDF(panda, lr=0.002, device=device)
    # if already trained, comment the training line and load the model directly
    # panda_hash_nn_sdf.train_hash_nn(epoches=5000)

    model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/SAMP_USDF_ALL.pt'
    model = torch.load(model_path, weights_only=False)
    # panda_hash_nn_sdf.create_surface_mesh(model, nbData=512, vis=True, save_mesh_name='hashgrid')

    joint_limits = [
        (-2.8973, 2.8973),
        (-1.7628, 1.7628),
        (-2.8973, 2.8973),
        (-3.0718, -0.0698),
        (-2.8973, 2.8973),
        (-0.0175, 3.7525),
        (-2.8973, 2.8973)
    ]

    # for display multiple poses
    thetas = np.array([
        [0, -0.3, 0, -2.2, 0, 2.0, np.pi/4],
        [0, -0.5, 0, -2.2, -0.8, 1.0, -0.5],
        [1.5, -0.3, 0, -2.2, 0, 1.0, -np.pi/4],
        [1.0, 1.0, -1.0, -2.0, 1.0, 2.0, 0.5],
        [-2.0, 0.0, 2.0, -1.5, 0.0, 1.5, -2.0],
        [2.0, -1.0, 1.0, -1.5, -1.0, 0.5, 2.0],
    ])
    thetas = torch.from_numpy(thetas).float().to(device)  # (100, 7)

    # Sample points
    sample_panda_points(panda, panda_hash_nn_sdf, model, device=device)
    
    batch_points = panda_hash_nn_sdf.batch_robot_points(thetas)
    print("Batch points shape:", batch_points.shape)
    batch_points_np = batch_points.detach().cpu().numpy()

    ############################## display sampled points with sdf color  ##############################
    for i in range(thetas.shape[0]):
        print(f"Points for theta {i}:", batch_points_np[i].shape)
        pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to("cuda").expand(len([thetas[i]]),4,4).float()  
        
        x = batch_points[i]
        points_sdf = panda_hash_nn_sdf.whole_body_nn_sdf(x, 
                                                         pose, 
                                                         thetas[i].unsqueeze(0), 
                                                         model)

        color_distance = 0.8
        all_sdf = points_sdf[0].detach().cpu().numpy()
        min_sdf = all_sdf.min()
        max_sdf = all_sdf.max()
        print("SDF min:", min_sdf, "max:", max_sdf)
        normalized_sdf =  (all_sdf - min_sdf) / (max_sdf-min_sdf + 1e-8)
        normalized_sdf = 1.0 - normalized_sdf
        print("normalized sdf min:", normalized_sdf.min(), "max:", normalized_sdf.max())
        
        cmap = plt.get_cmap('Oranges') 
        all_sdf_rgb = cmap(normalized_sdf)[:, :3]  # [N, 3]
        all_sdf_rgb = all_sdf_rgb.astype(np.float64)

        filtered_points = x.detach().cpu().numpy()
        filtered_colors = all_sdf_rgb

        trans_list = panda.get_transformations_each_link(
            torch.eye(4, device=device).unsqueeze(0), 
            thetas[i].unsqueeze(0)
        )

        panda_hash_nn_sdf.display_manipulator(model, trans_list, raw_points=batch_points_np[i], sdf_color=filtered_colors)

    ########################################## display sdf mesh ##########################################
    points_size = 150
    x = torch.meshgrid(torch.linspace(-0.8,0.8,points_size),
                       torch.linspace(-0.8,0.8,points_size),
                       torch.linspace(-0.1,1.2,points_size))
    x = torch.cat([x[0].reshape(-1,1),x[1].reshape(-1,1),x[2].reshape(-1,1)],dim=1).to(torch.float16).to(device)
    theta1 = torch.tensor([0, 0, 0.0, 0.0, 0, 0, 0]).float().to(device).reshape(-1,7)
    theta2 = torch.tensor([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]).float().to(device).reshape(-1,7)
    theta3 = torch.tensor([0, 0.3, 0, 2.2, 0, -2.0, -np.pi/4]).float().to(device).reshape(-1,7)
    theta4 = torch.tensor([0, 0.3, 0, -2.2, 0, 2.0, np.pi/4]).float().to(device).reshape(-1,7)
    theta5 = torch.tensor([0.5, -0.3, 0, 2.2, 0, -2.0, -np.pi/4]).float().to(device).reshape(-1,7)
    theta = torch.cat([theta4], dim=0).to(device).float()
    
    theta = theta.repeat(1,1).to(device).float()

    pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(device).expand(len(theta),4,4).float() 
    
    sdf_value_visual = panda_hash_nn_sdf.whole_body_nn_sdf(x, pose, theta, model).detach().cpu().numpy()
    display_mesh_list = []
    levels = np.linspace(0.01, 0.06, 5)  
    volume = sdf_value_visual.reshape(points_size, points_size, points_size)
    for lv in levels:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            volume, level=lv, spacing=(
                (0.8-(-0.8))/(points_size-1),
                (0.8-(-0.8))/(points_size-1),
                (1.2-(-0.1))/(points_size-1)
            )
        )
        verts[:, 0] += -0.8
        verts[:, 1] += -0.8
        verts[:, 2] += -0.1

        # mesh_mask = np.all(verts[faces][:, :, 1] >= 0.02, axis=1)  # front
        mesh_mask = np.all(verts[faces][:, :, 1] <= -0.02, axis=1) # back
        faces_filtered = faces[mesh_mask]

        sdf_mesh = o3d.geometry.TriangleMesh()
        sdf_mesh.vertices = o3d.utility.Vector3dVector(verts)
        sdf_mesh.triangles = o3d.utility.Vector3iVector(faces_filtered)
        # sdf_mesh.compute_triangle_normals()
        sdf_mesh.compute_vertex_normals()
        color = plt.get_cmap('coolwarm_r')((lv-levels[0])/(levels[-1]-levels[0]))[:3]
        sdf_mesh.paint_uniform_color(color)
        display_mesh_list.append(sdf_mesh)

    index = 0
    trans_list = panda.get_transformations_each_link(pose, theta[index].unsqueeze(0))
    panda_visual_mesh = []

    mesh_path = os.path.join(CUR_DIR,f"output_meshes/block_mesh_*.stl")
    mesh_files = glob.glob(mesh_path)
    mesh_files.sort()
    for i,mf in enumerate(mesh_files):
        mesh = trimesh.load(mf)
        mesh_dict = model[i]
        in_offset = mesh_dict['offset'].cpu().numpy()
        scale = mesh_dict['scale']
        mesh.vertices = mesh.vertices*scale + in_offset
        panda_visual_mesh.append(mesh)

    for i in range(len(trans_list)):
        trans_mat = trans_list[i].squeeze(0).cpu().numpy()
        mesh = copy.copy(panda_visual_mesh[i])
        mesh.apply_transform(trans_mat)
        o3d_mesh = panda_hash_nn_sdf.trimesh_to_o3d(mesh)
        o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
        display_mesh_list.append(o3d_mesh)

    o3d.visualization.draw_geometries(display_mesh_list, 
                                    window_name='point_near_data', 
                                    width=800, 
                                    height=600, 
                                    mesh_show_back_face=True)


''' 

Copy this parameter settings and paste into open3d to view the sdf mesh from different angles

# front

{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.66336071859270151, 0.19059214751992459, 0.72560222548926445 ],
			"boundingbox_min" : [ -0.21472102427642614, -0.1895178878067324, -0.060823533358989948 ],
			"field_of_view" : 60.0,
			"front" : [ -0.13859389601595964, -0.98929758761735875, 0.045629103887670067 ],
			"lookat" : [ 0.20147190962620268, 0.0026546099783746183, 0.31923202878474005 ],
			"up" : [ -0.029840682661525727, 0.050224431231536089, 0.99829206155601724 ],
			"zoom" : 0.49999999999999978
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}


# back


{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 0.66336071859270151, 0.19059214751992459, 0.72560222548926445 ],
			"boundingbox_min" : [ -0.21472102427642614, -0.1895178878067324, -0.060823533358989948 ],
			"field_of_view" : 60.0,
			"front" : [ 0.021730353438176274, 0.99956870434632683, 0.019753405550878456 ],
			"lookat" : [ 0.21429502561203631, 0.0036304571266203792, 0.31723256776093622 ],
			"up" : [ 0.014498898151989861, -0.020071072299535021, 0.99969342000891703 ],
			"zoom" : 0.49999999999999978
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''
