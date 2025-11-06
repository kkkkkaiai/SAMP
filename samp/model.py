import torch
import torch.nn.functional as F
import os, time

from torch import nn
from typing import List, Tuple, Sequence, Dict, Callable
from samp.utils import grid_normalize_points
from pointnet2_ops.pointnet2_modules import PointnetSAModule

class FlowMatchingDecoder(nn.Module):
    def __init__(self, cond_dim, action_dim, hidden_dim=512):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(cond_dim+action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, zt, t, context):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        x = torch.cat([zt, t, context], dim=1)
        return self.fc(x)  # [B, action_dim]

class PointCloudEncoder(nn.Module):
    def __init__(self, latent_dim, num_point=2500, point_dim=3, bn_decay=0.5):
        self.num_point = num_point
        self.point_dim = point_dim 
        self.latent_dim = latent_dim
        super(PointCloudEncoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, point_dim), padding=0)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=(1, 1), padding=0)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, self.latent_dim, kernel_size=(1, 1), padding=0)
        self.bn5 = nn.BatchNorm2d(self.latent_dim)

        self.fc1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.bn_fc1 = nn.BatchNorm1d(self.latent_dim)


    def forward(self, x):
        # Encoder
        actual_num_points = x.size(2)
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        point_feat = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(point_feat)))
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=(actual_num_points, 1), padding=0)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.bn_fc1(self.fc1(x)))

        return x, point_feat, None
    
class MPiNetsPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._build_model()

    def _build_model(self):
        """
        Assembles the model design into a ModuleList
        """
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.05,
                nsample=128,
                mlp=[1, 64, 64, 64],
                bn=False,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.3,
                nsample=128,
                mlp=[64, 128, 128, 256],
                bn=False,
            )
        )
        self.SA_modules.append(PointnetSAModule(mlp=[256, 512, 512, 1024], bn=False))

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.GroupNorm(16, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, 2048),
            nn.GroupNorm(16, 2048),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2048, 2048),
        )

    @staticmethod
    def _break_up_pc(pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Breaks up the point cloud into the xyz coordinates and segmentation mask

        :param pc torch.Tensor: Tensor with shape [B, N, M] where M is larger than 3.
                                The first three dimensions along the last axis will be x, y, z
        :rtype Tuple[torch.Tensor, torch.Tensor]: Two tensors, one with just xyz
            and one with the corresponding features
        """
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous()
        return xyz, features

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass of the network

        :param point_cloud torch.Tensor: Has dimensions (B, N, 4)
                                              B is the batch size
                                              N is the number of points
                                              4 is x, y, z, segmentation_mask
                                              This tensor must be on the GPU (CPU tensors not supported)
        :rtype torch.Tensor: The output from the network
        """
        assert point_cloud.size(2) == 4
        xyz, features = self._break_up_pc(point_cloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))
    

class MotionPlanningNetwork(nn.Module):
    def __init__(self, q_feature_size, anchor_size, output_size, dropout_p=0.0):
        super().__init__()


        self.cspace_feature_encoder = nn.Sequential(
            nn.Linear(q_feature_size, 32),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, 64),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 128),
        )


        self.robot_feature_encoder = PointCloudEncoder(latent_dim=256, num_point=1024, point_dim=4).to(device='cuda')

        self.pre_env_feature_encoder = nn.Sequential(
            nn.Conv3d(1, 4, 3, padding=1, stride=2, padding_mode='zeros'),
            nn.BatchNorm3d(4),
            nn.ELU(),
            nn.MaxPool3d(kernel_size=2, stride=1),
        )

        self.env_fc = nn.Sequential(
            nn.Linear(5120, 512),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
        )

        self.decoder = nn.Sequential(
            nn.Linear(640, 256),  
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(dropout_p),
            nn.Linear(32, output_size),
        )


    def forward(self, q: torch.Tensor, robot_xyz: torch.Tensor, env_xyz: torch.Tensor, robot_sdf: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            c_encoding = self.cspace_feature_encoder(q)

            batch_size = robot_xyz.size(0)
            if robot_xyz.dim() == 5:
                robot_points = robot_xyz.squeeze(1).squeeze(1)
            else:
                robot_points = robot_xyz.view(batch_size, -1, 3)
            robot_sdf_reshaped = robot_sdf.unsqueeze(-1)
            robot_points_with_sdf = torch.cat([robot_points, robot_sdf_reshaped], dim=-1)
            robot_points_with_sdf = robot_points_with_sdf.unsqueeze(1)
            robot_encoding, point_feat, _ = self.robot_feature_encoder(robot_points_with_sdf)
            robot_xyz = grid_normalize_points(robot_xyz)
            swap_idx = [2, 1, 0]  # z, y, x
            robot_xyz = robot_xyz[..., swap_idx]  

            f_1 = self.pre_env_feature_encoder(env_xyz)
            env_sdf_feature = F.grid_sample(f_1, robot_xyz, mode='bilinear', padding_mode='border', align_corners=True)
            robot_sdf = robot_sdf.unsqueeze(1).unsqueeze(1).unsqueeze(1) 

            sdf_features = torch.cat((robot_sdf, env_sdf_feature), dim=1)  
            sdf_features = sdf_features.reshape(sdf_features.size(0), -1)
            sdf_features = self.env_fc(sdf_features)
            
            all_features = torch.cat((c_encoding, robot_encoding, sdf_features), dim=1) 
            q = self.decoder(all_features)  

            return q

    def forward1(self, q: torch.Tensor, robot_xyz: torch.Tensor, env_xyz: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            c_encoding = self.cspace_feature_encoder(q)

            robot_encoding = self.robot_feature_encoder(robot_xyz.view(robot_xyz.size(0), -1, 3))  # [N, 1024, 32]
            robot_encoding, _ = robot_encoding.max(dim=1)
        
            f_0 = env_xyz
            f_1 = self.pre_env_feature_encoder(env_xyz)

            feature_0 = F.grid_sample(f_0, robot_xyz, mode='bilinear', padding_mode='border', align_corners=True)
            feature_1 = F.grid_sample(f_1, robot_xyz, mode='bilinear', padding_mode='border', align_corners=True)

            features = torch.cat((feature_0, feature_1), dim=1)  
            features = features.reshape(features.size(0), -1)
            features = self.env_fc(features)
            
            # print(c_encoding.shape, robot_encoding.shape, features.shape)
            all_features = torch.cat((c_encoding, robot_encoding, features), dim=1)  
            q = self.decoder(all_features)  

            return q
    

if __name__ == "__main__":
    N = 1
    q_dim = 14
    label_dim = 7
    C = 1
    D = H = W = 100
    q_feature = torch.randn(N, q_dim)
    label = torch.randn(N, label_dim)
    robot_feature = torch.randn(N, 1, 1, 1024, 3)
    obs_feature = torch.randn(N, 1, D, H, W)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fix: Assign the returned tensors back to their variables
    q_feature = q_feature.to(device)
    robot_feature = robot_feature.to(device)
    obs_feature = obs_feature.to(device)

    model = MotionPlanningNetwork(q_feature_size=q_dim, anchor_size=D, output_size=label_dim)
    model.to(device)
    
    start_time = time.time()
    # Optional: Use inference mode for speed
    model.eval()
    with torch.no_grad():
        for i in range(100):
            start_time = time.time()
            out = model(q_feature, robot_feature, obs_feature)
            print("Time taken:", time.time() - start_time)
    
    print("Output shape:", out.shape)  # Should be [N, label_dim]
