from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import torch
import numpy as np
import os
import time
import math
import h5py

from encode_panda_sdf_points_sampler import HashNNSDF, HashMLPModel
from robot_layer.panda_layer import PandaLayer
from samp.utils import normalize_franka_joints, unnormalize_franka_joints, set_random_seed, custom_collate_fn

from robofin.pointcloud.torch import FrankaSampler

RED = "\033[31m"          # red
GREEN = "\033[32m"        # green
YELLOW = "\033[33m"       # yellow
BLUE = "\033[34m"         # blue
MAGENTA = "\033[35m"      # purple red
CYAN = "\033[36m"         # cyan
WHITE = "\033[37m"        # white
RESET = "\033[0m"         # reset


class Trainer:
    def __init__(self, dataset, model=None, loss=None, lr=None, num_epochs=None, 
                 batch_size=None, weight_decay=None, device='cuda'):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.dataset = dataset
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_train = None
        self.num_val = None
        self.point_match_loss_weight = 1.0
        self.config_loss_weight = 1.0
        self.collision_loss_weight = 20.0
        self.final_config_loss_weight = 0.0

        self.batch_anchor_points = None

        self.train_loader, self.val_loader = self.data_split()
        print(f'Number of training samples: {self.num_train}')
        print(f'Number of validation samples: {self.num_val}')

        # Training Metrics
        self.train_loss = None 
        self.val_loss = None 

        self.scaler = GradScaler()

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=20,           # Initial restart interval (in epochs)
            T_mult=2,         # Multiply factor for restart interval
            eta_min=1e-6      # Minimum learning rate
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=1e-6)

        self.fk_sampler = None

        self.loss_fn = loss

        model_path = os.path.dirname(os.path.realpath(__file__)) + '/../models/SAMP_USDF_ALL.pt'
        
        self.sdf_model = torch.load(model_path, weights_only=False)
        self.panda = PandaLayer()
        self.panda_hash_nn_sdf = HashNNSDF(self.panda)

        self.dataset.set_joint_limits(self.panda_hash_nn_sdf.joint_limits)

        checkpoint_path = './trained_models/neural_motion_planning_08-04-192243_epoch_17.pth'
        checkpoint_path = None
        if checkpoint_path is not None:
            self.resume(checkpoint_path)  # Load the model and optimizer state from a checkpoint

        self.points_list = []
        with h5py.File('./panda_point_cloud_1024.hdf5', "r") as f:
            for key in sorted(f.keys()):
                pts = f[key][:]
                self.points_list.append(torch.from_numpy(pts).float().to(device))  # (N, 3)


    def resume(self, checkpoint_path):
        '''
        Resume training from a checkpoint.
        '''
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_loss = checkpoint.get('train_loss', None)
        self.val_loss = checkpoint.get('eval_loss', None)
        print(f"Resumed from {checkpoint_path}")


    def data_split(self):
        '''
        Splitting dataset into train and validation. 80% for training, and 20% for validation.
        '''
        set_random_seed(42)             # For reproducibility
        num_items = len(self.dataset)

        train_ratio = 0.9
        split = int(math.floor(train_ratio * num_items))

        shuffled_indices = torch.randperm(num_items).tolist()

        train_indices, val_indices = shuffled_indices[:split], shuffled_indices[split:]

        x_train = Subset(self.dataset, train_indices)
        self.num_train = len(x_train)

        x_val = Subset(self.dataset, val_indices)
        self.num_val = len(x_val)

        train_loader = DataLoader(x_train, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=10, pin_memory=True)
        val_loader = DataLoader(x_val, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=10, pin_memory=True)

        return train_loader, val_loader
    
    def test_overfit_single_batch(self):
        sample_batch = next(iter(self.train_loader))
        print("Testing model capacity with single batch overfit...")
        avg_loss = 0
        match_loss_sum = 0
        coll_loss_sum = 0
        config_loss_sum = 0
        final_loss_sum = 0
        num_batches = 0

        for i in range(1000):
            self.optimizer.zero_grad()

            features_x = sample_batch["q_feature"].to(self.device)
            next_joint_state = sample_batch["q_next"].to(self.device)
            obs_sdf = sample_batch["obs_feature"].to(self.device)
            obs_sdf = torch.clamp(obs_sdf, min=-0.5, max=1.0)
            goal_joint_state = features_x[:, -7:].to(self.device)
            cur_joint_state = features_x[:, :7].to(self.device)

            if self.fk_sampler is None:
                self.fk_sampler = FrankaSampler(
                    'cuda',
                    num_fixed_points=1024,
                    use_cache=True,
                    with_base_link=False,
                )
                     
            cur_robot_points = self.panda_hash_nn_sdf.batch_robot_points(cur_joint_state, device=self.device)
            cur_pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(self.device).expand(cur_joint_state.shape[0], 4, 4).float()
            cur_robot_sdf = self.panda_hash_nn_sdf.whole_body_nn_sdf_batch_points(cur_robot_points, cur_pose, cur_joint_state, self.sdf_model)
            cur_robot_sdf = torch.clamp(cur_robot_sdf, min=-1.0, max=1.0)
            robot_feature = cur_robot_points.unsqueeze(1).unsqueeze(1) 
            
            # concatenate the current joint state and the goal joint state into 2*B, 7
            stack_q = torch.cat([cur_joint_state, goal_joint_state], dim=0)
            norm_q = normalize_franka_joints(stack_q)  # Normalize the joint states

            norm_cur = norm_q[:cur_joint_state.shape[0]]  
            norm_goal = norm_q[cur_joint_state.shape[0]:2*cur_joint_state.shape[0]]
            norm_features_x = torch.cat([norm_cur, norm_goal], dim=1)  # [B, 14]
            
            y_pred = self.model(norm_features_x, robot_feature, obs_sdf, cur_robot_sdf)
            pred_theta = y_pred + norm_features_x[:, :7]
            pred_theta = torch.clamp(pred_theta, min=-1.0, max=1.0)  
            pred_theta_unnorm = unnormalize_franka_joints(pred_theta) 
            
            next_robot_points = self.panda_hash_nn_sdf.batch_robot_points(next_joint_state, device=self.device)
            pred_robot_points = self.panda_hash_nn_sdf.batch_robot_points(pred_theta_unnorm, device=self.device)
            pred_pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(self.device).expand(pred_theta.shape[0], 4, 4).float()
            pred_robot_sdf = self.panda_hash_nn_sdf.whole_body_nn_sdf_batch_points(pred_robot_points, pred_pose, pred_theta, self.sdf_model)
            pred_robot_sdf = torch.clamp(pred_robot_sdf, min=-1.0, max=1.0)


            match_loss, coll_loss, config_loss, final_loss = \
                self.loss_fn(pred_theta_unnorm, goal_joint_state, next_joint_state, pred_robot_points, next_robot_points, pred_robot_sdf, obs_sdf)

            match_loss_sum += match_loss.item()
            coll_loss_sum += coll_loss.item()
            config_loss_sum += config_loss.item()
            final_loss_sum += final_loss.item()
            num_batches += 1
            total_loss = self.final_config_loss_weight * final_loss + \
                            self.collision_loss_weight * coll_loss + \
                            self.point_match_loss_weight * match_loss + \
                            self.config_loss_weight * config_loss 
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("Loss nan/inf!")
                exit()
            avg_loss += total_loss.data.item()

            # match_loss.backward(retain_graph=True)
            # match_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
            # print(f"Match loss gradient norm: {match_grad_norm}")
            
            total_loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {total_loss.item():.6f}")
            
        return total_loss.item() < 0.01

    def batch_train(self):
        '''
        Training of each batch
        '''
        avg_loss = 0
        match_loss_sum = 0
        coll_loss_sum = 0
        config_loss_sum = 0
        final_loss_sum = 0
        num_batches = 0
        self.model.train()
        for item in tqdm(self.train_loader, desc="Training"):
            self.optimizer.zero_grad()
            with autocast():
                features_x = item["q_feature"].to(self.device) # include cur joint state and goal joint state
                next_joint_state = item["q_next"].to(self.device)
                obs_sdf = item["obs_feature"].to(self.device)
                obs_sdf = torch.clamp(obs_sdf, min=-0.5, max=1.0)
                goal_joint_state = features_x[:, -7:].to(self.device)
                cur_joint_state = features_x[:, :7].to(self.device)

                if self.fk_sampler is None:
                    self.fk_sampler = FrankaSampler(
                        'cuda',
                        num_fixed_points=1024,
                        use_cache=True,
                        with_base_link=False,
                    )
                
                cur_robot_points = self.panda_hash_nn_sdf.batch_robot_points(cur_joint_state, device=self.device)
                # cur_robot_points = self.fk_sampler.sample(cur_joint_state)
                cur_pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(self.device).expand(cur_joint_state.shape[0], 4, 4).float()
                cur_robot_sdf = self.panda_hash_nn_sdf.whole_body_nn_sdf_batch_points(cur_robot_points, cur_pose, cur_joint_state, self.sdf_model)
                cur_robot_sdf = torch.clamp(cur_robot_sdf, min=-0.5, max=1.0)
                robot_feature = cur_robot_points.unsqueeze(1).unsqueeze(1) 
                
                stack_q = torch.cat([cur_joint_state, goal_joint_state], dim=0)
                norm_q = normalize_franka_joints(stack_q)  # Normalize the joint states

                norm_cur = norm_q[:cur_joint_state.shape[0]]  
                norm_goal = norm_q[cur_joint_state.shape[0]:2*cur_joint_state.shape[0]]
                norm_features_x = torch.cat([norm_cur, norm_goal], dim=1)  # [B, 14]
                
                y_pred = self.model(norm_features_x, robot_feature, obs_sdf, cur_robot_sdf)
                pred_theta = y_pred + norm_features_x[:, :7]
                pred_theta = torch.clamp(pred_theta, min=-1.0, max=1.0)  
                pred_theta_unnorm = unnormalize_franka_joints(pred_theta) 
                
                next_robot_points = self.panda_hash_nn_sdf.batch_robot_points(next_joint_state, device=self.device)
                pred_robot_points = self.panda_hash_nn_sdf.batch_robot_points(pred_theta_unnorm, device=self.device)
                pred_pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(self.device).expand(pred_theta.shape[0], 4, 4).float()
                pred_robot_sdf = self.panda_hash_nn_sdf.whole_body_nn_sdf_batch_points(pred_robot_points, pred_pose, pred_theta, self.sdf_model)
                pred_robot_sdf = torch.clamp(pred_robot_sdf, min=-0.5, max=1.0)       

                match_loss, coll_loss, config_loss, final_loss = \
                    self.loss_fn(pred_theta_unnorm, goal_joint_state, next_joint_state, pred_robot_points, next_robot_points, pred_robot_sdf, obs_sdf)
                
                match_loss_sum += match_loss.item()
                coll_loss_sum += coll_loss.item()
                config_loss_sum += config_loss.item()
                final_loss_sum += final_loss.item()
                num_batches += 1
                total_loss = self.final_config_loss_weight * final_loss + \
                                self.collision_loss_weight * coll_loss + \
                                self.point_match_loss_weight * match_loss + \
                                self.config_loss_weight * config_loss 
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("Loss nan/inf!")
                    exit()
                avg_loss += total_loss.data.item()
                
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
        return avg_loss, match_loss_sum / num_batches, coll_loss_sum / num_batches, config_loss_sum / num_batches, final_loss_sum / num_batches

    def batch_eval(self):
        '''
        Evaluating the result
        '''
        eval_loss = 0
        match_loss_sum = 0
        coll_loss_sum = 0
        config_loss_sum = 0
        final_loss_sum = 0
        num_batches = 0
        self.model.eval()
        with torch.no_grad():
            for item in tqdm(self.val_loader, desc="Evaluating"):
                features_x = item["q_feature"].to(self.device) # include cur joint state and goal joint state
                next_joint_state = item["q_next"].to(self.device)
                obs_sdf = item["obs_feature"].to(self.device)
                obs_sdf = torch.clamp(obs_sdf, min=-0.5, max=1.0)
                goal_joint_state = features_x[:, -7:].to(self.device)
                cur_joint_state = features_x[:, :7].to(self.device)

                if self.fk_sampler is None:
                    self.fk_sampler = FrankaSampler(
                        'cuda',
                        num_fixed_points=1024,
                        use_cache=True,
                        with_base_link=False,
                    )
                
                cur_robot_points = self.panda_hash_nn_sdf.batch_robot_points(cur_joint_state, device=self.device)
                cur_pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(self.device).expand(cur_joint_state.shape[0], 4, 4).float()
                cur_robot_sdf = self.panda_hash_nn_sdf.whole_body_nn_sdf_batch_points(cur_robot_points, cur_pose, cur_joint_state, self.sdf_model)
                cur_robot_sdf = torch.clamp(cur_robot_sdf, min=-0.5, max=1.0)
                robot_feature = cur_robot_points.unsqueeze(1).unsqueeze(1) 
                
                stack_q = torch.cat([cur_joint_state, goal_joint_state], dim=0)
                norm_q = normalize_franka_joints(stack_q)  # Normalize the joint states

                norm_cur = norm_q[:cur_joint_state.shape[0]]  
                norm_goal = norm_q[cur_joint_state.shape[0]:2*cur_joint_state.shape[0]]
                norm_features_x = torch.cat([norm_cur, norm_goal], dim=1)  # [B, 14]
                
                y_pred = self.model(norm_features_x, robot_feature, obs_sdf, cur_robot_sdf)
                pred_theta = y_pred + norm_features_x[:, :7]
                pred_theta = torch.clamp(pred_theta, min=-1.0, max=1.0)  
                pred_theta_unnorm = unnormalize_franka_joints(pred_theta) 
                
                next_robot_points = self.panda_hash_nn_sdf.batch_robot_points(next_joint_state, device=self.device)
                pred_robot_points = self.panda_hash_nn_sdf.batch_robot_points(pred_theta_unnorm, device=self.device)
                pred_pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(self.device).expand(pred_theta.shape[0], 4, 4).float()
                pred_robot_sdf = self.panda_hash_nn_sdf.whole_body_nn_sdf_batch_points(pred_robot_points, pred_pose, pred_theta, self.sdf_model)
                pred_robot_sdf = torch.clamp(pred_robot_sdf, min=-0.5, max=1.0)
                
                match_loss, coll_loss, config_loss, final_loss = \
                    self.loss_fn(pred_theta_unnorm, goal_joint_state, next_joint_state, pred_robot_points, next_robot_points, pred_robot_sdf, obs_sdf)
                
                match_loss_sum += match_loss.item()
                coll_loss_sum += coll_loss.item()
                config_loss_sum += config_loss.item()
                final_loss_sum += final_loss.item()
                num_batches += 1
                total_loss = self.final_config_loss_weight * final_loss + \
                                self.collision_loss_weight * coll_loss + \
                                self.point_match_loss_weight * match_loss + \
                                self.config_loss_weight * config_loss 
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print("Loss nan/inf!")
                    exit()

                eval_loss += total_loss.data.item()
        return eval_loss, match_loss_sum / num_batches, coll_loss_sum / num_batches, config_loss_sum / num_batches, final_loss_sum / num_batches
                

    def train(self):
        '''
        Train the model for the given epochs
        '''
        for epoch in range(1, self.num_epochs + 1):
            # if epoch > int(self.num_epochs * 0.90):
            #     self.final_config_loss_weight = 0.0
            avg_loss, match_loss, coll_loss, config_loss, target_loss = self.batch_train()
            avg_loss = avg_loss / (self.num_train / self.batch_size)
            print(
                f"point_match_loss: {match_loss:.6f} (weighted: {self.point_match_loss_weight * match_loss:.6f})\n"
                f"config_loss: {config_loss:.6f} (weighted: {self.config_loss_weight * config_loss:.6f})\n"
                f"collision_loss: {coll_loss:.6f} (weighted: {self.collision_loss_weight * coll_loss:.6f})\n"
                f"final_config_loss: {target_loss:.6f} (weighted: {self.final_config_loss_weight * target_loss:.6f})"
            )
            print(f"{GREEN}Epoch: {epoch}, Training Loss is: {avg_loss}{RESET}")

            avg_loss, match_loss, coll_loss, config_loss, target_loss  = self.batch_eval()
            eval_loss = avg_loss / (self.num_val / self.batch_size)
            print(
                f"point_match_loss: {match_loss:.6f} (weighted: {self.point_match_loss_weight * match_loss:.6f})\n"
                f"config_loss: {config_loss:.6f} (weighted: {self.config_loss_weight * config_loss:.6f})\n"
                f"collision_loss: {coll_loss:.6f} (weighted: {self.collision_loss_weight * coll_loss:.6f})\n"
                f"final_config_loss: {target_loss:.6f} (weighted: {self.final_config_loss_weight * target_loss:.6f})"
            )
            print(f"{CYAN}Epoch: {epoch}, Evaluation loss is: {eval_loss}{RESET}")


            self.scheduler.step()

            if epoch % 5 == 0:
                print(f"{RED}Saving model at epoch {epoch}...{RESET}")
                self.save_model(epoch_name=f'{epoch}')

            print("========================================================")

        self.train_loss = avg_loss 
        self.val_loss = eval_loss
        print('Training is complete. Now save the model.')
        # print loss weight
        print(f"point_match_loss_weight: {self.point_match_loss_weight}, "
              f"config_loss_weight: {self.config_loss_weight}, "
              f"collision_loss_weight: {self.collision_loss_weight}, "
              f"final_config_loss_weight: {self.final_config_loss_weight}")
        

        self.save_model()

    def save_model(self, epoch_name=None):
        path = './trained_models'
        file_name = 'neural_motion_planning'

        timestr = time.strftime("%m-%d-%H%M%S")
        torch.save({
            "epoch": self.num_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_loss": self.train_loss,
            "eval_loss": self.val_loss,
        }, os.path.join(path, f"{file_name}_{timestr}_epoch_{epoch_name}.pth"))

    

if __name__ == '__main__':
    # Example usage
    from samp.dataset import CustomDataset
    from samp.loss import MathcLossAndCollsionLoss
    from samp.model import MotionPlanningNetwork

    dataset = CustomDataset('processed_dataset.pkl', gaussian_normalization=True)
    loss_fn = MathcLossAndCollsionLoss()
    model = MotionPlanningNetwork(q_feature_size=14, anchor_size=15, output_size=7)
    
    trainer = Trainer(dataset, model, loss_fn, lr=1e-4, num_epochs=50, batch_size=256, weight_decay=0.0)
    trainer.save_model(10)