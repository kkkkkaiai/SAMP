try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch
import time

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")

parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)

parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)

parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)

parser.add_argument(
    "--reach_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Reach partial pose",
    type=float,
    default=None,
)
parser.add_argument(
    "--hold_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Hold partial pose while moving to goal",
    type=float,
    default=None,
)

args = parser.parse_args()

############################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
# Standard Library
from typing import Dict

# Third Party
import carb
import numpy as np
from samp.helper import add_extensions, add_robot_to_scene, add_urdf_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper, set_prim_transform
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from omni.isaac.core.materials import OmniPBR
from omni.isaac.debug_draw import _debug_draw

########### OV #################

from robot_layer.panda_layer import PandaLayer
from encode_panda_sdf_points_sampler import HashNNSDF, HashMLPModel
from scipy.spatial.transform import Rotation as R
import os
import glob
from robofin.pointcloud.torch import FrankaSampler
from robofin.robots import FrankaRobot


from samp.utils import normalize_franka_joints, unnormalize_franka_joints, so3_angle_diff
from samp.utils import generate_anchor_points
from samp.model import MotionPlanningNetwork
from samp.loss import collision_loss
################################

def draw_sdf_points(pose, sdf, max_dist=0.5, show_range=0.05):
    # Third Party
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_points()
    if hasattr(pose, "position"):
        cpu_pos = pose.position.cpu().numpy()
    elif isinstance(pose, torch.Tensor):
        cpu_pos = pose.detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported pose type: {type(pose)}")
    sdf_np = sdf.detach().cpu().numpy().reshape(-1)

    if cpu_pos.ndim == 3 and cpu_pos.shape[0] == 1:
        cpu_pos = cpu_pos.squeeze(0)

    # only contain points that their |SDF| < show_range 
    mask = np.abs(sdf_np) < show_range
    cpu_pos = cpu_pos[mask]
    sdf_np = sdf_np[mask]

    b = cpu_pos.shape[0]
    point_list = []
    colors = []

    norm_isdf = np.clip(sdf_np / max_dist, 0, 1)

    for i in range(b):
        point_list.append((cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2]))
        t = norm_isdf[i]
        r = 1 - t
        g = t
        b_col = 0
        colors.append((r, g, b_col, 0.9))
    sizes = [20.0 for _ in range(b)]

    draw.draw_points(point_list, colors, sizes)

def draw_traj_list(traj_list, color=(1.0, 0.0, 0.0, 1.0), size=5.0):
    # Third Party
    draw = _debug_draw.acquire_debug_draw_interface()
    draw.clear_lines()
    point_list = []
    colors = []
    sizes = []

    for i, traj in enumerate(traj_list):
        point_list.append((traj[0], traj[1], traj[2]))
        colors.append(color)
        sizes.append(size)

    draw.draw_points(point_list, colors, sizes)

def grid_normalize_points(points, min_xyz=np.array([-1.0, -1.0, 0.05]), max_xyz=np.array([1.0, 1.0, 1.25])):
    # points: [N, 3]
    normed = 2 * (points - min_xyz) / (max_xyz - min_xyz) - 1
    if isinstance(normed, np.ndarray):
        normed = np.clip(normed, -1.0, 1.0)
    else:
        normed = torch.clamp(normed, -1.0, 1.0)
    return normed

def so3_angle_diff(q1, q2):
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    r_rel = r1.inv() * r2
    angle = r_rel.magnitude()
    return angle

def main():
    # create a curobo motion gen instance:
    # num_targets = 0
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]


    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    gripper, gripper_prim_path = add_urdf_to_scene(urdf_path = "robot/franka_description",
                                                    urdf_filename = "panda_gripper.urdf",
                                                    my_world=my_world,
                                                    base_link_name="gripper",
                                                    color=[0, 1, 0])

    articulation_controller = None

    collision_type = 0
    if collision_type == 0:
    ######################### Type 1: load mesh (use mesh world)
        world_file = "dataset_generation_scene_1.yml"   # scene chosen
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        vis_world_cfg = world_cfg.get_mesh_world()
    else:
    ######################### Type 2: load nvblox (use mesh generated from nvblox)
        world_file = "collision_nvblox.yml" 
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        world_cfg.objects[0].pose[2] += 0.2
        vis_world_cfg = world_cfg.get_mesh_world()

    ######################

    file_dir = get_world_configs_path()
    file_name = "generated_scene_" + world_file.split("_")[-1].split(".")[0] + ".obj"
    file_path = os.path.join(file_dir, file_name)
    print("Saving world as mesh to", file_path)
    world_cfg.save_world_as_mesh(file_path)

    mesh_cfg = {
        "mesh": {
            "base_scene": {
                "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "file_path": file_path,
            }
        }
    }
    world_cfg = WorldConfig.from_dict(
        mesh_cfg,
    )
    vis_world_cfg = world_cfg.get_mesh_world()

    ######################### Type 3: load motion plan

    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4
    interpolation_dt = 0.05
    enable_finetune_trajopt = True
    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt
        enable_finetune_trajopt = False
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_file,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    motion_gen = MotionGen(motion_gen_config)
    if not args.reactive:
        print("warming up...")
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5 if not args.reactive else 1.0,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(vis_world_cfg, base_frame="/World")

    ################ load robot sdf model #########################
    device = 'cuda'
    lr = 0.002
    panda = PandaLayer(device)
    hash_nn_sdf = HashNNSDF(panda,lr,device)
    model_path = os.path.dirname(os.path.realpath(__file__)) + '/models/SAMP_USDF_ALL.pt'
    sdf_model = torch.load(model_path, weights_only=False)

    ###################### anchor points ##########################
    points_size = 100
    anchor_points = tensor_args.to_device(generate_anchor_points(points_size, points_size, points_size, 1.0, 1.0, 1.2))
    ###################### sim model to get distance ##############
    act_distance = 3.0
    collision_checker_type = CollisionCheckerType.MESH

    sdf_config = RobotWorldConfig.load_from_config(
        args.robot,
        world_cfg,
        collision_activation_distance=act_distance,
        collision_checker_type = collision_checker_type,
    )

    sim_sdf_model = RobotWorld(sdf_config)
    my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None
    past_cmd = None
    target_orientation = None
    past_orientation = None
    pose_metric = None
    
    ########################### anchor points for the environment ##########################
    radius = 0.01
    anchor_spheres = torch.cat([
        anchor_points, 
        torch.full((anchor_points.shape[0], 1), radius, device=anchor_points.device)
    ], dim=1).unsqueeze(0).unsqueeze(2)  # [1, n, 1, 4]

    anchor_pose = Pose(
        position=tensor_args.to_device(np.zeros((3,), dtype=np.float32)),
        quaternion=tensor_args.to_device(np.array([0, 0, 0, 1], dtype=np.float32)),
    )
    anchor_pose = anchor_pose.repeat(anchor_points.shape[0])
    anchor_pose.position += anchor_points

    ########################### Motion Planning Network ####################################

    planning_model = MotionPlanningNetwork(
        q_feature_size=14,
        anchor_size=points_size,
        output_size=7,
    )
    # Load the trained model weights
    model_dir = "./trained_models/"
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    if len(model_files) == 0:
        print(f"No model weights found in {model_dir}, using default weights.")
    else:
        latest_model = max(model_files, key=os.path.getmtime)
        print(f"Loading model weights from {latest_model}")
        checkpoint = torch.load(latest_model, map_location=device)
    
    planning_model.load_state_dict(checkpoint["model_state_dict"])
    planning_model.to(device)
    planning_model.eval()  # Set the model to evaluation mode
    # Define the pose metric for the motion generation

    # preset parameters
    first_run = True
    stuck_cnt = 0
    max_stuck_cnt = 20
    last_joint_diff = None

    fk_sampler = FrankaSampler(
        'cuda',
        num_fixed_points=1024,
        use_cache=True,
        with_base_link=False,
    )

    eps = 1e-6

    dataset_cnt = 500 # evaluat the dataset starting from this number
    ee_pos = None
    ee_quat = None

    ee_traj_list = []
    q_traj_list = []
    comp_time_list = []
    success_list = []
    success_list_clean = []
    intensity_list = []
    total_plan_time_list = []
    path_length_list = []
    goal_reach = False

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            ee_traj_list = []
            q_traj_list = []
            comp_time_list = []
            i += 1
            continue

        step_index = my_world.current_time_step_index
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
        if step_index < 10:
            # robot._articulation_view.initialize()
            robot.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

            gripper.initialize()
            if ee_pos is not None and ee_quat is not None:
                gripper.set_world_pose(
                    position=ee_pos,
                    orientation=ee_quat,
                )

        if step_index < 20:
            continue

        ############## update world and obstacles ##################
        if step_index == 20: # or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()

            motion_gen.update_world(obstacles)
            sim_sdf_model.update_world(obstacles)


            distances = sim_sdf_model.get_collision_distance(anchor_spheres)
            distances = distances.view(-1).cpu()
            distances = sim_sdf_model.contact_distance - distances
            max_val = distances.max().item()
            min_val = distances.min().item()
            print(f"max: {max_val}, min: {min_val}")
            # draw_sdf_points(anchor_pose, distances)

            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        ############## position and orientation of target virtual cube:


        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names

        if np.any(np.isnan(sim_js.positions)):
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),  # * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )

        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0

        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if first_run:
            robot_static = True
            first_run = False

        if robot_static:
            with open("./isaac_sim_traj/dataset0001.pkl", 'rb') as f:  # scene chosen
                import pickle
                traj_data = pickle.load(f)
                # dataset_cnt = 109
                target_joint_solution = torch.tensor(traj_data[dataset_cnt]['joint_trajectory'])
                target_joint_solution = target_joint_solution[-1, :-2]
                target_joint_solution = target_joint_solution.to(device=motion_gen.tensor_args.device)
                start_joint = torch.tensor(traj_data[dataset_cnt]['joint_trajectory'][0, :-2])
                start_joint = start_joint.to(device=motion_gen.tensor_args.device)

            ee_traj_list = []
            q_traj_list = []
            comp_time_list = []

            traj_goal = JointState(
                position=target_joint_solution,
                joint_names=motion_gen.kinematics.joint_names,
                tensor_args=motion_gen.tensor_args,
            )

            fk_check_result = traj_goal.get_ordered_joint_state(motion_gen.kinematics.joint_names)
            fk_result = motion_gen.kinematics.compute_kinematics(fk_check_result)
            ee_pos = fk_result.ee_position.cpu().numpy().reshape(-1).astype(np.float32)
            ee_quat = fk_result.ee_quaternion.cpu().numpy().reshape(-1).astype(np.float32) 

            # when the position on z axis is too low, skip it
            if ee_pos[2] < 0.1:
                print("Skipping this target position due to low z value:", ee_pos[2])
                dataset_cnt += 1
                continue

            gripper.set_world_pose(
                position=ee_pos,
                orientation=ee_quat,
            )


            robot_static = False

            full_joint = sim_js.positions.copy()
            full_joint[:7] = start_joint.cpu().numpy()
            art_action = ArticulationAction(
                full_joint,
                joint_indices=idx_list
            )
            articulation_controller.apply_action(art_action)
            max_wait_steps = 100  
            wait_count = 0

            while wait_count < max_wait_steps:
                for _ in range(1): 
                    my_world.step(render=False)
                
                current_js = robot.get_joints_state()
                current_position = current_js.positions[:7]
                target_position = start_joint.cpu().numpy()

                position_diff = np.abs(current_position - target_position)
                wait_count += 1
            dataset_cnt += 1


        else:
            ###################### get robot sdf
            sim_js = robot.get_joints_state()
            theta = torch.tensor(sim_js.positions, device=device).reshape(1, -1)

            pose = torch.from_numpy(np.identity(4)).unsqueeze(0).to(device).expand(len(theta),4,4).float()
            robot_points = hash_nn_sdf.batch_robot_points(theta, device="cuda")
            robot_sdf = hash_nn_sdf.whole_body_nn_sdf_batch_points(robot_points, pose, theta, sdf_model)
            robot_sdf = torch.clamp(robot_sdf, min=-0.5, max=1.0)
            
            robot_feature = robot_points.unsqueeze(1)
            
            # import pickle
            # with open("./isaac_sim_traj/env_dataset_0003.pkl", 'rb') as f:  # scene chosen
            #     env_data = pickle.load(f)
            # obs_feature = torch.tensor(env_data[0]['environment_sdf_100'], device=device)  
            obs_feature = distances.view(points_size, points_size, points_size, -1).permute(3, 0, 1, 2)
            obs_feature = obs_feature.clone().detach().to(device=device, dtype=torch.float32)
            obs_feature = torch.clamp(obs_feature, min=-0.5, max=1.0)
            # obs_feature = -torch.ones_like(obs_feature) * 0.1  # use a constant value for testing
            

            grid_norm_robot_points = grid_normalize_points(robot_points.cpu().numpy())
            grid_norm_robot_points = torch.tensor(grid_norm_robot_points, device=device, dtype=torch.float32)
            new_robot_points = grid_norm_robot_points.clone()
            new_robot_points[:, :, 0] = grid_norm_robot_points[:, :, 2]
            new_robot_points[:, :, 2] = grid_norm_robot_points[:, :, 0] 

            # use grid_sample to get the sdf values at the robot points
            grid_obs_feature = torch.nn.functional.grid_sample(
                obs_feature.unsqueeze(0),
                new_robot_points.unsqueeze(0).unsqueeze(0),
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )

            # draw the sdf points
            draw_sdf_points(
                pose = robot_points,
                sdf = grid_obs_feature,
                max_dist=0.15,
                show_range=0.05
            )   

            sim_js_tensor = torch.tensor(sim_js.positions).to(device)  # [9]
            target_joint_solution_tensor = target_joint_solution.squeeze(0)

            norm_sim_js_tensor = normalize_franka_joints(sim_js_tensor[:7])
            norm_target_js_tensor = normalize_franka_joints(target_joint_solution_tensor)

            q_feature = torch.cat([
                norm_sim_js_tensor,  # current joint state
                norm_target_js_tensor,  # goal joint state
            ]).unsqueeze(0) 

            infer_time = time.time()
            q_output = planning_model(
                q_feature.to(device), 
                robot_feature.unsqueeze(0).to(device), 
                obs_feature.unsqueeze(0).to(device),
                robot_sdf.to(device)
            )
            infer_time = time.time() - infer_time
            pred_theta = q_output + norm_sim_js_tensor
            pred_theta = torch.clamp(pred_theta, min=-1.0, max=1.0)
            pred_theta_unnorm = unnormalize_franka_joints(pred_theta)

            theta_diff = pred_theta_unnorm - sim_js_tensor[:7]
            # add limit on theta diff
            # theta_diff = torch.clamp(theta_diff, min=-0.01, max=0.01)
            pred_theta_unnorm = sim_js_tensor[:7] + theta_diff

            pred_theta_unnorm_cpu = pred_theta_unnorm.detach().cpu().numpy()
            q_output_cpu = pred_theta_unnorm_cpu

            full_joint_target = sim_js.positions.copy()
            full_joint_target[:7] = pred_theta_unnorm_cpu


            art_action = ArticulationAction(
                full_joint_target,
                joint_indices=idx_list
            )

            articulation_controller.apply_action(art_action)
            max_wait_steps = 100  
            wait_count = 0

            for _ in range(5): 
                my_world.step(render=False)

            # here, I want to display the positions of end-effector with the red sphere
            sim_js = robot.get_joints_state()
            if sim_js is not None:
                current_actual_positions = sim_js.positions
                current_joint_tensor = torch.tensor(current_actual_positions[:7], device=device)

                temp_ee_position = JointState(
                    position=current_joint_tensor,
                    joint_names=motion_gen.kinematics.joint_names,
                    tensor_args=motion_gen.tensor_args,
                )

                ee_check_result = temp_ee_position.get_ordered_joint_state(motion_gen.kinematics.joint_names)
                ee_result = motion_gen.kinematics.compute_kinematics(ee_check_result)
                ee_cur_pos = ee_result.ee_position.cpu().numpy().reshape(-1).astype(np.float32)
                # ee_cur_quat = ee_result.ee_quaternion.cpu().numpy().reshape(-1).astype(np.float32) 

                ee_traj_list.append(ee_cur_pos)
                draw_traj_list(ee_traj_list, color=(0.0, 1.0, 0.0, 1.0), size=15.0)
                q_traj_list.append(current_joint_tensor.cpu().numpy().reshape(-1).astype(np.float32))
                comp_time_list.append(infer_time)

            
            if wait_count >= max_wait_steps:
                print(f"Warn: max waiting step {max_wait_steps}")
                print(f"Position error: {position_diff.max():.5f}")


            # time.sleep(0.02)  
            ####################### check if the robot has reached the goal
            full_dof = len(motion_gen.kinematics.joint_names)

            sim_js = robot.get_joints_state()
            sim_js = robot.get_joints_state()
            if sim_js is None:
                print("sim_js is None, skip this step.")
                continue
            current_actual_positions = sim_js.positions.copy()
            current_joint_tensor = torch.tensor(current_actual_positions[:7], device=device)
            full_current_tensor = torch.zeros((1, full_dof), device=device)
            full_current_tensor[0, :7] = current_joint_tensor

            target_joints_numpy = target_joint_solution_tensor[:7].detach().cpu().numpy()  
            target_joint_tensor = torch.tensor(target_joints_numpy, device=device) 
            full_target_tensor = torch.zeros((1, full_dof), device=device)
            full_target_tensor[0, :7] = target_joint_tensor

            
            with torch.no_grad(): 
                current_joints_list = full_current_tensor[0, :7].cpu().tolist()
                target_joints_list = full_target_tensor[0, :7].cpu().tolist()

                eff_pose_check = FrankaRobot.fk(current_joints_list,eff_frame="panda_link8")
                target_pose_check = FrankaRobot.fk(target_joints_list,eff_frame="panda_link8")
                
                pos_err = np.linalg.norm(eff_pose_check._xyz - target_pose_check._xyz)
                angle_err = np.abs(
                    np.degrees((eff_pose_check.so3._quat * target_pose_check.so3._quat.conjugate).radians)
                )
                
                if (pos_err < 0.02 and angle_err < 15):
                    print("Goal reach!")
                    stuck_cnt = 0
                    # print("dataset_cnt:", dataset_cnt)
                    robot_static = True
                    goal_reach = True

                elif (pos_err < 0.05 and angle_err < 30):
                    stuck_cnt += 1
                    # print(f"Position error: {pos_err:.4f}, Angle error: {angle_err:.4f} degrees")
                    if stuck_cnt > max_stuck_cnt:
                        robot_static = True
                        stuck_cnt = 0
                else:
                    cur_joint_solution = pred_theta_unnorm_cpu
                    cur_joint_solution_tensor = torch.tensor(cur_joint_solution, device=device)
                    joint_diff = torch.abs(cur_joint_solution_tensor - target_joint_solution_tensor[:7])
                    if last_joint_diff is not None:
                    # check the last joint diff is nearly the same as current joint diff
                        if torch.allclose(last_joint_diff, joint_diff, atol=0.002):
                            stuck_cnt += 1
                            if stuck_cnt > max_stuck_cnt:
                                print(f"Joint diff: {joint_diff}")
                                print()
                                robot_static = True
                        else:
                            stuck_cnt = 0
                    last_joint_diff = joint_diff
                
                if robot_static:
                    print(f"================={dataset_cnt}===================")
                    ee_traj_np = np.array(q_traj_list)  # [H, 3] 或 [H, 7]
                    ee_traj_tensor = torch.from_numpy(ee_traj_np).to(device).unsqueeze(0)
                    traj_mask = sim_sdf_model.validate_trajectory(ee_traj_tensor)
                    success = traj_mask.all().item()
                    if not goal_reach:
                        success = False
                    print("Success:", success)
                    success_list.append(success)

                    intensity = 1.0 - traj_mask.float().mean().item()
                    print(f"Trajectory collision intensity: {intensity:.4f}")
                    if intensity > 0.0:
                        intensity_list.append(intensity)
                        if goal_reach:
                            success = 1
                        success_list_clean.append(success)

                    ee_traj_np = np.array(ee_traj_list)
                    path_length = np.sum(np.linalg.norm(ee_traj_np[1:] - ee_traj_np[:-1], axis=1))
                    print(f"Path Length: {path_length:.4f}")
                    path_length_list.append(path_length)

                    if len(comp_time_list) > 0:
                        avg_comp_time = np.mean(comp_time_list)
                        min_comp_time = np.min(comp_time_list)
                        max_comp_time = np.max(comp_time_list)
                        print(f"Max, min, average Computation Time per Step: {max_comp_time:.4f}, {min_comp_time:.4f}, {avg_comp_time:.4f} seconds")
                        print(f"Time std: {np.std(comp_time_list):.4f} seconds")
                        total_plan_time = np.sum(comp_time_list)
                        total_plan_time_list.append(total_plan_time)
                        print(f"Total Planning Time: {total_plan_time:.4f} seconds")

                    if dataset_cnt % 100 == 0:
                        print("############################")
                        avg_success = np.mean(success_list)
                        avg_success_clean = np.mean(success_list_clean)
                        print(f"Average success rate until now: {avg_success*100:.2f}%")
                        print(f"Average clean success rate until now: {avg_success_clean*100:.2f}%")
                        print(f"Average total plan time until now: {np.mean(total_plan_time_list):.4f} seconds")
                        print(f"Std total plan time until now: {np.std(total_plan_time_list):.4f} seconds")
                        print(f"Mean path length until now: {np.mean(path_length_list):.4f} meters")
                        print(f"Std path length until now: {np.std(path_length_list):.4f} meters")
                        print(f"Mean intensity until now: {np.mean(intensity_list):.4f}")
                        print("############################")

                    goal_reach = False


    simulation_app.close()


if __name__ == "__main__":
    main()
