#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


# Explanation:
# This script is to continuously generate a target for the robot to plan.
# 

try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

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
from samp.helper import add_extensions, add_robot_to_scene
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
from curobo.util.usd_helper import UsdHelper
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

########### OV #################
from scipy.spatial.transform import Rotation as R
import os
import time
import pickle
from samp.utils import generate_anchor_points
################################


def main():
    # create a curobo motion gen instance:
    num_targets = 0
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
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

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

    articulation_controller = None

    world_file = "dataset_generation_scene_1.yml" 
    world_cfg = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), world_file))
    )
    vis_world_cfg = world_cfg.get_mesh_world()

    ######################

    file_dir = get_world_configs_path()
    # print(robot_cfg_path)
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
        enable_graph=True,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5 if not args.reactive else 1.0,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(vis_world_cfg, base_frame="/World")

    ###################### anchor points ##########################
    anchor_point_size = [100, 100, 100]
    anchor_points = tensor_args.to_device(generate_anchor_points(anchor_point_size[0], 
                                                                 anchor_point_size[1], 
                                                                 anchor_point_size[2], 
                                                                 1.0, 
                                                                 1.0, 
                                                                 1.2))

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

    cmd_plan = None
    my_world.scene.add_default_ground_plane()
    i = 0
    past_cmd = None
    target_orientation = None
    past_orientation = None
    pose_metric = None
    
    ########################### anchor points for the environment ##########################
    radius = 0.0
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


    # preset parameters
    first_run = True
    fail_count = 0
    max_fail_count = 8
    distances = None

    record_cnt = 0
    record_num = 2000
    dataset = []

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                my_world.play()
                # print("**** Click Play to start simulation *****")
            i += 1
            continue

        step_index = my_world.current_time_step_index
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
        if step_index < 10:
            robot.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue
        

        if step_index == 21 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()

            motion_gen.update_world(obstacles)
            sim_sdf_model.update_world(obstacles)

            ###################### get environment sdf
            distances = sim_sdf_model.get_collision_distance(anchor_spheres)
            distances = distances.view(-1).cpu()
            distances = act_distance/2 - distances
            max_val = distances.max().item()
            min_val = distances.min().item()
            print(f"max: {max_val}, min: {min_val}")
            # draw_sdf_points(anchor_pose, distances)

            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        ############## position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()
        
        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        if target_orientation is None:
            target_orientation = cube_orientation
        if past_orientation is None:
            past_orientation = cube_orientation

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
            ee_translation_goal = np.array([
                np.random.uniform(-0.8, 0.8),    # x
                np.random.uniform(-0.8, 0.8),    # y
                np.random.uniform(0.25, 1.10),   # z
            ], dtype=np.float32)
            if ee_translation_goal[2] < 0.35:
                pitch = np.random.uniform(-45, 45)
                yaw = np.random.uniform(-180, 180)
                roll = np.random.uniform(-20, 20) + 180
                down_rot = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
                ee_orientation_teleop_goal = down_rot.as_quat().astype(np.float32)
            else:
                ee_orientation_teleop_goal = R.random().as_quat().astype(np.float32)

            target.set_world_pose(
                position=ee_translation_goal,
                orientation=ee_orientation_teleop_goal
            )

            # compute curobo solution:
            goal_pose = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )

            plan_config.pose_cost_metric = pose_metric
            # start_time = time.time()
            result = motion_gen.plan_single(cu_js.unsqueeze(0), goal_pose, plan_config)
            # end_time = time.time()
            # print(f"Planning time: {end_time - start_time:.3f} sec")
            
            succ = result.success.item() 


            sim_js = robot.get_joints_state()

            if num_targets == 1:
                if args.constrain_grasp_approach:
                    pose_metric = PoseCostMetric.create_grasp_approach_metric()
                if args.reach_partial_pose is not None:
                    reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
                    pose_metric = PoseCostMetric(
                        reach_partial_pose=True, reach_vec_weight=reach_vec
                    )
                if args.hold_partial_pose is not None:
                    hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
                    pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
            if succ:
                num_targets += 1
                fail_count = 0
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)

                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                if cmd_plan.position.shape[0] > 55:
                    cmd_plan = cmd_plan[-55:-5]
                else:
                    print("Command plan is too short, skipping this iteration.")
                    continue

                if step_index >= 22 and distances is not None:

                    joint_states = JointState(
                        position=cmd_plan.position,  
                        joint_names=common_js_names, 
                        tensor_args=motion_gen.tensor_args
                    )
                    joint_states = joint_states.get_ordered_joint_state(motion_gen.kinematics.joint_names)
                    ee_pose = motion_gen.kinematics.compute_kinematics(joint_states)
                    ee_positions = ee_pose.ee_position.cpu().numpy()      # [T, 3]
                    ee_orientations = ee_pose.ee_quaternion.cpu().numpy() # [T, 4]

                    #  cat the ee positions and orientations
                    ee_pose = np.concatenate((ee_positions, ee_orientations), axis=1)
                    data_item = {
                        "joint_trajectory": cmd_plan.position.cpu().numpy().copy(),
                    }

                    dataset.append(data_item)

                    # display the position (only show the target position)
                    if cmd_plan is not None:
                        final_cmd_state = cmd_plan[-1]
                        robot.set_joint_positions(
                            final_cmd_state.position.cpu().numpy(),
                            idx_list  
                        )
                        for _ in range(2):
                            my_world.step(render=False)
                        cmd_plan = None
                        past_cmd = None
                        robot_static = True

                    record_cnt += 1
                if record_cnt % 50 == 0:
                    print(f"Record counnt/dataset size: {record_cnt}/{record_num}")
                
                # Save the environment sdf
                if record_cnt >= record_num:
                    env_dataset = []
                    anchor_size = [40, 70, 100, 130, 160]

                    for i in range(len(anchor_size)):
                        size = anchor_size[i]
                        anchor_points = tensor_args.to_device(generate_anchor_points(size, size, size, 1.0, 1.0, 1.2))
                        anchor_spheres = torch.cat([
                            anchor_points, 
                            torch.full((anchor_points.shape[0], 1), radius, device=anchor_points.device)
                        ], dim=1).unsqueeze(0).unsqueeze(2) 
                        distances = sim_sdf_model.get_collision_distance(anchor_spheres)
                        distances = distances.view(-1).cpu()
                        distances = act_distance/2 - distances
                        distances = distances.view(size, size, size, -1).permute(3, 0, 1, 2)
                        max_val = distances.max().item()
                        min_val = distances.min().item()
                        print(f"max: {max_val}, min: {min_val}")

                        data_item = {
                            "environment_sdf_"+str(size): distances.cpu().numpy().copy(),
                        }
                        print("environment_sdf_"+str(size))
                        env_dataset.append(data_item)
                    
                    dataset_dir = './isaac_sim_traj'
                    os.makedirs(dataset_dir, exist_ok=True)
                    # assign dataset name with the world file name number
                    scene_id = int(world_file.split('_')[-1].split('.')[0])
                    scene_id_str = f"{scene_id:04d}"

                    dataset_name = f"dataset{scene_id_str}.pkl"
                    with open(os.path.join(dataset_dir, dataset_name), "wb") as f:
                        pickle.dump(dataset, f)

                    env_dataset_name = f"env_dataset_{scene_id_str}.pkl"
                    with open(os.path.join(dataset_dir, env_dataset_name), "wb") as f:
                        pickle.dump(env_dataset, f)
                    print("Dataset saved to" + os.path.join(dataset_dir, dataset_name))
                    return

            else:
                fail_count += 1
                if fail_count > max_fail_count:
                    print("!!!!!!!!!!!! reset the robot !!!!!!!!!!!!!")
                    my_world.reset()
                    fail_count = 0
                    robot_static = True
                    continue

            target_pose = cube_position
            target_orientation = cube_orientation
        past_pose = cube_position
        past_orientation = cube_orientation

    simulation_app.close()


if __name__ == "__main__":
    main()

    
