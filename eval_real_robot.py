"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import traceback
import dill
import hydra
import logging
import pathlib
from omegaconf import OmegaConf
import scipy.spatial.transform as st

from renv.ril_env.spacemouse import Spacemouse
from renv.ril_env.keystroke_counter import KeystrokeCounter, Key, KeyCode
from renv.ril_env.precise_sleep import precise_wait
from renv.ril_env.xarm_controller import XArmConfig, XArm
from renv.ril_env.real_env import RealEnv

from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OmegaConf.register_new_resolver("eval", eval, replace=True)

def get_real_obs_dict_custom(env_obs, shape_meta, n_obs_steps=2):
    """A modified version of get_real_obs_dict that works with the actual keys in the environment"""
    # Print raw camera shapes
    # print("Raw env_obs camera shapes:")
    # for key in env_obs:
        # if key.startswith('camera_'):
            # print(f"{key}: {env_obs[key].shape}, dtype: {env_obs[key].dtype}")
    
    result = dict()
    
    # Process camera observations
    for key, value in shape_meta['obs'].items():
        if key.startswith('camera_'):
            if key in env_obs:
                # Get expected shape from shape_meta
                expected_channels = value['shape'][0]
                expected_height = value['shape'][1]
                expected_width = value['shape'][2]
                
                # Get last n_obs_steps observations from buffer
                this_data_in = env_obs[key][-n_obs_steps:]
                
                # Process each image in the sequence
                processed_images = []
                for img in this_data_in:
                    # Resize image to expected dimensions
                    resized = cv2.resize(img, (expected_width, expected_height))
                    # Transform from HWC to CHW format
                    transposed = np.transpose(resized, (2, 0, 1))
                    processed_images.append(transposed)
                
                # Stack processed images
                processed_data = np.stack(processed_images)
                # Normalize if needed
                if processed_data.dtype == np.uint8:
                    processed_data = processed_data.astype(np.float32) / 255.0
                
                result[key] = processed_data
    
    # Process low_dim state using correct keys mapping
    key_mapping = {
        'robot_eef_pose': 'TCPPose',
        'robot_eef_pose_vel': 'TCPSpeed',
        'robot_joint': 'JointAngles',
        'robot_joint_vel': 'JointSpeeds',
        'robot_gripper': 'Grasp',
        'robot_timestamp': 'robot_receive_timestamp'
    }
    
    for actual_key, model_key in key_mapping.items():
        if actual_key in env_obs and model_key in shape_meta['obs']:
            this_data_in = env_obs[actual_key][-n_obs_steps:]
            result[model_key] = this_data_in
    
    # Add stage with guaranteed correct shape
    if 'stage' in shape_meta['obs']:
        feature_dim = shape_meta['obs']['stage']['shape'][0]
        this_data_in = np.zeros((n_obs_steps, feature_dim), dtype=np.float32)
        result['stage'] = this_data_in
    
    # Add debugging to print shapes
    # print("Processed observation shapes:")
    # for k, v in result.items():
        # if isinstance(v, np.ndarray):
            # print(f"{k}: {v.shape}, dtype: {v.dtype}")
    
    return result
# @click.command()
# @click.option('--input', '-i', required=False, help="Ckpt path")
# Update the main function to add support for interpolation controller
def main(
    input="./checkpoints/epoch=0050-train_loss=0.026.ckpt",
    output="./output/",
    init_joints=True,
    frequency=30,
    command_latency=0.01,
    steps_per_inference=2,
    record_res=(1280, 720),
    spacemouse_deadzone=0.05,
    use_interpolation=True,  # Enable interpolation by default for smoother motion
    max_pos_speed=0.25,      # Position speed limit (m/s)
    max_rot_speed=0.6,       # Rotation speed limit (degrees/s)
):
    dt = 1.0 / frequency
    output_dir = pathlib.Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Setup for different policy types
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16  # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    xarm_config = XArmConfig()

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    logger.info(f"n_obs_steps: {n_obs_steps}")
    logger.info(f"steps_per_inference: {steps_per_inference}")
    logger.info(f"action_offset: {action_offset}")
    logger.info(f"use_interpolation: {use_interpolation}")

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, Spacemouse(
            deadzone=spacemouse_deadzone, shm_manager=shm_manager
        ) as sm, RealEnv(
            output_dir=output_dir,
            xarm_config=xarm_config,
            frequency=frequency,
            num_obs_steps=n_obs_steps,
            obs_image_resolution=record_res,
            max_obs_buffer_size=30,
            obs_float32=True,
            init_joints=init_joints,
            video_capture_fps=30,
            video_capture_resolution=record_res,
            record_raw_video=True,
            thread_per_video=3,
            video_crf=21,
            enable_multi_cam_vis=False,
            multi_cam_vis_resolution=(1280, 720),
            shm_manager=shm_manager,
            use_interpolation=use_interpolation,  # Use the new interpolation controller
            max_pos_speed=max_pos_speed,
            max_rot_speed=max_rot_speed,
        ) as env:
            logger.info("Configuring camera settings...")
            env.realsense.set_exposure(exposure=120, gain=0)
            env.realsense.set_white_balance(white_balance=5900)

            # Warm up policy inference
            logger.info("Warming up policy inference...")
            obs = env.get_obs()

            
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict_custom(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, n_obs_steps=n_obs_steps)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 7  # xy position
                del result

            time.sleep(1)
            logger.info("System initialized and ready!")

            while True:
                # ========= Human control loop ==========
                logger.info("Human in control!")
                state = env.get_robot_state()
                target_pose = np.array(state["TCPPose"], dtype=np.float32)
                logger.info(f"Initial pose: {target_pose}")

                t_start = time.monotonic()
                iter_idx = 0
                stop = False
                is_recording = False

                try:
                    while not stop:
                        # Calculate timing
                        t_cycle_end = t_start + (iter_idx + 1) * dt
                        t_command_target = t_cycle_end + dt
                        t_sample = t_cycle_end - command_latency

                        # Get observations
                        obs = env.get_obs()

                        # Handle key presses
                        press_events = key_counter.get_press_events()

                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char="q"):
                                logger.info("Quit requested...")
                                env.end_episode()
                                exit(0)
                            elif key_stroke == KeyCode(char="c"):
                                # Exit human control loop, hand control to policy
                                stop = True
                                break

                        stage_val = key_counter[Key.space]

                        # Visualize
                        episode_id = env.replay_buffer.n_episodes

                        precise_wait(t_sample)

                        # Get spacemouse state
                        sm_state = sm.get_motion_state_transformed()

                        dpos = sm_state[:3]
                        drot = sm_state[3:]
                        grasp = sm.grasp

                        # Check if movement is significant
                        input_magnitude = np.linalg.norm(dpos) + np.linalg.norm(drot)
                        significant_movement = input_magnitude > spacemouse_deadzone * 8.0
                        if significant_movement:
                            dpos *= xarm_config.position_gain
                            drot *= xarm_config.orientation_gain

                            curr_rot = st.Rotation.from_euler(
                                "xyz", target_pose[3:], degrees=True
                            )
                            delta_rot = st.Rotation.from_euler("xyz", drot, degrees=True)
                            final_rot = delta_rot * curr_rot

                            target_pose[:3] += dpos
                            target_pose[3:] = final_rot.as_euler("xyz", degrees=True)

                            action = np.concatenate([target_pose, [grasp]])

                            exec_timestamp = (t_command_target - time.monotonic() + time.time())
                            
                            # Use interpolation if enabled
                            if use_interpolation:
                                # print("USING.")
                                env.exec_action_waypoints(
                                    actions=[action],
                                    timestamps=[exec_timestamp],
                                    stages=[stage_val],
                                )
                            else:
                                env.exec_actions(
                                    actions=[action],
                                    timestamps=[exec_timestamp],
                                    stages=[stage_val],
                                )
                            logger.debug("Significant movement detected, executing action.")
                        else:
                            action = np.concatenate([target_pose, [grasp]])
                            exec_timestamp = (t_command_target - time.monotonic() + time.time())
                            
                            # Use interpolation if enabled
                            if use_interpolation:
                                # print("USING.")
                                env.exec_action_waypoints(
                                    actions=[action],
                                    timestamps=[exec_timestamp],
                                    stages=[stage_val],
                                )
                            else:
                                env.exec_actions(
                                    actions=[action],
                                    timestamps=[exec_timestamp],
                                    stages=[stage_val],
                                )
                            logger.debug("No significant movement detected.")

                        precise_wait(t_cycle_end)
                        iter_idx += 1

                    # ========== Policy control loop ==============
                    # Start policy evaluation
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # Wait for 1/30 sec to get the closest frame
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    logger.info("Policy evaluation started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    prev_target_pose = None
                    is_recording = True
                    
                    while True:
                        # Calculate timing for policy control
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # Get observations
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        logger.debug(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # Check for key presses during policy control
                        press_events = key_counter.get_press_events()
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char="s"):
                                # Stop episode, hand control back to human
                                env.end_episode()
                                is_recording = False
                                logger.info("Policy evaluation stopped.")
                                break
                            elif key_stroke == Key.backspace:
                                if click.confirm("Drop the most recently recorded episode?"):
                                    env.drop_episode()
                                    is_recording = False
                                    logger.info("Episode dropped.")

                        if not is_recording:
                            break

                        # Run policy inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict_custom(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            action = result['action'][0].detach().to('cpu').numpy()
                            logger.debug(f'Inference latency: {time.time() - s}')
                        
                        # Convert policy action to robot actions
                        if delta_action:
                            assert len(action) == 1
                            if prev_target_pose is None:
                                prev_target_pose = obs['robot_eef_pose'][-1]
                            this_target_pose = prev_target_pose.copy()
                            this_target_pose[[0,1]] += action[-1]
                            prev_target_pose = this_target_pose
                            this_target_poses = np.expand_dims(this_target_pose, axis=0)
                        else:
                            this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float32)
                            this_target_poses[:] = target_pose
                            this_target_poses = action[:,:6]

                        # Handle timing for actions
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 1
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # Exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # Schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            logger.debug(f'Over budget: {action_timestamp - curr_time}')
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # Execute actions
                        full_actions = []
                        for i in range(len(this_target_poses)):
                            # Add grasp parameter
                            full_action = np.concatenate([this_target_poses[i], [grasp]])
                            full_actions.append(full_action)
                            
                        # Use interpolation if enabled
                        if use_interpolation:
                            env.exec_action_waypoints(
                                actions=full_actions,
                                timestamps=action_timestamps,
                                stages=[stage_val] * len(action_timestamps)
                            )
                        else:
                            for i in range(len(full_actions)):
                                env.exec_actions(
                                    actions=[full_actions[i]],
                                    timestamps=[action_timestamps[i]],
                                    stages=[stage_val]
                                )
                        logger.info(f"Submitted {len(this_target_poses)} steps of actions.")

                        # Visualize
                        episode_id = env.replay_buffer.n_episodes

                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    logger.info("Interrupted!")
                    env.end_episode()
                except Exception:
                    logger.error("Exception occurred during control loop:")
                    traceback.print_exc()
                finally:
                    if is_recording:
                        env.end_episode()
                    logger.info("Control loop ended. Returning to human control.")

if __name__ == "__main__":
    main()
