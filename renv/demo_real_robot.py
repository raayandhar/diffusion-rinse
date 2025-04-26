import time
import traceback
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
import logging
import pathlib

from multiprocessing.managers import SharedMemoryManager

from ril_env.spacemouse import Spacemouse
from ril_env.keystroke_counter import KeystrokeCounter, Key, KeyCode
from ril_env.precise_sleep import precise_wait
from ril_env.xarm_controller import XArmConfig, XArm
from ril_env.real_env import RealEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
TODO:
- Actually check that we are recording things
- Fix the camera visualization
- Fix the spacemouse-robot weird movement problem
- Clean up the code, add some documentation
- Demonstrations!
"""


def main(
    output="./recordings/",
    vis_camera_idx=0,
    init_joints=True,  # Not used ATM
    frequency=30,  # Cannot increase frequency
    command_latency=0.01,
    record_res=(1280, 720),
    spacemouse_deadzone=0.05,
):
    dt = 1.0 / frequency
    output_dir = pathlib.Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    xarm_config = XArmConfig()

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, Spacemouse(
            deadzone=spacemouse_deadzone, shm_manager=shm_manager
        ) as sm, RealEnv(
            output_dir=output_dir,
            xarm_config=xarm_config,
            frequency=frequency,
            num_obs_steps=2,
            obs_image_resolution=record_res,
            max_obs_buffer_size=30,
            obs_float32=True,
            init_joints=init_joints,
            video_capture_fps=30,
            video_capture_resolution=record_res,
            record_raw_video=True,
            thread_per_video=3,
            video_crf=21,
            enable_multi_cam_vis=False,  # Totally broken RN
            multi_cam_vis_resolution=(1280, 720),
            shm_manager=shm_manager,
        ) as env:
            logger.info("Configuring camera settings...")
            env.realsense.set_exposure(exposure=120, gain=0)
            env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1)
            logger.info("System initialized")

            state = env.get_robot_state()
            target_pose = np.array(state["TCPPose"], dtype=np.float32)
            logger.info(f"Initial pose: {target_pose}")

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False

            try:
                logger.info("Starting main control loop...")
                while not stop:

                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_command_target = t_cycle_end + dt
                    t_sample = t_cycle_end - command_latency

                    # Pump obs
                    obs = env.get_obs()
                    # Let's get camera data as well.
                    # Should find out what the obs dict is doing.
                    # print(obs)

                    press_events = key_counter.get_press_events()

                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char="q"):
                            logger.info("Quit requested...")
                            stop = True
                        elif key_stroke == KeyCode(char="c"):
                            env.start_episode()
                            is_recording = True
                            logger.info("Recording started!")
                        elif key_stroke == KeyCode(char="s"):
                            env.end_episode()
                            is_recording = False
                            logger.info("Recording stopped.")
                        elif key_stroke == Key.backspace:
                            if click.confirm(
                                "Drop the most recently recorded episode?"
                            ):
                                env.drop_episode()
                                is_recording = False
                                logger.info("Episode dropped.")

                    stage_val = key_counter[Key.space]

                    # vis_img = obs[f"camera_{vis_camera_idx}"][-1, :, :, ::-1].copy()

                    episode_id = env.replay_buffer.n_episodes
                    text_str = f"Episode: {episode_id}, Stage: {stage_val}"
                    if is_recording:
                        text_str += " [RECORDING]"
                    """
                    cv2.putText(
                        vis_img,
                        text_str,
                        (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1.0,
                        color=(255, 255, 255),
                        thickness=2,
                    )

                    cv2.imshow("Robot Teleop", vis_img)
                    cv2.waitKey(1)

                    """
                    precise_wait(t_sample)

                    sm_state = sm.get_motion_state_transformed()

                    dpos = sm_state[:3]
                    drot = sm_state[3:]
                    grasp = sm.grasp

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

                        # Grasp does not work.
                        action = np.concatenate([target_pose, [grasp]])

                        exec_timestamp = (
                            t_command_target - time.monotonic() + time.time()
                        )
                        env.exec_actions(
                            actions=[action],
                            timestamps=[exec_timestamp],
                            stages=[stage_val],
                        )
                        logger.debug("Significant movement detected, executing action.")
                    else:
                        action = np.concatenate([target_pose, [grasp]])
                        exec_timestamp = (
                            t_command_target - time.monotonic() + time.time()
                        )
                        env.exec_actions(
                            actions=[action],
                            timestamps=[exec_timestamp],
                            stages=[stage_val],
                        )
                        logger.debug("No significant movement detected.")

                    precise_wait(t_cycle_end)
                    iter_idx += 1

            except KeyboardInterrupt:
                logger.info("\nInterrupted by user.")
            except Exception:
                logger.error("Exception occurred during the main loop:")
                traceback.print_exc()
            finally:
                logger.info("Exiting main loop. Cleaning up...")

if __name__ == "__main__":
    main()
