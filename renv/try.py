import time
import numpy as np
import zarr
import logging
import argparse
from ril_env.xarm_controller import XArmConfig, XArmController
from multiprocessing.managers import SharedMemoryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def replay_demonstration(episode_idx=0, zarr_path="./recordings/replay_buffer.zarr", debug=False):
    """
    Replay a demonstration from the replay buffer.
    
    Args:
        episode_idx (int): Index of the episode to replay (0-based)
        zarr_path (str): Path to the zarr replay buffer
        debug (bool): If True, don't actually move the robot, just log commands
    """
    # Open the replay buffer
    logger.info(f"Opening replay buffer at {zarr_path}")
    replay_buffer = zarr.open(zarr_path, mode='r')
    
    # Get episode information
    episode_ends = replay_buffer['meta']['episode_ends'][:]
    total_episodes = len(episode_ends)
    
    if total_episodes == 0:
        logger.error("No episodes found in the replay buffer")
        return
    
    if episode_idx < 0 or episode_idx >= total_episodes:
        logger.error(f"Episode index {episode_idx} is out of range. Available episodes: 0-{total_episodes-1}")
        return
    
    # Calculate episode boundaries
    start_idx = 0
    if episode_idx > 0:
        start_idx = episode_ends[episode_idx-1]
    end_idx = episode_ends[episode_idx]
    
    # Check if episode is valid
    episode_length = end_idx - start_idx
    if episode_length <= 0:
        logger.error(f"Episode {episode_idx} is empty (length: {episode_length})")
        return
    
    episode_slice = slice(start_idx, end_idx)
    
    # Extract episode data
    tcp_poses = replay_buffer['data']['TCPPose'][episode_slice]
    timestamps = replay_buffer['data']['timestamp'][episode_slice]
    
    # Also get grasp data if available
    grasp_data = None
    if 'Grasp' in replay_buffer['data']:
        grasp_data = replay_buffer['data']['Grasp'][episode_slice]
    
    logger.info(f"Episode {episode_idx} loaded: {len(tcp_poses)} steps")
    logger.info(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
    logger.info(f"First TCP Pose: {tcp_poses[0]}")
    logger.info(f"Last TCP Pose: {tcp_poses[-1]}")
    
    # Initialize the robot controller
    with SharedMemoryManager() as shm_manager:
        xarm_config = XArmConfig()
        robot = XArmController(
            shm_manager=shm_manager,
            xarm_config=xarm_config,
        )
        
        try:
            # Start the robot controller
            logger.info("Starting robot controller")
            robot.start(wait=True)
            
            # Move to the initial position
            initial_pose = tcp_poses[0]
            initial_grasp = 0.0  # Default grasp value if not available
            if grasp_data is not None and len(grasp_data) > 0:
                initial_grasp = grasp_data[0]
            
            logger.info(f"Moving to initial position: {initial_pose}")
            if not debug:
                # Actually move the robot to initial position
                robot.step(initial_pose, initial_grasp)
            else:
                logger.info("DEBUG MODE: Robot movement commands not executed")
            
            # Wait 5 seconds
            logger.info("Waiting 5 seconds before starting replay")
            time.sleep(5.0)
            
            # Start the replay
            logger.info(f"Starting replay of episode {episode_idx}")
            
            # Calculate time offsets from the first timestamp
            time_offsets = timestamps - timestamps[0]
            start_time = time.time()
            
            for i in range(len(tcp_poses)):
                # Wait until it's time to execute this action
                target_time = start_time + time_offsets[i]
                now = time.time()
                if target_time > now:
                    time.sleep(target_time - now)
                
                # Get the pose and grasp
                pose = tcp_poses[i]
                grasp = grasp_data[i]
                
                # Print the pose we would execute

                logger.info(f"Step {i+1}/{len(tcp_poses)}: {pose}")
                
                # Execute the step
                if not debug:
                    robot.step(pose, grasp)
            
            logger.info(f"Replay of episode {episode_idx} completed")
            
        except KeyboardInterrupt:
            logger.info("Replay interrupted by user")
        finally:
            # Stop the robot controller
            logger.info("Stopping robot controller")
            robot.stop(wait=True)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Replay a demonstration from the XArm replay buffer")
    parser.add_argument("episode", type=int, nargs="?", default=0, 
                        help="Index of the episode to replay (default: 0)")
    parser.add_argument("--path", type=str, default="./recordings/replay_buffer.zarr",
                        help="Path to the zarr replay buffer (default: ./recordings/replay_buffer.zarr)")
    parser.add_argument("--debug", action="store_true", 
                        help="Debug mode: Don't actually move the robot, just log commands")
    
    args = parser.parse_args()
    
    # Call the replay function with the specified episode
    replay_demonstration(
        episode_idx=args.episode,
        zarr_path=args.path,
        debug=args.debug
    )
