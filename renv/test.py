import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your replay buffer
zarr_path = Path("recordings/replay_buffer.zarr")

# Open the Zarr archive
replay_buffer = zarr.open(str(zarr_path), mode='r')

# Get episode ends
episode_ends = replay_buffer['meta']['episode_ends'][:]
print(f"Total episodes: {len(episode_ends)}")

# Process each episode
for ep_idx in range(len(episode_ends)):
    # Calculate episode boundaries
    start_idx = 0 
    if ep_idx > 0:
        start_idx = episode_ends[ep_idx-1]
    end_idx = episode_ends[ep_idx]
    
    # Skip empty episodes
    episode_length = end_idx - start_idx
    if episode_length <= 0:
        print(f"Episode {ep_idx+1} is empty. Skipping.")
        continue
    
    print(f"Episode {ep_idx+1} length: {episode_length} steps")
    
    # Get data for this episode
    episode_slice = slice(start_idx, end_idx)
    
    try:
        # Get TCP Poses and timestamps
        tcp_poses = replay_buffer['data']['TCPPose'][episode_slice]
        timestamps = replay_buffer['data']['timestamp'][episode_slice]
        
        # Print shape and sample for verification
        print(f"  TCP Poses shape: {tcp_poses.shape}")
        print(f"  First TCP Pose: {tcp_poses[0]}")
        print(f"  Last TCP Pose: {tcp_poses[-1]}")
        
        # Plot TCP position over time
        plt.figure(figsize=(12, 6))
        
        # Extract x, y, z positions
        x_pos = tcp_poses[:, 0]
        y_pos = tcp_poses[:, 1]
        z_pos = tcp_poses[:, 2]
        
        # Time in seconds from the start
        time_seconds = timestamps - timestamps[0]
        
        plt.plot(time_seconds, x_pos, label='X Position')
        plt.plot(time_seconds, y_pos, label='Y Position')
        plt.plot(time_seconds, z_pos, label='Z Position')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Position (mm)')
        plt.title(f'Robot End-Effector Position - Episode {ep_idx+1}')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f'tcp_position_episode_{ep_idx+1}.png')
        print(f"  Saved image for episode {ep_idx+1}")
        plt.close()  # Close the figure to free memory
        
        # Optional: create a 3D plot of the trajectory
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(x_pos, y_pos, z_pos, label=f'Episode {ep_idx+1} Path')
        
        # Mark start and end points
        ax.scatter(x_pos[0], y_pos[0], z_pos[0], color='green', s=100, label='Start')
        ax.scatter(x_pos[-1], y_pos[-1], z_pos[-1], color='red', s=100, label='End')
        
        ax.set_xlabel('X Position (mm)')
        ax.set_ylabel('Y Position (mm)')
        ax.set_zlabel('Z Position (mm)')
        ax.set_title(f'Robot End-Effector 3D Trajectory - Episode {ep_idx+1}')
        ax.legend()
        
        plt.savefig(f'tcp_trajectory_3d_episode_{ep_idx+1}.png')
        print(f"  Saved 3D trajectory for episode {ep_idx+1}")
        plt.close()
        
    except IndexError as e:
        print(f"  Error processing episode {ep_idx+1}: {e}")
    except Exception as e:
        print(f"  Unexpected error processing episode {ep_idx+1}: {e}")

print("Processing complete!")
