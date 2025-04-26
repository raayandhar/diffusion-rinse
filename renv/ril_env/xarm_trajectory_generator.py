import numpy as np
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import scipy.spatial.transform as st

class MotionType(Enum):
    """Type of motion to generate"""
    LINEAR = 0       # Linear interpolation with trapezoidal velocity profile
    JOINT_SPACE = 1  # Joint space interpolation with blending

class XArmTrajectoryGenerator:
    """
    Generates proper motion trajectories for XArm robot
    
    This class creates smooth trajectories with proper acceleration/deceleration
    profiles for the XArm robot, ensuring smooth motion between waypoints.
    """
    def __init__(
        self,
        max_vel: float = 0.05,           # m/s
        max_acc: float = 0.1,            # m/s^2
        max_angular_vel: float = 0.2,    # deg/s
        max_angular_acc: float = 0.4,    # deg/s^2
        min_duration: float = 0.1,       # seconds
        logger: Optional[logging.Logger] = None
    ):
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_angular_vel = max_angular_vel
        self.max_angular_acc = max_angular_acc
        self.min_duration = min_duration
        self.logger = logger or logging.getLogger(__name__)
        self.last_waypoint = None
        self.prev_trajectory_end_time = 0
        
    def compute_time_optimal_duration(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray
    ) -> float:
        """
        Compute the time-optimal duration for a trajectory between two poses.
        Takes into account max velocity and acceleration.
        
        Args:
            start_pose: Starting pose [x, y, z, rx, ry, rz] (position in m, rotation in degrees)
            end_pose: Ending pose [x, y, z, rx, ry, rz] (position in m, rotation in degrees)
            
        Returns:
            The minimum duration (in seconds) required to move between the poses
            given the velocity and acceleration constraints.
        """
        # Calculate positional distance
        pos_distance = np.linalg.norm(end_pose[:3] - start_pose[:3])
        
        # Calculate angular distance
        r1 = st.Rotation.from_euler('xyz', start_pose[3:], degrees=True)
        r2 = st.Rotation.from_euler('xyz', end_pose[3:], degrees=True)
        ang_distance = (r2 * r1.inv()).magnitude() * 180 / np.pi  # Convert to degrees
        
        # Time to move positional distance with trapezoidal velocity profile
        # For a trapezoidal profile, min time = sqrt(4*d/a) if max velocity is not reached
        # If max velocity is reached, time = d/v + v/a
        pos_time_acc_limited = np.sqrt(4 * pos_distance / self.max_acc)
        pos_time_vel_limited = pos_distance / self.max_vel + self.max_vel / self.max_acc
        pos_time = min(pos_time_acc_limited, pos_time_vel_limited)
        
        # Time to move angular distance with trapezoidal velocity profile
        ang_time_acc_limited = np.sqrt(4 * ang_distance / self.max_angular_acc)
        ang_time_vel_limited = ang_distance / self.max_angular_vel + self.max_angular_vel / self.max_angular_acc
        ang_time = min(ang_time_acc_limited, ang_time_vel_limited)
        
        # Take the longer of the two times
        duration = max(pos_time, ang_time)
        
        # Ensure minimum duration
        return max(duration, self.min_duration)
    
    def generate_trajectory(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        start_time: float,
        motion_type: MotionType = MotionType.LINEAR,
        steps: int = 50
    ) -> Dict:
        """
        Generate a trajectory between two poses.
        
        Args:
            start_pose: Starting pose [x, y, z, rx, ry, rz] (position in m, rotation in degrees)
            end_pose: Ending pose [x, y, z, rx, ry, rz] (position in m, rotation in degrees)
            start_time: Time to start the trajectory (in seconds)
            motion_type: Type of motion to generate
            steps: Number of steps in the trajectory
            
        Returns:
            Dictionary containing trajectory information:
                - 'poses': List of poses in the trajectory
                - 'times': List of times for each pose
                - 'duration': Total duration of the trajectory
        """
        # Compute required duration
        duration = self.compute_time_optimal_duration(start_pose, end_pose)
        
        # For safety, make sure start_time is after any previous trajectory
        if start_time < self.prev_trajectory_end_time:
            start_time = self.prev_trajectory_end_time
            
        end_time = start_time + duration
        self.prev_trajectory_end_time = end_time
        
        # Generate timestamps linearly spaced
        timestamps = np.linspace(start_time, end_time, steps)
        
        # Create trajectory based on motion type
        if motion_type == MotionType.LINEAR:
            poses = self._generate_linear_trajectory(start_pose, end_pose, steps)
        elif motion_type == MotionType.JOINT_SPACE:
            poses = self._generate_joint_space_trajectory(start_pose, end_pose, steps)
        else:
            raise ValueError(f"Unsupported motion type: {motion_type}")
            
        # Update last waypoint
        self.last_waypoint = end_pose
            
        return {
            'poses': poses,
            'times': timestamps,
            'duration': duration
        }
    
    def _generate_linear_trajectory(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """
        Generate a linear trajectory between two poses with a trapezoidal velocity profile.
        
        Args:
            start_pose: Starting pose [x, y, z, rx, ry, rz]
            end_pose: Ending pose [x, y, z, rx, ry, rz]
            steps: Number of steps in the trajectory
            
        Returns:
            Array of poses in the trajectory
        """
        # Create a normalized time vector with S-curve profile for smooth acceleration
        # This creates a smoother velocity profile than linear interpolation
        t_norm = np.linspace(0, 1, steps)
        
        # Apply smoothstep function for S-curve: t' = 3t^2 - 2t^3
        # This gives smoother acceleration/deceleration
        s_curve = 3 * t_norm**2 - 2 * t_norm**3
        
        # Linear interpolation for position
        positions = np.zeros((steps, 3))
        for i in range(3):  # x, y, z
            positions[:, i] = start_pose[i] + s_curve * (end_pose[i] - start_pose[i])
            
        # Spherical linear interpolation (SLERP) for orientation
        orientations = np.zeros((steps, 3))
        r1 = st.Rotation.from_euler('xyz', start_pose[3:], degrees=True)
        r2 = st.Rotation.from_euler('xyz', end_pose[3:], degrees=True)
        
        # Create interpolated rotations
        for i in range(steps):
            r_interp = st.Slerp([0, 1], st.Rotation.from_quat([r1.as_quat(), r2.as_quat()]))(s_curve[i])
            orientations[i] = r_interp.as_euler('xyz', degrees=True)
            
        # Combine into full poses
        poses = np.column_stack((positions, orientations))
        return poses
    
    def _generate_joint_space_trajectory(
        self,
        start_pose: np.ndarray,
        end_pose: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """
        Generate a joint space trajectory between two poses.
        This is typically smoother for the robot but doesn't follow a straight line.
        
        Args:
            start_pose: Starting pose [x, y, z, rx, ry, rz]
            end_pose: Ending pose [x, y, z, rx, ry, rz]
            steps: Number of steps in the trajectory
            
        Returns:
            Array of poses in the trajectory
        """
        # For joint space motion, we would ideally convert to joint angles first,
        # interpolate in joint space, then convert back to Cartesian.
        # Since we don't have inverse kinematics available, we'll use a similar
        # approach to linear but with a different profile.
        
        # Create a normalized time vector with smoother profile
        t_norm = np.linspace(0, 1, steps)
        
        # Apply a higher-order smoothstep: t' = 6t^5 - 15t^4 + 10t^3
        # This gives even smoother transitions at the endpoints
        smooth_profile = 6*t_norm**5 - 15*t_norm**4 + 10*t_norm**3
        
        # Linear interpolation for position
        positions = np.zeros((steps, 3))
        for i in range(3):  # x, y, z
            positions[:, i] = start_pose[i] + smooth_profile * (end_pose[i] - start_pose[i])
            
        # Spherical linear interpolation (SLERP) for orientation
        orientations = np.zeros((steps, 3))
        r1 = st.Rotation.from_euler('xyz', start_pose[3:], degrees=True)
        r2 = st.Rotation.from_euler('xyz', end_pose[3:], degrees=True)
        
        # Create interpolated rotations
        for i in range(steps):
            r_interp = st.Slerp([0, 1], st.Rotation.from_quat([r1.as_quat(), r2.as_quat()]))(smooth_profile[i])
            orientations[i] = r_interp.as_euler('xyz', degrees=True)
            
        # Combine into full poses
        poses = np.column_stack((positions, orientations))
        return poses
    
    def plan_trajectories(
        self,
        waypoints: List[np.ndarray],
        timestamps: List[float],
        motion_type: MotionType = MotionType.LINEAR
    ) -> List[Dict]:
        """
        Plan trajectories through a sequence of waypoints.
        
        Args:
            waypoints: List of waypoint poses
            timestamps: List of timestamps for each waypoint
            motion_type: Type of motion to generate
            
        Returns:
            List of trajectory dictionaries
        """
        if len(waypoints) < 2:
            self.logger.warning("Cannot plan trajectories with fewer than 2 waypoints")
            return []
            
        if len(waypoints) != len(timestamps):
            self.logger.error(f"Number of waypoints ({len(waypoints)}) must match number of timestamps ({len(timestamps)})")
            return []
            
        # Ensure waypoints are in chronological order
        sorted_indices = np.argsort(timestamps)
        waypoints = [waypoints[i] for i in sorted_indices]
        timestamps = [timestamps[i] for i in sorted_indices]
        
        # Get current pose as starting point if available
        if self.last_waypoint is not None:
            start_pose = self.last_waypoint
            
            # If first timestamp is in the past, adjust it
            current_time = time.time()
            if timestamps[0] <= current_time:
                timestamps[0] = current_time + 0.1  # Small offset for safety
        else:
            # Use first waypoint as starting point
            start_pose = waypoints[0]
            
        trajectories = []
        for i in range(len(waypoints) - 1):
            start_pose = waypoints[i]
            end_pose = waypoints[i + 1]
            start_time = timestamps[i]
            
            # Generate trajectory
            trajectory = self.generate_trajectory(
                start_pose=start_pose,
                end_pose=end_pose,
                start_time=start_time,
                motion_type=motion_type
            )
            
            trajectories.append(trajectory)
            
        return trajectories
