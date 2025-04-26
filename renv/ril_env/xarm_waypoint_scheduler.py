import numpy as np
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Union
from queue import Queue, Empty
from enum import Enum

class WaypointState(Enum):
    """State of a waypoint in the scheduler"""
    QUEUED = 0      # Waypoint is queued but not yet sent to the robot
    SENT = 1        # Waypoint has been sent to the robot
    COMPLETED = 2   # Waypoint has been completed
    ABORTED = 3     # Waypoint was aborted

class Waypoint:
    """
    Represents a single waypoint in a trajectory
    """
    def __init__(
        self,
        pose: np.ndarray,
        timestamp: float,
        waypoint_id: int,
        gripper_state: Optional[float] = None
    ):
        self.pose = pose
        self.timestamp = timestamp
        self.waypoint_id = waypoint_id
        self.gripper_state = gripper_state
        self.state = WaypointState.QUEUED
        self.creation_time = time.time()

class XArmWaypointScheduler:
    """
    Manages and schedules waypoints for XArm robot
    
    This class maintains a queue of waypoints, handles timing,
    and ensures smooth execution of trajectories.
    """
    def __init__(
        self,
        robot_api,  # XArm API instance
        max_queued_waypoints: int = 10,
        time_horizon: float = 1.0,  # How far in the future to queue waypoints
        execution_frequency: int = 25,  # Hz
        min_waypoint_spacing: float = 0.05,  # Minimum time between waypoints (seconds)
        logger: Optional[logging.Logger] = None
    ):
        self.robot_api = robot_api
        self.max_queued_waypoints = max_queued_waypoints
        self.time_horizon = time_horizon
        self.execution_frequency = execution_frequency
        self.min_waypoint_spacing = min_waypoint_spacing
        self.logger = logger or logging.getLogger(__name__)
        
        # Waypoint storage
        self.waypoints = []  # List of all waypoints
        self.waypoint_queue = Queue()  # Queue for execution
        self.next_waypoint_id = 0
        
        # State tracking
        self.last_executed_pose = None
        self.last_execution_time = 0
        self.is_running = False
        self.execution_thread = None
        self.lock = threading.Lock()
        
    def start(self):
        """Start the waypoint scheduler"""
        if self.is_running:
            return
            
        self.is_running = True
        self.execution_thread = threading.Thread(target=self._execution_loop)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        self.logger.info("Waypoint scheduler started")
        
    def stop(self):
        """Stop the waypoint scheduler"""
        self.is_running = False
        if self.execution_thread:
            self.execution_thread.join(timeout=1.0)
            self.execution_thread = None
        self.logger.info("Waypoint scheduler stopped")
        
    def add_waypoint(
        self,
        pose: np.ndarray,
        timestamp: float,
        gripper_state: Optional[float] = None
    ) -> int:
        """
        Add a waypoint to the scheduler
        
        Args:
            pose: 6-DOF pose [x, y, z, rx, ry, rz]
            timestamp: Time at which the waypoint should be reached
            gripper_state: Gripper state (0.0 = open, 1.0 = closed)
            
        Returns:
            Waypoint ID
        """
        with self.lock:
            # Ensure pose is numpy array
            pose = np.array(pose)
            
            # Ensure timestamp is in the future
            current_time = time.time()
            if timestamp <= current_time:
                timestamp = current_time + 0.1  # Small offset
                
            # Create waypoint
            waypoint_id = self.next_waypoint_id
            self.next_waypoint_id += 1
            
            waypoint = Waypoint(
                pose=pose,
                timestamp=timestamp,
                waypoint_id=waypoint_id,
                gripper_state=gripper_state
            )
            
            # Add to list and queue
            self.waypoints.append(waypoint)
            self.waypoint_queue.put(waypoint)
            
            self.logger.debug(f"Added waypoint {waypoint_id} at timestamp {timestamp:.3f}")
            return waypoint_id
            
    def add_trajectory(
        self,
        trajectory: Dict,
        gripper_state: Optional[float] = None
    ) -> List[int]:
        """
        Add a full trajectory to the scheduler
        
        Args:
            trajectory: Trajectory dict with 'poses' and 'times' keys
            gripper_state: Gripper state (0.0 = open, 1.0 = closed)
            
        Returns:
            List of waypoint IDs
        """
        poses = trajectory['poses']
        times = trajectory['times']
        
        waypoint_ids = []
        for i in range(len(poses)):
            # Skip waypoints that are too close in time to the previous one
            if i > 0 and times[i] - times[i-1] < self.min_waypoint_spacing:
                continue
                
            waypoint_id = self.add_waypoint(
                pose=poses[i],
                timestamp=times[i],
                gripper_state=gripper_state
            )
            waypoint_ids.append(waypoint_id)
            
        return waypoint_ids
        
    def _execution_loop(self):
        """Main execution loop"""
        dt = 1.0 / self.execution_frequency
        
        while self.is_running:
            loop_start = time.time()
            
            try:
                self._process_waypoints()
            except Exception as e:
                self.logger.error(f"Error in waypoint execution: {e}")
                
            # Sleep to maintain execution frequency
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    def _process_waypoints(self):
        """Process and execute waypoints"""
        with self.lock:
            current_time = time.time()
            horizon_time = current_time + self.time_horizon
            
            # Only execute waypoints within our time horizon
            executable_waypoints = []
            for _ in range(self.max_queued_waypoints):
                try:
                    # Get waypoint from queue (non-blocking)
                    waypoint = self.waypoint_queue.get_nowait()
                    
                    # Check if it's within our time horizon
                    if waypoint.timestamp <= horizon_time:
                        executable_waypoints.append(waypoint)
                    else:
                        # Put it back in the queue for later
                        self.waypoint_queue.put(waypoint)
                        break
                except Empty:
                    break
                    
            if not executable_waypoints:
                return
                
            # Sort by timestamp
            executable_waypoints.sort(key=lambda wp: wp.timestamp)
            
            # Execute the first waypoint
            if time.time() - self.last_execution_time >= self.min_waypoint_spacing:
                waypoint = executable_waypoints[0]
                try:
                    # Execute the waypoint
                    self._execute_waypoint(waypoint)
                    
                    # Update tracking
                    self.last_execution_time = time.time()
                    self.last_executed_pose = waypoint.pose
                    waypoint.state = WaypointState.COMPLETED
                except Exception as e:
                    self.logger.error(f"Failed to execute waypoint {waypoint.waypoint_id}: {e}")
                    waypoint.state = WaypointState.ABORTED
                    
                # Remove from queue
                self.waypoint_queue.task_done()
                
    def _execute_waypoint(self, waypoint: Waypoint):
        """Execute a single waypoint"""
        # Send the waypoint to the robot
        try:
            code = self.robot_api.set_servo_cartesian(
                list(waypoint.pose),
                is_radian=False
            )
            
            if code != 0:
                self.logger.warning(f"Robot returned code {code} for waypoint {waypoint.waypoint_id}")
                
                # Handle error
                if code in [1, 10, 31]:  # Common error codes
                    self.logger.info("Attempting to reset robot error state")
                    self.robot_api.clean_error()
                    self.robot_api.clean_warn()
                    self.robot_api.set_mode(1)
                    self.robot_api.set_state(0)
                    
                    # Retry
                    code = self.robot_api.set_servo_cartesian(
                        list(waypoint.pose),
                        is_radian=False
                    )
                    
                    if code != 0:
                        raise RuntimeError(f"Failed to execute waypoint after reset, code: {code}")
            
            # Handle gripper if specified
            if waypoint.gripper_state is not None:
                if waypoint.gripper_state == 1.0:  # Close
                    self.robot_api.set_gripper_position(0, wait=False)
                else:  # Open
                    self.robot_api.set_gripper_position(850, wait=False)
                    
            waypoint.state = WaypointState.SENT
            self.logger.debug(f"Executed waypoint {waypoint.waypoint_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error executing waypoint: {e}")
            raise
            
    def clear_waypoints(self):
        """Clear all waypoints"""
        with self.lock:
            # Clear queue
            while not self.waypoint_queue.empty():
                try:
                    self.waypoint_queue.get_nowait()
                    self.waypoint_queue.task_done()
                except Empty:
                    break
                    
            # Clear waypoint list
            self.waypoints = []
            self.logger.info("Cleared all waypoints")
