#!/usr/bin/env python3
"""
wall_follow_and_log.py

ROS 2 (Humble) script for iRobot Create 3 that:
- Sends a WallFollow action goal to /robot_1/wall_follow
- Default: follow left side for 1 hour
- Periodically prints robot pose, linear speed, and bumper state
- Logs each bumper "hit" event immediately (edge triggered)
- Uses ISO-like timestamps: [YYYY-MM-DDTHH:MM:SS.mmm]
- Requests cancellation of wall following when the script is stopped (Ctrl+C)

Run:
    python3 wall_follow_and_log.py
"""

import math
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data

from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry

from irobot_create_msgs.action import WallFollow
from irobot_create_msgs.msg import HazardDetectionVector, HazardDetection


def yaw_from_quaternion(q):
    """
    Compute yaw (rotation around Z) from a geometry_msgs/Quaternion.
    """
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class WallFollowClientNode(Node):
    def __init__(self):
        super().__init__("wall_follow_client")

        # Action client for /robot_1/wall_follow
        self._action_client = ActionClient(
            self,
            WallFollow,
            "/robot_1/wall_follow",
        )

        # Subscriptions for status/info
        # Use sensor-data QoS (BEST_EFFORT) to match Create 3 topics.
        self._odom_sub = self.create_subscription(
            Odometry,
            "/robot_1/odom",
            self.odom_callback,
            qos_profile_sensor_data,
        )

        self._hazard_sub = self.create_subscription(
            HazardDetectionVector,
            "/robot_1/hazard_detection",
            self.hazard_callback,
            qos_profile_sensor_data,
        )

        # Timer to print status once per second
        self._status_timer = self.create_timer(1.0, self.status_timer_callback)

        # State variables
        self._goal_handle = None
        self._latest_odom = None

        # Bumper-related state
        self._bumper_active = False          # bumper currently pressed
        self._bumper_was_active = False      # previous state, for edge detection
        self._bumper_hit_since_last_log = False
        self._bumper_hit_count = 0           # total hits since start

        self.get_logger().info("WallFollowClientNode initialized.")

    # -------------------------------------------------------------------------
    # Action client logic
    # -------------------------------------------------------------------------

    def send_wall_follow_goal(self, follow_side: int = 1, duration_sec: int = 3600):
        """
        Send the WallFollow goal to the robot.

        :param follow_side: +1 = left, -1 = right
        :param duration_sec: maximum runtime in seconds
        """
        self.get_logger().info(
            "Waiting for /robot_1/wall_follow action server..."
        )
        self._action_client.wait_for_server()
        self.get_logger().info("Action server available, sending goal...")

        goal_msg = WallFollow.Goal()
        # 1 = left, -1 = right (see irobot_create_msgs/action/WallFollow.action)
        goal_msg.follow_side = follow_side

        goal_msg.max_runtime = Duration()
        goal_msg.max_runtime.sec = int(duration_sec)
        goal_msg.max_runtime.nanosec = 0

        self.get_logger().info(
            f"Sending WallFollow goal: follow_side={goal_msg.follow_side}, "
            f"max_runtime={goal_msg.max_runtime.sec} s"
        )

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().error("WallFollow goal was rejected by the server.")
            self._goal_handle = None
            return

        self.get_logger().info("WallFollow goal accepted.")
        self._goal_handle = goal_handle

        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        # feedback.engaged is a bool indicating whether wall following is engaged
        self.get_logger().debug(
            f"WallFollow feedback: engaged={feedback.engaged}"
        )

    def get_result_callback(self, future):
        result_msg = future.result()
        if result_msg is None:
            # This can happen if the node/context is shutting down
            self.get_logger().warn("WallFollow result not available (shutdown in progress?)")
            self._goal_handle = None
            return

        result = result_msg.result
        runtime = result.runtime
        self.get_logger().info(
            f"WallFollow action finished, runtime = {runtime.sec} s"
        )
        # Once finished, clear handle so we do not try to cancel a completed goal
        self._goal_handle = None

    def request_cancel_wall_follow(self):
        """
        Request cancellation of the WallFollow goal if one is active.

        This is 'fire-and-forget' to avoid spinning the executor during shutdown.
        """
        if self._goal_handle is None:
            self.get_logger().info(
                "No active WallFollow goal to cancel."
            )
            return

        self.get_logger().info("Requesting cancellation of WallFollow goal...")
        try:
            _ = self._goal_handle.cancel_goal_async()
        except Exception as e:
            # If context is already shutting down, this might throw.
            self.get_logger().warn(f"Failed to send cancel request: {e}")
        finally:
            # We will not use this handle again.
            self._goal_handle = None

    # -------------------------------------------------------------------------
    # Subscriptions and status printing
    # -------------------------------------------------------------------------

    def odom_callback(self, msg: Odometry):
        """
        Store latest odometry.
        """
        self._latest_odom = msg

    def hazard_callback(self, msg: HazardDetectionVector):
        """
        Track bumper hits based on HazardDetectionVector.

        We treat a "hit" as a transition from no BUMP to at least one BUMP
        in the hazard vector (edge-triggered), so short contacts are not lost.
        """
        has_bump = any(
            detection.type == HazardDetection.BUMP
            for detection in msg.detections
        )

        self._bumper_active = has_bump

        # Edge detection: new hit when we go from no bump -> bump
        if has_bump and not self._bumper_was_active:
            self._bumper_hit_since_last_log = True
            self._bumper_hit_count += 1

            now_str = datetime.now().isoformat(timespec="milliseconds")

            # If we have odom, include pose & speed at the time of the hit
            if self._latest_odom is not None:
                pose = self._latest_odom.pose.pose
                twist = self._latest_odom.twist.twist

                x = pose.position.x
                y = pose.position.y
                yaw = yaw_from_quaternion(pose.orientation)
                linear_speed = twist.linear.x

                self.get_logger().info(
                    f"[{now_str}] BUMPER HIT #{self._bumper_hit_count}: "
                    f"x={x:.3f} m, y={y:.3f} m, yaw={yaw:.2f} rad, "
                    f"linear_speed={linear_speed:.3f} m/s"
                )
            else:
                self.get_logger().info(
                    f"[{now_str}] BUMPER HIT #{self._bumper_hit_count}: "
                    f"odom not yet available"
                )

        # Remember state for next callback
        self._bumper_was_active = has_bump

    def status_timer_callback(self):
        """
        Print pose, linear speed, and bumper state at 1 Hz,
        with ISO-like timestamp [YYYY-MM-DDTHH:MM:SS.mmm].
        """
        now_str = datetime.now().isoformat(timespec="milliseconds")

        if self._latest_odom is None:
            self.get_logger().info(f"[{now_str}] Waiting for /robot_1/odom...")
            return

        pose = self._latest_odom.pose.pose
        twist = self._latest_odom.twist.twist

        x = pose.position.x
        y = pose.position.y
        yaw = yaw_from_quaternion(pose.orientation)
        linear_speed = twist.linear.x  # m/s (forward component)

        # Latch value and reset so we report if anything happened since last print
        bumper_hit_recent = self._bumper_hit_since_last_log
        self._bumper_hit_since_last_log = False

        self.get_logger().info(
            f"[{now_str}] Pose: x={x:.3f} m, y={y:.3f} m, yaw={yaw:.2f} rad | "
            f"linear_speed={linear_speed:.3f} m/s | "
            f"bumper_active={self._bumper_active} | "
            f"bumper_hit_since_last_log={bumper_hit_recent} | "
            f"bumper_hit_count={self._bumper_hit_count}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowClientNode()

    # Default: follow left side (1) for one hour (3600 seconds)
    node.send_wall_follow_goal(follow_side=1, duration_sec=3600)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # On Ctrl+C, request cancellation and exit
        try:
            print("Ctrl+C received, stopping wall following...")
        except Exception:
            pass

        node.request_cancel_wall_follow()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
