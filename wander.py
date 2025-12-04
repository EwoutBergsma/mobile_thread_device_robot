#!/usr/bin/env python3

import math
import random

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from irobot_create_msgs.msg import (
    HazardDetectionVector,
    HazardDetection,
    IrIntensityVector,
)


class WanderNode(Node):
    def __init__(self):
        super().__init__('create3_wander_full_control')

        # === Tunables ===
        self.forward_speed = 0.3       # m/s - cruising speed
        self.backward_speed = 0.20      # m/s
        self.turn_speed = 1.5           # rad/s
        self.backward_time = 1.0        # s

        # IR threshold: above this, we EMERGENCY STOP + back off + turn
        # Tune this based on /robot_1/ir_intensity values
        self.ir_threshold = 700.0

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/robot_1/cmd_vel', 10)

        # Subscriptions (sensor_data QoS is important on Create3!)
        self.hazard_sub = self.create_subscription(
            HazardDetectionVector,
            '/robot_1/hazard_detection',
            self.hazard_callback,
            qos_profile_sensor_data,
        )

        self.ir_sub = self.create_subscription(
            IrIntensityVector,
            '/robot_1/ir_intensity',
            self.ir_callback,
            qos_profile_sensor_data,
        )

        # Simple state machine: FORWARD, BACKING_UP, TURNING
        self.state = 'FORWARD'
        self.state_end_time = None      # rclpy.time.Time or None
        self.turn_direction = 1.0
        self.turn_duration = 0.0

        # Control loop at 20 Hz
        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('Wander node started (full control, no reflexes).')

    # ---------- Callbacks ----------

    def hazard_callback(self, msg: HazardDetectionVector):
        """
        Handle bumper / cliff / wheel-drop hazards ourselves.
        """
        # Ignore empty messages to avoid log spam
        if not msg.detections:
            return

        if self.state in ('BACKING_UP', 'TURNING'):
            return  # already reacting

        react_to = {
            HazardDetection.BUMP,
            HazardDetection.CLIFF,
            HazardDetection.WHEEL_DROP,
            HazardDetection.STALL,
        }

        for d in msg.detections:
            if d.type in react_to:
                self.get_logger().info(
                    f'Hazard {d.type} detected -> back off + random turn.'
                )
                self.start_back_off_and_turn()
                break

    def ir_callback(self, msg: IrIntensityVector):
        """
        Immediate IR-based braking:
        - If any IR value >= threshold:
            * publish zero Twist (brake)
            * start BACKING_UP + TURNING behaviour
        """
        # If we're already reacting to something, ignore IR
        if self.state in ('BACKING_UP', 'TURNING'):
            return

        max_value = max((r.value for r in msg.readings), default=0.0)

        if max_value >= self.ir_threshold:
            self.get_logger().info(
                f'IR obstacle (max={max_value}) -> EMERGENCY STOP + back off + turn.'
            )

            # Immediate brake: send zero velocity NOW
            stop = Twist()
            self.cmd_pub.publish(stop)

            # Then do the usual back-off + random turn sequence
            self.start_back_off_and_turn()

    # ---------- Behaviour helpers ----------

    def start_back_off_and_turn(self):
        """Common behaviour: back up a bit, then random turn."""
        now = self.get_clock().now()

        # 1) Back up for a fixed time
        self.state = 'BACKING_UP'
        self.state_end_time = now + Duration(seconds=self.backward_time)

        # 2) Choose random angle in [-pi, pi]
        angle = random.uniform(-math.pi, math.pi)
        self.turn_direction = 1.0 if angle >= 0.0 else -1.0
        self.turn_duration = abs(angle) / max(self.turn_speed, 1e-3)

        self.get_logger().info(
            f'Will turn {math.degrees(angle):.1f}° for {self.turn_duration:.2f}s.'
        )

    # ---------- Control loop ----------

    def control_loop(self):
        now = self.get_clock().now()
        twist = Twist()

        if self.state == 'FORWARD':
            twist.linear.x = self.forward_speed

        elif self.state == 'BACKING_UP':
            if self.state_end_time is not None and now >= self.state_end_time:
                # Done backing up -> start turning
                self.state = 'TURNING'
                self.state_end_time = now + Duration(seconds=self.turn_duration)
                self.get_logger().debug('Finished backing up, now turning.')
            else:
                twist.linear.x = -self.backward_speed

        elif self.state == 'TURNING':
            if self.state_end_time is not None and now >= self.state_end_time:
                # Done turning -> go forward again
                self.state = 'FORWARD'
                self.state_end_time = None
                self.get_logger().debug('Finished turning, now going forward.')
            else:
                twist.angular.z = self.turn_direction * self.turn_speed

        # Publish the command for the current state
        self.cmd_pub.publish(twist)


def main():
    rclpy.init()
    node = WanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot on exit
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
