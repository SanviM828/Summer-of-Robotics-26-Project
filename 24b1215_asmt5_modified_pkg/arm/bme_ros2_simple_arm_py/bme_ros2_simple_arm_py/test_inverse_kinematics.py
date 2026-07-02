#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

def inverse_kinematics(coords, gripper_status, gripper_angle = 0):
    '''
    Calculates the joint angles according to the desired TCP coordinate and gripper angle
    :param coords: list, desired [X, Y, Z] TCP coordinates
    :param gripper_status: string, can be `closed` or `open`
    :param gripper_angle: float, gripper angle in woorld coordinate system (0 = horizontal, pi/2 = vertical)
    :return: list, the list of joint angles, including the 2 gripper fingers
    '''
    # link lengths
    ua_link = 0.2
    fa_link = 0.25
    tcp_link = 0.175
    # z offset (robot arm base height)
    z_offset = 0.05
    # default return list
    angles = [0,0,0,0,0,0]

    # Calculate the shoulder pan angle from x and y coordinates
    j0 = math.atan2(coords[1], coords[0])

    # Re-calculate target coordinated to the wrist joint (x', y', z')
    x = coords[0] - tcp_link * math.cos(j0) * math.cos(gripper_angle)
    y = coords[1] - tcp_link * math.sin(j0) * math.cos(gripper_angle)
    z = coords[2] - z_offset + math.sin(gripper_angle) * tcp_link

    # Solve the problem in 2D using x" and z'
    x = math.sqrt(y*y + x*x)

    # Let's calculate auxiliary lengths and angles
    c = math.sqrt(x*x + z*z)
    alpha = math.asin(z/c)
    beta = math.pi - alpha
    # Apply law of cosines
    gamma = math.acos((ua_link*ua_link + c*c - fa_link*fa_link)/(2*c*ua_link))

    j1 = math.pi/2.0 - alpha - gamma
    j2 = math.pi - math.acos((ua_link*ua_link + fa_link*fa_link - c*c)/(2*ua_link*fa_link)) # j2 = 180 - j2'
    delta = math.pi - (math.pi - j2) - gamma # delta = 180 - j2' - gamma

    j3 = math.pi + gripper_angle - beta - delta

    angles[0] = j0
    angles[1] = j1
    angles[2] = j2
    angles[3] = j3

    if gripper_status == "open":
        angles[4] = 0.04
        angles[5] = 0.04
    elif gripper_status == "closed":
        angles[4] = 0.01
        angles[5] = 0.01
    else:
        angles[4] = 0.04
        angles[5] = 0.04

    return angles

def forward_kinematics(joint_angles):
    '''
    Calculates the TCP coordinates from the joint angles
    :param joint_angles: list, joint angles [j0, j1, j2, j3, ...]
    :return: list, the list of TCP coordinates
    '''
    ua_link = 0.2
    fa_link = 0.25
    tcp_link = 0.175
    z_offset = 0.05

    x = math.cos(joint_angles[0]) * (math.sin(joint_angles[1]) * ua_link + math.sin(joint_angles[1] + joint_angles[2]) * fa_link + math.sin(joint_angles[1] + joint_angles[2] + joint_angles[3]) * tcp_link)
    y = math.sin(joint_angles[0]) * (math.sin(joint_angles[1]) * ua_link + math.sin(joint_angles[1] + joint_angles[2]) * fa_link + math.sin(joint_angles[1] + joint_angles[2] + joint_angles[3]) * tcp_link)
    z = z_offset + math.cos(joint_angles[1]) * ua_link + math.cos(joint_angles[1] + joint_angles[2]) * fa_link + math.cos(joint_angles[1] + joint_angles[2] + joint_angles[3]) * tcp_link

    return [x,y,z]



class IKPublisher(Node):

    def __init__(self):

        super().__init__("custom_inverse_kinematics")

        self.publisher = self.create_publisher(
            JointTrajectory,
            "/arm_controller/joint_trajectory",
            10,
        )

        self.timer = self.create_timer(
            2.0,
            self.publish_goal,
        )

        self.sent = False

    def publish_goal(self):

        if self.sent:
            return

        ##################################################
        # Change the target here
        ##################################################

        target = [0.40, -0.20, 0.15]

        ##################################################

        joint_angles = inverse_kinematics(
            target,
            "open",
            0.0,
        )

        fk = forward_kinematics(joint_angles)

        self.get_logger().info(f"Forward Kinematics = {fk}")

        msg = JointTrajectory()

        msg.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_joint",
            "left_finger_joint",
            "right_finger_joint",
        ]

        point = JointTrajectoryPoint()

        point.positions = joint_angles

        point.time_from_start = Duration(
            sec=3,
            nanosec=0,
        )

        msg.points.append(point)

        self.publisher.publish(msg)

        self.get_logger().info("Trajectory Published!")

        self.get_logger().info(f"Target = {target}")

        self.get_logger().info(
            f"Joint Angles = {joint_angles}"
        )

        self.sent = True





def main(args=None):

    rclpy.init(args=args)

    node = IKPublisher()

    rclpy.spin(node)

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()