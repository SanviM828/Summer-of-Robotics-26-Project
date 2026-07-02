#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import FollowWaypoints
from geometry_msgs.msg import PoseStamped
from action_msgs.msg import GoalStatus


class PatrolWaypoints(Node):

    def __init__(self):
        super().__init__('patrol_waypoints')

        self.client = ActionClient(
            self,
            FollowWaypoints,
            '/follow_waypoints'
        )

        self.get_logger().info("Waiting for Nav2 FollowWaypoints server...")
        self.client.wait_for_server()

        goal_msg = FollowWaypoints.Goal()

        # -----------------------
        # Waypoint 1
        # -----------------------
        pose1 = PoseStamped()
        pose1.header.frame_id = "map"
        pose1.pose.position.x = 3.97
        pose1.pose.position.y = -4.00
        pose1.pose.orientation.w = 1.0

        # -----------------------
        # Waypoint 2
        # -----------------------
        pose2 = PoseStamped()
        pose2.header.frame_id = "map"
        pose2.pose.position.x = -1.52
        pose2.pose.position.y = -4.22
        pose2.pose.orientation.w = 1.0

        # -----------------------
        # Waypoint 3
        # -----------------------
        pose3 = PoseStamped()
        pose3.header.frame_id = "map"
        pose3.pose.position.x = 6.95
        pose3.pose.position.y = 0.15
        pose3.pose.orientation.w = 1.0

        goal_msg.poses = [pose1, pose2, pose3]

        self.get_logger().info("Sending 3 waypoints...")

        self.send_goal_future = self.client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            return

        self.get_logger().info("Goal accepted")

        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):

        current = feedback_msg.feedback.current_waypoint
        self.get_logger().info(
            f"Currently travelling to waypoint {current + 1}"
        )

    def result_callback(self, future):

        status = future.result().status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("All waypoints reached!")
        else:
            self.get_logger().info(f"Navigation failed. Status = {status}")

        rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

    node = PatrolWaypoints()

    rclpy.spin(node)


if __name__ == "__main__":
    main()