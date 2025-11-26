#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import TransformStamped, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class OdomTFBroadcaster(Node):
    """Subscribe to nav_msgs/Odometry and re-broadcast it as TF."""

    def __init__(self) -> None:
        super().__init__("odom_tf_broadcaster")
        # Parameters allow overriding topic and frame IDs.
        self.declare_parameter("odom_topic", "ground_truth/odom")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_footprint")

        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.child_frame_id = self.get_parameter("child_frame_id").get_parameter_value().string_value

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self._on_odom, 10)

    def _on_odom(self, msg: Odometry) -> None:
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        # Force configured frame IDs so the TF tree is consistent even if the
        # incoming message uses a different frame.
        t.header.frame_id = self.frame_id
        t.child_frame_id = self.child_frame_id
        # Copy into a Vector3 to satisfy TF message type requirements.
        pos = msg.pose.pose.position
        t.transform.translation = Vector3(x=pos.x, y=pos.y, z=pos.z)
        t.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomTFBroadcaster()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
