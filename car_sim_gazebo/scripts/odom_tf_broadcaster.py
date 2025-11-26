#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from math import pi

class OdomTF(Node):
    def __init__(self):
        super().__init__('odom_tf_broadcaster')

        # static map->odom (필요 시 xyz/rpy 변경)
        self.static_br = StaticTransformBroadcaster(self)
        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()
        static_tf.header.frame_id = 'map'
        static_tf.child_frame_id = 'odom'
        static_tf.transform.translation.x = 0.0
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.0
        # rpy->quat(0,0,0)
        static_tf.transform.rotation.w = 1.0
        static_tf.transform.rotation.x = 0.0
        static_tf.transform.rotation.y = 0.0
        static_tf.transform.rotation.z = 0.0
        self.static_br.sendTransform(static_tf)

        # odom->base_link 브로드캐스터
        self.br = TransformBroadcaster(self)
        self.create_subscription(Odometry, '/odom', self.cb_odom, 10)
        self.get_logger().info('odom_tf_broadcaster started: static map->odom + odom->base_link')

    def cb_odom(self, msg: Odometry):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = msg.header.frame_id or 'odom'
        t.child_frame_id = msg.child_frame_id or 'base_link'
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)

def main():
    rclpy.init()
    rclpy.spin(OdomTF())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
