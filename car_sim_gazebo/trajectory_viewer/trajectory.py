#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.parameter import Parameter
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs.tf2_geometry_msgs  # register geometry types with tf2

class TrajectoryNode(Node):
    def __init__(self):
        super().__init__("trajectory_node")
        # params
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("path_topic", "/actual_path")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("min_dist", 0.5)      # m, 누적 간격
        self.declare_parameter("pub_hz", 10.0)
        try:
            self.declare_parameter("use_sim_time", True)
        except Exception:
            pass
        if self.get_parameter("use_sim_time").get_parameter_value().bool_value:
            self.set_parameters([Parameter("use_sim_time", value=True)])

        odom_topic = self.get_parameter("odom_topic").value
        path_topic = self.get_parameter("path_topic").value
        self.global_frame = self.get_parameter("global_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.min_dist = float(self.get_parameter("min_dist").value)
        pub_hz = float(self.get_parameter("pub_hz").value)

        qos_path = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.path_pub = self.create_publisher(Path, path_topic, qos_path)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.global_frame
        self.last_pt = None

        self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.create_timer(1.0 / pub_hz, self.on_timer)
        self.get_logger().info(f"trajectory_node: odom={odom_topic}, path->{path_topic}, frame={self.global_frame}")

    def on_odom(self, msg: Odometry):
        # odom 포즈를 global_frame으로 변환
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        # header가 world로 오는 경우 odom으로 교정
        if pose.header.frame_id == "world":
            pose.header.frame_id = "odom"
        try:
            if pose.header.frame_id != self.global_frame:
                if self.tf_buffer.can_transform(
                    self.global_frame,
                    pose.header.frame_id,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.1),
                ):
                    pose.header.stamp = rclpy.time.Time().to_msg()
                    pose = self.tf_buffer.transform(
                        pose, self.global_frame, timeout=Duration(seconds=0.1)
                    )
                else:
                    # fallback: frame이 TF에 없으면 좌표는 그대로 두고 frame만 global_frame으로 사용
                    self.get_logger().warn(
                        f"TF transform not available {pose.header.frame_id}->{self.global_frame}, using fallback",
                        throttle_duration_sec=2.0,
                    )
                    pose.header.frame_id = self.global_frame
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}", throttle_duration_sec=2.0)
            return

        # 간격 체크 후 누적
        if self.last_pt:
            dx = pose.pose.position.x - self.last_pt.pose.position.x
            dy = pose.pose.position.y - self.last_pt.pose.position.y
            if math.hypot(dx, dy) < self.min_dist:
                return
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.global_frame
        self.path_msg.poses.append(pose)
        self.last_pt = pose

    def on_timer(self):
        if not self.path_msg.poses:
            return
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
