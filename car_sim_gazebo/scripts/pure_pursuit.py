#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.parameter import Parameter
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import Path, Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs.tf2_geometry_msgs as tf2_geometry

def quat_yaw(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__("pure_pursuit")
        # parameters
        self.declare_parameter("path_topic", "recorded_path")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("cmd_topic", "/itusct/command_cmd")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("wheel_base", 2.65)
        self.declare_parameter("lookahead_ratio", 2.0)
        self.declare_parameter("min_lookahead", 2.0)
        self.declare_parameter("max_lookahead", 10.0)
        self.declare_parameter("target_speed", 3.0)
        self.declare_parameter("control_rate", 20.0)
        try:
            self.declare_parameter("use_sim_time", True)
        except Exception:
            pass
        # apply sim time
        if self.get_parameter("use_sim_time").get_parameter_value().bool_value:
            self.set_parameters([Parameter("use_sim_time", value=True)])
        # topics
        path_topic = self.get_parameter("path_topic").value
        odom_topic = self.get_parameter("odom_topic").value
        cmd_topic = self.get_parameter("cmd_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.global_frame = self.get_parameter("global_frame").value
        self.wheel_base = float(self.get_parameter("wheel_base").value)
        self.lookahead_ratio = float(self.get_parameter("lookahead_ratio").value)
        self.min_ld = float(self.get_parameter("min_lookahead").value)
        self.max_ld = float(self.get_parameter("max_lookahead").value)
        self.target_speed = float(self.get_parameter("target_speed").value)
        rate = float(self.get_parameter("control_rate").value)

        qos_path = QoSProfile(
            depth=10, durability=DurabilityPolicy.VOLATILE, reliability=ReliabilityPolicy.RELIABLE
        )
        self.sub_path = self.create_subscription(Path, path_topic, self.on_path, qos_path)
        self.sub_odom = self.create_subscription(Odometry, odom_topic, self.on_odom, 10)
        self.pub_cmd = self.create_publisher(AckermannDriveStamped, cmd_topic, 10)
        qos_markers = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.pub_marker = self.create_publisher(Marker, "~/debug_marker", qos_markers)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.path_points = []
        self.last_odom = None

        self.timer = self.create_timer(1.0 / rate, self.on_timer)
        self.get_logger().info(f"pure_pursuit listening path={path_topic}, odom={odom_topic}, publishing cmd={cmd_topic}")

    def on_path(self, msg: Path):
        # 강제로 경로 프레임을 global_frame으로 맞춘다
        self.path_points = []
        for p in msg.poses:
            p.header.frame_id = self.global_frame
            self.path_points.append(p)
        self.get_logger().info(f"Received path with {len(self.path_points)} poses")

    def on_odom(self, msg: Odometry):
        self.last_odom = msg

    def on_timer(self):
        if not self.last_odom or len(self.path_points) < 2:
            return
        # current pose (odometry frame)
        pose = PoseStamped()
        pose.header = self.last_odom.header
        pose.pose = self.last_odom.pose.pose

        # transform pose to global_frame if needed
        if pose.header.frame_id != self.global_frame:
            try:
                pose.header.stamp = rclpy.time.Time().to_msg()
                pose = self.tf_buffer.transform(
                    pose, self.global_frame, timeout=Duration(seconds=0.1)
                )
            except Exception as e:
                # fallback: assume pose already in global_frame coordinates
                self.get_logger().warn(
                    f"TF transform for current pose failed (using fallback): {e}",
                    throttle_duration_sec=2.0,
                )
                pose.header.frame_id = self.global_frame

        # compute lookahead distance
        speed = self.last_odom.twist.twist.linear.x
        ld = max(self.min_ld, min(self.max_ld, self.lookahead_ratio * abs(speed)))
        target = self.find_target_point(pose, ld)
        if target is None:
            self.get_logger().warn("No target point found on path", throttle_duration_sec=2.0)
            return

        # transform target to base frame
        try:
            target.header.frame_id = self.global_frame
            target.header.stamp = rclpy.time.Time().to_msg()  # use latest transform data
            target_base = self.tf_buffer.transform(
                target, self.base_frame, timeout=Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}", throttle_duration_sec=2.0)
            return

        tx = target_base.pose.position.x
        ty = target_base.pose.position.y
        Ld_sq = tx * tx + ty * ty
        if Ld_sq < 1e-6:
            return
        curvature = 2.0 * ty / Ld_sq
        steering = math.atan(self.wheel_base * curvature)

        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = self.base_frame
        cmd.drive.steering_angle = float(steering)
        cmd.drive.speed = float(self.target_speed)
        self.pub_cmd.publish(cmd)

        self.publish_markers(target, pose, ld, curvature)

    def find_target_point(self, pose: PoseStamped, lookahead: float):
        # find closest point on path
        min_dist = 1e9
        min_idx = 0
        px = pose.pose.position.x
        py = pose.pose.position.y
        for i, p in enumerate(self.path_points):
            dx = p.pose.position.x - px
            dy = p.pose.position.y - py
            d = dx * dx + dy * dy
            if d < min_dist:
                min_dist = d
                min_idx = i
        # accumulate distance forward until reach lookahead
        acc = 0.0
        for j in range(min_idx, len(self.path_points) - 1):
            p0 = self.path_points[j].pose.position
            p1 = self.path_points[j + 1].pose.position
            ds = math.hypot(p1.x - p0.x, p1.y - p0.y)
            acc += ds
            if acc >= lookahead:
                return self.path_points[j + 1]
        return self.path_points[-1]

    def publish_markers(self, target: PoseStamped, pose: PoseStamped, ld: float, curvature: float):
        now = self.get_clock().now().to_msg()

        # ensure pose frame matches target frame for visualization (both in global_frame)
        target_frame = self.global_frame
        pose_vis = pose

        # target point sphere
        m_target = Marker()
        m_target.header.frame_id = target_frame
        m_target.header.stamp = now
        m_target.ns = "pure_pursuit"
        m_target.id = 0
        m_target.type = Marker.SPHERE
        m_target.action = Marker.ADD
        m_target.pose = target.pose
        m_target.scale.x = m_target.scale.y = m_target.scale.z = 0.4
        m_target.color.r = 0.0
        m_target.color.g = 1.0
        m_target.color.b = 0.0
        m_target.color.a = 1.0
        self.pub_marker.publish(m_target)

        # trajectory arc (LINE_STRIP)
        m_arc = Marker()
        m_arc.header.frame_id = target_frame
        m_arc.header.stamp = now
        m_arc.ns = "pure_pursuit"
        m_arc.id = 1
        m_arc.type = Marker.LINE_STRIP
        m_arc.action = Marker.ADD
        m_arc.scale.x = 0.05
        m_arc.color.r = 1.0
        m_arc.color.g = 1.0
        m_arc.color.b = 1.0
        m_arc.color.a = 1.0

        # generate arc in base frame then transform
        try:
            yaw = quat_yaw(pose_vis.pose.orientation)
            # radius from curvature
            if abs(curvature) < 1e-6:
                # straight line
                for s in [i * ld / 20.0 for i in range(21)]:
                    p_local = Point(x=s, y=0.0, z=pose_vis.pose.position.z)
                    p_global = self._to_global(p_local, pose_vis.pose.position, yaw)
                    m_arc.points.append(p_global)
            else:
                R = 1.0 / curvature
                arc_angle = ld * curvature
                steps = max(20, int(abs(arc_angle) / 0.01))
                for i in range(steps + 1):
                    th = arc_angle * i / steps
                    x_local = R * math.sin(th)
                    y_local = R * (1 - math.cos(th))
                    p_local = Point(x=x_local, y=y_local, z=pose_vis.pose.position.z)
                    p_global = self._to_global(p_local, pose_vis.pose.position, yaw)
                    m_arc.points.append(p_global)
        except Exception as e:
            self.get_logger().warn(f"arc gen failed: {e}", throttle_duration_sec=2.0)

        self.pub_marker.publish(m_arc)
        # One-time info to help RViz users
        if not hasattr(self, "_marker_info_logged"):
            self.get_logger().info("Publishing debug markers on topic 'pure_pursuit/debug_marker' (ns: pure_pursuit)", once=True)
            self._marker_info_logged = True

    @staticmethod
    def _to_global(p_local: Point, origin: Point, yaw: float) -> Point:
        cy = math.cos(yaw)
        sy = math.sin(yaw)
        x = cy * p_local.x - sy * p_local.y + origin.x
        y = sy * p_local.x + cy * p_local.y + origin.y
        return Point(x=x, y=y, z=origin.z)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
