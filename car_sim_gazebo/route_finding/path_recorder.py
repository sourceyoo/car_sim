#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.exceptions import ParameterAlreadyDeclaredException

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time

import csv
import os
import math


class PathRecorder(Node):
    def __init__(self):
        super().__init__('path_recorder')

        # ===== 파라미터 설정 =====
        # 전역 기준 프레임 (RViz Fixed Frame과 맞추기)
        self.declare_parameter('global_frame', 'map')
        # 로봇 기준 프레임 (TF 트리의 최하위, 여기서는 base_footprint가 직계)
        self.declare_parameter('base_frame', 'base_footprint')
        # CSV 저장 경로
        default_csv = '/home/yoo/Final_P/car_sim_path.csv'
        self.declare_parameter('csv_path', default_csv)
        # 기록 주파수(Hz)
        self.declare_parameter('record_hz', 33.3333)  # ≈ 0.03 s
        # 리샘플링 여부 및 간격 [m]
        self.declare_parameter('enable_resample', True)
        self.declare_parameter('resample_spacing_m', 0.2)
        # 원본(raw) Path도 퍼블리시할지 여부
        self.declare_parameter('publish_raw_path', False)
        # 시뮬레이션 시간 사용 여부
        try:
            self.declare_parameter('use_sim_time', False)
        except ParameterAlreadyDeclaredException:
            pass

        self.global_frame = self.get_parameter('global_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.csv_path = self.get_parameter('csv_path').value
        record_hz = self.get_parameter('record_hz').value
        self.enable_resample = self.get_parameter('enable_resample').value
        self.resample_spacing = self.get_parameter('resample_spacing_m').value
        self.publish_raw_path = self.get_parameter('publish_raw_path').value
        use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value

        # 강제로 map 프레임 사용
        if self.global_frame != 'map':
            self.get_logger().warn(
                f"[PathRecorder] global_frame='{self.global_frame}' overridden to 'map'"
            )
            self.global_frame = 'map'

        # 시뮬레이션 시간 설정을 노드 파라미터로 반영
        if use_sim_time:
            self.set_parameters([Parameter(name='use_sim_time', value=True)])

        self.get_logger().info(
            f"[PathRecorder] global_frame={self.global_frame}, "
            f"base_frame={self.base_frame}"
        )
        self.get_logger().info(f"[PathRecorder] CSV file: {self.csv_path}")

        # TF 버퍼/리스너
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Path 퍼블리셔
        self.path_pub = self.create_publisher(Path, 'recorded_path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.global_frame
        # raw path (옵션)
        if self.publish_raw_path:
            self.raw_path_pub = self.create_publisher(Path, 'recorded_path_raw', 10)
            self.raw_path_msg = Path()
            self.raw_path_msg.header.frame_id = self.global_frame
        else:
            self.raw_path_pub = None
            self.raw_path_msg = None

        # CSV 파일 오픈 + 헤더 작성
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time_sec', 'x', 'y', 'z', 'yaw_rad'])

        # 타이머 (record_hz 주기로 콜백)
        timer_period = 1.0 / record_hz
        self.timer = self.create_timer(timer_period, self.timer_cb)

    def timer_cb(self):
        # 현재 시간 기준 TF 조회
        now = Time()
        try:
            tf = self.tf_buffer.lookup_transform(
                self.global_frame,  # 부모 (map)
                self.base_frame,    # 자식 (base_footprint)
                now
            )
        except Exception as e:
            self.get_logger().warn(
                f"[PathRecorder] TF lookup failed: {e}",
                throttle_duration_sec=2.0
            )
            return

        # 위치
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        z = tf.transform.translation.z

        # 쿼터니언 -> yaw
        q = tf.transform.rotation
        yaw = self.quaternion_to_yaw(q.x, q.y, q.z, q.w)

        stamp = self.get_clock().now().to_msg()

        # raw path 처리 (옵션)
        if self.raw_path_pub:
            pose_raw = PoseStamped()
            pose_raw.header.stamp = stamp
            pose_raw.header.frame_id = self.global_frame
            pose_raw.pose.position.x = x
            pose_raw.pose.position.y = y
            pose_raw.pose.position.z = z
            pose_raw.pose.orientation = tf.transform.rotation
            self.raw_path_msg.header.stamp = stamp
            self.raw_path_msg.poses.append(pose_raw)
            self.raw_path_pub.publish(self.raw_path_msg)

        # 리샘플링 처리
        if not self.enable_resample:
            # 그대로 사용
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.global_frame
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation = tf.transform.rotation
            self.path_msg.header.stamp = stamp
            self.path_msg.poses.append(pose)
            self.path_pub.publish(self.path_msg)

            t = self.get_clock().now()
            sec, nsec = t.seconds_nanoseconds()
            t_sec = sec + nsec * 1e-9
            self.csv_writer.writerow([t_sec, x, y, z, yaw])
            self.csv_file.flush()
            return

        # 리샘플링 모드: 원하는 간격으로 보간된 포인트를 추가
        if not hasattr(self, 'last_resampled_pose'):
            self.last_resampled_pose = None

        def yaw_to_quat(yaw_val):
            q = tf.transform.rotation.__class__()  # geometry_msgs.msg.Quaternion
            q.x = 0.0
            q.y = 0.0
            q.z = math.sin(0.5 * yaw_val)
            q.w = math.cos(0.5 * yaw_val)
            return q

        # 현재 원시 포인트
        raw_point = (x, y, z, yaw)

        # 첫 포인트면 바로 추가
        if self.last_resampled_pose is None:
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.global_frame
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            pose.pose.orientation = yaw_to_quat(yaw)
            self.path_msg.header.stamp = stamp
            self.path_msg.poses.append(pose)
            self.path_pub.publish(self.path_msg)
            self.last_resampled_pose = (x, y, z)

            t = self.get_clock().now()
            sec, nsec = t.seconds_nanoseconds()
            t_sec = sec + nsec * 1e-9
            self.csv_writer.writerow([t_sec, x, y, z, yaw])
            self.csv_file.flush()
            return

        # 이전 리샘플 포인트와의 거리 계산
        lx, ly, lz = self.last_resampled_pose
        dx = x - lx
        dy = y - ly
        dist = math.hypot(dx, dy)

        # 필요 간격만큼 선형보간하며 포인트 추가
        while dist >= self.resample_spacing:
            ratio = self.resample_spacing / dist
            nx = lx + ratio * dx
            ny = ly + ratio * dy
            nz = z  # 고도는 직접 사용
            yaw_seg = math.atan2(dy, dx) if dist > 1e-6 else yaw

            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = self.global_frame
            pose.pose.position.x = nx
            pose.pose.position.y = ny
            pose.pose.position.z = nz
            pose.pose.orientation = yaw_to_quat(yaw_seg)

            self.path_msg.header.stamp = stamp
            self.path_msg.poses.append(pose)

            # CSV 기록
            t = self.get_clock().now()
            sec, nsec = t.seconds_nanoseconds()
            t_sec = sec + nsec * 1e-9
            self.csv_writer.writerow([t_sec, nx, ny, nz, yaw_seg])

            # 다음 루프를 위해 업데이트
            lx, ly, lz = nx, ny, nz
            dx = x - lx
            dy = y - ly
            dist = math.hypot(dx, dy)

        # 마지막(현재) 포인트도 항상 추가해 0.03s마다 기록 유지
        pose_curr = PoseStamped()
        pose_curr.header.stamp = stamp
        pose_curr.header.frame_id = self.global_frame
        pose_curr.pose.position.x = x
        pose_curr.pose.position.y = y
        pose_curr.pose.position.z = z
        pose_curr.pose.orientation = yaw_to_quat(yaw)

        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(pose_curr)

        t = self.get_clock().now()
        sec, nsec = t.seconds_nanoseconds()
        t_sec = sec + nsec * 1e-9
        self.csv_writer.writerow([t_sec, x, y, z, yaw])

        # 퍼블리시 및 플러시
        self.path_pub.publish(self.path_msg)
        self.csv_file.flush()
        self.last_resampled_pose = (x, y, z)

    def quaternion_to_yaw(self, x, y, z, w):
        # Z축 yaw 계산
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def destroy_node(self):
        # 노드 종료 시 CSV 파일 닫기
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PathRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
