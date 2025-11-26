#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

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
        # 로봇 기준 프레임 (차량 베이스 링크)
        self.declare_parameter('base_frame', 'base_link')
        # CSV 저장 경로
        default_csv = '/home/yoo/Final_P/car_sim_path.csv'
        self.declare_parameter('csv_path', default_csv)
        # 기록 주파수(Hz)
        self.declare_parameter('record_hz', 10.0)

        self.global_frame = self.get_parameter('global_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.csv_path = self.get_parameter('csv_path').value
        record_hz = self.get_parameter('record_hz').value

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

        # Path 메시지에 Pose 추가
        stamp = self.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.global_frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation = tf.transform.rotation  # 방향 그대로 사용

        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

        # CSV에 기록
        t = self.get_clock().now()
        sec, nsec = t.seconds_nanoseconds()
        t_sec = sec + nsec * 1e-9

        self.csv_writer.writerow([t_sec, x, y, z, yaw])
        # 필요시 flush (안 하면 종료 시점까지 버퍼에 있을 수 있음)
        self.csv_file.flush()

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

