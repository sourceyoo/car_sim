#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion

import csv
import os
import math
from rclpy.exceptions import ParameterAlreadyDeclaredException


class CsvPathPlayer(Node):
    def __init__(self):
        super().__init__('csv_path_player')

        # ===== 파라미터 =====
        default_csv = '/home/yoo/Final_P/car_sim_path.csv'
        self.declare_parameter('csv_path', default_csv)
        self.declare_parameter('frame_id', 'map')
        try:
            self.declare_parameter('use_sim_time', False)
        except ParameterAlreadyDeclaredException:
            pass

        self.csv_path = self.get_parameter('csv_path').value
        self.frame_id = self.get_parameter('frame_id').value
        use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value

        if use_sim_time:
            self.set_parameters([rclpy.parameter.Parameter(name='use_sim_time', value=True)])

        self.get_logger().info(f"[CsvPathPlayer] CSV: {self.csv_path}")
        self.get_logger().info(f"[CsvPathPlayer] frame_id: {self.frame_id}")

        # Path 퍼블리셔
        self.path_pub = self.create_publisher(Path, 'csv_path', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id

        # CSV에서 Path 로드
        self._load_csv_to_path()

        # RViz에서 계속 보이게 주기적으로 publish
        self.timer = self.create_timer(0.5, self._timer_cb)

    def _load_csv_to_path(self):
        if not os.path.exists(self.csv_path):
            self.get_logger().error(f"[CsvPathPlayer] CSV not found: {self.csv_path}")
            return

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row['x'])
                y = float(row['y'])
                z = float(row['z'])
                yaw = float(row['yaw_rad'])

                pose = PoseStamped()
                pose.header.frame_id = self.frame_id
                pose.header.stamp = self.get_clock().now().to_msg()

                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z

                q = yaw_to_quaternion(yaw)
                pose.pose.orientation = q

                self.path_msg.poses.append(pose)

        self.get_logger().info(
            f"[CsvPathPlayer] Loaded {len(self.path_msg.poses)} poses from CSV."
        )

    def _timer_cb(self):
        now = self.get_clock().now().to_msg()
        self.path_msg.header.stamp = now
        for p in self.path_msg.poses:
            p.header.stamp = now
        self.path_pub.publish(self.path_msg)


def yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    q.w = cy
    q.x = 0.0
    q.y = 0.0
    q.z = sy
    return q


def main(args=None):
    rclpy.init(args=args)
    node = CsvPathPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
