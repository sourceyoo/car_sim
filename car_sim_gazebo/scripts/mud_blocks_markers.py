#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion
import math
import re

class MudBlocksPublisher(Node):
    def __init__(self):
        super().__init__('mud_blocks_publisher')

        # world 파일 경로 (네가 jarbay_mud_only.world 둔 위치로 수정)
        default_world = '/home/yoo/Final_P/car_sim_ws/src/car_sim/car_sim_gazebo/worlds/jarbay.world'
        self.declare_parameter('world_file', default_world)
        world_path = self.get_parameter('world_file').value

        self.get_logger().info(f'Loading mud blocks from: {world_path}')

        try:
            with open(world_path, 'r') as f:
                sdf = f.read()
        except Exception as e:
            self.get_logger().error(f'Failed to read world file: {e}')
            sdf = ''

        self.markers = self._parse_mud_blocks(sdf)

        # 퍼블리셔
        self.pub = self.create_publisher(MarkerArray, 'mud_blocks', 10)
        # 주기적으로 같은 MarkerArray를 재전송 (RViz에서 항상 보이게)
        self.timer = self.create_timer(0.5, self._timer_cb)

    def _timer_cb(self):
        if self.markers:
            self.pub.publish(self.markers)

    def _parse_mud_blocks(self, sdf_text: str) -> MarkerArray:
        """
        SDF에서 <model name='mud_box...'> 블럭을 찾아 pose/size를 MarkerArray로 변환
        """
        marker_array = MarkerArray()
        if not sdf_text:
            return marker_array

        # <model name='mud_box...'> ... </model> 전체 블럭 추출
        pattern = r'<model name=["\'](mud_box[^"\']*)["\']>(.*?)</model>'
        models = re.findall(pattern, sdf_text, re.DOTALL)


        self.get_logger().info(f'Found {len(models)} mud_box models in SDF.')

        marker_id = 0
        for name, body in models:
            # pose: x y z roll pitch yaw
            pose_match = re.search(r"<pose>(.*?)</pose>", body)
            if not pose_match:
                continue
            pose_vals = [float(v) for v in pose_match.group(1).split()]

            x, y, z, roll, pitch, yaw = pose_vals

            # box size: dx dy dz
            size_match = re.search(r"<box>.*?<size>(.*?)</size>.*?</box>", body, re.DOTALL)
            if not size_match:
                continue
            size_vals = [float(v) for v in size_match.group(1).split()]
            sx, sy, sz = size_vals

            # Marker 생성
            m = Marker()
            m.header.frame_id = 'map'   # Fixed Frame을 world로 맞추는 걸 추천
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'mud_blocks'
            m.id = marker_id
            marker_id += 1

            m.type = Marker.CUBE
            m.action = Marker.ADD

            # 위치
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = z

            # RPY -> 쿼터니언
            q = rpy_to_quaternion(roll, pitch, yaw)
            m.pose.orientation = q

            # 크기
            m.scale.x = sx
            m.scale.y = sy
            m.scale.z = sz

            # 색 (갈색 계열)
            m.color.r = 0.5
            m.color.g = 0.3
            m.color.b = 0.1
            m.color.a = 0.8  # 약간 투명

            # lifetime=0 → 계속 유지
            m.lifetime.sec = 0
            m.lifetime.nanosec = 0

            marker_array.markers.append(m)

        return marker_array


def rpy_to_quaternion(roll, pitch, yaw) -> Quaternion:
    """
    Roll, Pitch, Yaw [rad] -> geometry_msgs/Quaternion
    """
    q = Quaternion()
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy
    return q


def main(args=None):
    rclpy.init(args=args)
    node = MudBlocksPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

