'''
[EN]
ROS 2 node that feeds Isaac Sim camera data to BookDetector.
ROS 2 node that feeds Isaac Sim camera data to TargetDetector.
[KR] 
Isaac Sim 카메라 데이터를 BookDetector로 전달하는 ROS 2 노드.
Isaac Sim 카메라 데이터를 TargetDetector로 전달하는 ROS 2 노드.
'''

import rclpy
import cv2
import math
from ultralytics import YOLO

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError

from .book_detector import BookDetector
from .target_detector import TargetDetector



class VisionManager(Node):
    def __init__(self):
        super().__init__('vision_manager')

        self.bridge = CvBridge()
        self.rgb_topic = self.declare_parameter('rgb_topic', '/rgb').value
        self.depth_topic = self.declare_parameter('depth_topic', '/depth').value
        self.camera_info_topic = self.declare_parameter(
            'camera_info_topic', '/camera_info').value

        # 학습된 YOLO 모델(.pt) 파일의 절대 경로를 실행 시 model_path로 입력하세요.
        # 예: -p model_path:=/home/rokey/models/book_best.pt
        self.model_path = self.declare_parameter('model_path', '').value
        self.target_topic = self.declare_parameter(
            'target_topic', '/m0609/empty_shelf_position').value
        self.scan_radius = float(self.declare_parameter('scan_radius', 0.15).value)
        self.scan_pixel_u = int(self.declare_parameter('scan_pixel_u', -1).value)
        self.scan_pixel_v = int(self.declare_parameter('scan_pixel_v', -1).value)


        if not self.model_path:
            raise ValueError(
                'model_path parameter is required, for example '
                '-p model_path:=/path/to/book_best.pt')

        self.model = YOLO(self.model_path)
        self.book_detector = BookDetector()
        self.target_detector = TargetDetector(self.scan_radius)
        self.target_pub = self.create_publisher(PointStamped, self.target_topic, 10)
        self.latest_depth = None
        self.camera_info = None

        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.get_logger().info(
            f'Subscribed to rgb={self.rgb_topic}, depth={self.depth_topic}, '
            f'camera_info={self.camera_info_topic}')

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough')
        except CvBridgeError as error:
            self.get_logger().error(f'Depth conversion failed: {error}')

    def rgb_callback(self, msg):
        if self.latest_depth is None or self.camera_info is None:
            self.get_logger().debug('Waiting for depth image and CameraInfo')
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'RGB conversion failed: {error}')
            return

        info = self.camera_info
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]

        if fx <= 0 or fy <= 0:
            self.get_logger().warning('Invalid camera intrinsics in CameraInfo')
            return

        try:
            detections = self._detect_books(rgb_image)
            detected_targets = self.book_detector.process(
                detections,
                self.latest_depth,
                fx,
                fy,
                cx,
                cy,
            )
        except (IndexError, ValueError) as error:
            self.get_logger().error(f'Book detection failed: {error}')
            return

        for target in detected_targets:
            self.get_logger().info(
                f"Book detected: xyz={target['xyz']}, "
                f"center={target['center']}")

        scan_xyz = self._scan_position(
            self.latest_depth, fx, fy, cx, cy)
        empty_position = self.target_detector.process(detected_targets, scan_xyz)
        if empty_position is None:
            self.get_logger().info('No placeable empty shelf position found')
            return

        point = PointStamped()
        point.header = msg.header
        point.point.x, point.point.y, point.point.z = empty_position['xyz']
        self.target_pub.publish(point)

    def _scan_position(self, depth_image, fx, fy, cx, cy):
        height, width = depth_image.shape[:2]
        u = self.scan_pixel_u if self.scan_pixel_u >= 0 else width // 2
        v = self.scan_pixel_v if self.scan_pixel_v >= 0 else height // 2
        if not (0 <= u < width and 0 <= v < height):
            self.get_logger().warning('Scan pixel is outside the depth image')
            return None

        depth_value = depth_image[v, u]
        depth = float(depth_value)
        if not math.isfinite(depth) or depth <= 0:
            return None
        if depth_image.dtype.name == 'uint16':
            depth /= 1000.0

        return (
            (u - cx) * depth / fx,
            (v - cy) * depth / fy,
            depth,
        )

    def _detect_books(self, rgb_image):
        detections = []
        results = self.model(rgb_image, verbose=False)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                if class_name != 'book':
                    continue
                detections.append(tuple(map(int, box.xyxy[0])))
        return detections



def main(args=None):

    rclpy.init(args=args)

    node = VisionManager()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger.warn("강제 종료")

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()