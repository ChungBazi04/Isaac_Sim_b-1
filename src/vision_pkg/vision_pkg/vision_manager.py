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
        
        self.model_path = self.declare_parameter('model_path', '').value
        self.target_topic = self.declare_parameter(
            'target_topic', '/m0609/empty_shelf_position').value
        self.shelf_x_min = float(self.declare_parameter('shelf_x_min', -0.5).value)
        self.shelf_x_max = float(self.declare_parameter('shelf_x_max', 0.5).value)
        self.shelf_y_min = float(self.declare_parameter('shelf_y_min', -0.5).value)
        self.shelf_y_max = float(self.declare_parameter('shelf_y_max', 0.5).value)
        self.shelf_z = float(self.declare_parameter('shelf_z', 1.0).value)
        self.book_width = float(self.declare_parameter('book_width', 0.25).value)
        self.shelf_levels = [
            float(level) for level in self.declare_parameter(
                'shelf_levels', [0.0]).value
        ]


        if not self.model_path:
            raise ValueError(
                'model_path parameter is required, for example '
                '-p model_path:=/path/to/book_best.pt')

        self.model = YOLO(self.model_path)
        self.book_detector = BookDetector()
        self.target_detector = TargetDetector(
            shelf_x_min=self.shelf_x_min,
            shelf_x_max=self.shelf_x_max,
            shelf_y_min=self.shelf_y_min,
            shelf_y_max=self.shelf_y_max,
            shelf_z=self.shelf_z,
            book_width=self.book_width,
            shelf_levels=self.shelf_levels,
        )
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

        empty_position = self.target_detector.process(detected_targets)
        if empty_position is None:
            self.get_logger().info('No placeable empty shelf position found')
            return

        point = PointStamped()
        point.header = msg.header
        point.point.x, point.point.y, point.point.z = empty_position['xyz']
        self.target_pub.publish(point)

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