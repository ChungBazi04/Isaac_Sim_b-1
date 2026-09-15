#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32


class PCASub(Node):
    def __init__(self):
        super().__init__('pcA_sub')

        self.bridge = CvBridge()

        self.image_topic = self.declare_parameter(
            'image_topic', '/camera/color/image_raw').value
        self.blue_lower = np.array(
            self.declare_parameter('blue_lower', [90, 80, 50]).value,
            dtype=np.uint8,
        )
        self.blue_upper = np.array(
            self.declare_parameter('blue_upper', [130, 255, 255]).value,
            dtype=np.uint8,
        )
        self.green_lower = np.array(
            self.declare_parameter('green_lower', [35, 70, 50]).value,
            dtype=np.uint8,
        )
        self.green_upper = np.array(
            self.declare_parameter('green_upper', [85, 255, 255]).value,
            dtype=np.uint8,
        )
        self.min_area = int(self.declare_parameter('min_area', 1000).value)
        self.stable_frames = int(self.declare_parameter('stable_frames', 3).value)

        self.last_color = 0
        self.confirmed_color = 0
        self.stable_count = 0

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )
        self.detect_pub = self.create_publisher(Int32, '/m0609/detected_color', 10)
        self.debug_pub = self.create_publisher(Image, '/m0609/color_debug_image', 10)

        self.get_logger().info(
            f'Color detector started. image_topic={self.image_topic}, '
            f'min_area={self.min_area}, stable_frames={self.stable_frames}'
        )

    def _build_mask(self, hsv_image, lower, upper):
        mask = cv2.inRange(hsv_image, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _largest_contour_area(self, contours):
        if not contours:
            return 0
        return max(cv2.contourArea(c) for c in contours)

    def _largest_contour(self, contours):
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    def _publish_debug_image(self, image, blue_contour, green_contour, blue_area, green_area, result):
        debug = image.copy()

        if blue_contour is not None and blue_area > self.min_area:
            x, y, w, h = cv2.boundingRect(blue_contour)
            cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(debug, 'BLUE', (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

        if green_contour is not None and green_area > self.min_area:
            x, y, w, h = cv2.boundingRect(green_contour)
            cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(debug, 'GREEN', (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        if result == 0:
            cv2.putText(debug, 'NONE', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif result == 1:
            cv2.putText(debug, 'RESULT: BLUE', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        elif result == 2:
            cv2.putText(debug, 'RESULT: GREEN', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))
        except CvBridgeError as e:
            self.get_logger().error(f'Debug image publish failed: {e}')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        if cv_image is None or cv_image.size == 0:
            return

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        blue_mask = self._build_mask(hsv, self.blue_lower, self.blue_upper)
        green_mask = self._build_mask(hsv, self.green_lower, self.green_upper)

        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blue_area = self._largest_contour_area(blue_contours)
        green_area = self._largest_contour_area(green_contours)

        blue_contour = self._largest_contour(blue_contours) if blue_contours else None
        green_contour = self._largest_contour(green_contours) if green_contours else None

        color_code = 0
        if blue_area > self.min_area and blue_area >= green_area:
            color_code = 1
        elif green_area > self.min_area and green_area > blue_area:
            color_code = 2

        if color_code == 0:
            self.stable_count = 0
            self.last_color = 0
            if self.confirmed_color != 0:
                self.confirmed_color = 0
                self.detect_pub.publish(Int32(data=0))
            self._publish_debug_image(cv_image, blue_contour, green_contour, blue_area, green_area, 0)
            return

        if color_code == self.last_color:
            self.stable_count += 1
        else:
            self.last_color = color_code
            self.stable_count = 1

        if self.stable_count >= self.stable_frames:
            if self.confirmed_color != color_code:
                self.confirmed_color = color_code
                self.get_logger().info(f'Stable color confirmed: {color_code}')
            self.detect_pub.publish(Int32(data=color_code))

        self._publish_debug_image(cv_image, blue_contour, green_contour, blue_area, green_area, color_code)

    


def main(args=None):
    rclpy.init(args=args)
    node = PCASub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("강제종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
