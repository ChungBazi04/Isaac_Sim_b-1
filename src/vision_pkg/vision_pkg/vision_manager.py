''' object / target detector에서 받아온 좌표값을 보내줄 토픽노드 생성
 Vision System의 메인 ROS2 Node.
'''

import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

class VisionManager:
    def __init__(self):
        pass



def main(args=None):

    rclpy.init(args=args)

    node = VisionManager()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:

        cv2.destroyAllWindows()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()