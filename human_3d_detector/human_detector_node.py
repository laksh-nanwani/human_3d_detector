#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose
from geometry_msgs.msg import Pose

from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO

import message_filters


class Human3DDetector(Node):

    def __init__(self):
        super().__init__('human_3d_detector')

        # Load YOLO model
        self.model = YOLO('yolov8n.pt')  # Change to your model path
        self.bridge = CvBridge()

        # Camera intrinsics (replace with calibrated values)
        self.fx = 600.0
        self.fy = 600.0
        self.cx_cam = 320.0
        self.cy_cam = 240.0

        # Subscribers with synchronization
        rgb_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')

        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.synced_callback)

        # Publisher for Detection3DArray
        self.det_pub = self.create_publisher(Detection3DArray, '/human_detections_3d', 10)

        self.get_logger().info("Human 3D Detector Node with Median Depth Initialized")

    def synced_callback(self, rgb_msg, depth_msg):
        rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

        results = self.model(rgb_image, device='cuda' if self.model.device.type == 'cuda' else 'cpu')

        detections_msg = Detection3DArray()
        detections_msg.header = rgb_msg.header

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, class_id, score in zip(boxes, class_ids, scores):
                if int(class_id) != 0 or score < 0.5:
                    continue  # Only 'person' detections with confidence > 0.5

                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if not (0 <= cx < depth_image.shape[1] and 0 <= cy < depth_image.shape[0]):
                    continue

                # Median Depth over small region near center
                region_size = 10  # You can increase for more robust estimation
                x_start = max(cx - region_size // 2, x1)
                x_end = min(cx + region_size // 2, x2)
                y_start = max(cy - region_size // 2, y1)
                y_end = min(cy + region_size // 2, y2)

                depth_region = depth_image[y_start:y_end, x_start:x_end]
                valid_depths = depth_region[
                    np.isfinite(depth_region) &
                    (depth_region > 0.0) &
                    (depth_region < 10.0)
                ]

                if valid_depths.size == 0:
                    continue

                depth = np.median(valid_depths)

                # 3D position calculation
                X = (cx - self.cx_cam) * depth / self.fx
                Y = (cy - self.cy_cam) * depth / self.fy
                Z = depth

                detection = Detection3D()
                detection.results.append(ObjectHypothesisWithPose(
                    id=0,
                    score=float(score),
                    pose=Pose(position=self.create_point(X, Y, Z))
                ))

                detections_msg.detections.append(detection)

                # Visualization
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(rgb_image, (cx, cy), 5, (0, 0, 255), -1)

        if detections_msg.detections:
            self.det_pub.publish(detections_msg)
            self.get_logger().info(f"Published {len(detections_msg.detections)} human detections")

        cv2.imshow("Human Detections", rgb_image)
        cv2.waitKey(1)

    @staticmethod
    def create_point(x, y, z):
        p = Pose().position
        p.x = x
        p.y = y
        p.z = z
        return p


def main(args=None):
    rclpy.init(args=args)
    node = Human3DDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
