import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import threading
import time

from ultralytics import YOLO
from geometry_msgs.msg import Twist


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__('yolo_detector')

        # Better model than yolov8n
        self.model = YOLO("yolov8n.pt")

        self.get_logger().info("YOLO model loaded")

        self.subscription = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            1
        )

        self.depth_subscription = self.create_subscription(
            Image,
            "camera/depth_image",
            self.depth_callback,
            1
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10
        )

        self.bridge = CvBridge()

        self.latest_frame = None
        self.latest_depth = None
        self.frame_lock = threading.Lock()

        self.searching = True
        self.last_search_print = 0        

        self.running = True

        self.prev_time = time.time()
        self.last_distance_print = 0

        self.mission_completed = False

        self.stop_distance = 0.7

        self.target_object = input("Enter target object: ").strip().lower()
        print(f"Searching for: {self.target_object}")

        self.spin_thread = threading.Thread(
            target=self.spin_thread_func,
            daemon=True
        )
        self.spin_thread.start()


    def spin_thread_func(self):

        while rclpy.ok() and self.running:
            rclpy.spin_once(self, timeout_sec=0.05)

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        with self.frame_lock:
            self.latest_frame = frame

    def stop(self):

        self.running = False

        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=1)

    def display_image(self):

        cv2.namedWindow(
            "YOLO Detection",
            cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
        )

        cv2.resizeWindow("YOLO Detection", 1600, 900)

        while rclpy.ok() and self.running:

            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

            if frame is not None:

                result = self.run_yolo(frame)

                cv2.imshow("YOLO Detection", result)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                self.running = False
                break

        cv2.destroyAllWindows()

    def run_yolo(self, frame):

        CONF_THRESHOLD = 0.35
        results = self.model(
            frame,
            conf=CONF_THRESHOLD,
            imgsz=640,
            verbose=False
        )

        target_found = False

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                if class_name.lower() != self.target_object:
                    continue
                
                target_found = True

                detections.append(
                    f"{class_name} ({confidence:.2f})"
                )
                color = self.class_color(class_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2
                )

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                distance = None
                with self.frame_lock:
                    if self.latest_depth is not None:

                        h, w = self.latest_depth.shape[:2]

                        depth_x = int(cx * w / frame.shape[1])
                        depth_y = int(cy * h / frame.shape[0])

                        if 0 <= depth_x < w and 0 <= depth_y < h:
                            distance = float(self.latest_depth[depth_y, depth_x])
                            
                            if np.isnan(distance) or np.isinf(distance) or distance <= 0:
                                distance = None


                if distance is not None:

                    image_center = frame.shape[1] // 2
                    error = cx - image_center

                    angular = -0.002 * error

                    if distance > self.stop_distance:

                        self.move_robot(
                            linear=0.20,
                            angular=angular
                        )

                        if time.time() - self.last_distance_print > 1.0:
                            print(f"Target Locked")
                            print(f"Distance: {distance:.2f} m")
                            self.last_distance_print = time.time()

                    else:

                        self.move_robot(
                            linear=0.0,
                            angular=0.0
                        )

                        if not self.mission_completed:

                            print("\nMission Completed")
                            print("Target Reached Successfully")

                            self.mission_completed = True


                if distance is not None:
                    label = f"{class_name} {confidence:.2f} {distance:.2f} m"
                else:
                    label = f"{class_name} {confidence:.2f}"

                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                text_y = max(y1 - 10, th + 10)

                cv2.rectangle(frame, (x1, text_y - th - baseline), (x1 + tw + 10, text_y + baseline), color, -1
                )
                cv2.putText(frame, label, (x1 + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )

                cv2.circle(frame, (cx, cy), 5, color, -1
                )
                
        current_time = time.time()
        fps = 1.0 / max(current_time - self.prev_time, 1e-6)
        self.prev_time = current_time
        dashboard_width = 350
        dashboard = np.zeros(
            (frame.shape[0], dashboard_width, 3),
            dtype=np.uint8
        )

        cv2.putText(
            dashboard, "Detections", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
        )

        cv2.putText(dashboard,f"FPS : {fps:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.putText(dashboard, f"Objects : {len(detections)}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        y = 170

        for det in detections[:25]:

            cv2.putText(dashboard, det, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

            y += 30

        combined = np.hstack((frame, dashboard))

        if target_found:

            if self.searching:
                print("Target Found!")

            self.searching = False
            #self.mission_completed = False

        else:

            if not self.mission_completed:
                self.searching = True
                self.move_robot(
                    linear=0.0,
                    angular=0.3
                )
            #self.searching = True
            #self.rotate_robot(0.3)

                if time.time() - self.last_search_print > 1.0:
                    print("Searching...")
                    self.last_search_print = time.time()

        return combined

    def class_color(self, class_id):

        np.random.seed(class_id)

        return tuple(
            int(c)
            for c in np.random.randint(100, 255, 3)
        )

    def move_robot(self, linear, angular):

        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.cmd_pub.publish(msg)


    def depth_callback(self, msg):

        depth = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="passthrough"
        )

        with self.frame_lock:
            self.latest_depth = depth


def main(args=None):

    print("OpenCV Version:", cv2.__version__)

    rclpy.init(args=args)

    node = YoloDetectorNode()

    try:
        node.display_image()

    except KeyboardInterrupt:
        pass

    finally:

        node.stop()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()