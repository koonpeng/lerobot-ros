import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import rclpy
import rclpy.qos
from lerobot.cameras import Camera, CameraConfig, ColorMode
from numpy.typing import NDArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.subscription import Subscription
from sensor_msgs.msg import Image


@CameraConfig.register_subclass("ros2")
@dataclass
class ROS2CameraConfig(CameraConfig):
    name: str
    topic: str

    def __post_init__(self):
        pattern = r"^[a-zA-Z_]+$"
        if not re.match(pattern, self.name):
            raise ValueError("Invalid name: Only letters and underscores are allowed.")


class ROS2Camera(Camera):
    executor: SingleThreadedExecutor | None = None
    executor_thread: threading.Thread | None = None

    def __init__(self, config: ROS2CameraConfig) -> bool:
        super().__init__(config)
        self.config = config
        self.node: Node | None = None
        self.image_sub: Subscription[Image] | None = None
        self.last_image: Image | None = None

    @property
    def is_connected(self) -> bool:
        return self.image_sub is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        if not rclpy.ok():
            rclpy.init()
        node = Node("lerobot_ros_find_cameras")
        time.sleep(2)  # wait for discovery
        topics = node.get_topic_names_and_types()
        camera_topics = [t for t in topics if t[1][0] == "sensor_msgs/msg/Image"]
        return [{"topic": t[0]} for t in camera_topics]

    def connect(self, warmup: bool = True) -> None:
        if not rclpy.ok():
            rclpy.init()
        if self.executor is None:
            self.executor = SingleThreadedExecutor()
            self.executor_thread = threading.Thread(
                target=self.executor.spin, daemon=True
            )
            self.executor_thread.start()

        self.node = Node(f"lerobot_ros_camera_{self.config.name}")

        def on_recv(msg: Image):
            self.last_image = msg

        self.image_sub = self.node.create_subscription(
            Image,
            self.config.topic,
            on_recv,
            rclpy.qos.qos_profile_sensor_data,
        )

        self.executor.add_node(self.node)
        self.executor.wake()

    def read(self, color_mode: ColorMode | None = None) -> NDArray:
        if not self.last_image:
            return np.array([], dtype=np.uint8)
        if color_mode is not None and not self.last_image.encoding.startswith(
            color_mode
        ):
            raise NotImplementedError("image conversion not supported yet")
        return np.array(self.last_image.data, dtype=np.uint8).reshape(
            (self.config.height, self.config.width, 3)
        )

    def async_read(self, timeout_ms: float = 0) -> NDArray:
        return self.read()

    def disconnect(self) -> None:
        self.executor.remove_node(self.node)
        self.image_sub.destroy()
        self.image_sub = None
        self.node.destroy_node()
        self.node = None
