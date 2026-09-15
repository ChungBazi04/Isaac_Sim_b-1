"""
PC A — 색상 결과에 따라 목적지를 고르는 Pick & Place

    export ROS_DOMAIN_ID=135
    isaac_python 7_pick_place_color.py

기존 6_pick_place.py의 로봇, IK, 그리퍼, FSM 코드를 그대로 사용한다.
PC B가 /m0609/detected_color로 보내는 값을 기다린 뒤 동작한다.
  1: 파랑 마커
  2: 초록 마커
"""

import importlib.util
from pathlib import Path


# 숫자로 시작하는 기존 파일을 모듈로 불러온다.
THIS_DIR = Path(__file__).resolve().parent
BASE_PATH = THIS_DIR / "6_pick_place.py"
SPEC = importlib.util.spec_from_file_location("m0609_pick_place_base", BASE_PATH)
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


# SimulationApp이 생성된 뒤 ROS 2 Bridge를 켜야 한다.
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
base.simulation_app.update()

import numpy as np
import rclpy
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from rclpy.node import Node
from std_msgs.msg import Int32


# 기존 USD의 빨간 큐브 대신 카메라와 로봇만 있는 USD를 사용한다.
base.USD_PATH = str(base.M0609_DIR / "Collected_m0609_camera/m0609_camera.usd")

BLUE = 1
GREEN = 2

COLOR_NAME = {
    BLUE: "BLUE",
    GREEN: "GREEN",
}

COLOR_RGB = {
    BLUE: np.array([0.0, 0.0, 1.0]),
    GREEN: np.array([0.0, 1.0, 0.0]),
}

PLACE_BY_COLOR = {
    BLUE: np.array([0.45, -0.15]),
    GREEN: np.array([0.45, 0.15]),
}

CUBE_SIZE = 0.05
MARKER_SIZE = 0.10
MARKER_HEIGHT = 0.005


class PCASub(Node):
    """PC B가 판별한 색상 번호를 받는다."""

    def __init__(self):
        super().__init__("pc_a_pick_place")
        self.color_code = 0
        self.create_subscription(
            Int32,
            "/m0609/detected_color",
            self._color_callback,
            10,
        )

    def _color_callback(self, msg):
        if msg.data in PLACE_BY_COLOR:
            self.color_code = int(msg.data)

    def reset_color(self):
        self.color_code = 0


class ColorPickPlaceTask(base.M0609Task):
    """기존 로봇 씬에 무작위 큐브와 두 색상 마커를 추가한다."""

    def __init__(self, name):
        super().__init__(name=name)
        self.cube_color = int(np.random.choice([BLUE, GREEN]))

    def set_up_scene(self, scene):
        super().set_up_scene(scene)

        px, py = base.PICK_XY
        scene.add(
            DynamicCuboid(
                prim_path="/World/pick_cube",
                name="pick_cube",
                position=np.array([px, py, CUBE_SIZE / 2.0]),
                scale=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
                color=COLOR_RGB[self.cube_color],
            )
        )

        bx, by = PLACE_BY_COLOR[BLUE]
        scene.add(
            VisualCuboid(
                prim_path="/World/blue_marker",
                name="blue_marker",
                position=np.array([bx, by, MARKER_HEIGHT / 2.0]),
                scale=np.array([MARKER_SIZE, MARKER_SIZE, MARKER_HEIGHT]),
                color=COLOR_RGB[BLUE],
            )
        )

        gx, gy = PLACE_BY_COLOR[GREEN]
        scene.add(
            VisualCuboid(
                prim_path="/World/green_marker",
                name="green_marker",
                position=np.array([gx, gy, MARKER_HEIGHT / 2.0]),
                scale=np.array([MARKER_SIZE, MARKER_SIZE, MARKER_HEIGHT]),
                color=COLOR_RGB[GREEN],
            )
        )

        print(f"   cube color   {COLOR_NAME[self.cube_color]}")


def main():
    rclpy.init()
    color_sub = PCASub()

    world = base.World(stage_units_in_meters=1.0)

    base.section("SCENE")
    task = ColorPickPlaceTask(name="m0609_color_task")
    world.add_task(task)
    world.reset()

    robot = task.robot
    robot.initialize()
    base.init_gripper(robot, world)
    base.set_ready_pose(robot)
    for _ in range(30):
        world.step(render=True)

    base.print_dof_info(robot)

    base.section("SOLVER")
    ik_solver = base.create_ik_solver(robot)
    target_quat = base.make_target_quat(
        base.APPROACH_ROLL_DEG,
        base.APPROACH_PITCH_DEG,
        base.GRIPPER_YAW_DEG,
    )

    base.section("RUN")
    print("   press Play in the viewport")
    print("   waiting for /m0609/detected_color (1=BLUE, 2=GREEN)\n")

    fsm = None
    was_playing = False
    step = 0

    while base.simulation_app.is_running():
        world.step(render=True)
        base.time.sleep(0.005)

        is_playing = world.is_playing()

        if is_playing and not was_playing:
            world.reset()
            robot.initialize()
            base.init_gripper(robot, world)
            base.set_ready_pose(robot)
            color_sub.reset_color()
            fsm = None
            step = 0
            print("\n   waiting for color result...")

        rclpy.spin_once(color_sub, timeout_sec=0.0)

        if is_playing and fsm is None and color_sub.color_code in PLACE_BY_COLOR:
            color_code = color_sub.color_code
            base.PLACE_XY = PLACE_BY_COLOR[color_code].copy()
            fsm = base.PickPlaceFSM(robot)
            print(f"   detected      {COLOR_NAME[color_code]} ({color_code})")
            print(f"   place xy      {base.vec(base.PLACE_XY)}")
            base.print_target_info(target_quat)

        if is_playing and fsm is not None:
            target_tcp = fsm.current_target()
            flange_target = base.tcp_to_flange(target_tcp, target_quat)

            action, solved = ik_solver.compute_inverse_kinematics(
                target_position=flange_target,
                target_orientation=target_quat,
            )
            if solved:
                robot.apply_action(action)

            robot.apply_action(robot.gripper.forward(action=fsm.gripper))
            fsm.advance()

            if step % base.LOG_INTERVAL == 0:
                base.print_status(robot, solved, fsm, target_tcp)
            step += 1

        was_playing = is_playing

    color_sub.destroy_node()
    rclpy.shutdown()
    base.simulation_app.close()


if __name__ == "__main__":
    main()
