"""
[TEST] 랜덤 큐브 스폰 → 색상 번호 수신 → Pick & Place 반복

    ./run_isaac.sh                          감지 노드(PC B)와 함께
    ./run_isaac.sh --self-color             감지 노드 없이 스폰한 색으로 루프
    ./run_isaac.sh --no-ros --self-color    ROS 없이 씬·로봇·루프만

테스트용 단일 파일. 다른 파일을 import 하지 않는다.
run_isaac.sh 는 시스템 ROS 환경을 걷어낸 뒤 이 파일을 Isaac Sim python 으로 실행한다.
종료 원인은 화면과 logs/crash_*.txt 에 남는다.
  1. 실행하면 바로 Play 되고 red_block 을 끈다
  2. 파랑·초록 중 랜덤 색 큐브를 홈 자세 그리퍼 바로 아래 랜덤 위치에 스폰한다
  3. /m0609/detected_color (Int32) 를 기다린다
       1 파랑 → 파란 마커, 2 초록 → 초록 마커, 0·그 외 → 움직이지 않는다
  4. Pick & Place 후 홈으로 돌아오면 다음 큐브를 스폰한다 (반복)
  5. /m0609/mission_state, /m0609/mission_result 로 상태를 알리고
     logs/run_*.csv 에 미션마다 한 줄씩 기록한다

주의
  이 터미널에서는 source /opt/ros/jazzy/setup.bash 를 하지 않는다 (내장 rclpy 와 충돌)
  카메라 영상 토픽은 /rgb (USD 의 camera_graph)
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# rclpy 는 bridge extension 을 켠 뒤에 import 해야 한다
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import csv
import os
import random
import sys
import traceback
from collections import deque
from datetime import datetime

from pathlib import Path
import time

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.tasks import BaseTask
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver,
    ArticulationKinematicsSolver,
)

from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid


# ══════════════════════════════════════════════════════════════
#  색상 번호
# ══════════════════════════════════════════════════════════════
COLOR_NONE  = 0
COLOR_BLUE  = 1
COLOR_GREEN = 2

COLOR_NAMES = {
    COLOR_NONE:  "NONE",
    COLOR_BLUE:  "BLUE",
    COLOR_GREEN: "GREEN",
}

VALID_MISSION_COLORS = (COLOR_BLUE, COLOR_GREEN)


# ══════════════════════════════════════════════════════════════
#  토픽 계약
# ══════════════════════════════════════════════════════════════
COLOR_TOPIC  = "/m0609/detected_color"   # std_msgs/Int32
STATE_TOPIC  = "/m0609/mission_state"    # std_msgs/String
RESULT_TOPIC = "/m0609/mission_result"   # std_msgs/String
QOS_DEPTH    = 10


# ══════════════════════════════════════════════════════════════
#  미션 시작 판단
# ══════════════════════════════════════════════════════════════
class ColorGate:
    """
    색상 메시지를 "미션 시작" 한 번으로 바꾼다.

    색상 감지 노드는 같은 색을 매 프레임 계속 발행한다.
    그대로 쓰면 같은 큐브로 미션이 여러 번 시작되므로 여기서 거른다.

      armed   다음 미션을 받을 수 있는 상태
      busy    미션 수행 중. 들어오는 색상은 모두 무시한다

    미션이 끝나면 armed 로 바로 돌아가지 않는다.
    놓은 큐브가 카메라에 계속 보이면 같은 색이 또 들어오기 때문이다.
      - 0 (미검출) 을 한 번 받거나
      - 새 큐브를 스폰하고 rearm() 을 부르면
    다시 armed 가 된다.
    """

    def __init__(self, log=print):
        self._log = log
        self.reset()

    def reset(self):
        """처음 상태. 바로 미션을 받을 수 있다"""
        self.busy = False
        self.armed = True
        self.active_color = None

    # ── 입력 ─────────────────────────────────────────────
    def on_color(self, code):
        """
        색상 번호 하나를 처리한다.

        미션을 시작해야 하면 색상 번호(1 또는 2)를, 아니면 None 을 돌려준다.
        """
        # bool 은 int 의 하위 타입이라 따로 거른다
        if isinstance(code, bool) or not isinstance(code, int):
            self._log(f"   [color] WARN  invalid type {code!r} ignored")
            return None

        if code == COLOR_NONE:
            if not self.busy and not self.armed:
                self.armed = True
                self._log("   [color] cleared  ready for next mission")
            return None

        if code not in VALID_MISSION_COLORS:
            self._log(f"   [color] WARN  invalid code {code} ignored")
            return None

        if self.busy or not self.armed:
            return None

        self.busy = True
        self.armed = False
        self.active_color = code
        self._log(f"   [color] {code} {COLOR_NAMES[code]}  mission start")
        return code

    # ── 미션 쪽에서 부르는 함수 ──────────────────────────
    def mission_finished(self):
        """DONE 또는 ERROR 로 끝났을 때. 0 또는 rearm() 전까지 새 미션을 막는다"""
        self.busy = False
        self.armed = False
        self.active_color = None

    def rearm(self):
        """새 큐브를 스폰했을 때. 0 을 기다리지 않고 바로 받는다"""
        if not self.busy:
            self.armed = True


# ══════════════════════════════════════════════════════════════
#  ROS 연결
# ══════════════════════════════════════════════════════════════
class ColorLink:
    """
    색상 번호를 구독하고 미션 상태·결과를 발행한다.

    Isaac Sim 메인 루프는 멈추면 안 되므로 spin 을 쓰지 않는다.
    매 스텝 poll() 을 불러 쌓인 메시지만 꺼낸다.

        link = ColorLink()
        while simulation_app.is_running():
            world.step(render=True)
            for code in link.poll():
                ...
        link.close()
    """

    # 한 번의 poll 에서 처리할 최대 콜백 수 (루프가 막히지 않도록)
    MAX_CALLBACKS_PER_POLL = 20

    def __init__(self, node_name="m0609_pick_place", color_topic=COLOR_TOPIC,
                 state_topic=STATE_TOPIC, result_topic=RESULT_TOPIC,
                 domain_id=None):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from std_msgs.msg import Int32, String

        self._rclpy = rclpy
        self._String = String

        # domain_id 가 None 이면 ROS_DOMAIN_ID 환경변수를 따른다
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(domain_id=domain_id)

        qos = QoSProfile(
            depth=QOS_DEPTH,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )

        self._node = rclpy.create_node(node_name)
        self._inbox = deque(maxlen=QOS_DEPTH)
        self._node.create_subscription(
            Int32, color_topic, lambda msg: self._inbox.append(msg.data), qos)
        self._state_pub = self._node.create_publisher(String, state_topic, qos)
        self._result_pub = self._node.create_publisher(String, result_topic, qos)
        self._last_state = None

        self.log(f"subscribe {color_topic}  publish {state_topic}, {result_topic}")

    def poll(self):
        """도착한 색상 번호를 도착 순서대로 돌려준다. 기다리지 않는다"""
        for _ in range(self.MAX_CALLBACKS_PER_POLL):
            before = len(self._inbox)
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
            if len(self._inbox) == before:
                break

        codes = list(self._inbox)
        self._inbox.clear()
        return codes

    def publish_state(self, state):
        """상태가 바뀔 때만 발행한다"""
        if state == self._last_state:
            return
        self._last_state = state
        self._state_pub.publish(self._String(data=state))

    def publish_result(self, result):
        self._result_pub.publish(self._String(data=result))

    def log(self, text):
        self._node.get_logger().info(text)

    def warn(self, text):
        self._node.get_logger().warning(text)

    def close(self):
        self._node.destroy_node()
        if self._owns_context:
            self._rclpy.try_shutdown()


# ══════════════════════════════════════════════════════════════
#  경로
# ══════════════════════════════════════════════════════════════
THIS_DIR  = Path(__file__).resolve().parent
M0609_DIR = THIS_DIR.parent

USD_PATH         = str(M0609_DIR / "Collected_camera_cube/m0609_camera.usd")
URDF_PATH        = str(M0609_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
DESCRIPTION_PATH = str(M0609_DIR / "descriptor/m0609_description.yaml")


# ══════════════════════════════════════════════════════════════
#  로봇 설정
# ══════════════════════════════════════════════════════════════
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME    = "link_6"

# Drive 는 팔 6축에만 적용한다
ARM_JOINTS = ["joint_1", "joint_2", "joint_3",
              "joint_4", "joint_5", "joint_6"]

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8

# 로봇 base 의 월드 pose — Lula 가 월드 좌표를 base 좌표로 바꿀 때 쓴다
ROBOT_BASE_POS  = np.array([0.0, 0.0, 0.0])
ROBOT_BASE_QUAT = np.array([1.0, 0.0, 0.0, 0.0])

# 도달 범위 판정 기준 (URDF 실측)
#   어깨 높이 = base_link -> joint_1 = 0.1345
#   최대 반경 = 0.411 + 0.368 + 0.121 = 0.900
SHOULDER_Z = 0.1345
SPEC_REACH = 0.900

# 시작 자세 — 그리퍼가 아래를 향하도록 미리 굽혀 둔다
READY_JOINTS_DEG = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]


# ══════════════════════════════════════════════════════════════
#  그리퍼 설정
# ══════════════════════════════════════════════════════════════
# finger_joint 가 구동 관절이고 나머지 5개는 Mimic 으로 따라온다
# 두 번째 이름은 ParallelGripper 가 요구하는 형식상 필요하다
GRIPPER_JOINTS = ["finger_joint", "right_inner_knuckle_joint"]

# finger_joint 절대 목표값 (라디안)
#   Physics Inspector 는 도로 표시한다.  0.0 ~ 67.609 deg = 0.0 ~ 1.18 rad
GRIPPER_OPEN_POS  = 0.0     #   0.0 deg
GRIPPER_CLOSE_POS = 0.8     #  45.8 deg



# ══════════════════════════════════════════════════════════════
#  TCP 오프셋
# ══════════════════════════════════════════════════════════════
# link_6 로컬 좌표계에서 손가락 패드 끝까지의 거리 (실측)
#   손가락 패드 범위  0.13632 ~ 0.19671
#   링크 원점 0.14155 는 관절 위치이지 파지면이 아니다
FINGER_PAD_TIP_Z = 0.19671
TCP_OFFSET = np.array([0.0, 0.0, FINGER_PAD_TIP_Z])


# ══════════════════════════════════════════════════════════════
#  테스트 씬 설정
# ══════════════════════════════════════════════════════════════
LOG_DIR = THIS_DIR / "logs"

# 실행 옵션
#   --no-ros       rclpy 를 쓰지 않는다 (씬·로봇만 확인)
#   --self-color   스폰한 큐브 색을 감지 결과로 쓴다 (감지 노드 없이 루프 확인)
USE_ROS    = "--no-ros" not in sys.argv
SELF_COLOR = "--self-color" in sys.argv

# USD 에 들어 있는 red_block 은 스폰 영역·카메라 시야와 겹쳐서 끈다
DISABLE_PRIMS = ["/World/red_block"]

# 큐브
CUBE_SIZE       = 0.05
CUBE_MASS       = 0.05
CUBES_PER_COLOR = 4
CUBE_COLORS = {
    COLOR_BLUE:  np.array([0.0, 0.0, 1.0]),
    COLOR_GREEN: np.array([0.0, 1.0, 0.0]),
}

# 스폰 — 홈 자세 TCP 바로 아래를 중심으로 x, y 각각 ± 범위 안에서 랜덤
SPAWN_HALF_RANGE = np.array([0.06, 0.06])
SPAWN_DROP       = 0.002    # 바닥에 박히지 않도록 살짝 띄워 놓는다
SPAWN_SEED       = None     # 숫자로 두면 같은 스폰 순서를 재현한다

# 마커 — 홈 자세 카메라 시야 밖에 둔다
#   컬러 카메라는 바로 아래를 보고 화각이 90도라 넓다
#   USD 기준 바닥 시야  x -0.02 ~ +0.73,  y -0.33 ~ +0.43
#   시야 안에 있으면 감지 노드가 마커 색을 큐브로 착각한다
BLUE_MARKER_XY  = np.array([0.30, -0.45])
GREEN_MARKER_XY = np.array([0.30,  0.52])
MARKER_XY = {COLOR_BLUE: BLUE_MARKER_XY, COLOR_GREEN: GREEN_MARKER_XY}
MARKER_SIZE = np.array([0.24, 0.09, 0.004])
# 마커 하나에 큐브 3개를 x 방향으로 나란히 놓는다
MARKER_SLOT_OFFSETS_X = [-0.07, 0.0, 0.07]

# 주차 — 쓰지 않는 큐브를 로봇 뒤, 카메라 시야 밖에 둔다
PARK_X    = -0.45
PARK_Y0   = -0.35
PARK_STEP = 0.10


# 높이
#   PICK_Z    TCP(손가락 패드 끝) 높이. 큐브 윗부분을 물도록 큐브 높이보다 낮게 둔다
#   PLACE_Z   놓을 때는 살짝 높게 두어 큐브가 튀지 않도록 한다
#   APPROACH  집기 전 대기 높이
#   LIFT      들고 이동할 높이
PICK_Z          = CUBE_SIZE - 0.02      # 5cm 큐브의 윗부분 2cm 를 문다
PLACE_Z         = PICK_Z + 0.005
APPROACH_HEIGHT = 0.25
LIFT_HEIGHT     = 0.23

# 그리퍼를 닫고 기다리는 스텝 수
GRIPPER_WAIT = 120

# 보간 파라미터
#   스텝 수를 고정하면 시간이 고정되어 먼 구간일수록 빨라진다.
#   스텝당 이동 거리를 고정하고 구간 길이로 스텝 수를 계산한다.
TCP_SPEED  = 0.004     # 스텝당 TCP 이동 거리(m)
MIN_STEPS  = 60        # 짧은 구간이 순간이동하지 않도록
MAX_STEPS  = 600       # 스텝 수 폭주 방지
HOLD_STEPS = 60        # 지점 도착 후 멈춰 있는 시간

# 접근 방향 — 툴(link_6 로컬 +Z)이 어디를 향할지
#   roll  pitch      방향
#    180      0      바닥
#    180     90      +x 수평
#    180    -90      -x 수평
#     90      0      -y 수평
#    -90      0      +y 수평
#      0      0      하늘
APPROACH_ROLL_DEG  = 180.0
APPROACH_PITCH_DEG = 0.0

# 툴축 회전 — 접근 방향은 그대로, 손가락(로컬 +X)만 돌아간다
GRIPPER_YAW_DEG = 0.0


# ══════════════════════════════════════════════════════════════
#  실패 판정
# ══════════════════════════════════════════════════════════════
# 보간이 끝났는데 이만큼 떨어져 있으면 아직 도착하지 않은 것이다
REACH_TOL = 0.02         # m
# 보간이 끝난 뒤 도착을 기다리는 최대 스텝 수. 넘기면 ERROR
SETTLE_TIMEOUT = 240
# IK 가 연속으로 이만큼 실패하면 ERROR
IK_FAIL_LIMIT = 30
# Play 직후 시작 자세가 자리잡을 때까지 기다리는 스텝 수 (홈 TCP 기록용)
HOME_SETTLE_STEPS = 30

# 스폰 후 색상을 무시하는 스텝 수 — 큐브가 자리잡고 감지 노드가 새 색을 확정할 시간
SPAWN_SETTLE_STEPS = 90
# 미션이 끝나고 다음 큐브를 스폰하기까지 스텝 수
NEXT_SPAWN_DELAY = 60
# 색상을 기다리는 동안 안내를 찍는 간격
WAIT_HINT_STEPS = 600
# Play 후 이 스텝에서 Action Graph 에러와 ROS bridge 상태를 한 번 찍는다
GRAPH_REPORT_STEP = 120


# ══════════════════════════════════════════════════════════════
#  회전 유틸
# ══════════════════════════════════════════════════════════════
def quat_mul(a, b):
    """쿼터니언 곱. 순서는 (w, x, y, z)"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_axis(axis, deg):
    """회전축과 각도(도)로 쿼터니언을 만든다"""
    half = np.radians(deg) / 2.0
    a = np.array(axis, dtype=float)
    a = a / np.linalg.norm(a)
    return np.concatenate([[np.cos(half)], a * np.sin(half)])


def make_target_quat(roll_deg, pitch_deg, yaw_deg):
    """
    각도 세 개로 목표 자세를 만든다.

    roll, pitch 로 접근 방향을 정한 뒤 yaw 를 마지막에 곱한다.
    마지막에 곱하면 툴 로컬 Z축 회전이 되므로
    접근 방향은 유지되고 손가락 방향만 바뀐다.
    """
    q = quat_mul(quat_from_axis([1, 0, 0], roll_deg),
                 quat_from_axis([0, 1, 0], pitch_deg))
    q = quat_mul(q, quat_from_axis([0, 0, 1], yaw_deg))
    return q / np.linalg.norm(q)


def quat_to_matrix(q):
    """
    쿼터니언을 회전행렬로 바꾼다.
    각 열이 로컬 축의 월드 방향이다.
      1열 = 로컬 +X (손가락 방향)
      3열 = 로컬 +Z (툴 방향)
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


# ══════════════════════════════════════════════════════════════
#  TCP 변환
# ══════════════════════════════════════════════════════════════
def tcp_to_flange(tcp_pos, quat):
    """
    손가락 끝 목표를 플랜지 목표로 바꾼다.

    오프셋은 link_6 로컬 좌표이므로 목표 자세만큼 회전시킨 뒤 빼야 한다.
    """
    R = quat_to_matrix(quat)
    return np.array(tcp_pos) - R @ TCP_OFFSET


def get_tcp_pose(robot):
    """현재 플랜지 pose 로부터 손가락 끝의 월드 위치를 구한다"""
    pos, quat = robot.end_effector.get_world_pose()
    return pos + quat_to_matrix(quat) @ TCP_OFFSET


# ══════════════════════════════════════════════════════════════
#  궤적 보간
# ══════════════════════════════════════════════════════════════
def steps_for(start, goal):
    """구간 길이를 속도로 나눠 스텝 수를 정한다"""
    dist = float(np.linalg.norm(goal - start))
    return int(np.clip(dist / TCP_SPEED, MIN_STEPS, MAX_STEPS)), dist


def lerp(start, goal, alpha):
    """시작점에서 목표점까지 선형 보간"""
    return start + alpha * (goal - start)


class ColorPickPlaceFSM:
    """
    색상 번호 하나로 시작하는 Pick & Place 상태 기계.

      WAIT_COLOR     색상 번호 대기 (로봇 정지)
      APPROACH       큐브 위로 접근
      DESCEND        큐브까지 하강
      GRASP          그리퍼 닫기 (제자리)
      LIFT           들어올리기
      MOVE           색상별 놓을 곳 위로 이동
      LOWER          놓을 높이까지 하강
      RELEASE        그리퍼 열기 (제자리)
      RETREAT        놓은 자리에서 위로
      RETURN_HOME    시작 위치로 복귀
      → 끝나면 "DONE" 을 알리고 WAIT_COLOR 로 돌아간다

    실패하면 (IK 연속 실패, 제한 스텝 안에 도착 못함)
      → "ERROR" 를 알리고 RECOVER_OPEN, RECOVER_UP 을 수행한 뒤 WAIT_COLOR
      → 복구 중에 또 실패하면 HALTED 로 멈춘다 (무한 재시도 금지)

    advance() 는 이벤트를 돌려준다: None, "DONE", "ERROR", "RECOVERED", "HALTED"
    """

    GRIPPER_STAGES = {"GRASP": "close", "RELEASE": "open", "RECOVER_OPEN": "open"}

    def __init__(self, robot):
        self._robot = robot
        self.reset()

    def reset(self):
        self.mode = "WAIT"        # WAIT | MISSION | RECOVER | HALTED
        self.plan = []
        self.index = 0
        self.color = None
        self.error = None
        self.gripper = "open"
        self._clear_stage()

    def _clear_stage(self):
        self.step = 0
        self.start = None
        self.goal = None
        self.n_steps = MIN_STEPS
        self.ik_fail = 0

    # ── 상태 ─────────────────────────────────────────────
    @property
    def state_name(self):
        if self.mode == "WAIT":
            return "WAIT_COLOR"
        if self.mode == "HALTED":
            return "HALTED"
        return self.plan[self.index][0]

    @property
    def is_waiting(self):
        return self.mode == "WAIT"

    # ── 시작 ─────────────────────────────────────────────
    def start_mission(self, color, pick_xy, place_xy, home_tcp):
        """대기 중일 때만 미션을 만든다"""
        if self.mode != "WAIT":
            return False

        px, py = pick_xy
        if color not in VALID_MISSION_COLORS:
            raise ValueError(f"no place target for color {color}")
        gx, gy = place_xy
        self.plan = [
            ("APPROACH",    np.array([px, py, APPROACH_HEIGHT])),
            ("DESCEND",     np.array([px, py, PICK_Z])),
            ("GRASP",       np.array([px, py, PICK_Z])),
            ("LIFT",        np.array([px, py, LIFT_HEIGHT])),
            ("MOVE",        np.array([gx, gy, LIFT_HEIGHT])),
            ("LOWER",       np.array([gx, gy, PLACE_Z])),
            ("RELEASE",     np.array([gx, gy, PLACE_Z])),
            ("RETREAT",     np.array([gx, gy, LIFT_HEIGHT])),
            ("RETURN_HOME", np.array(home_tcp)),
        ]
        self.index = 0
        self.color = color
        self.error = None
        self.mode = "MISSION"
        self._clear_stage()
        print(f"\n   MISSION {COLOR_NAMES[color]}  pick {vec(pick_xy)}  place {vec([gx, gy])}")
        return True

    # ── 진행 ─────────────────────────────────────────────
    def current_target(self):
        """이번 스텝의 TCP 목표. 대기 중이면 None (팔에 명령하지 않는다)"""
        if self.mode in ("WAIT", "HALTED"):
            return None
        if self.start is None:
            return self.plan[self.index][1]
        alpha = min(1.0, self.step / float(self.n_steps))
        return lerp(self.start, self.goal, alpha)

    def advance(self, solved=True):
        """한 스텝 진행한다. solved 는 이번 스텝 IK 성공 여부"""
        if self.mode in ("WAIT", "HALTED"):
            return None

        name, goal = self.plan[self.index]

        # 단계에 처음 들어온 순간 시작점과 스텝 수를 정한다
        if self.start is None:
            self.start = get_tcp_pose(self._robot)
            self.goal = goal
            self.gripper = self.GRIPPER_STAGES.get(name, self.gripper)

            if name in self.GRIPPER_STAGES:
                self.n_steps = GRIPPER_WAIT
                dist = 0.0
            else:
                self.n_steps, dist = steps_for(self.start, self.goal)

            print(f"   {name:12s} goal {vec(self.goal)}"
                  f"  {dist:.4f} m  {self.n_steps} steps  gripper {self.gripper}")

        self.step += 1

        if solved:
            self.ik_fail = 0
        else:
            self.ik_fail += 1
            if self.ik_fail >= IK_FAIL_LIMIT:
                return self._fail(f"IK failed {self.ik_fail} steps")

        if self.step < self.n_steps:
            return None

        # 이동 단계는 보간이 끝난 뒤 실제로 도착했는지 확인한다
        if name not in self.GRIPPER_STAGES:
            err = float(np.linalg.norm(get_tcp_pose(self._robot) - self.goal))
            if err > REACH_TOL:
                if self.step >= self.n_steps + SETTLE_TIMEOUT:
                    return self._fail(f"not reached  err {err:.3f} m")
                return None

        return self._next()

    def _next(self):
        self.index += 1
        self._clear_stage()
        if self.index < len(self.plan):
            return None

        finished = self.mode
        self.mode = "WAIT"
        self.plan = []
        self.index = 0
        if finished == "MISSION":
            print(f"   DONE {COLOR_NAMES[self.color]}")
            return "DONE"
        print("   RECOVERED")
        return "RECOVERED"

    def _fail(self, reason):
        name = self.plan[self.index][0]
        self.error = f"{name}: {reason}"
        print(f"   ERROR {self.error}")

        # 복구 중 실패는 다시 복구하지 않는다
        if self.mode == "RECOVER":
            self.mode = "HALTED"
            self.gripper = "open"
            self.plan = []
            self.index = 0
            self._clear_stage()
            return "HALTED"

        tcp = get_tcp_pose(self._robot)
        up = np.array([tcp[0], tcp[1], max(tcp[2], LIFT_HEIGHT)])
        self.plan = [
            ("RECOVER_OPEN", tcp),
            ("RECOVER_UP",   up),
        ]
        self.index = 0
        self.mode = "RECOVER"
        self._clear_stage()
        return "ERROR"


# ══════════════════════════════════════════════════════════════
#  씬 구성 — Task
# ══════════════════════════════════════════════════════════════
def find_prim_path(root_path, name):
    """USD 계층에서 이름으로 prim 경로를 찾는다"""
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None

    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


class M0609Task(BaseTask):
    """
    set_up_scene 은 BaseTask 가 정한 이름이다. World 가 이 이름으로 부른다.
    _ 로 시작하는 메서드는 우리가 나눈 것이라 이름을 바꿔도 된다.
    """

    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self._robot = None
        self._pool = None

    # ── 프레임워크 규약 ──────────────────────────────────
    def set_up_scene(self, scene):
        """world.reset() 안에서 자동으로 불린다"""
        super().set_up_scene(scene)
        self._load_usd()
        self._setup_arm_drives()
        self._register_robot(scene)
        disable_prims(DISABLE_PRIMS)
        self._pool = build_test_scene(scene)
        print("   scene        ready")

    # ── 우리가 나눈 단계 ─────────────────────────────────
    def _load_usd(self):
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()

        world_prim.GetReferences().AddReference(USD_PATH)
        for _ in range(15):
            simulation_app.update()

        print("   USD          loaded")

    def _setup_arm_drives(self):
        """IK 결과를 로봇이 따라가도록 팔 관절의 Drive 를 강화한다"""
        stage = omni.usd.get_context().get_stage()
        count = 0

        for prim in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PRIM_PATH)):
            if prim.GetName() not in ARM_JOINTS:
                continue
            for drive_type in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    count += 1

        print(f"   arm drives   {count}")

    def _register_robot(self, scene):
        """로봇과 그리퍼를 등록한다. world.scene 이 아니라 인자 scene 을 쓴다"""
        ee_path = find_prim_path(ROBOT_PRIM_PATH, EE_LINK_NAME)
        if ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_PRIM_PATH}")

        gripper = ParallelGripper(
            end_effector_prim_path=ee_path,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array([GRIPPER_OPEN_POS] * 2),
            joint_closed_positions=np.array([GRIPPER_CLOSE_POS] * 2),
            action_deltas=None,
        )

        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="m0609_robot",
                end_effector_prim_path=ee_path,
                gripper=gripper,
            )
        )
        print(f"   EE frame     {ee_path}")

    @property
    def robot(self):
        return self._robot

    @property
    def pool(self):
        return self._pool


def init_gripper(robot, world):
    """그리퍼는 Articulation 초기화 이후에 따로 초기화한다"""
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )


def set_ready_pose(robot):
    """시작 자세로 보낸다"""
    q = np.zeros(robot.num_dof)
    q[:6] = np.deg2rad(READY_JOINTS_DEG)
    robot.set_joint_positions(q)


# ══════════════════════════════════════════════════════════════
#  IK 솔버
# ══════════════════════════════════════════════════════════════
def create_ik_solver(robot):
    """
    Lula 계산기를 만들고 로봇과 연결한다.

    LulaKinematicsSolver         : URDF 만 읽는 계산기. 로봇을 모른다
    ArticulationKinematicsSolver : 계산 결과를 로봇 관절 명령으로 바꾼다
    """
    lula = LulaKinematicsSolver(
        robot_description_path=DESCRIPTION_PATH,
        urdf_path=URDF_PATH,
    )

    # 월드 좌표와 base 좌표를 잇는다. 지금은 항등이지만 반드시 호출한다
    lula.set_robot_base_pose(
        robot_position=ROBOT_BASE_POS,
        robot_orientation=ROBOT_BASE_QUAT,
    )

    print(f"   controlled   {', '.join(lula.get_joint_names())}")

    return ArticulationKinematicsSolver(
        robot_articulation=robot,
        kinematics_solver=lula,
        end_effector_frame_name=EE_LINK_NAME,
    )


# ══════════════════════════════════════════════════════════════
#  테스트 씬 — 마커, 큐브 풀, 반복 루프, 기록
# ══════════════════════════════════════════════════════════════
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def disable_prims(paths):
    """USD 에 들어 있지만 테스트에 방해되는 prim 을 끈다. 파일은 바꾸지 않는다"""
    stage = omni.usd.get_context().get_stage()
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)
            print(f"   disabled     {path}")


def build_test_scene(scene):
    """마커 두 개와 색상별 큐브 풀을 만든다. 큐브는 로봇 뒤 주차 위치에서 시작한다"""
    for color, xy in MARKER_XY.items():
        name = COLOR_NAMES[color].lower()
        scene.add(VisualCuboid(
            prim_path=f"/World/test_markers/{name}",
            name=f"marker_{name}",
            position=np.array([xy[0], xy[1], MARKER_SIZE[2] / 2]),
            scale=MARKER_SIZE,
            color=CUBE_COLORS[color],
        ))

    cubes, parks = {}, {}
    index = 0
    for color in VALID_MISSION_COLORS:
        name = COLOR_NAMES[color].lower()
        cubes[color] = []
        for k in range(CUBES_PER_COLOR):
            park = np.array([PARK_X, PARK_Y0 + index * PARK_STEP, CUBE_SIZE / 2])
            cube = scene.add(DynamicCuboid(
                prim_path=f"/World/test_cubes/{name}_{k}",
                name=f"cube_{name}_{k}",
                position=park,
                scale=np.array([CUBE_SIZE] * 3),
                color=CUBE_COLORS[color],
                mass=CUBE_MASS,
            ))
            cubes[color].append(cube)
            parks[id(cube)] = park
            index += 1

    print(f"   markers      blue {vec(BLUE_MARKER_XY)}  green {vec(GREEN_MARKER_XY)}")
    print(f"   cubes        {CUBES_PER_COLOR} per color  size {CUBE_SIZE}")
    return CubePool(cubes, parks, MARKER_XY)


class CubePool:
    """
    미리 만든 큐브를 옮겨 쓰며 스폰·주차한다. 실행 중에 prim 을 만들거나 지우지 않는다.

      parked   로봇 뒤 주차 위치
      active   그리퍼 아래 스폰됨 (이번 미션 대상)
      placed   마커 슬롯에 놓임

    마커마다 슬롯이 여러 개이고 돌아가며 쓴다. 슬롯에 큐브가 있으면 주차시키고 비운다.
    같은 색 큐브가 모두 놓여 있으면 가장 먼저 놓인 큐브를 다시 쓴다.
    """

    def __init__(self, cubes, park_positions, marker_xy):
        self._cubes = cubes
        self._parks = park_positions
        self._marker_xy = marker_xy
        self._color = {id(c): color for color, group in cubes.items() for c in group}
        self._clear()

    def _clear(self):
        self._state = {id(c): "parked" for group in self._cubes.values() for c in group}
        self._slots = {color: [None] * len(MARKER_SLOT_OFFSETS_X) for color in self._marker_xy}
        self._next_slot = {color: 0 for color in self._marker_xy}
        self._placed_order = []

    def reset(self):
        """모든 큐브를 주차 위치로 보낸다. 물리가 초기화된 뒤에 부른다"""
        self._clear()
        for group in self._cubes.values():
            for cube in group:
                self._teleport(cube, self._parks[id(cube)])

    def color_of(self, cube):
        return self._color[id(cube)]

    def state_of(self, cube):
        return self._state[id(cube)]

    def spawn(self, color, xy):
        cube = self._take(color)
        self._teleport(cube, [xy[0], xy[1], CUBE_SIZE / 2 + SPAWN_DROP])
        self._state[id(cube)] = "active"
        return cube

    def reserve_place(self, marker_color):
        """다음 슬롯의 xy 와 슬롯 번호. 슬롯에 있던 큐브는 주차한다"""
        slot = self._next_slot[marker_color]
        self._next_slot[marker_color] = (slot + 1) % len(MARKER_SLOT_OFFSETS_X)
        occupant = self._slots[marker_color][slot]
        if occupant is not None:
            self.park(occupant)
        x, y = self._marker_xy[marker_color]
        return np.array([x + MARKER_SLOT_OFFSETS_X[slot], y]), slot

    def mark_placed(self, cube, marker_color, slot):
        self._slots[marker_color][slot] = cube
        self._placed_order.append(cube)
        self._state[id(cube)] = "placed"

    def park(self, cube):
        self._forget_place(cube)
        self._teleport(cube, self._parks[id(cube)])
        self._state[id(cube)] = "parked"

    def _take(self, color):
        for cube in self._cubes[color]:
            if self._state[id(cube)] == "parked":
                return cube
        for cube in self._placed_order:
            if self.color_of(cube) == color:
                self.park(cube)
                return cube
        raise RuntimeError(f"no {COLOR_NAMES[color]} cube available")

    def _forget_place(self, cube):
        for slots in self._slots.values():
            for i, placed in enumerate(slots):
                if placed is cube:
                    slots[i] = None
        self._placed_order = [c for c in self._placed_order if c is not cube]

    @staticmethod
    def _teleport(cube, position):
        cube.set_world_pose(position=np.array(position, dtype=float), orientation=IDENTITY_QUAT)
        cube.set_linear_velocity(np.zeros(3))
        cube.set_angular_velocity(np.zeros(3))


class Recorder:
    """미션마다 CSV 한 줄. path 가 None 이면 기록하지 않는다"""

    FIELDS = ["time", "mission", "spawned", "detected", "result", "error",
              "pick_x", "pick_y", "place_x", "place_y"]

    def __init__(self, path):
        self._file = None
        if path is None:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def write(self, **row):
        if self._file is None:
            return
        row.setdefault("time", datetime.now().isoformat(timespec="seconds"))
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def fmt(v):
    return "" if v is None else f"{v:.3f}"


class CubeLoop:
    """
    테스트 루프 — 스폰 → 색상 대기 → Pick & Place → 다음 스폰

      phase      발행 상태      내용
      INIT       INIT          홈 TCP 를 기록할 때까지
      SETTLE     SPAWN         큐브가 자리잡고 감지 노드가 새 색을 확정할 때까지 색상 무시
      READY      WAIT_COLOR    색상 번호를 받으면 미션 시작 (큐브 하나당 한 번)
      MISSION    FSM 상태      DONE 또는 ERROR → RECOVER 까지
      COOLDOWN   DONE 등       다음 스폰까지 잠깐 대기
      HALTED     HALTED        복구 실패. 더 이상 스폰하지 않는다

    스폰 색(spawned)은 테스트가 정하고, 로봇은 받은 색(detected)을 따른다.
    둘이 다르면 MISMATCH 로 기록한다.
    """

    STATE_NAMES = {"INIT": "INIT", "SETTLE": "SPAWN", "READY": "WAIT_COLOR", "HALTED": "HALTED"}

    def __init__(self, pool, fsm, gate, link, rng, recorder=None, log=print):
        self._pool = pool
        self._fsm = fsm
        self._gate = gate
        self._link = link
        self._rng = rng
        self._recorder = recorder or Recorder(None)
        self._log = log
        self._clear()

    def _clear(self):
        self.phase = "INIT"
        self.current_cube = None
        self.spawned_color = None
        self.detected_color = None
        self.mission_no = 0
        self.stats = {"SUCCESS": 0, "MISMATCH": 0, "ERROR": 0}
        self._count = 0
        self._wait_steps = 0
        self._cooldown_state = "DONE"
        self._slot = None
        self._pick_xy = None
        self._place_xy = None

    def reset(self):
        """Play 를 새로 시작할 때. 큐브를 모두 주차하고 처음부터"""
        self._pool.reset()
        self._clear()

    @property
    def state_name(self):
        if self.phase == "MISSION":
            return self._fsm.state_name
        if self.phase == "COOLDOWN":
            return self._cooldown_state
        return self.STATE_NAMES[self.phase]

    # ── 매 스텝 ──────────────────────────────────────────
    def update(self, home_tcp, extra_codes=()):
        codes = self._link.poll() + list(extra_codes)

        if self.phase == "INIT":
            if home_tcp is not None:
                self._spawn(home_tcp)

        elif self.phase == "SETTLE":
            self._count -= 1
            if self._count <= 0:
                self._gate.rearm()
                self.phase = "READY"
                self._wait_steps = 0

        elif self.phase == "READY":
            for code in codes:
                color = self._gate.on_color(code)
                if color is not None:
                    self._start(color, home_tcp)
                    break
            if self.phase == "READY":
                self._wait_steps += 1
                if self._wait_steps % WAIT_HINT_STEPS == 0:
                    self._log(f"   waiting      /m0609/detected_color  (last {codes[-1] if codes else 'none'})")

        elif self.phase == "COOLDOWN":
            self._count -= 1
            if self._count <= 0:
                self._spawn(home_tcp)

    def on_event(self, event):
        """FSM 이벤트를 처리한다"""
        if event is None:
            return

        if event == "ERROR":
            self._link.publish_state("ERROR")
            self._link.publish_result(f"ERROR #{self.mission_no} {self._fsm.error}")
            self._link.warn(f"mission #{self.mission_no} error  {self._fsm.error}")
            return

        if event == "DONE":
            self._pool.mark_placed(self.current_cube, *self._slot)
            result = "SUCCESS" if self.detected_color == self.spawned_color else "MISMATCH"
            self._link.publish_state("DONE")
            self._link.publish_result(
                f"{result} #{self.mission_no} detected {COLOR_NAMES[self.detected_color]}"
                f" spawned {COLOR_NAMES[self.spawned_color]}")
            self._finish(result, "", "DONE")

        elif event == "RECOVERED":
            self._pool.park(self.current_cube)
            self._link.publish_state("RECOVERED")
            self._finish("ERROR", self._fsm.error, "RECOVERED")

        elif event == "HALTED":
            self._link.publish_state("HALTED")
            self._link.publish_result(f"HALTED #{self.mission_no} {self._fsm.error}")
            self._finish("ERROR", self._fsm.error, "HALTED")
            self.phase = "HALTED"

    def close(self):
        self._recorder.close()

    # ── 내부 ─────────────────────────────────────────────
    def _spawn(self, home_tcp):
        color = self._rng.choice(VALID_MISSION_COLORS)
        offset = np.array([self._rng.uniform(-h, h) for h in SPAWN_HALF_RANGE])
        xy = np.array(home_tcp[:2], dtype=float) + offset

        self.current_cube = self._pool.spawn(color, xy)
        self.spawned_color = color
        self.detected_color = None
        self._pick_xy = None
        self._place_xy = None
        self.mission_no += 1
        self.phase = "SETTLE"
        self._count = SPAWN_SETTLE_STEPS
        self._log(f"\n   SPAWN #{self.mission_no}  {COLOR_NAMES[color]:5s}  xy {vec(xy)}")

    def _start(self, color, home_tcp):
        pos, _ = self.current_cube.get_world_pose()
        pick_xy = np.array(pos[:2], dtype=float)
        place_xy, slot = self._pool.reserve_place(color)

        if not self._fsm.start_mission(color, pick_xy, place_xy, home_tcp):
            self._gate.mission_finished()
            self._gate.rearm()
            return

        self.detected_color = color
        self._slot = (color, slot)
        self._pick_xy = pick_xy
        self._place_xy = place_xy
        self.phase = "MISSION"
        tag = "match" if color == self.spawned_color else "MISMATCH"
        self._link.log(f"mission #{self.mission_no}  detected {COLOR_NAMES[color]}"
                       f"  spawned {COLOR_NAMES[self.spawned_color]}  ({tag})")

    def _finish(self, result, error, next_state):
        self.stats[result] += 1
        pick = self._pick_xy if self._pick_xy is not None else [None, None]
        place = self._place_xy if self._place_xy is not None else [None, None]
        self._recorder.write(
            mission=self.mission_no,
            spawned=COLOR_NAMES[self.spawned_color],
            detected=COLOR_NAMES.get(self.detected_color, ""),
            result=result,
            error=error or "",
            pick_x=fmt(pick[0]), pick_y=fmt(pick[1]),
            place_x=fmt(place[0]), place_y=fmt(place[1]),
        )
        self._gate.mission_finished()
        self.current_cube = None
        self.phase = "COOLDOWN"
        self._count = NEXT_SPAWN_DELAY
        self._cooldown_state = next_state

        total = sum(self.stats.values())
        self._log(f"   RESULT #{self.mission_no} {result}   total {total}"
                  f"  success {self.stats['SUCCESS']}  mismatch {self.stats['MISMATCH']}"
                  f"  error {self.stats['ERROR']}")


class NullLink:
    """--no-ros 일 때 ColorLink 대신 쓴다. 결과만 화면에 찍는다"""

    def poll(self):
        return []

    def publish_state(self, state):
        pass

    def publish_result(self, result):
        print(f"   [result] {result}")

    def log(self, text):
        print(f"   [ros-off] {text}")

    def warn(self, text):
        print(f"   [ros-off] WARN {text}")

    def close(self):
        pass


def ros_env_report():
    """rclpy 를 불러오기 전에 환경을 찍는다. import 실패 원인 확인용"""
    print(f"   python       {sys.version.split()[0]}")
    for key in ("ROS_DISTRO", "RMW_IMPLEMENTATION", "ROS_DOMAIN_ID", "FASTRTPS_DEFAULT_PROFILES_FILE"):
        print(f"   {key:31s}{os.environ.get(key, '(unset)')}")
    ros_libs = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if "ros" in p]
    print(f"   LD_LIBRARY_PATH (ros)          {ros_libs or '(none)'}")
    system_ros = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p.startswith("/opt/ros")]
    if system_ros:
        print(f"   WARN  system ROS on PYTHONPATH {system_ros}")
        print("         Isaac Sim 은 Python 3.11 내장 rclpy 를 써야 한다. run_isaac.sh 로 실행할 것")


def write_crash_log():
    """예외를 화면과 logs/crash_*.txt 에 남긴다"""
    text = traceback.format_exc()
    bar = "=" * 66
    print(f"\n{bar}\n [CRASH] 종료 원인\n{bar}\n{text}", flush=True)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"crash_{datetime.now():%Y%m%d_%H%M%S}.txt"
        path.write_text(text)
        print(f" saved {path}", flush=True)
    except OSError:
        pass


def graph_report(graph_root="/World/Graph"):
    """ROS bridge 상태와 Action Graph 노드의 에러·경고를 찍는다. 카메라 토픽이 안 나올 때 확인용"""
    try:
        import omni.kit.app
        import omni.graph.core as og
    except Exception as e:  # noqa: BLE001
        print(f"   graph report unavailable  {e}")
        return

    try:
        manager = omni.kit.app.get_app().get_extension_manager()
        print(f"   ros2 bridge  enabled={manager.is_extension_enabled('isaacsim.ros2.bridge')}")

        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(graph_root)
        if not root.IsValid():
            print(f"   WARN  {graph_root} not found  camera topics will not be published")
            return

        found = False
        for prim in Usd.PrimRange(root):
            if not prim.HasAttribute("node:type"):
                continue
            node = og.Controller.node(str(prim.GetPath()))
            for severity, tag in ((og.Severity.ERROR, "ERROR"), (og.Severity.WARNING, "WARN")):
                for text in node.get_compute_messages(severity):
                    found = True
                    print(f"   graph {tag:5s} {prim.GetPath()}  {text}")
        if not found:
            print("   graph        no node errors")
        print("   expected     /rgb /depth /camera_info /clock")
    except Exception as e:  # noqa: BLE001
        print(f"   graph report failed  {e}")


# ══════════════════════════════════════════════════════════════
#  출력
# ══════════════════════════════════════════════════════════════
def section(title):
    print(f"\n{'─' * 66}")
    print(f" {title}")
    print(f"{'─' * 66}")


def vec(v, digits=3):
    """벡터를 고정폭으로 찍는다"""
    return "[" + " ".join(f"{x:+.{digits}f}" for x in v) + "]"


def print_target_info(target_quat):
    """테스트 계획을 확인한다"""
    R = quat_to_matrix(target_quat)

    section("PLAN")
    print(f"   spawn range  +-{vec(SPAWN_HALF_RANGE[:2])} around home tcp xy")
    print(f"   blue  marker {vec(BLUE_MARKER_XY)}   color 1")
    print(f"   green marker {vec(GREEN_MARKER_XY)}   color 2")
    print(f"   slot x       {MARKER_SLOT_OFFSETS_X}")
    print()
    print(f"   cube size    {CUBE_SIZE}  mass {CUBE_MASS}")
    print(f"   approach z   {APPROACH_HEIGHT}")
    print(f"   pick z       {PICK_Z}")
    print(f"   lift z       {LIFT_HEIGHT}")
    print(f"   place z      {PLACE_Z}")
    print()
    print(f"   tcp speed    {TCP_SPEED} m/step")
    print(f"   gripper      open {GRIPPER_OPEN_POS}  close {GRIPPER_CLOSE_POS}")
    print(f"   reach tol    {REACH_TOL} m  settle timeout {SETTLE_TIMEOUT} steps")
    print(f"   spawn settle {SPAWN_SETTLE_STEPS} steps  next spawn {NEXT_SPAWN_DELAY} steps")
    print()
    print(f"   tool   +Z    {vec(R @ np.array([0, 0, 1]))}   approach direction")
    print(f"   finger +X    {vec(R @ np.array([1, 0, 0]))}   finger direction")


def print_dof_info(robot):
    """어떤 관절이 몇 번인지 확인한다"""
    section("DOF")
    for i, name in enumerate(robot.dof_names):
        tag = "arm" if name in ARM_JOINTS else "gripper"
        print(f"   [{i:2d}] {name:28s} {tag}")
    print()
    print(f"   finger index {robot.get_dof_index('finger_joint')}")
    print(f"   num_dof      {robot.num_dof}")


def print_gripper_state(robot, command):
    """명령값과 실제값, Mimic 관절 전체를 함께 본다"""
    q = robot.get_joint_positions()
    actual = q[robot.get_dof_index("finger_joint")]
    print(f"   gripper {command:5s}   finger {actual:+.4f}")
    print(f"   dof[6:12] {vec(q[6:12], 4)}")


def print_status(robot, solved, fsm, target_tcp):
    """현재 단계와 손가락 끝 위치를 함께 찍는다"""
    name = fsm.state_name

    if not solved:
        print(f"   {name:12s} IK FAILED  target {vec(target_tcp)}")
        return

    tcp = get_tcp_pose(robot)
    finger = robot.get_joint_positions()[robot.get_dof_index("finger_joint")]
    print(f"   {name:12s} tcp {vec(tcp)}   finger {finger:+.4f}")


# ══════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════
LOG_INTERVAL = 60


def main():
    world = World(stage_units_in_meters=1.0)

    section("SCENE")
    # world.reset() 이 Task.set_up_scene() 을 자동으로 부른다
    task = M0609Task(name="m0609_task")
    world.add_task(task)
    world.reset()

    robot = task.robot
    robot.initialize()
    init_gripper(robot, world)
    set_ready_pose(robot)
    for _ in range(30):
        world.step(render=True)

    print_dof_info(robot)

    section("SOLVER")
    ik_solver = create_ik_solver(robot)

    target_quat = make_target_quat(
        APPROACH_ROLL_DEG, APPROACH_PITCH_DEG, GRIPPER_YAW_DEG
    )
    print_target_info(target_quat)

    section("ROS")
    ros_env_report()
    if USE_ROS:
        link = ColorLink()
    else:
        link = NullLink()
        print("   --no-ros     rclpy disabled")
    if SELF_COLOR:
        print("   --self-color spawned color is used as the detected color")
    gate = ColorGate()

    log_path = LOG_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.csv"
    recorder = Recorder(log_path)
    print(f"   log          {log_path}")

    fsm = ColorPickPlaceFSM(robot)
    loop = CubeLoop(task.pool, fsm, gate, link, random.Random(SPAWN_SEED), recorder)

    section("RUN")
    print("   auto play  cubes spawn under the gripper and wait for a color\n")
    world.play()

    home_tcp = None
    was_playing = False
    step = 0

    try:
        while simulation_app.is_running():
            world.step(render=True)
            time.sleep(0.005)

            is_playing = world.is_playing()

            # Play 가 시작되는 순간 처음 상태로 되돌린다
            if is_playing and not was_playing:
                world.reset()
                robot.initialize()
                init_gripper(robot, world)
                set_ready_pose(robot)
                fsm.reset()
                gate.reset()
                loop.reset()
                home_tcp = None
                step = 0
                print()

            if not is_playing:
                link.poll()      # 멈춘 동안 온 색상은 버린다
                link.publish_state("STOPPED")
                was_playing = is_playing
                continue

            if step == GRAPH_REPORT_STEP:
                section("GRAPH CHECK")
                graph_report()

            if home_tcp is None and step >= HOME_SETTLE_STEPS:
                home_tcp = get_tcp_pose(robot)
                print(f"   home tcp     {vec(home_tcp)}")

            # --self-color 이면 감지 노드 대신 스폰한 색을 넣는다
            extra = [loop.spawned_color] if SELF_COLOR and loop.phase == "READY" else ()
            loop.update(home_tcp, extra)

            # 팔 — 미션이 없으면 명령하지 않고 제자리를 유지한다
            target_tcp = fsm.current_target()
            solved = True
            if target_tcp is not None:
                flange_target = tcp_to_flange(target_tcp, target_quat)
                action, solved = ik_solver.compute_inverse_kinematics(
                    target_position=flange_target,
                    target_orientation=target_quat,
                )
                if solved:
                    robot.apply_action(action)

            # 그리퍼 — 현재 단계가 정한 상태를 유지한다
            robot.apply_action(robot.gripper.forward(action=fsm.gripper))

            event = fsm.advance(solved)
            loop.on_event(event)
            link.publish_state(loop.state_name)

            if target_tcp is not None and step % LOG_INTERVAL == 0:
                print_status(robot, solved, fsm, target_tcp)
            step += 1

            was_playing = is_playing
    finally:
        loop.close()
        link.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n   interrupted")
    except BaseException:
        write_crash_log()
    finally:
        simulation_app.close()
