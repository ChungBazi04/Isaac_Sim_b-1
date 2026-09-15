"""
pick_place_color_loop.py 의 ColorLink 송수신 테스트 — 시스템 ROS 2 필요 (Isaac Sim 불필요)

    source /opt/ros/jazzy/setup.bash
    cd src_pra/M0609/_tmp_color_loop_test
    python3 -m pytest tests/test_color_link_ros.py -v

팀 로봇이 반응하지 않도록 도메인 77, localhost 한정, 테스트 전용 토픽을 쓴다.
"""

import os
import sys
import time
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")
from std_msgs.msg import Int32, String  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_isaac import load_script  # noqa: E402

M = load_script("color_loop_ros")

TEST_DOMAIN = 77
PREFIX = f"/test_{os.getpid()}"


@pytest.fixture
def link(monkeypatch):
    monkeypatch.setenv("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
    monkeypatch.delenv("ROS_STATIC_PEERS", raising=False)
    link = M.ColorLink(node_name="color_loop_test",
                       color_topic=PREFIX + "/detected_color",
                       state_topic=PREFIX + "/mission_state",
                       result_topic=PREFIX + "/mission_result",
                       domain_id=TEST_DOMAIN)
    yield link
    link.close()


def wait_until(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_receive_colors_and_publish_state(link):
    tester = rclpy.create_node("color_loop_tester")
    pub = tester.create_publisher(Int32, PREFIX + "/detected_color", 10)
    states, results = [], []
    tester.create_subscription(String, PREFIX + "/mission_state", lambda m: states.append(m.data), 10)
    tester.create_subscription(String, PREFIX + "/mission_result", lambda m: results.append(m.data), 10)
    try:
        assert wait_until(lambda: pub.get_subscription_count() > 0)
        assert wait_until(lambda: link._state_pub.get_subscription_count() > 0)

        for code in (2, 2, 0, 1):
            pub.publish(Int32(data=code))
        received = []
        assert wait_until(lambda: received.extend(link.poll()) or len(received) >= 4)
        assert received == [2, 2, 0, 1]

        for state in ("SPAWN", "SPAWN", "WAIT_COLOR", "DONE"):
            link.publish_state(state)
        link.publish_result("SUCCESS #1 detected BLUE spawned BLUE")

        def spin():
            rclpy.spin_once(tester, timeout_sec=0.05)
            return len(states) >= 3 and results
        assert wait_until(spin)
        assert states == ["SPAWN", "WAIT_COLOR", "DONE"]
        assert results == ["SUCCESS #1 detected BLUE spawned BLUE"]
    finally:
        tester.destroy_node()
