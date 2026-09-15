"""
pick_place_color_loop.py 로직 테스트 — Isaac Sim 없이 실행

    cd src_pra/M0609/_tmp_color_loop_test
    python3 -m pytest tests -v

로봇은 목표를 그대로 따라가고, 큐브는 그리퍼가 닫혀 있는 동안 TCP 를 따라간다고 단순화한다.
감지 노드는 "그리퍼 아래에 스폰된 큐브의 색"을 매 스텝 보내는 가짜로 대신한다.
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_isaac import load_script  # noqa: E402

M = load_script()

HOME = np.array([0.37, 0.006, 0.23])
OPPOSITE = {M.COLOR_BLUE: M.COLOR_GREEN, M.COLOR_GREEN: M.COLOR_BLUE}


class FakeCube:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)

    def get_world_pose(self):
        return self.pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def set_world_pose(self, position=None, orientation=None):
        if position is not None:
            self.pos = np.array(position, dtype=float)

    def set_linear_velocity(self, v):
        pass

    def set_angular_velocity(self, v):
        pass


class FakeRobot:
    def __init__(self, tcp):
        self.tcp = np.array(tcp, dtype=float)
        self.offset = np.zeros(3)
        self.end_effector = self

    def get_world_pose(self):
        return self.tcp + self.offset - M.TCP_OFFSET, np.array([1.0, 0.0, 0.0, 0.0])


class FakeLink:
    def __init__(self):
        self.inbox, self.states, self.results, self.warnings = [], [], [], []

    def poll(self):
        codes, self.inbox = self.inbox, []
        return codes

    def publish_state(self, state):
        if not self.states or self.states[-1] != state:
            self.states.append(state)

    def publish_result(self, result):
        self.results.append(result)

    def log(self, _):
        pass

    def warn(self, text):
        self.warnings.append(text)


class Harness:
    def __init__(self, seed=0, detector=None, solved=None, recorder=None):
        cubes, self.parks = {}, {}
        index = 0
        for color in M.VALID_MISSION_COLORS:
            cubes[color] = []
            for _ in range(M.CUBES_PER_COLOR):
                park = np.array([M.PARK_X, M.PARK_Y0 + index * M.PARK_STEP, M.CUBE_SIZE / 2])
                cube = FakeCube(park)
                cubes[color].append(cube)
                self.parks[id(cube)] = park
                index += 1

        self.pool = M.CubePool(cubes, self.parks, M.MARKER_XY)
        self.robot = FakeRobot(HOME)
        self.fsm = M.ColorPickPlaceFSM(self.robot)
        self.gate = M.ColorGate(log=lambda _: None)
        self.link = FakeLink()
        self.loop = M.CubeLoop(self.pool, self.fsm, self.gate, self.link,
                               random.Random(seed), recorder, log=lambda _: None)
        self.detector = detector or (lambda h: h.truth())
        self.solved = solved or (lambda h: True)

        self.spawns = []      # (color, position)
        self.starts = []      # (detected, release_xy, spawned, mission_no)

        spawn = self.pool.spawn

        def record_spawn(color, xy):
            cube = spawn(color, xy)
            self.spawns.append((color, cube.pos.copy()))
            return cube
        self.pool.spawn = record_spawn

        start = self.fsm.start_mission

        def record_start(color, pick_xy, place_xy, home_tcp):
            ok = start(color, pick_xy, place_xy, home_tcp)
            if ok:
                release = dict(self.fsm.plan)["RELEASE"][:2].copy()
                self.starts.append((color, release, self.loop.spawned_color, self.loop.mission_no))
            return ok
        self.fsm.start_mission = record_start

    def truth(self):
        """카메라가 보는 색 — 스폰된 큐브가 그리퍼 아래 있을 때만"""
        cube = self.loop.current_cube
        if cube is None or self.pool.state_of(cube) != "active":
            return 0
        if np.linalg.norm(cube.pos[:2] - HOME[:2]) > 0.2:
            return 0
        return self.pool.color_of(cube)

    def step(self):
        code = self.detector(self)
        if code is not None:
            self.link.inbox.append(code)

        self.loop.update(HOME)

        target = self.fsm.current_target()
        if target is not None:
            self.robot.tcp = np.array(target, dtype=float)

        cube = self.loop.current_cube
        if cube is not None and self.fsm.gripper == "close":
            cube.pos = np.array([self.robot.tcp[0], self.robot.tcp[1], M.CUBE_SIZE / 2])

        event = self.fsm.advance(self.solved(self))
        self.loop.on_event(event)
        self.link.publish_state(self.loop.state_name)
        return event

    def run_until(self, n_results, max_steps=400000):
        for _ in range(max_steps):
            self.step()
            if len(self.link.results) >= n_results or self.loop.phase == "HALTED":
                return
        raise AssertionError(f"only {len(self.link.results)} results")


def near_slot(xy, color):
    marker = M.MARKER_XY[color]
    return (abs(xy[1] - marker[1]) < 1e-9 and
            min(abs(xy[0] - (marker[0] + o)) for o in M.MARKER_SLOT_OFFSETS_X) < 1e-9)


# ══════════════════════════════════════════════════════════════
def test_markers_outside_home_camera_view():
    """USD 기준 바닥 시야 x -0.021~0.727, y -0.325~0.427 밖에 마커 전체가 있어야 한다"""
    half = M.MARKER_SIZE[1] / 2
    assert M.BLUE_MARKER_XY[1] + half < -0.325
    assert M.GREEN_MARKER_XY[1] - half > 0.427
    assert M.PARK_X + M.CUBE_SIZE / 2 < -0.021


def test_idle_until_home_known():
    h = Harness()
    for _ in range(1000):
        h.loop.update(None)
        assert h.fsm.current_target() is None
    assert h.spawns == [] and h.loop.state_name == "INIT"


def test_spawn_under_gripper_within_range():
    h = Harness(seed=1)
    h.run_until(10)
    assert len(h.spawns) >= 10
    for _, pos in h.spawns:
        assert np.all(np.abs(pos[:2] - HOME[:2]) <= M.SPAWN_HALF_RANGE + 1e-9)
        assert pos[2] == pytest.approx(M.CUBE_SIZE / 2 + M.SPAWN_DROP)


def test_both_colors_spawn_randomly():
    h = Harness(seed=3)
    h.run_until(20)
    colors = [c for c, _ in h.spawns[:20]]
    assert set(colors) == {M.COLOR_BLUE, M.COLOR_GREEN}


def test_each_cube_placed_on_matching_marker():
    h = Harness(seed=5)
    h.run_until(10)
    assert all(r.startswith("SUCCESS") for r in h.link.results[:10])
    for detected, release_xy, spawned, _ in h.starts[:10]:
        assert detected == spawned
        assert near_slot(release_xy, spawned)
    assert h.loop.stats["MISMATCH"] == 0 and h.loop.stats["ERROR"] == 0


def test_one_mission_per_cube_despite_repeated_messages():
    h = Harness(seed=7)
    h.run_until(6)
    nos = [no for *_, no in h.starts]
    assert len(nos) == len(set(nos))


def test_stale_color_right_after_spawn_ignored():
    def detector(h):
        if h.loop.phase == "SETTLE":
            return OPPOSITE[h.loop.spawned_color]     # 이전 큐브 색이 늦게 도착
        return h.truth()

    h = Harness(seed=9, detector=detector)
    h.run_until(8)
    assert h.loop.stats["MISMATCH"] == 0 and h.loop.stats["SUCCESS"] >= 8


@pytest.mark.parametrize("code", [0, 3, -1, 99])
def test_robot_does_not_move_for_zero_or_invalid(code):
    h = Harness(detector=lambda h: code)
    for _ in range(3000):
        h.step()
        assert h.fsm.current_target() is None
    assert h.starts == [] and h.loop.state_name == "WAIT_COLOR"


def test_no_color_message_keeps_waiting():
    h = Harness(detector=lambda h: None)
    for _ in range(3000):
        h.step()
    assert h.starts == [] and h.loop.state_name == "WAIT_COLOR" and len(h.spawns) == 1


def test_mismatch_follows_detected_color_and_is_recorded(tmp_path):
    def detector(h):
        seen = h.truth()
        return OPPOSITE.get(seen, seen)

    recorder = M.Recorder(tmp_path / "run.csv")
    h = Harness(seed=11, detector=detector, recorder=recorder)
    h.run_until(4)
    recorder.close()

    for detected, release_xy, spawned, _ in h.starts[:4]:
        assert detected != spawned
        assert near_slot(release_xy, detected)

    with open(tmp_path / "run.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["result"] for r in rows[:4]] == ["MISMATCH"] * 4
    assert rows[0]["spawned"] != rows[0]["detected"]


def test_error_recovers_then_next_cube_spawns():
    def solved(h):
        return not (h.loop.mission_no == 1 and h.fsm.mode == "MISSION")   # 첫 미션만 IK 실패

    h = Harness(seed=13, solved=solved)
    h.run_until(3)
    assert h.link.results[0].startswith("ERROR #1")
    assert "RECOVERED" in h.link.states
    assert h.loop.stats["ERROR"] == 1
    assert h.link.results[1].startswith("SUCCESS #2")
    assert h.fsm.gripper == "open"


def test_halted_stops_spawning():
    h = Harness(seed=15)
    h.robot.offset = np.array([0.0, 0.0, 0.05])      # 목표에 절대 도착하지 못함
    h.run_until(10 ** 9)
    assert h.loop.phase == "HALTED"

    spawned = len(h.spawns)
    for _ in range(3000):
        h.step()
    assert len(h.spawns) == spawned
    assert h.fsm.current_target() is None


def test_many_missions_recycle_cubes_and_slots(tmp_path):
    recorder = M.Recorder(tmp_path / "run.csv")
    h = Harness(seed=17, recorder=recorder)
    h.run_until(14)
    recorder.close()

    assert h.loop.stats["SUCCESS"] >= 14
    states = [h.pool.state_of(c) for g in h.pool._cubes.values() for c in g]
    assert states.count("active") <= 1
    assert states.count("placed") <= 2 * len(M.MARKER_SLOT_OFFSETS_X)

    with open(tmp_path / "run.csv") as f:
        assert len(list(csv.DictReader(f))) >= 14


def test_reset_parks_everything():
    h = Harness(seed=19)
    h.run_until(3)
    h.fsm.reset()
    h.gate.reset()
    h.loop.reset()

    assert h.loop.state_name == "INIT"
    for group in h.pool._cubes.values():
        for cube in group:
            assert h.pool.state_of(cube) == "parked"
            assert np.allclose(cube.pos, h.parks[id(cube)])

    h.run_until(len(h.link.results) + 2)
    assert h.link.results[-1].startswith("SUCCESS #2")


def test_state_sequence_for_one_mission():
    h = Harness(seed=21)
    h.run_until(1)
    expected = ["SPAWN", "WAIT_COLOR", "APPROACH", "DESCEND", "GRASP", "LIFT", "MOVE",
                "LOWER", "RELEASE", "RETREAT", "RETURN_HOME", "DONE"]
    assert h.link.states[:len(expected)] == expected
