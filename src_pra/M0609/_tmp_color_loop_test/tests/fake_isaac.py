"""Isaac Sim 모듈을 가짜로 채워 pick_place_color_loop.py 를 import 한다"""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy  # noqa: F401  가짜 모듈 범위 밖에서 먼저 불러 재로드 경고를 막는다

SCRIPT = Path(__file__).resolve().parents[1] / "pick_place_color_loop.py"

FAKE_MODULES = [
    "isaacsim", "isaacsim.core", "isaacsim.core.api", "isaacsim.core.api.tasks",
    "isaacsim.core.api.objects",
    "isaacsim.core.utils", "isaacsim.core.utils.extensions",
    "isaacsim.robot", "isaacsim.robot.manipulators",
    "isaacsim.robot.manipulators.grippers", "isaacsim.robot.manipulators.manipulators",
    "isaacsim.robot_motion", "isaacsim.robot_motion.motion_generation",
    "omni", "omni.usd", "pxr",
]


def load_script(name="color_loop"):
    fake = {n: mock.MagicMock(name=n) for n in FAKE_MODULES}
    fake["isaacsim.core.api.tasks"].BaseTask = type("BaseTask", (), {})

    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, fake):
        spec.loader.exec_module(module)
    return module
