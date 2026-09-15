# from import 하는 라이브러리는 항상 위로 올라가지만
# 이 파일은 python 형식과는 조금 위배되는 구조를 가지고 있음
# LLM이 이 부분을 갑자기 갈아엎을 수도 있으니 조심!!!!!

# 여기까지는 수업 내용상 구조는 동일하게 유지
# -------------------------------

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False}) # headless 모드 비활성화

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge") # bridge extension 활성화
simulation_app.update()

from pathlib import Path
import time
import omni.usd
from pxr import Usd, UsdGeom

USD_PATH = str(Path(__file__).resolve().parent / "Collected_camera_cube/m0609_camera.usd")

# /World prim 명시적 생성 후 USD reference 연결
stage = omni.usd.get_context().get_stage()
UsdGeom.Xform.Define(stage, "/World")
world_prim = stage.GetPrimAtPath("/World")
world_prim.GetReferences().AddReference(USD_PATH)

# -------------------------------
# 여기까지는 수업 내용상 구조는 동일하게 유지

for _ in range(15):
    simulation_app.update()

# # 로드된 prim 구조 출력
# print("\n" + "=" * 60)
# print("Stage prim 구조")
# print("=" * 60)
# for prim in Usd.PrimRange(stage.GetPseudoRoot()):
#     depth = len(str(prim.GetPath()).split("/")) - 2
#     indent = "  " * depth
#     print(f"{indent}{prim.GetName()}  [{prim.GetTypeName()}]")

print("\n시뮬레이션 실행 중 (Play 버튼을 눌러 확인하세요)")

while simulation_app.is_running():
    simulation_app.update()
    time.sleep(0.016)

simulation_app.close()
