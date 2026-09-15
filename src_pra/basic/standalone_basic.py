# from isaacsim import SimulationApp
# simulation_app = SimulationApp({"headless": False})     # 1. Application

# import numpy as np
# import time
# import omni.usd
# from isaacsim.core.api import World
# from isaacsim.core.api.objects import DynamicCuboid

# world = World(stage_units_in_meters=1.0)                # 2. World
# stage = omni.usd.get_context().get_stage()              # 3. Stage

# # cube_prim1 = DynamicCuboid(                              # 4. Prim
# #     prim_path="/World/BlueCube",
# #     name="blue_cube",
# #     position=np.array([1.0, 0.0, 1.0]),
# #     scale=np.array([0.15, 0.15, 0.15]),
# #     color=np.array([0.0, 0.0, 1.0]),
# # )

# cube_prim2 = DynamicCuboid(                              # 4. Prim
#     prim_path="/World/RedCube",
#     name="red_cube",
#     position=np.array([0.0, 0.0, 1.0]),
#     scale=np.array([0.3, 0.3, 0.3]),
#     color=np.array([1.0, 0.0, 0.0]),
# )

# cube_prim3 = DynamicCuboid(                              # 4. Prim
#     prim_path="/World/GreenCube",
#     name="green_cube",
#     position=np.array([0.0, 0.0, 1.3]),
#     scale=np.array([0.1, 0.1, 0.1]),
#     color=np.array([0.0, 1.0, 0.0]),
# )
# world.scene.add_default_ground_plane()                  # 5. Scene
# step_count = 0

# while simulation_app.is_running() and step_count < 500:
#     world.step(render=True)
#     time.sleep(0.01)
#     step_count += 1
#     print(f"Step: {step_count}")
#     # if step_count >= 500:
#     #     step_count = 0

# #world.scene.add(cube_prim1)
# world.scene.add(cube_prim2)
# world.scene.add(cube_prim3)

# world.reset()

# # while simulation_app.is_running():                      # 6. Simulation
# #     world.step(render=True)

# simulation_app.close()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})     # 1. Application

import time
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)                # 2. World
stage = omni.usd.get_context().get_stage()              # 3. Stage

# 미션 사진과 동일하게 빨간 큐브 하나만 생성합니다.
cube_prim = DynamicCuboid(                              # 4. Prim
    prim_path="/World/RedCube",
    name="red_cube",
    position=np.array([0.0, 0.0, 0.15]),  # 크기가 0.3이므로 중심을 0.15로 두어 바닥에 둡니다.
    scale=np.array([0.3, 0.3, 0.3]),
    color=np.array([1.0, 0.0, 0.0]),
)

world.scene.add_default_ground_plane()                  # 5. Scene
world.scene.add(cube_prim)                              # 큐브를 시작 전에 씬에 추가합니다.

world.reset()

step_count = 0
is_playing_prev = False  # 이전 프레임의 Play 상태 저장용 변수

while simulation_app.is_running():                      # 6. Simulation
    world.step(render=True)
    
    # 현재 시뮬레이션이 Play 상태인지 확인
    is_playing_curr = world.is_playing()
    
    # [조건 2] Stop -> Play 전환 감지 및 리셋
    if is_playing_curr and not is_playing_prev:
        step_count = 0
        print("[리셋] Play 시작 -> step_count = 0")
        
    # Play 상태일 때만 스텝을 세고 제어 로직을 실행합니다.
    if is_playing_curr:
        step_count += 1
        
        # 터미널 출력 (이미지 참고)
        if step_count % 100 == 0:
            print(f"step: {step_count}")
            
        # [조건 1] 300 스텝 후 큐브 1m 높이로 순간이동
        if step_count == 300:
            current_position, current_orientation = cube_prim.get_world_pose()
            current_position[2] = 1.0  # z축(높이)을 1.0m로 변경
            cube_prim.set_world_pose(position=current_position, orientation=current_orientation)

        elif step_count == 500:
            print("시뮬레이션 종료")
            time.sleep(1)
            simulation_app.close()
            
    # 다음 루프를 위해 현재 상태를 이전 상태로 저장
    is_playing_prev = is_playing_curr

simulation_app.close()