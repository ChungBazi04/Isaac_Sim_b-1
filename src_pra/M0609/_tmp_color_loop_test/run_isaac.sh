#!/usr/bin/env bash
# [TEST] 시스템 ROS 환경을 걷어내고 Isaac Sim python 으로 pick_place_color_loop.py 를 실행한다
#
#   ./run_isaac.sh                          감지 노드(PC B)와 함께
#   ./run_isaac.sh --self-color             감지 노드 없이 스폰한 색으로 루프
#   ./run_isaac.sh --no-ros --self-color    ROS 없이 씬·로봇·루프만
#
# Isaac Sim 위치를 못 찾으면  ISAAC_SIM_PATH=/path/to/isaacsim ./run_isaac.sh
#
# 왜 필요한가
#   ~/.bashrc 가 /opt/ros/jazzy/setup.bash 를 자동으로 source 하면
#   시스템 rclpy(Python 3.12)가 Isaac Sim(Python 3.11)에 섞여 시작하자마자 죽는다.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Isaac Sim 위치 ────────────────────────────────────────
if [ -z "${ISAAC_SIM_PATH:-}" ]; then
    for d in "$HOME/isaacsim" "$HOME/isaac-sim" "$HOME"/isaacsim* "$HOME"/isaac-sim-* \
             "$HOME"/.local/share/ov/pkg/isaac-sim-* "$HOME"/.local/share/ov/pkg/isaac_sim-*; do
        if [ -x "$d/python.sh" ]; then
            ISAAC_SIM_PATH="$d"
            break
        fi
    done
fi
if [ -z "${ISAAC_SIM_PATH:-}" ]; then
    # isaac_python alias 에 적힌 python.sh 경로를 쓴다
    alias_def="$(bash -ic 'alias isaac_python' 2>/dev/null || true)"
    py_sh="$(printf '%s' "$alias_def" | grep -oE "[^' =\"]+/python\.sh" | head -1 || true)"
    py_sh="${py_sh/#\~/$HOME}"
    py_sh="${py_sh/#\$HOME/$HOME}"
    if [ -n "$py_sh" ] && [ -x "$py_sh" ]; then
        ISAAC_SIM_PATH="$(dirname "$py_sh")"
    fi
fi
if [ -z "${ISAAC_SIM_PATH:-}" ] || [ ! -x "$ISAAC_SIM_PATH/python.sh" ]; then
    echo "Isaac Sim python.sh 를 찾지 못했다. ISAAC_SIM_PATH=/path/to/isaacsim $0 $*" >&2
    exit 1
fi

# ── 시스템 ROS 흔적 제거 ─────────────────────────────────
strip_ros() {
    printf '%s' "$1" | tr ':' '\n' | grep -v -e '^/opt/ros' -e '^$' | paste -sd: - || true
}
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH \
      ROS_PACKAGE_PATH ROS_PYTHON_VERSION ROS_VERSION ROS_LOCALHOST_ONLY CYCLONEDDS_URI
export PATH="$(strip_ros "$PATH")"
export LD_LIBRARY_PATH="$(strip_ros "${LD_LIBRARY_PATH:-}")"

# ── Isaac Sim 내장 ROS 2 Jazzy ───────────────────────────
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-135}"
if [ -f "$HOME/.ros/fastdds_whitelist.xml" ]; then
    export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/.ros/fastdds_whitelist.xml"
fi
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ISAAC_SIM_PATH/exts/isaacsim.ros2.bridge/jazzy/lib"

echo "isaac sim   $ISAAC_SIM_PATH"
echo "domain      $ROS_DOMAIN_ID  rmw $RMW_IMPLEMENTATION"
echo "fastdds xml ${FASTRTPS_DEFAULT_PROFILES_FILE:-(none)}"
echo "args        $*"

cd "$HERE"
exec "$ISAAC_SIM_PATH/python.sh" pick_place_color_loop.py "$@"
