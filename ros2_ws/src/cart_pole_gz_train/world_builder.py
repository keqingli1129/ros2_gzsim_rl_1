import os
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
XACRO_REL_PATH = "ros2_ws/src/robot_description/robot/cart_pole.urdf.xacro"


def _run_in_ros_env(cmd: str) -> str:
    """Run a shell command with a colcon-built ros2_ws sourced and the venv
    stripped from PATH. Required because xacro (ament_index_python-based)
    fails under the venv's python3 and can't resolve the robot_description
    package unless ros2_ws/install/setup.bash has been sourced."""
    script = (
        'PATH=$(echo "$PATH" | tr ":" "\\n" | grep -v "\\.venv" | paste -sd:); '
        'unset VIRTUAL_ENV; '
        f'source {REPO_ROOT}/ros2_ws/install/setup.bash; '
        + cmd
    )
    result = subprocess.run(["bash", "-c", script], cwd=REPO_ROOT,
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd!r}\n{result.stderr}")
    return result.stdout


def run_xacro() -> str:
    return _run_in_ros_env(f"xacro {XACRO_REL_PATH}")


def convert_urdf_to_sdf(urdf_text: str, scratch_dir: str) -> str:
    tmp_urdf = os.path.join(scratch_dir, "_generated.urdf")
    with open(tmp_urdf, "w") as f:
        f.write(urdf_text)
    return _run_in_ros_env(f"gz sdf -p {tmp_urdf}")
