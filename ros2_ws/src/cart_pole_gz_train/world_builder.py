import os
import subprocess
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
XACRO_REL_PATH = "ros2_ws/src/robot_description/robot/cart_pole.urdf.xacro"

_PRIMITIVE_GEOMETRY = {
    "base_footprint": ("box", "0.4 0.4 0.6"),
    "cart_link": ("box", "0.3 0.3 0.15"),
    "pole_link": ("cylinder", None),  # handled specially below
}


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


def postprocess_model_sdf(model_sdf_text: str) -> str:
    """Strip visuals and replace mesh collision with primitives sized to
    roughly match the real robot_description meshes, and drop tip_link
    (mass 0.0001, physically negligible mount point) - keeps the physics
    parameters (mass/inertia/joint limits) sourced live from the xacro
    while collision shape stays hand-simplified, since headless training
    never renders and shouldn't pay for mesh-based collision."""
    root = ET.fromstring(model_sdf_text)
    model = root if root.tag == "model" else root.find("model")

    for link in list(model.findall("link")):
        name = link.get("name")
        for visual in link.findall("visual"):
            link.remove(visual)
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh")
            if mesh is None:
                continue
            geometry.remove(mesh)
            if name == "pole_link":
                cylinder = ET.SubElement(geometry, "cylinder")
                ET.SubElement(cylinder, "radius").text = "0.02"
                ET.SubElement(cylinder, "length").text = "1.0"
            else:
                shape, size = _PRIMITIVE_GEOMETRY.get(name, ("box", "0.2 0.2 0.2"))
                box = ET.SubElement(geometry, shape)
                ET.SubElement(box, "size").text = size

    for joint in list(model.findall("joint")):
        if joint.get("name") == "tip_joint":
            model.remove(joint)
    for link in list(model.findall("link")):
        if link.get("name") == "tip_link":
            model.remove(link)

    return ET.tostring(model, encoding="unicode")


def wrap_in_world(model_sdf_text: str) -> str:
    """Wrap the processed <model> in a full <world> TestFixture can load.

    The model gets an initial pose lifting it to z=2, clear of the ground
    plane - verified necessary: at z=0 the primitive collision boxes
    interpenetrate the ground plane on load, and the resulting contact
    forces dominate cart_joint's tiny prismatic motion, making commanded
    force look like it has no effect even though it's being applied
    correctly (confirmed by lifting the model and re-testing: a 1,000,000N
    force went from producing ~0.02 m/s after 200ms to producing the
    expected effort-limit-clamped ~11 m/s^2 acceleration).
    """
    model_sdf_text = model_sdf_text.replace(
        '<model name="cart_pole">',
        '<model name="cart_pole"><pose>0 0 2 0 0 0</pose>',
    )
    return f"""<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="cart_pole_train">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
      </link>
    </model>
    {model_sdf_text}
  </world>
</sdf>
"""


def generate_training_world(output_path: str) -> str:
    scratch_dir = os.path.dirname(output_path)
    urdf = run_xacro()
    model_sdf = convert_urdf_to_sdf(urdf, scratch_dir)
    processed = postprocess_model_sdf(model_sdf)
    world = wrap_in_world(processed)
    with open(output_path, "w") as f:
        f.write(world)
    return world
