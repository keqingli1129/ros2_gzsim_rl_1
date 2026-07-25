import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from world_builder import run_xacro, convert_urdf_to_sdf

scratch_dir = os.path.dirname(__file__)
urdf = run_xacro()
assert "<robot" in urdf, "xacro did not produce a URDF"
model_sdf = convert_urdf_to_sdf(urdf, scratch_dir)
assert "cart_joint" in model_sdf and "pole_joint" in model_sdf, \
    "converted SDF is missing expected joints"
os.remove(os.path.join(scratch_dir, "_generated.urdf"))
print("PASS: xacro -> SDF conversion produced cart_joint and pole_joint")
