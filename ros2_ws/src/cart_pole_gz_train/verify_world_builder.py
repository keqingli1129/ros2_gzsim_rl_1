import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from world_builder import generate_training_world

scratch_dir = os.path.dirname(__file__)
output_path = os.path.join(scratch_dir, "cart_pole_train.sdf")
world_text = generate_training_world(output_path)

assert "<mesh>" not in world_text, "primitive replacement left a mesh behind"
assert "tip_link" not in world_text, "tip_link should have been dropped"
assert 'cart_joint' in world_text and 'pole_joint' in world_text
assert '<world name="cart_pole_train">' in world_text
assert '<model name="cart_pole"><pose>0 0 2 0 0 0</pose>' in world_text, "pose-lift fix must be present in cart_pole model"
print("PASS: generated world has no meshes, no tip_link, correct joints, and pose-lift fix")
