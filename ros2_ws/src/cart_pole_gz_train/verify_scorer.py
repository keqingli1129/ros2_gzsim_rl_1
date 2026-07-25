import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from gz_scorer import GzCartPoleScorer

scorer = GzCartPoleScorer()
obs, _info = scorer.reset()
assert obs.shape == (4,)

for _ in range(20):
    obs, reward, terminated, truncated, _info = scorer.step(1)
    assert not terminated, "should not fall over in 20 steps of 5ms each"

assert obs[1] > 0.0, f"cart should have positive velocity after pushing right, got {obs[1]}"
scorer.close()
print(f"PASS: after 20 steps of action=1, cart_vel={obs[1]:.3f} (positive as expected)")
