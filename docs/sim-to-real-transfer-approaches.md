# Bridging cart_pole's PPO into ros2_ws's commander: approaches considered

Context: `cart_pole/cart_pole_env.py` trains a PPO policy (Stable-Baselines3) directly
against `gz.sim8` bindings, headless and fast, with no ROS involved. `ros2_ws`'s
`commander` package trains a from-scratch DQN against the same conceptual robot,
but over ROS 2 topics/services (`ros_gz_bridge`) talking to a live `gz sim` world
(`robomaster_rale`). The question explored in this discussion: can/should the
already-trained `cart_pole` policy be used as a starting point for training the
`commander` side, instead of starting `commander` from scratch?

This file summarizes the approaches discussed, in the order they came up, and
which one was settled on.

## Why this isn't a simple copy-paste

- **Different algorithms in the original design.** `cart_pole` uses SB3 PPO
  (actor-critic network, outputs an action distribution + value estimate).
  `commander`'s original DQN (`dqn_learning.py`) uses a hand-rolled `QNet`
  (plain regression network, outputs one Q-value per action). These are not
  weight-compatible — you cannot load PPO weights into a DQN's Q-network.
- **Different action spaces.** `cart_pole` uses `Discrete(2)` (bang-bang
  ±2000N force). `commander`'s DQN used `Discrete(10)`, mapped to a continuous
  force via `force = action * 16/9 - 8`.
- **Different observations.** `cart_pole` reads the pole's real pitch directly
  from the physics engine. `commander` hand-integrates an angle estimate from
  angular velocity (`yaw_angle += y_angular * time_interval`), even though the
  real joint position is already present (and unused) in the `/joint_states`
  message it subscribes to.
- **Different physical plants.** `cart_pole.sdf`'s `vehicle_green` model and
  `robot_description`'s URDF-based robot (spawned as `cart_pole` in world
  `robomaster_rale`) are different SDF/URDF definitions with likely different
  mass/friction — so even with matching architectures, transferred weights are
  being adapted to a new plant, not a free win.

## Approaches discussed

1. **Imitation / behavior-cloning pretraining (teacher-student, informal).**
   Roll out the trained PPO policy as a scripted "teacher," record
   `(state, action)` pairs, and pretrain the DQN's `QNet` via supervised
   learning on those pairs before continuing normal DQN training. Chosen
   first as the general direction, but immediately ran into the
   `Discrete(2)` vs `Discrete(10)` action-space mismatch (would need
   approximate mapping, e.g. teacher action 0/1 → DQN's extreme action
   indices 0/9).

2. **Reconfigure `cart_pole`'s PPO to use `Discrete(10)`, retrain, then
   imitate.** Removes the action-space mismatch — the teacher's chosen
   action directly becomes the imitation label, no approximate mapping.
   Still requires behavior cloning rather than weight copying, since PPO's
   actor-critic network and the DQN's Q-network remain structurally
   different.

3. **Replace the DQN with SB3 PPO on the `commander` side too (chosen
   direction).** Since the DQN was originally just a learning exercise and
   SB3 is the more mature/stable implementation already used successfully in
   `cart_pole`, the DQN gets superseded rather than kept. With the *same*
   algorithm and network architecture on both sides, this becomes genuine
   weight-reuse transfer learning: `PPO.load("cart_pole_ppo.zip")`, point it
   at a new Gym-compatible environment wrapping the ROS/`gz-sim` world, and
   continue `.learn()` — no distillation or imitation step needed.

   This requires two new pieces of work:
   - A new Gymnasium `Env` wrapping the ROS/gz-sim world (subscribing
     `/joint_states`, publishing to `/cart_controller/command`, calling
     `/world/robomaster_rale/control` for reset) with the *same*
     `Discrete(2)` action space and 4-dim `Box` observation space
     `cart_pole_env.py` uses, so the pretrained weights are meaningful.
   - Reading the pole's real position directly from `/joint_states`
     (currently received but discarded) instead of the hand-integrated
     estimate, since the pretrained network expects the real signal it was
     trained on.

## The environment-mixing problem this creates

Running PPO training against the ROS/gz-sim world needs both `rclpy`/ROS
message packages (system dist-packages, used by `colcon build`/`ros2 run`)
**and** `stable-baselines3`/`torch`/`gymnasium` (only in the project's
uv-managed venv) in the same process — something neither `cart_pole/` (needs
gz bindings + SB3, no ROS) nor the original `commander` (needs only `rclpy`)
required simultaneously before. Two ways to resolve it, discussed but not
yet decided:

1. Run the training script standalone via `uv run`, with `PYTHONPATH`
   extended to the ROS install (mirrors how `cart_pole/` already bridges
   `uv run` + system gz bindings). No `ros2 run`/colcon entry point involved.
2. Install SB3/torch/gymnasium into system dist-packages so a normal
   `ros2 run commander train_ppo` console-script entry point works. Keeps
   the ROS launch convention, at the cost of duplicating large dependencies
   outside `uv`'s lockfile management.

## Industry context for this kind of transfer (for reference)

The overall shape of "pretrain fast/cheap, then fine-tune closer to the
real target" is standard practice in robotics RL, generally called
**sim-to-real transfer** or **progressive-fidelity training**:

- **Domain randomization + fine-tune** — train across randomized physics
  parameters (mass, friction, latency, sensor noise) in a fast simulator so
  the policy is robust to plant mismatch, then deploy/fine-tune on the real
  target (e.g. OpenAI's Dactyl, legged-robot locomotion work from
  ETH/ANYbotics).
- **Two-simulator pipeline (closest match to our plan)** — pretrain in a
  cheap/fast physics backend (e.g. NVIDIA Isaac Gym, GPU-parallel, no ROS),
  then continue training in a slower but more representative one (e.g.
  Gazebo with the real ROS stack) before real hardware. Our
  `gz.sim8`-fast-pretrain → ROS/`ros_gz_bridge`-fine-tune plan maps directly
  onto this pattern, just without domain randomization (a possible future
  addition if the fine-tuning gap turns out large).
- **Teacher-student distillation** — train a teacher with *privileged*
  simulation-only information (true friction, full contact state, etc.),
  then distill a differently-architected student that only sees realistic
  sensor observations, via supervised learning on the teacher's action
  choices (examples: ANYmal locomotion (Lee et al. 2020), RMA - Rapid Motor
  Adaptation (Kumar et al. 2021), "Learning by Cheating" for autonomous
  driving). Considered and set aside for this project: there's no real
  privileged-information gap between `cart_pole`'s `gz.sim8` training and
  `commander`'s ROS `/joint_states` path — both read the same kind of
  simulated joint data — so weight-reuse fine-tuning (same architecture,
  continued training) fits better than distillation (different architecture,
  supervised imitation).
- **System identification first** — some teams calibrate the fast
  simulator's physics parameters to match the target plant *before*
  pretraining, shrinking the eventual fine-tuning gap. Not yet done here;
  `cart_pole.sdf` and `robot_description`'s URDF likely have unmatched
  mass/friction values.

## Status

Direction settled: replace `commander`'s DQN with an SB3 PPO training script,
reusing `cart_pole`'s pretrained weights via `PPO.load(...)` + continued
`.learn()` against a new ROS-wrapped Gym environment. Not yet decided: the
`uv run` + `PYTHONPATH` vs. system-dist-packages-install question for running
the new environment/training script. No implementation started yet — this
is still at the design/brainstorming stage.
