import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".05"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# @title Import packages for plotting and creating graphics
import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np
import onnxruntime as rt

# Graphics and plotting.
# print("Installing mediapy:")
# !command -v ffmpeg >/dev/null || (apt update && apt install -y ffmpeg)
# !pip install -q mediapy
import mediapy as media
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)
# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp

  #@title Import The Playground

from mujoco_playground import wrapper
from mujoco_playground import registry
from etils import epath
from mujoco_playground._src import mjx_env
import mujoco.viewer
import csv

from mujoco_playground.config import locomotion_params
env_name = 'HunterJoystick'
# env_name = 'HunterJoystickRoughTerrain'
_PLAY_ONLY = True
# _LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250913-131738/checkpoints"
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250913-151258/checkpoints"

#3rd Reduced obs
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250919-005307/checkpoints" #200k steps
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250919-022259/checkpoints" # 1mils200k steps
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250919-074629/checkpoints"
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250919-091701/checkpoints"
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250919-105650/checkpoints"
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250919-140651/checkpoints"
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250919-154753/checkpoints" ## Current best

#Fix kp kd
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250928-143848/checkpoints"
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250928-164017/checkpoints"
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250929-030300/checkpoints"

#Fix forceRange
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250929-091157/checkpoints"
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250929-133124/checkpoints"

#Add noises
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250930-041904/checkpoints"
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20250930-150416/checkpoints"
_LOAD_CHECKPOINT_PATH = "/content/drive/MyDrive/HACHIX-project/mujoco_playground/logs/HunterJoystick-20251001-020621/checkpoints"

_LOAD_CHECKPOINT_PATH = "/home/sandbox/Work/mujoco_playground/learning/notebooks/logs/HunterJoystickRoughTerrain-20251015-150843/checkpoints"
_CSV_PATH = "/home/sandbox/Work/mujoco_playground/learning/notebooks/logs/HunterJoystickRoughTerrain-20251015-150843"

episode = 30
push_step = 150  # When to apply the push
push_magnitude = -20.0  # Newtons
push_force_x = jp.array([push_magnitude, 0.0, 0.0])  # Newtons in -X direction
push_force_y = jp.array([0.0, push_magnitude, 0.0])  # Newtons in +Y direction
push_duration = 10  # How many steps the push lasts
command = jp.array([0.0, 0.0, 0.0])  # Initial command
gait_freq = 1.75 
foot_height = 0.16
phase = jp.array([0, jp.pi])
gait = 0  # 0: walk, 1: stand
if gait == 1:
   phase = jp.array([0.0, 0.0])
name = _LOAD_CHECKPOINT_PATH.split("/")[-2]
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
ppo_params = locomotion_params.brax_ppo_config(env_name)

## For inference
if _PLAY_ONLY:
    ppo_params["num_timesteps"] = 0
else:
    ppo_params["num_timesteps"]= 100000000
    # ppo_params["num_timesteps"]= 20000000

x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
#   keys = sorted(metrics.keys())
# #   raise ValueError(keys)

#   row = {"step": int(num_steps)}
#   row.update({k: metrics[k] for k in keys})

#   file_exists = os.path.exists(_CSV_PATH)
#   with open(_CSV_PATH, "a", newline="") as f:
#     writer = csv.DictWriter(f, fieldnames=["step"] + keys)
#     if not file_exists:
#        writer.writeheader()
#     writer.writerow(row)

  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

#   display(plt.gcf())

randomizer = registry.get_domain_randomizer(env_name)
ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

# Handle checkpoint loading
if _LOAD_CHECKPOINT_PATH is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

if not _PLAY_ONLY:
    # Generate unique experiment name
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{env_name}-{timestamp}"
    print(f"Experiment name: {exp_name}")

    # Set up logging directory
    logdir = epath.Path("/home/sandbox/Work/mujoco_playground/learning/notebooks/checkpoints/").resolve() / exp_name
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"Logs are being stored in: {logdir}")

    # Set up checkpoint directory
    ckpt_path = logdir / "checkpoints"
    ckpt_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint path: {ckpt_path}")

    with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
      json.dump(env_cfg.to_dict(), fp, indent=4)
else:
    ckpt_path = None

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    save_checkpoint_path=ckpt_path,
    restore_checkpoint_path=restore_checkpoint_path,
    randomization_fn=randomizer,
    progress_fn=progress
)
env_cfg.push_config.enable = False
make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=registry.load(env_name, config=env_cfg),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
if len(times) > 1:
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

#@title Rollout and Render
from mujoco_playground._src.gait import draw_joystick_command
import mujoco

env = registry.load(env_name)
eval_env = registry.load(env_name)
jit_reset = jax.jit(eval_env.reset)
jit_step = jax.jit(eval_env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

phase_dt = 2 * jp.pi * eval_env.dt * gait_freq
# phase = jp.array([0, 0.0])

output_path = "/home/sandbox/Work/mujoco_playground/learning/notebooks/onnx/stand_202510151615.onnx"
# Run inference with ONNX Runtime
output_names = ['continuous_actions']
providers = ['CPUExecutionProvider']
m = rt.InferenceSession(output_path, providers=providers)

log_push = []

body_id = eval_env.mj_model.body("base_link").id

def draw_force_arrow(scene: mujoco.MjvScene, data: mujoco.MjData):
    body_pos = data.xpos[body_id]
    applied_force = np.array(data.xfrc_applied[body_id][:3])

    force_magnitude = np.linalg.norm(applied_force)
    if force_magnitude < 1e-3:
        return

    # Visualization scale: adjust this value to your preference
    visualization_scale = 0.01
    arrow_length = force_magnitude * visualization_scale

    # Direction of the force
    arrow_dir = applied_force / force_magnitude

    # Orientation matrix (Z axis → arrow_dir)
    z_axis = np.array([0, 0, 1])
    if np.allclose(arrow_dir, z_axis) or np.allclose(arrow_dir, -z_axis):
        rot_mat = np.eye(3)
    else:
        v = np.cross(z_axis, arrow_dir)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, arrow_dir)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        rot_mat = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2 + 1e-8))

    geom_id = scene.ngeom
    if geom_id >= len(scene.geoms):
        return

    # Create a new geom object and add it to the scene
    geom = scene.geoms[geom_id]

    # Arrow geom type
    geom.type = mujoco.mjtGeom.mjGEOM_ARROW

    # Set the geom size: [shaft_radius, head_radius, arrow_length]
    geom.size[:] = [0.01, 0.02, arrow_length]

    # Position the geom at the body and orient it
    geom.pos[:] = body_pos
    geom.mat[:] = rot_mat

    geom.rgba[:] = [1, 0, 0, 1]
    scene.ngeom += 1

# # Log eval action to csv
# joint_names = []
# for i in range(env._mj_model.nu):
#   name = mujoco.mj_id2name(env._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
#   print(i, name)
#   joint_names.append(name)
# csv_header = ["steps"] + joint_names
steps = []
for j in range(episode): 
    print(f"episode {j}")
    rollout = []
    modify_scene_fns = []
    rng = jax.random.PRNGKey(j)
    state = jit_reset(rng)
    state.info["phase_dt"] = phase_dt
    state.info["phase"] = phase
    state.info["gait_freq"] = gait_freq
    state.info["foot_height"] = foot_height
    state.info["gait"] = gait
    # with open("actions_log.csv", mode="w", newline="") as file:
    #   writer = csv.writer(file)
    #   writer.writerow(csv_header)
    step = 0
    csv_path = os.path.join(_CSV_PATH, f"episode-{j}.csv")

    for i in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        onnx_input = {
            'obs': np.array(state.obs["state"], dtype=np.float32).reshape(1, -1)
        }
        ctrl = m.run(output_names, onnx_input)[0][0]
        # ctrl, _ = jit_inference_fn(state.obs, act_rng)

        keys = sorted(state.metrics.keys())
        row = {"step": int(i + 1)}
        row.update({k: state.metrics[k] for k in keys})

        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step"] + keys)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        data = state.data
        if push_step <= i < push_step + push_duration:
            # Create a force vector (6D: force + torque)
            force = jp.concatenate([push_force_x, jp.zeros(3)])  # No torque
            data = data.replace(
                xfrc_applied=data.xfrc_applied.at[body_id].set(force)
            )
        elif push_step*2 <= i < push_step*2 + push_duration:
            # Create a force vector (6D: force + torque)
            force = jp.concatenate([push_force_y, jp.zeros(3)])  # No torque
            data = data.replace(
                xfrc_applied=data.xfrc_applied.at[body_id].set(force)
            )
        else:
            # No force
            data = data.replace(
                xfrc_applied=data.xfrc_applied.at[body_id].set(jp.zeros(6))
            )

        state = state.replace(data=data)

        state = jit_step(state, ctrl)
        if state.done:
          break
        state.info["command"] = command
        rollout.append(state)
        step += 1

        # Log step + action values
        # writer.writerow([step] + list(ctrl))
        xyz = np.array(state.data.xpos[eval_env.mj_model.body("base_link").id])
        xyz += np.array([0.0, 0.0, 0.0])
        x_axis = state.data.xmat[eval_env.mj_model.body("base_link").id, 0]
        yaw = -np.arctan2(x_axis[1], x_axis[0])
        modify_scene_fns.append(
            functools.partial(
                draw_joystick_command,
                cmd=state.info["command"],
                xyz=xyz,
                theta=yaw,
                scl=np.linalg.norm(state.info["command"])
            )
        )
        # modify_scene_fns.append(
        #     functools.partial(
        #         draw_force_arrow,
        #         data=state.data
        #     )
        # )
    steps.append(step * eval_env.dt)

    render_every = 1
    fps = 1.0 / eval_env.dt / render_every
    print(f"fps: {fps}")
    traj = rollout[::render_every]
    mod_fns = modify_scene_fns[::render_every]

    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[2] = True
    scene_option.geomgroup[3] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
    frames = eval_env.render(
        traj,
        camera="track",
        scene_option=scene_option,
        width=640,
        height=480,
        modify_scene_fns=mod_fns,
    )
    # media.show_video(frames, fps=fps, loop=False)
    if gait == 1:
       folder_name = f"/home/sandbox/Work/mujoco_playground/learning/notebooks/videos/{name}-gait-{gait}-push-{abs(push_magnitude)}"
       file_name = f"/home/sandbox/Work/mujoco_playground/learning/notebooks/videos/{name}-gait-{gait}-push-{abs(push_magnitude)}/seed{j}.mp4"
    else:
       folder_name = f"/home/sandbox/Work/mujoco_playground/learning/notebooks/videos/{name}-gait-{gait}-gait-freq-{gait_freq}-foot-height-{foot_height}-push-{abs(push_magnitude)}"
       file_name = f"/home/sandbox/Work/mujoco_playground/learning/notebooks/videos/{name}-gait-{gait}-gait-freq-{gait_freq}-foot-height-{foot_height}-push-{abs(push_magnitude)}/seed{j}.mp4"
    os.makedirs(folder_name, exist_ok=True)
    media.write_video(file_name, frames, fps=fps, qp=18)
print("steps:", steps)
print("steps > 5s: ", sum([1 for s in steps if s > 5.0]))
print("steps > 10s: ", sum([1 for s in steps if s > 10.0]))
print("steps > 15s: ", sum([1 for s in steps if s > 15.0]))
print("steps > 25s: ", sum([1 for s in steps if s == 20.0]))