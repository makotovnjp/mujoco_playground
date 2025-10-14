# @title Import packages for plotting and creating graphics
import json
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

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
import os
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

# Make model, data, and renderer
env_name = 'HunterStand'
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)

from mujoco_playground._src.locomotion.hunter import hunter_constants as consts
def get_assets() -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH, "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "meshes", "*.STL")
  return assets

mj_model = mujoco.MjModel.from_xml_string(epath.Path(env.xml_path).read_text(), assets=get_assets())
mj_data = mujoco.MjData(mj_model)
renderer = mujoco.Renderer(mj_model)

#----------------------------------#
# Visualize init position in Mujoco
#----------------------------------#
_init_q = jp.zeros(mj_model.nq)
_init_q = _init_q.at[2].set(-0.01)  # z position - proper standing height
_init_q = _init_q.at[3:7].set(jp.array([1, 0, 0, 0]))  # quat
joint_init = jp.array([0.0, 0.0, 0.2, 0.4, 0.15, 0.0, 0.0, -0.2, 0.4, -0.15])  # 10 joints
# joint_init = jp.array([0.0, 0.0, 0.0, 0.0, 0.0] * 2)  # 10 joints

_lowers = jp.array([0.0, 0.0, -0.2, 0.6, -0.3] * 2)
_uppers = jp.array([2.0, 2.0, 2.0, 2.0, 2.0] * 2)

_init_q = _init_q.at[7:].set(joint_init)
mj_data.qpos = _init_q

mujoco.mj_forward(mj_model, mj_data)
print(mj_data.qpos)

# Initialize paused state
paused = True
# Define a key callback function to toggle the paused state with the spacebar
def key_callback(keycode):
    global paused
    if chr(keycode) == ' ':  # Spacebar keycode
        paused = not paused
        
with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
    while viewer.is_running():
        if not paused:
            mujoco.mj_step(mj_model, mj_data)
        viewer.sync()

