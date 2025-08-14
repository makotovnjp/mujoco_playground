# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Default configuration for Hunter robot tasks."""

from ml_collections import config_dict


def default_config() -> config_dict.ConfigDict:
  """Default configuration for Hunter robot tasks."""
  config = config_dict.ConfigDict()
  config.sim_dt = 0.002
  config.ctrl_dt = 0.02
  config.impl = 'mjx'
  config.episode_length = 1000
  config.action_repeat = 1
  config.action_scale = 0.3
  config.obs_noise = 0.01
  
  # Reward configuration
  config.reward_config = config_dict.ConfigDict()
  config.reward_config.scales = config_dict.ConfigDict()
  config.reward_config.scales.upright = 1.0
  config.reward_config.scales.stability = -0.1
  config.reward_config.scales.effort = -0.001
  config.reward_config.scales.joint_limits = -0.5
  config.reward_config.scales.height = 0.5
  
  # MJX specific settings
  config.nconmax = 8 * 1024
  config.njmax = 19 + 8 * 4
  
  return config
