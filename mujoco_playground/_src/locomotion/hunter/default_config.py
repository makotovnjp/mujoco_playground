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
  config.reward_config.scales.upright = 2.0  # 姿勢を維持する報酬を強化
  config.reward_config.scales.height = 1.0  # 高さを維持する報酬を強化
  config.reward_config.scales.stability = -0.05  # 安定性のペナルティを軽減
  config.reward_config.scales.effort = -0.001  # 努力のペナルティはそのまま
  config.reward_config.scales.joint_limits = -0.5  # 関節制限のペナルティはそのまま
  
  # MJX specific settings
  config.nconmax = 8 * 1024
  config.njmax = 19 + 8 * 4
  
  return config
