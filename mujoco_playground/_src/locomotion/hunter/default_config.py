from ml_collections import config_dict

def default_config():
    config = config_dict.ConfigDict()
    config.sim_dt = 0.002
    config.ctrl_dt = 0.02
    config.impl = 'mjx'
    config.episode_length = 1000
    config.action_repeat = 1
    config.action_scale = 0.5
    config.obs_noise = 0.0
    # Add more default config values as needed for Hunter
    return config
