from ml_collections import config_dict

def default_config():
    config = config_dict.ConfigDict()
    config.sim_dt = 0.002
    config.impl = 'mjx'
    # Add more default config values as needed for Hunter
    return config
