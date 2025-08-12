from typing import Any, Dict, Optional, Union

from etils import epath
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.hunter import hunter_constants as consts

def get_assets() -> Dict[str, bytes]:
    assets = {}
    mjx_env.update_assets(assets, consts.ROOT_PATH, "*.xml")
    mjx_env.update_assets(assets, consts.ROOT_PATH, "meshes", "*.STL")
    return assets

class HunterEnv(mjx_env.MjxEnv):
    """Base class for Hunter environments."""
    def __init__(
        self,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ) -> None:
        super().__init__(config, config_overrides)
        xml_path = epath.Path(consts.HUNTER_XML)
        self._model_assets = get_assets()
        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(), assets=self._model_assets
        )
        self._mj_model.opt.timestep = self._config.sim_dt
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        self._xml_path = xml_path

    def reset(self, rng: jax.Array):
        # Example: Reset the environment state. This is a placeholder and should be adapted to your state structure.
        # Typically, you would randomize initial state using rng if needed.
        # Here, we just call the parent reset if available, or return a dummy state.
        # Replace with your own logic as needed.
        if hasattr(super(), 'reset'):
            return super().reset(rng)
        else:
            # Dummy state: replace with your own state structure
            import jax.numpy as jnp
            obs = jnp.zeros(self.action_size)
            return obs

    def step(self, state, action):
        # Example: Step the environment. This is a placeholder and should be adapted to your state structure.
        # Typically, you would apply the action, simulate, and return the new state, reward, done, info.
        if hasattr(super(), 'step'):
            return super().step(state, action)
        else:
            # Dummy next state, reward, done, info
            import jax.numpy as jnp
            next_state = state
            reward = 0.0
            done = False
            info = {}
            return next_state, reward, done, info

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GRAVITY_SENSOR)

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR)

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR)

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.LOCAL_LINVEL_SENSOR)

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.ACCELEROMETER_SENSOR)

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
