"""Standing task for Hunter robot."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.hunter import base as hunter_base
from mujoco_playground._src.locomotion.hunter import hunter_constants


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.3,
        obs_noise=0.01,
        reward_config=config_dict.create(
            scales=config_dict.create(
                upright=2.0,
                stability=1.0, 
                effort=0.1,
                joint_limits=1.0,
                height=1.0,
                pose=1.0,
            ),
        ),
        impl="jax",
        nconmax=8 * 1024,
        njmax=19 + 8 * 4,
    )


class Stand(hunter_base.HunterEnv):
    """
    A standing environment for the Hunter robot.
    The goal is to maintain an upright standing posture.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[
            Dict[str, Union[str, int, list[Any]]]
        ] = None,
    ):
        super().__init__(
            hunter_constants.HUNTER_XML.as_posix(), config, config_overrides
        )
        self._post_init()

    def _post_init(self) -> None:
        """Initialize environment-specific parameters."""
        # Default standing pose with slightly bent knees
        self._init_q = jp.zeros(self._mjx_model.nq)
        # Set floating base position (x, y, z, quat)
        self._init_q = self._init_q.at[2].set(-0.17)  # z position - proper standing height
        self._init_q = self._init_q.at[3:7].set(jp.array([1, 0, 0, 0]))  # quat
        # Set joint positions for stable standing
        joint_init = jp.array([0.0, 0.0, 0.1, 0.2, -0.1] * 2)  # 10 joints
        self._init_q = self._init_q.at[7:].set(joint_init)

        self._default_pose = joint_init
        self._lowers = jp.array([-0.2, -0.5, -0.8, 0.0, -1.1] * 2)
        self._uppers = jp.array([0.5, 1.0, 1.2, 1.5, 1.1] * 2)

        # Body and sensor IDs
        self._base_body_id = self._mj_model.body(hunter_constants.ROOT_BODY).id
        self._imu_site_id = self._mj_model.site("imu").id

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset the environment to the initial standing position."""
        print("resetting...")
        rng, noise_rng = jax.random.split(rng, 2)

        # Initialize robot to standing position without noise
        qpos = self._init_q  # Use the pre-defined stable standing pose
        qvel = jp.zeros(self._mjx_model.nv)

        data = mjx_env.make_data(
            self.mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self.mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        
        data = mjx.forward(self.mjx_model, data)

        info = {
            "rng": rng,
            "last_act": jp.zeros(self._mjx_model.nu),
            "step": 0,
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs = self._get_obs(data, info, noise_rng)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step the environment forward."""
        print("stepping...")        
        rng, noise_rng = jax.random.split(state.info["rng"], 2)

        # Convert action to joint targets for training
        motor_targets = self._default_pose + action * self._config.action_scale
        motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)

        # Step the simulation
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        # Get observation
        obs = self._get_obs(data, state.info, noise_rng)

        # Check termination conditions
        base_z = data.xpos[self._base_body_id, 2]
        base_quat = data.xquat[self._base_body_id]
        up_dot = jp.abs(base_quat[0])  # Simplified upright check

        done = base_z < 0.3  # Robot fell down
        done |= up_dot < 0.7  # Robot tipped over significantly
        done |= jp.any(data.qpos[7:] < self._lowers)  # Joint limits
        done |= jp.any(data.qpos[7:] > self._uppers)

        # Calculate rewards for training
        rewards = self._get_reward(data, action, state.info)
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, -1000.0, 1000.0)

        # Update info
        state.info["last_act"] = action
        state.info["step"] += 1
        state.info["rng"] = rng

        # Update reward metrics for training
        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = jp.float32(done)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_obs_size(self) -> int:
        """Get the size of the observation vector."""
        return 3 + 3 + 10 + 10 + 10  # gravity + angular_vel + qpos + qvel + action

    def _get_obs(
        self,
        data: mjx.Data,
        info: dict[str, Any],
        rng: jax.Array,
    ) -> jax.Array:
        """Get the observation vector."""
        # Gravity vector in base frame (3)
        gravity = self._get_gravity_vector(data)

        # Angular velocity (3)
        angvel = self._get_angular_velocity(data)

        # Joint positions relative to default (10)
        joint_pos = data.qpos[7:] - self._default_pose

        # Joint velocities (10)
        joint_vel = data.qvel[6:]

        # Previous action (10)
        last_act = info["last_act"]

        obs = jp.concatenate([
            gravity,      # 3
            angvel,       # 3
            joint_pos,    # 10
            joint_vel,    # 10
            last_act,     # 10
            # total: 36
        ])

        # Add noise if specified
        if self._config.obs_noise > 0.0:
            noise = self._config.obs_noise * jax.random.normal(
                rng, obs.shape
            )
            obs = obs + noise

        return obs

    def _get_gravity_vector(self, data: mjx.Data) -> jax.Array:
        """Get gravity vector in the base frame."""
        # Extract rotation matrix from base quaternion
        base_quat = data.xquat[self._base_body_id]
        # Convert quaternion to rotation matrix
        w, x, y, z = base_quat
        rot_mat = jp.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
        gravity_world = jp.array([0, 0, -1])
        gravity_base = rot_mat.T @ gravity_world
        return gravity_base

    def _get_angular_velocity(self, data: mjx.Data) -> jax.Array:
        """Get angular velocity of the base."""
        return data.qvel[3:6]

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> dict[str, jax.Array]:
        """Calculate reward components."""
        # Upright reward - higher when base is upright
        base_quat = data.xquat[self._base_body_id]
        upright = base_quat[0] ** 2  # Reward for staying upright (w component squared)

        # Stability reward - penalize high velocities
        lin_vel = jp.linalg.norm(data.qvel[:3])
        ang_vel = jp.linalg.norm(data.qvel[3:6])
        joint_vel = jp.linalg.norm(data.qvel[6:])
        stability = jp.exp(-2.0 * (lin_vel + ang_vel + 0.1 * joint_vel))

        # Effort penalty - encourage minimal action
        effort = jp.exp(-0.1 * jp.linalg.norm(action))

        # Joint limit penalty
        joint_pos = data.qpos[7:]
        lower_violation = jp.sum(jp.maximum(0, self._lowers - joint_pos))
        upper_violation = jp.sum(jp.maximum(0, joint_pos - self._uppers))
        joint_limits = jp.exp(-10.0 * (lower_violation + upper_violation))

        # Height reward - encourage staying at proper height
        base_height = data.xpos[self._base_body_id, 2]
        target_height = 0.0  # Adjusted target height for proper standing
        height = jp.exp(-10.0 * jp.abs(base_height - target_height))

        # Pose reward - encourage staying close to default standing pose
        pose_error = jp.linalg.norm(joint_pos - self._default_pose)
        pose_reward = jp.exp(-2.0 * pose_error)

        return {
            "upright": upright,
            "stability": stability, 
            "effort": effort,
            "joint_limits": joint_limits,
            "height": height,
            "pose": pose_reward,
        }
