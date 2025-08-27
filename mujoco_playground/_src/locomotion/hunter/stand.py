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
        episode_length=100,
        action_repeat=1,
        # action_scale=1.6,  # Direct torque scaling
        action_scale=0.6,
        obs_noise=0.01,
        reward_config=config_dict.create(
            scales=config_dict.create(
                upright=1.0,
                stability=0.5, 
                # effort=-0.0002,  # Lower effort penalty for torque control
                effort=0.0,  # Lower effort penalty for torque control
                joint_limits=0.0,
                height=2.0,
                pose=-1.0,
                # penalty_fall=100,
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
        self._init_q = self._init_q.at[2].set(0.0)  # z position - proper standing height
        self._init_q = self._init_q.at[3:7].set(jp.array([1, 0, 0, 0]))  # quat
        
        # Set joint positions for stable standing
        # joint_init = jp.array([0.0, 0.0, 0.1, 0.2, -0.1] * 2)  # 10 joints
        # joint_init = jp.array([0.0, 0.0, -0.2, 0.4, -0.15] * 2)  # 10 joints
        joint_init = jp.array([0.0, 0.0, 0.2, 0.4, 0.15, 0.0, 0.0, -0.2, 0.4, -0.15])  # 10 joints
        self._init_q = self._init_q.at[7:].set(joint_init)

        self._default_pose = joint_init
        # self._lowers = jp.array([-0.2, -0.5, -0.8, 0.0, -1.1] * 2)
        # self._uppers = jp.array([0.5, 1.0, 1.2, 1.5, 1.1] * 2)
        self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self._mj_model.actuator_ctrlrange[:, 1]

        # Body and sensor IDs
        self._base_body_id = self._mj_model.body(hunter_constants.ROOT_BODY).id
        self._imu_site_id = self._mj_model.site("imu").id

        # # --- Get motor IDs ---
        # # Each actuator controls exactly one joint in most torque-control setups
        # motor_ids = []
        # for i in range(self._mj_model.nu):  # nu = number of actuators
        #     joint_id = self._mj_model.actuator_trnid[i][0]  # first entry is joint ID
        #     motor_ids.append(joint_id)
        # self.motor_id = jp.array(motor_ids)

        # # --- Store default pose for relative joint positions ---
        # self._default_qpos = self._init_q[7:]  # first keyframe as default pose
        

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

        # Convert action to joint torques (direct torque control)
        motor_targets = self._default_pose + action * self._config.action_scale
        
        # Clip torques to safe limits
        # torque_limits = 100.0  # Maximum torque in Nm
        # motor_targets = jp.clip(motor_targets, -torque_limits, torque_limits)
        motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)

        # Step the simulation
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        # # Step physics n_substeps
        # data = state.data
        # for _ in range(self.n_substeps):
        #     data = data.replace(ctrl=motor_targets)
        #     data = mjx.step(self.mjx_model, data)

        # Get observation
        obs = self._get_obs(data, state.info, noise_rng)

        # Check termination conditions for torque control
        base_z = data.xpos[self._base_body_id, 2]
        base_quat = data.xquat[self._base_body_id]
        up_dot = jp.abs(base_quat[0])  # Simplified upright check

        done = base_z < 0.3  # Robot fell down (below ground level)
        done |= up_dot < 0.5  # Robot tipped over significantly
        # done |= jp.any(data.qpos[7:] < self._lowers)  # Joint limits
        # done |= jp.any(data.qpos[7:] > self._uppers)

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
        # return 3 + 3 + 3 + 10  # gravity + linear_accel + angular_vel + joint_torques
        # Observation vector: gravity(3) + accelerometer(3) + gyro(3) + joint_torques(10) + joint_pos(10) + joint_vel(10) = 39
        return 3 + 3 + 3 + 10 + 10 + 10

    def _get_obs(
        self,
        data: mjx.Data,
        info: dict[str, Any],
        rng: jax.Array,
    ) -> jax.Array:
        """Get the observation vector with IMU data and joint torques."""
        # IMU data: Gravity vector in base frame (3)
        gravity = self.get_gravity(data)

        # IMU data: Linear acceleration (3)
        # linear_accel = self._get_linear_acceleration(data)
        linear_accel = self.get_accelerometer(data)
        
        # IMU data: Angular velocity (3)
        # angular_vel = self._get_angular_velocity(data)
        angular_vel = self.get_gyro(data)

        # Current joint action (10) 
        joint_torques = info["last_act"]

        # # --- Joint states ---
        # qpos = data.qpos[self.motor_id] - self._default_qpos  # relative joint angles
        # qvel = data.qvel[self.motor_id]                       # joint velocities

        obs = jp.concatenate([
            gravity,        # 3
            linear_accel,   # 3
            angular_vel,    # 3
            joint_torques,  # 10
            # qpos,
            # qvel,
            # total: 19

            data.qpos[7:] - self._default_pose,  # 10
            data.qvel[6:],  # 10
            # total: 39
        ])

        # Add noise if specified
        if self._config.obs_noise > 0.0:
            noise = self._config.obs_noise * jax.random.normal(
                rng, obs.shape
            )
            obs = obs + noise

        return obs


    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
    ) -> dict[str, jax.Array]:
        """Calculate reward components for torque control."""
        # Upright reward - higher when base is upright
        base_quat = data.xquat[self._base_body_id]
        upright = base_quat[0] ** 2  # Reward for staying upright (w component squared)

        # Stability reward - penalize high velocities
        lin_vel = jp.sum(jp.square(data.qvel[:3]))
        ang_vel = jp.sum(jp.square(data.qvel[3:6]))
        joint_vel = jp.sum(jp.square(data.qvel[6:]))
        # stability = jp.exp(-1.0 * (lin_vel + ang_vel + 0.1 * joint_vel))
        stability = -1.0 * (lin_vel + ang_vel + 0.1 * joint_vel)


        # Effort penalty - penalize high torques (more important for torque control)
        torque_magnitude = jp.linalg.norm(action)
        # effort = jp.exp(-0.01 * torque_magnitude)
        effort = torque_magnitude

        # Joint limit penalty
        joint_pos = data.qpos[7:]
        # lower_violation = jp.sum(jp.maximum(0, self._lowers - joint_pos))
        # upper_violation = jp.sum(jp.maximum(0, joint_pos - self._uppers))
        # joint_limits = jp.exp(1.0 * (lower_violation + upper_violation))
        joint_limits = 0.0

        # Height reward - encourage staying at proper height
        base_height = data.xpos[self._base_body_id, 2]
        target_height = 0.69  # Match the current init height
        height = jp.exp(-1.0 * jp.abs(base_height - target_height))

        # is_upright = (base_height>0.3)
        # penalty_fall = -1.0 * (1 - is_upright)

        # Pose reward - encourage staying close to default standing pose
        # pose_error = jp.linalg.norm(joint_pos - self._default_pose)
        pose_error = jp.sum(jp.square(joint_pos - self._default_pose))
        # pose_reward = jp.exp(-5.0 * pose_error)
        pose_reward = pose_error

        return {
            "upright": upright,
            "stability": stability, 
            "effort": effort,
            "joint_limits": joint_limits,
            "height": height,
            "pose": pose_reward,
            # "penalty_fall":penalty_fall,
        }
