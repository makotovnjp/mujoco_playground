# Joystick control for the hunter robot

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.hunter import base as hunter_base
from mujoco_playground._src.locomotion.hunter import hunter_constants

_PHASES = np.array([
    [0, np.pi],  # walk
    [0.0, 0.0], # stand
    # [0, np.pi], # run
])

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.01,
      sim_dt=0.001,
      episode_length=1000,
      early_termination=True,
      action_repeat=1,
      action_scale=0.5,
      dof_vel_scale=0.05,
      lin_vel_scale=2.0,
      history_len=1,
      noise_config=config_dict.create(
          level=1.0,
          # level=0.6,
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=1.5,
              gyro=0.2,
              linvel=0.1,
              gravity=0.05,
          ),
      ),
      # reward_config=config_dict.create(
      #     scales=config_dict.create(
      #         # Rewards.
      #         feet_phase=5.0,
      #         tracking_lin_vel=3.5,
      #         tracking_ang_vel=0.75,
      #         # feet_air_time=2.0,

      #         # feet_phase=3.0,
      #         # tracking_lin_vel=0.0,
      #         # tracking_ang_vel=0.0,
      #         feet_air_time=2.0,
      #         feet_contact=0.5,
      #         feet_clearance=-1.0,

      #         # Costs.
      #         ang_vel_xy=-0.0,
      #         lin_vel_z=-0.0,
      #         orientation=-2.0,
      #         pose=-1.0,
      #         stand_still=+0.0,
      #         foot_slip=-0.1,
      #         action_rate=-0.01,
      #         feet_distance=-0.0,
      #     ),
      #     tracking_sigma=0.5,
      # ),
      # command_config=config_dict.create(
      #     lin_vel_x=[-1.5, 1.5],
      #     lin_vel_y=[-0.5, 0.5],
      #     ang_vel_yaw=[-1.0, 1.0],
      #     lin_vel_threshold=0.1,
      #     ang_vel_threshold=0.1,
      # ),

      reward_config=config_dict.create(
          scales=config_dict.create(
              # Rewards.
              # feet_phase=5.0,
              tracking_lin_vel=3.5,
              tracking_ang_vel=0.75,
              # feet_air_time=2.0,

              feet_phase=3.0,
              # tracking_lin_vel=0.0,
              # tracking_ang_vel=0.0,
              feet_air_time=2.0,
              feet_contact=0.0,
              # feet_air_time=0.0,
              # feet_contact=0.0,
          
              feet_clearance=-0.0,

              # Costs.
              ang_vel_xy=-0.15,  # previous: -0.0
              # lin_vel_z=-0.0,
              lin_vel_z=-0.0,  # previous: -5.0
              orientation=-2.0,
              joint_deviation_knee=-0.1,
              joint_deviation_hip=-0.5,
              pose=-1.0,  # previous: -0.1
              stand_still=0.5,  # previous: +4.0
              # stand_still=+0.0,
              termination=-1.0,
              foot_slip=-0.25,
              # action_rate=-0.01,  # previous: -0.5
              action_rate=-0.1,  # previous: -0.5
              # feet_distance=-0.3,
              feet_distance=-2.0,
              collision=-0.1,
          ),
          tracking_sigma=0.5,
      ),
      command_config=config_dict.create(
          lin_vel_x=[-1.5, 1.5],
          lin_vel_y=[-1.0, 1.0],
          # ang_vel_yaw=[-1.2, 1.2]
          ang_vel_yaw=[-2*np.pi, 2*np.pi]
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
    #   gait_frequency=[1.25, 2.0],
      # gait_frequency=[0.0, 0.5],
      gait_frequency=[0.5, 4.0],
    #   gaits=["walk"],
      gaits=["walk", "stand"],
      # gaits=["walk","stand","run"],
      foot_height=[0.08, 0.2],
      impl="jax",
      nconmax=8 * 1024,
      njmax=10 + 8 * 4,
  )

class Joystick(hunter_base.HunterEnv):
  """Hunter environment with joystick control."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    if task.startswith("rough"):
      config.nconmax = 100 * 8192
      config.njmax = 12 + 100 * 4
    super().__init__(
        hunter_constants.task_to_xml(task).as_posix(),
        config, 
        config_overrides
    )
    self._post_init()
  
  def _post_init(self):
    # # Default standing pose with slightly bent knees
    self._init_q = jp.zeros(self._mjx_model.nq)
    self._init_q = self._init_q.at[3:7].set(jp.array([1, 0, 0, 0]))  # quat
    
    # # Set joint positions for stable standing
    # # Set floating base position (x, y, z, quat)
    # # self._init_q = self._init_q.at[2].set(-0.014)   # z position - proper standing height
    # # joint_init = jp.array([0.0, 0.0, -0.2, 0.5, -0.3, 0.0, 0.0, -0.2, 0.5, -0.3])   # 10 joints

    # # SAME AS ROS1
    # # self._init_q = self._init_q.at[2].set(-0.05)  # z position - proper standing height
    # # joint_init = jp.array([0.1, 0.0, -0.4, 0.93, -0.53, -0.1, 0.0, -0.4, 0.93, -0.53]) 

    # # SAME AS CUSTOMER DOC
    self._init_q = self._init_q.at[2].set(-0.029)  # z position - proper standing height
    joint_init = jp.array([0.0, 0.0, -0.36, 0.72, -0.36, 0.0, -0.05, -0.36, 0.72, -0.36]) 

    self._init_q = self._init_q.at[7:].set(joint_init)

    self._default_pose = joint_init
    # self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    # self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    # Set joint limits
    self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
    self._uppers = self._mj_model.actuator_ctrlrange[:, 1]

    self._hx_idxs = jp.array([
        0, 1, 2, 3, 4,  # left leg
        5, 6, 7, 8, 9,  # right leg
    ])  # fmt: skip
    self._hip_indices = jp.array([0, 1, 5, 6])
    self._knee_indices = jp.array([3, 8])
    self._weights = jp.array([
        1.0, 1.0, 0.01, 0.01, 1.0,
        1.0, 1.0, 0.01, 0.01, 1.0,
    ])  # fmt: skip

    self._hx_default_pose = self._default_pose[self._hx_idxs]

    self._base_body_id = self._mj_model.body(hunter_constants.ROOT_BODY).id
    self._imu_site_id = self._mj_model.site("imu").id

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in hunter_constants.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("ground").id
    self._left_feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in hunter_constants.LEFT_FEET_GEOMS]
    )
    self._right_feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in hunter_constants.RIGHT_FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in hunter_constants.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._left_foot_box_geom_id = self._mj_model.geom("left_foot").id
    self._right_foot_box_geom_id = self._mj_model.geom("right_foot").id
   
  def sample_command(self, rng: jax.Array) -> jax.Array:
    """Samples a random command with a 10% chance of being zero."""
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    cmd_config = self._config.command_config
    lin_vel_x = jax.random.uniform(
        rng1, minval=cmd_config.lin_vel_x[0], maxval=cmd_config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=cmd_config.lin_vel_y[0], maxval=cmd_config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=cmd_config.ang_vel_yaw[0],
        maxval=cmd_config.ang_vel_yaw[1],
    )    
    # With 10% chance, set everything to zero.
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )

    cmd = jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw])
    return cmd
  
  def reset(self, rng: Optional[Union[int, jp.ndarray]] = None):
    rng, noise_rng, gait_freq_rng, gait_rng, foot_height_rng, cmd_rng = (  # pylint: disable=redefined-outer-name
        jax.random.split(rng, 6)
    )

    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (10,), minval=0.5, maxval=1.5)
    )

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # Initialize history buffers.
    qpos_error_history = jp.zeros(self._config.history_len * 10)
    qvel_history = jp.zeros(self._config.history_len * 10)

    # Sample gait parameters.
    gait_freq = jax.random.uniform(
        gait_freq_rng,
        minval=self._config.gait_frequency[0],
        maxval=self._config.gait_frequency[1],
    )
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    gait = jax.random.randint(  # pylint: disable=redefined-outer-name
        gait_rng, minval=0, maxval=len(self._config.gaits), shape=()
    )
    phase = jp.array(_PHASES)[gait]
    foot_height = jax.random.uniform(
        foot_height_rng,
        minval=self._config.foot_height[0],
        maxval=self._config.foot_height[1],
    )

    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)


    # info = {
    #     "rng": rng,
    #     "last_act": jp.zeros(self.mjx_model.nu),
    #     "last_vel": jp.zeros(self.mjx_model.nv - 6),
    #     "command": self.sample_command(cmd_rng),
    #     "step": 0,
    # }

    info = {
        "command": self.sample_command(cmd_rng),
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "step": 0,
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "qpos_error_history": qpos_error_history,
        "qvel_history": qvel_history,
        "swing_peak": jp.zeros(2),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "lin_vel": jp.zeros(3),
        "ang_vel": jp.zeros(3),
        "gait_freq": gait_freq,
        "gait": gait,
        "phase": phase,
        "phase_dt": phase_dt,
        "foot_height": foot_height,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    left_feet_contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._left_feet_geom_id
    ])
    right_feet_contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._right_feet_geom_id
    ])
    contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])

    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)


  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    
    # Push related
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)

    rng, cmd_rng, noise_rng = jax.random.split(state.info["rng"], 3)

    motor_targets = self._default_pose + action * self._config.action_scale
    motor_targets = jp.clip(motor_targets, self._lowers, self._uppers)
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps  # pytype: disable=attribute-error
    )
    state.info["motor_targets"] = motor_targets

    left_feet_contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._left_feet_geom_id
    ])
    right_feet_contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._right_feet_geom_id
    ])
    contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    # obs = self._get_obs(data, state.info, state.obs, noise_rng)
    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    # joint_angles = data.qpos[7:]
    # joint_vel = data.qvel[6:]
    # base_z = data.xpos[self._base_body_id, 2]

    # done = self.get_gravity(data)[-1] < 0.59
    # done |= jp.any(joint_angles < self._lowers)
    # done |= jp.any(joint_angles > self._uppers)
    # done |= base_z < 0.65

    # rewards = self._get_reward(data, action, state.info, state.metrics, done)
    # rewards = {
    #     k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    # }

    pos, neg = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    pos = {k: v * self._config.reward_config.scales[k] for k, v in pos.items()}
    neg = {k: v * self._config.reward_config.scales[k] for k, v in neg.items()}
    rewards = pos | neg

    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0)

    
    # Bookkeeping.
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    # state.info["last_vel"] = joint_vel
    state.info["step"] += 1
    state.info["push_step"] += 1
    phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    state.info["rng"] = rng
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact

    state.info["command"] = jp.where(
        state.info["step"] > 500,
        self.sample_command(cmd_rng),
        state.info["command"],
    )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )

    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = jp.float32(done)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_gravity(data)[-1] < 0.85
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self,
      data: mjx.Data,
      info: dict[str, Any],
      contact: jax.Array,
  ) -> jp.ndarray:
    # IMU data: Gravity vector in base frame (3)
    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:] 
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    # TODO: Disable linvel noise for now as it causes instability
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    cos = jp.cos(info["phase"])
    sin = jp.sin(info["phase"])
    phase = jp.concatenate([cos, sin])

    state = jp.concatenate([
        noisy_linvel * self._config.lin_vel_scale,  # 3
        noisy_gyro,    # 3
        noisy_gravity,        # 3
        noisy_joint_angles - self._default_pose,  # 10
        noisy_joint_vel * self._config.dof_vel_scale,  # 10
        info["last_act"],  # 10
        info["command"],  # 3
        phase,  # 4
        #info["gait"],   #1
        #info["gait_freq"],   #1
        #info["foot_height"]   #1
        # total: 49
      ],
    )

    accelerometer = self.get_accelerometer(data)
    global_angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel * self._config.lin_vel_scale,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel * self._config.dof_vel_scale,
        root_height,  # 1
        data.actuator_force,  # 29
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
        info["gait"],   #1
        info["gait_freq"],   #1
        info["foot_height"]   #1
    ])


    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_local_angvel(self, data: mjx.Data) -> jax.Array:
    return self.get_gyro(data)

  def _get_global_linvel(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "global_linvel")

  def _get_global_angvel(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "global_angvel")

  def _get_local_linvel(self, data: mjx.Data) -> jax.Array:
    return self._get_sensor_data(data, "local_linvel")
  
  def _get_sensor_data(self, data: mjx.Data, sensor_name: str) -> jax.Array:
    sensor_id = self._mj_model.sensor(sensor_name).id
    sensor_adr = self._mj_model.sensor_adr[sensor_id]
    sensor_dim = self._mj_model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    del metrics  # Unused.
    pos = {
        "tracking_lin_vel": self._reward_tracking_lin_vel(
            info["command"], self.get_local_linvel(data)
        ),
        "tracking_ang_vel": self._reward_tracking_ang_vel(
            info["command"], self.get_gyro(data)
        ),
        "feet_phase": self._reward_feet_phase(
            data, info["phase"], info["foot_height"]
        ),
        "feet_air_time": self._reward_feet_air_time(
            info["feet_air_time"], first_contact, info["command"]
        ),
        "feet_contact": self._reward_feet_contact(data),
    }
    neg = {
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "lin_vel_z": self._cost_lin_vel_z(
            self.get_global_linvel(data), info["gait"]
        ),
        "orientation": self._cost_orientation(self.get_gravity(data)),
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:], info["command"]
        ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        "pose": self._cost_pose(data.qpos[7:]),
        "foot_slip": self._cost_feet_slip(data),
        "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "action_rate": self._cost_action_rate(
            info["last_act"], info["last_last_act"], action
        ),
        "termination": self._cost_termination(done),
        "feet_clearance": self._cost_feet_clearance(data),
        "feet_distance": self._cost_feet_distance(data),
        "collision": self._cost_collision(data)
    }
    return pos, neg

  def _reward_feet_phase(
      self, data: mjx.Data, phase: jax.Array, foot_height: jax.Array
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    return jp.exp(-error / 0.01)

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of linear velocity commands (xy axes).
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    # Tracking of angular velocity commands (yaw).
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)
  
  def _reward_feet_air_time(
      self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
  ) -> jax.Array:
    # Reward air time.
    cmd_norm = jp.linalg.norm(commands[:2])
    rew_air_time = jp.sum((air_time - 0.1) * first_contact)
    rew_air_time *= cmd_norm > 0.05  # No reward for zero commands.
    return rew_air_time

  def _reward_feet_contact(
    self, data:mjx.Data
  ):
    left_feet_contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._left_feet_geom_id
    ])
    right_feet_contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._right_feet_geom_id
    ])
    feet_contact = jp.hstack(
        [left_feet_contact.any(), right_feet_contact.any()]
    )
    return jp.mean(feet_contact)

  def _cost_joint_deviation_hip(
      self, qpos: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    cost = jp.sum(
        jp.abs(qpos[self._hip_indices] - self._default_pose[self._hip_indices])
    )
    cost *= jp.abs(cmd[1]) > 0.1
    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(
        jp.abs(
            qpos[self._knee_indices] - self._default_pose[self._knee_indices]
        )
    )

  def _cost_pose(self, joint_angles: jax.Array) -> jax.Array:
    # Penalize deviation from the default pose for certain joints.
    current = joint_angles[self._hx_idxs]
    return jp.sum(jp.square(current - self._hx_default_pose) * self._weights)

  def _cost_lin_vel_z(self, global_linvel, gait: jax.Array) -> jax.Array:  # pylint: disable=redefined-outer-name
    # Penalize z axis base linear velocity unless pronk or bound.
    cost = jp.square(global_linvel[2])
    return cost * (gait > 0)

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))
  
  def _cost_foot_slip(self, data: mjx.Data, contact: jax.Array) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact)

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    # Penalize first and second derivative of actions.
    c1 = jp.sum(jp.square(act - last_act))
    c2 = jp.sum(jp.square(act - 2 * last_act + last_last_act))
    return c1 + c2

  def _cost_stand_still(
      self,
      commands: jax.Array,
      joint_angles: jax.Array,
  ) -> jax.Array:
    # Penalize motion at zero commands.
    unit_cmd = commands[:2] / jp.linalg.norm(commands[:2])
    return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
        unit_cmd[1] < 0.1
    )

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _cost_feet_slip(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    left_feet_contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._left_feet_geom_id
    ])
    right_feet_contact = jp.array([
        collision.geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._right_feet_geom_id
    ])
    feet_contact = jp.hstack(
        [left_feet_contact.any(), right_feet_contact.any()]
    )
    return jp.sum(vel_xy_norm_sq * feet_contact)

  def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = (foot_z - self._config.foot_height[1]) ** 2
    return jp.sum(delta * vel_norm)

  def _cost_feet_distance(
      self, data: mjx.Data
  ) -> jax.Array:
    left_foot_pos = data.site_xpos[self._feet_site_id[0]]
    right_foot_pos = data.site_xpos[self._feet_site_id[1]]
    base_xmat = data.site_xmat[self._imu_site_id]
    base_yaw = jp.arctan2(base_xmat[1, 0], base_xmat[0, 0])
    feet_distance = jp.abs(
        jp.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1])
        - jp.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
    )
    return jp.clip(0.25 - feet_distance, min=0.0, max=0.1)

  def _cost_collision(self, data: mjx.Data) -> jax.Array:
    return collision.geoms_colliding(
        data, self._left_foot_box_geom_id, self._right_foot_box_geom_id
    )