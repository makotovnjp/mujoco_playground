# mujoco_playground/envs/custom/xml_standing_env.py
import os
import math
import numpy as np
import gymnasium as gym
import mujoco

class HunterStandingEnv(gym.Env):
    """
    任意XMLを MuJoCo で読み込み、観測 = [IMU(加速度,角速度), 全関節角, 全関節角速度]
    行動 = 各関節トルク（clip）で安定立位を学習する最小Env
    """
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        xml_path: str,
        torso_body: str = "torso",      # IMU/姿勢評価に使う胴体ボディ名
        frame_skip: int = 10,
        episode_length: int = 1000,
        push_every: int = 200,          # 外乱インパルスの周期（step数）
        push_strength: float = 80.0,    # 外乱の強さ (N) 方向は水平ランダム
        render_mode=None,
        seed: int | None = None,
    ):
        super().__init__()
        assert os.path.exists(xml_path), f"XML not found: {xml_path}"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        # 胴体ボディID
        self.torso_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, torso_body)
        if self.torso_bid < 0:
            raise ValueError(f"torso body '{torso_body}' not found in XML")

        # アクチュエータ（トルク）本数と制御範囲
        self.nu = self.model.nu
        self.ctrl_range = np.zeros((self.nu, 2))
        if self.model.actuator_ctrlrange.size == 2 * self.nu:
            self.ctrl_range = self.model.actuator_ctrlrange.copy()
        else:
            # XMLで範囲未定義なら ±1 を既定とする
            self.ctrl_range[:, 0] = -1.0
            self.ctrl_range[:, 1] =  1.0

        # 観測次元：IMU(加速度3, 角速度3) + qpos(ndof) + qvel(ndof)
        nq, nv = self.model.nq, self.model.nv
        self.obs_dim = 6 + nq + nv

        self.frame_skip = frame_skip
        self.episode_length = episode_length
        self._t = 0
        self.push_every = push_every
        self.push_strength = push_strength
        self.render_mode = render_mode

        # Gym API: spaces
        high_act = self.ctrl_range[:, 1]
        low_act  = self.ctrl_range[:, 0]
        self.action_space = gym.spaces.Box(low=low_act, high=high_act, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # 初期姿勢（少し膝を曲げて安定）
        self._qpos_init = None
        if nq > 0:
            self._qpos_init = np.zeros(nq, dtype=np.float64)
            # 例：最初の2関節を少し曲げる（必要に応じて調整）
            self._qpos_init[:2] = 0.1

    # ---------- 観測の作成 ----------
    def _get_imu(self):
        """
        XMLに <sensor> accelerometer/gyro が定義されていれば data.sensordata から取り出す。
        なければ代替として、角速度は data.qvel（胴体角速度成分）、
        加速度は重力と躯体の線加速度から計算（簡易）。
        """
        # センサーがあれば使う
        if self.model.nsensordata >= 6:
            s = self.data.sensordata.ravel()
            # 仮に最初の加速度3、次の角速度3をIMUとする（XML側で順番を合わせておくのが無難）
            acc = s[0:3]
            gyro = s[3:6]
            return acc.astype(np.float32), gyro.astype(np.float32)

        # センサー未定義の簡易代替（ワールド→ローカル変換は簡略化）
        # 角速度：胴体ボディの角速度
        angvel = self.data.cvel[self.torso_bid, 3:] if self.data.cvel.shape[0] > self.torso_bid else np.zeros(3)
        # 加速度：重力 + 躯体線加速度（近似）
        gravity = self.model.opt.gravity
        # caccはないため、差分で近似するのはノイズが大きい。簡易に重力のみ
        acc = gravity
        return np.array(acc, dtype=np.float32), np.array(angvel, dtype=np.float32)

    def _obs(self):
        acc, gyro = self._get_imu()
        qpos = self.data.qpos.copy().ravel().astype(np.float32)
        qvel = self.data.qvel.copy().ravel().astype(np.float32)
        return np.concatenate([acc, gyro, qpos, qvel], dtype=np.float32)

    # ---------- 1ステップ ----------
    def step(self, action):
        action = np.clip(action, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        self.data.ctrl[: self.nu] = action

        # 外乱：一定間隔で胴体に水平インパルス
        if self.push_every > 0 and (self._t % self.push_every == 0) and self._t > 0:
            theta = self.rng.uniform(0, 2 * math.pi)
            fx = self.push_strength * math.cos(theta)
            fy = self.push_strength * math.sin(theta)
            mujoco.mj_applyFT(self.model, self.data, np.array([fx, fy, 0.0]), np.zeros(3), self.torso_bid, self.data.xipos[self.torso_bid])

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._t += 1
        obs = self._obs()

        # 報酬：直立 + 静止 + トルク節約
        # 胴体の上向きベクトルと世界Zのコサイン近似（小角近似で roll/pitch を使ってもよい）
        up_world = np.array([0, 0, 1.0])
        # 胴体のz軸方向（ボディ姿勢から）
        torso_xmat = self.data.xmat[self.torso_bid].reshape(3, 3)
        z_axis = torso_xmat[:, 2]
        upright = float(np.clip(z_axis.dot(up_world), -1.0, 1.0))  # 1が直立

        linvel = np.linalg.norm(self.data.cvel[self.torso_bid, :3]) if self.data.cvel.shape[0] > self.torso_bid else 0.0
        angvel = np.linalg.norm(self.data.cvel[self.torso_bid, 3:]) if self.data.cvel.shape[0] > self.torso_bid else 0.0
        torque_pen = 1e-4 * float(np.sum(np.square(action)))

        reward = 1.0 * upright - 0.05 * linvel - 0.02 * angvel - torque_pen

        # 終了条件：過度に傾いた／転倒（高さが閾値未満）
        height = self.data.xipos[self.torso_bid, 2]
        terminated = (upright < 0.5) or (height < 0.3)
        truncated = (self._t >= self.episode_length)

        info = {"upright": upright, "height": height}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        if self._qpos_init is not None:
            self.data.qpos[: len(self._qpos_init)] = self._qpos_init
        # ほんの少し姿勢をゆらがせる
        if self.model.nq > 0:
            self.data.qpos[:] += 0.01 * self.rng.standard_normal(self.model.nq)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self._t = 0
        return self._obs(), {}

    # 任意：動画化のためのレンダリング
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        # 必要に応じて mjv/mjr で実装（省略）
        return None

    def close(self):
        pass
