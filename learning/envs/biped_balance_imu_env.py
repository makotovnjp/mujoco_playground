# envs/biped_balance_imu_env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv

class BipedBalanceImuEnv(MujocoEnv):
    """
    観測: [ torso_quat(4), torso_angvel(3), prev_torque(n_act) ]
    行動: 各関節トルク（連続）
    目的: 倒れずに直立を維持（外力が来ても復元）
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 240}

    def __init__(
        self,
        xml_path="../../assets/XBot-L-terrain.xml",
        frame_skip=5,
        render_mode=None,
        max_episode_steps=1000,
        push_every_steps=(300, 600),   # 外力をかける間隔（乱数で決定）
        push_force_range=(100.0, 250.0),  # 押しの強さ[N]
        push_duration=10               # 押しを与えるステップ数
    ):
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

        # 外力関連
        self.push_every_steps = push_every_steps
        self.push_force_range = push_force_range
        self.push_duration = push_duration
        self.push_timer = 0
        self.push_force_world = np.zeros(3, dtype=np.float32)

        # 前回トルク
        self.prev_torque = None

        super().__init__(
            model_path=xml_path,
            frame_skip=frame_skip,
            observation_space=None,
            render_mode=render_mode,
            width=960,
            height=540,
        )

        # アクチュエータ数
        self.n_act = int(self.model.nu)

        # 行動空間（MuJoCoのctrlrangeに合わせる）
        cr = self.model.actuator_ctrlrange.copy()
        self.action_space = spaces.Box(low=cr[:, 0], high=cr[:, 1], dtype=np.float32)

        # 観測空間
        self._set_observation_space()

        # ID キャッシュ
        self._torso_id = self.model.body("torso").id

    # ---------- 観測 ----------
    def _get_imu(self):
        # 姿勢（四元数）とワールド角速度
        quat = self.data.xquat[self._torso_id].astype(np.float32)      # (w,x,y,z)
        angvel = self.data.cvel[self._torso_id, 3:].astype(np.float32) # (3,)
        # 正規化
        n = np.linalg.norm(quat)
        if n > 0:
            quat = quat / n
        # 角速度はクリップ（安定化）
        angvel = np.clip(angvel, -50.0, 50.0)
        return quat, angvel

    def _get_obs(self):
        quat, angvel = self._get_imu()
        return np.concatenate([quat, angvel, self.prev_torque], dtype=np.float32)

    def _set_observation_space(self):
        low = np.concatenate([
            -np.ones(4, dtype=np.float32),               # quat
            -np.ones(3, dtype=np.float32) * 50.0,        # angvel clip
            -np.ones(self.model.nu, dtype=np.float32) * np.inf,  # prev torque
        ])
        high = -low
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    # ---------- 外力（プッシュ） ----------
    def _maybe_apply_push(self):
        # 既に押している最中
        if self.push_timer > 0:
            # ワールド座標の外力を torso に適用（1ステップ有効）
            self.data.xfrc_applied[self._torso_id, :3] = self.push_force_world
            self.push_timer -= 1
            if self.push_timer == 0:
                # 終了したら解除
                self.data.xfrc_applied[self._torso_id, :6] = 0.0
            return

        # 次の押し開始タイミングか？
        # 乱数で「次の押しまでの残り」をスケジュール
        if not hasattr(self, "_next_push_at"):
            low, high = self.push_every_steps
            self._next_push_at = self._elapsed_steps + self.np_random.integers(low, high)

        if self._elapsed_steps >= self._next_push_at:
            # 横方向（x-y 平面）にランダムな押し
            mag = float(self.np_random.uniform(*self.push_force_range))
            theta = float(self.np_random.uniform(0, 2*np.pi))
            fx, fy = mag*np.cos(theta), mag*np.sin(theta)
            self.push_force_world = np.array([fx, fy, 0.0], dtype=np.float32)
            self.push_timer = int(self.push_duration)

            # 次回の予定も決める
            low, high = self.push_every_steps
            self._next_push_at = self._elapsed_steps + self.np_random.integers(low, high)

    # ---------- 報酬・終了 ----------
    def _upright_cos(self):
        # 胴体のローカルz軸が世界の+zとどれだけ揃っているか（cosθ）
        # xmat[body] は 3x3 回転行列（行フラット）
        R = self.data.xmat[self._torso_id].reshape(3, 3)
        torso_z_world = R[:, 2]              # 第3列がローカルz軸のワールド表現
        return float(np.clip(torso_z_world[2], -1.0, 1.0))  # = cos(傾き)

    def _compute_reward(self, action):
        # 直立：cos(upright) が 1 に近いほど良い
        upright = self._upright_cos()

        # 胴体高さ（あまり低いと減点）
        height = float(self.data.xipos[self._torso_id, 2])

        # 速度落ち着き（横揺れ抑制のため角速度小を好む）
        _, angvel = self._get_imu()
        angvel_pen = 0.001 * float(np.sum(angvel**2))

        # トルクコスト
        ctrl_cost = 0.001 * float(np.sum(np.square(action)))

        # ボーナス＆ペナルティ
        alive_bonus = 1.0
        upright_bonus = 2.0 * upright                 # [-2, 2]
        height_bonus = 1.0 * np.clip(height - 0.9, 0.0, 0.3)  # モデルに合わせ調整

        reward = alive_bonus + upright_bonus + height_bonus - ctrl_cost - angvel_pen

        info = {
            "upright": upright,
            "height": height,
            "ctrl_cost": ctrl_cost,
            "angvel_pen": angvel_pen,
            "push_timer": int(self.push_timer),
        }
        return float(reward), info

    def _fell_down(self):
        z = float(self.data.xipos[self._torso_id, 2])
        too_low = z < 0.6                   # 倒れた高さのしきい値（モデルに合わせて調整）
        # 姿勢が大きく傾いたら終了（upright<0 = 90度超）
        too_tilt = self._upright_cos() < 0.1
        bad = np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel))
        return bool(too_low or too_tilt or bad)

    # ---------- Gymnasium API ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._elapsed_steps = 0
        self.push_timer = 0
        self.push_force_world[:] = 0.0
        if hasattr(self, "_next_push_at"):
            delattr(self, "_next_push_at")

        # 初期姿勢に微小ノイズ
        qpos = self.init_qpos + self.np_random.uniform(-0.005, 0.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-0.005, 0.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        # 前回トルクゼロ
        self.prev_torque = np.zeros(self.n_act, dtype=np.float32)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self._elapsed_steps += 1

        # たまに外力（押し）をかける
        self._maybe_apply_push()

        # 行動を安全にクリップ
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 物理シミュレーション実行
        self.do_simulation(action, self.frame_skip)

        # 報酬・終了
        reward, info = self._compute_reward(action)
        terminated = self._fell_down()
        truncated = self._elapsed_steps >= self._max_episode_steps

        # 観測更新：直前トルク = 今回のctrl
        self.prev_torque = self.data.ctrl.copy().astype(np.float32)

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
