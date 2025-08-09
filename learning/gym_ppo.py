# train_biped_imu_torque_ppo.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# --- 観測を「IMU(胴体姿勢+角速度)+直前トルク」にするラッパー ---
class ImuAndPrevTorqueObs(gym.Wrapper):
    """
    ベース環境: Humanoid-v4 (MuJoCo)
    観測 = [ 胴体クォータニオン(4), 胴体角速度(3), 直前トルク(n_act) ]
    行動 = ベース環境と同じ（連続トルク）
    """
    def __init__(self, env):
        super().__init__(env)
        # アクチュエータ数
        self.n_act = int(self.env.action_space.shape[0])
        # 直前トルク（直前のctrl値をそのまま保持）
        self.prev_torque = np.zeros(self.n_act, dtype=np.float32)

        # 観測空間を定義：クォータニオン(4)+角速度(3)+直前トルク(n_act)
        low = np.concatenate([
            -np.ones(4, dtype=np.float32),        # クォータニオンは[-1,1]に収まる
            -np.full(3, np.inf, dtype=np.float32),# 角速度は無限に近い範囲
            -np.ones(self.n_act, dtype=np.float32) * np.inf
        ])
        high = np.concatenate([
            np.ones(4, dtype=np.float32),
            np.full(3, np.inf, dtype=np.float32),
            np.ones(self.n_act, dtype=np.float32) * np.inf
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 胴体(body)の名前（Humanoidは"torso"）
        self.torso_body_name = "torso"
        # body id を事前に取得（reset後でmodel/dataが有効）
        self._torso_id_cached = None

    # MuJoCo の model/data ハンドル取得ヘルパ
    @property
    def _model(self):
        return self.env.unwrapped.model  # gymnasium.mujoco.MujocoEnv 互換

    @property
    def _data(self):
        return self.env.unwrapped.data

    def _get_torso_id(self):
        if self._torso_id_cached is None:
            self._torso_id_cached = self._model.body(self.torso_body_name).id
        return self._torso_id_cached

    def _get_imu_obs(self):
        """
        IMU相当:
          - 胴体の世界姿勢クォータニオン: xquat[body_id] -> (w,x,y,z) 形
          - 胴体の角速度: cvel[body_id, 3:] (ワールド座標系の角速度)
        """
        torso_id = self._get_torso_id()
        quat = self._data.xquat[torso_id].astype(np.float32)            # (4,)
        angvel = self._data.cvel[torso_id, 3:].astype(np.float32)       # (3,)
        # 正規化（念のため）
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat = quat / norm
        return quat, angvel

    def _build_obs(self):
        quat, angvel = self._get_imu_obs()
        return np.concatenate([quat, angvel, self.prev_torque], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        # reset時は prev_torque をゼロに
        self.prev_torque = np.zeros(self.n_act, dtype=np.float32)
        # MuJoCo の内部状態からIMU観測を作る
        return self._build_obs(), info

    def step(self, action):
        # ベース環境にそのまま行動を渡す（Humanoidはトルク/目標値）
        obs, reward, terminated, truncated, info = self.env.step(action)
        # 今ステップの制御値を「直前トルク」として保存
        # MuJoCo の data.ctrl は現在の制御入力
        self.prev_torque = np.copy(self._data.ctrl).astype(np.float32)
        # 新しい観測（IMU+prev_torque）を返す
        return self._build_obs(), reward, terminated, truncated, info


def make_env(seed_offset=0):
    def _thunk():
        # ベース環境
        env = gym.make("Humanoid-v4")  # 2Dで良ければ Walker2d-v4 でもOK
        env.reset(seed=42 + seed_offset)
        # 観測ラッパ
        env = ImuAndPrevTorqueObs(env)
        return env
    return _thunk


def main():
    # 並列環境（CPU数に合わせて）
    n_envs = 4
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)

    # PPO ハイパーパラメータは控えめ（安定寄り）
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=4096 // n_envs,   # 1更新あたりの各環境ステップ数
        batch_size=1024,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        n_epochs=10,
        device="auto",
        verbose=1,
    )

    # 学習
    total_timesteps = 2_000_000  # まずは ~2M から
    model.learn(total_timesteps=total_timesteps)

    # 保存
    model.save("ppo_humanoid_imu_prevtorque")
    vec_env.close()


if __name__ == "__main__":
    main()
