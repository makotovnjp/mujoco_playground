# train_balance.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from envs.biped_balance_imu_env import BipedBalanceImuEnv

def make_env(seed_offset=0):
    def _thunk():
        env = BipedBalanceImuEnv(
            xml_path="./../assets/XBot-L-terrain.xml",   # あなたの2足モデル
            frame_skip=5,
            max_episode_steps=1000,
            push_every_steps=(300, 600),
            push_force_range=(120.0, 250.0),
            push_duration=10,
        )
        env.reset(seed=42 + seed_offset)
        return env
    return _thunk

if __name__ == "__main__":
    n_envs = 1
    vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=2048 // n_envs,
        batch_size=1024,
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        n_epochs=10,
        vf_coef=0.5,
        device="auto",
        verbose=1,
    )

    model.learn(total_timesteps=2_000_000)
    model.save("ppo_biped_balance_imu")
    vec_env.close()
