"""
Push-Resistant Bipedal Walker (CPU-only, with visualization)

- Env: Gymnasium's BipedalWalker with random external impulses ("pushes").
- Algo: PPO (Stable-Baselines3, PyTorch CPU)
- Viz: saves rollout video to rollout.mp4

Usage (CPU-only):
  # 1) Install deps (Python 3.9+ recommended)
  pip install -U gymnasium[box2d] stable-baselines3 imageio imageio-ffmpeg

  # 2) Train
  python push_resistant_bipedal_rl.py --mode train --timesteps 300000

  # 3) Evaluate & save video
  python push_resistant_bipedal_rl.py --mode play --model_path models/ppo_bipedal_push.zip --video rollout.mp4

Notes:
- Training time on CPU varies (tweak timesteps).
- You can adjust push strength/frequency by CLI flags below.
"""

import argparse
import os
from typing import Optional, Tuple

import gymnasium as gym
from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
    from stable_baselines3.common.callbacks import EvalCallback
except Exception as e:
    raise RuntimeError(
        "stable-baselines3 is required. Install with: pip install -U stable-baselines3"
    ) from e

try:
    import imageio
except Exception as e:
    imageio = None


# -----------------------------
# Custom environment with pushes
# -----------------------------
class PushResistantBipedalWalker(BipedalWalker):
    """BipedalWalker with random external impulses applied to the hull.

    We inject short impulses at random timesteps to simulate external pushes.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        hardcore: bool = False,
        push_prob: float = 0.02,
        push_impulse_range: Tuple[float, float] = (10.0, 60.0),
        push_cooldown: int = 20,
        seed: Optional[int] = None,
    ):
        super().__init__(render_mode=render_mode, hardcore=hardcore)
        self.push_prob = push_prob
        self.push_impulse_range = push_impulse_range
        self.push_cooldown_default = push_cooldown
        self._push_cooldown = 0
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

    def step(self, action):
        # Apply random push before stepping physics
        if self._push_cooldown <= 0 and self.np_random.random() < self.push_prob:
            self._apply_random_push()
            self._push_cooldown = self.push_cooldown_default
        else:
            self._push_cooldown = max(0, self._push_cooldown - 1)

        obs, reward, terminated, truncated, info = super().step(action)
        # Optionally, provide whether a push occurred recently
        info["recent_push"] = int(self._push_cooldown > self.push_cooldown_default - 3)
        return obs, reward, terminated, truncated, info

    # -- helpers --
    def _apply_random_push(self):
        # Choose direction: left or right (x-axis impulse)
        direction = 1.0 if self.np_random.random() < 0.5 else -1.0
        mag = self.np_random.uniform(*self.push_impulse_range)
        # Apply linear impulse to hull at its center of mass
        # Box2D expects impulse in N*s (kg*m/s). Using ApplyLinearImpulse.
        self.hull.ApplyLinearImpulse(impulse=(direction * mag, 0.0), point=self.hull.worldCenter, wake=True)


# -----------------------------
# Utility functions
# -----------------------------

def make_env_factory(args, seed_offset=0):
    def _thunk():
        env = PushResistantBipedalWalker(
            render_mode=None,
            hardcore=args.hardcore,
            push_prob=args.push_prob,
            push_impulse_range=(args.push_min_impulse, args.push_max_impulse),
            push_cooldown=args.push_cooldown,
            seed=args.seed + seed_offset,
        )
        return env

    return _thunk


def train(args):
    os.makedirs("models", exist_ok=True)

    # Vectorized env for faster CPU training
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env_factory(args, i) for i in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env_factory(args, 0)])

    env = VecMonitor(env)

    # PPO hyperparams tuned for CPU
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=args.tb_logdir if args.tb_logdir else None,
        seed=args.seed,
        device="cpu",
    )

    # Eval callback on a single-env instance (deterministic pushes via same seed)
    eval_env = DummyVecEnv([make_env_factory(args, 1000)])
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/best",
        log_path="models/eval",
        eval_freq=max(10000 // args.num_envs, 1000),
        n_eval_episodes=3,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    save_path = os.path.join("models", "ppo_bipedal_push")
    model.save(save_path)
    print(f"Saved model to {save_path}.zip")


def play(args):
    # Create a single env with rendering to RGB frames
    env = PushResistantBipedalWalker(
        render_mode="rgb_array",
        hardcore=args.hardcore,
        push_prob=args.push_prob,
        push_impulse_range=(args.push_min_impulse, args.push_max_impulse),
        push_cooldown=args.push_cooldown,
        seed=args.seed,
    )

    model = PPO.load(args.model_path, device="cpu")

    frames = []
    obs, info = env.reset()
    ep_reward = 0.0
    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += float(reward)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if terminated or truncated:
            break

    print(f"Episode reward: {ep_reward:.2f}, steps: {len(frames)}")

    # Save video
    if args.video:
        if imageio is None:
            raise RuntimeError("imageio is required to write video. Install with: pip install imageio imageio-ffmpeg")
        imageio.mimsave(args.video, frames, fps=50)
        print(f"Saved video to {args.video}")

    env.close()


# -----------------------------
# CLI
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Push-Resistant Bipedal Walker (CPU)")
    p.add_argument("--mode", choices=["train", "play"], default="train")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--num_envs", type=int, default=4, help="parallel envs for training (CPU)")

    # PPO hyperparams (reasonable CPU defaults)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_steps", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.0)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)

    # Env options
    p.add_argument("--hardcore", action="store_true", help="use Hardcore BipedalWalker")
    p.add_argument("--push_prob", type=float, default=0.02, help="probability of a push at each step")
    p.add_argument("--push_min_impulse", type=float, default=10.0)
    p.add_argument("--push_max_impulse", type=float, default=60.0)
    p.add_argument("--push_cooldown", type=int, default=20, help="min steps between pushes")

    # Eval / Viz
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--video", type=str, default="rollout.mp4")
    p.add_argument("--model_path", type=str, default="models/ppo_bipedal_push.zip")
    p.add_argument("--tb_logdir", type=str, default=None)

    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        play(args)


if __name__ == "__main__":
    main()
