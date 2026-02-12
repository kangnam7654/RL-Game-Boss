"""PPO 학습 스크립트"""
from __future__ import annotations

import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback,
)

# SubprocVecEnv의 워커에서 임포트가 가능하도록 경로 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def make_env(rank: int, player_difficulty: str = 'medium'):
    def _init():
        from ai.boss_env import BossEnv
        env = BossEnv(render_mode=None, player_policy=player_difficulty)
        return env
    return _init


class LoggingCallback(BaseCallback):
    """에피소드 통계 로깅"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_count += 1
                if self.episode_count % 100 == 0:
                    ep = info['episode']
                    print(f"[Episode {self.episode_count}] "
                          f"reward={ep['r']:.2f}, length={ep['l']}")
        return True


def train(
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    save_path: str = 'models/',
    player_difficulty: str = 'medium',
    resume_path: str | None = None,
):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'best'), exist_ok=True)

    print(f"=== Boss AI PPO Training ===")
    print(f"  timesteps: {total_timesteps:,}")
    print(f"  envs: {n_envs}")
    print(f"  player difficulty: {player_difficulty}")
    print(f"  save path: {save_path}")

    # 병렬 환경
    envs = SubprocVecEnv([make_env(i, player_difficulty) for i in range(n_envs)])
    envs = VecMonitor(envs)

    # 평가 환경
    eval_env = SubprocVecEnv([make_env(0, 'medium')])
    eval_env = VecMonitor(eval_env)

    if resume_path:
        print(f"  Resuming from: {resume_path}")
        model = PPO.load(resume_path, env=envs)
    else:
        model = PPO(
            "MlpPolicy",
            envs,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            verbose=1,
            tensorboard_log="./tb_logs/",
            device='auto',
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            ),
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=save_path,
        name_prefix='boss_ppo',
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, 'best'),
        eval_freq=max(20_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )
    log_cb = LoggingCallback()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList([checkpoint_cb, eval_cb, log_cb]),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")

    model.save(os.path.join(save_path, 'boss_ppo_final'))
    print(f"Model saved to {save_path}/boss_ppo_final.zip")

    envs.close()
    eval_env.close()


if __name__ == '__main__':
    train()
