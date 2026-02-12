"""교대 학습 (Co-training) - 보스와 플레이어를 번갈아 학습하며 목표 클리어율 수렴"""
from __future__ import annotations

import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def make_player_env(rank: int, boss_model_path: str | None):
    def _init():
        from ai.player_env import PlayerEnv
        return PlayerEnv(render_mode=None, boss_model_path=boss_model_path)
    return _init


def make_boss_env(rank: int, player_model_path: str | None, kill_reward_coef: float):
    def _init():
        from ai.boss_env_vs_player import BossEnvVsPlayer
        return BossEnvVsPlayer(
            render_mode=None,
            player_model_path=player_model_path,
            kill_reward_coef=kill_reward_coef,
        )
    return _init


def train_player(boss_model_path: str | None, timesteps: int = 500_000,
                 n_envs: int = 8, save_dir: str = 'models/player/',
                 resume_path: str | None = None) -> str:
    """플레이어 학습. 반환: 저장된 모델 경로"""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'best'), exist_ok=True)

    envs = SubprocVecEnv([make_player_env(i, boss_model_path) for i in range(n_envs)])
    envs = VecMonitor(envs)

    if resume_path:
        model = PPO.load(resume_path, env=envs)
    else:
        model = PPO(
            "MultiInputPolicy" if False else "MlpPolicy",
            envs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./tb_logs/",
            device='auto',
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

    model.learn(
        total_timesteps=timesteps,
        callback=CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=save_dir,
            name_prefix='player_ppo',
        ),
        progress_bar=True,
    )

    path = os.path.join(save_dir, 'player_latest.zip')
    model.save(path)
    envs.close()
    return path


def train_boss_vs_player(player_model_path: str, kill_reward_coef: float,
                         timesteps: int = 500_000, n_envs: int = 8,
                         save_dir: str = 'models/boss_co/',
                         resume_path: str | None = None) -> str:
    """보스 재학습 (학습된 플레이어 상대). 반환: 저장된 모델 경로"""
    os.makedirs(save_dir, exist_ok=True)

    envs = SubprocVecEnv([
        make_boss_env(i, player_model_path, kill_reward_coef)
        for i in range(n_envs)
    ])
    envs = VecMonitor(envs)

    if resume_path:
        model = PPO.load(resume_path, env=envs)
    else:
        model = PPO(
            "MlpPolicy",
            envs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./tb_logs/",
            device='auto',
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

    model.learn(
        total_timesteps=timesteps,
        callback=CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=save_dir,
            name_prefix='boss_co_ppo',
        ),
        progress_bar=True,
    )

    path = os.path.join(save_dir, 'boss_latest.zip')
    model.save(path)
    envs.close()
    return path


def evaluate(player_model_path: str, boss_model_path: str | None,
             n_episodes: int = 100) -> float:
    """플레이어 클리어율 측정"""
    from ai.player_env import PlayerEnv

    env = PlayerEnv(render_mode=None, boss_model_path=boss_model_path)
    player_model = PPO.load(player_model_path)

    wins = 0
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = player_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        if info.get('player_won', False):
            wins += 1

    env.close()
    return wins / n_episodes


def co_train(
    target_winrate: float = 0.5,
    rounds: int = 5,
    steps_per_round: int = 500_000,
    n_envs: int = 8,
    initial_boss_path: str = 'models/best/best_model.zip',
):
    """교대 학습 메인 루프"""
    boss_path = initial_boss_path
    player_path = None
    kill_lambda = 1.0

    print(f"=== Co-Training ===")
    print(f"  Target win rate: {target_winrate:.0%}")
    print(f"  Rounds: {rounds}")
    print(f"  Steps/round: {steps_per_round:,}")
    print(f"  Initial boss: {boss_path}")
    print()

    for rnd in range(rounds):
        print(f"{'='*50}")
        print(f"ROUND {rnd + 1}/{rounds}")
        print(f"{'='*50}")

        # Phase A: 플레이어 학습
        print(f"\n[Phase A] Training Player (vs boss)")
        player_path = train_player(
            boss_model_path=boss_path,
            timesteps=steps_per_round,
            n_envs=n_envs,
            save_dir=f'models/player/round_{rnd}/',
        )

        # 클리어율 측정
        win_rate = evaluate(player_path, boss_path, n_episodes=100)
        print(f"\n  >>> Win rate: {win_rate:.1%} (target: {target_winrate:.1%})")

        # λ 조정
        error = win_rate - target_winrate
        kill_lambda += 0.5 * error  # 조정 폭
        kill_lambda = max(0.1, min(5.0, kill_lambda))
        print(f"  >>> λ adjusted: {kill_lambda:.3f}")

        # Phase B: 보스 재학습
        print(f"\n[Phase B] Training Boss (vs player, λ={kill_lambda:.3f})")
        boss_path = train_boss_vs_player(
            player_model_path=player_path,
            kill_reward_coef=kill_lambda,
            timesteps=steps_per_round,
            n_envs=n_envs,
            save_dir=f'models/boss_co/round_{rnd}/',
        )

        # 최종 클리어율
        final_wr = evaluate(player_path, boss_path, n_episodes=100)
        print(f"\n  >>> Final win rate this round: {final_wr:.1%}")
        print(f"  >>> λ = {kill_lambda:.3f}")

    print(f"\n{'='*50}")
    print(f"Co-Training Complete!")
    print(f"  Final boss: {boss_path}")
    print(f"  Final player: {player_path}")
    print(f"  λ = {kill_lambda:.3f}")


if __name__ == '__main__':
    co_train()
