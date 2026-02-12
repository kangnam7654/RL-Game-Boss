"""Boss AI - Top-Down Action Game 진입점"""
import argparse
import sys
import os

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description='Boss AI - Top-Down Action Game with Reinforcement Learning')
    parser.add_argument('mode', choices=['train', 'play', 'eval'],
                        help='실행 모드: train(학습), play(플레이), eval(평가)')
    parser.add_argument('--model', type=str, default='models/best/best_model.zip',
                        help='학습된 모델 경로 (play/eval 모드)')
    parser.add_argument('--timesteps', type=int, default=2_000_000,
                        help='학습 총 타임스텝')
    parser.add_argument('--difficulty', choices=['easy', 'medium', 'hard'],
                        default='medium', help='학습 시 플레이어 난이도')
    parser.add_argument('--n-envs', type=int, default=8,
                        help='병렬 환경 수')
    parser.add_argument('--resume', type=str, default=None,
                        help='학습 재개 시 모델 경로')
    args = parser.parse_args()

    if args.mode == 'train':
        from ai.train import train
        train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            player_difficulty=args.difficulty,
            resume_path=args.resume,
        )

    elif args.mode == 'play':
        from modes.play_mode import PlayMode
        PlayMode(model_path=args.model).run()

    elif args.mode == 'eval':
        _run_eval(args.model)


def _run_eval(model_path: str):
    """학습된 모델 평가"""
    from stable_baselines3 import PPO
    from ai.boss_env import BossEnv
    import numpy as np

    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"Model not found: {model_path}")
        return

    env = BossEnv(render_mode=None, player_policy='medium')
    n_episodes = 50
    wins = 0
    total_reward = 0.0
    total_steps = 0

    print(f"Evaluating {model_path} over {n_episodes} episodes...")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        if env.world.player.health.current <= 0:
            wins += 1
        total_reward += ep_reward
        total_steps += env.world.step_count

    env.close()

    print(f"\n=== Evaluation Results ===")
    print(f"  Win rate:      {wins}/{n_episodes} ({100*wins/n_episodes:.1f}%)")
    print(f"  Avg reward:    {total_reward/n_episodes:.2f}")
    print(f"  Avg steps:     {total_steps/n_episodes:.0f}")


if __name__ == '__main__':
    main()
