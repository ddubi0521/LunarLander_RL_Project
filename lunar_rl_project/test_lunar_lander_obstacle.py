import time
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

from stable_baselines3 import DQN, PPO, A2C

import lunar_lander_obstacle

RESULT_DIR = "results"

try:
    register(
        id="LunarLanderObstacle-v0",
        entry_point="lunar_lander_obstacle:LunarLander",
        max_episode_steps=1000,
    )
except Exception as e:
    print("Register warning:", e)


def make_env(render_mode="human"):
    env = gym.make(
        "LunarLanderObstacle-v0",
        continuous=False,
        render_mode=render_mode
    )
    return env


if __name__ == "__main__":
    ALGO = "A2C"
    MODEL_PATH = f"{RESULT_DIR}/{ALGO}_lunar_obstacle_final.zip"

    algorithms = {
        "DQN": DQN,
        "PPO": PPO,
        "A2C": A2C,
    }

    env = make_env(render_mode=None)
    model = algorithms[ALGO].load(MODEL_PATH, env=env)

    n_episodes = 100
    success_count = 0
    obstacle_collision_count = 0
    total_rewards = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if terminated or truncated:
                base_env = env.unwrapped

                obstacle_contact = base_env.obstacle_contact
                game_over = base_env.game_over

                # 현재 reward 구조 기준:
                # obstacle_contact면 장애물 충돌 실패
                # obstacle_contact가 아니고 total_reward가 양수면 성공으로 간주
                success = (not obstacle_contact) and (total_reward > 0)

                if success:
                    success_count += 1

                if obstacle_contact:
                    obstacle_collision_count += 1

                total_rewards.append(total_reward)

                print(
                    f"Episode {ep + 1} | "
                    f"total_reward={total_reward:.2f} | "
                    f"success={success} | "
                    f"obstacle_contact={obstacle_contact} | "
                    f"game_over={game_over}"
                )

                time.sleep(1.0)
                break

    print("\n===== Evaluation Result =====")
    print(f"Mean reward: {np.mean(total_rewards):.2f}")
    print(f"Reward std: {np.std(total_rewards):.2f}")
    print(f"Success rate: {success_count / n_episodes:.2f}")
    print(f"Obstacle collision rate: {obstacle_collision_count / n_episodes:.2f}")

    env.close()