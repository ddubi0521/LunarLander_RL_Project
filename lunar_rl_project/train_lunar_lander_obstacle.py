import os
import gymnasium as gym
from gymnasium.envs.registration import register

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# 커스텀 환경 파일
import lunar_lander_obstacle

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 커스텀 환경 등록
try:
    register(
        id="LunarLanderObstacle-v0",
        entry_point="lunar_lander_obstacle:LunarLander",
        max_episode_steps=1000,
    )
except Exception as e:
    print("Register warning:", e)


# 환경 생성 함수
def make_env(render_mode=None):
    env = gym.make(
        "LunarLanderObstacle-v0",
        continuous=False,
        render_mode=render_mode
    )
    env = Monitor(env)
    return env


# 학습 함수
def train_model(algo_name="DQN", total_timesteps=50000):
    algorithms = {
        "DQN": DQN,
        "PPO": PPO,
        "A2C": A2C,
    }

    AlgoClass = algorithms[algo_name]

    train_env = make_env(render_mode=None)
    eval_env = make_env(render_mode=None)

    print("Observation space:", train_env.observation_space)
    print("Action space:", train_env.action_space)

    # 평가 callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{RESULT_DIR}/{algo_name}_best",
        log_path=f"{RESULT_DIR}/{algo_name}_logs",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # 알고리즘별 기본 설정
    if algo_name == "DQN":
        model = DQN(
            policy="MlpPolicy", # policy 모델
            env=train_env, # 환경
            learning_rate=3e-5, # 학습률
            gamma=0.99, # discount factor 값

            exploration_fraction=0.35, # 초기 10%의 학습단계 동안 exploration에 더 많은 초점
            exploration_initial_eps=1.0, # exploration을 위한 초기값
            exploration_final_eps=0.02, # exploration을 위한 최종값
            buffer_size=200_000, # experience replay memory 크기
            learning_starts=20_000, # 학습을 하기 전 random action으로 experience data를 수집할 step 수
            batch_size=128, # 각 학습단계에서 사용할 batch의 크기
            target_update_interval=2000, # 몇 번의 step이 지난 후에 Qtarget을 Q로 복사할 것인지 결정하기 위한 C-step 수
            train_freq=4, # 몇 번의 interaction이 있은 후에 training을 할지 결정하는 변수
            gradient_steps=1, # 한 training당 몇 번의 gradient update를 수행할지 결정하는 gradient step 수
            max_grad_norm=10, # gradient explosion 방지를 위한 gradient clipping 최대값

            tensorboard_log=f"{RESULT_DIR}/tensorboard", # tensorboard 사용 시 log 데이터를 저장할 디렉토리
            verbose=1, # 학습 과정 중 message를 print 할지 말지
            seed=42, # seed 값
            device='cpu', # 코드가 구동되는 device
        )

    elif algo_name == "PPO":
        model = PPO(
            policy="MlpPolicy", # policy 모델
            env=train_env, # 환경
            learning_rate=5e-4, # 학습률
            gamma=0.99, # discount factor 값

            n_steps=4096, # 각 환경마다 몇 step으로 rollout을 하고 gradient update를 할지 결정하는 파라미터
            gae_lambda=0.95, # TD의 lambda를 지정하기 위한 파라미터
            vf_coef=0.5, # value loss가 얼마나 중요한지를 결정하는 가중치
            ent_coef=0.01, # exploration을 위한 entropy 가중치
            normalize_advantage=True, # advantage normalization
            max_grad_norm=0.5, # gradient explosion 방지를 위한 gradient clipping 최대값
            n_epochs=10, # 환경에서 10 step 동안 데이터를 모음
            clip_range=0.2, # 새 정책이 이전 정책보다 너무 많이 변하지 않도록 +/- 20% 범위에서 제한

            batch_size=64, # 각 학습단계에서 사용할 batch의 크기
            tensorboard_log=f"{RESULT_DIR}/tensorboard", # tensorboard 사용 시 log 데이터를 저장할 디렉토리
            verbose=1, # 학습 과정 중 message를 print 할지 말지
            seed=42, # seed 값
            device='cpu', # 코드가 구동되는 device
        )

    elif algo_name == "A2C":
        model = A2C(
            policy="MlpPolicy", # policy 모델
            env=train_env, # 환경
            learning_rate=7e-4, # 학습률
            gamma=0.99, # discount factor 값

            n_steps=32, # 각 환경마다 몇 step으로 rollout을 하고 gradient update를 할지 결정하는 파라미터
            gae_lambda=0.95, # TD의 lambda를 지정하기 위한 파라미터
            vf_coef=0.5, # value loss가 얼마나 중요한지를 결정하는 가중치
            ent_coef=0.02, # exploration을 위한 entropy 가중치
            use_rms_prop=True, # optimizer로 RMSProp을 사용할지 Adam을 사용할지 결정
            rms_prop_eps=1e-5, # RMSProp 최적화 알고리즘에서 사용되는 상수(epsilon) 빈도를 설정하는 변수
            normalize_advantage=True, # advantage normalization
            max_grad_norm=0.5, # gradient explosion 방지를 위한 gradient clipping 최대값

            tensorboard_log=f"{RESULT_DIR}/tensorboard", # tensorboard 사용 시 log 데이터를 저장할 디렉토리
            verbose=1, # 학습 과정 중 message를 print 할지 말지
            seed=42, # seed 값
            device='cpu', # 코드가 구동되는 device
        )

    print(f"\n===== Training {algo_name} =====")

    model.learn(
        total_timesteps=total_timesteps, # 에이전트가 학습을 진행할 동안 수행할 총 타임스텝의 수
        callback=eval_callback, # 학습 과정 중 특정조건 만족 시 호출되는 callback 함수
        tb_log_name=f"{algo_name}_LunarObstacle", # tensorboard에 표기될 log 이름
        reset_num_timesteps=True, # reset 시 0으로 초기화
    )

    final_path = f"{RESULT_DIR}/{algo_name}_lunar_obstacle_final"
    model.save(final_path)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
    )

    print("\n===== 학습 완료 =====")
    print(f"Algorithm: {algo_name}")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Saved model: {final_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    # total_timesteps=50000
    # total_timesteps=200000
    # total_timesteps=500000
    # total_timesteps=1000000

    # 처음에는 DQN 추천
    train_model(
        algo_name="A2C",
        total_timesteps=1000000
    )