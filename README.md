# LunarLander 환경에서 장애물 회피 착륙을 위한 강화학습 알고리즘 비교

## 1. 프로젝트 배경 및 목표

LunarLander는 OpenAI Gym/Gymnasium에서 제공하는 대표적인 강화학습 환경으로, 
달 착륙선을 조종하여 지정된 착륙 지점에 안전하게 착륙시키는 문제이다. 
에이전트는 착륙선의 위치, 속도, 각도, 각속도, 다리 접촉 여부 등의 상태 정보를 바탕으로 엔진을 제어하며, 안정적으로 착륙할수록 높은 보상을 받는다.

기본 LunarLander 환경은 장애물이 없는 단순 착륙 문제이기 때문에, 환경의 난이도를 높이기 위해 공중에 작은 장애물 3개를 추가하였다. 
에이전트는 착륙 과정에서 장애물과 충돌하지 않도록 회피하면서, 최종적으로 착륙 지점에 안전하게 도달해야 한다.

수정된 LunarLander 환경에서 강화학습 알고리즘을 적용하여 장애물 회피 및 착륙 성능을 분석한다. 
최종적으로 DQN, PPO, A2C 알고리즘의 학습 결과를 비교하여 장애물 회피 착륙 문제에 가장 적합한 알고리즘을 평가한다.

## 2. Environment Modification

The original `LunarLander` source code was modified to create a custom environment named:

```python
LunarLanderObstacle-v0
```

The main modifications are:

* Added three small static obstacles in the air
* Added obstacle collision detection
* Added obstacle-relative position information to the observation space
* Applied a large penalty when the lander collides with an obstacle
* Evaluated whether the agent can avoid obstacles and land successfully

### Observation Space

The original LunarLander observation consists of 8 values:

* Lander x-position
* Lander y-position
* x-velocity
* y-velocity
* angle
* angular velocity
* left leg contact
* right leg contact

In this project, obstacle-relative position values are added.

For three obstacles:

```text
obstacle 1 relative x, y
obstacle 2 relative x, y
obstacle 3 relative x, y
```

Therefore, the modified observation space has:

```text
8 original values + 6 obstacle-relative values = 14 values
```

### Action Space

This project mainly uses the discrete action version of LunarLander.

```text
0: Do nothing
1: Fire left engine
2: Fire main engine
3: Fire right engine
```

Since the action space is discrete, the main algorithms compared are:

* DQN
* PPO
* A2C

## 3. Algorithms

### DQN

DQN is a value-based reinforcement learning algorithm suitable for discrete action spaces. Since the modified LunarLander environment uses `Discrete(4)` actions, DQN is a natural baseline algorithm.

Best DQN hyperparameters:

```python
learning_rate = 3e-5
exploration_fraction = 0.35
exploration_final_eps = 0.02
target_update_interval = 2000
batch_size = 128
```

### PPO

PPO is a policy-gradient-based algorithm that can be used with both discrete and continuous action spaces. PPO limits excessive policy updates using clipping, which helps stabilize training.

Best PPO hyperparameters:

```python
learning_rate = 5e-4
ent_coef = 0.01
n_steps = 4096
clip_range = 0.2
n_epochs = 10
batch_size = 64
```

### A2C

A2C is a simpler policy-gradient-based algorithm. It was used as a baseline comparison with DQN and PPO.

Best A2C hyperparameters:

```python
learning_rate = 7e-4
n_steps = 32
ent_coef = 0.02
gae_lambda = 0.95
```

## 4. Project Structure

Recommended repository structure:

```text
lunar_lander_obstacle_rl/
│
├── lunar_lander_obstacle.py      # Custom LunarLander environment with obstacles
├── train_lunar_obstacle.py       # Training script for DQN, PPO, A2C
├── evaluate_lunar_obstacle.py    # Evaluation script
├── results/                      # Saved models and logs
│   ├── DQN_lunar_obstacle_final.zip
│   ├── PPO_lunar_obstacle_final.zip
│   └── A2C_lunar_obstacle_final.zip
│
├── README.md
└── requirements.txt
```

## 5. Installation

Install the required packages:

```bash
pip install gymnasium[box2d]
pip install stable-baselines3
pip install numpy
pip install pygame
```

If Box2D installation fails, install `swig` first:

```bash
pip install swig
pip install gymnasium[box2d]
```

## 6. Training

Run the training script:

```bash
python train_lunar_obstacle.py
```

In the training script, choose the algorithm by changing:

```python
train_model(
    algo_name="PPO",
    total_timesteps=1000000
)
```

Available algorithm names:

```python
"DQN"
"PPO"
"A2C"
```

Example:

```python
train_model(
    algo_name="DQN",
    total_timesteps=1000000
)
```

## 7. Evaluation

Run the evaluation script:

```bash
python evaluate_lunar_obstacle.py
```

To evaluate a specific algorithm, change:

```python
ALGO = "PPO"
```

Available options:

```python
ALGO = "DQN"
ALGO = "PPO"
ALGO = "A2C"
```

The evaluation script measures:

* Mean reward
* Reward standard deviation
* Success rate
* Obstacle collision rate

## 8. Evaluation Results

The final evaluation was conducted using 100 episodes.

| Algorithm | Mean Reward | Reward Std | Success Rate | Obstacle Collision Rate |
| --------- | ----------: | ---------: | -----------: | ----------------------: |
| DQN       |      112.89 |     134.85 |         0.80 |                    0.20 |
| PPO       |      273.80 |      22.11 |         1.00 |                    0.00 |
| A2C       |       60.08 |     170.44 |         0.54 |                    0.05 |

## 9. Result Analysis

Among the three algorithms, PPO showed the best performance.

PPO achieved:

* Highest mean reward
* Lowest reward standard deviation
* 100% success rate
* 0% obstacle collision rate

DQN also learned a partially successful policy, achieving an 80% success rate. However, its reward standard deviation was high, meaning that some episodes succeeded while others failed due to collision or unstable landing.

A2C showed the lowest stability among the three algorithms. Although it sometimes succeeded, its reward standard deviation was high and its success rate was lower than DQN and PPO.

Overall, PPO was the most suitable algorithm for the obstacle avoidance LunarLander task.

## 10. Conclusion

This project modified the original LunarLander environment by adding aerial obstacles and obstacle-relative observation features. The agent was required to avoid obstacles while safely landing on the landing pad.

The experiment compared DQN, PPO, and A2C in the discrete action environment.

Final conclusion:

```text
PPO > DQN > A2C
```

PPO achieved the most stable and successful landing performance, making it the best algorithm for this modified obstacle avoidance LunarLander environment.

## 11. Future Work

Future improvements may include:

* Randomizing obstacle positions for better generalization
* Increasing the number of obstacles
* Adding moving obstacles
* Comparing continuous action algorithms such as SAC, TD3, and DDPG
* Using `continuous=True` to test continuous-control LunarLander
* Adding more detailed metrics such as landing accuracy, fuel usage, and crash rate

## 12. Notes on Discrete and Continuous Action Spaces

The current project uses the discrete version of LunarLander.

For discrete action spaces:

```text
DQN, PPO, A2C
```

are suitable.

For continuous action spaces:

```text
SAC, TD3, DDPG
```

are more suitable.

To use the continuous version, the environment can be created with:

```python
env = gym.make(
    "LunarLanderObstacle-v0",
    continuous=True,
    render_mode=None
)
```

In that case, the action space changes from:

```text
Discrete(4)
```

to:

```text
Box(-1, 1, shape=(2,))
```

and the agent directly controls the engine power using continuous values.
