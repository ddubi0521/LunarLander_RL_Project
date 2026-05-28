# 🚀 LunarLander Obstacle Avoidance with Reinforcement Learning

> Gymnasium Box2D LunarLander 환경을 수정하여 공중 장애물이 포함된 착륙 환경을 만들고,
> DQN, PPO, A2C 강화학습 알고리즘의 장애물 회피 및 착륙 성능을 비교한 프로젝트입니다.

---

## 📌 1. Project Overview

`LunarLander`는 OpenAI Gym/Gymnasium에서 제공하는 대표적인 강화학습 환경으로,
달 착륙선을 조종하여 지정된 착륙 지점에 안전하게 착륙시키는 문제입니다.

기본 LunarLander 환경에서는 장애물이 존재하지 않기 때문에, 에이전트는 단순히 착륙 지점에 안정적으로 도달하는 것만 학습합니다.
본 프로젝트에서는 원본 Box2D LunarLander 환경을 수정하여 **공중에 작은 장애물 3개**를 추가하였습니다.

따라서 에이전트는 다음 두 가지 목표를 동시에 달성해야 합니다.

* 공중 장애물과 충돌하지 않도록 회피
* 최종적으로 착륙 지점에 안전하게 착륙

본 프로젝트의 최종 목표는 수정된 LunarLander 환경에서 **DQN, PPO, A2C** 알고리즘을 학습시키고,
각 알고리즘의 장애물 회피 및 착륙 성능을 비교하여 가장 적합한 알고리즘을 평가하는 것입니다.

---

## 🛠️ 2. Environment Modification

본 프로젝트에서는 원본 Box2D LunarLander 코드를 수정하여
기존의 단순 착륙 문제를 **장애물 회피 착륙 문제**로 확장했습니다.

### 주요 수정 사항

| 수정 항목  | 내용                            |
| ------ | ----------------------------- |
| 장애물 생성 | 공중에 작은 Box2D 정적 장애물 3개 배치     |
| 관측값 수정 | 기존 상태값에 장애물의 상대 위치 정보 추가      |
| 충돌 처리  | 착륙선 본체 또는 다리가 장애물과 충돌하면 실패 처리 |
| 보상 설계  | 장애물 충돌 시 큰 패널티 부여             |

---

## 🌕 3. Modified LunarLander Environment

### 기존 LunarLander 상태값

기본 LunarLander의 observation은 다음 8개의 값으로 구성됩니다.

| 상태값               | 설명               |
| ----------------- | ---------------- |
| x position        | 착륙선의 x축 위치       |
| y position        | 착륙선의 y축 위치       |
| x velocity        | x축 속도            |
| y velocity        | y축 속도            |
| angle             | 착륙선의 각도          |
| angular velocity  | 각속도              |
| left leg contact  | 왼쪽 다리의 지면 접촉 여부  |
| right leg contact | 오른쪽 다리의 지면 접촉 여부 |

### 수정된 상태값

본 프로젝트에서는 장애물 3개에 대해 각각 상대 위치 정보를 추가했습니다.

```text
obstacle 1: relative x, relative y
obstacle 2: relative x, relative y
obstacle 3: relative x, relative y
```

따라서 최종 observation은 다음과 같이 구성됩니다.

```text
기존 상태값 8개 + 장애물 상대 위치 6개 = 총 14차원 observation
```

---

## 🎮 4. Action Space

본 프로젝트에서는 LunarLander의 **이산형 action space**를 사용했습니다.

```text
0: 아무것도 하지 않음
1: 왼쪽 방향 엔진 점화
2: 메인 엔진 점화
3: 오른쪽 방향 엔진 점화
```

즉, 에이전트는 매 step마다 4개의 행동 중 하나를 선택합니다.

```text
Action Space: Discrete(4)
```

따라서 본 프로젝트에서는 이산형 action space에 적용 가능한 알고리즘인
**DQN, PPO, A2C**를 비교했습니다.

---

## 🧠 5. Algorithms

## 5.1 DQN

DQN은 이산형 행동 공간에 적합한 대표적인 value-based 강화학습 알고리즘입니다.
현재 환경은 `Discrete(4)` action space를 사용하기 때문에 DQN을 기본 비교 알고리즘으로 사용했습니다.

### Best DQN Hyperparameters

```python
learning_rate = 3e-5
exploration_fraction = 0.35
exploration_final_eps = 0.02
target_update_interval = 2000
batch_size = 128
```

### DQN Best Result

| Metric         |           Value |
| -------------- | --------------: |
| ep_rew_mean    |             122 |
| episode_reward |  202.35 ± 34.76 |
| Episode length | 623.30 ± 132.73 |
| Mean reward    |  85.86 ± 139.85 |

DQN은 일부 episode에서 성공적인 착륙을 수행했지만, 표준편차가 크게 나타났습니다.
이는 episode에 따라 착륙 성공과 장애물 충돌 또는 추락이 혼재되어 있음을 의미합니다.

---

## 5.2 PPO

PPO는 policy gradient 기반 알고리즘으로, 정책이 한 번에 너무 크게 변하지 않도록 제한하는 구조를 가지고 있습니다.
이러한 특성 덕분에 장애물 회피 착륙과 같이 불안정한 환경에서도 안정적인 성능을 기대할 수 있습니다.

### Best PPO Hyperparameters

```python
learning_rate = 5e-4
ent_coef = 0.01
n_steps = 4096
clip_range = 0.2
n_epochs = 10
batch_size = 64
```

### PPO Best Result

| Metric         |          Value |
| -------------- | -------------: |
| ep_rew_mean    |            272 |
| episode_reward | 268.89 ± 17.69 |
| Episode length | 200.50 ± 19.51 |
| Mean reward    | 281.84 ± 15.42 |

PPO는 세 알고리즘 중 가장 높은 평균 reward와 가장 낮은 표준편차를 기록했습니다.
이는 PPO가 장애물 회피와 안정적인 착륙을 가장 일관적으로 수행했음을 의미합니다.

---

## 5.3 A2C

A2C는 advantage actor-critic 구조를 사용하는 policy-based 알고리즘입니다.
DQN, PPO와 함께 비교하기 위해 baseline 알고리즘으로 사용했습니다.

### Best A2C Hyperparameters

```python
learning_rate = 7e-4
n_steps = 32
ent_coef = 0.02
gae_lambda = 0.95
```

### A2C Best Result

| Metric         |          Value |
| -------------- | -------------: |
| ep_rew_mean    |          -72.9 |
| episode_reward | 31.02 ± 121.32 |
| Episode length | 209.00 ± 70.01 |
| Mean reward    | 49.58 ± 157.36 |

A2C는 일부 episode에서 성공 가능성을 보였지만, 전체적으로 reward 편차가 크게 나타났습니다.
따라서 장애물 회피 착륙 문제에서는 PPO보다 안정성이 낮은 것으로 확인되었습니다.

---

## 📊 6. Final Evaluation

최종 평가는 각 알고리즘의 best model을 대상으로 100 episode 동안 수행했습니다.

### Evaluation Metrics

| Metric                  | Description                |
| ----------------------- | -------------------------- |
| Mean Reward             | 100 episode의 평균 누적 보상      |
| Reward Std              | reward의 표준편차               |
| Success Rate            | 장애물 충돌 없이 양수 reward를 얻은 비율 |
| Obstacle Collision Rate | 장애물과 충돌한 episode 비율        |

---

## 🏆 7. Final Results

| Algorithm | Mean Reward | Reward Std | Success Rate | Obstacle Collision Rate |
| --------- | ----------: | ---------: | -----------: | ----------------------: |
| DQN       |      112.89 |     134.85 |         0.80 |                    0.20 |
| PPO       |      273.80 |      22.11 |         1.00 |                    0.00 |
| A2C       |       60.08 |     170.44 |         0.54 |                    0.05 |

---

## 📈 8. Result Analysis

### DQN

DQN은 이산형 action space에 적합한 알고리즘답게 학습이 진행되었고,
최종 평가에서 80%의 success rate를 기록했습니다.

그러나 reward 표준편차가 크게 나타났으며, 장애물 충돌률도 20%로 확인되었습니다.
즉, 일부 episode에서는 성공적으로 착륙했지만 모든 상황에서 안정적인 정책을 보이지는 못했습니다.

### PPO

PPO는 가장 안정적인 성능을 보였습니다.

* Mean reward: 273.80
* Reward std: 22.11
* Success rate: 1.00
* Obstacle collision rate: 0.00

PPO는 평균 보상뿐만 아니라 안정성 측면에서도 가장 우수했습니다.
모든 평가 episode에서 장애물과 충돌하지 않고 성공적으로 착륙했습니다.

### A2C

A2C는 일부 episode에서 착륙에 성공했지만, reward 표준편차가 가장 크게 나타났습니다.
이는 학습된 정책이 episode마다 크게 흔들렸다는 것을 의미합니다.

A2C는 구조가 비교적 단순하기 때문에, 장애물 회피 착륙과 같은 복잡한 문제에서는 PPO보다 낮은 안정성을 보였습니다.

---

## ✅ 9. Conclusion

본 프로젝트에서는 원본 LunarLander 환경에 공중 장애물을 추가하여
장애물 회피 착륙 문제를 구성했습니다.

이후 DQN, PPO, A2C 알고리즘을 동일한 환경에서 학습시키고 성능을 비교했습니다.

최종 결과는 다음과 같습니다.

```text
PPO > DQN > A2C
```

PPO는 가장 높은 평균 reward, 가장 낮은 reward 표준편차, 100% success rate, 0% obstacle collision rate를 기록했습니다.

따라서 본 프로젝트에서 구성한 장애물 회피 LunarLander 환경에서는
**PPO가 가장 적합한 강화학습 알고리즘**으로 판단됩니다.

---

## 🚧 10. Future Work

향후 개선 방향은 다음과 같습니다.

* 장애물 위치를 매 episode마다 랜덤화
* 장애물 개수 증가
* 움직이는 장애물 추가
* 착륙 정확도 및 연료 사용량 추가 분석
* 연속형 action space로 확장
* SAC, TD3, DDPG 등 연속 제어 알고리즘 비교

---

## 🔁 11. Discrete vs Continuous Action Space

LunarLander는 설정에 따라 이산형과 연속형 action space를 모두 사용할 수 있습니다.

| Type       | Action Space             | Suitable Algorithms      |
| ---------- | ------------------------ | ------------------------ |
| Discrete   | `Discrete(4)`            | DQN, PPO, A2C            |
| Continuous | `Box(-1, 1, shape=(2,))` | SAC, TD3, DDPG, PPO, A2C |

본 프로젝트에서는 이산형 action space를 사용했기 때문에 DQN, PPO, A2C를 비교했습니다.

연속형 환경으로 확장하려면 다음과 같이 설정할 수 있습니다.

```python
env = gym.make(
    "LunarLanderObstacle-v0",
    continuous=True,
    render_mode=None
)
```

이 경우 에이전트는 엔진 출력 세기를 연속값으로 직접 제어하게 됩니다.

---

## 📁 12. Project Structure

```text
lunar_lander_obstacle_rl/
│
├── lunar_lander_obstacle.py      # Custom LunarLander environment with obstacles
├── train_lunar_obstacle.py       # Training script
├── evaluate_lunar_obstacle.py    # Evaluation script
├── results/                      # Saved models and logs
│   ├── DQN_lunar_obstacle_final.zip
│   ├── PPO_lunar_obstacle_final.zip
│   └── A2C_lunar_obstacle_final.zip
│
├── README.md
└── requirements.txt
```

---

## ⚙️ 13. Installation

```bash
pip install gymnasium[box2d]
pip install stable-baselines3
pip install numpy
pip install pygame
```

If Box2D installation fails, install `swig` first.

```bash
pip install swig
pip install gymnasium[box2d]
```

---

## 🚀 14. How to Train

```bash
python train_lunar_obstacle.py
```

In the training script, choose the algorithm.

```python
train_model(
    algo_name="PPO",
    total_timesteps=1000000
)
```

Available algorithms:

```python
"DQN"
"PPO"
"A2C"
```

---

## 🎥 15. How to Evaluate

```bash
python evaluate_lunar_obstacle.py
```

To evaluate a specific model, change the algorithm name.

```python
ALGO = "PPO"
```

Available options:

```python
ALGO = "DQN"
ALGO = "PPO"
ALGO = "A2C"
```

---

## 📌 16. Summary

This project modified the original LunarLander environment by adding aerial obstacles and obstacle-relative observations.
The agent learned to avoid obstacles and land safely using reinforcement learning.

Among DQN, PPO, and A2C, PPO achieved the best performance.

```text
Best Algorithm: PPO
Mean Reward: 273.80
Success Rate: 1.00
Obstacle Collision Rate: 0.00
```
