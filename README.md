# 🚀 LunarLander Obstacle Avoidance with Reinforcement Learning

> Gymnasium Box2D LunarLander 환경을 수정하여 공중 장애물이 포함된 착륙 환경을 만들고,
> DQN, PPO, A2C 강화학습 알고리즘의 장애물 회피 및 착륙 성능을 비교한 프로젝트이다.

---

## 📌 1. Project Overview

`LunarLander`는 OpenAI Gym/Gymnasium에서 제공하는 대표적인 강화학습 환경으로,
달 착륙선을 조종하여 지정된 착륙 지점에 안전하게 착륙시키는 문제이다.

기본 LunarLander 환경에서는 장애물이 존재하지 않기 때문에, 에이전트는 단순히 착륙 지점에 안정적으로 도달하는 것만 학습한다.
본 프로젝트에서는 원본 Box2D LunarLander 환경을 수정하여 **공중에 작은 장애물 3개**를 추가하였다.

따라서 에이전트는 다음 두 가지 목표를 동시에 달성해야 한다.

* 공중 장애물과 충돌하지 않도록 회피
* 최종적으로 착륙 지점에 안전하게 착륙

본 프로젝트의 최종 목표는 수정된 LunarLander 환경에서 **DQN, PPO, A2C** 알고리즘을 학습시키고,
각 알고리즘의 장애물 회피 및 착륙 성능을 비교하여 가장 적합한 알고리즘을 평가하는 것이다.

---

## 🛠️ 2. Environment Modification

본 프로젝트에서는 원본 Box2D LunarLander 코드를 수정하여
기존의 단순 착륙 문제를 **장애물 회피 착륙 문제**로 확장했다.

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

기본 LunarLander의 observation은 다음 8개의 값으로 구성된다.

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

본 프로젝트에서는 장애물 3개에 대해 각각 상대 위치 정보를 추가했다.

```text
obstacle 1: relative x, relative y
obstacle 2: relative x, relative y
obstacle 3: relative x, relative y
```

따라서 최종 observation은 다음과 같이 구성된다.

```text
기존 상태값 8개 + 장애물 상대 위치 6개 = 총 14차원 observation
```

---

## 🎮 4. Action Space

본 프로젝트에서는 LunarLander의 **이산형 action space**를 사용했다.

```text
0: 아무것도 하지 않음
1: 왼쪽 방향 엔진 점화
2: 메인 엔진 점화
3: 오른쪽 방향 엔진 점화
```

즉, 에이전트는 매 step마다 4개의 행동 중 하나를 선택한다.

```text
Action Space: Discrete(4)
```

따라서 본 프로젝트에서는 이산형 action space에 적용 가능한 알고리즘인
**DQN, PPO, A2C**를 비교했다.

---

## 🧠 5. Algorithms and Hyperparameter Experiments

본 프로젝트에서는 수정된 LunarLanderObstacle 환경에서 **DQN, PPO, A2C** 알고리즘을 학습하고 성능을 비교하였다.

모든 실험은 동일한 장애물 환경에서 수행하였으며, 주요 성능 지표는 다음과 같다.

| Metric           | Description                             |
| ---------------- | --------------------------------------- |
| `ep_rew_mean`    | 학습 중 최근 episode들의 평균 reward             |
| `episode_reward` | EvalCallback에서 측정한 평가 episode 평균 reward |
| `Episode length` | episode가 종료되기까지 지속된 평균 step 수           |
| `Mean reward`    | 최종 evaluation에서 측정한 평균 누적 reward        |

---

# 5.1 DQN

DQN은 이산형 action space에 적합한 대표적인 value-based 강화학습 알고리즘이다.
현재 환경은 `Discrete(4)` action space를 사용하므로 DQN을 기본 비교 알고리즘으로 사용하였다.

## DQN 1차 실험: timestep 변화

먼저 동일한 하이퍼파라미터 설정에서 `num_timesteps`에 따른 성능 변화를 확인하였다.

### DQN 초기 설정

```python
exploration_fraction = 0.35
learning_rate = 5e-5
target_update_interval = 2000
batch_size = 128
exploration_final_eps = 0.02
```

| num_timesteps | ep_rew_mean |  episode_reward |  Episode length |     Mean reward |
| ------------: | ----------: | --------------: | --------------: | --------------: |
|        50,000 |        -328 | -108.40 ± 61.92 | 743.60 ± 391.75 | -109.88 ± 59.82 |
|       200,000 |       -2.58 |  -78.21 ± 71.54 |  148.00 ± 72.73 | -32.29 ± 130.95 |
|       500,000 |       -87.3 | -114.00 ± 73.70 | 506.90 ± 358.38 | -30.06 ± 170.99 |
|     1,000,000 |        47.1 |  50.00 ± 143.75 | 456.50 ± 285.76 |  58.54 ± 145.58 |

1,000,000 timestep에서 평균 reward가 양수로 전환되며 가장 좋은 성능을 보였다.
그러나 표준편차가 매우 크게 나타나 episode마다 성능 차이가 컸다.
즉, 일부 episode에서는 성공적으로 착륙했지만, 다른 episode에서는 장애물 충돌 또는 추락으로 실패하는 경우가 있었다.

따라서 이후 DQN의 하이퍼파라미터 조정을 수행하였다.

---

## DQN 2차 실험: 하이퍼파라미터 조정

| 실험 | exploration_fraction | learning_rate | target_update_interval | batch_size | exploration_final_eps | ep_rew_mean |  episode_reward |  Episode length |     Mean reward |
| -: | -------------------: | ------------: | ---------------------: | ---------: | --------------------: | ----------: | --------------: | --------------: | --------------: |
|  1 |                 0.35 |          5e-5 |                   2000 |        128 |                  0.02 |        47.1 |  50.00 ± 143.75 | 456.50 ± 285.76 |  58.54 ± 145.58 |
|  2 |                 0.45 |          5e-5 |                   2000 |        128 |                  0.02 |       -17.2 |  -44.36 ± 76.64 | 749.30 ± 348.78 |  -79.69 ± 89.32 |
|  3 |                 0.35 |          3e-5 |                   2000 |        128 |                  0.02 |         122 |  202.35 ± 34.76 | 623.30 ± 132.73 |  85.86 ± 139.85 |
|  4 |                 0.35 |          3e-5 |                   5000 |        128 |                  0.02 |         101 |  84.40 ± 177.43 | 319.00 ± 159.78 | 108.12 ± 167.40 |
|  5 |                 0.35 |          3e-5 |                   2000 |        256 |                  0.02 |       -92.6 | -106.33 ± 91.80 | 593.90 ± 381.55 |  -78.00 ± 93.80 |
|  6 |                 0.35 |          3e-5 |                   2000 |        128 |                  0.01 |       -58.3 |  -84.70 ± 59.84 | 642.60 ± 439.90 |  -65.96 ± 64.20 |

## DQN Best Setting

```python
exploration_fraction = 0.35
learning_rate = 3e-5
target_update_interval = 2000
batch_size = 128
exploration_final_eps = 0.02
```

4번째 설정은 `Mean reward`가 가장 높았지만 표준편차가 가장 커서 안정성이 낮았다.
따라서 평균 성능과 안정성을 함께 고려하여 **3번째 설정**을 최종 DQN 설정으로 선정하였다.

---

# 5.2 PPO

PPO는 policy gradient 기반 알고리즘으로, 정책이 한 번에 너무 크게 변하지 않도록 제한하는 구조를 가진다.
이러한 특성 덕분에 장애물 회피 착륙과 같이 불안정한 환경에서도 안정적인 성능을 기대할 수 있다.

## PPO 하이퍼파라미터 실험 결과

| 실험 | learning_rate | ent_coef | n_steps | clip_range | n_epochs | batch_size | ep_rew_mean |  episode_reward |  Episode length |     Mean reward |
| -: | ------------: | -------: | ------: | ---------: | -------: | ---------: | ----------: | --------------: | --------------: | --------------: |
|  1 |          3e-4 |     0.01 |    2048 |        0.2 |       10 |         64 |         260 |  263.72 ± 17.87 |   171.90 ± 5.37 |  256.30 ± 95.13 |
|  2 |          1e-4 |     0.01 |    2048 |        0.2 |       10 |         64 |        -128 | -127.94 ± 59.01 | 928.60 ± 211.55 | -118.97 ± 48.51 |
|  3 |          5e-4 |     0.01 |    2048 |        0.2 |       10 |         64 |         259 |  273.53 ± 20.48 |  195.40 ± 13.32 |  277.90 ± 17.75 |
|  4 |          7e-4 |     0.01 |    2048 |        0.2 |       10 |         64 |         253 |  264.97 ± 58.41 |  176.80 ± 21.60 |  287.83 ± 18.88 |
|  5 |          5e-4 |     0.02 |    2048 |        0.2 |       10 |         64 |        99.9 | 131.87 ± 182.79 | 260.80 ± 254.73 | 155.29 ± 177.01 |
|  6 |          5e-4 |    0.005 |    2048 |        0.2 |       10 |         64 |         241 | 231.96 ± 143.17 |  166.60 ± 37.31 |  288.20 ± 21.91 |
|  7 |          5e-4 |     0.01 |    1024 |        0.2 |       10 |         64 |         250 |  296.05 ± 20.61 |  219.40 ± 17.03 | 199.91 ± 151.63 |
|  8 |          5e-4 |     0.01 |    4096 |        0.2 |       10 |         64 |         272 |  268.89 ± 17.69 |  200.50 ± 19.51 |  281.84 ± 15.42 |
|  9 |          5e-4 |     0.01 |    4096 |        0.1 |       10 |         64 |         239 |  279.42 ± 13.91 |  189.90 ± 18.89 | 252.65 ± 102.55 |
| 10 |          5e-4 |     0.01 |    4096 |        0.2 |       20 |         64 |         237 |  295.44 ± 10.82 |  173.70 ± 24.34 | 242.93 ± 132.05 |
| 11 |          5e-4 |     0.01 |    4096 |        0.2 |        5 |         64 |         214 |  261.06 ± 21.08 |  285.50 ± 23.39 | 153.04 ± 164.13 |
| 12 |          5e-4 |     0.01 |    4096 |        0.2 |       10 |        128 |         269 |  275.85 ± 21.97 |  188.80 ± 16.62 | 264.62 ± 103.64 |

## PPO Best Setting

```python
learning_rate = 5e-4
ent_coef = 0.01
n_steps = 4096
clip_range = 0.2
n_epochs = 10
batch_size = 64
```

8번째 설정은 평균 성능이 높고 표준편차가 가장 작아 가장 안정적인 성능을 보였다.
따라서 해당 설정을 최종 PPO 설정으로 선정하였다.

---

# 5.3 A2C

A2C는 advantage actor-critic 구조를 사용하는 policy-based 강화학습 알고리즘이다.
DQN, PPO와 함께 비교하기 위해 baseline 알고리즘으로 사용하였다.

## A2C 하이퍼파라미터 실험 결과

| 실험 | n_steps | learning_rate | ent_coef | gae_lambda | ep_rew_mean |   episode_reward |  Episode length |     Mean reward |
| -: | ------: | ------------: | -------: | ---------: | ----------: | ---------------: | --------------: | --------------: |
|  1 |       5 |          7e-4 |     0.01 |       0.95 |        -7.2 |  -53.62 ± 140.68 | 292.40 ± 268.99 |   0.05 ± 145.00 |
|  2 |      16 |          7e-4 |     0.01 |       0.95 |         -46 |  -58.97 ± 148.64 | 196.10 ± 112.01 |   5.72 ± 162.45 |
|  3 |      32 |          7e-4 |     0.01 |       0.95 |          70 |   70.03 ± 151.99 | 244.20 ± 110.89 |  37.83 ± 156.79 |
|  4 |      64 |          7e-4 |     0.01 |       0.95 |       -36.1 |  -82.73 ± 113.35 |  174.80 ± 97.26 | -57.01 ± 129.44 |
|  5 |      32 |          3e-4 |     0.01 |       0.95 |        6.75 |   50.13 ± 168.20 |  418.00 ± 91.66 | -69.96 ± 142.31 |
|  6 |      32 |          1e-4 |     0.01 |       0.95 |        -114 |  -241.15 ± 14.20 | 920.20 ± 239.40 | -201.26 ± 98.76 |
|  7 |      32 |          7e-4 |    0.005 |       0.95 |        -128 |  -158.38 ± 51.17 | 440.00 ± 282.74 | -146.40 ± 39.62 |
|  8 |      32 |          7e-4 |     0.02 |       0.95 |       -72.9 |   31.02 ± 121.32 |  209.00 ± 70.01 |  49.58 ± 157.36 |
|  9 |      32 |          7e-4 |     0.01 |       0.90 |       -36.6 |  -133.93 ± 57.11 | 622.60 ± 386.25 | -157.94 ± 83.16 |
| 10 |      32 |          5e-4 |     0.02 |       0.95 |       -50.3 |  174.79 ± 100.78 | 470.90 ± 272.69 |  23.58 ± 153.95 |
| 11 |      32 |          1e-4 |     0.02 |       0.95 |        -151 | -168.34 ± 163.42 | 636.20 ± 285.09 | -256.04 ± 93.57 |

## A2C Best Setting

```python
n_steps = 32
learning_rate = 7e-4
ent_coef = 0.02
gae_lambda = 0.95
```

A2C에서는 8번째 설정이 최종 `Mean reward` 기준으로 가장 높은 성능을 보였다.
다만 표준편차가 매우 크기 때문에, PPO에 비해 안정적인 정책을 학습하지는 못한 것으로 판단된다.

---

## 📊 6. Final Evaluation

최종 평가는 각 알고리즘의 best model을 대상으로 100 episode 동안 수행하였다.

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

## DQN

DQN은 이산형 action space에 적합한 알고리즘답게 학습이 진행되었고, 최종 평가에서 80%의 success rate를 기록하였다.

그러나 reward 표준편차가 134.85로 크게 나타났으며, 장애물 충돌률도 20%로 확인되었다.
즉, 일부 episode에서는 성공적으로 착륙했지만 모든 상황에서 안정적인 정책을 보이지는 못했다.

## PPO

PPO는 가장 안정적인 성능을 보였다.

```text
Mean reward = 273.80
Reward std = 22.11
Success rate = 1.00
Obstacle collision rate = 0.00
```

PPO는 평균 보상뿐만 아니라 안정성 측면에서도 가장 우수하였다.
모든 평가 episode에서 장애물과 충돌하지 않고 성공적으로 착륙하였다.

## A2C

A2C는 일부 episode에서 착륙에 성공했지만, reward 표준편차가 170.44로 가장 크게 나타났다.
이는 학습된 정책이 episode마다 크게 흔들렸다는 것을 의미한다.

A2C는 구조가 비교적 단순하기 때문에, 장애물 회피 착륙과 같은 복잡한 문제에서는 PPO보다 낮은 안정성을 보였다.

---

## ✅ 9. Conclusion

본 프로젝트에서는 원본 LunarLander 환경에 공중 장애물을 추가하여 장애물 회피 착륙 문제를 구성하였다.

이후 DQN, PPO, A2C 알고리즘을 동일한 환경에서 학습시키고 성능을 비교하였다.

최종 결과는 다음과 같다.

```text
PPO > DQN > A2C
```

PPO는 가장 높은 평균 reward, 가장 낮은 reward 표준편차, 100% success rate, 0% obstacle collision rate를 기록하였다.

따라서 본 프로젝트에서 구성한 장애물 회피 LunarLander 환경에서는 **PPO가 가장 적합한 강화학습 알고리즘**으로 판단된다.

---

## 🚧 10. Future Work

향후 개선 방향은 다음과 같다.

* 장애물 위치를 매 episode마다 랜덤화
* 장애물 개수 증가
* 움직이는 장애물 추가
* 착륙 정확도 및 연료 사용량 추가 분석
* 연속형 action space로 확장
* SAC, TD3, DDPG 등 연속 제어 알고리즘 비교

---

## 🔁 11. Discrete vs Continuous Action Space

LunarLander는 설정에 따라 이산형과 연속형 action space를 모두 사용할 수 있다.

| Type       | Action Space             | Suitable Algorithms      |
| ---------- | ------------------------ | ------------------------ |
| Discrete   | `Discrete(4)`            | DQN, PPO, A2C            |
| Continuous | `Box(-1, 1, shape=(2,))` | SAC, TD3, DDPG, PPO, A2C |

본 프로젝트에서는 이산형 action space를 사용했기 때문에 DQN, PPO, A2C를 비교했다.

연속형 환경으로 확장하려면 다음과 같이 설정할 수 있다.

```python
env = gym.make(
    "LunarLanderObstacle-v0",
    continuous=True,
    render_mode=None
)
```

이 경우 에이전트는 엔진 출력 세기를 연속값으로 직접 제어하게 된다.

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

## 🚀 13. How to Train

```bash
python train_lunar_obstacle.py
```

알고리즘을 선택한다.

```python
train_model(
    algo_name="PPO",
    total_timesteps=1000000
)
```

사용 가능한 알고리즘:

```python
"DQN"
"PPO"
"A2C"
```

---

## 🎥 14. How to Evaluate

```bash
python evaluate_lunar_obstacle.py
```

특정 모델을 평가하려면 알고리즘 이름을 변경한다.

```python
ALGO = "PPO"
```

사용 가능한 알고리즘:

```python
ALGO = "DQN"
ALGO = "PPO"
ALGO = "A2C"
```

---

## 📌 15. Summary

이 프로젝트는 공중 장애물과 장애물 상대 관측을 추가하여 원래의 달 착륙선 환경을 수정했습니다.
에이전트는 강화 학습을 통해 장애물을 피하고 안전하게 착륙하는 방법을 배웠습니다.

DQN, PPO, A2C 중에서 PPO가 가장 우수한 성과를 거두었습니다.

```text
Best Algorithm: PPO
Mean Reward: 273.80
Success Rate: 1.00
Obstacle Collision Rate: 0.00
```
