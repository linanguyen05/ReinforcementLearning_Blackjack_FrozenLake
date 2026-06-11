"""
FrozenLake - Q-Learning Agent
==============================
Sử dụng thư viện Gymnasium (Farama Foundation) để huấn luyện agent
giải bài toán FrozenLake bằng thuật toán Tabular Q-Learning.

Tham khảo:
- https://gymnasium.farama.org/environments/toy_text/frozen_lake/
- https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/
- https://github.com/moripiri/Reinforcement-Learning-on-FrozenLake
- Sutton & Barto, "Reinforcement Learning: An Introduction"

Môi trường: FrozenLake-v1
- Action Space: Discrete(4) — 0: Left, 1: Down, 2: Right, 3: Up
- Observation: Discrete(16) cho 4x4, Discrete(64) cho 8x8
- Map tiles: S(Start), F(Frozen), H(Hole), G(Goal)
- Reward: +1 (goal), 0 (hole/frozen)
"""

from __future__ import annotations

import os
import sys
from typing import NamedTuple

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no popup)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

sns.set_theme()

# Đường lưu kết quả
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 1. Định nghĩa Parameters
# ============================================================
class Params(NamedTuple):
    """Tham số huấn luyện."""
    total_episodes: int     # Tổng số episodes
    learning_rate: float    # Tốc độ học (alpha)
    gamma: float            # Hệ số chiết khấu
    epsilon: float          # Xác suất khám phá (exploration)
    map_size: int           # Kích thước bản đồ (map_size x map_size)
    seed: int               # Seed cho reproducibility
    is_slippery: bool       # Mặt băng trơn trượt?
    n_runs: int             # Số lần chạy (để lấy trung bình)
    action_size: int        # Số hành động
    state_size: int         # Số trạng thái
    proba_frozen: float     # Xác suất ô băng (không phải hố)


params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
)

# Seed cho reproducibility
rng = np.random.default_rng(params.seed)


# ============================================================
# 2. Định nghĩa Q-Learning Agent
# ============================================================
class Qlearning:
    """
    Tabular Q-Learning Agent.

    Q-table: matrix [state_size x action_size]
    Update rule: Q(s,a) = Q(s,a) + α * [R + γ * max Q(s',a') - Q(s,a)]
    """

    def __init__(self, learning_rate: float, gamma: float, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    def update(self, state: int, action: int, reward: float, new_state: int) -> float:
        """
        Cập nhật Q-value cho cặp (state, action).

        Q(s,a) := Q(s,a) + lr * [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        """
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    def reset_qtable(self):
        """Reset Q-table về 0."""
        self.qtable = np.zeros((self.state_size, self.action_size))


# ============================================================
# 3. Định nghĩa Epsilon-Greedy Strategy
# ============================================================
class EpsilonGreedy:
    """
    Chiến lược Epsilon-Greedy cho việc chọn hành động.

    - Với xác suất epsilon: chọn random (exploration)
    - Với xác suất (1-epsilon): chọn action có Q-value cao nhất (exploitation)
    """

    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def choose_action(self, action_space, state: int, qtable: np.ndarray) -> int:
        """Chọn hành động theo epsilon-greedy."""
        if rng.uniform(0, 1) < self.epsilon:
            # Exploration: chọn random
            action = action_space.sample()
        else:
            # Exploitation: chọn action tốt nhất (break ties randomly)
            max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
            action = rng.choice(max_ids)
        return action


# ============================================================
# 4. Hàm chạy môi trường
# ============================================================
def run_env(env, params, learner, explorer):
    """
    Chạy huấn luyện trên môi trường FrozenLake.

    Returns:
        rewards, steps, episodes, qtables, all_states, all_actions
    """
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):
        learner.reset_qtable()

        for episode in tqdm(
            episodes,
            desc=f"  Run {run + 1}/{params.n_runs}",
            leave=False,
        ):
            state = env.reset(seed=params.seed)[0]
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space,
                    state=state,
                    qtable=learner.qtable,
                )

                all_states.append(state)
                all_actions.append(action)

                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                learner.qtable[state, action] = learner.update(
                    state, action, reward, new_state
                )

                total_rewards += reward
                step += 1
                state = new_state

            rewards[episode, run] = total_rewards
            steps[episode, run] = step

        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions


# ============================================================
# 5. Hàm xử lý kết quả
# ============================================================
def postprocess(episodes, params, rewards, steps, map_size):
    """Chuyển kết quả sang DataFrame để dễ vẽ biểu đồ."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(order="F"),
            "Steps": steps.flatten(order="F"),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])

    return res, st


# ============================================================
# 6. Hàm vẽ biểu đồ
# ============================================================
def qtable_directions_map(qtable, map_size):
    """Chuyển Q-table thành bản đồ hướng (mũi tên) cho từng ô."""
    qtable_val_max = qtable.max(axis=1).reshape(map_size, map_size)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_size, map_size)
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}

    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps

    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            qtable_directions[idx] = directions[val]

    qtable_directions = qtable_directions.reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size, save_name=""):
    """Vẽ frame cuối của simulation và policy đã học."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    fig.suptitle(
        f"FrozenLake {map_size}x{map_size} — Learned Policy",
        fontsize=14,
        fontweight="bold",
    )

    # Frame cuối
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last Frame")

    # Policy heatmap
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")

    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    if save_name:
        plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches="tight")

    plt.close()


def plot_states_actions_distribution(states, actions, map_size, save_name=""):
    """Vẽ phân phối states và actions."""
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    fig.suptitle(
        f"FrozenLake {map_size}x{map_size} — State & Action Distribution",
        fontsize=14,
        fontweight="bold",
    )

    sns.histplot(data=states, ax=ax[0], kde=True, color="#2196F3")
    ax[0].set_title("States Distribution")
    ax[0].set_xlabel("State")

    sns.histplot(data=actions, ax=ax[1], color="#4CAF50")
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions Distribution")
    ax[1].set_xlabel("Action")

    fig.tight_layout()
    if save_name:
        plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches="tight")
    plt.close()


def plot_steps_and_rewards(rewards_df, steps_df):
    """Vẽ biểu đồ rewards tích lũy và số bước trung bình."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    fig.suptitle(
        "FrozenLake Q-Learning — Training Progress",
        fontsize=14,
        fontweight="bold",
    )

    sns.lineplot(data=rewards_df, x="Episodes", y="cum_rewards", hue="map_size", ax=ax[0])
    ax[0].set(ylabel="Cumulated Rewards")
    ax[0].set_title("Cumulated Rewards over Episodes")

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", hue="map_size", ax=ax[1])
    ax[1].set(ylabel="Averaged Steps")
    ax[1].set_title("Average Steps per Episode")

    for axi in ax:
        axi.legend(title="Map Size")

    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "frozenlake_training_progress.png"), dpi=150)
    plt.close()


# ============================================================
# 7. MAIN — Chạy huấn luyện trên nhiều kích thước bản đồ
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  FROZENLAKE - Q-Learning Agent Training")
    print("=" * 60)
    print(f"  Learning rate:     {params.learning_rate}")
    print(f"  Gamma:             {params.gamma}")
    print(f"  Epsilon:           {params.epsilon}")
    print(f"  Episodes/map:      {params.total_episodes}")
    print(f"  Runs/map:          {params.n_runs}")
    print(f"  Slippery:          {params.is_slippery}")
    print(f"  Frozen probability:{params.proba_frozen}")
    print("=" * 60)

    # Chạy trên nhiều kích thước bản đồ
    map_sizes = [4, 7, 9, 11]
    res_all = pd.DataFrame()
    st_all = pd.DataFrame()

    for map_size in map_sizes:
        print(f"\n{'─' * 40}")
        print(f"  📍 Map size: {map_size}x{map_size}")
        print(f"{'─' * 40}")

        env = gym.make(
            "FrozenLake-v1",
            is_slippery=params.is_slippery,
            render_mode="rgb_array",
            desc=generate_random_map(
                size=map_size,
                p=params.proba_frozen,
                seed=params.seed,
            ),
        )

        # Cập nhật kích thước state và action
        current_params = params._replace(
            action_size=env.action_space.n,
            state_size=env.observation_space.n,
            map_size=map_size,
        )

        print(f"  Action size: {current_params.action_size}")
        print(f"  State size:  {current_params.state_size}")

        env.action_space.seed(current_params.seed)

        learner = Qlearning(
            learning_rate=current_params.learning_rate,
            gamma=current_params.gamma,
            state_size=current_params.state_size,
            action_size=current_params.action_size,
        )
        explorer = EpsilonGreedy(epsilon=current_params.epsilon)

        # Huấn luyện
        rewards, steps, episodes, qtables, all_states, all_actions = run_env(
            env, current_params, learner, explorer
        )

        # Xử lý kết quả
        res, st = postprocess(episodes, current_params, rewards, steps, map_size)
        res_all = pd.concat([res_all, res])
        st_all = pd.concat([st_all, st])

        # Q-table trung bình qua các runs
        qtable = qtables.mean(axis=0)

        # Vẽ biểu đồ cho map size này
        plot_states_actions_distribution(
            states=all_states,
            actions=all_actions,
            map_size=map_size,
            save_name=f"frozenlake_{map_size}x{map_size}_distribution.png",
        )

        plot_q_values_map(
            qtable,
            env,
            map_size,
            save_name=f"frozenlake_{map_size}x{map_size}_policy.png",
        )

        # Đánh giá nhanh
        eval_wins = 0
        n_eval = 100
        for _ in range(n_eval):
            state = env.reset(seed=None)[0]
            done = False
            while not done:
                max_ids = np.where(qtable[state, :] == max(qtable[state, :]))[0]
                action = rng.choice(max_ids)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            if reward > 0:
                eval_wins += 1

        print(f"  ✅ Win rate (eval): {eval_wins}/{n_eval} = {eval_wins}%")

        env.close()

    # Vẽ biểu đồ tổng hợp
    plot_steps_and_rewards(res_all, st_all)

    print("\n" + "=" * 60)
    print("  ✅ HOÀN TẤT!")
    print(f"  📁 Kết quả được lưu tại: {OUTPUT_DIR}")
    print("=" * 60)
