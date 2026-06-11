"""
Blackjack - Q-Learning Agent
=============================
Sử dụng thư viện Gymnasium (Farama Foundation) để huấn luyện agent
chơi Blackjack bằng thuật toán Tabular Q-Learning.

Tham khảo:
- https://gymnasium.farama.org/environments/toy_text/blackjack/
- https://gymnasium.farama.org/tutorials/training_agents/blackjack_q_learning/
- Sutton & Barto, "Reinforcement Learning: An Introduction"

Môi trường: Blackjack-v1
- Action Space: Discrete(2) — 0: Stick, 1: Hit
- Observation: (player_sum, dealer_card, usable_ace)
- Reward: +1 (win), -1 (lose), 0 (draw)
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no popup)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym

# ============================================================
# 1. Tạo môi trường Blackjack
# ============================================================
# sab=True: theo đúng luật trong sách Sutton & Barto
env = gym.make("Blackjack-v1", sab=True)

# Đường lưu kết quả
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 2. Định nghĩa Blackjack Agent (Q-Learning)
# ============================================================
class BlackjackAgent:
    """
    Agent sử dụng Tabular Q-Learning với chiến lược epsilon-greedy.

    Q-value được lưu dưới dạng dictionary: q_values[obs] -> np.array([q_stick, q_hit])
    """

    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """
        Khởi tạo agent.

        Args:
            env: Gymnasium environment
            learning_rate: Tốc độ học (alpha)
            initial_epsilon: Giá trị epsilon ban đầu (xác suất khám phá)
            epsilon_decay: Mức giảm epsilon mỗi episode
            final_epsilon: Giá trị epsilon tối thiểu
            discount_factor: Hệ số chiết khấu (gamma)
        """
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, env, obs: tuple[int, int, bool]) -> int:
        """
        Chọn hành động theo chiến lược epsilon-greedy.

        - Với xác suất epsilon: chọn ngẫu nhiên (exploration)
        - Với xác suất (1 - epsilon): chọn action tốt nhất (exploitation)
        """
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """
        Cập nhật Q-value theo công thức Q-Learning:

        Q(s,a) = Q(s,a) + α * [R + γ * max Q(s',a') - Q(s,a)]
        """
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Giảm dần epsilon để agent chuyển từ exploration sang exploitation."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# ============================================================
# 3. Hyperparameters
# ============================================================
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # Giảm dần trong nửa đầu
final_epsilon = 0.1

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

# ============================================================
# 4. Vòng lặp huấn luyện
# ============================================================
print("=" * 60)
print("  BLACKJACK - Q-Learning Agent Training")
print("=" * 60)
print(f"  Số episodes:       {n_episodes:,}")
print(f"  Learning rate:     {learning_rate}")
print(f"  Epsilon:           {start_epsilon} → {final_epsilon}")
print(f"  Discount factor:   0.95")
print("=" * 60)

env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

for episode in tqdm(range(n_episodes), desc="Training Blackjack Agent"):
    obs, info = env.reset()
    done = False

    # Chơi 1 episode
    while not done:
        action = agent.get_action(env, obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Cập nhật Q-value
        agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()

print("\n✅ Huấn luyện hoàn tất!")

# ============================================================
# 5. Visualize quá trình huấn luyện
# ============================================================
rolling_length = 500

fig, axs = plt.subplots(ncols=3, figsize=(14, 5))
fig.suptitle("Blackjack Q-Learning — Training Progress", fontsize=14, fontweight="bold")

# --- Episode Rewards ---
axs[0].set_title("Episode Rewards (Moving Avg)")
reward_moving_average = (
    np.convolve(
        np.array(env.return_queue).flatten(),
        np.ones(rolling_length),
        mode="valid",
    )
    / rolling_length
)
axs[0].plot(range(len(reward_moving_average)), reward_moving_average, color="#2196F3")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")

# --- Episode Lengths ---
axs[1].set_title("Episode Lengths (Moving Avg)")
length_moving_average = (
    np.convolve(
        np.array(env.length_queue).flatten(),
        np.ones(rolling_length),
        mode="same",
    )
    / rolling_length
)
axs[1].plot(range(len(length_moving_average)), length_moving_average, color="#4CAF50")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

# --- Training Error ---
axs[2].set_title("Training Error (Moving Avg)")
training_error_moving_average = (
    np.convolve(
        np.array(agent.training_error),
        np.ones(rolling_length),
        mode="same",
    )
    / rolling_length
)
axs[2].plot(
    range(len(training_error_moving_average)),
    training_error_moving_average,
    color="#FF5722",
)
axs[2].set_xlabel("Step")
axs[2].set_ylabel("TD Error")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "blackjack_training_progress.png"), dpi=150)
plt.close()
print(f"📊 Đã lưu biểu đồ training tại: blackjack_training_progress.png")


# ============================================================
# 6. Visualize Policy đã học được
# ============================================================
def create_grids(agent, usable_ace=False):
    """Tạo lưới giá trị trạng thái và chính sách từ Q-table."""
    state_value = defaultdict(float)
    policy = defaultdict(int)

    for obs, action_values in agent.q_values.items():
        state_value[obs] = float(np.max(action_values))
        policy[obs] = int(np.argmax(action_values))

    player_count, dealer_count = np.meshgrid(
        np.arange(12, 22),  # Player's sum: 12-21
        np.arange(1, 11),   # Dealer's face-up card: 1(Ace)-10
    )

    # Value grid
    value = np.apply_along_axis(
        lambda obs: state_value[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )
    value_grid = player_count, dealer_count, value

    # Policy grid
    policy_grid = np.apply_along_axis(
        lambda obs: policy[(obs[0], obs[1], usable_ace)],
        axis=2,
        arr=np.dstack([player_count, dealer_count]),
    )

    return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
    """Vẽ State Values (3D) và Policy (heatmap)."""
    player_count, dealer_count, value = value_grid
    fig = plt.figure(figsize=plt.figaspect(0.4))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # 3D Surface — State Values
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(
        player_count,
        dealer_count,
        value,
        rstride=1,
        cstride=1,
        cmap="viridis",
        edgecolor="none",
    )
    plt.xticks(range(12, 22), range(12, 22))
    plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
    ax1.set_title(f"State Values: {title}")
    ax1.set_xlabel("Player Sum")
    ax1.set_ylabel("Dealer Showing")
    ax1.zaxis.set_rotate_label(False)
    ax1.set_zlabel("Value", fontsize=14, rotation=90)
    ax1.view_init(20, 220)

    # Heatmap — Policy
    fig.add_subplot(1, 2, 2)
    ax2 = sns.heatmap(
        policy_grid,
        linewidth=0,
        annot=True,
        cmap="Accent_r",
        cbar=False,
    )
    ax2.set_title(f"Policy: {title}")
    ax2.set_xlabel("Player Sum")
    ax2.set_ylabel("Dealer Showing")
    ax2.set_xticklabels(range(12, 22))
    ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
        Patch(facecolor="grey", edgecolor="black", label="Stick"),
    ]
    ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))

    return fig


# --- Với Usable Ace ---
value_grid, policy_grid = create_grids(agent, usable_ace=True)
fig1 = create_plots(value_grid, policy_grid, title="With Usable Ace")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "blackjack_policy_usable_ace.png"), dpi=150)
plt.close()

# --- Không có Usable Ace ---
value_grid, policy_grid = create_grids(agent, usable_ace=False)
fig2 = create_plots(value_grid, policy_grid, title="Without Usable Ace")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "blackjack_policy_no_usable_ace.png"), dpi=150)
plt.close()

print(f"📊 Đã lưu policy plots tại thư mục: {OUTPUT_DIR}")

# ============================================================
# 7. Đánh giá agent sau khi huấn luyện
# ============================================================
print("\n" + "=" * 60)
print("  ĐÁNH GIÁ AGENT (1000 episodes)")
print("=" * 60)

eval_env = gym.make("Blackjack-v1", sab=True)
wins = 0
losses = 0
draws = 0
n_eval = 1000

for _ in range(n_eval):
    obs, info = eval_env.reset()
    done = False
    while not done:
        action = int(np.argmax(agent.q_values[obs]))  # Greedy policy
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated

    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        draws += 1

eval_env.close()

print(f"  🏆 Wins:   {wins:4d}  ({wins/n_eval*100:.1f}%)")
print(f"  ❌ Losses: {losses:4d}  ({losses/n_eval*100:.1f}%)")
print(f"  🤝 Draws:  {draws:4d}  ({draws/n_eval*100:.1f}%)")
print(f"  📈 Win Rate: {wins/(wins+losses)*100:.1f}% (excluding draws)")
print("=" * 60)

env.close()
print("\n✅ Hoàn tất! Tất cả biểu đồ đã được lưu.")
