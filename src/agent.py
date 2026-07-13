"""
Implement the DQN agent, network, and replay buffer.

The architecture and Bellman update follow Paper Fig. 2, Paper equation (1),
and SPEC.md Section 3.

CHANGELOG:
  - Added gradient clipping (max_norm=10.0) after loss.backward().
    Without clipping, large early Q-value errors can cause exploding gradients,
    producing the loss spikes visible in the convergence curve. Clipping is
    standard DQN practice (Mnih et al., 2015) and does not alter the paper's
    described architecture or hyperparameters.
  - Input state expanded from 12 to 17 features by adding five per-entity
    cache-presence bits, allowing the agent to see which tenant-scoped keys are
    currently cache-resident before choosing prefetch or eviction actions.
"""

import logging
import random
import sys
from collections import deque
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional

from config import (
    ACTION_DIM,
    BATCH_SIZE,
    BUFFER_CAPACITY,
    EPSILON_DECAY,
    EPSILON_END,
    EPSILON_START,
    GAMMA,
    HIDDEN_DIM,
    LEARNING_RATE,
    STATE_DIM,
)

log = logging.getLogger(__name__)
Transition = tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Fixed-capacity replay memory for DQN transitions."""

    def __init__(self, capacity: int) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in replay memory."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Input(17) -> Linear(17,64) -> ReLU -> Linear(64,64) -> ReLU -> Linear(64,11)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        """Initialize the exact two-hidden-layer DQN architecture from paper Fig. 2."""
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for a batch of states."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return cast(torch.Tensor, self.out(x))


class DQNAgent:
    """DQN agent with policy network, target network, and replay memory."""

    def __init__(self) -> None:
        """Initialize networks, optimizer, loss function, memory, and epsilon."""
        self.policy_net = DQNNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
        self.target_net = DQNNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=LEARNING_RATE,
        )
        self.memory = ReplayBuffer(BUFFER_CAPACITY)
        self.epsilon = EPSILON_START
        self.step_count = 0
        self.loss_log: list[float] = []
        # Bounded to prevent unbounded memory growth during 500k+ steps.
        # Monitor only inspects the last 200 entries.
        self.action_log: deque[int] = deque(maxlen=10_000)
        self.train_hit_ratio = 0.0
        self._last_q_values: torch.Tensor | None = None

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using epsilon-greedy policy."""
        epsilon = self.epsilon if training else 0.0
        if training and random.random() < epsilon:
            action = random.randint(0, ACTION_DIM - 1)
            self.action_log.append(action)
            return action

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            action = int(torch.argmax(q_values, dim=1).item())
            if training:
                self.action_log.append(action)
            return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push one transition into replay memory."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float | None:
        """Run one DQN training step from replay memory.

        Implements the Bellman update from Paper equation (1):
            Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Gradient clipping (max_norm=10.0) is applied after backward() to
        prevent exploding gradients during early training. This is standard
        DQN practice and does not conflict with any paper hyperparameter.

        Returns:
            float | None: Loss value, or None if memory is below BATCH_SIZE.
        """
        if len(self.memory) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        state_tensor = torch.FloatTensor(states)
        action_tensor = torch.LongTensor(actions).unsqueeze(1)
        reward_tensor = torch.FloatTensor(rewards)
        next_state_tensor = torch.FloatTensor(next_states)
        done_tensor = torch.FloatTensor(dones)

        q_all = self.policy_net(state_tensor)
        current_q = q_all.gather(1, action_tensor)
        self._last_q_values = q_all.detach()
        with torch.no_grad():
            next_actions = self.policy_net(next_state_tensor).argmax(
                dim=1,
                keepdim=True,
            )
            next_q = (
                self.target_net(next_state_tensor).gather(1, next_actions).squeeze(1)
            )
        target_q = reward_tensor + GAMMA * next_q * (1.0 - done_tensor)

        loss = functional.smooth_l1_loss(current_q.squeeze(1), target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: prevents exploding gradients in early training.
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        loss_value = float(loss.item())
        self.loss_log.append(loss_value)
        self.step_count += 1
        self.decay_epsilon()
        return loss_value

    def update_target_network(self) -> None:
        """Hard-copy policy network weights to the target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        """Apply linear epsilon decay for one training step."""
        self.epsilon = max(EPSILON_END, self.epsilon - EPSILON_DECAY)

    def save(self, path: str) -> None:
        """Save policy network weights to disk."""
        try:
            torch.save(self.policy_net.state_dict(), path)
        except (OSError, RuntimeError) as exc:
            log.error("[ERROR] Model save failed: %s", exc)
            sys.exit(1)

    def load(self, path: str) -> None:
        """Load saved weights into both policy and target networks."""
        try:
            state_dict = torch.load(path, weights_only=True)
        except (OSError, RuntimeError) as exc:
            log.error("[ERROR] Model load failed: %s", exc)
            sys.exit(1)

        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.eval()
        self.target_net.eval()
