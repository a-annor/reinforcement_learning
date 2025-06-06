import numpy as np
from tqdm import tqdm
from model import Model, Actions
from typing import Callable
from matplotlib.pyplot import grid
import sys
import random
from model import Model, Actions


def qlearning(
    model: Model, alpha: float = 0.1, n_episodes: int = 100, epsilon=0.1, max_steps=50
):

    Q = np.zeros((model.num_states, len(Actions)))  # initialize Q
    rewards_per_episode = []
    policy_diffs = []
    pi_old = np.argmax(Q, axis=1)

    for i in tqdm(range(n_episodes)):
        s = model.start_state  # initial states
        total_reward = 0
        for step in range(max_steps):
            a = epsilon_greedy(
                Q, s, epsilon
            )  # Choose A from S using policy derived from Q, e-greedy
        
            s_prime = model.next_state(s, Actions(a))
            R = model.reward(s, Actions(a))
            total_reward += R

            # SARSA update
            Q[s, a] += alpha * (R + model.gamma * np.max(Q[s_prime, :]) - Q[s, a])

            if s_prime == model.goal_state:  # If terminal state stop
                break

            else:
                s = s_prime
        rewards_per_episode.append(total_reward)
        pi = np.argmax(Q, axis=1)  # get policy from Q
        diff = np.sum(pi_old != pi)  # differences in policy
        policy_diffs.append(diff)
        pi_old = np.copy(pi)

    # get policy from Q
    V = np.max(Q, axis=1)
    pi = np.argmax(Q, axis=1)
    return V, pi, rewards_per_episode, policy_diffs


def epsilon_greedy(Q, s, epsilon):

    if np.random.rand() < epsilon:
        return np.random.choice(
            len(Actions)
        )  # exploring (choosing random action index)
    return np.argmax(Q[s])  # exploiting (choosing index of the action that maximises Q)
