import numpy as np
from tqdm import tqdm
from model import Model, Actions
from typing import Callable
from matplotlib.pyplot import grid
import sys
import random
from model import Model, Actions


def sarsa(
    model: Model, alpha: float = 0.1, n_episodes: int = 100, epsilon=0.1, max_steps=50
):

    Q = np.zeros((model.num_states, len(Actions)))  # initialize Q
    rewards_per_episode = []
    policy_diffs = []
    pi_old = np.argmax(Q, axis=1)

    for i in tqdm(range(n_episodes)):
        s = model.start_state  # initial states
        a = epsilon_greedy(
            Q, s, epsilon
        )  # Choose A from S using policy from Q e greedy
        # print("initial a: ", a)
        total_reward = 0
        for step in range(max_steps):
            # print("current action: ", Actions(a))
            # print("current state: ", s)
            s_prime = model.next_state(s, Actions(a))  # next state based on likelihood
            R = model.reward(s, Actions(a))

            total_reward += R
            a_prime = epsilon_greedy(
                Q, s_prime, epsilon
            )  # next action using current policy

            # SARSA update
            Q[s, a] += alpha * (R + model.gamma * Q[s_prime, a_prime] - Q[s, a])

            # If terminal state stop
            if s_prime == model.goal_state:  # or s_prime == model.fictional_end_state:
                # R = model.reward(s, Actions(a))
                # Q[s, a] += alpha* (R  - Q[s, a])
                # terminal = True
                break

            else:
                s, a = s_prime, a_prime
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


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from world_config import cliff_world, small_world, grid_world
#     from plot_vp import plot_vp

#     if len(sys.argv) > 1:
#         if sys.argv[1] == 'cliff':
#             model = Model(cliff_world)
#         elif sys.argv[1] == 'small':
#             model = Model(small_world)
#         elif sys.argv[1] == 'grid':
#             model = Model(grid_world)
#         else:
#             print("Error: unknown world type:", sys.argv[1])
#     else:
#         model = Model(small_world)

#     if len(sys.argv) > 2:
#         n_episodes = int(sys.argv[2])
#         V, pi, rewards_per_episode, policy_diffs = sarsa(model, alpha=0.1, epsilon=0.1, n_episodes=n_episodes, max_steps=50)
#     else:
#         # V, pi, rewards_per_episode, policy_diffs = sarsa(model, alpha=0.1, epsilon=0.1, n_episodes=n_episodes, max_steps=50)

#     plot_vp(model, V, pi)
#     plt.show()
