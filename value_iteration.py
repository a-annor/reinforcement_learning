from cgitb import small
from typing import Callable
from matplotlib.pyplot import grid
from tqdm import tqdm
import numpy as np
import sys
import random
from model import Model, Actions


def value_iteration(
    model: Model, n_episodes: int = 100, asynchronous: bool = False, threshold=1e-4
):

    V = np.zeros((model.num_states,))
    episode_count = 0
    diff_per_episode = []

    def compute_value(s, a, reward: Callable):
        return np.sum(
            [
                model.transition_probability(s, s_, a)
                * (reward(s, a) + model.gamma * V[s_])
                for s_ in model.states
            ]
        )

    pi = np.zeros((model.num_states,), dtype=int)  # initialise policy just for tracking
    for i in tqdm(range(n_episodes)):
        episode_count += 1
        delta = 0
        pi_old = np.copy(pi)

        if asynchronous:
            for _ in range(len(model.states)):
                s = random.choice(model.states)  # randomly choose a state like in pg85
                action_values = [compute_value(s, a, model.reward) for a in Actions]
                max_value = max(action_values)
                delta = max(delta, abs(V[s] - max_value))
                V[s] = max_value  # update straight away
        else:
            V_new = np.copy(V)
            for s in model.states:
                action_values = [compute_value(s, a, model.reward) for a in Actions]
                V_new[s] = max(action_values)
                delta = max(delta, abs(V[s] - V_new[s]))  # STOPPING criteria
            V = V_new  # apply all updates at once

        for s in model.states:
            pi[s] = np.argmax(
                [
                    sum(
                        model.transition_probability(s, s_, a)
                        * (model.reward(s, a) + model.gamma * V[s_])
                        for s_ in model.states
                    )
                    for a in Actions
                ]
            )

        # diffs = sum(pi_old != pi)
        diff_per_episode.append(delta)

        if delta < threshold:  # STOPPING criteria
            break

    print(f"Value Iteration converged after {episode_count} episode.")
    # print(f"Value Iteration converged after {iteration_count} iterations.")

    # for s in model.states:
    #     # pi_old= np.copy(pi)
    #     pi[s] = np.argmax(
    #         [
    #             sum(
    #                 model.transition_probability(s, s_, a)
    #                 * (model.reward(s, a) + model.gamma * V[s_])
    #                 for s_ in model.states
    #             )
    #             for a in Actions
    #         ]
    #     )
    # abs(pi_old- pi)
    # diffs = sum((pi_old != pi))
    # diff_per_episode.append(diffs)

    return V, pi, np.array(diff_per_episode)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from world_config import cliff_world, small_world, grid_world
    from plot_vp import plot_vp

    if len(sys.argv) > 1:
        if sys.argv[1] == "cliff":
            model = Model(cliff_world)
        elif sys.argv[1] == "small":
            model = Model(small_world)
        elif sys.argv[1] == "grid":
            model = Model(grid_world)
        else:
            print("Error: unknown world type:", sys.argv[1])
    else:
        model = Model(small_world)

    if len(sys.argv) > 2:
        n_episodes = int(sys.argv[2])
        (
            V,
            pi,
        ) = value_iteration(model, n_episodes=n_episodes)
    else:
        V, pi = value_iteration(model)

    plot_vp(model, V, pi)
    plt.show()
