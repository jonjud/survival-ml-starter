"""Train a Q-learning agent on SurvivalEnv and compare it with a random agent."""
import random

import numpy as np

from survival_env import SurvivalEnv, ACTION_NAMES

# ---- hyperparameters: try changing these and see what happens! ----
EPISODES = 600000    # more state (water) -> needs much more practice
ALPHA = 0.2          # learning rate: how fast new info overwrites old
GAMMA = 0.95         # discount: how much future reward matters
EPS_START = 1.0      # exploration: 1.0 = fully random
EPS_END = 0.05
EPS_DECAY_EPISODES = 400000


def evaluate(env, policy, episodes=300):
    """Average survival steps and food eaten per episode (no learning)."""
    steps, foods = [], []
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            s, _, done = env.step(policy(s))
        steps.append(env.steps)
        foods.append(env.foods_eaten)
    return float(np.mean(steps)), float(np.mean(foods))


def main():
    env = SurvivalEnv(size=5, seed=0)  # smaller map keeps the Q-table learnable
    rng = random.Random(1)
    Q = np.zeros((env.n_states, env.n_actions))

    for ep in range(EPISODES):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / EPS_DECAY_EPISODES)
        s = env.reset()
        done = False
        while not done:
            # epsilon-greedy: sometimes explore, otherwise use what we know
            if rng.random() < eps:
                a = rng.randrange(env.n_actions)
            else:
                a = int(np.argmax(Q[s]))
            s2, r, done = env.step(a)
            # the Q-learning update (the heart of it all)
            target = r + (0.0 if done else GAMMA * np.max(Q[s2]))
            Q[s, a] += ALPHA * (target - Q[s, a])
            s = s2

        if (ep + 1) % 100000 == 0:
            avg_steps, avg_food = evaluate(env, lambda st: int(np.argmax(Q[st])), 100)
            print(f"episode {ep+1:>6} | eps={eps:.2f} | survive={avg_steps:5.1f} steps | food={avg_food:4.1f}")

    # ---- final comparison ----
    rand_rng = random.Random(2)
    r_steps, r_food = evaluate(env, lambda st: rand_rng.randrange(env.n_actions))
    q_steps, q_food = evaluate(env, lambda st: int(np.argmax(Q[st])))
    print("\n=== result (higher is better) ===")
    print(f"random agent : survive {r_steps:5.1f}/{env.max_steps} steps, food {r_food:4.1f}")
    print(f"trained agent: survive {q_steps:5.1f}/{env.max_steps} steps, food {q_food:4.1f}")

    # ---- watch the trained agent play a few steps ----
    print("\n=== demo ===")
    s = env.reset()
    for _ in range(8):
        a = int(np.argmax(Q[s]))
        print(env.render(), "->", ACTION_NAMES[a], "\n")
        s, _, done = env.step(a)
        if done:
            break

    np.save("q_table.npy", Q)
    print("saved q_table.npy")


if __name__ == "__main__":
    main()
