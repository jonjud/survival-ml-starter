"""DQN: replace the Q-table with a small neural network (pure numpy, no PyTorch).

Q-table : a lookup table, one row per state  -> explodes when the world grows.
DQN     : a function  observation(10 numbers) -> 4 Q-values. It *generalizes*:
          states it has never seen look similar to ones it has.

Extra tricks that make it stable:
  * replay buffer  - learn from random old experiences, not just the latest
  * target network - a slowly-updated copy used to compute the learning target
"""
import random
import time

import numpy as np

from survival_env import SurvivalEnv

# ---- hyperparameters ----
EPISODES = 3000
GAMMA = 0.95
LR = 1e-3
BATCH = 64
BUFFER = 50_000
WARMUP = 1_000            # steps of pure random play before learning starts
TRAIN_EVERY = 2           # do one learning step every N env steps
TARGET_SYNC = 500         # copy online net -> target net every N learning steps
EPS_START, EPS_END, EPS_DECAY_EPISODES = 1.0, 0.05, 2000
HIDDEN = 64


class MLP:
    """in -> HIDDEN -> HIDDEN -> out, ReLU, trained with Adam. Backprop by hand."""

    def __init__(self, n_in, n_hidden, n_out, rng):
        def init(i, o):  # He initialisation
            return rng.standard_normal((i, o)) * np.sqrt(2.0 / i)
        self.p = {
            "W1": init(n_in, n_hidden), "b1": np.zeros(n_hidden),
            "W2": init(n_hidden, n_hidden), "b2": np.zeros(n_hidden),
            "W3": init(n_hidden, n_out) * 0.1, "b3": np.zeros(n_out),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}  # Adam state
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x):
        p = self.p
        z1 = x @ p["W1"] + p["b1"]; a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]; a2 = np.maximum(z2, 0)
        q = a2 @ p["W3"] + p["b3"]
        return q, (x, z1, a1, z2, a2)

    def train_step(self, x, actions, targets, lr):
        q, (x, z1, a1, z2, a2) = self.forward(x)
        B = len(x)
        idx = np.arange(B)
        # gradient of a Huber loss on the chosen action's Q-value only
        dq = np.zeros_like(q)
        dq[idx, actions] = np.clip(q[idx, actions] - targets, -1.0, 1.0) / B
        p = self.p
        g = {}
        g["W3"] = a2.T @ dq;            g["b3"] = dq.sum(0)
        dz2 = (dq @ p["W3"].T) * (z2 > 0)
        g["W2"] = a1.T @ dz2;           g["b2"] = dz2.sum(0)
        dz1 = (dz2 @ p["W2"].T) * (z1 > 0)
        g["W1"] = x.T @ dz1;            g["b1"] = dz1.sum(0)
        # Adam update
        self.t += 1
        for k in p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t)
            vh = self.v[k] / (1 - 0.999 ** self.t)
            p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)

    def copy_from(self, other):
        for k in self.p:
            self.p[k] = other.p[k].copy()


def evaluate(env, act_fn, episodes=200):
    steps, foods = [], []
    for _ in range(episodes):
        env.reset()
        done = False
        while not done:
            _, _, done = env.step(act_fn(env.observation()))
        steps.append(env.steps)
        foods.append(env.foods_eaten)
    return float(np.mean(steps)), float(np.mean(foods))


def main():
    env = SurvivalEnv(size=5, seed=0)      # same map as train.py -> fair comparison
    rng = np.random.default_rng(0)
    py_rng = random.Random(1)
    net = MLP(env.N_FEATURES, HIDDEN, env.n_actions, rng)
    target = MLP(env.N_FEATURES, HIDDEN, env.n_actions, rng)
    target.copy_from(net)

    # replay buffer (ring buffer of numpy arrays)
    S = np.zeros((BUFFER, env.N_FEATURES)); S2 = np.zeros_like(S)
    A = np.zeros(BUFFER, dtype=int); R = np.zeros(BUFFER); D = np.zeros(BUFFER)
    n = 0
    total_steps = 0
    learn_steps = 0
    t0 = time.time()

    def greedy(obs):
        return int(np.argmax(net.forward(np.array([obs]))[0][0]))

    for ep in range(EPISODES):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * ep / EPS_DECAY_EPISODES)
        env.reset()
        obs = env.observation()
        done = False
        while not done:
            a = py_rng.randrange(env.n_actions) if py_rng.random() < eps else greedy(obs)
            _, r, done = env.step(a)
            obs2 = env.observation()
            i = n % BUFFER
            S[i], A[i], R[i], S2[i], D[i] = obs, a, r, obs2, float(done)
            n += 1
            total_steps += 1
            obs = obs2

            if n >= WARMUP and total_steps % TRAIN_EVERY == 0:
                size = min(n, BUFFER)
                b = rng.integers(0, size, BATCH)
                q_next = target.forward(S2[b])[0].max(axis=1)
                y = R[b] + GAMMA * q_next * (1.0 - D[b])
                net.train_step(S[b], A[b], y, LR)
                learn_steps += 1
                if learn_steps % TARGET_SYNC == 0:
                    target.copy_from(net)

        if (ep + 1) % 500 == 0:
            st, fd = evaluate(env, greedy, 100)
            print(f"episode {ep+1:>5} | eps={eps:.2f} | survive={st:5.1f} steps | "
                  f"food={fd:4.1f} | {time.time()-t0:4.0f}s")

    rand_rng = random.Random(2)
    r_steps, r_food = evaluate(env, lambda o: rand_rng.randrange(env.n_actions))
    d_steps, d_food = evaluate(env, greedy)
    print("\n=== result (higher is better) ===")
    print(f"random agent : survive {r_steps:5.1f}/{env.max_steps} steps, food {r_food:4.1f}")
    print(f"DQN agent    : survive {d_steps:5.1f}/{env.max_steps} steps, food {d_food:4.1f}")

    np.savez("dqn_weights.npz", **net.p)
    print("saved dqn_weights.npz")


if __name__ == "__main__":
    main()
