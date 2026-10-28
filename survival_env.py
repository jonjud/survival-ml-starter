"""Mini survival environment (grid world).

The agent walks on a small grid. Every step costs 1 energy.
Stepping on food restores energy (and food respawns elsewhere).
Water also drops by 1 every step; the lake (W, bottom-right corner) refills it.
Energy or water hits 0 -> the agent dies.

This mimics the "find food before you starve" loop of a survival game,
but is small enough for a tabular Q-learning agent to solve in seconds.
"""
import random

# action index -> (dx, dy)
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
ACTION_NAMES = ["up", "down", "left", "right"]


class SurvivalEnv:
    def __init__(self, size=6, max_energy=15, food_energy=8, max_steps=100, seed=None):
        self.size = size
        self.max_energy = max_energy
        self.food_energy = food_energy
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.n_actions = len(ACTIONS)
        self.max_water = max_energy
        self.water_pos = (size - 1, size - 1)  # fixed lake in the bottom-right corner
        # state = (agent_x, agent_y, food_x, food_y, energy, water)
        self.n_states = size**4 * (max_energy + 1) * (self.max_water + 1)

    # ---- state encoding: tuple -> single integer index (for the Q-table) ----
    def _encode(self):
        s = self.size
        idx = self.ax
        idx = idx * s + self.ay
        idx = idx * s + self.fx
        idx = idx * s + self.fy
        idx = idx * (self.max_energy + 1) + self.energy
        idx = idx * (self.max_water + 1) + self.water
        return idx

    def _spawn_food(self):
        while True:
            fx, fy = self.rng.randrange(self.size), self.rng.randrange(self.size)
            if (fx, fy) != (self.ax, self.ay):
                self.fx, self.fy = fx, fy
                return

    def reset(self):
        self.ax = self.rng.randrange(self.size)
        self.ay = self.rng.randrange(self.size)
        self.energy = self.max_energy
        self.water = self.max_water
        self.steps = 0
        self.foods_eaten = 0
        self._spawn_food()
        return self._encode()

    def step(self, action):
        """Returns (next_state, reward, done)."""
        dx, dy = ACTIONS[action]
        self.ax = min(max(self.ax + dx, 0), self.size - 1)
        self.ay = min(max(self.ay + dy, 0), self.size - 1)
        self.energy -= 1
        self.water -= 1
        self.steps += 1

        reward = -0.05  # small cost per step -> encourages efficiency

        # drinking has NO reward on purpose: otherwise the agent would loop
        # in and out of the lake forever (reward hacking). Fear of dying
        # from thirst is motivation enough.
        if (self.ax, self.ay) == self.water_pos:
            self.water = self.max_water

        if (self.ax, self.ay) == (self.fx, self.fy):
            self.energy = min(self.energy + self.food_energy, self.max_energy)
            self.foods_eaten += 1
            reward += 1.0
            self._spawn_food()

        done = False
        if self.energy <= 0 or self.water <= 0:
            reward -= 5.0  # death penalty
            done = True
        elif self.steps >= self.max_steps:
            done = True  # survived the whole episode

        return self._encode(), reward, done

    def render(self):
        rows = []
        for y in range(self.size):
            row = ""
            for x in range(self.size):
                if (x, y) == (self.ax, self.ay):
                    row += "A "
                elif (x, y) == (self.fx, self.fy):
                    row += "F "
                elif (x, y) == self.water_pos:
                    row += "W "
                else:
                    row += ". "
            rows.append(row)
        return "\n".join(rows) + f"\nenergy={self.energy} water={self.water}"
