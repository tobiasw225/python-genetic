# __filename__: ga.py
#
# __description__: methods for ga
#
# Created by Tobias Wenzel in ~ Summer 2019
# Copyright (c) 2019 Tobias Wenzel

import numpy as np
from scipy.spatial import distance

from genetic.eval_funcs import eval_function


class GeneticAlgorithm:
    def __init__(
        self,
        num_particles: int,
        dims: int,
        max_val: int,
        step_size: float,
        func_name: str,
        max_runs: int,
    ):
        self.num_particles = num_particles
        self.dims = dims
        self.max_val = max_val
        self.step_size = step_size * max_val
        # points in [0, 1)
        self.swarm = np.random.random((num_particles, dims))
        # [-n, n)
        self.swarm = 2 * max_val * self.swarm - max_val
        self.func = eval_function(func_name)
        self.solutions = None
        self.min_solution_in_rounds = []
        # size of random sub populations.
        n_sub_population = self.num_particles // 2
        if n_sub_population % 2 != 0:
            n_sub_population += 1
        self.n_sub_population = n_sub_population
        # number of elements selected each round.
        fittest = n_sub_population // 2
        if fittest % 2 != 0:
            fittest += 1
        self.num_fittest = fittest
        self.num_weakest = self.num_fittest // 2
        self.num_runs = max_runs
        weights = np.linspace(1, 10, max_runs) / 100
        self.weights = np.flip(weights)

    def fitness_of_population(self, sub_pop: list) -> list:
        return [self.func(self.swarm[i, :]) for i in sub_pop]

    def diversity_of_sub_population(self, sub_pop: np.array) -> np.array:
        def diversity(row):
            return np.sum([distance.cosine(row, self.swarm[j, :]) for j in sub_pop])

        return np.array([diversity(self.swarm[i, :]) for i in sub_pop])

    def crossover(self, xx: int, xy: int) -> np.ndarray:
        # shuffle xx, xy
        xx, xy = np.random.choice([xx, xy], 2)
        row = self.swarm[xy, :]
        if self.dims > 2:
            ri = np.random.randint(1, self.dims + 1)
            row[ri:] = self.swarm[xx, ri:]
        elif self.dims == 2:
            # 0, -1 or 1, 0
            i = np.random.randint(0, 2)
            row[i] = self.swarm[xx, i - 1]
        return row

    def mutation(self, row: np.ndarray, weight: np.float64) -> np.ndarray:
        f = self.step_size * weight
        if np.random.randint(0, 2):
            f *= -1
        row[np.random.randint(0, self.dims)] += f
        return row

    def run(self, target_array: np.ndarray):
        # todo indices look odd
        for j in range(self.num_runs):
            self._run_interation(j, target_array)
        self.solutions = self.fitness_of_population(list(range(self.num_particles)))

    def _run_interation(
        self,
        j,
        target_array,
    ):
        particle_indices = range(self.num_particles)
        target_array[j, :] = self.swarm
        # choose random subpopulation
        sub_pop = np.random.choice(
            particle_indices, self.n_sub_population, replace=False
        )
        solutions = self.fitness_of_population(sub_pop)
        # print(f"{np.min(solutions):.2f}")
        self.min_solution_in_rounds.append(np.min(solutions))
        diversity = self.diversity_of_sub_population(sub_pop)
        # choose fittest and most diverse elements
        # indices in subpopulation.
        measure = solutions + (1 - self.weights[j] * diversity)
        indices = list(np.argpartition(measure, self.num_fittest)[: self.num_fittest])
        weak_indices = list(
            np.argpartition(measure, -self.num_weakest)[-self.num_weakest :]
        )
        z = 0
        for i in range(0, len(indices), 2):
            self.next_generation(i, indices, z, weak_indices)
            z += 1
        # set maximum/ minimum (so particles can't escape area)
        self.swarm = np.clip(self.swarm, -self.max_val, self.max_val)

    def next_generation(self, i, indices, j, weak_indices):
        row = self.crossover(xx=indices[i], xy=indices[i + 1])
        if np.random.randint(0, 2):
            row = self.mutation(row, self.weights[j])
        # replace weakest elements in sub-population
        # no permanent solution.
        if np.random.randint(0, 2):
            self.swarm[np.random.randint(0, self.num_particles), :] = row
        else:
            self.swarm[weak_indices[j], :] = row
