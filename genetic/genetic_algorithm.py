# __filename__: ga.py
#
# __description__: methods for ga
#
# __remark__:
#
# __todos__:
#
# Created by Tobias Wenzel in ~ Summer 2019
# Copyright (c) 2019 Tobias Wenzel

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

    def fitness_of_sub_population(self, sub_pop: list):
        """

        :param sub_pop:
        :return:
        """
        fitness = []
        for i in sub_pop:
            fitness.append(self.func(self.swarm[i, :]))
        return fitness

    def diversity_of_sub_population(self, sub_pop: list) -> np.array:
        """

        :param sub_pop:
        :return:
        """

        def diversity(row):
            ds = []
            for j in sub_pop:
                ds.append(distance.cosine(row, self.swarm[j, :]))
            return np.sum(ds)

        diversities = []
        for i in sub_pop:
            diversities.append(diversity(self.swarm[i, :]))
        return np.array(diversities)

    def crossover(self, xx: int, xy: int) -> np.ndarray:
        """

        :param xx:
        :param xy:
        :return:
        """
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
        """

        :param row:
        :param weight:
        :return:
        """
        f = self.step_size * weight
        if np.random.randint(0, 2):
            f *= -1
        row[np.random.randint(0, self.dims)] += f
        return row

    def mutate_flip(self, row: np.ndarray):
        return np.flip(row)

    def run(self, max_runs: int, target_array: np.ndarray):
        particle_indices = range(self.num_particles)
        # size of random sub-populations.
        n_sub_population = self.num_particles // 2
        if n_sub_population % 2 != 0:
            n_sub_population += 1
        # number of elements selected each round.
        fittest = n_sub_population // 2
        if fittest % 2 != 0:
            fittest += 1

        weights = np.linspace(1, 10, max_runs) / 100
        weights = np.flip(weights)

        for j in range(max_runs):
            target_array[j, :] = self.swarm
            # choose random sub-population
            sub_pop = np.random.choice(
                particle_indices, n_sub_population, replace=False
            )

            solutions = self.fitness_of_sub_population(sub_pop)
            # print(f"{np.min(solutions):.2f}")
            self.min_solution_in_rounds.append(np.min(solutions))
            diversity = self.diversity_of_sub_population(sub_pop)
            # choose fittest and most diverse elements
            # indices in sub-population.
            measure = solutions + (1 - weights[j] * diversity)
            indices = list(np.argpartition(measure, fittest)[:fittest])
            weakest = fittest // 2
            weak_indices = list(np.argpartition(measure, -weakest)[-weakest:])
            j = 0
            for i in range(0, len(indices), 2):
                xx, xy = indices[i], indices[i + 1]
                row = self.crossover(xx, xy)
                if np.random.randint(0, 2):
                    row = self.mutation(row, weights[j])
                # replace weakest elements in sub-population
                # no permanent solution.
                if np.random.randint(0, 2):
                    ri = np.random.randint(0, self.num_particles)
                    self.swarm[ri, :] = row
                else:
                    self.swarm[weak_indices[j], :] = row

                j += 1

            # set maximum/ minimum (so particles can't escape area)
            self.swarm = np.clip(self.swarm, -self.max_val, self.max_val)

        self.solutions = self.fitness_of_sub_population(list(particle_indices))


def run_on_function(
    dims: int,
    n: int,
    num_runs: int,
    func_name: str,
    num_particles: int,
    step_size: float,
):
    target_array = np.zeros((num_runs, num_particles, dims))
    ga = GeneticAlgorithm(
        num_particles=num_particles,
        dims=dims,
        max_val=n,
        step_size=step_size,
        func_name=func_name,
    )
    ga.run(target_array=target_array, max_runs=num_runs)
    return ga.solutions, ga.min_solution_in_rounds, target_array
