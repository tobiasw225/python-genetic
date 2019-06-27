import numpy as np
from scipy.spatial import distance


import sys
sys.path.append('/home/tobias/mygits/python_vis/genetic')
# error+2d vis
from vis.PSOVisualization import Particle_2DVis
from genetic.background_function import *


def eval_rastrigin(row: np.ndarray):
    """
        apply rastrigin-function to row

    :param row:
    :return:
    """
    f = lambda d: d ** 2 - 10 * np.cos(np.pi * d) + 10
    f_arr = np.frompyfunc(f, 1, 1)
    return np.sum(f_arr(row))


def eval_square(row: np.ndarray):
    """

    :param row:
    :return:
    """
    f_arr = np.frompyfunc(lambda d: d ** 2, 1, 1)
    return np.sum(f_arr(row))


def eval_rosenbrock(row: np.ndarray):
    """

    :param row:
    :return:
    """
    assert len(row) == 2
    a = 1. - row[0]
    b = row[1]- row[0]*row[0]
    return a*a + b*b*100


def eval_eggholder(row: np.ndarray):
    """

    :param row:
    :return:
    """
    return -(row[1]+47)*np.sin(np.sqrt(np.abs(row[0]/2+ (row[1]+47))))\
    -row[0]*np.sin(np.sqrt(np.abs(row[0]-(row[1]+47))))


def eval_styblinsky_tang(row: np.ndarray):
    """
        not working
    :param row:
    :return:
    """
    f = lambda d: d**4-(16*d**2)+(5*d)
    f_arr = np.frompyfunc(f, 1, 1)
    return np.sum(f_arr(row))/2


class GA:
    def __init__(self,
                 num_particles: int,
                 dims: int,
                 max_val: int,
                 step_size: float):
        """

        :param num_particles:
        :param dims:
        :param max_val:
        :param step_size:
        """
        self.num_particles = num_particles
        self.dims = dims
        self.max_val = max_val
        self.step_size = step_size * max_val
        # points in [0, 1)
        self.swarm = np.random.random((num_particles, dims))
        # [-n, n)
        self.swarm = 2 * max_val * self.swarm - max_val
        self.func = eval_rastrigin

    def fitness_of_sub_population(self, sub_pop: list):
        """

        :param sub_pop:
        :return:
        """
        fitness = []
        for i in sub_pop:
            fitness.append(self.func(self.swarm[i,:]))
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
        row = self.swarm[xy,:]
        if self.dims > 2:
            ri = np.random.randint(1, self.dims + 1)

            row[ri:] = self.swarm[xx, ri:]
        elif self.dims == 2:
            # 0, -1 or 1, 0
            i = np.random.randint(0, 2)
            row[i] = self.swarm[xx, i-1]
        return row

    def mutation(self, row: np.ndarray, weight: np.float64) -> np.ndarray:
        """

        :param row:
        :param weight:
        :return:
        """
        f = (self.step_size * weight)
        if np.random.randint(0, 2):
            f *= (-1)
        row[np.random.randint(0, self.dims)] += f
        # set maximum (so particles can't escape area)
        for d in range(self.dims):
            row[d] = min(
                max(row[d], -self.max_val),
                self.max_val)
        return row

    def run(self, max_runs: int,
            target_array: np.ndarray):
        """

        :param max_runs:
        :return:
        """
        particle_indices = range(self.num_particles)
        # size of random sub-populations.
        n_sub_population = self.num_particles//2
        if n_sub_population % 2 != 0:
            n_sub_population += 1
        # number of elements selected each round.
        fittest = n_sub_population //2
        if fittest % 2 != 0:
            fittest += 1

        weights = np.linspace(1, 1000, max_runs) / 100
        weights = np.flip(weights)

        for j in range(max_runs):
            target_array[j, :] = self.swarm
            # choose random subpopulation
            sub_pop = np.random.choice(particle_indices,
                                       n_sub_population, replace=False)

            solutions = self.fitness_of_sub_population(sub_pop)
            print(f"{np.min(solutions):.2f}")
            diversity = self.diversity_of_sub_population(sub_pop)
            # choose fittest and most diverse elements
            # indices in sub-population.
            measure = solutions-(1-weights[j]*diversity)
            indices = list(np.argpartition(measure,
                                           fittest)[:fittest])
            weakest = fittest//2
            weak_indices = list(np.argpartition(measure,
                                                -weakest)[-weakest:])
            j = 0
            for i in range(0, len(indices), 2):
                xx, xy = indices[i], indices[i+1]
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

        solutions = self.fitness_of_sub_population(particle_indices)


if __name__ == '__main__':
    max_runs = 100
    dims = 2
    num_particles = 10
    max_val = 5
    func_name = 'rastrigin'
    print(background_function.keys())
    ga = GA(num_particles=num_particles,
            dims=dims,
            max_val=max_val,
            step_size=0.005)
    target_array = np.zeros((max_runs, num_particles, dims))

    ga.run(max_runs=max_runs, target_array=target_array)

    if dims == 2:
        vis2d = Particle_2DVis(n=max_val, num_runs=max_runs)
        values, t_m = background_function[func_name]()
        vis2d.set_data(target_array, values, t_m)
        vis2d.plot_contours()
        vis2d.set_point_size(2.5)
        vis2d.animate()
