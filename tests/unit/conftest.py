import numpy as np
import pytest

from genetic.genetic_algorithm import GeneticAlgorithm


@pytest.fixture()
def seed():
    def _func():
        np.random.seed(1)

    return _func


@pytest.fixture
def pseudo_random_swarm():
    return np.array(
        [
            [-33.69644182, -30.42319644],
            [-35.31770945, 5.71282286],
            [24.1841216, 44.99983344],
            [-48.59330915, 25.78939106],
            [5.99316147, -2.72579467],
            [25.44051296, -7.80891628],
            [5.46702377, -25.47720454],
            [35.98856979, -14.89162702],
            [-15.41348534, -17.63940369],
            [-36.38783025, -6.65614741],
        ]
    )


@pytest.fixture
def ga(pseudo_random_swarm):
    ga = GeneticAlgorithm(
        num_particles=10,
        dims=2,
        max_val=50,
        step_size=0.01,
        func_name="rastrigin",
        max_runs=50,
    )
    ga.swarm = pseudo_random_swarm
    return ga
