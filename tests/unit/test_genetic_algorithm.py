from unittest.mock import MagicMock

import numpy as np
import pytest


def test_fitness_of_population(ga):
    ga.func = MagicMock()
    _ = ga.fitness_of_population(sub_pop=list(range(ga.num_particles)))
    for i in range(ga.num_particles):
        assert (ga.swarm[i, :] == ga.func.call_args_list[i].args[0]).all()


def test_diversity_of_sub_population(ga):
    sub_pop = np.array([7, 5, 8, 0, 3, 1])
    div = ga.diversity_of_sub_population(sub_pop)
    assert (
        div
        == np.array(
            [
                6.722815923142539,
                6.898005574500473,
                3.9776175830261784,
                3.982341006317326,
                5.460908507210796,
                4.861933185739215,
            ]
        )
    ).all()


@pytest.mark.parametrize("dims", (2, 5))
def test_crossover(ga, dims):
    ga.dims = dims
    res = ga.crossover(0, 9)
    if dims == 2:
        assert (res == np.array([-6.65614741, -6.65614741])).all()
    else:
        assert (res == np.array([-36.38783025, -6.65614741])).all()


def test_mutation(ga):
    res = ga.mutation(
        weight=np.float64(0.09816326530612246), row=np.array([38.54407894, 38.54407894])
    )
    assert (np.array([38.54407894, 38.494997307346935]) == res).all()


@pytest.fixture
def swarm_before_first_iteration():
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


def test_next_generation(ga, swarm_before_first_iteration):
    assert (ga.swarm == swarm_before_first_iteration).all()
    ga.next_generation(i=0, indices=[1, 5, 0, 4], j=0, weak_indices=[3, 2])
    assert (
        ga.swarm
        == np.array(
            [
                [-33.69644182, -30.42319644],
                [-35.31770945, 5.71282286],
                [24.1841216, 44.99983344],
                [-48.59330915, 25.78939106],
                [5.99316147, -2.72579467],
                [-7.80891628, -7.80891628],
                [5.46702377, -25.47720454],
                [35.98856979, -14.89162702],
                [-15.41348534, -17.63940369],
                [-7.80891628, -7.80891628],
            ]
        )
    ).all()


def test__run_iteration(ga, swarm_before_first_iteration):
    target_array = np.zeros((ga.num_runs, ga.num_particles, ga.dims))
    assert (ga.swarm == swarm_before_first_iteration).all()
    ga._run_interation(j=0, target_array=target_array)
    assert (
        ga.swarm
        == np.array(
            [
                [-33.69644182, -30.42319644],
                [-35.31770945, -35.31770945],
                [44.99983344, 44.99983344],
                [-48.59330915, 25.78939106],
                [5.99316147, -2.72579467],
                [-35.31770945, -35.31770945],
                [5.46702377, -25.47720454],
                [35.98856979, -14.89162702],
                [-15.41348534, -17.63940369],
                [-36.38783025, -6.65614741],
            ]
        )
    ).all()


def test_run(ga, swarm_before_first_iteration):
    target_array = np.zeros((ga.num_runs, ga.num_particles, ga.dims))
    assert (ga.swarm == swarm_before_first_iteration).all()
    ga.run(target_array=target_array)
    assert (
        ga.swarm
        == np.array(
            [
                [25.64122779469388, 25.64122779469388],
                [25.64122779469388, 25.64122779469388],
                [25.59122779469388, 25.64122779469388],
                [25.79030942734694, 25.74030942734694],
                [25.541227794693878, 25.590309427346938],
                [25.541227794693878, 25.59122779469388],
                [25.74030942734694, 25.59122779469388],
                [25.64030942734694, 25.69030942734694],
                [25.590309427346938, 25.64030942734694],
                [25.74030942734694, 25.69122779469388],
            ]
        )
    ).all()
    assert ga.solutions == [
        1326.3598007935577,
        1326.3598007935577,
        1325.2639094788697,
        1332.943531258501,
        1323.1274180724272,
        1323.1467352717204,
        1327.7949844086154,
        1327.5220081969985,
        1325.2235727347274,
        1330.0974666333661,
    ]
