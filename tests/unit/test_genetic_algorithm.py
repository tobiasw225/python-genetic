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
def test_crossover(ga, dims, seed):
    ga.dims = dims
    seed()
    res = ga.crossover(0, 9)
    if dims == 2:
        assert (res == np.array([-6.65614741, -6.65614741])).all()
    else:
        assert (res == np.array([-36.38783025, -6.65614741])).all()


def test_mutation(ga, seed):
    seed()
    res = ga.mutation(
        weight=np.float64(0.09816326530612246), row=np.array([38.54407894, 38.54407894])
    )
    assert (np.array([38.54407894, 38.494997307346935]) == res).all()


def test_next_generation():
    assert False, "implement me"


def test__run_iteration():
    assert False, "implement me"


def test_run():
    assert False, "implement me"
