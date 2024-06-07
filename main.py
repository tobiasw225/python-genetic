import time

import numpy as np

from genetic.genetic_algorithm import GeneticAlgorithm
from vis.ScatterVisualizer import ScatterVisualizer
from vis.draw import animate_solutions

if __name__ == "__main__":
    dims = 2
    n = 50
    num_runs = 50
    num_particles = 150
    step_size = 0.01
    func_name = "eggholder"
    target_array = np.zeros((num_runs, num_particles, dims))
    ga = GeneticAlgorithm(
        num_particles=num_particles,
        dims=dims,
        max_val=n,
        step_size=step_size,
        func_name=func_name,
        max_runs=num_runs,
    )
    ga.run(target_array=target_array)
    print(ga.solutions)
    print(ga.min_solution_in_rounds)
    print(min(ga.min_solution_in_rounds))
    vis = ScatterVisualizer(
        interactive=False,
        log_scale=False,
        xlim=num_runs,
        ylim=max(ga.min_solution_in_rounds),
        offset=1,
    )
    vis.plot_data(range(num_runs), ga.min_solution_in_rounds)
    animate_solutions(dims, n, num_runs, func_name, 10, 10, target_array)
    time.sleep(5)
