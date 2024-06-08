import numpy as np

from genetic.genetic_algorithm import GeneticAlgorithm
from vis.visualization import animate

if __name__ == "__main__":
    dims = 2
    n = 50
    num_runs = 50
    num_particles = 150
    step_size = 0.01
    func_name = "rastrigin"
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
    print(f"Solutions: {ga.solutions}")
    print(f"Best solutions {ga.min_solution_in_rounds}")
    print(f"Best solution {min(ga.min_solution_in_rounds)}")
    # todo something is still of here: IndexError: index 50 is out of bounds for axis 0 with size 50
    animate(solutions=target_array, n=n, func_name=func_name)
