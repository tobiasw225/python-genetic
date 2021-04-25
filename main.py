from genetic.background_function import background_function
from genetic.genetic_algorithm import run_on_function
from vis.PSOVisualization import Particle2DVis

if __name__ == '__main__':
    dims = 2
    n = 100  # ~ value height
    num_runs = 100
    num_particles = 10
    func_name = 'square'
    solution, target_array = run_on_function(dims, n, num_runs, func_name, num_particles)
    print(target_array)
    if dims == 2:
        vis = Particle2DVis(n=n, num_runs=num_runs, func_name=func_name)
        # background_function = background_function(func_name, n)
        # vis.set_background_function(background_function)
        for i in range(num_runs):
            # not stopping ?
            vis.animate(solution=target_array[i, :])
