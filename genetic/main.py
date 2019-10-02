"""


"""
import time

from vis.PSOVisualization import Particle2DVis
from genetic.ga import GA
from genetic.background_function import *


if __name__ == '__main__':
    dims = 2
    n = 100  # ~ value height
    num_runs = 100
    num_particles = 30
    func_name = 'rastrigin'

    target_array = np.zeros((num_runs, num_particles, dims))

    ga = GA(num_particles=num_particles,
            dims=dims,
            max_val=n,
            step_size=.05)
    ga.run(target_array=target_array, max_runs=num_runs)

    vis = Particle2DVis(n=n, num_runs=num_runs)
    _, background_function = generate_2d_background(func_name, n)
    vis.set_background_function(background_function)

    if dims == 2:
        for i in range(num_runs):
            # not stopping ?
            vis.animate(solution=target_array[i, :])
            time.sleep(.1)
