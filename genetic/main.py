"""


"""

import sys
sys.path.append('/home/tobias/mygits/python_vis/genetic')
# error+2d vis
from vis.PSOVisualization import Particle_2DVis
from genetic.genetic_algorithm import GA
from genetic.background_function import *


if __name__ == '__main__':
    dims = 2
    n = 10 # ~ value height
    num_runs = 50
    num_particles = 20
    func_name = 'schaffer_f6'

    target_array = np.zeros((num_runs, num_particles, dims))

    ga = GA(num_particles=num_particles,
            dims=dims,
            n=n,
            step_size=.2)
    ga.set_func_name(func_name)
    ga.run(target_array=target_array, num_runs=num_runs)

    # for t in target_array:
    #     print(t)

    if dims == 2:
        vis2d = Particle_2DVis(n=n, num_runs=num_runs)
        values, t_m = background_function[func_name]()
        vis2d.set_data(target_array, values, t_m)
        vis2d.plot_contours()
        vis2d.set_point_size(3.5)
        vis2d.animate()
