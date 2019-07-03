"""


"""

import sys
sys.path.append('/home/tobias/mygits/python_vis/genetic')
# error+2d vis
from vis.PSOVisualization import Particle_2DVis
from genetic.ga import GA
from genetic.background_function import *


if __name__ == '__main__':
    dims = 2
    n = 100 # ~ value height
    num_runs = 100
    num_particles = 30
    func_name = 'styblinsky_tang'

    target_array = np.zeros((num_runs, num_particles, dims))

    ga = GA(num_particles=num_particles,
            dims=dims,
            max_val=n,
            step_size=.05)
    ga.run(target_array=target_array, max_runs=num_runs)

    if dims == 2:
        vis2d = Particle_2DVis(n=n, num_runs=num_runs)
        values, t_m = generate_2d_background(func_name, n)
        vis2d.set_data(target_array, values, t_m)
        vis2d.plot_contours()
        vis2d.set_point_size(3.5)
        vis2d.animate()
