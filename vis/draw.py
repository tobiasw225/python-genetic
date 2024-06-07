from vis.ParticleVisualization import Particle2DVis


def animate_solutions(dims, n, num_runs, func_name, x_limit, y_limit, solutions):
    assert dims == 2
    vis = Particle2DVis(
        n=n, num_runs=num_runs, func_name=func_name, x_limit=x_limit, y_limit=y_limit
    )
    for i in range(num_runs):
        vis.animate(solution=solutions[i, :])
