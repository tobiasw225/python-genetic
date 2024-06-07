from genetic.genetic_algorithm import run_on_function
from vis.draw import animate_solutions
from vis.ErrorVis import ErrorVis

if __name__ == "__main__":
    dims = 2
    n = 10
    num_runs = 300
    num_particles = 50
    step_size = 0.1
    func_name = "rastrigin"
    # todo something is still broken
    solution, min_solution_in_rounds, target_array = run_on_function(
        dims, n, num_runs, func_name, num_particles, step_size
    )
    vis = ErrorVis(
        interactive=False,
        log_scale=False,
        xlim=num_runs,
        ylim=max(min_solution_in_rounds),
    )
    # vis.plot_data(range(num_runs), min_solution_in_rounds)
    # note: do not use pip install matplotlib in ubuntu, use apt instead
    # apt-get install tcl-dev tk-dev python-tk python3-tk
    animate_solutions(dims, n, num_runs, func_name, 10, 10, target_array)
