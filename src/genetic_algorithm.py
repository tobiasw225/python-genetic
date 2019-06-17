import numpy as np
from scipy.spatial import distance
from particle import Particle



class GA:
    class __GA:
        def __init__(self,
                     num_particles: int=0,
                     dims: int=0,
                     n: int=0,
                     step_size: float=0.1):

            self.num_particles = num_particles
            self.dims = dims
            self.n = n  # feldausdehnung
            self.population = [Particle(n, dims) for _ in range(num_particles)]
            self.step_size = step_size*n
            self.k_pairs = round(len(self.population)* 0.1) # choose at at each round
            if (self.k_pairs % 2) != 0:
                self.k_pairs += 1

            self.func_name = ""

        def set_func_name(self, func_name=""):
            self.func_name = func_name

        def crossover_mutation(self,
                               pa: Particle,
                               pb: Particle,
                               weight: np.int64) -> Particle:
            """
                crossover elements of best particles and add some randomness
                for mutation. adjust the effect with the parameter step-size.

            :param pa:
            :param pb:
            :return:
            """
            assert len(pa.x) == len(pb.x)
            n = len(pa.x)
            # set the 2nd part of the first particle to the 2nd part
            # of the 2nd particle (
            pa.x[n//2:] = pb.x[n//2:]
            # mutate one dimension.
            pa.x[np.random.randint(0, len(pa.x))] += (self.step_size*weight)
            # set maximum (so particles can't escape area)
            for d in range(self.dims):
                pa.x[d] = min(max(pa.x[d], -self.n), self.n)
            return pa

        def diversity(self, j: int) -> np.array:
            """
                compute diversity of particle.

            :param j: index of particle
            :return:
            """
            x_0 = self.population[j].x
            ds = []
            for i, p in enumerate(self.population):
                ds.append(distance.cosine(x_0, p.x))
            return np.sum(ds)

        def natural_selection(self):
            """
            remove solution with worst value.

            :return:
            """
            solutions = list(self.evaluate_population())
            i = np.argmax(solutions)
            self.population.pop(i)

        def evaluate_population(self) -> np.array:
            """

            :return:
            """
            solutions = np.zeros(len(self.population))
            for i, particle in enumerate(self.population):
                solutions[i] = particle.func[self.func_name]()
            for i, _ in enumerate(self.population):
                solutions[i] += self.diversity(i)
            return solutions

        def run(self, target_array, num_runs):
            """
            1.0) initialize population (@done)
            1.1) compute fitness
            2) select fittest particles
            3.1) crossover solutions
            3.2) mutation (-> stepsize)
            return to 1.1

            :return:
            """
            weights = np.linspace(0.9, 0.0, num_runs)
            for j in range(num_runs):
                # for vis.
                array = np.zeros((len(self.population), self.dims))

                solutions = self.evaluate_population()


                #print(f"{len(self.population)}{j}: {np.min(solutions)}")
                print(f"{j}: {np.min(solutions)}")

                # select best solutions (only fitness)
                indices = np.argpartition(solutions,
                                          self.k_pairs)[:self.k_pairs]
                np.random.shuffle(indices)
                for i in range(0, len(indices), 2):
                    pa = self.population[indices[i]]
                    pb = self.population[indices[i+1]]
                    p_next = self.crossover_mutation(pa, pb, weights[j])
                    self.population.append(p_next)
                    # ... for every new particle delete worst solution
                    self.natural_selection()

                # wird leider nicht angezeigt.
                for i, particle in enumerate(self.population):
                    array[i] = particle.x
                target_array[j, :] = array
    instance = None

    def __init__(self,
                 num_particles=0,
                 dims=0,
                 n=0,
                 step_size=0):
        if not GA.instance:
            GA.instance = GA.__GA(num_particles, dims, n, step_size)

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __str__(self):
        res = ""
        res += "\n"+ "GA\n"+"\n"
        res += "number of particles \t"+str(self.num_particles)+"\n"
        res += "dims \t"+str(self.dims) +"\n"
        res += "n\t"+ str(self.n)
        return res
