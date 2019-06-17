import numpy as np
import sys


class Particle:
    def __init__(self, n, dims):
        self.n = n
        self.x = np.zeros(dims)
        for i in range(dims):
            self.x[i] = 2 * n * np.random.random() - n
        self.dims = dims
        # better: class-func
        self.func = dict()
        self.func['square'] = self.eval_square
        self.func['rastrigin'] = self.eval_rastrigin
        self.func['schaffer_f6'] = self.eval_schaffer_f6
        self.func['griewank'] = self.eval_griewank
        self.func['rosenbrock'] = self.eval_rosenbrock
        self.func['eggholder'] = self.eval_eggholder
        self.func['hoelder_table'] = self.eval_hoelder_table
        self.func['eval_styblinsky_tang'] = self.eval_styblinsky_tang
    """
        Test function evaluation.
    """
    def eval_square(self):
        res = 0.0
        for i in range(self.dims):
            res += self.x[i]**2
        return res

    def eval_rastrigin(self):
        res = 0.0
        for i in range(self.dims):
            res += self.x[i]**2 - 10 * np.cos(np.pi * self.x[i]) + 10
        return res

    def eval_schaffer_f6(self):
        # nur 2d möglich (?)
        if self.dims != 2:
            print("Invalid dim!", self.dims)
            return
        return (0.5- ((np.sin(np.sqrt(self.x[0]**2+self.x[1]**2))**2-0.5)\
                      / (1+0.001*(self.x[0]**2+self.x[1]**2)**2)))

    def eval_griewank(self):
        prod = (np.cos(self.x[0]/0.0000001)+1)*(np.cos(self.x[1]/1)+1)
        return (1/4000)*(self.x[0]**2-prod +self.x[1]**2-prod)

    def eval_rosenbrock(self):
        a = 1. - self.x[0]
        b = self.x[1]- self.x[0]*self.x[0]
        return a*a + b*b*100

    def eval_eggholder(self):
        return -(self.x[1]+47)*np.sin(np.sqrt(np.abs(self.x[0]/2+ (self.x[1]+47))))\
        -self.x[0]*np.sin(np.sqrt(np.abs(self.x[0]-(self.x[1]+47))))

    def eval_hoelder_table(self):
        return np.abs(np.sin(self.x[0])*np.cos(self.x[1])\
                      * np.exp(np.abs(1-(np.sqrt(self.x[0]**2+self.x[1]**2)/np.pi))))

    def eval_styblinsky_tang(self):
        sum = 0.0
        for i in range(self.dims):
            sum += (self.x[i]**4-16*self.x[i]+5*self.x[i])
        return sum/2

