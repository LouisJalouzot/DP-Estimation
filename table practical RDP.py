from estimatorRDP import Estimator
from RDPtruncatedLaplace import TruncatedLaplace
import matplotlib.pyplot as plt
import numpy as np

def experiments(runs):
    delta = .9
    for B in [5, 3, 2, 1.5]:
        mechanism = TruncatedLaplace(mu=0, B=B)
        mechanism_ = TruncatedLaplace(mu=1, B=B)
        epsilon = mechanism.epsilon
        print("eps:", int(1000*epsilon)/1000)
        print("C:", int(100 * mechanism.C) / 100)
        # for gamma in [10, 1, .5, .1, .01]:
        for gamma in [1, .5, .1]:
            print("    gamma:", gamma)
            est = Estimator(gamma=gamma, epsilon=epsilon, C=mechanism.C)
            n = 100
            prop = 0
            while prop < delta and n <= 5e8:
                _, _, prop = est.run_estimator(mechanism, mechanism_, n, runs)
                n *= 2
            if n > 1e9 and prop < delta:
                n = None
            else:
                a, b = n//4, n//2
                while b - a > 10:
                    k = (b + a)//2
                    _, _, prop = est.run_estimator(mechanism, mechanism_, k, runs)
                    if prop >= delta:
                        b = k
                    else: a = k
                n = b
            print("        m:", est.m)
            print("        Theoretical n:", est.n)
            print("        Practical n:", n)
            
    

if __name__ == "__main__":
    runs = 1000
    experiments(runs)