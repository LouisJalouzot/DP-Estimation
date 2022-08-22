from estimator import Estimator
from truncatedLaplace import TruncatedLaplace
import matplotlib.pyplot as plt
import numpy as np

def experiments(runs):
    delta = .8
    # for epsilon in [.5, 1, 3, 5, 10]:
    for epsilon in [.5, .7, 1, 2]:
        mechanism = TruncatedLaplace(mu=0, epsilon=epsilon)
        mechanism_ = TruncatedLaplace(mu=1, epsilon=epsilon)
        epsilon = mechanism.epsilon
        print("eps:", epsilon)
        print("C:", int(100 * mechanism.C) / 100)
        # for gamma in [10, 1, .5, .1, .01]:
        for gamma in [1, .5, .1, .05]:
            print("    gamma:", gamma)
            try:
                est = Estimator(gamma=gamma, delta=delta, epsilon=epsilon,
                                C=mechanism.C,
                                enable_safety=False)
            except:
                est = Estimator(gamma=gamma, delta=delta, epsilon=epsilon,
                                C=mechanism.C, m=int(100/gamma),
                                enable_safety=False)
                est.n = None
            m = est.m
            n = 100
            prop = 0
            while prop < delta and n <= 5e8:
                prop = est.run_estimator2(mechanism, mechanism_, n, runs)
                n *= 2
            if n > 1e9 and prop < delta:
                n = None
            else:
                a, b = n//4, n//2
                while b - a > m / gamma:
                    k = (b + a)//2
                    prop = est.run_estimator2(mechanism, mechanism_,
                                                    k, runs)
                    if prop >= delta:
                        b = k
                    else: a = k
                n = b
            print("        m:", est.m)
            print("        Theoretical n:", est.n)
            print("        Practical n:", n)
            
    

if __name__ == "__main__":
    runs = 100
    experiments(runs)