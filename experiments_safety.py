import matplotlib.pyplot as plt
import numpy as np

from estimator import Estimator
from truncatedLaplace import TruncatedLaplace

gammaR = r"$\mathbf{\gamma}$"
deltaR = r"$\mathbf{\delta}$"


def experiments(gamma, delta, epsilon, runs, mu1, mu2, C, safety_conf):
    bound, props, actualC = [], [], []
    prop = 1
    while prop > safety_conf or len(bound) < 10:
        mechanism = TruncatedLaplace(mu=mu1, epsilon=epsilon)
        mechanism_ = TruncatedLaplace(mu=mu2, epsilon=epsilon)
        est = Estimator(
            gamma=gamma,
            delta=delta,
            C=C,
            enable_safety=True,
            safety_conf=safety_conf,
        )
        print(f"m = {est.m}, n = {est.n}")
        print(f"Claimed C = {C}, actual C = {mechanism.C}")
        actualC.append(mechanism.C)
        print(f"Safety bound = {est.safety_bound}")
        bound.append(est.safety_bound)
        print(f"Safety proba = {est.safety_conf}")
        prop = est.run_estimator(mechanism, mechanism_, est.n, runs)
        print(prop)
        props.append(prop)
        print("\n")
        epsilon += 0.15
    plt.figure(figsize=(10, 6))
    plt.rc(
        "axes",
        titlesize=16,
        labelsize=12,
        titleweight="bold",
        labelweight="bold",
    )
    plt.rc("figure", titlesize=18, titleweight="bold")
    plt.plot(actualC, props, linewidth=3)
    plt.axhline(
        safety_conf,
        label="required safety check confidence "
        + f"({int(safety_conf*100)}%)",
        color="orange",
        linewidth=3,
    )
    plt.xlabel(f"Actual lipschitz constant C for a claimed C = {C}")
    plt.ylabel(f"Proportion across {runs} executions")
    plt.title(
        "Proportion of executions which "
        + """did not trigger the safety check
              """
        + f"(m = {est.m}, n = {est.n}, "
        + gammaR
        + f" = {gamma}, "
        + deltaR
        + f" = {delta})"
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    gamma = 2
    delta = 0.8
    epsilon = 0.5
    mu1 = 0
    mu2 = 1
    C = 1.7
    runs = 100
    safety_conf = 0.5
    experiments(gamma, delta, epsilon, runs, mu1, mu2, C, safety_conf)
