import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from tqdm.auto import tqdm

from histogramEstimator import HistogramEstimator
from truncatedGaussian import TruncatedGaussian
from truncatedLaplace import TruncatedLaplace

a, b = 0, 1
runs = 100
laplace_parameters = [
    (
        5.0,
        {"D": 0.44, "C": 0.22, "epsilon": 0.2, "gamma": 0.05, "delta": 0.95},
    ),
    (
        2.0,
        {"D": 1.27, "C": 0.64, "epsilon": 0.5, "gamma": 0.1, "delta": 0.95},
    ),
    (
        1.0,
        {"D": 3.16, "C": 1.58, "epsilon": 1.00, "gamma": 0.2, "delta": 0.95},
    ),
    (
        0.8,
        {"D": 4.38, "C": 2.19, "epsilon": 1.25, "gamma": 0.3, "delta": 0.95},
    ),
    (
        0.5,
        {"D": 9.25, "C": 4.63, "epsilon": 2.00, "gamma": 0.5, "delta": 0.95},
    ),
]

gaussian_parameters = [
    (
        2.0,
        {"D": 0.13, "C": 0.23, "epsilon": 0.13, "gamma": 0.05, "delta": 0.95},
    ),
    (
        1.0,
        {"D": 0.54, "C": 0.70, "epsilon": 0.50, "gamma": 0.1, "delta": 0.95},
    ),
    (
        0.6,
        {"D": 1.62, "C": 1.49, "epsilon": 1.39, "gamma": 0.3, "delta": 0.95},
    ),
    (
        0.5,
        {"D": 2.42, "C": 2.03, "epsilon": 2.00, "gamma": 0.5, "delta": 0.95},
    ),
    (
        0.3,
        {"D": 7.06, "C": 5.40, "epsilon": 5.56, "gamma": 1.4, "delta": 0.95},
    ),
]

# rdp_laplace_parameters = [
#     (
#         5.0,
#         {"D": 1e-3, "C": 0.22, "epsilon": 0.013, "gamma": 0.004, "delta": 0.95},
#     ),
#     (
#         3.0,
#         {"D": 1e-3, "C": 0.39, "epsilon": 0.037, "gamma": 0.01, "delta": 0.95},
#     ),
#     (
#         2.0,
#         {"D": 1e-3, "C": 0.64, "epsilon": 0.082, "gamma": 0.02, "delta": 0.95},
#     ),
#     (
#         1.5,
#         {"D": 1e-3, "C": 0.91, "epsilon": 0.143, "gamma": 0.04, "delta": 0.95},
#     ),
# ]

# rdp_gaussian_parameters = [
#     (
#         2.0,
#         {"D": 0.13, "C": 0.23, "epsilon": 0.125, "gamma": 0.004, "delta": 0.95},
#     ),
#     (
#         1.0,
#         {"D": 0.54, "C": 0.70, "epsilon": 0.5, "gamma": 0.01, "delta": 0.95},
#     ),
#     (
#         0.6,
#         {"D": 1.62, "C": 1.49, "epsilon": 1.39, "gamma": 0.02, "delta": 0.95},
#     ),
#     (
#         0.5,
#         {"D": 2.42, "C": 2.03, "epsilon": 2.0, "gamma": 0.04, "delta": 0.95},
#     ),
#     (
#         0.3,
#         {"D": 7.06, "C": 5.40, "epsilon": 5.56, "gamma": 0.04, "delta": 0.95},
#     ),
# ]

all_experiments = [
    (TruncatedLaplace, False, laplace_parameters),
    (TruncatedGaussian, False, gaussian_parameters),
    (TruncatedLaplace, True, rdp_laplace_parameters),
    # (TruncatedGaussian, True, rdp_gaussian_parameters),
]


def run_experiment(mechanism_class, scale, params, is_rdp, runs):
    mechanism = mechanism_class(a=a, b=b, scale=scale)
    estimator = HistogramEstimator(
        mechanism=mechanism,
        a=a,
        b=b,
        C=params["C"],
        D=params["D"],
        epsilon=params["epsilon"],
        delta=params["delta"],
        gamma=params["gamma"],
        renyi=is_rdp,
        alpha=2.0,
    )
    n_th = estimator.n
    m = estimator.m
    if not m:
        estimator.m = int(np.ceil(100 / estimator.gamma))

    n_pr = 1
    # while True:
    #     valid = 0
    #     for _ in range(runs):
    #         estimator.n = n_pr
    #         estimate = estimator.estimate(0, 1)
    #         if abs(estimate - estimator.epsilon) < estimator.gamma:
    #             valid += 1
    #     if valid / runs >= estimator.delta:
    #         break
    #     n_pr *= 2

    return [
        mechanism.__class__.__name__.replace("Truncated", ""),
        a,
        b,
        estimator.delta,
        estimator.gamma,
        runs,
        scale,
        params["epsilon"],
        params["D"],
        params["C"],
        estimator.k if estimator.k is not None else "Und.",
        m if m is not None else "Und.",
        f"{n_th:.1g}" if n_th is not None else "Und.",
        f"{n_pr:.1g}",
        is_rdp,
    ]


with open("results.csv", "w") as f:
    f.write(
        "mechanism,a,b,delta,gamma,runs,scale,epsilon,D,C,k,m,n_th,n_pr,renyi\n"
    )

    for mechanism_class, is_rdp, params_list in tqdm(all_experiments):
        for scale, params in tqdm(params_list):
            result = run_experiment(
                mechanism_class, scale, params, is_rdp, runs
            )
            f.write(",".join(str(i) for i in result) + "\n")
