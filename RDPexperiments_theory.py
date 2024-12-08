import matplotlib.pyplot as plt
import numpy as np

from estimatorRDP import Estimator
from RDPtruncatedLaplace import TruncatedLaplace


def experiments(gamma, delta, B, runs, mu1, mu2):
    mechanism = TruncatedLaplace(mu=mu1, B=B)
    mechanism_ = TruncatedLaplace(mu=mu2, B=B)
    epsilon = mechanism.epsilon
    print("epsilon =", epsilon, "C =", mechanism.C)
    est = Estimator(gamma=gamma, delta=delta, epsilon=epsilon, C=mechanism.C)
    print(f"m = {est.m}, n = {est.n}")
    indices = [est.n]
    while indices[-1] > 10:
        indices.append(indices[-1] // 8)
    indices.reverse()
    nb, means, stds, props = [], [], [], []
    practical_n = None
    for i in indices:
        # print('i =', i)
        mean, std, prop = est.run_estimator(mechanism, mechanism_, i, runs)
        if prop >= delta and practical_n is None:
            a, b = i // 2, i
            while b - a > 2:
                k = (b + a) // 2
                mean, std, prop = est.run_estimator(
                    mechanism, mechanism_, k, runs
                )
                if prop >= delta:
                    b = k
                else:
                    a = k
            while prop < delta:
                mean, std, prop = est.run_estimator(
                    mechanism, mechanism_, b, runs
                )
            i = b
            practical_n = b
            practical_n_index = len(nb)
            print("practical", b)
        if mean is not None:
            nb.append(i)
            means.append(mean)
            stds.append(std)
            props.append(prop)
            print(i, prop, mean)
    nb = np.array(nb)
    means = np.array(means)
    stds = np.array(stds)
    plt.rc(
        "axes",
        titlesize=16,
        labelsize=12,
        titleweight="bold",
        labelweight="bold",
    )
    plt.rc("figure", titlesize=18, titleweight="bold")
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(211)
    ax.axhline(y=epsilon, color="red", label=r"real $\epsilon$", linestyle="--")
    ax.axhspan(
        epsilon - gamma,
        epsilon + gamma,
        facecolor="orange",
        alpha=0.2,
        label=r"$\gamma$-width band around $\epsilon$ " + "(required output)",
    )
    # ax.axhline(y=epsilon + gamma, color='red', linestyle=':',
    # label=r'$\epsilon + \gamma$ and $\epsilon - \gamma$')
    # ax.axhline(y=epsilon - gamma, color='red', linestyle=':')
    ax.fill_between(
        nb,
        means - stds,
        means + stds,
        alpha=0.2,
        label=r"standard deviation of $\tilde{\epsilon}$ "
        + f"across {runs} executions",
    )
    # ax.errorbar(nb, means, stds, linestyle='None',
    # marker='o', capsize=3,
    # label='Confidence intervals of level '
    # + rf'$\delta=${delta} across {runs} executions')
    ax.plot(nb, means, label=r"mean of $\tilde{\epsilon}$")
    ax.set_title(
        r"Estimated $\mathbf{\tilde{\epsilon}}$ "
        + "against the number of samples"
    )
    # ax.set_ylabel(r'Estimated $\mathbf{\tilde{\epsilon}}$')
    ax.set_xlabel(r"Number of samples $\mathbf{n}$ (log scale)")
    ax.set_xscale("log")
    # ax.set_ylim(top=np.max(means))
    ax.legend()
    ax2 = fig.add_subplot(212)
    ax2.axhline(y=delta, color="purple", label=r"$\delta$")
    bar = ax2.bar([str(i) for i in nb], props, color="purple", alpha=0.4)
    labels = [""] * len(props)
    labels[practical_n_index] = "Practical n"
    labels[-1] = "Theoretical n"
    ax2.bar_label(bar, labels, label_type="center")
    ax2.legend()
    ax2.set_xlabel(r"Number of samples $\mathbf{n}$")
    ax2.set_ylabel(f"Proportion across {runs} executions")
    ax2.set_title(
        r"Proportion of estimated $\mathbf{\tilde{\epsilon}}$ "
        + r"close to $\mathbf{\epsilon}$ within $\mathbf{\gamma}$"
    )
    fig.tight_layout(pad=5)
    plt.show(block=False)

    X, Y = mechanism.true_density()
    _, Y_ = mechanism_.true_density()
    hist = {}
    std = {}
    # for n in [practical_n, est.n]:
    #     samples = mechanism.sample(n, runs)
    #     nj = est.count(samples) / (n * est.w)
    #     samples = mechanism_.sample(n, runs)
    #     mj = est.count(samples) / (n * est.w)
    #     mask = np.logical_and(np.all(nj > 0, axis=1), np.all(mj > 0, axis=1))
    #     nj = nj[mask]
    #     mj = mj[mask]
    #     hist[n] = (np.mean(nj, axis=0), np.mean(mj, axis=0))
    #     std[n] = (np.std(nj, axis=0), np.std(mj, axis=0))
    # fig = plt.figure(figsize=(20, 10))
    # fig.suptitle('Estimated densities with standard deviations '
    #                  + f'across {runs} executions')
    # ax = fig.add_subplot(121)
    # ax.set_title('For the practical number of samples '
    #              + f'n = {practical_n}')
    # x = np.linspace(0, 1, est.m)
    # # h, s = hist[practical_n], std[practical_n]
    # width=.4/est.m
    # # ax.bar(x-width/2, h[0], yerr=s[0], alpha=.5, width=width, ecolor='gray',
    # #        label=rf'estimated density for $\mu=${mu1}')
    # # ax.bar(x+width/2, h[1], yerr=s[1], alpha=.5, width=width, ecolor='gray',
    # #        label=rf'estimated density for $\mu=${mu2}')
    # ax.plot(X-width/2, Y, linewidth=3,
    #         label=rf'true density for $\mu=${mu1}')
    # ax.plot(X+width/2, Y_, linewidth=3,
    #         label=rf'true density for $\mu=${mu2}')
    # ax.set_ylabel('Density')
    # ax.set_xlabel('z')
    # ax.legend()
    # ax2 = fig.add_subplot(122)
    # ax2.set_title('For the theoretical number of samples '
    #              + f'n = {est.n}')
    # # h, s = hist[est.n], std[est.n]
    # # ax2.bar(x-width/2, h[0], yerr=s[0], alpha=.5, width=width, ecolor='gray',
    #         # label=rf'estimated density for $\mu=${mu1}')
    # # ax2.bar(x+width/2, h[1], yerr=s[1], alpha=.5, width=width, ecolor='gray',
    #         # label=rf'estimated density for $\mu=${mu2}')
    # ax2.plot(X-width/2, Y, linewidth=3,
    #          label=rf'true density for $\mu=${mu1}')
    # ax2.plot(X+width/2, Y_, linewidth=3,
    #          label=rf'true density for $\mu=${mu2}')
    # ax2.set_ylim(ax.get_ylim())
    # ax2.set_ylabel('Density')
    # ax2.set_xlabel('z')
    # ax2.legend()
    plt.show()


if __name__ == "__main__":
    gamma = 0.5
    delta = 0.9
    runs = 100
    mu1 = 0
    mu2 = 1
    B = 3.5
    experiments(gamma, delta, B, runs, mu1, mu2)
    # input()
    # mechanism = TruncatedLaplace(mu=mu1, B=B)
    # mechanism_ = TruncatedLaplace(mu=mu2, B=B)
    # epsilon = mechanism.epsilon
    # print('epsilon =', epsilon, 'C =', mechanism.C)
    # est = Estimator(gamma=gamma, delta=delta, epsilon=epsilon,
    #                 C=mechanism.C)
    # print(f'm = {est.m}, n = {est.n}')
    # print(est.run_estimator(mechanism, mechanism_, est.n, runs)[0])
