import math

import matplotlib.pyplot as plt
import numpy as np


class TruncatedLaplace:
    def __init__(self, mu, epsilon=None, B=None, a=0, b=1):
        self.mu = mu
        self.a = a
        self.b = b
        self.W = self.b - self.a
        self.epsilon = epsilon
        self.B = B
        if not self.a <= self.mu <= self.b:
            raise Exception("Error: mu must belong to [a, b]")
        if self.epsilon is None and self.B is None:
            raise Exception("Error: either epsilon or B needs to be defined")
        if self.epsilon is not None:
            if self.B is not None and self.epsilon * self.B < self.b - self.a:
                raise Exception(
                    "Error: with such a B the resulting mechanism "
                    + "is not epsilon-DP"
                )
            elif self.B is None:
                self.B = self.W / self.epsilon
        else:
            self.epsilon = self.W / self.B
        self.K = self.normalizationConstant()
        self.C = self.K / self.B

    def normalizationConstant(self):
        exp_a = math.exp(-(self.mu - self.a) / self.B)
        exp_b = math.exp(-(self.b - self.mu) / self.B)
        return 1 / (self.B * (2 - exp_a - exp_b))

    def sample(self, n, runs):
        """
        the estimator asks for n samples and we run it runs times so we need
        an array of samples with the shape (n, runs)

        to do that we sample uniformly in (0, 1) and apply the inverse CDF of
        the truncated Laplace distribution

        all of that is done in a vectorized way for efficiency
        """
        exp_a = math.exp(-(self.mu - self.a) / self.B)
        f_mu = self.K * self.B * (1 - exp_a)
        tab = np.random.uniform(size=(runs, n))
        sgn = np.sign(f_mu - tab)
        res = np.log(1 + sgn * (tab / (self.K * self.B) + exp_a - 1))
        return self.mu + sgn * self.B * res

    def true_density(self):
        X = np.linspace(0, 1, 1000)
        Y = self.K * np.exp(-np.abs(X - self.mu) / self.B)
        return X, Y


def plot_distrib():
    epsilon = 8
    l = [(0.3, "blue"), (0.95, "red")]
    for mu, color in l:
        mechanism = TruncatedLaplace(mu=mu, epsilon=epsilon)
        X, Y = mechanism.true_density()
        plt.plot(
            X, Y, color="dark" + color, label=rf"True density for $\mu = ${mu}"
        )
        plt.hist(
            mechanism.sample(1000, 1000).reshape(-1),
            bins=50,
            label=rf"Histogram of samples for $\mu = ${mu}",
            density=True,
            color=color,
            alpha=0.5,
        )
    plt.xlabel("z")
    plt.ylabel("Density")
    plt.legend()
    plt.title(
        """Sampling from a truncated Laplace distribution with parameter
              """
        + rf"$b=${mechanism.B} to ensure "
        + rf"$\epsilon$-DP with $\epsilon=${epsilon}"
    )
    plt.show()


if __name__ == "__main__":
    plot_distrib()
