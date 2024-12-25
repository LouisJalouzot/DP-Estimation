import numpy as np
from tqdm.auto import tqdm

np.random.seed(0)


class HistogramEstimator:
    def __init__(
        self,
        mechanism,
        a,
        b,
        C,
        D,
        epsilon,
        delta=0.95,
        gamma=None,
        renyi=False,
        alpha=2.0,
    ):
        self.mechanism = mechanism
        self.a = a
        self.b = b
        self.W = self.b - self.a
        self.C = C
        self.D = D
        self.epsilon = epsilon
        self.delta = delta
        self.delta_1 = np.sqrt(delta)  # Delta for Algo 1 = sqrt(delta)
        if gamma is None:
            gamma = 0.1 * epsilon
        self.gamma = gamma
        self.gamma_1 = gamma / 3  # Gamma for Algo 1 = gamma / 3
        self.renyi = renyi
        self.alpha = alpha
        if self.renyi:
            self.tau_0 = 1 / self.W - (self.C * self.W) / 2
            self.tau_1 = 1 / self.W + (self.C * self.W) / 2
            if self.tau_0 <= 0:
                self.k = self.m = self.n = None
                print(
                    "Negative or zero tau_0, define k, m, n manually for renyi."
                )
            else:
                # K, K', gamma'
                self.K = (
                    2
                    * (self.tau_1**self.alpha)
                    / (self.tau_0 ** (self.alpha - 1))
                )
                self.Kp = (self.tau_0**self.alpha) / (
                    self.tau_1 ** (self.alpha - 1)
                )
                self.gamma_p = min(
                    (self.gamma * self.Kp * (self.alpha - 1))
                    / (2 * self.K * (2 * self.alpha - 1)),
                    np.log(2) / (2 * self.alpha - 1),
                )
                # m from (defM)
                # gamma/2 >= C*w*K*(2alpha -1)/(2*tau_0*K'*(alpha-1))
                # w = W/m => m >= ...
                denom = (
                    (self.gamma / 2)
                    * 2
                    * self.tau_0
                    * self.Kp
                    * (self.alpha - 1)
                )
                num = self.C * self.W * self.K * (2 * self.alpha - 1)
                self.m = int(np.ceil(num / denom))
                # n from (nRDPdef)
                self.n = self.compute_n_renyi()
        else:
            self.tau = 1 / self.W - (self.C * self.W) / 2
            if self.tau > 0:
                # From Algo 2
                self.k = int(
                    np.ceil(3 * self.D * self.W / (self.tau * self.gamma))
                )
                self.m = int(
                    np.ceil(6 * self.C * self.W / (self.tau * self.gamma_1))
                )
                self.n = self.compute_n()
                print(
                    f"{self.mechanism.__class__.__name__} with C = {C}, D = {D}, delta = {delta}, gamma = {gamma}:"
                )
                print(f"m = {self.m}, n = {self.n:,}, k = {self.k}")
                print(f"Number of samples for one pair: {2 * self.n:.3g}")
                print(
                    f"Number of samples for global estimation: {self.n * self.k * (self.k - 1) / 2:.3g}\n"
                )
            else:
                self.k = self.m = self.n = None
                print("Negative tau, define k, m, n manually")

    def f(self, x, y, z):
        exp_1 = np.exp(-x * y * (np.exp(z) - 1) ** 2 / (1 + np.exp(z)))
        exp_2 = np.exp(-x * y * (1 - np.exp(-z)) ** 2 / 2)

        return (exp_1 + exp_2) / (1 - (1 - y) ** x)

    def ndef_equation_satisfied(self, n, w_tau):
        first_term = 2 * self.m * (1 - w_tau) ** n
        second_term = 4 * self.f(n, w_tau, self.gamma_1 / 12)

        return first_term + second_term <= 1 - self.delta_1

    def compute_n(self):
        w_tau = self.W * self.tau / self.m

        # Find upper bound
        n_high = 1
        while not self.ndef_equation_satisfied(n_high, w_tau):
            n_high *= 2

        # Binary search for smallest valid n
        n_low = n_high // 2
        while n_low < n_high - 1:
            n_mid = (n_low + n_high) // 2
            if self.ndef_equation_satisfied(n_mid, w_tau):
                n_high = n_mid
            else:
                n_low = n_mid

        return n_high

    def compute_n_renyi(self):
        """
        Compute n for LRDP following the definitions from the renyi_dp.tex file.
        """
        w_tau_0 = self.W * self.tau_0 / self.m

        def ndef_equation_satisfied_renyi(n):
            # eqn: 1 - 2m(1 - w_tau_0)^n - 2m*f(n, w_tau_0, self.gamma_p) >= self.delta
            lhs = (
                1
                - 2 * self.m * (1 - w_tau_0) ** n
                - 2 * self.m * self.f(n, w_tau_0, self.gamma_p)
            )
            return lhs >= self.delta

        # Find upper bound
        n_high = 1
        while not ndef_equation_satisfied_renyi(n_high):
            n_high *= 2

        # Binary search
        n_low = n_high // 2
        while n_low < n_high - 1:
            n_mid = (n_low + n_high) // 2
            if ndef_equation_satisfied_renyi(n_mid):
                n_high = n_mid
            else:
                n_low = n_mid

        return n_high

    def estimate(
        self,
        x1=None,
        x2=None,
    ):
        """
        Estimate epsilon between x1 and x2 if specified, otherwise estimate epsilon globally
        """
        # Setup bins
        if x1 is None or x2 is None:
            means = np.linspace(self.a, self.b, self.k)
        else:
            means = [x1, x2]

        counts = []
        for x in tqdm(means, disable=len(means) <= 2):
            samples = self.mechanism.sample(x, self.n)
            counts.append(
                np.histogram(samples, bins=self.m, range=(self.a, self.b))[0]
            )
        counts = np.stack(counts)

        if (counts == 0).any():
            return np.nan

        divs = counts[:, None] / counts[None]

        if self.renyi:
            # Example LRDP approach: 1/(alpha-1) * log( sum_j((N_j/M_j)^alpha * 1/n * M_j) )
            # ...new code using counts...
            return float("nan")  # placeholder
        else:
            return np.log(max(divs.max(), 1.0 / divs.min())).item()


if __name__ == "__main__":
    from truncatedGaussian import TruncatedGaussian
    from truncatedLaplace import TruncatedLaplace

    a, b = 0, 1
    delta = 0.95
    gamma = 0.2

    laplace = TruncatedLaplace(a=a, b=b, scale=2)
    laplace_estimator = HistogramEstimator(
        mechanism=laplace,
        a=a,
        b=b,
        C=0.63,
        D=1.27,
        epsilon=0.5,
        delta=delta,
        gamma=gamma,
    )

    gaussian = TruncatedGaussian(a=a, b=b, scale=1)
    gaussian_estimator = HistogramEstimator(
        mechanism=gaussian,
        a=a,
        b=b,
        C=0.7,
        D=0.54,
        epsilon=0.5,
        delta=delta,
        gamma=gamma,
    )
