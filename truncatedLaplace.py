import numpy as np
import pandas as pd
import plotly.express as px


class TruncatedLaplace:
    def __init__(self, a, b, scale):
        self.a = a
        self.b = b
        self.W = self.b - self.a
        self.scale = scale

    def normalization_constant(self, mean):
        exp_a = np.exp(-(mean - self.a) / self.scale)
        exp_b = np.exp(-(self.b - mean) / self.scale)

        return 1 / (self.scale * (2 - exp_a - exp_b))

    def sample(self, mean, n):
        normalization_constant = self.normalization_constant(mean)
        exp_a = np.exp(-(mean - self.a) / self.scale)
        f_mu = normalization_constant * self.scale * (1 - exp_a)
        tab = np.random.uniform(size=n)
        sgn = np.sign(f_mu - tab)
        res = np.log(
            1 + sgn * (tab / (normalization_constant * self.scale) + exp_a - 1)
        )

        return mean + sgn * self.scale * res

    def true_density(self, mean, n):
        normalization_constant = self.normalization_constant(mean)
        Z = np.linspace(self.a, self.b, n)
        Y = normalization_constant * np.exp(-np.abs(Z - mean) / self.scale)

        return Z, Y


if __name__ == "__main__":
    m = TruncatedLaplace(a=-0.5, b=2, scale=0.5)
    x1, x2 = 0.3, 1.95
    n = 50000
    samples = np.concatenate((m.sample(x1, n), m.sample(x2, n)))
    Z_1, Y_1 = m.true_density(x1, n)
    Z_2, Y_2 = m.true_density(x2, n)
    Z = np.concatenate((Z_1, Z_2))
    Y = np.concatenate((Y_1, Y_2))
    means = np.array([x1, x2])
    df = pd.DataFrame(
        {
            "mean": np.repeat(means, n),
            "samples": samples,
            "Z": Z,
            "density": Y,
        }
    )
    fig = px.histogram(
        df,
        x="samples",
        color="mean",
        histnorm="probability density",
        nbins=50,
        barmode="overlay",
    )
    line_traces = px.line(
        df, x="Z", y="density", color="mean", line_group="mean"
    ).data
    fig.add_traces(line_traces)
    fig.show()
