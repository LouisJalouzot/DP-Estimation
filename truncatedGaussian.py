import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import truncnorm


class TruncatedGaussian:
    def __init__(self, a, b, scale):
        self.a = a
        self.b = b
        self.scale = scale

    def normalization_constant(self, mean):
        a_norm = (self.a - mean) / self.scale
        b_norm = (self.b - mean) / self.scale

        return truncnorm.cdf(b_norm, a_norm, b_norm) - truncnorm.cdf(
            a_norm, a_norm, b_norm
        )

    def sample(self, mean, n):
        a_norm = (self.a - mean) / self.scale
        b_norm = (self.b - mean) / self.scale
        samples = truncnorm.rvs(
            a_norm,
            b_norm,
            loc=mean,
            scale=self.scale,
            size=n,
        )

        return samples

    def true_density(self, mean, n):
        Z = np.linspace(self.a, self.b, n)
        a_norm = (self.a - mean) / self.scale
        b_norm = (self.b - mean) / self.scale
        normalization_constant = self.normalization_constant(mean)
        Y = (
            truncnorm.pdf(Z, a_norm, b_norm, loc=mean, scale=self.scale)
            / normalization_constant
        )

        return Z, Y


if __name__ == "__main__":
    m = TruncatedGaussian(a=-0.5, b=2, scale=0.5)
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
            "samples": samples.reshape(-1),
            "Z": Z.reshape(-1),
            "density": Y.reshape(-1),
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
