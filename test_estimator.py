import numpy as np
import plotly.graph_objects as go

from histogramEstimator import HistogramEstimator
from truncatedGaussian import TruncatedGaussian
from truncatedLaplace import TruncatedLaplace


def main():
    # Parameters from the paper (section 4.3)
    a, b = 0, 1

    # Parameters for B=1 case from Table 1
    laplace_params = {"epsilon": 1.0, "C": 1.58, "D": 3.16, "scale": 1.0}

    # Parameters for σ=0.6 case from Table 1
    gaussian_params = {"epsilon": 1.39, "C": 1.49, "D": 1.62, "scale": 0.6}

    # Initialize mechanisms
    laplace = TruncatedLaplace(a=a, b=b, scale=laplace_params.pop("scale"))
    gaussian = TruncatedGaussian(a=a, b=b, scale=gaussian_params.pop("scale"))

    # Create estimators
    laplace_estimator = HistogramEstimator(
        mechanism=laplace,
        a=a,
        b=b,
        **laplace_params,
    )

    gaussian_estimator = HistogramEstimator(
        mechanism=gaussian,
        a=a,
        b=b,
        **gaussian_params,
    )

    # Print estimator parameters
    print("Parameters for Truncated Laplace mechanism:")
    print(f"Number of samples (n): {laplace_estimator.n:,}")
    print(f"Number of histogram bins (m): {laplace_estimator.m}")
    print(f"Number of grid points (k): {laplace_estimator.k}\n")

    print("Parameters for Truncated Gaussian mechanism:")
    print(f"Number of samples (n): {gaussian_estimator.n:,}")
    print(f"Number of histogram bins (m): {gaussian_estimator.m}")
    print(f"Number of grid points (k): {gaussian_estimator.k}\n")

    # Compute histograms
    print("Computing Laplace histogram...")
    laplace_hist = laplace_estimator.estimate()
    print("Computing Gaussian histogram...")
    gaussian_hist = gaussian_estimator.estimate()

    # Prepare visualization data
    x = np.linspace(
        laplace_params["a"], laplace_params["b"], laplace_estimator.k
    )
    y = np.linspace(
        laplace_params["a"], laplace_params["b"], laplace_estimator.k
    )

    # Create and show heatmap figure
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=np.max(laplace_hist, axis=2),
            x=x,
            y=y,
            colorscale="Viridis",
            name="Truncated Laplace",
            visible=True,
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=np.max(gaussian_hist, axis=2),
            x=x,
            y=y,
            colorscale="Viridis",
            name="Truncated Gaussian",
            visible=False,
        )
    )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        args=[{"visible": [True, False]}],
                        label="Truncated Laplace",
                        method="update",
                    ),
                    dict(
                        args=[{"visible": [False, True]}],
                        label="Truncated Gaussian",
                        method="update",
                    ),
                ],
                direction="down",
                showactive=True,
            )
        ],
        title="Estimated Privacy Loss",
        xaxis_title="x1",
        yaxis_title="x2",
    )

    # Compute and display global estimates
    print("\nComputing global estimates...")
    laplace_global = laplace_estimator.estimate(global_estimate=True)
    gaussian_global = gaussian_estimator.estimate(global_estimate=True)

    print(
        f"Global privacy loss estimate for Truncated Laplace: {laplace_global:.3f} "
        f"(expected: {laplace_params['epsilon']:.3f} ± {laplace_estimator.gamma:.3f})"
    )
    print(
        f"Global privacy loss estimate for Truncated Gaussian: {gaussian_global:.3f} "
        f"(expected: {gaussian_params['epsilon']:.3f} ± {gaussian_estimator.gamma:.3f})"
    )

    # Show the interactive plot
    fig.show()


if __name__ == "__main__":
    main()
