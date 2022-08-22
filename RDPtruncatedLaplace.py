import numpy as np
import math
import matplotlib.pyplot as plt


class TruncatedLaplace():
    def __init__(self, mu, B, alpha=2, a=0, b=1):
        self.mu = mu
        self.a = a
        self.b = b
        self.W = self.b - self.a
        self.B = B
        self.alpha = alpha
        if not self.a <= self.mu <= self.b:
            raise Exception('Error: mu must belong to [a, b]')
        self.epsilon = math.exp(self.W*(self.alpha - 1) / self.B)
        self.epsilon -= math.exp(-self.W*self.alpha / self.B)
        self.epsilon /= 1 - math.exp(-self.W / self.B)
        self.epsilon /= 2*self.alpha - 1
        self.epsilon = math.log(self.epsilon) / (self.alpha - 1)
        self.K = self.normalizationConstant()
        self.C = self.K / self.B
    
    
    def normalizationConstant(self):
        exp_a = math.exp(-(self.mu - self.a) / self.B)
        exp_b = math.exp(-(self.b - self.mu) / self.B)
        return 1 / (self.B*(2 - exp_a - exp_b))


    def sample(self, n, runs):
        """
        the estimator asks for n samples and we run it runs times so we need
        an array of samples with the shape (n, runs)
        
        to do that we sample uniformly in (0, 1) and apply the inverse CDF of
        the truncated Laplace distribution
        
        all of that is done in a vectorized way for efficiency
        """
        exp_a = math.exp(-(self.mu - self.a)/self.B)
        f_mu = self.K*self.B*(1 - exp_a)
        tab = np.random.uniform(size=(runs, n))
        sgn = np.sign(tab - f_mu)
        res = np.log(1 + sgn*(1 - exp_a - tab/(self.K*self.B)))
        return self.mu - sgn*self.B*res
    
    
    def true_density(self):
        X = np.linspace(0, 1, 1000)
        Y = self.K*np.exp(-np.abs(X - self.mu)/self.B)
        return X, Y


def plot_distrib():
    B = .5
    l = [(.3, "blue"), (.95, "red")]
    for mu, color in l:
        mechanism = TruncatedLaplace(mu=mu, B=B)
        X, Y = mechanism.true_density()
        plt.plot(X, Y, color='dark'+color,
                 label=fr'True density for $\mu = ${mu}')
        plt.hist(mechanism.sample(1000, 10000).reshape(-1),
                 bins=50,
                 label=fr'Histogram of samples for $\mu = ${mu}',
                 density=True,
                 color=color,
                 alpha=.5)
    epsilon = mechanism.epsilon
    plt.xlabel('z')
    plt.ylabel('Density')
    plt.legend()
    plt.title('''Sampling from a truncated Laplace distribution with parameter
              '''
              + rf'$b=${mechanism.B} to ensure '
              + rf'$\epsilon$-DP with $\epsilon=${epsilon}')
    plt.show()

if __name__ == "__main__":
    plot_distrib()