import numpy as np
import math
import matplotlib.pyplot as plt

class Estimator:
    def __init__(
            self, gamma=1, delta=.9, alpha=2,
            epsilon=None, C=1, W=1, m=None):
        self.gamma = gamma
        self.delta = delta
        self.alpha = alpha
        if self.alpha <= 1:
            raise Exception('alpha needs to be > 1')
        self.epsilon = epsilon
        self.W = W
        if C is None:
            raise Exception('C needs to be defined')
        self.C = C
        self.tau = 1/self.W - self.C*self.W/2
        self.tau_ = 1/self.W + self.C*self.W/2
        if self.tau <= 0:
            raise Exception('Condition not met: C < 2/W^2')    
        self.K = 2*self.tau_**self.alpha / self.tau**(self.alpha - 1)        
        self.K_ = self.tau**self.alpha / self.tau_**(self.alpha - 1)
        self.gamma_ = self.gamma*self.K_*(self.alpha - 1)
        self.gamma_ /= 2*self.K*(2*self.alpha - 1)
        self.gamma_ = min(self.gamma_, math.log(2)/(2*self.alpha - 1))
        try:
            self.m = self.defineM()
            self.w = self.W/self.m
            self.n = self.defineN()
        except:
            self.m = m
            self.w = self.W/self.m


    def defineM(self):
        d = self.C*self.W*self.K*(2*self.alpha - 1)
        n = self.gamma*self.tau*self.K_*(self.alpha - 1)
        return math.ceil(d/n)


    def compute_f(self, x, y, z):
        exp = math.exp(-x*y*(math.exp(z) - 1)**2 / (1 + math.exp(z)))
        exp_ = math.exp(-x*y*(1 - math.exp(-z))**2 / 2)
        return (exp + exp_) / (1 - (1 - y)**x)
    
    
    def compute_bound(self, n):
        exp = 2*self.m*(1 - self.w*self.tau)**n
        f = self.compute_f(n, self.w*self.tau, self.gamma_)
        return exp + 2*self.m*f


    def defineN(self):
        n = 1
        while self.compute_bound(n) > 1 - self.delta:
            n *= 2
        i, j = n//2, n
        while j - i > 2:
            k = (i + j)//2
            if self.compute_bound(k) > 1 - self.delta:
                i = k
            else:
                j = k
        return j
    
    
    def count(self, samples):
        samples = (samples / self.w).astype(int)
        # replace values by the number of the sub-interval they belong to
        samples = np.apply_along_axis(np.bincount, 1, samples,
                                      minlength=self.m)
        # bincount on each row/execution to build the N_j
        return samples
    
    
    def exec(self, nj, mj, runs, n):
        # remove all rows with at least one Nj or Mj equal to 0
        mask = np.logical_and(np.all(nj > 0, axis=1),
                                np.all(mj > 0, axis=1))
        nj = nj[mask]
        mj = mj[mask]
        if len(nj) > 0:
            # estimated epsilon
            nj = np.power(nj, self.alpha)
            mj = np.power(mj, self.alpha - 1)
            estimates = np.log(np.sum(nj / mj, axis=1) / n)
            estimates /= self.alpha - 1
            # mean of estimated epsilons across runs
            mean = np.mean(estimates)
            # standard deviation of estimated epsilons across runs
            std = np.std(estimates)
            # number of estimated epsilons that are
            # close to epsilon within gamma
            good_estimates = np.abs(estimates - self.epsilon) <= self.gamma
            # proportion of 'good' estimates, we want prop >= delta
            prop = np.sum(good_estimates) / runs
            return mean, std, prop
        else:
            return None, 0, 0
        
    def run_estimator(self, mechanism, mechanism_, n, runs):
        samples = mechanism.sample(n, runs)
        nj = self.count(samples)
        samples = mechanism_.sample(n, runs)
        mj = self.count(samples)
        return self.exec(nj, mj, runs, n)
    

def plot_dependencies():
    plt.rc('axes', titlesize=12, labelsize=12,
           titleweight='bold', labelweight='bold')
    plt.rc('figure', titlesize=14, titleweight='bold')
    # fig = plt.figure(figsize=(20, 10))
    fig = plt.figure()
    # fig.suptitle("Log of the number of samples as functions of "
                #  + r"$\mathbf{\gamma, \delta}$ and C")
    alpha = np.linspace(1.5, 5, 10)
    gamma = np.linspace(1, 10, 10)
    delta = np.linspace(.5, .999, 10)
    C = np.linspace(1, .1, 10)

    # n as a function of gamma and delta
    ax = fig.add_subplot(131, projection='3d')
    X, Y = np.meshgrid(gamma, C)
    f = np.vectorize(lambda g, c: Estimator(gamma=g, C=c).n)
    Z = np.log(f(X, Y))
    ax.plot_surface(X, Y, Z, cmap='copper')
    ax.set_xlabel(r'$\mathbf{\gamma}$')
    ax.set_ylabel('C')
    ax.set_zlabel(r'log$\mathbf{(n)}$')
    # ax.set_title(title + r'$\mathbf{\gamma}$ and $\mathbf{\delta}$')
    
    # n as a function of C and delta
    ax = fig.add_subplot(132, projection='3d')
    X, Y = np.meshgrid(alpha, C)
    f = np.vectorize(lambda a, c: Estimator(alpha=a, C=c).n)
    Z = np.log(f(X, Y))
    ax.plot_surface(X, Y, Z, cmap='copper')
    ax.set_xlabel(r'$\mathbf{\alpha}$')
    ax.set_ylabel('C')
    ax.set_zlabel(r'log$\mathbf{(n)}$')
    # ax.set_title(title + r'$\mathbf{\delta}$ and $\mathbf{C}$')
    
    # n as a function of gamma and C
    ax = fig.add_subplot(133, projection='3d')
    X, Y = np.meshgrid(delta, alpha)
    f = np.vectorize(lambda d, a: Estimator(delta=d, alpha=a).n)
    Z = np.log(f(X, Y))
    ax.plot_surface(X, Y, Z, cmap='copper')
    ax.set_xlabel(r'$\mathbf{\delta}$')
    ax.set_ylabel(r'$\mathbf{\alpha}$')
    ax.set_zlabel(r'log$\mathbf{(n)}$')
    # ax.set_title(title + r'$\mathbf{\gamma}$ and $\mathbf{C}$')#, y=-.2)
    
    plt.show()
    

if __name__ == "__main__":
    plot_dependencies()
    # print(Estimator(gamma=1, delta=.8, C=.5, W=1).n)