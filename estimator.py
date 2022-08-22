import numpy as np
import math
import matplotlib.pyplot as plt

class Estimator:
    def __init__(
            self, gamma, delta, epsilon=None,
            C=None, W=1, m=None,
            enable_safety=False, safety_c=-1, safety_conf=.95):
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon # expected epsilon
        self.W = W
        if m is not None:
            # manually setting m for use with 
            # practical n instead of theoretical one
            self.m = m
            self.w = self.W/self.m
            try:
                self.C = C
                self.tau = 1/self.W - self.C*self.W/2
                self.n = self.defineN()
            except:
                self.n = None
        else:
            if C is None:
                raise Exception('C needs to be defined')
            
            self.C = C
            self.tau = 1/self.W - self.C*self.W/2
            
            if self.tau <= 0:
                raise Exception('Condition not met: C < 2/W^2')            
            
            self.m = self.defineM()
            self.w = self.W/self.m
            self.n = self.defineN()
        self.enable_safety = enable_safety
        if self.enable_safety:
            self.safety_conf = safety_conf
            self.safety_tau = 1/self.W + self.C*self.W/2
            # safety_c is the constant c used for the safety check
            log = math.log((1 - self.safety_conf) / (8*self.m))
            if safety_c is None:
                # if we don't choose c then we will set it so that
                # the theoretical lower bound on the probability
                # of the safety check to be met is >= safety_conf
                if self.n is None:
                    raise Exception('Either n of c needs to be defined')
                self.safety_c = math.sqrt(-3*self.w*self.safety_tau*log / self.n)
            else:
                if safety_c == -1:
                    # ask for c to be set of the same order of magnitude as
                    # Cw**2
                    self.safety_c = self.C*self.w**2 / 2
                # otherwise we set n to reach safety_conf
                else:
                    self.safety_c = safety_c
                safety_n = math.ceil(-3*self.w*self.safety_tau*log
                                    / self.safety_c**2)
                print('safety_n', safety_n)
                if self.n is None:
                    self.n = safety_n
                else:
                    self.n = max(self.n, safety_n)
            # safety_bound is 2c + Cw^2
            # it's the maximum difference allowed between
            # 2 consecutive Nj or Mj
            self.safety_bound = 2*self.safety_c + self.C*self.w**2


    def defineM(self):
        d = 6*self.C*self.W
        n = self.tau * self.gamma
        return math.ceil(d/n)


    def compute_f(self, x, y, z):
        exp = math.exp(-x*y*(math.exp(z) - 1)**2 / (1 + math.exp(z)))
        exp_ = math.exp(-x*y*(1 - math.exp(-z))**2 / 2)
        return (exp + exp_) / (1 - (1 - y)**x)
    
    
    def compute_bound(self, n):
        exp = 2*self.m*(1 - self.w*self.tau)**n
        f = self.compute_f(n, self.w*self.tau, self.gamma/12)
        return exp + 4*f


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
        if self.enable_safety:
            # computing each |Nj - Nj+1| and |Mj - Mj+1|
            # and check whether any of them is too large
            nj = np.abs(nj[::, :-1] - nj[::, 1:]) > n*self.safety_bound
            mj = np.abs(mj[::, :-1] - mj[::, 1:]) > n*self.safety_bound
            count = np.logical_or(nj, mj).any(axis=1)
            return 1 - np.sum(count) / runs
        else:
            # remove all rows with at least one Nj or Mj equal to 0
            mask = np.logical_and(np.all(nj > 0, axis=1),
                                  np.all(mj > 0, axis=1))
            nj = nj[mask]
            mj = mj[mask]
            if len(nj) > 0:
                # estimated epsilon
                estimates = np.log(np.max(np.maximum(nj / mj, mj / nj),
                                          axis=1))
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
    
    def exec(self, nj, mj, runs, n):
        if self.enable_safety:
            # computing each |Nj - Nj+1| and |Mj - Mj+1|
            # and check whether any of them is too large
            nj = np.abs(nj[::, :-1] - nj[::, 1:]) > n*self.safety_bound
            mj = np.abs(mj[::, :-1] - mj[::, 1:]) > n*self.safety_bound
            count = np.logical_or(nj, mj).any(axis=1)
            return 1 - np.sum(count) / runs
        else:
            # remove all rows with at least one Nj or Mj equal to 0
            mask = np.logical_and(np.all(nj > 0, axis=1),
                                  np.all(mj > 0, axis=1))
            nj = nj[mask]
            mj = mj[mask]
            if len(nj) > 0:
                # estimated epsilon
                estimates = np.log(np.max(np.maximum(nj / mj, mj / nj),
                                          axis=1))
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
            
            
    def exec2(self, nj, mj):
        mask = np.logical_and(np.all(nj > 0, axis=1),
                                np.all(mj > 0, axis=1))
        nj = nj[mask]
        mj = mj[mask]
        if len(nj) > 0:
            estimate = np.log(np.max(np.maximum(nj / mj, mj / nj)))
            return np.abs(estimate - self.epsilon) <= self.gamma
        return False

        
    def run_estimator2(self, mechanism, mechanism_, n, runs):
        prop = 0
        for _ in range(runs):
            samples = mechanism.sample(n, 1)
            nj = self.count(samples)
            samples = mechanism_.sample(n, 1)
            mj = self.count(samples)
            prop += 1 if self.exec2(nj, mj) else 0
        return prop / runs
    
    
def plot_dependencies():
    plt.rc('axes', titlesize=12, labelsize=12,
           titleweight='bold', labelweight='bold')
    plt.rc('figure', titlesize=14, titleweight='bold')
    fig = plt.figure(figsize=(20, 10))
    # fig.suptitle("Log of the number of samples as functions of "
                #  + r"$\mathbf{\gamma, \delta}$ and C")
    gamma = np.linspace(.5, .001, 100)
    delta = np.linspace(.5, .999, 10)
    C = np.linspace(.5, 1.7, 10)
    f = lambda g, d, c: Estimator(gamma=g, delta=d, C=c,
                                  W=1, enable_safety=False).n
    title = r'log$\mathbf{(n)}$ as a function of '

    # n as a function of gamma and delta
    ax = fig.add_subplot(131, projection='3d')
    X, Y = np.meshgrid(gamma, delta)
    Z = np.log(np.vectorize(lambda g, d: f(g, d, 1))(X, Y))
    ax.plot_surface(X, Y, Z, cmap='copper')
    ax.set_xlabel(r'$\mathbf{\gamma}$')
    ax.set_ylabel(r'$\mathbf{\delta}$')
    ax.set_zlabel(r'log$\mathbf{(n)}$')
    # ax.set_title(title + r'$\mathbf{\gamma}$ and $\mathbf{\delta}$')
    
    # n as a function of gamma and C
    ax = fig.add_subplot(132, projection='3d')
    X, Y = np.meshgrid(gamma, C)
    Z = np.log(np.vectorize(lambda g, c: f(g, .9, c))(X, Y))
    ax.plot_surface(X, Y, Z, cmap='copper')
    ax.set_xlabel(r'$\mathbf{\gamma}$')
    ax.set_ylabel('C')
    ax.set_zlabel(r'log$\mathbf{(n)}$')
    # ax.set_title(title + r'$\mathbf{\gamma}$ and $\mathbf{C}$')#, y=-.2)
    
    # n as a function of C and delta
    ax = fig.add_subplot(133, projection='3d')
    X, Y = np.meshgrid(delta, C)
    Z = np.log(np.vectorize(lambda d, c: f(.1, d, c))(X, Y))
    ax.plot_surface(X, Y, Z, cmap='copper')
    ax.set_xlabel(r'$\mathbf{\delta}$')
    ax.set_ylabel('C')
    ax.set_zlabel(r'log$\mathbf{(n)}$')
    # ax.set_title(title + r'$\mathbf{\delta}$ and $\mathbf{C}$')
    
    plt.show()
    

if __name__ == "__main__":
    plot_dependencies()