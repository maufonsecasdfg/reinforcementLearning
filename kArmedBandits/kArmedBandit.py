import numpy as np
from scipy.stats import norm
import pandas as pd

class kArmedBandit:
    def __init__(self, k, distributions, dist_params):
        '''
        - k: 
            number of arms
        - distributions: 
            list of base structure of the distributions, one per arm, 
            without parameters defined (for example, just scipy.stats.norm for normal distributions)
        - dist_params: 
            list of lists, one per arm, each list containing the parameters to initialize each distribution,
            ordered by the order of definition for the respective base distribution ([mean,std], for scipy.stats.norm)
        '''
        if k != len(distributions):
            raise ValueError("distributions must have k entries.")
        if k != len(dist_params):
            raise ValueError("dist_params must have k entries.")
        self.k = k
        self.base_distributions = distributions
        self.dist_params = dist_params
        self.distributions = [distributions[i](*dist_params[i]) for i in range(len(distributions))]
        self.best_arm = np.argmax([self.dist_params[i][0] for i in range(len(self.dist_params))])

    def get_reward(self,arm):
        return self.distributions[arm].rvs()


class kArmedBanditNonStationary(kArmedBandit):
    '''
    For a k-Armed Bandit problem where each distribution's mean is updated in each step by performing
    a gaussian random walk with mean 0 and defined STD.
    Note: The update method assumes all distributions are Gaussian.
    '''
    def __init__(self, k, distributions, dist_params, rand_walk_update_std, save_dist_mean_history=False):
        '''
        - rand_walk_update_std: 
            Standard deviation of the gayssian random walk performed on the mean
            of the distributions to update the distributions at each step.
        - save_dist_mean_history:
            Save the mean of each distribution at each step.
        '''
        super().__init__(k, distributions, dist_params)
        self.rand_walk_update_std = rand_walk_update_std
        self.best_arm_history = []
        self.distribution_mean_history = []
        self.save_dist_mean_history = save_dist_mean_history
        for i in range(k):
            self.distribution_mean_history.append([])

    def get_reward(self,arm):
        self.best_arm_history.append(self.best_arm)
        if self.save_dist_mean_history:
            for i in range(self.k):
                self.distribution_mean_history[i].append(self.distributions[i].mean())
        R = self.distributions[arm].rvs()
        self.update_all_distributions()
        return R

    def update_one_distribution(self, arm):
        distribution = self.base_distributions[arm]
        self.dist_params[arm] = [self.dist_params[arm][0]+norm(0,self.rand_walk_update_std).rvs(),self.dist_params[arm][1]]
        self.distributions[arm] = distribution[self.dist_params[arm]]
        self.best_arm = np.argmax([self.dist_params[i][0] for i in range(len(self.dist_params))])

    def update_all_distributions(self):
        distributions = self.base_distributions
        self.dist_params = [[self.dist_params[i][0]+norm(0,self.rand_walk_update_std).rvs(),self.dist_params[i][1]] for i in range(self.k)]
        self.distributions = [distributions[i](*self.dist_params[i]) for i in range(len(distributions))]
        self.best_arm = np.argmax([self.dist_params[i][0] for i in range(len(self.dist_params))])

    def get_best_arm_history(self):
        return pd.DataFrame({'best_arm':self.best_arm_history})

    def get_distribution_mean_history(self):
        df = pd.DataFrame()
        for i in range(self.k):
            df[i] = self.distribution_mean_history[i]
        return df


