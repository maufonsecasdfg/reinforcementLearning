import numpy as np

class Agent:
    def __init__(self, k, method, epsilon, alpha):
        '''
        - k:
            number of arms
        - epislon:
            for epsilon-greedy method
        -alpha:
            step parameter
            string "1/n" implements incremental sample average as estimation
            float where 0 < alpha <= 1 implements wheighted moving average
        '''
        self.Q = np.zeros(k)
        self.k = k
        self.epsilon = epsilon
        self.N = np.zeros(k)
        self.method = method
        if type(alpha) == float:
            self.alpha_style = 'constant'
            self.alpha = np.full(k,alpha)
        elif alpha == '1/n':
            self.alpha_style = '1/n'
            self.alpha = np.zeros(k)

    def action(self,bandit):
        '''
        - bandit:
            The k-armed bandit that the agent interacts with
        '''
        if self.method == 'epsilon-greedy':
            arm = self.epsilon_greedy_choice()
        R = bandit.get_reward(arm)
        self.update_Q(arm,R)
        return arm, R
        
    def update_Q(self,arm,R):
        self.update_alpha(arm)
        self.Q[arm] = self.Q[arm] + self.alpha[arm]*(R-self.Q[arm])

    def update_alpha(self,arm):
        self.N[arm] += 1
        if self.alpha_style == '1/n':
            self.alpha[arm] = 1/self.N[arm]

    def epsilon_greedy_choice(self):
        if np.random.uniform() <= 1.0-self.epsilon:
            #Exploit
            arm = np.random.choice(np.argwhere(self.Q == np.amax(self.Q)).transpose()[0])
        else:
            #Explore
            arm = np.random.choice(np.arange(self.k))
        return arm



