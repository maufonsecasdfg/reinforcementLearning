import numpy as np
from scipy.special import softmax

class Agent:
    def __init__(self, k, method, alpha, Q_1=None, epsilon=None, c=None):
        '''
        -k:
            number of arms
        -method:
            +'epsilon-greedy' : requires specification of epsilon
            +'ucb' : (Upper-Confidence-Bound) requires specification of c
            +'gradient-bandit' : requires specification of numeric alpha
        -epislon:
            for epsilon-greedy method
        -Q_1:
            initial estimations. ordered numpy array of length k
            if None, initial estimations are all 0
        -alpha:
            step parameter
            if method is 'epsilon-greedy' or 'ucb'
                + string "1/n" implements incremental sample average as estimation
                + float where 0 < alpha <= 1 implements wheighted moving average
            if method is 'gradient-bandit'
                + float where 0 < alpha <= 1 for the update of the preferences
        -c:
            degree of exploration for upper-confidence-bound selection method
        '''
        if method == 'epsilon-greedy' or method == 'ucb':
            self.action_value_estimation = True
            self.H = None
            self.pi = None
            self.R_mean = None
            if Q_1 is None:
                self.Q = np.zeros(k)
            else:
                self.Q = Q_1
        elif method == 'gradient-bandit':
            self.action_value_estimation = False
            self.Q = None
            self.H = np.zeros(k)
            self.pi = softmax(self.H)
            self.R_mean = np.zeros(k)
        self.k = k
        self.epsilon = epsilon
        self.c = c
        self.N = np.zeros(k)
        self.method = method
        self.t = 1
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
        elif self.method == 'ucb':
            arm = self.ucb_choice()
        elif self.method == 'gradient-bandit':
            arm = self.gradient_bandit_choice()
        R = bandit.get_reward(arm)
        
        if self.action_value_estimation:
            self.update_Q(arm,R)
        else:
            self.update_pi(arm,R)
        self.update_step(arm)
        return arm, R
        
    def update_Q(self,arm,R):
        self.update_alpha(arm)
        self.Q[arm] = self.Q[arm] + self.alpha[arm]*(R-self.Q[arm])

    def update_R_mean(self,R):
        self.R_mean = self.R_mean + (1/self.t)*(R-self.R_mean)

    def update_pi(self,arm,R):
        for i in range(self.k):
            if i == arm:
                self.H[i] = self.H[i] + self.alpha*(R-self.R_mean)*(1-self.pi[i])
            else:
                self.H[i] = self.H[i] - self.alpha*(R-self.R_mean)*self.pi[i]
        self.pi = softmax(self.H)
        self.update_R_mean(R)

    def update_step(self,arm):
        self.N[arm] += 1
        if self.alpha_style == '1/n':
            self.alpha[arm] = 1/self.N[arm]
        self.t += 1

    def epsilon_greedy_choice(self):
        if np.random.uniform() <= 1.0-self.epsilon:
            #Exploit
            arm = np.random.choice(np.argwhere(self.Q == np.amax(self.Q)).transpose()[0])
        else:
            #Explore
            arm = np.random.choice(np.arange(self.k))
        return arm
    
    def ucb_choice(self):
        with np.errstate(divide='ignore'):
            m = self.Q + self.c*np.sqrt(np.log(self.t)/self.N)
        arm = np.random.choice(np.argwhere(m == np.amax(m)).transpose()[0])
        return arm

    def gradient_bandit_choice(self):
        arm = np.choice(list(range(self.k)),p=(softmax(self.H)))
        return arm
