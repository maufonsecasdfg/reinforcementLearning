import numpy as np
from kArmedBandit import *
from Agent import *
import pandas as pd

class kArmedBanditRun:
    def __init__(self, bandit, agent, steps):
        self.arms = bandit
        self.agent = agent
        self.steps = steps
        self.rewards = []
        self.chosen_arms = []

    def run(self):
        for s in range(self.steps):
            arm, R = self.agent.action(self.arms)
            self.rewards.append(R)
            self.chosen_arms.append(arm)

    def get_results_dataframe(self):
        return pd.DataFrame({'step': list(range(self.steps)),'reward':self.rewards,'chosen_arm':self.chosen_arms})
        

    