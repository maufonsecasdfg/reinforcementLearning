import pandas as pd
from datetime import datetime
from kArmedBanditRun import *
from Agent import *
from kArmedBandit import *
from scipy.stats import norm


def run_experiment(runs, steps, k, bandit_type, reward_mean, reward_std, arm_dist_std, nonstat_bandit_rand_walk_update_std, agent_method, agent_epsilon, agent_alpha):
    results = pd.DataFrame()
    experiment_id = datetime.now().strftime("%Y%m%d%H%M%S")
    experiment_parameters = pd.DataFrame(pd.Series({
        'experiment_id' : experiment_id,
        'runs' : runs,
        'steps' : steps,
        'k' : k,
        'bandit_type' : bandit_type,
        'reward_mean' : reward_mean,
        'reward_std' : reward_std,
        'arm_dist_std' : arm_dist_std,
        'nonstat_bandit_rand_walk_update_std' : nonstat_bandit_rand_walk_update_std,
        'agent_method' : agent_method,
        'agent_epsilon' : agent_epsilon,
        'agent_alpha' : agent_alpha
    })).transpose()

    for j in range(runs):
        distributions = [norm]*k
        dist_params = [[norm(reward_mean,reward_std).rvs(),arm_dist_std] for i in range(k)]
        if bandit_type == 'stationary':
            arms = kArmedBandit(k=k, distributions=distributions, dist_params=dist_params)
        elif bandit_type == 'non-stationary':
            arms = kArmedBanditNonStationary(k=k, distributions=distributions, dist_params=dist_params, rand_walk_update_std=nonstat_bandit_rand_walk_update_std)
        agent = Agent(k = k, method=agent_method, epsilon=agent_epsilon,  alpha=agent_alpha)
        runner = kArmedBanditRun(arms, agent, steps=steps)
        runner.run()
        
        r = runner.get_results_dataframe()
        if bandit_type == 'stationary':
            r['best_arm'] = runner.arms.best_arm
        elif bandit_type == 'non-stationary':
            r = r.join(runner.arms.get_best_arm_history())
        r['run'] = j
        results = pd.concat([results,r])
        
    results['chosen_best'] = results['best_arm'] == results['chosen_arm']
    results['experiment_id'] = experiment_id
    results.to_csv(f'experiments/kArmedBanditExperiment_{experiment_id}.csv',index=False)
    with open('experiments/experiment_parameters.csv', 'a') as f:
        experiment_parameters.to_csv(f, index=False, header=False, line_terminator='\n')
    
    return results, experiment_parameters

def process_results(results, parameters):
    mean_results = results.groupby(['experiment_id','step'])[['reward','chosen_best']].mean()
    mean_results = mean_results.reset_index().set_index('experiment_id').join(parameters.set_index('experiment_id')).reset_index()
    return(mean_results)