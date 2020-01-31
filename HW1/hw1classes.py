class Bandit():
    def __init__(self, n_arms, probs):
        ''' Takes the number of arms, and a dict of 
            arm_number -> prize function 
            where prize function returns some number (randomly or otherwise)
        '''
        self.n = n_arms
        self.probs = probs

    def pull_lever(self, lever):
        return self.probs[lever]()

import random 
import numpy as np
class EGreedy():
    def __init__(self, bandit, epsilon=0.0, alpha=None, q0=None):
        # Static fields
        self.bandit = bandit
        self.epsilon = epsilon
        self.q0 = q0
        self.alpha = alpha
        
        # Reset after each run
        self.Q = {}
        self.N = [0] * bandit.n
        self.rewards = []
        self.running_total = 0 

        if self.alpha == None:
            self.alpha = lambda x : 1

        if self.q0 == None:
            self.q0 = [0] * bandit.n

        self.init_q()

    def init_q(self):
        for i in range(self.bandit.n):
            self.Q[i] = self.q0[i]


    ''' Resets stats for use in repeated trials
    '''
    def reset(self):
        self.Q = {}
        self.N = [0] * self.bandit.n 
        self.rewards = []
        self.running_total = 0
        self.init_q()


    ''' Update Q(a) for a given Q and reward R
    '''
    def update_q(self, q, r):
        self.running_total += r
        self.N[q] += 1
        self.Q[q] = self.Q[q] + self.alpha(self.N[q]) * (r - self.Q[q]) 


    ''' Chooses either best Q(a) or random using e-greedy alg
    '''
    def choose(self):
        if (random.random() > self.epsilon):
            qt = max(self.Q, key=self.Q.get)
        else:
            qt = random.randint(0, self.bandit.n-1)
        
        rt = self.bandit.pull_lever(qt)

        self.update_q(qt, rt)
        self.rewards.append(rt)


    ''' Returns the average of 100 runs for 1000 steps
    '''
    def run(self, trials=100, steps=1000):
        rewards = []
        totals = []
        Qs = []

        # Run 100 trials
        for i in range(trials):
            for j in range(steps):
                self.choose()
            
            # Save all data
            rewards.append(self.rewards)
            totals.append(self.running_total)
            Qs.append([self.Q[i] for i in range(len(self.Q))])

            # Clear data
            self.reset()

        return {
            'reward_per_iter': np.mean(rewards, axis=0),
            'final_amt' : np.mean(totals),
            'std_qs': np.array(Qs).std(axis=0).tolist(),
            'avg_qs': np.array(Qs).mean(axis=0).tolist()
        }