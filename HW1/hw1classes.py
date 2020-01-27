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
class EGreedy():
    def __init__(self, bandit, epsilon=0.0):
        self.bandit = bandit
        self.epsilon = epsilon
        self.Q = {}
        self.N = [0] * bandit.n
        self.reward = 0
        self.running_total = [] 

        for i in range(bandit.n):
            self.Q[i] = 0


    ''' Update Q(a) for a given Q and reward R
    '''
    def update_q(self, q, r):
        self.reward += r
        self.N[q] += 1
        self.Q[q] = self.Q[q] + 1/self.N[q] * (r - self.Q[q]) 


    ''' Chooses either best Q(a) or random using e-greedy alg
    '''
    def choose(self):
        if (random.random() > self.epsilon):
            qt = max(self.Q, key=self.Q.get)
        else:
            qt = random.randint(0, self.bandit.n-1)
        
        rt = self.bandit.pull_lever(qt)

        self.update_q(qt, rt)
        self.running_total.append(rt)