import random
import numpy as np

from math import inf

class Agent():
    def __init__(self, states, actions, policy):
        self.states = states
        self.actions = actions
        self.policy = policy
        self.curstate = None

    def state_transition(self, action):
        pass

class HW2Agent(Agent):
    def __init__(self, states, statemap, p, gamma, theta):
        self.actions_map = ['l', 'r', 'u', 'd']
        policy = np.random.randint(0, high=4, size=states.shape)
        super().__init__(states, list(range(4)), policy)

        coords = np.where(states == statemap['start'])
        self.start = (coords[0][0], coords[1][0])
        self.statemap = statemap
        self.polymap = {'l':0, 'r':1, 'u':2, 'd':3}
        self.p = p
        self.values = np.zeros(states.shape)
        self.gamma = gamma
        self.theta = theta
        self.num_improvements = 0
        self.num_evals = 0

        self.curstate = self.start


    ''' Assuming all steps are taken perfectly, generates a sequence of steps 
        that are the most optimal path 
    '''
    def generate_best_path(self, stochastic=False, cutoff=None, start=None):
        cur_s = self.statemap['start']
        goal = self.statemap['goal']
        
        if start == None:
            self.curstate = self.start
        else:
            self.curstate = start

        pathx = [self.curstate[0]]
        pathy = [self.curstate[1]]

        path_len = 0
        if cutoff == None:
            cutoff = inf

        while(cur_s != goal and path_len < cutoff):
            if stochastic:
                self.state_transition(self.choose_action(self.policy[self.curstate]))
            
            else:
                self.state_transition(self.policy[self.curstate])
            
            pathx.append(self.curstate[0])
            pathy.append(self.curstate[1])
            
            cur_s = self.states[self.curstate]
            path_len += 1

        return pathx, pathy

    ''' Chooses an action with random noise p as specified in specs
    '''
    def choose_action(self, action):
        if (random.random() > self.p):
            return action 
        else:
            return np.random.choice(
                list(
                    set({0,1,2,3}).difference({action})
                )
            )


    ''' Calculates reward for action
        and moves to next state
    '''
    def state_transition(self, a, curstate=None, change_state=True):
        reward = -1
        
        # Can actually run, or predict reward for an input state
        if curstate == None:
            curstate = self.curstate
        
        if self.actions_map[a] == 'l':
            next_s = (curstate[0], curstate[1]-1)
        elif self.actions_map[a] == 'r':
            next_s = (curstate[0], curstate[1]+1)
        elif self.actions_map[a] == 'u':
            next_s = (curstate[0]-1, curstate[1])
        elif self.actions_map[a] == 'd':
            next_s = (curstate[0]+1, curstate[1])

        if (next_s[0] < self.states.shape[0] and next_s[1] < self.states.shape[1]):
            statetype = self.states[next_s]
        else:
            return curstate, reward

        if statetype == self.statemap['wall']:
            next_s = curstate
        elif statetype == self.statemap['oil']:
            reward = -5
        elif statetype == self.statemap['hole']:
            reward = -10
        elif statetype == self.statemap['goal']:
            reward = 200

        if change_state:
            self.curstate = next_s

        return next_s, reward


    ''' Returns Q(s,a) = Sum_{s', r} P(s', r|s, a) * [r + gamma*V(s')]
    '''
    def Q(self, s, a):
        q_sum = 0
        for a_prime in self.actions:
            if a_prime == a:
                p_of_sr = 1-self.p
            else:
                p_of_sr = self.p/3
            
            s_prime, r = self.state_transition(a_prime, curstate=s, change_state=False)
            q_sum += p_of_sr * (r + self.gamma*self.values[s_prime])
        
        return q_sum


    ''' Iterates between policy eval and policy improvement until stability 
        is reached. 
    '''
    def policy_iteration(self):
        stable = False
        while(not stable):
            self.policy_evaluation()
            stable = self.policy_improvement()


    ''' Runs policy evaluation on agent 
    '''
    def policy_evaluation(self):
        delta = 0
        rows, cols = self.states.shape

        while(True):
            self.num_evals += 1
            for row in range(rows):
                for col in range(cols):
                    v = self.values[row, col]
                    
                    # Because Pi(a|s) = 0 for all actions other than the deterministic policy
                    # we can skip the summation of Pi(a|s) for all actions 
                    v_prime = self.Q((row,col), self.policy[(row,col)])

                    self.values[row,col] = v_prime
                    delta = max(delta, abs(v-v_prime))

            # Continues until convergence reached
            if delta < self.theta:
                break
            
            delta = 0

        return True


    ''' Continually calls policy evaluation until policy stable
        call policy improvement once first to initialize V
    '''
    def policy_improvement(self):
        self.num_improvements += 1
        if not self.num_improvements % 50 and self.num_improvements > 0:
            print("Agent has made " + str(self.num_improvements) + " Improvements...") 

        policy_stable = True
        rows, cols = self.states.shape

        for row in range(rows):
            for col in range(cols):
                b = self.policy[row, col]

                argmax = (-inf, None)
                for a in self.actions:
                    val = self.Q((row, col), a)
                    
                    if val > argmax[0]:
                        argmax = (val, a)

                self.policy[row,col] = argmax[1]

                if b != argmax[1]: 
                    policy_stable = False

        return policy_stable
        

    ''' Generates policy using value iteration
    '''
    def value_iteration(self):
        delta = inf
        rows, cols = self.values.shape
        
        # Iterate through all states evaluating V(x,y) until no change 
        while(delta > self.theta):
            self.num_evals += 1
            delta = 0

            for row in range(rows):
                for col in range(cols):
                    s = (row, col)
                    v = self.values[(row, col)]

                    max_a = -inf
                    for a in self.actions:
                        q = self.Q(s, a)
            
                        if q > max_a:
                            max_a = q
                    
                    self.values[s] = max_a
                    delta = max(delta, abs(v-max_a))

        # Then set each Pi to the maximum valued action (as determined by V(x,y))
        for row in range(rows):
            for col in range(cols):
                s = (row, col)

                max_a = (-inf, None)
                for a in self.actions:
                    q = self.Q(s, a)

                    if q > max_a[0]:
                        max_a = (q, a)
                
                self.policy[s] = max_a[1]


class HW3Agent(HW2Agent):
    SA_TUPLE = 0
    REWARD = 1

    def __init__(self, states, statemap, p, default_r=-1):
        # Just give dummy values for gamma and theta
        super().__init__(states, statemap, p, 1, 1) 
        self.returns = {}
        self.default_r = default_r

    def random_walk(self, episode_len, state=[]):
        if state == []:
            state = self.start
        
        walk = []
        for _ in range(episode_len):
            s_prime, r, a = self.step(state)
            walk.append(((state,a), r))

            s = s_prime
        
        return walk
        
    def step(self, s, a=None):
        if a == None:
            a = self.choose_action(self.policy[s])

        s_prime, r = self.state_transition(a, curstate=s, change_state=False)
        return s_prime, r, a

    def reset(self):
        self.returns.clear()
        self.policy = np.random.randint(0, high=4, size=self.states.shape)

    def Q(self, s, a):
        if (s,a) in self.returns:
            return np.average(self.returns[(s,a)])
        else: 
            return self.default_r

    def on_policy_mc(self, episode_len, num_episodes):
        for e in range(num_episodes):
            start = np.random.randint((1,2), high=self.states.shape)
            start = (start[0], start[1])
            episode = self.random_walk(episode_len, state=start)

            s_list = set()
            for e in episode:
                sa = e[self.SA_TUPLE]
                r = e[self.REWARD]
                s_list.add(sa[0])

                # Add/Append reward to returns dict
                if sa in self.returns:
                    r_sa = self.returns[sa]
                    self.returns[sa] = np.append(r_sa, r)
                else:
                    self.returns[sa] = np.array([r])

            for s in s_list:
                a_vals = [self.Q(s,a) for a in self.actions]
                
                a_star = np.random.choice(
                    np.where(
                        a_vals == np.max(a_vals)
                    )[0]
                )

                self.policy[s] = a_star

           #self.returns.clear()