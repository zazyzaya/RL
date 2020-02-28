
import numpy as np

class Agent():
    def __init__(self, states, actions, policy):
        self.states = states
        self.actions = actions
        self.policy = policy
        self.curstate = None

    def state_transition(self, action):
        pass


class HW2Agent(Agent):
    def __init__(self, states, statemap, gamma, p, theta):
        actions = ['l', 'r', 'u', 'd']
        policy = np.random.randint(0, high=4, size=states.shape)
        super().__init__(states, actions, policy)

        self.value = np.zeros(states.shape)
        self.curstate = np.where(states == statemap['start'])


    ''' Calculates reward for action
        and moves to next state
    '''
    def state_transition(self, action, change_state=True):
        reward = -1
        
        if action == 'l':
            next_s = (self.curstate[0], self.curstate[1]-1)
        elif action == 'r'
            next_s = (self.curstate[0], self.curstate[1]+1)
        elif action == 'u'
            next_s = (self.curstate[0]-1, self.curstate[1])
        elif action == 'd':
            next_s = (self.curstate[1]+1, self.curstate[1])

        statetype = self.states(next_s)

        if statetype == self.statemap['wall']:
            next_s = self.curstate
        elif statetype == self.statemap['oil']:
            reward -= 5
        elif statetype == self.statemap['hole']:
            reward -= 10
        elif statetype == self.statemap['goal']:
            reward += 200

        if change_state:
            self.curstate = next_s

        return reward