import matplotlib.pyplot as plt
import numpy as np
import os 

from math import inf, exp
from agent_class import SARSA_Agent, Q_learning_agent
from build_environment import build_map, STATEMAP
from joblib import Parallel, delayed

NUM_RUNS = 100
NUM_EPISODES = 10000

MAP = build_map()

HOME, fn = os.path.split(__file__)
SARSA_FILE = 'sarsa_'
Q_FILE = 'q-learning_'

# Best SARSA
gamma = 0.99
alpha =     lambda x : exp(-x/50000)
epsilon =   lambda x : exp(-x/500)

# Best Q-learning
qgamma = 0.99
qalpha =     lambda x : exp(-x/500000)
qepsilon =   lambda x : exp(-x/50)

# Make sure each proc uses a different object (learned this 
# the hard way...)
def task(g, a, e, Agent):
    agent = Agent(MAP, STATEMAP, 1, g, a, e)
    return agent.run(NUM_EPISODES)

def generate_arr(
    g=qgamma, a=qalpha, e=qepsilon, 
    workers=10, save_policy=False, Agent='q'
    ):

    if Agent == 'sarsa':
        Agent = SARSA_Agent
        fname = SARSA_FILE
    elif Agent == 'q':
        Agent = Q_learning_agent
        fname = Q_FILE

    avg = Parallel(n_jobs=workers, prefer='processes')(
        delayed(task)(g,a,e,Agent)
        for _ in range(NUM_RUNS)
    )

    # Run one more time to get a save on the policy 
    if save_policy:
        Agent(MAP, STATEMAP, 1, g, a, e, save_policy=True, fname=fname).run(NUM_EPISODES)

    avg = np.average(avg, axis=0)
    np.save(fname + 'avg.npy', avg)
    return avg

def test():
    plt.plot(range(NUM_EPISODES), generate_arr(), '-g', label='0.99')
    plt.plot(range(NUM_EPISODES), generate_arr(g=0.9), '-r', label='0.90')
    plt.plot(range(NUM_EPISODES), generate_arr(g=0.8), '-b', label='0.80')
    plt.plot(range(NUM_EPISODES), generate_arr(g=0.75), '-', label='0.75')

    plt.legend()
    plt.title('Q-learning Different gamma values')
    plt.show()

def run_all():
    plt.plot(range(NUM_EPISODES), generate_arr(save_policy=True), '-b', label='Q-Learning')
    plt.plot(range(NUM_EPISODES), generate_arr(Agent='sarsa', g=gamma, a=alpha, e=epsilon, save_policy=True), '-r', label='SARSA')

    plt.legend()
    plt.title('SARSA v. Q-learning')

    plt.show()

def draw_arrow(q, s):
    direction = np.argmax(q[s]) # L, R, U, D

    kargs = dict(
        length_includes_head=True, head_width=0.25, head_length=0.25
    )

    y = s[0]-0.5
    x = s[1]-0.5
    inc = 1
    off = 0.5

    if direction == 0:
        plt.arrow(x+inc,y+off,-inc,0, **kargs)
    elif direction == 1:
        plt.arrow(x,y+off,inc,0, **kargs)
    elif direction == 2:
        plt.arrow(x+off,y+inc,0,-inc, **kargs)
    else:
        plt.arrow(x+off,y,0,inc, **kargs)
    

def plot_policies(fname, title):
    qp = np.load(fname)
    for x in range(qp.shape[0]):
        for y in range(qp.shape[1]):
            if MAP[(x,y)] != 1:
                draw_arrow(qp, (x,y))
    
    plt.imshow(MAP)
    plt.title(title + ' Best Policy')
    plt.show()

run_all()
[plot_policies(f[0]+'_policy.npy', f[1]) for f in [('sarsa', 'SARSA'), ('q-learning', 'Q Learning')]]