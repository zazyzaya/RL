import matplotlib.pyplot as plt
import numpy as np

from math import inf, exp
from agent_class import SARSA_Agent
from build_environment import build_map, STATEMAP

NUM_RUNS = 100
NUM_EPISODES = 10000

MAP = build_map()

SARSA_FILE = 'sarsa.npy'

# SARSA
gamma = 0.75
alpha =     lambda x : exp(-x/50)
epsilon =   lambda x : exp(-x)

def generate_sarsa(g=gamma, a=alpha, e=epsilon):
    SARSA_avg = []
    SARSA = SARSA_Agent(MAP, STATEMAP, 1, g, a, e)
    for i in range(NUM_RUNS):
        print('%d/%d' % (i, NUM_RUNS))
        SARSA_avg.append(SARSA.run(NUM_EPISODES))

    SARSA_avg = np.average(SARSA_avg, axis=0)
    np.save(SARSA_FILE, SARSA_avg)
    return SARSA_avg

generate_sarsa()