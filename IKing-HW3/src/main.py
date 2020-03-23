import matplotlib.pyplot as plt

from math import inf
from agent_class import HW3Agent
from build_environment import build_map, STATEMAP

MAP = build_map()

def test(episode_len, num_episodes, epsilon, default_r=-1):
    Agent = HW3Agent(MAP, STATEMAP, epsilon, default_r=default_r)
    Agent.on_policy_mc(episode_len, num_episodes)
    
    y,x = Agent.generate_best_path(cutoff=1000, stochastic=True)
    pathlen = len(x) if len(x) < 1000 else 'INF'

    pathlen = "%d episodes with %d steps: %s" % (num_episodes, episode_len, str(pathlen))
    print(pathlen)
    plt.imshow(MAP)
    plt.scatter(x,y)
    
    title = "%d episodes of length %d" % (num_episodes, episode_len)
    plt.suptitle(title, fontsize=15)
    plt.title(pathlen)
    #plt.show()


for episodes in range(6):
    for elen in range(6):
        test(10 ** elen, 10 ** episodes, 0.25, default_r=-inf)


'''
for episodes in range(10000, 130000, 30000):
    for elen in range(100, 1200, 300):
        test(elen, episodes, 0.02, default_r=-inf)
'''