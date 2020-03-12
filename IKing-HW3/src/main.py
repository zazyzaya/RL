import matplotlib.pyplot as plt

from agent_class import HW3Agent
from build_environment import build_map, STATEMAP

MAP = build_map()

def test(episode_len, num_episodes, epsilon):
    Agent = HW3Agent(MAP, STATEMAP, epsilon)
    Agent.on_policy_mc(episode_len, num_episodes)
    
    y,x = Agent.generate_best_path(cutoff=1000, stochastic=True)
    pathlen = len(x) if len(x) < 1000 else 'INF'

    print("Len of best path: " + str(pathlen))
    plt.imshow(MAP)
    plt.scatter(x,y)
    plt.show()

test(50, 10000, 0.02)