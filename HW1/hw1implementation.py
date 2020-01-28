from hw1classes import Bandit, EGreedy
from numpy import random 
import matplotlib.pyplot as plt
import json

# Create bandit with specified distributions
l1 = lambda : random.normal(5, 10**(1/2))
l2 = lambda : sum(random.normal([10,4], [15**(1/2), 10**(1/2)])) 
bandit = Bandit(2, [l1, l2])

# Need to ask about this one, depending on how it's initialized it gets
# too small too quickly, or remains the same
alpha3_lookup_table = [1.0]
for i in range(1, 1000):
    alpha3_lookup_table.append(alpha3_lookup_table[i-1] ** -i)

# Section a.) 
def parta():
    e_values = [0, 0.1, 0.2, 0.5]
    a_values = [
        lambda x : 1,
        lambda x : 0.9 ** x,
        lambda x : alpha3_lookup_table[x],
        lambda x : 1/(x+1)
    ]

    titles = [
        'a = 1', 
        'a = 0.9^k',
        'a = a^(-k)',
        'a = 1/k'
    ]

    i = 0
    for a in a_values:
        stats = {} 
        lines = []
        for e in e_values:
            stats[e] = EGreedy(bandit, e, a).run()
            name, = plt.plot(stats[e].pop('reward_per_iter'), label='epsilon = ' + str(e))
            lines.append(name)

        print(titles[i])
        print(json.dumps(stats, indent=4))

        plt.legend(handles=lines)
        plt.title(titles[i])
        plt.show()

        i+=1
    
# Part b
def partb():
    Q0 = [(0,0), (5,5), (10,10), (7,7)]
    e = 0.1
    a = lambda x : 0.1

    i = 0
    stats = {} 
    lines = []
    for q in Q0:
        stats[str(q)] = EGreedy(bandit, e, a, q0=q).run()
        name, = plt.plot(stats[str(q)].pop('reward_per_iter'), label='Q0 = ' + str(q))
        lines.append(name)

    print(json.dumps(stats, indent=4))

    plt.legend(handles=lines)
    plt.title('Different Q0 Values')
    plt.show()

parta()
partb()