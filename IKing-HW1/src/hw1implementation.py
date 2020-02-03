from hw1classes import Bandit, EGreedy
from numpy import random 
import matplotlib.pyplot as plt
import json
import math
import csv

# Create bandit with specified distributions
l1 = lambda : random.normal(5, 10**(1/2))
l2 = lambda : sum(random.normal([10,4], [15**(1/2), 10**(1/2)])) 
bandit = Bandit(2, [l1, l2])

# Section a.) 
def parta():
    f = open('varying_a_e.csv', 'w+', newline='\n')
    writer = csv.writer(f)
    writer.writerow(['Epsilon', 'Alpha', 'Final Average', 'Std Dev Q1', 'Std Dev Q2', 'Avg Q1', 'Avg Q2'])

    e_values = [0, 0.1, 0.2, 0.5]
    a_values = [ 
        lambda x : 1,
        lambda x : 0.9 ** x,
        lambda x : 1/(1+math.log(1 + x)),
        lambda x : 1/(x+1)
    ]

    titles = [
        'a = 1', 
        'a = 0.9^k',
        'a = 1/(1+ln(1+x))',
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

            writer.writerow([
                e, titles[i], 
                stats[e]['final_amt'], 
                stats[e]['std_qs'][0], stats[e]['std_qs'][1], 
                stats[e]['avg_qs'][0], stats[e]['avg_qs'][1] 
            ])

        print(titles[i])
        print(json.dumps(stats, indent=4))

        plt.legend(handles=lines)
        plt.title(titles[i])
        plt.ylim((0,15))
        plt.show()

        i+=1
    
    f.close()
    
# Part b
def partb():
    Q0 = [(0,0), (5,5), (10,10), (7,7)]
    e = 0.1
    a = lambda x : 0.1

    f = open('varying_q0.csv', 'w+', newline='\n')
    writer = csv.writer(f)
    writer.writerow(['Q0', 'Final Average', 'Std Dev Q1', 'Std Dev Q2', 'Avg Q1', 'Avg Q2'])

    i = 0
    stats = {} 
    lines = []
    for q in Q0:
        stats[str(q)] = EGreedy(bandit, e, a, q0=q).run()
        name, = plt.plot(stats[str(q)].pop('reward_per_iter'), label='Q0 = ' + str(q))
        lines.append(name)

        writer.writerow([
                str(q), 
                stats[str(q)]['final_amt'], 
                stats[str(q)]['std_qs'][0], stats[str(q)]['std_qs'][1], 
                stats[str(q)]['avg_qs'][0], stats[str(q)]['avg_qs'][1] 
            ])

    print(json.dumps(stats, indent=4))

    plt.legend(handles=lines)
    plt.title('Different Q0 Values')
    plt.ylim((0,15))
    plt.show()

    f.close()

parta()
partb()