
from agent_class import HW2Agent
import build_environment as env 
import matplotlib.pyplot as plt

# Part 1: Policy iteration
# Improvements: 13
# Evaluations:  748
def part1a():
    Agent = HW2Agent(env.build_map(), env.STATEMAP, 0.02, 0.95, 0.01)
    Agent.policy_iteration()
    
    print("Agent A:")
    print("\tNum improvements: " + str(Agent.num_improvements))
    print("\tNum evaluations:  " + str(Agent.num_evals))
    print(Agent.values)
    
    x,y = Agent.generate_best_path()
    plt.imshow(env.build_map())
    plt.scatter(y,x)
    plt.show()

    plt.imshow(Agent.values)
    plt.show()


# Improvements: 30
# Evaluations:  432
def part1b():
    Agent = HW2Agent(env.build_map(), env.STATEMAP, 0.5, 0.95, 0.01)
    Agent.policy_iteration()
    
    print("Agent B:")
    print("\tNum improvements: " + str(Agent.num_improvements))
    print("\tNum evaluations:  " + str(Agent.num_evals))
    print(Agent.values)
    
    x,y = Agent.generate_best_path()
    plt.imshow(env.build_map())
    plt.scatter(y,x)
    plt.show()

    plt.imshow(Agent.values)
    plt.show()


# Improvements: 38
# Evaluations:  84
def part1c():
    Agent = HW2Agent(env.build_map(), env.STATEMAP, 0.02, 0.55, 0.01)
    Agent.policy_iteration()
    
    print("Agent C:")
    print("\tNum improvements: " + str(Agent.num_improvements))
    print("\tNum evaluations:  " + str(Agent.num_evals))
    print(Agent.values)
    
    x,y = Agent.generate_best_path(cutoff=1000)
    plt.imshow(env.build_map())
    plt.scatter(y,x)
    plt.show()

    plt.imshow(Agent.values)
    plt.show()

def part1():
    part1a()
    part1b()
    part1c()


# Part 2: Value iteration
# 105 Evals
def part2a():
    Agent = HW2Agent(env.build_map(), env.STATEMAP, 0.02, 0.95, 0.01)
    Agent.value_iteration()
    
    print("Agent A:")
    print("\tNum evaluations:  " + str(Agent.num_evals))
    print(Agent.values)
    
    x,y = Agent.generate_best_path()
    plt.imshow(env.build_map())
    plt.scatter(y,x)
    plt.show()

    plt.imshow(Agent.values)
    plt.show()

# 106 Evals
def part2b():
    Agent = HW2Agent(env.build_map(), env.STATEMAP, 0.5, 0.95, 0.01)
    Agent.value_iteration()
    
    print("Agent B:")
    print("\tNum evaluations:  " + str(Agent.num_evals))
    print(Agent.values)
    
    x,y = Agent.generate_best_path()
    plt.imshow(env.build_map())
    plt.scatter(y,x)
    plt.show()

    plt.imshow(Agent.values)
    plt.show()

# 16 Evals
def part2c():
    Agent = HW2Agent(env.build_map(), env.STATEMAP, 0.02, 0.55, 0.01)
    Agent.value_iteration()
    
    print("Agent C:")
    print("\tNum evaluations:  " + str(Agent.num_evals))
    print(Agent.values)
    
    x,y = Agent.generate_best_path(stochastic=True, cutoff=1000)
    plt.imshow(env.build_map())
    plt.scatter(y,x)
    plt.show()

    plt.imshow(Agent.values)
    plt.show()

def part2():
    part2a()
    part2b()
    part2c()


part1()
part2()