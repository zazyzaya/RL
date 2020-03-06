
from agent_class import HW2Agent
import build_environment as env 
import matplotlib.pyplot as plt

# Part 1: Policy iteration
def part1a():
    Agent = HW2Agent(env.build_map(), env.STATEMAP, 0.02, 0.95, 1e-6)
    Agent.policy_iteration()
    
    print("Agent A:")
    print("\tNum improvements: " + str(Agent.num_improvements))
    print("\tNum evaluations:  " + str(Agent.num_evals))
    print(Agent.values)
    
    x,y = Agent.generate_best_path()
    plt.imshow(env.build_map())
    plt.scatter(y,x)
    plt.show()

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

def part1c():
    Agent = HW2Agent(env.build_map(), env.STATEMAP, 0.02, 0.55, 0.01)
    Agent.policy_iteration()
    
    print("Agent A:")
    print("\tNum improvements: " + str(Agent.num_improvements))
    print("\tNum evaluations:  " + str(Agent.num_evals))
    print(Agent.values)
    
    x,y = Agent.generate_best_path()
    plt.imshow(env.build_map())
    plt.scatter(y,x)
    plt.show()

part1a()
part1b()
part1c()