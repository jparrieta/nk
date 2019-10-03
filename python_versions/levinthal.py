#!/usr/bin/env python
# coding: utf-8

# # Tutorial on Learning under Complexity
# 
# In this tutorial, you will be introduced to a simple model that replicates the main finding from the paper by Dan Levinthal, published in 1997 in Management Science.  
#   
# This tutorial provides a barebones description of the model. If you want to explore a more flexible version or explore how different agents or bandit distributions would affect Dan's results please employ Maciej Workiewicz's code (https://www.maciejworkiewicz.com/coding). There you will find code on how to replicate also other seminal papers on NK landscapes.  
#   
# **Reference:** Levinthal, D. A. (1997). Adaptation on rugged landscapes. Management science, 43(7), 934-950.

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# # Basic building blocks
# Below I introduce the code for searching in NK landscape, equivalent to the one from Levithnal (1997). It builds on two main objects, the landscape and the agent. Additionally, we need to create a function to run the simulation and a couple of miscellaneous functions to make small transformations to the code. We present each in a separate section.  
#   
# ## 1. Landscape
# An NK landscape outputs a payoff value for an input policy. It is is equivalent to getting an elevation value after providing the longitude and latitude in a map. That is the reason why it is called a landscape. However, the inputs of an NK landscape are binary thus the analogy does not go to far.  
# Overall, the landscape receives a policy and outputs a payoff for this policy. The policy consists in N binary values stored as a string of 0s and 1s (e.g. '101'). The payoff depends on the value of K of the NK model. 
# 
# K represents the amount of interdependencies that are linked to the performance of each of the N variables. In an environment where K=0, then the performance of the policy value depends only on its own value, if 0 then x0 if 1 then x1. It does not depend on the other N-1 elements. If K > 0 then the payoff for this policy element depends on the value of some of the N-1 elements of the policy. If K = N-1 then the payoff of this policy element depends on the value of all the other N-1 policy element. The higher the K the higher the complexity of the interrelationships when calculating the performance of each policy.    
# ### 1.1 Create dependencies
# The first step to create a landscape is creating the dependency matrix. Here one takes N and K and fills an N by N matrix with 1s in the diagonal and N*K 1s in the off diagonal. On each row there are on average K 1s in the off-diagonal, and always a 1 in the diagonal. 
# 
# ### 1.2 Fitness contribution
# Having the interdependency matrix. We can create the fitness contributions. These basically determine the payoff each policy element receives. There are in total N different sets of fitness contribution functions. Each of the N sets has 2^k+1 different values for all the combinations of the policies that matter for the policy element at hand.  
# The fitness contribution is created by filling a list with N dictionaries. Each dictionary has as key value a combination of policies that affect the payoff of the policy element (e.g. '010') and as value a draw from an uniform distribution.  
# 
# ### 1.3 Payoff
# The fitness contributions are used to calculate the total payoff of each policy. For this we need a way of taking a policy and determining the interactions that each policy element makes in order to estimate its payoff contribution.   
# This requires several steps. Firts we need to obtain the keys for the fitness contributions. that is we take a policy (e.g. '101']) and determine for every policy element the key value for the fitness contribution.  
#   
# Let's imagine that the interdependency matrix is:  
#   
# |1 0 0|  
# |0 1 1|  
# |1 1 1|  
# 
# From Levinthal (1997), we would see that the first key value is '1', the second '01', and the third '101'. In the code below, the functions transfor_matrix and transform_row are in charge of making the translation from a policy to the key values. 
#   
# After having the key values, we can calculate the performance of each policy element by addressing the fitness contribution list of each element, and then averaging them. The total payoff for one fitness contribution is this average. 
# 
# ### 1.4 Calculate landscape
# After having a function that calculate the performance of one policy we can calculate the full landscape by running a for-loop over each policy value. 
# 
# ### 1.6 Landscape initiation
# To initiate a landscape one just needs to give the N and K value. The landscape is created with the use of the reset function.
# 
# ### 1.5 Reset
# The reset function creates a landscape by first building the dependency matrix, then the fitness contributions, and lastly the full-landscape from the prior two. For doing this it just requires the N and K values given at the initiation of the class. 
# 
# ### 1.6 Summary
# The summary function outputs the maximum, minimum, and number of peaks in the landscape. 

# In[1]:


import numpy as np
import random # random.sample is 10x faster than np.random.choice!

class landscape:
    def __init__(self, n, k, land_style):
        self.n = n
        self.k = k
        self.land_style = land_style
        self.reset()
    def calc_landscape(self):
        land = {}
        for i in range(2**self.n):
            pol = int2pol(i,self.n)
            land[pol] = self.payoff(pol)
        self.lands = land
    def create_dependencies(self):
        self.dep_mat = np.zeros((self.n,self.n)).astype(int)
        inter_row = [1]*self.k+[0]*(self.n-self.k-1) # number of off-diagonals 1s and 0s
        if self.land_style == "Rand_mat": 
            inter = random.sample(inter_row*self.n, self.n*(self.n-1))
        for i in range(self.n):
            if self.land_style == "Rand_row": inter = random.sample(inter_row, self.n-1)
            elif self.land_style == "Levinthal": inter = inter_row # The original order is the one from Levinthal (1997)
            range_row = list(range(self.n))
            range_row = range_row[i:]+range_row[:i]
            for j in range_row:
                if i != j: 
                    self.dep_mat[i][j] = inter[0]
                    inter = inter[1:]
                else: self.dep_mat[i][i] = 1
    def fitness_contribution(self):
        self.fit_con = []
        for i in range(self.n):
            if  self.land_style != "Rand_mat": q = self.k+1
            elif self.land_style == "Rand_mat": q = sum(self.dep_mat[i])
            self.fit_con.append({int2pol(j,q):random.random() for j in range(2**q)})
        return(self.fit_con)
    def payoff(self, policy):
        keys = transform_matrix(policy, self.dep_mat)
        pay = np.sum([self.fit_con[i][keys[i]]/self.n for i in range(self.n)])
        return(pay)
    def reset(self):
        self.create_dependencies()
        self.fitness_contribution()
        self.calc_landscape()
    def reset_one(self, which_pol):
        q = sum(self.dep_mat[which_pol])
        self.fit_con[which_pol] = {int2pol(i,q): random.random() for i in range(2**q)}
        self.calc_landscape()        
    def summary(self):
        num_peaks = 0
        for current_row in self.lands.keys():
            counter = 1
            for neighbor in find_neighbors(policy = current_row, randomizer = False):
                if self.lands[current_row] < self.lands[neighbor]: 
                    counter = 0
                    break
            num_peaks += counter
        return([max(self.lands.values()), min(self.lands.values()), num_peaks])


# ## 2. Agents
# 
# ### 2.1 Initiation
# The agent is created by giving the probability of making a long jump instead of searching its neighbors, the position in the landscape the agent starts and the level of noise in its evaluation of distant positions in the landscape.  
# 
# ### 2.2 Search
# The search function receives a landscape and the number of periods it has to search the landscape. On every periods, it has the opportunity of making a long jump and staying in the position if it has a higher payoff or searching the neighboring positions.  
# The agent decides to move to a new position only if the payoff is higher than the current payoff. Any change of position is logged into a short log, and even if there is no change, the current position is logged into a longer log. If the global maximum is found, then the agent stores a 1 and the period when the global maximum was reached.  

# In[2]:


class agent:
    def __init__(self, long_jump, start_pos, noise):
        self.long_jump = long_jump
        self.start_pos = start_pos
        self.noise = noise
    def search(self, lands, num_periods):
        # Where to start?
        if self.start_pos == "Minimum": current_row = min(lands, key=lands.get)
        elif self.start_pos == "Random": current_row = random.sample(lands.keys(),1)[0]
        # Initialize logs and find maximum
        policy_short = [current_row]
        policy_long = [current_row]
        payoff_short = [lands[current_row]]
        payoff_long = [lands[current_row]]
        global_max = max(lands, key=lands.get)
        # Start search
        for j in range(num_periods):
            # Local search or Jump?
            walk_or_jump = np.random.choice(["Walk", "Jump"], p = [1-long_jump, long_jump])
            if walk_or_jump == "Walk":
                for proposed_row in find_neighbors(policy = current_row, randomizer = True):
                    ruido = (random.random()-0.5)*self.noise # just 1 difference
                    if lands[proposed_row] + ruido > lands[current_row]: break
            elif walk_or_jump == "Jump": 
                proposed_row = random.sample(list(lands.keys()),1)[0] #could be improved not past or current
                distance = np.sum([1*(proposed_row[i] != current_row[i]) for i in range(len(proposed_row))])
                ruido = (random.random()-0.5)*self.noise*distance
            # Store new position if higher
            if lands[proposed_row] + ruido > lands[current_row]: # same ruido as before
                current_row = proposed_row
                policy_short.append(current_row)
                payoff_short.append(lands[current_row])
            policy_long.append(current_row)
            payoff_long.append(lands[current_row])
            # Check if search is finished
            if current_row == global_max:
                if j < num_periods-1: # somehow the & did not work
                    for k in range(num_periods-j-1): 
                        policy_long.append(current_row)
                        payoff_long.append(lands[current_row])
                    break # stop the main for-loop after global maximum is found
        # Store data
        policy_short.append(global_max) # add global max for comparison later on
        policy_long.append(global_max)
        payoff_short.append(lands[global_max])
        payoff_long.append(lands[global_max])
        reached_max = 1*(current_row == global_max)
        num_steps = len(policy_short)-1
        num_trials = j+1
        return([reached_max, num_steps, num_trials, [policy_short, payoff_short], [policy_long, payoff_long]])


# ## 3 Miscellaneous functions  
# We need four functions that perform simple transformations to the data of the agents or the landscape.  
#   
# ### 3.1 Int2Pol
# This function translates an integer value to a string of 0s and 1s. This string is made so that it has length N.
# For example 5 is translated in to '101' in the case of N=3 or '0101' in case N = 4.  

# In[3]:


def int2pol(pol_int, n):
    pol = bin(pol_int)[2:] # removes the '0b'
    if len(pol) < n: pol = '0'*(n-len(pol)) + pol
    return(pol)


# ### 3.2 Transform matrix
# Handles the transformation of a policy into the key values for estimating its payoff. For this it uses transform_row for every policy element. The input is an interdependency matrix and a policy and the output are the N keys for each policy element.  

# In[4]:


def transform_matrix(policy, dep_mat):
    int_mat = [transform_row(policy, dep_mat[i]) for i in range(len(dep_mat))]
    return(int_mat)


# ### 3.3 Transform row
# Uses the policy and interacts it with a row of the dependency matrix. The length of the output depends on the number of 1s in the row of the interdependency matrix. For example if the policy is '101' and the row is [0,1,1] then the output is '01'. If the row was [1,0,0] then the output would be '1'.  
# This transformation gives always the same value and thus can be used with the fitness contribution to estimate the payoff for each policy element.  

# In[5]:


def transform_row(policy, dep_row):
    interact_row = [policy[i] for i in range(len(policy)) if dep_row[i] == 1]
    trans_pol = ''
    for pol_i in interact_row: trans_pol += pol_i
    return(trans_pol)


# ### 3.4 Find neighbors
# The main form of movement for the agent is local search. This implies moving to each neighboring position and staying in the first that gives a higher payoff. In order to do this the agent needs to find its neighbors. This can be done in several ways, here what we do is to morph the current position by flipping one policy element and storing it as a neighbor. We do this for every policy element to make a list of neighbors. We randomize the order of neighbors to avoid bias and use this for local search.  

# In[6]:


def find_neighbors(policy, randomizer):
    neighbors = []
    n = len(policy)
    if randomizer: random_order = random.sample(range(n), n)
    else: random_order = range(n)
    for i in random_order:
        neighbor = list(policy)
        if policy[i] == '1': neighbor[i] = '0'
        else: neighbor[i] = '1'
        neighbors.append(''.join(neighbor))
    return(neighbors)


# ## 4. Run simulation
# Having the agent and the landscape it is turn to define the simulation. For this we define a class that holds the different types of simulations to run. The class takes a landscape when initialized and will reset it when needed.
#   
# ### 4.1 Describe
# This function creates a given number of landscapes and describes the maximum, minimum, and number of peaks they have.  
#   
# ### 4.2 Search
# This function has multiple agents serching the landscape and repeats this process with multiple landscapes. It takes four attributes, the number of landscapes to simulate, the number of agents that search each landscape and the number of periods each agent has to learn. The last parameter it takes is the Agent class itself. The simulation is ran by running a for-loop over every landscape and calling the search lanscape function for every landscape. After the search landscape is finished the data is stored and a new landscape is generated. 
#   
# ### 4.3 Search landscape
# This function takes three attributes, an Agent class, the number of times the agent needs to search the landscape and the number of periods the agent has to search. It stores five variables. The number of time the global maximum is reached, the number of steps to reach the maximum, the number of trials before reaching the maximum, the average payoff on every period and the number of differnt configurations on every period.  

# In[7]:


class run_simulation:
    def __init__(self, Environment):
        self.Env = Environment
    def describe(self, num_reps):
        all_max = []
        all_min = []
        all_num_peaks = []
        for i in range(num_reps): 
            self.Env.reset()
            max_val, min_val, peaks = self.Env.summary()
            all_max.append(max_val)
            all_min.append(min_val)
            all_num_peaks.append(peaks)
        return([all_max, all_min, all_num_peaks])
    def search(self, num_lands, num_agents, num_periods, Alice):
        all_reached_max = []
        all_num_steps = []
        all_num_trials = []
        all_payoffs = np.zeros(num_periods+2)
        all_choices = np.zeros(num_periods+2)
        for i in range(num_lands):
            self.Env.reset()
            data_landscape = self.search_landscape(num_agents, num_periods, Alice)
            all_reached_max.append(data_landscape[0])
            all_num_steps.append(data_landscape[1])
            all_num_trials.append(data_landscape[2])
            all_payoffs = np.add(all_payoffs, data_landscape[3])
            all_choices = np.add(all_choices, data_landscape[4])
        return([all_reached_max, all_num_steps, all_num_trials, all_payoffs, all_choices])
    def search_landscape(self, num_agents, num_periods, Alice):
        reached_max = 0
        num_steps = 0
        num_trials = 0
        payoffs = np.zeros(num_periods+2)
        choices = []
        policies = []
        for i in range(num_agents):
            search_data = Alice.search(lands = self.Env.lands, num_periods = num_periods)
            reached_max += search_data[0]
            num_steps += search_data[1]
            num_trials += search_data[2]
            payoffs = np.add(payoffs, search_data[4][1])
            policies.append(search_data[4][0]) 
        for j in range(num_periods + 2):
            diff_forms = set([choice_list[j] for choice_list in policies]) # store all different forms of one period
            choices.append(len(diff_forms)) # count number of different forms
        return([reached_max, num_steps, num_trials, payoffs, choices])


# # Levinthal (1997)
# Below we run the simulation once and show the steps followed by one agent while searching the landscape. 
#   
# ## 1. Initialize values

# In[8]:


# Simulation
num_periods = 50
num_agents = 100
num_lands = 100
num_reps = 500
# Agent
long_jump = 0.1
noise = 0.0 # no noise 0.0, low noise 0.01, high noise 0.025
start_pos = "Random" # "Minimum", "Random"
# Landscape
n = 10
k = 1
land_style = "Rand_row" # "Levinthal", "Rand_mat", "Rand_row"


# ## 2. Initialize agent and landscape

# In[9]:


Environment = landscape(n,k, land_style)    
Alice = agent(long_jump = long_jump, start_pos = start_pos, noise = noise)
Simulation = run_simulation(Environment)


# ## 3. Run simple search
# Below we show one search of the environment and the short output of the search

# In[10]:


import pandas as pd 
reached_max, n_step, n_trial, output_short, output_long = Alice.search(Environment.lands, num_periods)
pd.DataFrame({'policy': output_short[0], 'payoff': output_short[1]})


# ## 4. Run simulation
# Additionally we run a simulation 1000 times to show the average form in which the agents explore the landscape. We print the percentage of times it reaches the global maximum, the number of movements done before the total number of periods were reached, the number of periods needed to find the maximum, and the time required to make the simulation. On average 50% of the agents reached the global maximum and they reached it around period 36. 

# In[11]:


import time
start_time = time.time()
reached_max, num_steps, num_trials, payoffs, choices = Simulation.search(num_lands, num_agents, num_periods, Alice)
print("% of time global max is reached: " + str(100*sum(reached_max)/(num_agents*num_lands)))
print("# of steps from start: " + str(sum(num_steps)/(num_agents*num_lands)))
print("# of periods to find global max: " + str(sum(num_trials)/(num_agents*num_lands)))
print("Computation time: " + str(round(time.time()-start_time,2)) + " s")


# ### 4.1 Number of organizational forms
# We can now plot the number of different solutions the agents go to on every period.

# In[12]:


import matplotlib.pyplot as plt
plt.scatter(range(num_periods+2), 100*choices[:num_periods+2]/(num_lands*num_agents))


# ### 4.2 Performance over time
# We also plot the growth in payoff as the agents search the landscape. The last value is the average of the global maxima. Clearly, the search process is still distante to reaching the highest peak on every search ocassion. 

# In[13]:


plt.scatter(range(num_periods+2), payoffs[:num_periods+2]/(num_lands*num_agents))


# ## 5. Landscape descriptives  
# 
# Finally we create 500 landscapes and see their characteristics. Although making 1000 landscapes takes around 3 seconds, estimating their characteristics takes one order of magnitude longer. This is not a crucial step so I have not optimized it, yet.

# In[14]:


start_time = time.time()
all_max, all_min, all_num_peaks = Simulation.describe(num_reps)
print("Computation time: "+ str(round(time.time()-start_time, 2)) + " s")


# In[15]:


plt.hist(all_min)


# In[16]:


plt.hist(all_max)


# In[17]:


plt.hist(all_num_peaks)


# Levinthal (1997) includes more analyzes in specific on competition dynamics. This tutorial does not include them. Competition is a more general process. The main theoretical contribution of Levinthal (1997) was the introduction of NK models and that is what this tutorial presents. 
# 
# **Note:** The code below produced the table of contents.

# In[18]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")

