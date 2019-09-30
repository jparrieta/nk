#!/usr/bin/env python
# coding: utf-8

# # Tutorial on Levinthal (1997)
# 
# In this tutorial, you will be introduced to a simple model that replicates the main finding from the paper by Dan Levinthal, published in 1997 in Management Science. 
# 
# This tutorial provides a step by step description on the logic about how to build NK landscapes.
# 
# **Reference:** Levinthal, D. A. (1997). Adaptation on rugged landscapes. Management science, 43(7), 934-950.

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>
# <script type="text/javascript" src="https://raw.github.com/kmahelona/ipython_notebook_goodies/master/ipython_notebook_toc.js">

# # NK Landscape
# In Levinthal (1997) the agent is quite simple. The environment does have some intricancies. 
# 
# ## 1. Create Dependencies
# The k interdependencies in Levinthal's are created at random. Basically, one needs a matrix where the diagonal has a 1 and the off-diagonal has k ones and n-k zeroes. A one represents an interdependency and a zero the lack of it.  
# This function includes two variables N and K and outputs a NxN interdependency matrix. 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_dependencies(n, k):
    dep_mat = np.zeros((n,n)).astype(int)
    for i in range(n):
        inter = np.random.choice([1]*k+[0]*(n-k-1), replace = False, size = n-1)
        for j in range(n):
            if i != j: 
                dep_mat[i][j] = inter[0]
                inter = inter[1:]
            else: dep_mat[i][i] = 1
    return(dep_mat)


# ### 1.1 Example: How to create a interdependency matrix
# Below you see how an interdependency matrix is built. If you run the code again, the matrix will change. 

# In[2]:


n = 3
k = 1
dep_mat = create_dependencies(n, k)

dep_mat


# ## 2. Fitness Contributions
# The second step is building the fitness contributions for each item in the interdependency matrix. Before showing the code, it is important to present the logic of it, here we do so by an example.
# 
# ### 2.1 Example
# Let's imagine that our interdependency matrix is: 
#    
# |1 0 0|  
# |0 1 1|  
# |1 1 1|  
# 
# What this means is that the fitness contributions of the first value of a policy P = [a,b,c] will depend ONLY on the value of the first policy value, namely whether a is 1 or 0. Formally the fitness contribution will have twwo values:  
# * f1[0] = z0  
# * F1[1] = z1  
# Where z0 and z1 are numbers drawn from a uniform distribution. 
#   
# The second row is more complicated as there is one interdependency. Here we will have that the second and third policy values are needed to calculate the fitness contribution. For this we need a function with four values as there are four possible combinations of a and b. The functions have the form f2[b,a].  
# * f2[00] = y0
# * f2[01] = y1
# * f2[10] = y2
# * f2[11] = y3
# 
# The final row has 3 interdependencies. Now the fitness contributions depends on the values of a, 2, and c. The fitness contribution has the for f3[c,a,b]. To depict it, we draw a truth table.   
#   
# c  a  b  |  f3  
# 0  0  0  |  x0  
# 0  0  1  |  x1  
# 0  1  0  |  x2  
# 0  1  1  |  x3  
# 1  0  0  |  x4  
# 1  0  1  |  x5  
# 1  1  0  |  x6  
# 1  1  1  |  x7  
# 
# To create the fitness contributions we need a function that takes one interdependence matrix and outputs the fitness contributions functions for each position. That is, a function that outputs the two f1, four f2, and eight f3.  
# The function later on should have a structure such that if one tells it that a = 0 and b = 1 it can give f2[b,a] = y2. For this the ideal structure is a list of dictionaries. Each fitness contribution function is a dictionary that one gives the values and it outputs the  and the dictionaries are joined together in a list. Below you find a function that does just that.
# 
# ### 2.2 Fitness contribution generator
# The function first creates an empty list. This list will be filled with the dictionaries of fitness contribution functions.   
# The next step is entering a for-loop over the N rows of the interdependency matrix. For each iteration in the for-loop, we do a list comprehension where the binary value of the counter is stored as the key of a dictionary with the value drawn from a uniform distribution. The second for-loop will go over 2^k+1 iterations, where k is the number of 1's in the off-diagonal. At the end the function output a list of with the dictionaries as its entry.  
# You see the code below and the outputs is the list of dictionaries of the dependency matrix created in the prior section. 

# In[3]:


def fitness_contribution(dep_mat):
    fit_con = []
    n = len(dep_mat)
    for i in range(n): 
        epi_row = {bin(j): np.random.random() for j in range(2**sum(dep_mat[i]))}
        fit_con.append(epi_row)
    return(fit_con)

fit_con = fitness_contribution(dep_mat)
fit_con


# ## 3. Calculate policy payoffs
# The next step is to calculate the payoff of a policy based upon the fitness contribution functions. To do this we require two things. First to calculate the payoff contribution of every value in a policy and then sum all of them together. We start with the first task. 
# 
# ### 3.1 Transform Row
# Let's continue with the example. But now we have the polict P = [0,1,1].  
# In the case of the first value we know we should get z0 as fitness contribution. In the second row, y2 and so on. To do this we need to create a function that takes the values of the policy and matches them to the values other values that are interdependent with it.  
# THe function below is given two inputs, a policy and a row of a dependency matrix.  It creates an empty list and starts to populate it. It does this by starting a for-loop for every element of the policy. If the item of that index is 1 in the interdependency row then it appends the value of policy to the list. If not, it continues to the next policy value. In this way, only the interdependent items are stored. With this, the programs has a list with items that are relevant to calculate the fitness contribution for this policy.     
# For example, in the case from before we would have as an output of the for-loop [0] in the first row, [1,1] in the second row and [1,0,1] for the last row. This output is called interact_row.
# These values however are not understandable by the dictionaries of the fitness contribution. The last rows of the function translate the list into a binary value. So that later the dictonary can be queried.   
# The process starts by starting trans_pol = 0. This value will store the key value for the dictionary. Then a for-loop starts. The for-loop has the range reversed so that the we keep the items to the left of the list being more significant. This is important because in the the next step we multiply the value of interact_row[i] with the 2^index value. By reversing the order we have that the item most to left in the list will be multiplied by 2^3 if we follow the example from before, the item next to it by 2^2 and so on. The product of the multiplications is added on every loop to the trans_pol value. At the end we have a decimal value. We transform the decimal value and the output is a key we can use in the dictionaries from the fitness contribution. For example for the first row from before we get '0b0', for the second '0b11', and for the last row, '0b101'.  
# Below you see an example of this function for the same policy we have used in this example but for the randomly generate dependency matrix from before. 

# In[4]:


def list2int(pol_list):
    pol_list.reverse()
    pol_int = np.sum([(2**j)*pol_list[j] for j in range(len(pol_list))])
    return(pol_int)

def transform_row(policy, dep_row):
    interact_row = [policy[i] for i in range(len(policy)) if dep_row[i] == 1]
    trans_pol = list2int(interact_row)
    return(bin(trans_pol))

transform_row([1,0,1], dep_mat[1])


# ### 3.2 Transform Matrix
# The transform_row function is called by a transform_matrix function whose job is to take a policy and output a set of keys for the the list of fitness contributions. To do this basically what it does is to to call the transform_row N times can fill out a list with the output of each of the calls of the function. In the example from above we would have: ['0b0',  '0b11', '0b101'] as an output. Below you see an example of the function working.  

# In[5]:


def transform_matrix(policy, dep_mat):
    int_mat = [transform_row(policy, dep_mat[i]) for i in range(len(dep_mat))]
    return(int_mat)

transform_matrix([1,0,1],dep_mat)


# ### 3.3 Payoff
# The payoff function has three inputs, the policy for which to calculate a payoff, the interdependency matrix, and the list of fitness contributions.  The first action it does it to transform the policy into keys to the fitness cotnribution dictionaries. After this is done it sums the entries of all the fitness contributions of the key values and divides the sum by the length of the policy. The last part is done to get a value between 0 and 1. Below we see and example of the code working. 

# In[6]:


def payoff(policy,dep_mat,fit_con):
    keys = transform_matrix(policy, dep_mat)
    pay = np.sum([fit_con[i][keys[i]]/len(policy) for i in range(len(policy))])
    return(pay)

payoff([1,0,1],dep_mat,fit_con)


# ## 4. Make full-landscape
# Now that we can calculate the payoff for one policy we can make the full-landscape. However, to calculate the landscape we need to make a function that takes integers and makes policies.
# 
# ### 4.1 Integer to List
# This function takes two arguments first a number in integer and then the length of the desired list. This is important because when one transforms an integer to binary in Python, the output cuts the 0s to the left. So the policy length is altered. This function prevents that.  Below you can see an example.  
# 
# **Note:** In prior functions we had to translate policies from list to integer, this function does the opposite. If we used this functions only here, then it would be a waste. However, there are benefits to having policies as lists. Especially when regarding agents searching in the landscape.

# In[7]:


def int2list(pol_int, n):
    pol_str = bin(pol_int)
    policy = [int(pol) for pol in pol_str[2:]]
    if len(policy) < n: policy = [0]*(n-len(policy))+policy
    return(policy)

int2list(5, 4)


# ### 4.2 Calculate Landscape
# Having the translating function, the function that calculates the landscape is just a for-loop that fills up a dataframe. The dataframe consists of three entries per policy: The policy in list-value, the policy in integer-value, and the payoff. The integer value of the policy is used for accessing the dataframe, the list-value for searching the landscape.   
# Below you will see an example.  

# In[8]:


def calc_landscape(dep_mat, fit_con):
    land = []
    for i in range(2**n):
        pol_list = int2list(i, n)
        land.append({'int_pol':i, 'policy': np.asarray(pol_list), 'payoff': payoff(pol_list, dep_mat, fit_con)})
    return(pd.DataFrame(land))

lands = calc_landscape(dep_mat, fit_con)
lands


# ### 4.3 Descriptives of the environment
# We can characterize the environment. Find the maximum, minimum, number of peaks.  
# For this we need a function that finds the peaks, which has to find whether a policy gives the highest performance for every neighbor. 
# #### 4.3.1 FInd neighbors
# For every policy there are N neighboring positions. These are policies that differ by one change from the current policy. The function shown here takes the starting policy and generates at random the N neighbors. It outputs the integer value of each of these N neighbors. The neighbors are given at random becuase this is useful for the search algorithm. It is not necessary here but does not affect the code. 
# Below you see an example. 

# In[9]:


def find_neighbors(policy):
    policy = (policy) #policy changed 
    neighbors = []
    random_order = np.random.choice(range(len(policy)), replace = False, size = len(policy))
    for i in random_order:
        neighbor = list(policy)
        if policy[i] == 1: neighbor[i] = 0
        else: neighbor[i] = 1
        neighbors.append(list2int(neighbor))
    return(neighbors)

find_neighbors([1,0,1])


# #### 4.3.1 Summary
# This functions gives a summary of the landscape. The maximum, minimum, and number of peaks. It is most useful for collecting statistics of different landscape configurations (i.e., different n and ks).
# 

# In[10]:


def summary(lands):
    max_global = max(lands.payoff)
    min_global = min(lands.payoff)
    num_peaks = 0
    for i in range(lands.shape[0]):
        randomized_neighbors = find_neighbors(lands.policy[i])
        if lands.loc[i, "payoff"] > np.max(lands.loc[randomized_neighbors, "payoff"]): num_peaks += 1
    return([max_global, min_global, num_peaks])

summary(lands)


# # Create Landscape from Scratch
# Below you can find the short way of creating the landscape from scratch

# In[11]:


n = 6
k = 2
dep_mat = create_dependencies(n, k)
fit_con = fitness_contribution(dep_mat)
Environment = calc_landscape(dep_mat, fit_con)    

summary(Environment)


# In[12]:


Environment


# # Search
# With the environment finished it is time to search in this rugged landscape. 
# 
#   
# **Note:** The code below builds the table of content.

# In[13]:


get_ipython().run_cell_magic('javascript', '', "$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")

