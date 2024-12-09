#%% import necessary functions and libraries
import numpy as np
import scipy as sp 
import random
from collections import defaultdict, Counter 
from typing import List, Dict, Any, Tuple
import pandas as pd
import multiprocessing
import concurrent.futures 
import time

def print_colored_square(color):
    colors = {
        'red': '\033[41m',    # red background
        'blue': '\033[44m',   # blue background
        'green': '\033[42m',  # green background
        'grey': '\033[100m',  # grey background
        'yellow': '\033[43m', # yellow background
        'reset': '\033[0m'    # reset to default
    }
    print(f"{colors[color]}  {colors['reset']}", end='')

# # Print a row of squares
# for color in ['red', 'blue', 'green', 'yellow', 'grey']:
#     print_colored_square(color)
# print() 


# def word_comparison(comp,test):

#     """
#     compares two words (the assumed solution and a test word) and outputs the wordle response, a list of colors (grey, yellow, green), that will determine the grouping for the test word.

#     takes in:

#     comp - the assumed solution
#     test - all the other words in the list

#     returns:

#     colors - list of colors corresponding to the wordle response

#     """
#     if len(test) != len(comp):
#         raise Exception('Cannot compare words of different dimensions') 

#     common_letters = set(comp) & set(test)
#     comp_idx = []
#     test_idx = []

#     for letter in common_letters:
#         comp_idx.extend([n for n, char in enumerate(comp) if char == letter])
#         test_idx.extend([n for n, char in enumerate(test) if char == letter])

#     colors = []
#     for x in range(0,len(test)):

#         if x in test_idx:
#             if x in comp_idx:
#                 if test[x] == comp[x]:
#                     colors.append('green')
#                 else:
#                     colors.append('yellow')
#             else:
#                 colors.append('yellow')
#         else:       
#             colors.append('grey')
    
#     return colors

def word_comparison(comp,test):
    """
    compares two words (the assumed solution and a test word) and outputs the wordle response, 
    a list of colors (grey, yellow, green) that will determine the grouping for the test word.

    inputs:
    comp - the assumed solution
    test - all the other words in the list

    outputs:
    colors - list of colors corresponding to the wordle response

    """
    if len(test) != len(comp):
        raise Exception('Cannot compare words of different dimensions')
    
    colors = []    
    skipComp = []
    for i in range(0, len(test)):
        if test[i] == comp[i]:  
            color = 'green'
            skipComp.append(i) 

        else:     
            color = 'grey'
            for j in range(0, len(comp)):
                if j in skipComp:
                    continue
                if test[i] == comp[j]:
                    if test[j] != comp[j]:
                        color = 'yellow'
                        skipComp.append(j)
                        break                    

        colors.append(color)
    
    return colors

# # Unit tests for comparison function
# colors0 = word_comparison("funny","innie") #GYEGG  
# colors1 = word_comparison("white","night") #GYGYY
# colors2 = word_comparison("plate","blade") #GEEGE
# colors3 = word_comparison("guile","genie") #EGGYE
# colors4 = word_comparison("payee","blend") #GGYGG
# colors5 = word_comparison("missy","spams") #YGGYY
# colors6 = word_comparison("canal","bloat") #GYGEG
# colors7 = word_comparison("angel","angle") #EEEYY
# colors8 = word_comparison("teeth","tooth") #EGGEE
# colors9 = word_comparison("roomy","bossy") #GEGGE
# colors10 = word_comparison("aaecb","ebebc") #GYEGY
# colors11 = word_comparison("aeeaa","aaeec") #EYEYG
# colors12 = word_comparison("aaeeb","ebebe") #YYEGG


def get_groups(word_list,a_sol):

    """
    takes a list of words and compares it with an assumed solution word. It then groups the words based on the unique coloring response and outputs as a dictionary. Each key is the unique color grouping, and the values are the list of words corresponding to that color grouping.

    inputs:

    word_list:  list of words to be grouped
    a_sol: assumed solution (assigned outside the function)

    outputs:

    grp_dict: dictionary with groupings and associated words. It also contains the assumed solution as an entry.
    
    """

    grp_dict = {}
    #grp_dict['assumed solution'] = a_sol
    for word in word_list:
        
        group = word_comparison(a_sol,word) # get the grouping between the word and assumed solution

        group_key = tuple(group)  # Convert to tuple to make it hashable
        if group_key not in grp_dict:
            grp_dict[group_key] = []
        grp_dict[group_key].append(word)
        
    return grp_dict

# # Unit tests for grouping function
# word_list1 = ["funny", "innie", "plate", "blade", "guile", "genie", "payee", "blend", "canal", "bloat"]
# grp_dict1 = get_groups(word_list1,"blend")
# grp_dict2 = get_groups(word_list1,"funny")
# word_list2 = ["float", "bloat", "gloat", "sloan", "vegan"]
# grp_dict3 = get_groups(word_list2, "bloat")
# grp_dict4 = get_groups(word_list2, "vegan")

def get_dict_stats(group_dict):

    #getting the largest group size
    if not group_dict:
        max_length = 0
    max_length = max(len(group) for group in group_dict.values())  
    
    no_groups = len(group_dict)
    
    return no_groups, max_length

# # Unit tests for dictionary stats function
#no_groups1, max_length1 = get_dict_stats(grp_dict1)
# no_groups2, max_length2 = get_dict_stats(grp_dict2)
# no_groups3, max_length3 = get_dict_stats(grp_dict3)
# no_groups4, max_length4 = get_dict_stats(grp_dict4)

def create_dict_list(word_list):
    """
    takes the current state (list of remaining valid words) and creates lists with information necessary for
    generating the next state

    inputs:

    word_list:  list of valid words remaining    

    outputs:

    dict_list: list of dictionaries where each entry is associated with a word from the word_list being chosen as the next guess.
               the individual dictionaries contain all possible observations from the given guess (keys representing the color patern)
               and the set of words that would be contained in the new state associated with the given observation (values).
               The entry associated with the current guess being the solution (patern of 5 green tiles) will be empty, signifying that
               the game ended
    stat_list: list containing the number of dictionaries and the size of the largest dictionary associated with each potential guess
    tran_list: list containing the probability of transitioning to each possible future state from a given guess
    prob_list: probability of choosing a word as the next guess (action), considering the heuristic: [Number of Groups / Size of Largest Group]
    
    """
    dict_list = []
    stat_list = []
    tran_list = []
    prob_list = []
    nword = len(word_list)

    for guess in word_list:
        grp_dict = {}

        for sol in word_list:
            group = word_comparison(sol, guess)
            group_key = tuple(group)
            if group_key not in grp_dict:
                grp_dict[group_key] = []
            grp_dict[group_key].append(sol)
        
        tran_dict = {}
        for key in grp_dict.keys():
            tran_dict[key] = []
            tran_dict[key].append(len(grp_dict[key]) / nword)

        no_groups, max_length = get_dict_stats(grp_dict)
        grp_dict[('green', 'green', 'green', 'green', 'green')] = []

        dict_list.append(grp_dict)
        stat_list.append([no_groups, max_length])
        tran_list.append(tran_dict)
        prob_list.append(no_groups / max_length )

    total_prob = sum(prob_list)
    prob_list = [x / total_prob for x in prob_list]

    return dict_list, stat_list, tran_list, prob_list

def create_dict_p(input):
    word_list = input[0]
    guess = input[1]

    grp_dict = {}
    for sol in word_list:
        group = word_comparison(sol, guess)
        group_key = tuple(group)
        if group_key not in grp_dict:
            grp_dict[group_key] = []
        grp_dict[group_key].append(sol)
    
    tran_dict = {}
    nword = len(word_list)
    for key in grp_dict.keys():
        tran_dict[key] = []
        tran_dict[key].append(len(grp_dict[key]) / nword)

    no_groups, max_length = get_dict_stats(grp_dict)
    grp_dict[('green', 'green', 'green', 'green', 'green')] = []

    return grp_dict, [no_groups, max_length], tran_dict, no_groups / max_length

def create_dict_list_p(word_list):    
    dict_list = []
    stat_list = []
    tran_list = []
    prob_list = []

    inputs = []
    for guess in word_list:
        inputs.append((word_list, guess))

    with concurrent.futures.ProcessPoolExecutor(max_workers = 8) as executor:
        results = list(executor.map(create_dict_p, inputs))
        # # Display the number of active processes              
        # while any(future.running() for future in futures):                    
        #     active_processes = multiprocessing.active_children()
        #     print(f"Active processes: {len(active_processes)}")
        #     time.sleep(2)  # Check every 2s
        # results = [future.result() for future in futures]

    for result in results:
        dict_list.append(result[0])
        stat_list.append(result[1])
        tran_list.append(result[2])
        prob_list.append(result[3])

    total_prob = sum(prob_list)
    prob_list = [x / total_prob for x in prob_list]

    return dict_list, stat_list, tran_list, prob_list

# # Unit tests for dictionary list creation
# word_list3 = ["float", "bloat", "gloat", "sloan", "vegan"]
# dict_list, stat_list, tran_list, prob_list = create_dict_list(word_list3)

def state_transition(word_list, solution, guess):
    """
    takes the current guess and compares it against the actual solution (for the game simulator) and then trims down the
    list of remaining possible solutions (next state), according to the color patern observation given by the game

    inputs:

    word_list:  list of valid words remaining before the guess
    solution: mystery word which is the solution to the game
    guess: current guess (action) taken    

    outputs:

    new_word_list: updated list of valid words after taking into account the color patern information given by the game
    obs_group: list of color observations for the given guess-solution pair
    
    """
    
    new_word_list = []
    obs_group = word_comparison(solution, guess)
    if guess != solution:     
        for word in word_list:         
            group = word_comparison(word, guess)
            if group == obs_group:
                new_word_list.append(word)    

    return new_word_list, obs_group  

# # Unit test for the state transition function
# word_list4 = ["float", "bloat", "gloat", "sloan", "vegan"]
# solution4 = "bloat"
# guess4 = "gloat"
# new_word_list4, obs_group4 = state_transition(word_list4, solution4, guess4)
# solution5 = "float"
# guess5 = "sloan"
# new_word_list5, obs_group5 = state_transition(word_list4, solution5, guess5)
# solution6 = "vegan"
# guess6 = "bloat"
# new_word_list6, obs_group6 = state_transition(word_list4, solution6, guess6)
# word_list5 = ["float", "bloat"]
# solution7 = "bloat"
# guess7 = "float"
# new_word_list7, obs_group7 = state_transition(word_list5, solution7, guess7)
# solution8 = "bloat"
# guess8 = "bloat"
# new_word_list8, obs_group8 = state_transition(word_list5, solution8, guess8)

def compute_score(trajectory):
    """
    takes a trajectory obtained by a Monte Carlo simulation and evaluates its score. A trajectory consists of a list
    states (word list) traversed by a sequence of actions. The score returned will be associated with a state-action
    pair, corresponding to the state from which the trajectory was rolled out and the first action taken in that state.
    The total score of that state-action pair R(s,a) is computed as the sum of the reward obtained at each step of the
    trajectory. For the states before the terminal state (which will be empty if we have found the solution, or non-empty
    if we exhausted the available guesses without finding a solution), we penalize the size of the state and the number of
    guesses it has taken us to reach that state. For the terminal state, we either apply a penalty for failing to find the 
    solution or a reward for doing so. The reward computation is shown below, where alpha and beta are hyperparameters.
    R(Si, Ai) = -[(|s|- 1) + alpha*(i-1)] + beta(if |St| == 1) - beta(if |St| > 1)

    Obs.: a trajectory always has n+1 states, where n is the number of guesses actually taken (max n = 6, max states = 7).
          also, we do not need to evaluate the score associated with the first state, since it will be identical for all
          trajectories taken from that state.

    inputs:

    trajectory:  list of the sequence of states generated by a single Monte Carlo simulation

    outputs:

    score: utility of the action taken at the initial state that lead to the given trajectory rollout
    
    """
    alpha = 1000
    beta = 10000
    nstates = len(trajectory)
    score = 0.0
    for guess, state in enumerate(trajectory):
        if guess == 0:
            continue
        if guess == (nstates - 1):
            if len(state) == 0:
                score += beta
            else:
                score -= beta
        else:
            score -= ((len(state) - 1) + alpha * guess)

    return score

# # Unit test for score function
# trajectory1 = [["a", "b", "c", "d"], ["a", "b", "d"], ["d"], []]
# score1 = compute_score(trajectory1)
# trajectory2 = [["a", "b", "c", "d"], ["a", "b", "d"], ["b", "d"], ["d"]]
# score2 = compute_score(trajectory2)