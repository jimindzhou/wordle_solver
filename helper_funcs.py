#%% import necessary functions and libraries

import numpy as np
import scipy as sp 
import random
from collections import defaultdict, Counter 
from typing import List, Dict, Any, Tuple
import pandas as pd

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
# no_groups1, max_length1 = get_dict_stats(grp_dict1)
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
               and the set of words that would be contained in the new state associated with the given observation (values)
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
            group_key = tuple(group)  # Convert to tuple to make it hashable
            if group_key not in grp_dict:
                grp_dict[group_key] = []
            grp_dict[group_key].append(sol)
        
        tran_dict = {}
        for key in grp_dict.keys():
            tran_dict[key] = []
            tran_dict[key].append(len(grp_dict[key]) / nword)

        no_groups, max_length = get_dict_stats(grp_dict)

        dict_list.append(grp_dict)
        stat_list.append([no_groups, max_length])
        tran_list.append(tran_dict)
        prob_list.append(no_groups / max_length )

    total_prob = sum(prob_list)
    prob_list = [x / total_prob for x in prob_list]

    return dict_list, stat_list, tran_list, prob_list

# # Unit tests for dictionary list creation
# word_list3 = ["float", "bloat", "gloat", "sloan", "vegan"]
# dict_list, stat_list, tran_list, prob_list = create_dict_list(word_list3)