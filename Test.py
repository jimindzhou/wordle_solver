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

# Print a row of squares
for color in ['red', 'blue', 'green']:
    print_colored_square(color)
print()  #


def word_comparison(comp,test):

    """
    compares two words (the assumed solution and a test word) and outputs the wordle response, a list of colors (grey, yellow, green), that will determine the grouping for the test word.

    takes in:

    comp - the assumed solution
    test - all the other words in the list

    returns:

    colors - list of colors corresponding to the wordle response

    """

    common_letters = set(comp) & set(test)
    comp_idx = []
    test_idx = []

    for letter in common_letters:
        comp_idx.extend([n for n, char in enumerate(comp) if char == letter])
        test_idx.extend([n for n, char in enumerate(test) if char == letter])

    colors = []
    for x in range(0,len(test)):

        if x in test_idx:
            if x in comp_idx:
                if int(test_idx.index(x)) == int(comp_idx.index(x)):
                    colors.append('green')
                else:
                    colors.append('yellow')
            else:
                colors.append('yellow')
        else:
       
            colors.append('grey')
    
    return colors



def get_groups(word_list,a_sol):
    groupings = []
    grp_list = []
    grps = pd.DataFrame()
    grp_no = 0 
    wrd_lst = []
    grp_dict = {}
    for word in word_list:
        
        group = word_comparison(a_sol,word) # get the grouping between the word and assumed solution

        group_key = tuple(group)  # Convert to tuple to make it hashable
        if group_key not in grp_dict:
            grp_dict[group_key] = []
        grp_dict[group_key].append(word)
        
    return grp_dict



#%% import the wordlist

with open('valid-wordle-words.txt','r') as file:
    wordlist = [line.strip() for line in file]


#%% test on toy subset
random.seed(12)
rand_idx = [random.randint(1,len(wordlist)) for _ in range(10)]

sublist = []
for idx in rand_idx:
    sublist.append(wordlist[idx])


for i in range(len(sublist)):

    #assign word to be the assumed solution
    assumed_sol = sublist[i]
    groupings = get_groups(sublist,assumed_sol)
    #print(groupings)
    for key,value in groupings.items():
        print(f"Key: {key}, Value: {value}")

