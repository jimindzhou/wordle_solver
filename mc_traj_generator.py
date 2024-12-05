from helper_funcs import create_dict_list, compute_score
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Any  # New import
import numpy as np

def generate_mc_trajectory(word_list: List[str],
                           depth: int,
                           max_depth: int = 6) -> Tuple[List[List[str]], str]:
    """
    Generates a single Monte Carlo trajectory starting from the current state.
    
    Args:
        word_list: Current list of valid words
        depth: Current depth in the game
        max_depth: Maximum allowed depth (default 6 for Wordle)
    
    Returns:
        Tuple of (trajectory: List[List[str]], first_action: str)
        trajectory contains the sequence of states after the first action
        first_action is the initial action taken from the current state
    """
    trajectory = []  # Start with empty state as the current state is not included
    current_words = word_list
    current_depth = depth
    
    while current_depth < max_depth and len(current_words) > 0:
        dict_list, stat_list, tran_list, prob_list = create_dict_list(current_words)
        
        if not prob_list:  # No valid moves left
            break
            
        # Choose action based on probability distribution
        action = random.choices(current_words, weights=prob_list, k=1)[0]
        if current_depth == depth:
            # Save the first action for scoring
            first_action = action
            trajectory = []

        action_idx = current_words.index(action) # Find the transition probabilities for this action
        possible_transitions = tran_list[action_idx]
        
        # Choose a transition based on probabilities
        patterns = list(possible_transitions.keys())
        probs = [possible_transitions[p][0] for p in patterns]
        chosen_pattern = random.choices(patterns, weights=probs, k=1)[0]
        
        # Get next state from dict_list
        next_words = dict_list[action_idx][chosen_pattern]
        
        # terminal state
        if chosen_pattern == ('green', 'green', 'green', 'green', 'green') or not next_words:
            trajectory.append(next_words)
            break
            
        trajectory.append(next_words)
        current_words = next_words
        current_depth += 1
    
    return trajectory, first_action


def run_monte_carlo(word_list: List[str], n_simulations: int, depth: int) -> str:
    """
    Runs multiple Monte Carlo simulations to determine the best next guess.
    
    Args:
        word_list: Current list of valid words
        n_simulations: Number of Monte Carlo simulations to run
        depth: Current depth in the game
    
    Returns:
        Best guess based on Monte Carlo simulations
    """
    action_scores = defaultdict(list)
    
    # Run simulations
    for _ in range(n_simulations):
        trajectory, first_action = generate_mc_trajectory(word_list, depth)
        score = compute_score(trajectory)
        action_scores[first_action].append(score)
    
    # Average scores for each action
    avg_scores = {
        action: np.mean(scores) for action, scores in action_scores.items()
    }
    
    # Return action with highest average score
    if not avg_scores:
        return random.choice(word_list)
    return max(avg_scores.items(), key=lambda x: x[1])[0]

