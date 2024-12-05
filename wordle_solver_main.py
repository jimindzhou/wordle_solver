from mc_traj_generator import run_monte_carlo
from helper_funcs import state_transition
from typing import List, Dict, Tuple, Any

def play_wordle(solution: str,
                 word_list: List[str],
                 n_simulations: int = 100) -> Tuple[bool, int]:
    """
    Plays a complete Wordle game using Monte Carlo simulation.
    
    Args:
        solution: The target word to guess
        word_list: List of valid words
        n_simulations: Number of Monte Carlo simulations per turn
    
    Returns:
        Tuple of (success: bool, num_guesses: int)
    """
    current_words = word_list.copy()
    depth = 0
    max_depth = 6
    
    while depth < max_depth:
        # Get best guess using Monte Carlo simulation
        guess = run_monte_carlo(current_words, n_simulations, depth)
        if guess == solution:
            return True, depth + 1
            
        current_words = state_transition(current_words, solution, guess) # Update word list based on the game's response
        if not current_words:
            # If no valid words left, return False
            return False, depth + 1
        if len(current_words) == 1 and current_words[0] == solution:
            # If only the solution word is left, return True
            return True, depth + 2
            
        depth += 1
    
    return False, max_depth

def main():
    # Load your word list
    with open('valid-wordle-words-small.txt', 'r') as f:
        word_list = [word.strip() for word in f.readlines()]

    # Play a game
    solution = "aback"
    success, num_guesses = play_wordle(solution, word_list)
    print(f"Solved in {num_guesses} guesses: {success}")

if __name__ == "__main__":
    main()