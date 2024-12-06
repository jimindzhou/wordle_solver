from typing import Dict, Any
from tqdm import tqdm
import numpy as np
import logging
from collections import Counter
import argparse
import matplotlib.pyplot as plt
from wordle_solver_base import WordleSolverBase, SolverConfig
from wordle_mc_solver import WordleMCSolver
from wordle_random_solver import WordleRandomSolver
import yaml
from enum import Enum

class SolverType(Enum):
    MCTS = "mcts" # Monte Carlo Tree Search
    RANDOM = "random" # Random guessing

def create_solver(config: 'SolverConfig') -> WordleSolverBase:
    """Create appropriate solver based on configuration"""
    solver_type = SolverType(config.type)
    
    if solver_type == SolverType.MCTS:
        return WordleMCSolver( config )
    elif solver_type == SolverType.RANDOM:
        return WordleRandomSolver( config )
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

def evaluate_solver(solver: WordleMCSolver) -> Dict[str, Any]:
    """Run evaluation of solver according to config"""
    n_games = solver.config.n_games
    success_count = 0
    all_attempts = []
    guess_distribution = Counter()
    game_histories = []

    games_pbar = tqdm(range(n_games), 
                     desc="Games Progress", 
                     position=0, 
                     leave=True)
    
    for game_num in games_pbar:
        solver.reset()
        solution = solver.true_solution # either one from today's wordle game or random
        game_record = {
            'random_seed': solver.config.random_seed,
            'solution': solution,
            'guesses': [],
            'states': []
        }
        
        # Print game header before attempts
        print(f"\nGame {game_num + 1}/{n_games} | Target: {solution}")
        success = False
        for attempt in range(6):
            # Get guess based on whether it's first attempt or not
            if attempt == 0 and solver.config.initial_guesses:
                guess = np.random.choice(solver.initial_guesses)
                state_size = len(solver.initial_guesses)
            else:
                guess = solver.get_next_guess()
                state_size = len(solver.current_word_list)
            
            # Record the guess and state
            game_record['guesses'].append(guess)
            game_record['states'].append(state_size)
            
            # Print attempt information
            print(f"  → Attempt {attempt + 1}: {guess} (Possible states: {state_size})")
            if guess == solution:
                success = True
                success_count += 1
                guess_distribution[attempt + 1] += 1
                all_attempts.append(attempt + 1)
                print(f"  ✓ Solution found in {attempt + 1} attempts!")
                break

            solver.update_state(guess, solution)
            if not solver.current_word_list:
                print("  × No valid words remaining!")
                break
        
        if not success:
            print("  × Failed to find solution")
        
        # Record game results
        game_record['success'] = success
        game_record['attempts'] = len(game_record['guesses'])
        game_histories.append(game_record)
        
        # Update main progress bar
        win_rate = (success_count / (len(game_histories))) * 100
        avg_attempts = np.mean(all_attempts) if all_attempts else 0
        games_pbar.set_postfix({
            'Win Rate': f"{win_rate:.1f}%",
            'Avg Attempts': f"{avg_attempts:.2f}",
            'Last Result': 'Success' if success else 'Fail'
        })
        
        # Add separator between games
        print("\n" + "="*50)
    
    return {
        'solver_type': solver.config.type,
        'n_games': n_games,
        'success_rate': (success_count / n_games) * 100,
        'avg_attempts': np.mean(all_attempts),
        'std_attempts': np.std(all_attempts),
        'guess_distribution': dict(guess_distribution),
        'game_histories': game_histories
    }

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Wordle solver with configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--solver', type=str, required=True, help='Name of solver to use from config')
    args = parser.parse_args()
    
    # Load solver configuration
    config = SolverConfig.from_yaml(args.config, 
                                    args.solver)
    logger.info(f"Initializing {args.solver} solver")
    solver = create_solver(config) # WordleMCSolver or WordleRandomSolver
    
    try:
        logger.info(f"Starting evaluation with {config.n_games} games")
        results = evaluate_solver(solver)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"\nSolver: {results['solver_type']}")
        print(f"Number of games: {results['n_games']}")
        print(f"Success rate: {results['success_rate']:.1f}%")
        print(f"Average attempts: {results['avg_attempts']:.2f} ± {results['std_attempts']:.2f}")
        
        # Save results
        results_file = f"results_{args.solver}.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f)
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()