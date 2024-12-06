from typing import Dict, Any
from tqdm import tqdm
import logging
from collections import Counter
import argparse
import matplotlib.pyplot as plt
from wordle_solver_base import WordleSolverBase, SolverConfig
from wordle_mc_solver import WordleMCSolver
from wordle_random_solver import WordleRandomSolver
import yaml
from enum import Enum
from pathlib import Path
from datetime import datetime
import json
from copy import deepcopy
import random 
import numpy as np
class SolverType(Enum):
    MCTS = "mcts" # Monte Carlo Tree Search
    RANDOM = "random" # Random guessing


def clean_for_yaml(obj):
    """Convert numpy types and arrays to native Python types for clean YAML output"""
    if isinstance(obj, dict):
        return {k: clean_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_yaml(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj
    
def save_results(results: Dict[str, Any], 
                 config: 'SolverConfig',
                   output_dir: str) -> None:
    """
    Save solver results in a structured format for easy comparison across different scenarios.
    
    Args:
        results: Dictionary containing solver results and configuration
        output_dir: Base directory for saving results
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract key configuration parameters for folder structure
    solver_type = config.type
    mc_runs = config.max_simulations
    if config.initial_guesses:
        initial_guesses = '_'.join(config.initial_guesses)
    else:
        initial_guesses = 'default'
    
    # Create timestamp for unique identification
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if mc_runs is None:
        scenario_path = output_path / solver_type / timestamp
    else:
        scenario_path = output_path / solver_type / f"mc_num__{mc_runs}" / f"init_{initial_guesses}" / timestamp
    scenario_path.mkdir(parents=True, exist_ok=True)
    results_file = scenario_path / "detailed_results.yaml"

    clean_results = clean_for_yaml(deepcopy(results))
    
    # Save detailed results as YAML with clean formatting
    results_file = scenario_path / "detailed_results.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(clean_results, f, default_flow_style=False)
    
    summary = {
        'solver_type': solver_type,
        'max_simulations': mc_runs,
        'initial_guesses': initial_guesses,
        'timestamp': timestamp,
        'success_rate': results['success_rate'],
        'avg_attempts': results['avg_attempts'],
        'std_attempts': results['std_attempts'],
        'n_games': results['n_games'],
        'guess_distribution': results['guess_distribution']
    }
    
    summary_file = scenario_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results_file
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
    np.random.seed(solver.config.random_seed)
    random.seed(solver.config.random_seed)

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
                guess = np.random.choice(solver.config.initial_guesses)
                state_size = len(solver.config.initial_guesses)
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
    parser.add_argument('--output', type=str, default='./output', help='Output directory for results')
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
        results_file = save_results(results, config, args.output)
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()