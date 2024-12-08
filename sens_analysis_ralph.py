import pandas as pd
import numpy as np
from typing import List, Dict
from pathlib import Path
from wordle_solver_base import SolverConfig
from tqdm import tqdm
import yaml
from copy import deepcopy
from wordle_main import evaluate_solver, create_solver

def run_comparative_study(
    config_path: str,
    solution_list_path: str,
    random_seeds: List[int],
    mc_simulations_range: List[int],
    output_path: str = "./sens_output/beat_ralph_stats.csv"
) -> pd.DataFrame:
    """
    Run comparative study between MCTS and Random solvers.
    
    Args:
        config_path: Path to solver configuration YAML
        solution_list_path: Path to file containing true words
        random_seeds: List of random seeds to use
        mc_simulations_range: List of max_simulations values to test for MCTS
        output_path: Path to save resulting statistics CSV
    """
    # Read true words
    with open(solution_list_path, 'r') as f:
        true_words = [word.strip() for word in f.readlines()]
    
    # Load base configuration
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Prepare results storage
    results = []
    
    # Calculate total runs for progress bar
    total_mcts_runs = len(true_words) * len(random_seeds) * len(mc_simulations_range)
    total_random_runs = len(true_words) * len(random_seeds)
    total_runs = total_mcts_runs + total_random_runs
    
    # Progress bar for overall progress
    with tqdm(total=total_runs, desc="Running comparative study") as pbar:
        # For each true word
        for true_word in true_words:
            # Test MCTS with different max_simulations
            mcts_config = base_config['solvers']['mcts_baseline']
            for mc_sims in mc_simulations_range:
                for seed in random_seeds:
                    # Create modified config
                    modified_config = deepcopy(mcts_config)
                    modified_config['max_simulations'] = mc_sims
                    modified_config['random_seed'] = seed
                    modified_config['n_games'] = 1  # Test one word at a time
                    
                    # Create solver config and run evaluation
                    solver_config = SolverConfig(modified_config)
                    solver = create_solver(solver_config)
                    solver.true_solution = true_word  # Override solution
                    
                    eval_results = evaluate_solver(solver)
                    
                    # Extract key metrics
                    run_result = {
                        'true_word': true_word,
                        'solver_type': 'mcts',
                        'max_simulations': mc_sims,
                        'random_seed': seed,
                        'success': eval_results['success_rate'] == 100,
                        'attempts': eval_results['avg_attempts'],
                        'final_state_size': eval_results['game_histories'][0]['states'][-1]
                    }
                    
                    results.append(run_result)
                    pbar.update(1)
            
            # Test Random solver
            random_config = base_config['solvers']['random_guesser']
            for seed in random_seeds:
                # Create modified config
                modified_config = deepcopy(random_config)
                modified_config['random_seed'] = seed
                modified_config['n_games'] = 1
                
                # Create solver config and run evaluation
                solver_config = SolverConfig(modified_config)
                solver = create_solver(solver_config)
                solver.true_solution = true_word
                
                eval_results = evaluate_solver(solver)
                
                # Extract key metrics
                run_result = {
                    'true_word': true_word,
                    'solver_type': 'random',
                    'max_simulations': None,  # Not applicable for random solver
                    'random_seed': seed,
                    'success': eval_results['success_rate'] == 100,
                    'attempts': eval_results['avg_attempts'],
                    'final_state_size': eval_results['game_histories'][0]['states'][-1]
                }
                
                results.append(run_result)
                pbar.update(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary statistics
    summary_stats = df.groupby(['solver_type', 'max_simulations', 'true_word']).agg({
        'success': 'mean',
        'attempts': ['mean', 'std'],
        'final_state_size': 'mean'
    }).round(3)
    
    # Save both detailed and summary results
    df.to_csv(output_path, index=False)
    summary_path = output_path.replace('.csv', '_summary.csv')
    summary_stats.to_csv(summary_path)
    
    return df, summary_stats

if __name__ == "__main__":
    # Configuration
    CONFIG_PATH = "./config/wordle_solver_mcts.yaml"
    SOLUTION_LIST_PATH = "./two-week-solutions.txt"
    RANDOM_SEEDS = list(range(1, 11))  # 10 random seeds
    MC_SIMULATIONS_RANGE = [10, 25, 50, 75, 100, 150]  # Different max_simulations values to test
    
    # Run study
    detailed_results, summary_results = run_comparative_study(
        config_path=CONFIG_PATH,
        solution_list_path=SOLUTION_LIST_PATH,
        random_seeds=RANDOM_SEEDS,
        mc_simulations_range=MC_SIMULATIONS_RANGE
    )
    
    print("\nSummary Statistics:")
    print(summary_results)