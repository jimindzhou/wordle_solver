from typing import List, Any
import yaml
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from wordle_solver_base import SolverConfig
from wordle_main import evaluate_solver, create_solver
import numpy as np

def run_sensitivity_analysis(
    base_config_path: str,
    mc_simulations_range: List[int],
    random_seeds: List[int],
    output_dir: str
) -> pd.DataFrame:
    """
    Run sensitivity analysis across different max_simulations values, solvers, and random seeds.
    
    Args:
        base_config_path: Path to base configuration YAML
        mc_simulations_range: List of max_simulations values to test
        random_seeds: List of random seeds to test
        output_dir: Directory to save results
    """
    # Load base configuration
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    results = []
    
    # Test MCTS with different max_simulations and random seeds
    mcts_config = base_config['solvers']['mcts_baseline']
    for mc_sims in mc_simulations_range:
        for seed in random_seeds:
            # Create modified config
            modified_config = deepcopy(mcts_config)
            modified_config['max_simulations'] = mc_sims
            modified_config['random_seed'] = seed
            
            print(f"\nRunning MCTS with max_simulations={mc_sims}, seed={seed}")
            
            # Create solver config and run evaluation
            solver_config = SolverConfig(modified_config)
            solver = create_solver(solver_config)
            eval_results = evaluate_solver(solver)
            
            # Store results
            results.append({
                'solver_type': 'MCTS',
                'max_simulations': mc_sims,
                'random_seed': seed,
                'success_rate': eval_results['success_rate'],
                'avg_attempts': eval_results['avg_attempts'],
                'std_attempts': eval_results['std_attempts']
            })
    
    # Run random solver with different seeds
    random_config = base_config['solvers']['random_guesser']
    for seed in random_seeds:
        modified_random_config = deepcopy(random_config)
        modified_random_config['random_seed'] = seed
        
        print(f"\nRunning Random solver with seed={seed}")
        
        random_solver_config = SolverConfig(modified_random_config)
        random_solver = create_solver(random_solver_config)
        random_results = evaluate_solver(random_solver)
        
        results.append({
            'solver_type': 'Random',
            'max_simulations': None,
            'random_seed': seed,
            'success_rate': random_results['success_rate'],
            'avg_attempts': random_results['avg_attempts'],
            'std_attempts': random_results['std_attempts']
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    plot_sensitivity_results(df, output_dir)
    
    return df

def plot_sensitivity_results(df: pd.DataFrame, output_dir: str):
    """Create visualizations for sensitivity analysis results with seed variation."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn')
    
    # Success Rate vs Max Simulations with seed variation
    plt.figure(figsize=(12, 7))
    
    # Plot MCTS results
    mcts_df = df[df['solver_type'] == 'MCTS']
    # Plot individual seed lines
    for seed in mcts_df['random_seed'].unique():
        seed_data = mcts_df[mcts_df['random_seed'] == seed]
        plt.plot(seed_data['max_simulations'], 
                seed_data['success_rate'], 
                'o-', 
                alpha=0.3, 
                label=f'MCTS Seed {seed}')
    
    # Plot MCTS mean
    mcts_mean = mcts_df.groupby('max_simulations')['success_rate'].mean()
    plt.plot(mcts_mean.index, 
            mcts_mean.values, 
            'b-', 
            linewidth=2, 
            label='MCTS Mean')
    
    # Plot Random solver results
    random_df = df[df['solver_type'] == 'Random']
    random_mean = random_df['success_rate'].mean()
    random_std = random_df['success_rate'].std()
    plt.axhline(y=random_mean, 
                color='r', 
                linestyle='--', 
                label=f'Random Mean ({random_mean:.1f}%)')
    plt.fill_between(plt.xlim(), 
                    random_mean - random_std, 
                    random_mean + random_std, 
                    color='r', 
                    alpha=0.2)
    
    plt.title('Success Rate vs Max Simulations (with Seed Variation)')
    plt.xlabel('Max Simulations')
    plt.ylabel('Success Rate (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path / 'success_rate_sensitivity.png', bbox_inches='tight')
    
    # Average Attempts vs Max Simulations with seed variation
    plt.figure(figsize=(12, 7))
    
    # Plot MCTS results
    for seed in mcts_df['random_seed'].unique():
        seed_data = mcts_df[mcts_df['random_seed'] == seed]
        plt.plot(seed_data['max_simulations'], 
                seed_data['avg_attempts'], 
                'o-', 
                alpha=0.3, 
                label=f'MCTS Seed {seed}')
    
    # Plot MCTS mean
    mcts_mean = mcts_df.groupby('max_simulations')['avg_attempts'].mean()
    plt.plot(mcts_mean.index, 
            mcts_mean.values, 
            'b-', 
            linewidth=2, 
            label='MCTS Mean')
    
    # Plot Random solver results
    random_mean = random_df['avg_attempts'].mean()
    random_std = random_df['avg_attempts'].std()
    plt.axhline(y=random_mean, 
                color='r', 
                linestyle='--', 
                label=f'Random Mean ({random_mean:.2f})')
    plt.fill_between(plt.xlim(), 
                    random_mean - random_std, 
                    random_mean + random_std, 
                    color='r', 
                    alpha=0.2)
    
    plt.title('Average Attempts vs Max Simulations (with Seed Variation)')
    plt.xlabel('Max Simulations')
    plt.ylabel('Average Attempts')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path / 'avg_attempts_sensitivity.png', bbox_inches='tight')
    
    # Save statistical summary
    summary = df.groupby(['solver_type', 'max_simulations'])[
        ['success_rate', 'avg_attempts', 'std_attempts']
    ].agg(['mean', 'std']).round(2)
    summary.to_csv(output_path / 'statistical_summary.csv')
    
    # Save detailed results to CSV
    df.to_csv(output_path / 'sensitivity_analysis_results.csv', index=False)

if __name__ == "__main__":
    # Example usage with multiple seeds
    mc_simulations_range = [10]
    random_seeds = [42]  # Multiple seeds for robustness
    output_dir = Path('./sens_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run sensitivity analysis
    results_df = run_sensitivity_analysis(
        base_config_path='./config/wordle_solver_mcts.yaml',
        mc_simulations_range=mc_simulations_range,
        random_seeds=random_seeds,
        output_dir='./sens_output/'
    )
    
    # Save results to CSV
    csv_path = output_dir / 'sensitivity_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")