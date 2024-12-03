import random

from typing import List, Optional
import re
from tqdm import tqdm
import yaml
import argparse
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum
from mcts import WordleMCTS, WordleState
from llmmcts import LLMMCTSHelper, LLMEnhancedWordleMCTS

class SolverType(Enum):
    MCTS = "mcts"
    LLM_MCTS = "llm_mcts"
    HYBRID = "hybrid"

@dataclass
class SolverConfig:
    solver_type: SolverType
    initial_guesses: list
    exploration_constant: float
    max_simulations: int
    n_games: int
    word_file: str
    llm_config: Dict[str, Any] = None

def load_config(config_file: str) -> Dict[str, Any]:
    """Load solver configuration from YAML file"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def create_solver(config: Dict[str, Any]) -> WordleMCTS:
    """Create appropriate solver based on configuration"""
    solver_type = SolverType(config['type'])
    
    if solver_type == SolverType.MCTS:
        return WordleMCTS(
            word_list=load_word_list(config['word_file']),
            initial_guesses=config['initial_guesses'],
            exploration_constant=config['exploration_constant'],
            max_simulations=config['max_simulations']
        )
    elif solver_type == SolverType.LLM_MCTS:
        llm_helper = LLMMCTSHelper(**config['llm_config'])
        return LLMEnhancedWordleMCTS(
            word_list=load_word_list(config['word_file']),
            initial_guesses=config['initial_guesses'],
            llm_helper=llm_helper,
            exploration_constant=config['exploration_constant'],
            max_simulations=config['max_simulations'],
            llm_suggestion_weight=config['llm_config']['suggestion_weight']
        )
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

def load_word_list(word_file: str) -> List[str]:
    """Load word list from file"""
    with open(word_file, 'r') as f:
        return [word.strip() for word in f]

def run_evaluation(solver: WordleMCTS, n_games: int) -> Dict[str, float]:
    """Run evaluation and return metrics"""
    wins = 0
    total_attempts = 0
    
    for _ in tqdm(range(n_games), desc="Playing games"):
        target_word = random.choice(solver.word_list)
        #target_word = "tract"
        print(f"\nTarget word: {target_word}")
        result = play_wordle(solver, target_word)
        
        if result > 0:
            wins += 1
            total_attempts += result
            print(f"Won in {result} attempts")
        else:
            total_attempts += 6
            print("Failed to solve")
    
    return {
        "total_wins": wins,
        "total_losses": n_games - wins,
        "average_attempts": total_attempts / n_games,
        "win_rate": wins / n_games
    }

def play_wordle(mcts: WordleMCTS, target_word: str) -> int:
    """Enhanced play function with proper state handling"""
    initial_valid_actions = mcts._get_valid_actions(
        WordleState(target_word, valid_actions=[]))
    
    state = WordleState(target_word, valid_actions=initial_valid_actions)
    
    while not state.is_terminal():
        action = mcts.get_best_action(state)
        if action is None:
            return -1
            
        green_letters, yellow_letters, black_letters = mcts._play_guess(
            action, target_word)
            
        # Get valid actions for new state
        new_valid_actions = mcts._get_valid_actions(state)
            
        state = WordleState(
            target_word=target_word,
            current_guess=action,
            green_letters={**state.green_letters, **green_letters},
            yellow_letters=state.yellow_letters | yellow_letters,
            black_letters=state.black_letters | black_letters,
            depth=state.depth + 1,
            valid_actions=new_valid_actions
        )
        
        if state.get_result() == 'win':
            return state.depth
            
    return -1

def main():
    parser = argparse.ArgumentParser(description='Run Wordle solver with configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--solver', type=str, required=True, help='Name of solver to use from config')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    solver_config = config['solvers'][args.solver]
    
    # Create solver
    solver = create_solver(solver_config)
    
    # Run evaluation
    metrics = run_evaluation(solver, solver_config['n_games'])
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Total wins: {metrics['total_wins']}")
    print(f"Total losses: {metrics['total_losses']}")
    print(f"Average attempts: {metrics['average_attempts']:.2f}")
    print(f"Win rate: {metrics['win_rate']:.2f}")

if __name__ == "__main__":
    main()