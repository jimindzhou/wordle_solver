
from typing import List, Dict, Any
from pathlib import Path
import yaml
import random
import numpy as np

class SolverConfig:
    """Configuration class for solver parameters"""
    def __init__(self, config_dict: Dict[str, Any]):
        self.type = config_dict['type']
        
        self.alpha = float(config_dict.get('alpha', 1000.0))
        self.beta = float(config_dict.get('beta', 10000.0))
        self.max_depth = 6
        self.max_simulations = int(config_dict.get('max_simulations', 100))
        self.n_games = int(config_dict.get('n_games', 100))
        self.word_file = config_dict.get('word_file', 'valid-wordle-words.txt')
        self.initial_guesses = config_dict.get('initial_guesses', None)
        self.true_solution = config_dict.get('true_solution', None)
        self.random_seed = config_dict.get('random_seed', 42)

    @classmethod
    def from_yaml(cls, yaml_path: str, solver_name: str) -> 'SolverConfig':
        """Create config from YAML file"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'solvers' not in config:
            raise ValueError("Config must have 'solvers' section")
        if solver_name not in config['solvers']:
            raise ValueError(f"Solver {solver_name} not found in config")
            
        return cls(config['solvers'][solver_name])

class WordleSolverBase:
    """Base class for Wordle solvers"""
    def __init__(self, config: 'SolverConfig'):
        """Initialize solver with configuration"""
        self.config = config
        self.word_list = self._load_words()

        self.initial_word_list = self.word_list.copy()
        self.current_word_list = self.word_list.copy()
        self.current_depth = 0
        self.trajectory_history = []  # history of action taken at each depth [during game]
        self.random_seed = config.random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        if config.true_solution:
            self.true_solution = config.true_solution # one from today's wordle game
        else:
            self.true_solution = np.random.choice(self.word_list)
        
    def _load_words(self) -> List[str]:
        """Load word list from file specified in config"""
        word_file = Path(self.config.word_file)
        if not word_file.exists():
            raise FileNotFoundError(f"Word file {word_file} not found")
            
        with open(word_file, 'r') as f:
            return [word.strip() for word in f.readlines()]
    
    def reset(self, seed: int):
        """Reset solver state"""
        np.random.seed(seed)
        random.seed(seed)

        self.current_word_list = self.initial_word_list.copy()
        self.current_depth = 0
        self.trajectory_history = []

        if self.config.true_solution:
            pass
        else:
            # a random word from the word list
            self.true_solution = np.random.choice(self.word_list)

    
    def get_next_guess(self) -> str:
        """
        Get the next word guess. This method should be implemented by derived classes.
        Returns:
            str: The next word to guess
        """
        raise NotImplementedError("Derived classes must implement get_next_guess()")
    
    def update_state(self, guess: str, solution: str) -> bool:
        """
        Update solver state based on guess and actual solution.
        
        Args:
            guess: Our guess word
            solution: Actual solution word
            
        Returns:
            bool: True if solution was found, False otherwise
        """
        from helper_funcs import state_transition  # Import here to avoid circular imports
        
        self.current_depth += 1
        new_word_list, _ = state_transition(self.current_word_list, solution, guess)
        self.current_word_list = new_word_list
        self.trajectory_history.append(new_word_list)
        
        return guess == solution or len(new_word_list) == 0