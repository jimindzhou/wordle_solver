from typing import List, Dict, Tuple, Any
from collections import defaultdict
from helper_funcs import create_dict_list, create_dict_list_p, state_transition
import numpy as np
import random
from pathlib import Path
import yaml
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm
import multiprocessing
import concurrent.futures 

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
        self.initial_guesses = config_dict.get('initial_guesses', self.word_file)
        
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
    
class WordleMCSolver:
    """Monte Carlo Tree Search solver for Wordle"""
    def __init__(self, config: SolverConfig):
        """Initialize solver with configuration"""
        self.config = config
        self.word_list = self._load_words()
        self.initial_word_list = self.word_list.copy()
        self.current_word_list = self.word_list.copy()
        self.alpha = config.alpha
        self.beta = config.beta
        self.max_depth = config.max_depth
        self.debug = False
        self.current_depth = 0
        self.trajectory_history = [] # history of action taked at each depth [during game]
        self.n_simulations = config.max_simulations # number of MC simulations to run at each depth
        
    def _load_words(self) -> List[str]:
        """Load word list from file specified in config"""
        word_file = Path(self.config.word_file)
        if not word_file.exists():
            raise FileNotFoundError(f"Word file {word_file} not found")
            
        with open(word_file, 'r') as f:
            return [word.strip() for word in f.readlines()]
    
    def reset(self):
        """Reset solver state"""
        self.current_word_list = self.initial_word_list.copy()
        self.current_depth = 0
        self.trajectory_history = []
    
    def _compute_score(self, trajectory: List[List[str]]) -> float:
        """
        Compute the score for a given trajectory.
        
        Args:
            trajectory: List of states (word lists) in the trajectory
            
        Returns:
            float: Computed score based on trajectory
        """
        score = 0.0
        nstates = len(trajectory)
        
        for guess, state in enumerate(trajectory):
            if guess == 0:
                continue
            if guess == (nstates - 1):
                if len(state) == 0:
                    score += self.beta
                else:
                    score -= self.beta
            else:
                score -= ((len(state) - 1) + self.alpha * guess)
        
        return score
    
    def _generate_trajectory(self) -> Tuple[List[List[str]], str]:
        """
        Generate a single Monte Carlo trajectory from current state.
        
        Returns:
            Tuple of (trajectory, first_action)
        """
        trajectory = []
        current_words = self.current_word_list
        current_depth = self.current_depth
        
        while current_depth < self.max_depth and len(current_words) > 0:
            dict_list, stat_list, tran_list, prob_list = create_dict_list(current_words)
            
            if not prob_list:
                break
            
            action = random.choices(current_words, weights=prob_list, k=1)[0]
            if current_depth == self.current_depth:
                first_action = action
                trajectory = []
            
            action_idx = current_words.index(action)
            possible_transitions = tran_list[action_idx]
            
            patterns = list(possible_transitions.keys())
            probs = [possible_transitions[p][0] for p in patterns]
            chosen_pattern = random.choices(patterns, weights=probs, k=1)[0]
            
            next_words = dict_list[action_idx][chosen_pattern]
            
            if chosen_pattern == ('green', 'green', 'green', 'green', 'green') or not next_words:
                trajectory.append(next_words)
                break
            
            trajectory.append(next_words)
            current_words = next_words
            current_depth += 1
        
        return trajectory, first_action
    
    def get_next_guess(self) -> str:
        """
        Run multiple Monte Carlo simulations to determine the best next guess.
        """
        action_scores = defaultdict(list)
        
        desc = f"Depth {self.current_depth + 1} MC sims"
        # Use regular range if debug is False
        iterator = tqdm(range(self.n_simulations), 
                       desc=desc,
                       disable=not self.debug,  # Disable tqdm when debug is False
                       leave=False) if self.debug else range(self.n_simulations)
        
        if len(self.current_word_list) < 2000:
        # Perform multiple MC simulations 
            for _ in iterator:
                trajectory, first_action = self._generate_trajectory()
                score = self._compute_score(trajectory)
                action_scores[first_action].append(score)
                
                # Only update postfix if debug is True and using tqdm
                if self.debug:
                    if action_scores:
                        current_best_score = max(
                            np.mean(scores) for scores in action_scores.values()
                        )
                        iterator.set_postfix({
                            'words': len(self.current_word_list),
                            'best_score': f"{current_best_score:.1f}"
                        })

        else:
        # Perform multiple MC simulations
            mcruns = [self._generate_trajectory] * self.n_simulations
            with concurrent.futures.ProcessPoolExecutor(max_workers = 8) as executor:
                futures = [executor.submit(func) for func in mcruns]  
                # # Display the number of active processes              
                # while any(future.running() for future in futures):                    
                #     active_processes = multiprocessing.active_children()
                #     print(f"Active processes: {len(active_processes)}")
                #     time.sleep(0.5)  # Check every 500ms
                results = [future.result() for future in futures]
                

            for trajectory, first_action in results:
                score = self._compute_score(trajectory)
                action_scores[first_action].append(score)
        
        # Calculate average scores and return best guess
        avg_scores = {
            action: np.mean(scores) for action, scores in action_scores.items()
        }
        
        if not avg_scores:
            return random.choice(self.current_word_list)
        
        best_guess = max(avg_scores.items(), key=lambda x: x[1])[0]
        return best_guess
    
    def update_state(self, guess: str, solution: str) -> bool:
        """
        Update solver state based on guess and actual solution.
        
        Args:
            guess: Our guess word from MC simulation or initial guess
            solution: Actual solution word
            
        Returns:
            bool: True if solution was found, False otherwise
        """
        self.current_depth += 1
        new_word_list, _ = state_transition(self.current_word_list, solution, guess)
        self.current_word_list = new_word_list
        self.trajectory_history.append(new_word_list)
        
        return guess == solution or len(new_word_list) == 0