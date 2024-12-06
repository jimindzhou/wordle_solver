from typing import List, Dict, Tuple, Any
from collections import defaultdict
from wordle_solver_base import WordleSolverBase, SolverConfig
from helper_funcs import create_dict_list
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import multiprocessing

class WordleMCSolver(WordleSolverBase):
    """Monte Carlo Tree Search solver for Wordle"""
    def __init__(self, config: 'SolverConfig'):
        super().__init__(config)
        self.alpha = config.alpha
        self.beta = config.beta
        self.max_depth = config.max_depth
        self.debug = False
        self.n_simulations = config.max_simulations
        self.mc_process_num = config.mc_process_num
        
    def _load_words(self) -> List[str]:
        """Load word list from file specified in config"""
        word_file = Path(self.config.word_file)
        if not word_file.exists():
            raise FileNotFoundError(f"Word file {word_file} not found")
            
        with open(word_file, 'r') as f:
            return [word.strip() for word in f.readlines()]
    
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
        
        if len(self.current_word_list) < 500:
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
            # Perform multiple MC simulations using multiprocessing
            n_workers = min(multiprocessing.cpu_count(), self.mc_process_num)  # Optimal number of processes
            def run_trajectory(_):
                return self._generate_trajectory()

            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(run_trajectory, range(self.n_simulations)))
                
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