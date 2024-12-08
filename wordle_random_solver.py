from wordle_solver_base import WordleSolverBase, SolverConfig
import random

class WordleRandomSolver(WordleSolverBase):
    """Simple random word selection solver"""
    def __init__(self, config: SolverConfig):
        """Initialize solver with base class configuration"""
        super().__init__(config)

    def get_next_guess(self) -> str:
        """
        Randomly select a word from the current word list.
        
        Returns:
            str: Randomly selected word
        """
        return random.choice(self.current_word_list)
