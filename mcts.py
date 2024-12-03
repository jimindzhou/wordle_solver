from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple
import string
from tqdm import tqdm
import math

class WordleState:
    def __init__(self, target_word: str, current_guess: str = None, 
                 green_letters: Dict[int, str] = None, 
                 yellow_letters: Set[str] = None,
                 black_letters: Set[str] = None,
                 depth: int = 0,
                 valid_actions: List[str] = None):  # Added valid_actions
        self.target_word = target_word
        self.current_guess = current_guess
        self.green_letters = green_letters if green_letters is not None else {}
        self.yellow_letters = yellow_letters if yellow_letters is not None else set()
        self.black_letters = black_letters if black_letters is not None else set()
        self.depth = depth
        self.valid_actions = valid_actions  # Store available actions
        
    def is_terminal(self) -> bool:
        # Modified to check for valid actions
        return (self.current_guess == self.target_word or 
                self.depth >= 6 or 
                (self.valid_actions is not None and len(self.valid_actions) == 0))
        
    def get_result(self) -> str:
        if self.current_guess == self.target_word:
            return 'win'
        elif self.depth >= 6 or (self.valid_actions is not None and len(self.valid_actions) == 0):
            return 'loss'
        return None

class MCTSNode:
    def __init__(self, state: WordleState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[str, MCTSNode] = {}
        self.visits = 0
        self.value = 0.0
        self.untried_actions = state.valid_actions.copy() if state.valid_actions is not None else None
        
    def add_child(self, action: str, state: WordleState) -> 'MCTSNode':
        child = MCTSNode(state, self, action)
        self.children[action] = child
        return child
        
    def update(self, result: float):
        self.visits += 1
        self.value += result
        
    def get_ucb1(self, exploration_constant: float) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits)

class WordleMCTS:
    def __init__(self, word_list: List[str], initial_guesses: List[str], 
                 exploration_constant: float = 1.414, max_simulations: int = 1000):
        self.word_list = word_list
        self.initial_guesses = initial_guesses
        self.exploration_constant = exploration_constant
        self.max_simulations = max_simulations
        self.letter_frequencies = self._calculate_letter_frequencies()
        self.position_frequencies = self._calculate_position_frequencies()
        
    def _calculate_letter_frequencies(self) -> Dict[str, float]:
        freq = defaultdict(int)
        total = 0
        for word in self.word_list:
            for letter in set(word):
                freq[letter] += 1
                total += 1
        return {letter: count/total for letter, count in freq.items()}
    
    def _calculate_position_frequencies(self) -> Dict[Tuple[str, int], float]:
        freq = defaultdict(int)
        total = len(self.word_list)
        for word in self.word_list:
            for pos, letter in enumerate(word):
                freq[(letter, pos)] += 1
        return {key: count/total for key, count in freq.items()}

    def _play_guess(self, word: str, target: str) -> Tuple[Dict[int, str], Set[str], Set[str]]:
        """Simulate playing a guess and return feedback"""
        green_letters = {}
        yellow_letters = set()
        black_letters = set()
        
        # First pass: Find green letters
        for i, (guess_letter, target_letter) in enumerate(zip(word, target)):
            if guess_letter == target_letter:
                green_letters[i] = guess_letter
                
        # Second pass: Find yellow and black letters
        remaining_target_letters = list(target)
        for pos, letter in green_letters.items():
            remaining_target_letters.remove(letter)
            
        for i, letter in enumerate(word):
            if i in green_letters:
                continue
            if letter in remaining_target_letters:
                yellow_letters.add(letter)
                remaining_target_letters.remove(letter)
            else:
                black_letters.add(letter)
                
        return green_letters, yellow_letters, black_letters
    
    def _get_valid_actions(self, state: WordleState) -> List[str]:
        """Enhanced valid actions checking"""
        if state.depth == 0:
            return self.initial_guesses.copy()  # Return copy to prevent modification
            
        valid_words = []
        used_patterns = set()  # Track patterns we've already tried
        
        if state.current_guess:
            used_patterns.add(state.current_guess)  # Avoid repeating same guess
            
        for word in self.word_list:
            if word in used_patterns:
                continue
                
            if not all(word[pos] == letter 
                      for pos, letter in state.green_letters.items()):
                continue
                
            if not all(letter in word for letter in state.yellow_letters):
                continue
                
            skip_word = False
            for pos, letter in enumerate(word):
                if letter in state.black_letters:
                    # Allow black letter only if it appears elsewhere as yellow
                    if letter not in state.yellow_letters:
                        skip_word = True
                        break
                    # Ensure letter isn't used in a position we know is wrong
                    if pos in state.green_letters and state.green_letters[pos] != letter:
                        skip_word = True
                        break
            if skip_word:
                continue
                
            valid_words.append(word)
            
        return valid_words if valid_words else []  # Return empty list instead of fallback
    
    def _evaluate_word_score(self, word: str, state: WordleState) -> float:
        """Enhanced word scoring"""
        score = 0.0
        
        # Base scores
        unique_letters = set(word)
        freq_score = sum(self.letter_frequencies[letter] for letter in unique_letters)
        pos_score = sum(self.position_frequencies.get((letter, pos), 0) 
                       for pos, letter in enumerate(word))
        green_score = sum(1.0 for pos, letter in enumerate(word) 
                         if pos in state.green_letters and state.green_letters[pos] == letter)
        yellow_score = sum(0.5 for letter in word if letter in state.yellow_letters)
        pattern_diversity = len(unique_letters) / 5.0  # Reward words with unique letters
        
        # Combine scores with weights
        score = (0.25 * freq_score + 
                0.25 * pos_score + 
                0.25 * green_score + 
                0.15 * yellow_score +
                0.10 * pattern_diversity)
                
        return score
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """Enhanced selection with proper terminal state handling"""
        while not node.state.is_terminal():
            if node.untried_actions is None:
                valid_actions = self._get_valid_actions(node.state)
                node.untried_actions = valid_actions
                if not valid_actions:  # No valid actions available
                    break
            
            if node.untried_actions:
                return node
                
            if not node.children:  # No children available
                break
                
            node = max(node.children.values(), 
                      key=lambda n: n.get_ucb1(self.exploration_constant))
        return node
    
    def expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Enhanced expansion with validity checks"""
        if not node.untried_actions:
            return None
            
        action = node.untried_actions.pop()
        green_letters, yellow_letters, black_letters = self._play_guess(
            action, node.state.target_word)
            
        # Get valid actions for new state
        new_state = WordleState(
            target_word=node.state.target_word,
            current_guess=action,
            green_letters={**node.state.green_letters, **green_letters},
            yellow_letters=node.state.yellow_letters | yellow_letters,
            black_letters=node.state.black_letters | black_letters,
            depth=node.state.depth + 1,
            valid_actions=self._get_valid_actions(node.state)  # Pre-compute valid actions
        )
        
        return node.add_child(action, new_state)
    
    def simulate(self, state: WordleState) -> float:
        """Enhanced simulation with proper state handling"""
        current_state = WordleState(
            target_word=state.target_word,
            current_guess=state.current_guess,
            green_letters=state.green_letters.copy(),
            yellow_letters=state.yellow_letters.copy(),
            black_letters=state.black_letters.copy(),
            depth=state.depth,
            valid_actions=state.valid_actions.copy() if state.valid_actions else None
        )
        
        while not current_state.is_terminal():
            valid_actions = self._get_valid_actions(current_state)
            if not valid_actions:
                break
                
            action = max(valid_actions, 
                        key=lambda w: self._evaluate_word_score(w, current_state))
            
            green_letters, yellow_letters, black_letters = self._play_guess(
                action, current_state.target_word)
                
            current_state = WordleState(
                target_word=current_state.target_word,
                current_guess=action,
                green_letters={**current_state.green_letters, **green_letters},
                yellow_letters=current_state.yellow_letters | yellow_letters,
                black_letters=current_state.black_letters | black_letters,
                depth=current_state.depth + 1,
                valid_actions=valid_actions
            )
        
        if current_state.get_result() == 'win':
            return 1.0 / current_state.depth
        return 0.0
    
    def backpropagate(self, node: MCTSNode, result: float):
        """Backpropagate the result up the tree"""
        while node is not None:
            node.update(result)
            node = node.parent

    def get_best_action(self, state: WordleState) -> Optional[str]:
        """Enhanced best action selection with profiling"""
        if not state.valid_actions:
            state.valid_actions = self._get_valid_actions(state)
        
        if not state.valid_actions:
            return None
            
        root = MCTSNode(state)
        
        for _ in tqdm(range(self.max_simulations), 
                    desc=f"Depth {state.depth}: Simulations",
                    leave=False):
            node = self.select(root)
            if not node.state.is_terminal():
                new_node = self.expand(node)
                if new_node:
                    result = self.simulate(new_node.state)
                    self.backpropagate(new_node, result)
        
        if not root.children:
            return None
            
        return max(root.children.items(), 
                key=lambda item: (item[1].visits, item[1].value))[0]
