from openai import OpenAI
from typing import Optional, List, Dict, Any
from mcts import MCTSNode, WordleMCTS, WordleState
import random
import re

class LLMMCTSHelper:
    def __init__(self, **config):
        """Initialize OpenAI client and set up prompting"""
        self.config = config
        api_key_file = config['api_key']
        try:
            with open(api_key_file, 'r') as f:
                api_key = f.read().strip()
            # Initialize OpenAI client with the read API key
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            raise Exception(f"Failed to read API key from {api_key_file}: {str(e)}")

        
        # Store config values with defaults
        self.model = config.get('model', 'gpt-3.5-turbo')
        self.temperature = config.get('temperature', 0.3)
        self.suggestion_weight = config.get('suggestion_weight', 0.3)
        
        self.system_prompt = """You are a Wordle solving assistant. Given the current game state, 
        suggest the best next word to guess. Consider:
        1. Letter frequency in English
        2. Position-specific letter probabilities
        3. Pattern recognition from common English words
        4. Information gained from previous guesses
        
        Provide your response in the format:
        WORD: [your suggestion]
        REASONING: [brief explanation]
        """
        
    def format_game_state(self, state: WordleState) -> str:
        """Format the current game state for the LLM prompt"""
        pattern = ['_'] * 5
        for pos, letter in state.green_letters.items():
            pattern[pos] = letter
            
        prompt = f"""Current Wordle state:
        Pattern: {' '.join(pattern)}
        Known letters (yellow): {', '.join(sorted(state.yellow_letters)) if state.yellow_letters else 'None'}
        Excluded letters: {', '.join(sorted(state.black_letters)) if state.black_letters else 'None'}
        Depth: {state.depth}
        
        Based on this information, what would be the best word to guess next? 
        Consider only valid 5-letter English words."""
        
        return prompt

    def get_suggestion(self, state: WordleState, valid_words: List[str]) -> Optional[str]:
        """Get word suggestion from LLM using configured parameters"""
        try:
            prompt = self.format_game_state(state)
            
            # Limit number of words to prevent token overflow
            max_words = self.config.get('max_valid_words', 50)
            if len(valid_words) > max_words:
                valid_words = valid_words[:max_words]
                words_note = f"\nShowing first {max_words} valid words of {len(valid_words)} total."
            else:
                words_note = ""

            # Get base prompt and add filtered words
            prompt = self.format_game_state(state)
            prompt += f"\n\nValid words to choose from: {', '.join(valid_words)}{words_note}"
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=150
            )
            
            # Extract suggested word from response
            suggestion = response.choices[0].message.content
            word_match = re.search(r'WORD:\s*(\w+)', suggestion)
            
            if word_match:
                suggested_word = word_match.group(1).lower()
                # Verify the suggestion is in valid words
                if suggested_word in valid_words:
                    return suggested_word
                    
            return None
            
        except Exception as e:
            print(f"LLM suggestion error: {e}")
            return None

    def get_config(self) -> Dict[str, Any]:
        """Return current configuration settings"""
        return {
            'model': self.model,
            'temperature': self.temperature,
            'suggestion_weight': self.suggestion_weight
        }

class LLMEnhancedWordleMCTS(WordleMCTS):
    def __init__(self, word_list: List[str],
                  initial_guesses: List[str], 
                 llm_helper: LLMMCTSHelper,
                 exploration_constant: float = 1.414, 
                 max_simulations: int = 1000,
                 llm_suggestion_weight: float = 0.3):
        super().__init__(word_list, initial_guesses, exploration_constant, max_simulations)
        self.llm_helper = llm_helper
        self.llm_suggestion_weight = llm_suggestion_weight
        
    def _evaluate_word_score(self, word: str, state: WordleState) -> float:
        """Enhanced word scoring with LLM suggestions"""
        # Get base score from original heuristics
        base_score = super()._evaluate_word_score(word, state)
        llm_suggestion = self.llm_helper.get_suggestion(state, self._get_valid_actions(state))
        
        # Apply LLM boost if this word matches the suggestion
        llm_boost = 1.0 if word == llm_suggestion else 0.0
        final_score = ((1 - self.llm_suggestion_weight) * base_score + 
                      self.llm_suggestion_weight * llm_boost)
                      
        return final_score
        
    def select(self, node: MCTSNode) -> MCTSNode:
        """Modified selection with LLM guidance"""
        while not node.state.is_terminal():
            if node.untried_actions is None:
                valid_actions = self._get_valid_actions(node.state)
                node.untried_actions = valid_actions
                if not valid_actions:
                    break
            
            if node.untried_actions:
                # Try to get LLM suggestion first
                llm_suggestion = self.llm_helper.get_suggestion(
                    node.state, node.untried_actions)
                    
                if llm_suggestion and llm_suggestion in node.untried_actions:
                    # Use LLM suggestion with high probability
                    if random.random() < 0.7:  # 70% chance to use LLM suggestion
                        node.untried_actions.remove(llm_suggestion)
                        return node
                
                return node
                
            if not node.children:
                break
                
            node = max(node.children.values(), 
                      key=lambda n: n.get_ucb1(self.exploration_constant))
        return node