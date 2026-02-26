"""
Text generation utilities
"""

from typing import List, Optional, Dict, Any
from rwkvserve.inference.pipeline import InferencePipeline


class TextGenerator:
    """Text generator - provides a high-level generation interface"""
    
    PRESETS = {
        "greedy": {
            "temperature": 1.0,
            "top_p": 0.0,
            "top_k": 0,
        },
        "creative": {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 0,
        },
        "precise": {
            "temperature": 0.3,
            "top_p": 0.5,
            "top_k": 0,
        },
        "balanced": {
            "temperature": 0.6,
            "top_p": 0.8,
            "top_k": 0,
        },
    }
    
    def __init__(self, pipeline: InferencePipeline):
        """
        Args:
            pipeline: Inference pipeline instance
        """
        self.pipeline = pipeline
    
    def generate(
        self,
        prompt: str,
        preset: str = "balanced",
        max_tokens: int = 100,
        **kwargs
    ) -> str:
        """
        Generate text using a preset configuration.
        
        Args:
            prompt: Input prompt
            preset: Preset name (greedy/creative/precise/balanced)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments that override the preset
        
        Returns:
            Generated text
        """
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(self.PRESETS.keys())}")
        
        gen_kwargs = {**self.PRESETS[preset], **kwargs}
        
        return self.pipeline.generate(prompt, max_tokens=max_tokens, **gen_kwargs)
    
    def complete(
        self,
        prompt: str,
        max_tokens: int = 50,
        n: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple completions.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            n: Number of completions
            **kwargs: Additional arguments passed to generate
        
        Returns:
            List of completions
        """
        result = self.pipeline.generate(
            prompt, 
            max_tokens=max_tokens, 
            num_generations=n,
            **kwargs
        )
        return result if isinstance(result, list) else [result]
    
    def interactive_chat(self, system_prompt: Optional[str] = None):
        """
        Interactive chat session.
        
        Args:
            system_prompt: System prompt (optional)
        """
        print("=" * 60)
        print("RWKV-Kit Interactive Chat")
        print("Type 'quit' or 'exit' to end the session")
        print("=" * 60)
        
        if system_prompt:
            print(f"\nSystem: {system_prompt}\n")
        
        history = []
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("\nAssistant: ", end="")
                response = self.pipeline.chat(
                    user_input,
                    system_prompt=system_prompt,
                    history=history,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=200,
                    stream=True,
                )
                
                history.append((user_input, response))
                
                if len(history) > 5:
                    history = history[-5:]
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue
