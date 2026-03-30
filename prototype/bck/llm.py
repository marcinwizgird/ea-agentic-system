"""
Component 3: LLM (Claude Opus 4.6)
Anthropic Claude integration for text generation.
"""

import os
import json
import re
from typing import Optional, Dict

# =============================================================================
# LLM Class
# =============================================================================

class LLM:
    """Claude Opus 4.6 LLM provider."""
    
    def __init__(self, api_key: str = None, model: str = "claude-3-5-sonnet-2024102"):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.llm = None
        
        if self.api_key:
            try:
                from langchain_anthropic import ChatAnthropic
                self.llm = ChatAnthropic(
                    model=model,
                    api_key=self.api_key,
                    max_tokens=2048,
                    temperature=0.7
                )
                print(f"✓ LLM: {model}")
            except ImportError:
                raise ImportError("pip install langchain-anthropic")
        else:
            print("✓ LLM: Mock mode (no API key)")
    
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate text response."""
        if self.llm is None:
            return self._mock_response(prompt)
        
        from langchain_core.messages import HumanMessage, SystemMessage
        
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        
        response = self.llm.invoke(messages)
        return response.content
    
    def generate_json(self, prompt: str, system: str = "") -> Dict:
        """Generate JSON response."""
        response = self.generate(prompt + "\n\nRespond with valid JSON only.", system)
        
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return {}
    
    def _mock_response(self, prompt: str) -> str:
        """Mock response for testing."""
        if "refine" in prompt.lower():
            return "This unit coordinates key activities, ensuring alignment with objectives and collaboration across teams."
        return f"Mock response for: {prompt[:40]}..."


# =============================================================================
# Factory
# =============================================================================

def create_llm(api_key: str = None, use_mock: bool = False, model: str = "claude-3-5-sonnet-2024102") -> LLM:
    """Create LLM instance."""
    if use_mock:
        return LLM(api_key=None)
    return LLM(api_key=api_key, model = model)


if __name__ == "__main__":
    llm = create_llm(use_mock=True)
    response = llm.generate("Refine this description")
    print(f"Response: {response}")
