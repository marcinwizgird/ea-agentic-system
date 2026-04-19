"""
LLM Module — Anthropic Claude Integration
Provides text generation via Claude API with mock fallback for testing.
"""

import os
import json
import re
from typing import Optional, Dict


# =============================================================================
# LLM Class
# =============================================================================

class LLM:
    """Claude LLM provider with mock fallback."""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
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
        except Exception:
            pass
        return {}

    def _mock_response(self, prompt: str) -> str:
        """Mock response for testing without API key."""
        if "CAPABILITY_KEYWORDS" in prompt:
            return (
                "REFINED_DESCRIPTION:\n"
                "This business capability encompasses the organizational functions and "
                "processes required to deliver value in its domain area. It enables the "
                "European Parliament to effectively manage operations, coordinate resources, "
                "and achieve strategic objectives aligned with its institutional mandate.\n\n"
                "CAPABILITY_KEYWORDS:\n"
                "business capability, organizational function, process management, "
                "strategic alignment, service delivery, institutional operations"
            )
        elif "MATCH_SCORE" in prompt and "EVALUATION" in prompt:
            # LLM judge for bipartite matching — single pair evaluation
            return (
                "MATCH_SCORE: 0.72\n"
                "MATCH_TYPE: MODERATE\n"
                "JUSTIFICATION: The organizational unit's activities show functional "
                "alignment with this capability area, particularly in process management "
                "and service delivery aspects that map to the capability's core functions.\n"
                "KEY_OVERLAPS: operational coordination, process management, service delivery\n"
                "GAPS: Some specialized activities may not be fully covered by this capability"
            )
        elif "CANDIDATE CAPABILITIES" in prompt and "TOP 3" in prompt:
            # LLM batch ranking for bipartite matching
            return (
                "MATCH_1:\n"
                "CAP_ID: FIRST_CANDIDATE\n"
                "SCORE: 0.75\n"
                "MATCH_TYPE: MODERATE\n"
                "JUSTIFICATION: Strong functional alignment between the unit's core "
                "activities and this capability's process management functions.\n"
                "KEY_OVERLAPS: operational management, service delivery, coordination\n"
                "\n"
                "MATCH_2:\n"
                "CAP_ID: SECOND_CANDIDATE\n"
                "SCORE: 0.62\n"
                "MATCH_TYPE: WEAK\n"
                "JUSTIFICATION: Partial overlap in administrative coordination and "
                "support functions.\n"
                "KEY_OVERLAPS: administrative support, resource coordination\n"
            )
        elif "SELECTED_CATEGORIES" in prompt and "AVAILABLE CATEGORIES" in prompt:
            # LLM category screening for prescreened mode
            return (
                "SELECTED_CATEGORIES: FIRST_CAT, SECOND_CAT\n"
                "REASONING: These categories best align with the unit's operational "
                "mandate and core activities."
            )
        elif "activity" in prompt.lower() and "ACTIVITY_KEYWORDS" in prompt:
            return (
                "REFINED_DESCRIPTION:\n"
                "This activity involves coordinating key operational tasks within the "
                "organizational unit, ensuring alignment with the unit's strategic objectives "
                "and facilitating collaboration across teams.\n\n"
                "ACTIVITY_KEYWORDS:\n"
                "coordination, operational management, strategic alignment, collaboration, "
                "institutional support, policy implementation"
            )
        elif "refine" in prompt.lower() or "augment" in prompt.lower() or "REFINED_DESCRIPTION" in prompt:
            return (
                "REFINED_DESCRIPTION:\n"
                "This organizational unit coordinates key activities across its mandate, "
                "ensuring alignment with institutional objectives and effective collaboration "
                "with partner units. It manages its assigned portfolio through structured "
                "work allocation with clear percentage-based resource distribution.\n\n"
                "REFINED_ACTIVITY_SUMMARY:\n"
                "• Core Operations (40-60%): Primary mandate activities and service delivery\n"
                "• Coordination & Support (20-30%): Cross-unit collaboration and administrative tasks\n"
                "• Strategic & Oversight (10-20%): Planning, reporting, and institutional representation"
            )
        elif "merge" in prompt.lower():
            return (
                "REFINED_DESCRIPTION:\n"
                "This unit integrates top-down strategic direction with bottom-up operational "
                "expertise, delivering on its mandate through well-structured activity allocation. "
                "It bridges institutional priorities with practical implementation.\n\n"
                "REFINED_ACTIVITY_SUMMARY:\n"
                "• Core Mandate (40-60%): Primary service delivery and operational activities\n"
                "• Institutional Coordination (20-30%): Cross-unit and interinstitutional tasks\n"
                "• Strategic Management (10-20%): Planning, oversight, and resource management"
            )
        return f"Mock response for: {prompt[:50]}..."


# =============================================================================
# Factory
# =============================================================================

def create_llm(api_key: str = None, use_mock: bool = False,
               model: str = "claude-sonnet-4-20250514") -> LLM:
    """Create LLM instance."""
    if use_mock:
        return LLM(api_key=None)
    return LLM(api_key=api_key, model=model)


if __name__ == "__main__":
    llm = create_llm(use_mock=True)
    response = llm.generate("Refine this description")
    print(f"Response: {response}")
