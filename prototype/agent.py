"""
Component 5: Refinement Agent
Refines organizational descriptions using LLM.
"""

from typing import Dict, List, Optional, Any

from graph_builder import GraphBuilder
from llm import LLM, create_llm


# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an expert organizational analyst.
Refine organizational unit descriptions to be clear, concise, and professional.
Focus on key responsibilities and how units relate to their hierarchy."""

REFINE_PROMPT = """Refine the description for this organizational unit:

Unit: {name} (Code: {code})
Level: {level} | Parent: {parent}

Current Activities:
{activities}

Context:
- Parent: {parent_desc}
- Children: {children_desc}

Write a refined description (2-3 sentences) that:
1. Summarizes key responsibilities
2. Shows relationship to parent/children
3. Uses professional language

Description:"""


# =============================================================================
# Agent
# =============================================================================

class RefinementAgent:
    """Agent for refining organizational descriptions."""
    
    def __init__(self, graph: GraphBuilder, llm: LLM):
        self.graph = graph
        self.llm = llm
        self.results = {}
    
    def _format_activities(self, node: str) -> str:
        data = self.graph.graph.nodes[node]
        acts = data.get("activities", [])
        weights = data.get("weights", [])
        if not acts:
            return "No activities listed"
        return "\n".join([f"- ({w}%) {a}" for a, w in zip(acts, weights)])
    
    def _get_context(self, node: str) -> Dict[str, str]:
        """Get parent/children context."""
        parent = self.graph.get_parent(node)
        children = self.graph.get_children(node)
        
        parent_desc = "Root level"
        if parent:
            p_data = self.graph.graph.nodes[parent]
            parent_desc = self.results.get(parent, p_data.get("name", parent))
        
        children_desc = "No subordinate units"
        if children:
            c_names = [self.graph.graph.nodes[c].get("name", c)[:30] for c in children[:3]]
            children_desc = ", ".join(c_names)
            if len(children) > 3:
                children_desc += f" (+{len(children)-3} more)"
        
        return {"parent": parent_desc, "children": children_desc}
    
    def refine_node(self, node: str) -> str:
        """Refine a single node's description."""
        data = self.graph.graph.nodes[node]
        context = self._get_context(node)
        
        prompt = REFINE_PROMPT.format(
            name=data.get("name", node),
            code=node,
            level=data.get("level", 0),
            parent=self.graph.get_parent(node) or "None",
            activities=self._format_activities(node),
            parent_desc=context["parent"],
            children_desc=context["children"]
        )
        
        description = self.llm.generate(prompt, SYSTEM_PROMPT)
        self.results[node] = description
        self.graph.update_node(node, refined_description=description)
        
        return description
    
    def run(self, strategy: str = "top_down", verbose: bool = True) -> Dict[str, Any]:
        """
        Run refinement on all nodes.
        
        Args:
            strategy: "top_down" or "bottom_up"
            verbose: Print progress
        """
        if verbose:
            print(f"\n{'='*50}")
            print(f"REFINEMENT: {strategy.upper()}")
            print(f"{'='*50}")
        
        if strategy == "top_down":
            nodes = self.graph.traverse_top_down()
        else:
            nodes = self.graph.traverse_bottom_up()
        
        for node in nodes:
            if verbose:
                name = self.graph.graph.nodes[node].get("name", node)[:30]
                print(f"\n[{node}] {name}")
            
            desc = self.refine_node(node)
            
            if verbose:
                print(f"  → {desc[:70]}...")
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Refined {len(self.results)} nodes")
            print(f"{'='*50}")
        
        return {
            "strategy": strategy,
            "nodes_processed": len(self.results),
            "descriptions": self.results
        }


# =============================================================================
# Factory
# =============================================================================

def create_agent(graph: GraphBuilder, llm: LLM = None, use_mock: bool = False) -> RefinementAgent:
    """Create refinement agent."""
    if llm is None:
        llm = create_llm(use_mock=use_mock)
    return RefinementAgent(graph, llm)


if __name__ == "__main__":
    from graph_builder import create_graph_builder
    
    builder = create_graph_builder()
    builder.load_excel("/mnt/user-data/uploads/BUDG_Activities.xlsx")
    
    agent = create_agent(builder, use_mock=True)
    result = agent.run(strategy="top_down")
    
    print(f"\nProcessed: {result['nodes_processed']} nodes")
