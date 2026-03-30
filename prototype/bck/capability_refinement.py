"""
Component 6B: Capability Refinement Module
Refines business capability descriptions with dual storage (NetworkX + FalkorDB).

Adapts refinement for capability hierarchy:
- Category (L0) → Business Area (L1) → Sub-Business Area (L2)
- Generates refined_description and capability_keywords
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass
class CapabilityRefinementConfig:
    """Configuration for capability refinement."""
    strategy: str = "top_down"
    sync_to_falkordb: bool = True
    verbose: bool = True
    max_nodes: Optional[int] = None


CAPABILITY_SYSTEM_PROMPT = """You are an expert business capability analyst for the European Parliament.
Your task is to create AUGMENTED descriptions of business capabilities.

CRITICAL RULES:
1. PRESERVE ALL original capability details
2. ENHANCE clarity while maintaining detail level
3. EXPLAIN how capabilities relate to organizational functions
4. USE professional business architecture language
5. EXTRACT relevant keywords for semantic matching"""

CAPABILITY_REFINEMENT_PROMPT = """Analyze this business capability and create an augmented description.

═══════════════════════════════════════════════════════════════════════════════
CAPABILITY INFORMATION
═══════════════════════════════════════════════════════════════════════════════
Capability Name: {name}
Capability ID: {cap_id}
Level: {level} (0=Category, 1=Business Area, 2=Sub-Business Area)
Type: {node_type}

═══════════════════════════════════════════════════════════════════════════════
ORIGINAL DESCRIPTION
═══════════════════════════════════════════════════════════════════════════════
{description}

═══════════════════════════════════════════════════════════════════════════════
CAPABILITY CONTEXT
═══════════════════════════════════════════════════════════════════════════════
Parent Capability: {parent_info}
Sub-Capabilities: {children_info}
Sibling Capabilities: {siblings_info}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Generate TWO outputs:

1. REFINED_DESCRIPTION (3-5 sentences):
   - Define the capability's purpose and scope
   - Explain what organizational functions it enables
   - Reference parent/child capabilities for context
   - Use business architecture terminology

2. CAPABILITY_KEYWORDS (comma-separated list):
   - Extract 5-10 key terms for semantic matching
   - Include synonyms and related concepts
   - Focus on actionable/functional terms

FORMAT:
REFINED_DESCRIPTION:
[Your comprehensive capability description]

CAPABILITY_KEYWORDS:
[keyword1, keyword2, keyword3, ...]
"""


def get_parent(graph: nx.DiGraph, node_id: str) -> Optional[str]:
    preds = list(graph.predecessors(node_id))
    return preds[0] if preds else None


def get_children(graph: nx.DiGraph, node_id: str) -> List[str]:
    return list(graph.successors(node_id))


def get_capability_context(graph: nx.DiGraph, node_id: str) -> dict:
    """Get hierarchical context for a capability."""
    parent_id = get_parent(graph, node_id)
    
    if parent_id:
        p_data = graph.nodes[parent_id]
        p_refined = p_data.get("refined_description", "")
        parent_info = f"{p_data.get('name', parent_id)}"
        if p_refined:
            parent_info += f": {p_refined[:150]}..."
    else:
        parent_info = "ROOT CAPABILITY"
    
    children = get_children(graph, node_id)
    if children:
        child_lines = [f"  • {graph.nodes[c].get('name', c)}" for c in children[:5]]
        children_info = "\n".join(child_lines)
        if len(children) > 5:
            children_info += f"\n  • ... +{len(children)-5} more"
    else:
        children_info = "LEAF CAPABILITY - No sub-capabilities"
    
    if parent_id:
        siblings = [s for s in get_children(graph, parent_id) if s != node_id]
        if siblings:
            sib_names = [graph.nodes[s].get("name", s)[:25] for s in siblings[:4]]
            siblings_info = ", ".join(sib_names)
            if len(siblings) > 4:
                siblings_info += f", +{len(siblings)-4} more"
        else:
            siblings_info = "Only capability at this level"
    else:
        siblings_info = "N/A (root level)"
    
    return {"parent_info": parent_info, "children_info": children_info, "siblings_info": siblings_info}


def parse_capability_response(response: str) -> dict:
    """Parse LLM response for capability refinement."""
    result = {"refined_description": "", "capability_keywords": ""}
    
    if "REFINED_DESCRIPTION:" in response:
        parts = response.split("REFINED_DESCRIPTION:", 1)
        if len(parts) > 1:
            desc = parts[1]
            if "CAPABILITY_KEYWORDS:" in desc:
                desc = desc.split("CAPABILITY_KEYWORDS:", 1)[0]
            result["refined_description"] = desc.strip()
    
    if "CAPABILITY_KEYWORDS:" in response:
        parts = response.split("CAPABILITY_KEYWORDS:", 1)
        if len(parts) > 1:
            result["capability_keywords"] = parts[1].strip()
    
    if not result["refined_description"]:
        result["refined_description"] = response.strip()
    
    return result


def traverse_top_down(graph: nx.DiGraph, roots: List[str]) -> List[str]:
    visited, queue = [], list(roots)
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            queue.extend(get_children(graph, node))
    return visited


def traverse_bottom_up(graph: nx.DiGraph) -> List[str]:
    by_level = {}
    for n in graph.nodes():
        lvl = graph.nodes[n].get("level", 0)
        by_level.setdefault(lvl, []).append(n)
    result = []
    for lvl in sorted(by_level.keys(), reverse=True):
        result.extend(sorted(by_level[lvl]))
    return result


class CapabilityRefinementAgent:
    """Agent for refining business capability descriptions with dual storage."""
    
    def __init__(self, capability_builder, llm, config: CapabilityRefinementConfig = None):
        self.capability_builder = capability_builder
        self.graph = capability_builder.graph
        self.llm = llm
        self.config = config or CapabilityRefinementConfig()
        self.results: Dict[str, dict] = {}
    
    def _sync_to_falkordb(self, node_id: str, result: dict):
        """Sync refinement to FalkorDB."""
        if not self.config.sync_to_falkordb:
            return
        
        try:
            falkordb = self.capability_builder.falkordb
            if hasattr(falkordb, 'graph') and falkordb.graph:
                query = """
                MATCH (n {node_id: $node_id})
                SET n.refined_description = $refined_description,
                    n.capability_keywords = $capability_keywords
                RETURN n.node_id
                """
                falkordb.graph.query(query, {
                    "node_id": node_id,
                    "refined_description": result["refined_description"],
                    "capability_keywords": result["capability_keywords"]
                })
            elif hasattr(falkordb, 'nodes') and node_id in falkordb.nodes:
                falkordb.nodes[node_id]["refined_description"] = result["refined_description"]
                falkordb.nodes[node_id]["capability_keywords"] = result["capability_keywords"]
        except Exception as e:
            if self.config.verbose:
                print(f"  ⚠ FalkorDB sync failed for {node_id}: {e}")
    
    def refine_capability(self, node_id: str) -> dict:
        """Refine a single capability. Updates NetworkX and FalkorDB."""
        data = self.graph.nodes[node_id]
        context = get_capability_context(self.graph, node_id)
        
        prompt = CAPABILITY_REFINEMENT_PROMPT.format(
            name=data.get("name", node_id),
            cap_id=node_id,
            level=data.get("level", 0),
            node_type=data.get("node_type", "capability"),
            description=data.get("description", "No description available"),
            parent_info=context["parent_info"],
            children_info=context["children_info"],
            siblings_info=context["siblings_info"]
        )
        
        response = self.llm.generate(prompt, CAPABILITY_SYSTEM_PROMPT)
        result = parse_capability_response(response)
        
        # Update NetworkX
        self.graph.nodes[node_id]["refined_description"] = result["refined_description"]
        self.graph.nodes[node_id]["capability_keywords"] = result["capability_keywords"]
        
        # Update FalkorDB
        self._sync_to_falkordb(node_id, result)
        
        return result
    
    def run_top_down(self, max_nodes: int = None) -> Dict[str, dict]:
        """Run top-down refinement."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("CAPABILITY TOP-DOWN REFINEMENT")
            print("═" * 70)
        
        roots = getattr(self.capability_builder, 'categories', None) or \
                [n for n in self.graph.nodes() if not get_parent(self.graph, n)]
        
        nodes = traverse_top_down(self.graph, roots)
        if max_nodes:
            nodes = nodes[:max_nodes]
        
        if self.config.verbose:
            print(f"Processing {len(nodes)} capabilities...\n")
        
        results = {}
        for i, node_id in enumerate(nodes):
            data = self.graph.nodes[node_id]
            result = self.refine_capability(node_id)
            results[node_id] = result
            
            if self.config.verbose:
                print(f"[{i+1:3}/{len(nodes)}] {node_id:15} {data.get('name', '')[:35]}")
                print(f"           → {result['refined_description'][:50]}...\n")
        
        if self.config.verbose:
            print(f"✓ Top-down complete: {len(results)} capabilities")
        return results
    
    def run_bottom_up(self, max_nodes: int = None, clear_existing: bool = False) -> Dict[str, dict]:
        """Run bottom-up refinement."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("CAPABILITY BOTTOM-UP REFINEMENT")
            print("═" * 70)
        
        if clear_existing:
            for node_id in self.graph.nodes():
                self.graph.nodes[node_id]["refined_description"] = ""
                self.graph.nodes[node_id]["capability_keywords"] = ""
        
        nodes = traverse_bottom_up(self.graph)
        if max_nodes:
            nodes = nodes[:max_nodes]
        
        if self.config.verbose:
            print(f"Processing {len(nodes)} capabilities...\n")
        
        results = {}
        for i, node_id in enumerate(nodes):
            data = self.graph.nodes[node_id]
            result = self.refine_capability(node_id)
            results[node_id] = result
            
            if self.config.verbose:
                print(f"[{i+1:3}/{len(nodes)}] {node_id:15} {data.get('name', '')[:35]}")
                print(f"           → {result['refined_description'][:50]}...\n")
        
        if self.config.verbose:
            print(f"✓ Bottom-up complete: {len(results)} capabilities")
        return results
    
    def run(self) -> Dict[str, Any]:
        """Run refinement based on strategy."""
        max_nodes = self.config.max_nodes
        
        if self.config.strategy == "top_down":
            self.results = self.run_top_down(max_nodes)
        elif self.config.strategy == "bottom_up":
            self.results = self.run_bottom_up(max_nodes, clear_existing=True)
        elif self.config.strategy == "bidirectional":
            self.run_top_down(max_nodes)
            self.results = self.run_bottom_up(max_nodes)
        
        refined_count = sum(1 for n in self.graph.nodes() if self.graph.nodes[n].get("refined_description"))
        
        return {
            "strategy": self.config.strategy,
            "total_capabilities": self.graph.number_of_nodes(),
            "refined": refined_count,
            "synced_to_falkordb": self.config.sync_to_falkordb
        }
    
    def export_refinements(self, path: str) -> str:
        """Export capability refinements to JSON."""
        refinements = {}
        for node_id in self.graph.nodes():
            data = self.graph.nodes[node_id]
            if data.get("refined_description"):
                refinements[node_id] = {
                    "name": data.get("name"),
                    "level": data.get("level"),
                    "node_type": data.get("node_type"),
                    "original_description": data.get("description", ""),
                    "refined_description": data.get("refined_description"),
                    "capability_keywords": data.get("capability_keywords", "")
                }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(refinements, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported capability refinements: {path} ({len(refinements)} capabilities)")
        return path


def create_capability_refinement_agent(
    capability_builder, llm,
    strategy: str = "top_down",
    sync_to_falkordb: bool = True,
    verbose: bool = True,
    max_nodes: int = None
) -> CapabilityRefinementAgent:
    """Create CapabilityRefinementAgent instance."""
    config = CapabilityRefinementConfig(
        strategy=strategy, sync_to_falkordb=sync_to_falkordb, verbose=verbose, max_nodes=max_nodes
    )
    return CapabilityRefinementAgent(capability_builder, llm, config)


if __name__ == "__main__":
    print("CapabilityRefinementAgent module loaded. Use create_capability_refinement_agent() to instantiate.")
