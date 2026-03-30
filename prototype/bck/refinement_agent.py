"""
Component 5: Refinement Agent Module
Augmented activity description refinement with dual storage (NetworkX + FalkorDB).

Features:
- Preserves ALL original activity details in augmented descriptions
- Creates refined_description and refined_activity_summary
- Bidirectional traversal (top-down + bottom-up)
- Syncs to both NetworkX graph and FalkorDB immediately
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass
class RefinementConfig:
    """Configuration for refinement agent."""
    strategy: str = "bidirectional"
    sync_to_falkordb: bool = True
    verbose: bool = True
    max_nodes: Optional[int] = None


SYSTEM_PROMPT = """You are an expert organizational analyst for the European Parliament.
Your task is to create AUGMENTED descriptions that PRESERVE ALL original details.

CRITICAL RULES:
1. NEVER reduce or summarize away any activity details
2. ENHANCE clarity while MAINTAINING the same or greater level of detail
3. ADD context about organizational relationships
4. USE professional, precise language
5. INCLUDE all percentage allocations in your response"""

AUGMENTED_REFINEMENT_PROMPT = """Analyze this organizational unit and create an augmented description.

═══════════════════════════════════════════════════════════════════════════════
UNIT INFORMATION
═══════════════════════════════════════════════════════════════════════════════
Unit Name: {name}
Unit Code: {code}
Hierarchy Level: {level} (0=DG, 1=Direction, 2=Unit)

═══════════════════════════════════════════════════════════════════════════════
ORIGINAL ACTIVITIES (MUST BE FULLY PRESERVED IN YOUR OUTPUT)
═══════════════════════════════════════════════════════════════════════════════
{activities_full}

═══════════════════════════════════════════════════════════════════════════════
ORGANIZATIONAL CONTEXT
═══════════════════════════════════════════════════════════════════════════════
Parent Unit: {parent_info}
Child Units: {children_info}
Sibling Units: {siblings_info}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK - Generate TWO outputs:
═══════════════════════════════════════════════════════════════════════════════

1. REFINED_DESCRIPTION (4-6 sentences):
   - Start with unit's primary mission/role
   - INCLUDE ALL activities with their percentage allocations
   - Explain HOW activities contribute to organizational goals
   - Reference relationships with parent/child units where relevant
   - Use professional European institutional language
   
2. REFINED_ACTIVITY_SUMMARY (structured bullet points):
   - Group related activities into 3-5 thematic areas
   - Show percentage ranges for each area
   - Highlight primary focus areas (highest % allocations)

FORMAT YOUR RESPONSE EXACTLY AS:

REFINED_DESCRIPTION:
[Your comprehensive augmented description here - must include all activity details]

REFINED_ACTIVITY_SUMMARY:
• [Theme 1] (X-Y%): [grouped activities]
• [Theme 2] (X-Y%): [grouped activities]
• [Theme 3] (X-Y%): [grouped activities]
"""

MERGE_PROMPT = """Merge these two perspectives into a comprehensive organizational description.

══ TOP-DOWN PERSPECTIVE ══
{td_desc}

Summary: {td_summary}

══ BOTTOM-UP PERSPECTIVE ══  
{bu_desc}

Summary: {bu_summary}

══ TASK ══
Create a FINAL merged description that integrates BOTH perspectives and preserves ALL activity details.

FORMAT:
REFINED_DESCRIPTION:
[Comprehensive 5-7 sentence description]

REFINED_ACTIVITY_SUMMARY:
• [Merged thematic areas with percentages]
"""


def format_activities_full(graph: nx.DiGraph, node_id: str) -> str:
    """Format activities with COMPLETE details."""
    data = graph.nodes[node_id]
    activities = data.get("activities", [])
    weights = data.get("activity_weights", [])
    
    if not activities:
        return "No activities documented for this unit."
    
    lines = []
    for i, (activity, weight) in enumerate(zip(activities, weights), 1):
        lines.append(f"Activity {i} [{weight}% of workload]:")
        lines.append(f"  {activity}")
        lines.append("")
    
    lines.append(f"Total documented workload: {sum(weights)}%")
    return "\n".join(lines)


def get_parent(graph: nx.DiGraph, node_id: str) -> Optional[str]:
    preds = list(graph.predecessors(node_id))
    return preds[0] if preds else None


def get_children(graph: nx.DiGraph, node_id: str) -> List[str]:
    return list(graph.successors(node_id))


def get_hierarchical_context(graph: nx.DiGraph, node_id: str) -> dict:
    """Get complete hierarchical context for a node."""
    parent_id = get_parent(graph, node_id)
    
    if parent_id:
        p_data = graph.nodes[parent_id]
        p_refined = p_data.get("refined_description", "")
        parent_info = f"{p_data.get('name', parent_id)}"
        if p_refined:
            parent_info += f"\n  Context: {p_refined[:200]}..."
    else:
        parent_info = "ROOT LEVEL - Reports directly to Secretary General"
    
    children = get_children(graph, node_id)
    if children:
        child_lines = []
        for c_id in children[:6]:
            c_data = graph.nodes[c_id]
            c_refined = c_data.get("refined_description", "")
            if c_refined:
                child_lines.append(f"  • {c_data.get('name', c_id)}: {c_refined[:80]}...")
            else:
                child_lines.append(f"  • {c_data.get('name', c_id)} ({len(c_data.get('activities', []))} activities)")
        children_info = "\n".join(child_lines)
        if len(children) > 6:
            children_info += f"\n  • ... and {len(children)-6} more units"
    else:
        children_info = "LEAF NODE - No subordinate units"
    
    if parent_id:
        siblings = [s for s in get_children(graph, parent_id) if s != node_id]
        if siblings:
            sib_names = [graph.nodes[s].get("name", s)[:25] for s in siblings[:5]]
            siblings_info = ", ".join(sib_names)
            if len(siblings) > 5:
                siblings_info += f", +{len(siblings)-5} more"
        else:
            siblings_info = "Only unit under parent"
    else:
        siblings_info = "N/A (root level)"
    
    return {"parent_info": parent_info, "children_info": children_info, "siblings_info": siblings_info}


def parse_llm_response(response: str) -> dict:
    """Parse LLM response into description and summary."""
    result = {"refined_description": "", "refined_activity_summary": ""}
    
    if "REFINED_DESCRIPTION:" in response:
        parts = response.split("REFINED_DESCRIPTION:", 1)
        if len(parts) > 1:
            desc = parts[1]
            if "REFINED_ACTIVITY_SUMMARY:" in desc:
                desc = desc.split("REFINED_ACTIVITY_SUMMARY:", 1)[0]
            result["refined_description"] = desc.strip()
    
    if "REFINED_ACTIVITY_SUMMARY:" in response:
        parts = response.split("REFINED_ACTIVITY_SUMMARY:", 1)
        if len(parts) > 1:
            result["refined_activity_summary"] = parts[1].strip()
    
    if not result["refined_description"]:
        result["refined_description"] = response.strip()
    
    return result


def traverse_top_down(graph: nx.DiGraph, roots: List[str]) -> List[str]:
    """BFS traversal from roots."""
    visited, queue = [], list(roots)
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            queue.extend(get_children(graph, node))
    return visited


def traverse_bottom_up(graph: nx.DiGraph) -> List[str]:
    """Traverse from leaves to roots."""
    by_level = {}
    for n in graph.nodes():
        lvl = graph.nodes[n].get("level", 0)
        by_level.setdefault(lvl, []).append(n)
    result = []
    for lvl in sorted(by_level.keys(), reverse=True):
        result.extend(sorted(by_level[lvl]))
    return result


class RefinementAgent:
    """Agent for refining organizational unit activity descriptions with dual storage."""
    
    def __init__(self, graph_builder, llm, config: RefinementConfig = None):
        self.graph_builder = graph_builder
        self.graph = graph_builder.graph
        self.llm = llm
        self.config = config or RefinementConfig()
        self.results: Dict[str, dict] = {}
        self.td_snapshot: Dict[str, dict] = {}
    
    def _sync_to_falkordb(self, node_id: str, result: dict):
        """Sync refinement to FalkorDB."""
        if not self.config.sync_to_falkordb:
            return
        
        try:
            falkordb = self.graph_builder.falkordb
            if hasattr(falkordb, 'graph') and falkordb.graph:
                query = """
                MATCH (n {node_id: $node_id})
                SET n.refined_description = $refined_description,
                    n.refined_activity_summary = $refined_activity_summary
                RETURN n.node_id
                """
                falkordb.graph.query(query, {
                    "node_id": node_id,
                    "refined_description": result["refined_description"],
                    "refined_activity_summary": result["refined_activity_summary"]
                })
            elif hasattr(falkordb, 'nodes') and node_id in falkordb.nodes:
                falkordb.nodes[node_id]["refined_description"] = result["refined_description"]
                falkordb.nodes[node_id]["refined_activity_summary"] = result["refined_activity_summary"]
        except Exception as e:
            if self.config.verbose:
                print(f"  ⚠ FalkorDB sync failed for {node_id}: {e}")
    
    def refine_node(self, node_id: str) -> dict:
        """Refine a single node. Updates NetworkX and FalkorDB."""
        data = self.graph.nodes[node_id]
        context = get_hierarchical_context(self.graph, node_id)
        
        prompt = AUGMENTED_REFINEMENT_PROMPT.format(
            name=data.get("name", node_id),
            code=node_id,
            level=data.get("level", 0),
            activities_full=format_activities_full(self.graph, node_id),
            parent_info=context["parent_info"],
            children_info=context["children_info"],
            siblings_info=context["siblings_info"]
        )
        
        response = self.llm.generate(prompt, SYSTEM_PROMPT)
        result = parse_llm_response(response)
        
        # Update NetworkX
        self.graph.nodes[node_id]["refined_description"] = result["refined_description"]
        self.graph.nodes[node_id]["refined_activity_summary"] = result["refined_activity_summary"]
        
        # Update FalkorDB
        self._sync_to_falkordb(node_id, result)
        
        return result
    
    def run_top_down(self, max_nodes: int = None) -> Dict[str, dict]:
        """Run top-down refinement."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("TOP-DOWN REFINEMENT (Root → Leaves)")
            print("═" * 70)
        
        roots = getattr(self.graph_builder, 'org_roots', None) or \
                [n for n in self.graph.nodes() if not get_parent(self.graph, n)]
        
        nodes = traverse_top_down(self.graph, roots)
        if max_nodes:
            nodes = nodes[:max_nodes]
        
        if self.config.verbose:
            print(f"Processing {len(nodes)} nodes...\n")
        
        results = {}
        for i, node_id in enumerate(nodes):
            data = self.graph.nodes[node_id]
            result = self.refine_node(node_id)
            results[node_id] = result
            
            if self.config.verbose:
                print(f"[{i+1:3}/{len(nodes)}] {node_id:10} {data.get('name', '')[:40]}")
                print(f"           → {result['refined_description'][:60]}...\n")
        
        if self.config.verbose:
            print(f"✓ Top-down complete: {len(results)} nodes")
        return results
    
    def run_bottom_up(self, max_nodes: int = None, clear_existing: bool = False) -> Dict[str, dict]:
        """Run bottom-up refinement."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("BOTTOM-UP REFINEMENT (Leaves → Root)")
            print("═" * 70)
        
        if clear_existing:
            for node_id in self.graph.nodes():
                self.graph.nodes[node_id]["refined_description"] = ""
                self.graph.nodes[node_id]["refined_activity_summary"] = ""
        
        nodes = traverse_bottom_up(self.graph)
        if max_nodes:
            nodes = nodes[:max_nodes]
        
        if self.config.verbose:
            print(f"Processing {len(nodes)} nodes...\n")
        
        results = {}
        for i, node_id in enumerate(nodes):
            data = self.graph.nodes[node_id]
            result = self.refine_node(node_id)
            results[node_id] = result
            
            if self.config.verbose:
                print(f"[{i+1:3}/{len(nodes)}] {node_id:10} {data.get('name', '')[:40]}")
                print(f"           → {result['refined_description'][:60]}...\n")
        
        if self.config.verbose:
            print(f"✓ Bottom-up complete: {len(results)} nodes")
        return results
    
    def run_bidirectional_merge(self, td_results: dict, bu_results: dict) -> int:
        """Merge top-down and bottom-up results."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("BIDIRECTIONAL MERGE")
            print("═" * 70)
        
        merged_count = 0
        for node_id in self.graph.nodes():
            td_data = self.td_snapshot.get(node_id, {})
            bu_data = bu_results.get(node_id, {})
            
            td_desc, td_summ = td_data.get("refined_description", ""), td_data.get("refined_activity_summary", "")
            bu_desc, bu_summ = bu_data.get("refined_description", ""), bu_data.get("refined_activity_summary", "")
            
            if td_desc and bu_desc and td_desc != bu_desc:
                prompt = MERGE_PROMPT.format(td_desc=td_desc, td_summary=td_summ, bu_desc=bu_desc, bu_summary=bu_summ)
                response = self.llm.generate(prompt, SYSTEM_PROMPT)
                result = parse_llm_response(response)
                
                self.graph.nodes[node_id]["refined_description"] = result["refined_description"]
                self.graph.nodes[node_id]["refined_activity_summary"] = result["refined_activity_summary"]
                self._sync_to_falkordb(node_id, result)
                merged_count += 1
                
                if self.config.verbose:
                    print(f"[MERGED] {node_id} {self.graph.nodes[node_id].get('name', '')[:35]}")
        
        if self.config.verbose:
            print(f"\n✓ Merged {merged_count} nodes")
        return merged_count
    
    def run(self) -> Dict[str, Any]:
        """Run refinement based on configured strategy."""
        max_nodes = self.config.max_nodes
        
        if self.config.strategy == "top_down":
            self.results = self.run_top_down(max_nodes)
        elif self.config.strategy == "bottom_up":
            self.results = self.run_bottom_up(max_nodes, clear_existing=True)
        elif self.config.strategy == "bidirectional":
            td_results = self.run_top_down(max_nodes)
            for node_id in self.graph.nodes():
                data = self.graph.nodes[node_id]
                self.td_snapshot[node_id] = {
                    "refined_description": data.get("refined_description", ""),
                    "refined_activity_summary": data.get("refined_activity_summary", "")
                }
            bu_results = self.run_bottom_up(max_nodes)
            self.run_bidirectional_merge(td_results, bu_results)
            self.results = bu_results
        
        refined_count = sum(1 for n in self.graph.nodes() if self.graph.nodes[n].get("refined_description"))
        summary_count = sum(1 for n in self.graph.nodes() if self.graph.nodes[n].get("refined_activity_summary"))
        
        return {
            "strategy": self.config.strategy,
            "total_nodes": self.graph.number_of_nodes(),
            "refined_descriptions": refined_count,
            "activity_summaries": summary_count,
            "synced_to_falkordb": self.config.sync_to_falkordb
        }
    
    def export_refinements(self, path: str) -> str:
        """Export refinements to JSON file."""
        refinements = {}
        for node_id in self.graph.nodes():
            data = self.graph.nodes[node_id]
            if data.get("refined_description"):
                refinements[node_id] = {
                    "name": data.get("name"),
                    "level": data.get("level"),
                    "original_activities": data.get("activities", []),
                    "original_weights": data.get("activity_weights", []),
                    "refined_description": data.get("refined_description"),
                    "refined_activity_summary": data.get("refined_activity_summary", "")
                }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(refinements, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Exported refinements: {path} ({len(refinements)} nodes)")
        return path


def create_refinement_agent(
    graph_builder, llm,
    strategy: str = "bidirectional",
    sync_to_falkordb: bool = True,
    verbose: bool = True,
    max_nodes: int = None
) -> RefinementAgent:
    """Create RefinementAgent instance."""
    config = RefinementConfig(strategy=strategy, sync_to_falkordb=sync_to_falkordb, verbose=verbose, max_nodes=max_nodes)
    return RefinementAgent(graph_builder, llm, config)


if __name__ == "__main__":
    print("RefinementAgent module loaded. Use create_refinement_agent() to instantiate.")
