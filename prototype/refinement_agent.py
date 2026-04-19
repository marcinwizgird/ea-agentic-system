"""
Refinement Agent Module
Augmented description refinement for organizational units and activities
with FalkorDB sync and vector index support.

Features:
- Refines org unit descriptions using hierarchical context
- Refines activity descriptions using parent org context
- Configurable strategy: top_down, bottom_up, bidirectional
- Syncs refined descriptions + embeddings to FalkorDB
- Creates/updates vector indexes for semantic search

Usage:
    from refinement_agent import create_refinement_agent
    from llm import create_llm

    llm = create_llm()
    agent = create_refinement_agent(
        graph=graph,
        graph_store=pipeline.graph_store,
        vector_store=pipeline.vector_store,
        llm=llm,
        strategy="bidirectional",
        verbose=True
    )
    result = agent.run()
    agent.export_refinements("output/refinements.json")
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import networkx as nx


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RefinementConfig:
    """Configuration for refinement agent."""
    strategy: str = "bidirectional"       # top_down, bottom_up, bidirectional
    refine_org_units: bool = True         # Refine org unit nodes
    refine_activities: bool = True        # Refine activity nodes
    sync_to_falkordb: bool = True         # Sync results to FalkorDB
    update_embeddings: bool = True        # Recompute embeddings after refinement
    create_vector_index: bool = True      # Create/update vector indexes
    embedding_dimension: int = 384        # Embedding vector dimension
    verbose: bool = True
    max_nodes: Optional[int] = None       # Limit nodes to process (for testing)


# =============================================================================
# LLM Prompts
# =============================================================================

SYSTEM_PROMPT = """You are an expert organizational analyst for the European Parliament.
Your task is to create AUGMENTED descriptions that PRESERVE ALL original details.

CRITICAL RULES:
1. NEVER reduce or summarize away any activity details
2. ENHANCE clarity while MAINTAINING the same or greater level of detail
3. ADD context about organizational relationships
4. USE professional, precise language
5. INCLUDE all percentage allocations in your response"""


ORG_UNIT_REFINEMENT_PROMPT = """Analyze this organizational unit and create an augmented description.

═══════════════════════════════════════════════════════════════════════════════
UNIT INFORMATION
═══════════════════════════════════════════════════════════════════════════════
Unit Name: {name}
Unit Code: {code}
Hierarchy Level: {level}

═══════════════════════════════════════════════════════════════════════════════
ACTIVITIES ASSIGNED TO THIS UNIT
═══════════════════════════════════════════════════════════════════════════════
{activities_detail}

═══════════════════════════════════════════════════════════════════════════════
ORGANIZATIONAL CONTEXT
═══════════════════════════════════════════════════════════════════════════════
Parent Unit: {parent_info}
Child Units: {children_info}
Sibling Units: {siblings_info}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK — Generate TWO outputs:
═══════════════════════════════════════════════════════════════════════════════

1. REFINED_DESCRIPTION (4-6 sentences):
   - Start with the unit's primary mission/role
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
[Your comprehensive augmented description here]

REFINED_ACTIVITY_SUMMARY:
• [Theme 1] (X-Y%): [grouped activities]
• [Theme 2] (X-Y%): [grouped activities]
• [Theme 3] (X-Y%): [grouped activities]
"""


ACTIVITY_REFINEMENT_PROMPT = """Refine this activity description for semantic matching.

═══════════════════════════════════════════════════════════════════════════════
ACTIVITY
═══════════════════════════════════════════════════════════════════════════════
Activity ID: {activity_id}
Weight: {weight}% of parent unit workload
Parent Org Unit: {parent_org_name} ({parent_org_code})

Original Description:
{description}

═══════════════════════════════════════════════════════════════════════════════
PARENT UNIT CONTEXT
═══════════════════════════════════════════════════════════════════════════════
{parent_context}

═══════════════════════════════════════════════════════════════════════════════
SIBLING ACTIVITIES (same org unit)
═══════════════════════════════════════════════════════════════════════════════
{sibling_activities}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK — Generate TWO outputs:
═══════════════════════════════════════════════════════════════════════════════

1. REFINED_DESCRIPTION (2-4 sentences):
   - Clarify what this activity involves operationally
   - Explain its role within the parent org unit
   - Use precise terminology for semantic matching against IT capabilities

2. ACTIVITY_KEYWORDS (comma-separated):
   - Extract 5-10 key terms for capability matching
   - Include functional verbs, domain terms, and synonyms

FORMAT:
REFINED_DESCRIPTION:
[Your refined activity description]

ACTIVITY_KEYWORDS:
[keyword1, keyword2, keyword3, ...]
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


# =============================================================================
# Helper Functions
# =============================================================================

def get_parent(graph: nx.DiGraph, node_id: str) -> Optional[str]:
    """Get parent node ID."""
    preds = list(graph.predecessors(node_id))
    # Filter out non-org parents (activity nodes don't have org parents as predecessors)
    for p in preds:
        if graph.nodes[p].get('type') != 'activity':
            return p
    return preds[0] if preds else None


def get_children(graph: nx.DiGraph, node_id: str, include_activities: bool = False) -> List[str]:
    """Get child node IDs, optionally including activities."""
    children = list(graph.successors(node_id))
    if not include_activities:
        children = [c for c in children if graph.nodes[c].get('type') != 'activity']
    return children


def get_activities(graph: nx.DiGraph, node_id: str) -> List[str]:
    """Get activity node IDs attached to an org unit."""
    return [c for c in graph.successors(node_id)
            if graph.nodes[c].get('type') == 'activity']


def get_org_units(graph: nx.DiGraph) -> List[str]:
    """Get all org unit node IDs (excluding activities)."""
    return [n for n, d in graph.nodes(data=True) if d.get('type') == 'org_unit']


def get_activity_nodes(graph: nx.DiGraph) -> List[str]:
    """Get all activity node IDs."""
    return [n for n, d in graph.nodes(data=True) if d.get('type') == 'activity']


def format_activities_for_org(graph: nx.DiGraph, org_id: str) -> str:
    """Format all activities of an org unit with full details."""
    activity_ids = get_activities(graph, org_id)
    if not activity_ids:
        return "No activities documented for this unit."

    lines = []
    total_weight = 0
    for i, act_id in enumerate(sorted(activity_ids), 1):
        data = graph.nodes[act_id]
        weight = data.get('weight_pct', 0)
        desc = data.get('description', '')
        total_weight += weight
        lines.append(f"Activity {i} [{act_id}] — {weight}% of workload:")
        lines.append(f"  {desc}")
        lines.append("")

    lines.append(f"Total documented workload: {total_weight}%")
    return "\n".join(lines)


def get_hierarchical_context(graph: nx.DiGraph, node_id: str) -> dict:
    """Get complete hierarchical context for an org unit node."""
    parent_id = get_parent(graph, node_id)

    # Parent info
    if parent_id:
        p_data = graph.nodes[parent_id]
        parent_info = f"{p_data.get('name', parent_id)}"
        p_refined = p_data.get("refined_description", "")
        if p_refined:
            parent_info += f"\n  Context: {p_refined[:200]}..."
    else:
        parent_info = "ROOT LEVEL — Reports directly to Secretary General"

    # Children info (org units only)
    children = get_children(graph, node_id, include_activities=False)
    if children:
        child_lines = []
        for c_id in children[:6]:
            c_data = graph.nodes[c_id]
            c_refined = c_data.get("refined_description", "")
            act_count = len(get_activities(graph, c_id))
            if c_refined:
                child_lines.append(f"  • {c_data.get('name', c_id)}: {c_refined[:80]}...")
            else:
                child_lines.append(f"  • {c_data.get('name', c_id)} ({act_count} activities)")
        children_info = "\n".join(child_lines)
        if len(children) > 6:
            children_info += f"\n  • ... and {len(children) - 6} more units"
    else:
        children_info = "LEAF UNIT — No subordinate organizational units"

    # Siblings info
    if parent_id:
        siblings = [s for s in get_children(graph, parent_id, include_activities=False)
                     if s != node_id]
        if siblings:
            sib_names = [graph.nodes[s].get("name", s)[:30] for s in siblings[:5]]
            siblings_info = ", ".join(sib_names)
            if len(siblings) > 5:
                siblings_info += f", +{len(siblings) - 5} more"
        else:
            siblings_info = "Only unit under parent"
    else:
        siblings_info = "N/A (root level)"

    return {
        "parent_info": parent_info,
        "children_info": children_info,
        "siblings_info": siblings_info
    }


def get_activity_context(graph: nx.DiGraph, activity_id: str) -> dict:
    """Get context for an activity node."""
    data = graph.nodes[activity_id]
    parent_org = data.get('parent_org', '')

    # Parent org context
    if parent_org and parent_org in graph.nodes:
        p_data = graph.nodes[parent_org]
        parent_context = f"{p_data.get('name', parent_org)}"
        p_refined = p_data.get("refined_description", "")
        if p_refined:
            parent_context += f"\n{p_refined[:300]}"
    else:
        parent_context = "Unknown parent organization"

    # Sibling activities
    sibling_acts = []
    if parent_org and parent_org in graph.nodes:
        for act_id in get_activities(graph, parent_org):
            if act_id != activity_id:
                act_data = graph.nodes[act_id]
                weight = act_data.get('weight_pct', 0)
                desc = act_data.get('description', '')[:100]
                sibling_acts.append(f"  • [{act_id}] ({weight}%) {desc}...")

    sibling_text = "\n".join(sibling_acts) if sibling_acts else "No other activities in this unit"

    return {
        "parent_org_name": graph.nodes.get(parent_org, {}).get('name', parent_org) if parent_org else "Unknown",
        "parent_org_code": parent_org,
        "parent_context": parent_context,
        "sibling_activities": sibling_text
    }


def parse_org_response(response: str) -> dict:
    """Parse LLM response for org unit refinement."""
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


def parse_activity_response(response: str) -> dict:
    """Parse LLM response for activity refinement."""
    result = {"refined_description": "", "activity_keywords": ""}

    if "REFINED_DESCRIPTION:" in response:
        parts = response.split("REFINED_DESCRIPTION:", 1)
        if len(parts) > 1:
            desc = parts[1]
            if "ACTIVITY_KEYWORDS:" in desc:
                desc = desc.split("ACTIVITY_KEYWORDS:", 1)[0]
            result["refined_description"] = desc.strip()

    if "ACTIVITY_KEYWORDS:" in response:
        parts = response.split("ACTIVITY_KEYWORDS:", 1)
        if len(parts) > 1:
            result["activity_keywords"] = parts[1].strip()

    if not result["refined_description"]:
        result["refined_description"] = response.strip()

    return result


# =============================================================================
# Traversal Functions
# =============================================================================

def traverse_top_down(graph: nx.DiGraph) -> List[str]:
    """BFS traversal of org units from roots to leaves."""
    roots = [n for n in get_org_units(graph)
             if get_parent(graph, n) is None or
             graph.nodes.get(get_parent(graph, n), {}).get('type') != 'org_unit']
    visited, queue = [], list(roots)
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            queue.extend(get_children(graph, node, include_activities=False))
    return visited


def traverse_bottom_up(graph: nx.DiGraph) -> List[str]:
    """Traverse org units from leaves to roots by depth."""
    org_units = get_org_units(graph)

    # Compute depth from root for each org unit
    depths = {}
    for n in org_units:
        depth = 0
        current = n
        while True:
            parent = get_parent(graph, current)
            if parent is None or parent not in graph.nodes or graph.nodes[parent].get('type') != 'org_unit':
                break
            depth += 1
            current = parent
        depths[n] = depth

    # Sort by depth descending (leaves first)
    return sorted(org_units, key=lambda n: (-depths.get(n, 0), n))


# =============================================================================
# Refinement Agent
# =============================================================================

class RefinementAgent:
    """
    Agent for refining organizational unit and activity descriptions.

    Supports three strategies:
    - top_down: Refine from root to leaves (parent context available)
    - bottom_up: Refine from leaves to root (child context available)
    - bidirectional: Top-down pass, then bottom-up pass, then merge

    Syncs results to FalkorDB and updates vector embeddings.
    """

    def __init__(self, graph: nx.DiGraph, graph_store, vector_store,
                 llm, config: RefinementConfig = None):
        self.graph = graph
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.llm = llm
        self.config = config or RefinementConfig()
        self.results: Dict[str, dict] = {}
        self.td_snapshot: Dict[str, dict] = {}
        self._stats = defaultdict(int)

    # -----------------------------------------------------------------
    # FalkorDB Sync
    # -----------------------------------------------------------------

    def _sync_org_to_falkordb(self, node_id: str, result: dict):
        """Sync org unit refinement to FalkorDB."""
        if not self.config.sync_to_falkordb or not self.graph_store:
            return

        try:
            if hasattr(self.graph_store, '_graph') and self.graph_store._connected:
                query = """
                MATCH (n {node_id: $node_id})
                SET n.refined_description = $refined_description,
                    n.refined_activity_summary = $refined_activity_summary
                RETURN n.node_id
                """
                self.graph_store._graph.query(query, {
                    "node_id": node_id,
                    "refined_description": result.get("refined_description", ""),
                    "refined_activity_summary": result.get("refined_activity_summary", "")
                })
                self._stats['synced_to_falkordb'] += 1
        except Exception as e:
            if self.config.verbose:
                print(f"  ⚠ FalkorDB sync failed for {node_id}: {e}")

    def _sync_activity_to_falkordb(self, activity_id: str, result: dict):
        """Sync activity refinement to FalkorDB."""
        if not self.config.sync_to_falkordb or not self.graph_store:
            return

        try:
            if hasattr(self.graph_store, '_graph') and self.graph_store._connected:
                query = """
                MATCH (n {node_id: $node_id})
                SET n.refined_description = $refined_description,
                    n.activity_keywords = $activity_keywords
                RETURN n.node_id
                """
                self.graph_store._graph.query(query, {
                    "node_id": activity_id,
                    "refined_description": result.get("refined_description", ""),
                    "activity_keywords": result.get("activity_keywords", "")
                })
                self._stats['synced_to_falkordb'] += 1
        except Exception as e:
            if self.config.verbose:
                print(f"  ⚠ FalkorDB sync failed for {activity_id}: {e}")

    def _update_embedding(self, node_id: str, text: str):
        """Recompute and sync embedding for a node after refinement."""
        if not self.config.update_embeddings or not self.vector_store:
            return

        try:
            embedding = self.vector_store.get_embedding(text)

            # Update in FalkorDB
            if self.config.sync_to_falkordb and hasattr(self.graph_store, '_graph') and self.graph_store._connected:
                query = """
                MATCH (n {node_id: $node_id})
                SET n.refined_embedding = vecf32($embedding)
                RETURN n.node_id
                """
                self.graph_store._graph.query(query, {
                    "node_id": node_id,
                    "embedding": embedding
                })
                self._stats['embeddings_updated'] += 1
        except Exception as e:
            if self.config.verbose:
                print(f"  ⚠ Embedding update failed for {node_id}: {e}")

    # -----------------------------------------------------------------
    # Refine Single Nodes
    # -----------------------------------------------------------------

    def refine_org_unit(self, node_id: str) -> dict:
        """Refine a single org unit node."""
        data = self.graph.nodes[node_id]
        context = get_hierarchical_context(self.graph, node_id)

        prompt = ORG_UNIT_REFINEMENT_PROMPT.format(
            name=data.get("name", node_id),
            code=node_id,
            level=data.get("level", 0),
            activities_detail=format_activities_for_org(self.graph, node_id),
            parent_info=context["parent_info"],
            children_info=context["children_info"],
            siblings_info=context["siblings_info"]
        )

        response = self.llm.generate(prompt, SYSTEM_PROMPT)
        result = parse_org_response(response)

        # Update local graph
        self.graph.nodes[node_id]["refined_description"] = result["refined_description"]
        self.graph.nodes[node_id]["refined_activity_summary"] = result["refined_activity_summary"]

        # Sync to FalkorDB
        self._sync_org_to_falkordb(node_id, result)

        # Update embedding based on refined description
        self._update_embedding(node_id, result["refined_description"])

        self._stats['org_units_refined'] += 1
        return result

    def refine_activity(self, activity_id: str) -> dict:
        """Refine a single activity node."""
        data = self.graph.nodes[activity_id]
        context = get_activity_context(self.graph, activity_id)

        prompt = ACTIVITY_REFINEMENT_PROMPT.format(
            activity_id=activity_id,
            weight=data.get('weight_pct', 0),
            description=data.get('description', ''),
            parent_org_name=context["parent_org_name"],
            parent_org_code=context["parent_org_code"],
            parent_context=context["parent_context"],
            sibling_activities=context["sibling_activities"]
        )

        response = self.llm.generate(prompt, SYSTEM_PROMPT)
        result = parse_activity_response(response)

        # Update local graph
        self.graph.nodes[activity_id]["refined_description"] = result["refined_description"]
        self.graph.nodes[activity_id]["activity_keywords"] = result["activity_keywords"]

        # Sync to FalkorDB
        self._sync_activity_to_falkordb(activity_id, result)

        # Update embedding
        self._update_embedding(activity_id, result["refined_description"])

        self._stats['activities_refined'] += 1
        return result

    # -----------------------------------------------------------------
    # Strategy Runners
    # -----------------------------------------------------------------

    def run_top_down(self, max_nodes: int = None) -> Dict[str, dict]:
        """Run top-down refinement on org units."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("TOP-DOWN REFINEMENT (Root → Leaves)")
            print("═" * 70)

        nodes = traverse_top_down(self.graph)
        if max_nodes:
            nodes = nodes[:max_nodes]

        if self.config.verbose:
            print(f"Processing {len(nodes)} org units...\n")

        results = {}
        for i, node_id in enumerate(nodes):
            data = self.graph.nodes[node_id]
            result = self.refine_org_unit(node_id)
            results[node_id] = result

            if self.config.verbose:
                name = data.get('name', '')[:40]
                desc = result['refined_description'][:60]
                print(f"[{i+1:3}/{len(nodes)}] {node_id:10} {name}")
                print(f"           → {desc}...\n")

        if self.config.verbose:
            print(f"✓ Top-down complete: {len(results)} org units refined")
        return results

    def run_bottom_up(self, max_nodes: int = None,
                       clear_existing: bool = False) -> Dict[str, dict]:
        """Run bottom-up refinement on org units."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("BOTTOM-UP REFINEMENT (Leaves → Root)")
            print("═" * 70)

        if clear_existing:
            for node_id in get_org_units(self.graph):
                self.graph.nodes[node_id]["refined_description"] = ""
                self.graph.nodes[node_id]["refined_activity_summary"] = ""

        nodes = traverse_bottom_up(self.graph)
        if max_nodes:
            nodes = nodes[:max_nodes]

        if self.config.verbose:
            print(f"Processing {len(nodes)} org units...\n")

        results = {}
        for i, node_id in enumerate(nodes):
            data = self.graph.nodes[node_id]
            result = self.refine_org_unit(node_id)
            results[node_id] = result

            if self.config.verbose:
                name = data.get('name', '')[:40]
                desc = result['refined_description'][:60]
                print(f"[{i+1:3}/{len(nodes)}] {node_id:10} {name}")
                print(f"           → {desc}...\n")

        if self.config.verbose:
            print(f"✓ Bottom-up complete: {len(results)} org units refined")
        return results

    def run_bidirectional_merge(self, td_results: dict, bu_results: dict) -> int:
        """Merge top-down and bottom-up refinement results."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("BIDIRECTIONAL MERGE")
            print("═" * 70)

        merged_count = 0
        for node_id in get_org_units(self.graph):
            td_data = self.td_snapshot.get(node_id, {})
            bu_data = bu_results.get(node_id, {})

            td_desc = td_data.get("refined_description", "")
            td_summ = td_data.get("refined_activity_summary", "")
            bu_desc = bu_data.get("refined_description", "")
            bu_summ = bu_data.get("refined_activity_summary", "")

            if td_desc and bu_desc and td_desc != bu_desc:
                prompt = MERGE_PROMPT.format(
                    td_desc=td_desc, td_summary=td_summ,
                    bu_desc=bu_desc, bu_summary=bu_summ
                )
                response = self.llm.generate(prompt, SYSTEM_PROMPT)
                result = parse_org_response(response)

                self.graph.nodes[node_id]["refined_description"] = result["refined_description"]
                self.graph.nodes[node_id]["refined_activity_summary"] = result["refined_activity_summary"]
                self._sync_org_to_falkordb(node_id, result)
                self._update_embedding(node_id, result["refined_description"])
                merged_count += 1

                if self.config.verbose:
                    name = self.graph.nodes[node_id].get('name', '')[:35]
                    print(f"[MERGED] {node_id} {name}")

        if self.config.verbose:
            print(f"\n✓ Merged {merged_count} org units")
        return merged_count

    def refine_all_activities(self, max_nodes: int = None) -> Dict[str, dict]:
        """Refine all activity nodes."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("ACTIVITY REFINEMENT")
            print("═" * 70)

        activity_ids = sorted(get_activity_nodes(self.graph))
        if max_nodes:
            activity_ids = activity_ids[:max_nodes]

        if self.config.verbose:
            print(f"Processing {len(activity_ids)} activities...\n")

        results = {}
        for i, act_id in enumerate(activity_ids):
            data = self.graph.nodes[act_id]
            result = self.refine_activity(act_id)
            results[act_id] = result

            if self.config.verbose:
                desc = result['refined_description'][:60]
                weight = data.get('weight_pct', 0)
                parent = data.get('parent_org', '?')
                print(f"[{i+1:3}/{len(activity_ids)}] {act_id:20} ({parent}, {weight}%)")
                print(f"           → {desc}...\n")

        if self.config.verbose:
            print(f"✓ Activity refinement complete: {len(results)} activities refined")
        return results

    # -----------------------------------------------------------------
    # Vector Index Management
    # -----------------------------------------------------------------

    def setup_vector_indexes(self):
        """Create vector indexes in FalkorDB for refined embeddings."""
        if not self.config.create_vector_index:
            return

        if not hasattr(self.graph_store, 'create_vector_index'):
            if self.config.verbose:
                print("  ⚠ Graph store does not support vector indexes")
            return

        if self.config.verbose:
            print("\n" + "═" * 70)
            print("CREATING VECTOR INDEXES")
            print("═" * 70)

        dim = self.config.embedding_dimension

        # Index for org unit refined embeddings
        self.graph_store.create_vector_index(
            index_name="org_refined_embedding_idx",
            node_label="OrganizationalUnit",
            property_name="refined_embedding",
            dimension=dim
        )

        # Index for DG refined embeddings
        self.graph_store.create_vector_index(
            index_name="dg_refined_embedding_idx",
            node_label="DG",
            property_name="refined_embedding",
            dimension=dim
        )

        # Index for activity refined embeddings
        self.graph_store.create_vector_index(
            index_name="activity_refined_embedding_idx",
            node_label="Activity",
            property_name="refined_embedding",
            dimension=dim
        )

        if self.config.verbose:
            print("✓ Vector indexes created for OrganizationalUnit, DG, and Activity")

    # -----------------------------------------------------------------
    # Main Entry Point
    # -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Run the full refinement pipeline.

        1. Set up vector indexes in FalkorDB
        2. Refine org units (using configured strategy)
        3. Refine activities
        4. Return summary statistics
        """
        self._stats = defaultdict(int)
        max_nodes = self.config.max_nodes

        if self.config.verbose:
            print("\n" + "=" * 70)
            print("REFINEMENT AGENT")
            print("=" * 70)
            print(f"  Strategy:          {self.config.strategy}")
            print(f"  Refine org units:  {self.config.refine_org_units}")
            print(f"  Refine activities: {self.config.refine_activities}")
            print(f"  Sync to FalkorDB:  {self.config.sync_to_falkordb}")
            print(f"  Update embeddings: {self.config.update_embeddings}")
            print(f"  Vector indexes:    {self.config.create_vector_index}")

        # Step 1: Set up vector indexes
        self.setup_vector_indexes()

        # Step 2: Refine org units
        if self.config.refine_org_units:
            if self.config.strategy == "top_down":
                self.results = self.run_top_down(max_nodes)

            elif self.config.strategy == "bottom_up":
                self.results = self.run_bottom_up(max_nodes, clear_existing=True)

            elif self.config.strategy == "bidirectional":
                td_results = self.run_top_down(max_nodes)

                # Snapshot top-down results before bottom-up overwrites
                for node_id in get_org_units(self.graph):
                    data = self.graph.nodes[node_id]
                    self.td_snapshot[node_id] = {
                        "refined_description": data.get("refined_description", ""),
                        "refined_activity_summary": data.get("refined_activity_summary", "")
                    }

                bu_results = self.run_bottom_up(max_nodes)
                self.run_bidirectional_merge(td_results, bu_results)
                self.results = bu_results

        # Step 3: Refine activities
        if self.config.refine_activities:
            activity_results = self.refine_all_activities(max_nodes)
            self.results.update(activity_results)

        # Summary
        org_refined = sum(1 for n in get_org_units(self.graph)
                          if self.graph.nodes[n].get("refined_description"))
        act_refined = sum(1 for n in get_activity_nodes(self.graph)
                          if self.graph.nodes[n].get("refined_description"))

        summary = {
            "strategy": self.config.strategy,
            "total_org_units": len(get_org_units(self.graph)),
            "total_activities": len(get_activity_nodes(self.graph)),
            "org_units_refined": org_refined,
            "activities_refined": act_refined,
            "synced_to_falkordb": self._stats.get('synced_to_falkordb', 0),
            "embeddings_updated": self._stats.get('embeddings_updated', 0)
        }

        if self.config.verbose:
            print("\n" + "=" * 70)
            print("REFINEMENT COMPLETE")
            print("=" * 70)
            for k, v in summary.items():
                print(f"  {k}: {v}")

        return summary

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

    def export_refinements(self, path: str) -> str:
        """Export all refinements to JSON."""
        refinements = {"org_units": {}, "activities": {}}

        for node_id in get_org_units(self.graph):
            data = self.graph.nodes[node_id]
            if data.get("refined_description"):
                refinements["org_units"][node_id] = {
                    "name": data.get("name"),
                    "refined_description": data.get("refined_description"),
                    "refined_activity_summary": data.get("refined_activity_summary", "")
                }

        for act_id in get_activity_nodes(self.graph):
            data = self.graph.nodes[act_id]
            if data.get("refined_description"):
                refinements["activities"][act_id] = {
                    "parent_org": data.get("parent_org"),
                    "original_description": data.get("description", ""),
                    "refined_description": data.get("refined_description"),
                    "activity_keywords": data.get("activity_keywords", ""),
                    "weight_pct": data.get("weight_pct", 0)
                }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(refinements, f, indent=2, ensure_ascii=False)

        org_count = len(refinements["org_units"])
        act_count = len(refinements["activities"])
        print(f"✓ Exported refinements: {path} ({org_count} org units, {act_count} activities)")
        return path


# =============================================================================
# Factory Function
# =============================================================================

def create_refinement_agent(
    graph: nx.DiGraph,
    graph_store,
    vector_store,
    llm,
    strategy: str = "bidirectional",
    refine_org_units: bool = True,
    refine_activities: bool = True,
    sync_to_falkordb: bool = True,
    update_embeddings: bool = True,
    create_vector_index: bool = True,
    embedding_dimension: int = 384,
    verbose: bool = True,
    max_nodes: int = None
) -> RefinementAgent:
    """
    Create a RefinementAgent instance.

    Args:
        graph: NetworkX DiGraph with org units and activity nodes
        graph_store: FalkorDBGraphStore (or NetworkXGraphStore) instance
        vector_store: VectorStore instance for embedding computation
        llm: LLM instance with .generate(prompt, system) method
        strategy: "top_down", "bottom_up", or "bidirectional"
        refine_org_units: Whether to refine org unit nodes
        refine_activities: Whether to refine activity nodes
        sync_to_falkordb: Sync results to FalkorDB
        update_embeddings: Recompute embeddings after refinement
        create_vector_index: Create vector indexes in FalkorDB
        embedding_dimension: Embedding vector dimension (default 384 for MiniLM)
        verbose: Print progress
        max_nodes: Limit number of nodes (for testing)

    Returns:
        Configured RefinementAgent instance
    """
    config = RefinementConfig(
        strategy=strategy,
        refine_org_units=refine_org_units,
        refine_activities=refine_activities,
        sync_to_falkordb=sync_to_falkordb,
        update_embeddings=update_embeddings,
        create_vector_index=create_vector_index,
        embedding_dimension=embedding_dimension,
        verbose=verbose,
        max_nodes=max_nodes
    )
    return RefinementAgent(graph, graph_store, vector_store, llm, config)


if __name__ == "__main__":
    print("RefinementAgent module loaded.")
    print("Usage: create_refinement_agent(graph, graph_store, vector_store, llm)")
