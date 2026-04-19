"""
Capability Refinement Module
LLM-based description enrichment for business capabilities.

- Refines capability descriptions using hierarchical context
- Generates semantic keywords for capability matching
- Syncs to FalkorDB (shared org_hierarchy graph)
- Updates vector embeddings for semantic search
- Configurable: top_down, bottom_up, bidirectional strategies

Usage:
    from capability_refinement import create_capability_refinement_agent
    from capability_graph import create_capability_builder
    from llm import create_llm

    builder = create_capability_builder(falkordb_url=URL)
    llm = create_llm()
    agent = create_capability_refinement_agent(builder, llm)
    agent.run()
    agent.export_refinements("output/capability_refinements.json")
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import networkx as nx


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CapabilityRefinementConfig:
    """Configuration for capability refinement."""
    strategy: str = "top_down"              # top_down, bottom_up, bidirectional
    sync_to_falkordb: bool = True           # Sync results to capability FalkorDB graph
    update_embeddings: bool = True          # Recompute embeddings after refinement
    create_vector_index: bool = True        # Create/update vector index
    embedding_dimension: int = 384
    verbose: bool = True
    max_nodes: Optional[int] = None


# =============================================================================
# Prompts
# =============================================================================

CAPABILITY_SYSTEM_PROMPT = """You are an expert business capability analyst for the European Parliament.
Your task is to create AUGMENTED descriptions of business capabilities.

CRITICAL RULES:
1. PRESERVE ALL original capability details
2. ENHANCE clarity while maintaining detail level
3. EXPLAIN how capabilities relate to organizational functions
4. USE professional business architecture language (TOGAF aligned)
5. EXTRACT relevant keywords for semantic matching against organizational activities"""


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
   - Focus on actionable/functional terms that match organizational activities

FORMAT:
REFINED_DESCRIPTION:
[Your comprehensive capability description]

CAPABILITY_KEYWORDS:
[keyword1, keyword2, keyword3, ...]
"""

CAPABILITY_MERGE_PROMPT = """Merge these two perspectives into a comprehensive capability description.

══ TOP-DOWN PERSPECTIVE ══
{td_desc}

Keywords: {td_keywords}

══ BOTTOM-UP PERSPECTIVE ══
{bu_desc}

Keywords: {bu_keywords}

══ TASK ══
Create a FINAL merged description that integrates BOTH perspectives.

FORMAT:
REFINED_DESCRIPTION:
[Comprehensive 3-5 sentence description]

CAPABILITY_KEYWORDS:
[merged keyword list, 8-12 terms]
"""


# =============================================================================
# Helper Functions
# =============================================================================

def get_capability_context(graph: nx.DiGraph, node_id: str) -> dict:
    """Get hierarchical context for a capability node."""
    preds = list(graph.predecessors(node_id))
    parent_id = preds[0] if preds else None

    # Parent info
    if parent_id:
        p_data = graph.nodes[parent_id]
        parent_info = f"{p_data.get('name', parent_id)}"
        p_refined = p_data.get("refined_description", "")
        if p_refined:
            parent_info += f": {p_refined[:200]}..."
    else:
        parent_info = "ROOT CAPABILITY — Top-level category"

    # Children info
    children = list(graph.successors(node_id))
    if children:
        child_lines = []
        for c in children[:6]:
            c_data = graph.nodes[c]
            c_refined = c_data.get("refined_description", "")
            if c_refined:
                child_lines.append(f"  • {c_data.get('name', c)}: {c_refined[:80]}...")
            else:
                child_lines.append(f"  • {c_data.get('name', c)}")
        children_info = "\n".join(child_lines)
        if len(children) > 6:
            children_info += f"\n  • ... +{len(children) - 6} more"
    else:
        children_info = "LEAF CAPABILITY — No sub-capabilities"

    # Siblings info
    if parent_id:
        siblings = [s for s in graph.successors(parent_id) if s != node_id]
        if siblings:
            sib_names = [graph.nodes[s].get("name", s)[:30] for s in siblings[:5]]
            siblings_info = ", ".join(sib_names)
            if len(siblings) > 5:
                siblings_info += f", +{len(siblings) - 5} more"
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


# =============================================================================
# Capability Refinement Agent
# =============================================================================

class CapabilityRefinementAgent:
    """
    Agent for refining business capability descriptions.

    Works with CapabilityGraphBuilder for dual storage (NetworkX + FalkorDB).
    Supports top_down, bottom_up, and bidirectional strategies.
    """

    def __init__(self, capability_builder, llm, vector_store=None,
                 config: CapabilityRefinementConfig = None):
        self.builder = capability_builder
        self.graph = capability_builder.graph
        self.llm = llm
        self.vector_store = vector_store
        self.config = config or CapabilityRefinementConfig()
        self.results: Dict[str, dict] = {}
        self.td_snapshot: Dict[str, dict] = {}
        self._stats = defaultdict(int)

    # -----------------------------------------------------------------
    # FalkorDB Sync
    # -----------------------------------------------------------------

    def _sync_to_falkordb(self, node_id: str, result: dict):
        """Sync capability refinement to FalkorDB."""
        if not self.config.sync_to_falkordb:
            return
        if not self.builder._connected:
            return

        try:
            self.builder._fgraph.query(
                "MATCH (n {node_id: $nid}) "
                "SET n.refined_description = $rdesc, "
                "    n.capability_keywords = $kw "
                "RETURN n.node_id",
                {
                    "nid": node_id,
                    "rdesc": result.get("refined_description", ""),
                    "kw": result.get("capability_keywords", "")
                }
            )
            self._stats['synced_to_falkordb'] += 1
        except Exception as e:
            if self.config.verbose:
                print(f"  ⚠ FalkorDB sync failed for {node_id}: {e}")

    def _update_embedding(self, node_id: str, text: str):
        """Recompute and sync embedding for refined description."""
        if not self.config.update_embeddings or not self.vector_store:
            return

        try:
            embedding = self.vector_store.get_embedding(text)

            if self.config.sync_to_falkordb and self.builder._connected:
                self.builder._fgraph.query(
                    "MATCH (n {node_id: $nid}) "
                    "SET n.refined_embedding = vecf32($emb) "
                    "RETURN n.node_id",
                    {"nid": node_id, "emb": embedding}
                )
                self._stats['embeddings_updated'] += 1
        except Exception as e:
            if self.config.verbose:
                print(f"  ⚠ Embedding update failed for {node_id}: {e}")

    # -----------------------------------------------------------------
    # Refine Single Capability
    # -----------------------------------------------------------------

    def refine_capability(self, node_id: str) -> dict:
        """Refine a single capability node."""
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

        # Update local graph
        self.graph.nodes[node_id]["refined_description"] = result["refined_description"]
        self.graph.nodes[node_id]["capability_keywords"] = result["capability_keywords"]

        # Sync to FalkorDB
        self._sync_to_falkordb(node_id, result)

        # Update embedding
        self._update_embedding(node_id, result["refined_description"])

        self._stats['capabilities_refined'] += 1
        return result

    # -----------------------------------------------------------------
    # Strategy Runners
    # -----------------------------------------------------------------

    def run_top_down(self, max_nodes: int = None) -> Dict[str, dict]:
        """Refine capabilities from root categories to leaves."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("CAPABILITY TOP-DOWN REFINEMENT (Categories → Leaves)")
            print("═" * 70)

        nodes = self.builder.traverse_top_down()
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
                name = data.get('name', '')[:35]
                desc = result['refined_description'][:50]
                print(f"[{i+1:3}/{len(nodes)}] {node_id:20} {name}")
                print(f"           → {desc}...\n")

        if self.config.verbose:
            print(f"✓ Top-down complete: {len(results)} capabilities")
        return results

    def run_bottom_up(self, max_nodes: int = None,
                      clear_existing: bool = False) -> Dict[str, dict]:
        """Refine capabilities from leaves to root categories."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("CAPABILITY BOTTOM-UP REFINEMENT (Leaves → Categories)")
            print("═" * 70)

        if clear_existing:
            for node_id in self.graph.nodes():
                self.graph.nodes[node_id]["refined_description"] = ""
                self.graph.nodes[node_id]["capability_keywords"] = ""

        nodes = self.builder.traverse_bottom_up()
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
                name = data.get('name', '')[:35]
                desc = result['refined_description'][:50]
                print(f"[{i+1:3}/{len(nodes)}] {node_id:20} {name}")
                print(f"           → {desc}...\n")

        if self.config.verbose:
            print(f"✓ Bottom-up complete: {len(results)} capabilities")
        return results

    def run_bidirectional_merge(self, td_results: dict, bu_results: dict) -> int:
        """Merge top-down and bottom-up perspectives."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("CAPABILITY BIDIRECTIONAL MERGE")
            print("═" * 70)

        merged_count = 0
        for node_id in self.graph.nodes():
            td_data = self.td_snapshot.get(node_id, {})
            bu_data = bu_results.get(node_id, {})

            td_desc = td_data.get("refined_description", "")
            td_kw = td_data.get("capability_keywords", "")
            bu_desc = bu_data.get("refined_description", "")
            bu_kw = bu_data.get("capability_keywords", "")

            if td_desc and bu_desc and td_desc != bu_desc:
                prompt = CAPABILITY_MERGE_PROMPT.format(
                    td_desc=td_desc, td_keywords=td_kw,
                    bu_desc=bu_desc, bu_keywords=bu_kw
                )
                response = self.llm.generate(prompt, CAPABILITY_SYSTEM_PROMPT)
                result = parse_capability_response(response)

                self.graph.nodes[node_id]["refined_description"] = result["refined_description"]
                self.graph.nodes[node_id]["capability_keywords"] = result["capability_keywords"]
                self._sync_to_falkordb(node_id, result)
                self._update_embedding(node_id, result["refined_description"])
                merged_count += 1

                if self.config.verbose:
                    name = self.graph.nodes[node_id].get('name', '')[:35]
                    print(f"[MERGED] {node_id} {name}")

        if self.config.verbose:
            print(f"\n✓ Merged {merged_count} capabilities")
        return merged_count

    # -----------------------------------------------------------------
    # Vector Index
    # -----------------------------------------------------------------

    def setup_vector_indexes(self):
        """Create vector indexes for refined embeddings in capability graph."""
        if not self.config.create_vector_index:
            return
        if not self.builder._connected:
            if self.config.verbose:
                print("  ⚠ Not connected to FalkorDB")
            return

        if self.config.verbose:
            print("\n" + "═" * 70)
            print("CREATING CAPABILITY VECTOR INDEXES")
            print("═" * 70)

        dim = self.config.embedding_dimension

        # Refined embedding index (for semantic search after refinement)
        self.builder.create_vector_index(
            index_name="capability_refined_idx",
            property_name="refined_embedding",
            dimension=dim
        )

    # -----------------------------------------------------------------
    # Main Entry
    # -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Run full capability refinement pipeline."""
        self._stats = defaultdict(int)
        max_nodes = self.config.max_nodes

        if self.config.verbose:
            print("\n" + "=" * 70)
            print("CAPABILITY REFINEMENT AGENT")
            print("=" * 70)
            print(f"  Strategy:          {self.config.strategy}")
            print(f"  Sync to FalkorDB:  {self.config.sync_to_falkordb}")
            print(f"  Update embeddings: {self.config.update_embeddings}")
            print(f"  Vector indexes:    {self.config.create_vector_index}")

        # Set up vector indexes
        self.setup_vector_indexes()

        # Run strategy
        if self.config.strategy == "top_down":
            self.results = self.run_top_down(max_nodes)

        elif self.config.strategy == "bottom_up":
            self.results = self.run_bottom_up(max_nodes, clear_existing=True)

        elif self.config.strategy == "bidirectional":
            td_results = self.run_top_down(max_nodes)

            # Snapshot
            for node_id in self.graph.nodes():
                data = self.graph.nodes[node_id]
                self.td_snapshot[node_id] = {
                    "refined_description": data.get("refined_description", ""),
                    "capability_keywords": data.get("capability_keywords", "")
                }

            bu_results = self.run_bottom_up(max_nodes)
            self.run_bidirectional_merge(td_results, bu_results)
            self.results = bu_results

        # Summary
        refined_count = sum(1 for n in self.graph.nodes()
                            if self.graph.nodes[n].get("refined_description"))

        summary = {
            "strategy": self.config.strategy,
            "total_capabilities": self.graph.number_of_nodes(),
            "refined": refined_count,
            "synced_to_falkordb": self._stats.get('synced_to_falkordb', 0),
            "embeddings_updated": self._stats.get('embeddings_updated', 0)
        }

        if self.config.verbose:
            print("\n" + "=" * 70)
            print("CAPABILITY REFINEMENT COMPLETE")
            print("=" * 70)
            for k, v in summary.items():
                print(f"  {k}: {v}")

        return summary

    # -----------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------

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

        if self.config.verbose:
            print(f"✓ Exported: {path} ({len(refinements)} capabilities)")
        return path


# =============================================================================
# Factory
# =============================================================================

def create_capability_refinement_agent(
    capability_builder,
    llm,
    vector_store=None,
    strategy: str = "top_down",
    sync_to_falkordb: bool = True,
    update_embeddings: bool = True,
    create_vector_index: bool = True,
    embedding_dimension: int = 384,
    verbose: bool = True,
    max_nodes: int = None
) -> CapabilityRefinementAgent:
    """
    Create CapabilityRefinementAgent.

    Args:
        capability_builder: CapabilityGraphBuilder instance
        llm: LLM instance with .generate(prompt, system)
        vector_store: VectorStore for embedding computation (optional)
        strategy: "top_down", "bottom_up", or "bidirectional"
        sync_to_falkordb: Sync results to capability FalkorDB graph
        update_embeddings: Recompute embeddings after refinement
        create_vector_index: Create vector index in FalkorDB
        embedding_dimension: Embedding dimension (384 for MiniLM)
        verbose: Print progress
        max_nodes: Limit nodes (for testing)

    Returns:
        Configured CapabilityRefinementAgent
    """
    config = CapabilityRefinementConfig(
        strategy=strategy,
        sync_to_falkordb=sync_to_falkordb,
        update_embeddings=update_embeddings,
        create_vector_index=create_vector_index,
        embedding_dimension=embedding_dimension,
        verbose=verbose,
        max_nodes=max_nodes
    )
    return CapabilityRefinementAgent(capability_builder, llm, vector_store, config)
