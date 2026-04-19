"""
Bipartite Capability Matcher
Maps **activity leaf nodes** of organizational units to business capabilities
at a configurable level of the capability tree.

The matching subject is an individual :Activity node (not its parent org
unit). Each activity describes one action performed by the owning org unit;
each activity is evaluated and linked to its best-matching capability.

Traversal strategy:
    1. Walk the org hierarchy and collect every Activity leaf node whose
       parent org unit sits at or above `max_org_level`.
    2. For each activity, match it against capabilities at
       `target_capability_level` (default: the leaves of the capability
       tree — typically L2 Sub-Business Areas).
    3. If no match at the target level, fall back to higher capability
       levels (target-1 → target-2 → …).
    4. Store :MAPS_TO_CAPABILITY edges (Activity → Capability) in the
       shared FalkorDB graph.

Matching modes:
    - hybrid:         Semantic pre-filter + LLM judge (default, balanced)
    - llm_only:       Pure LLM matching (most accurate, expensive)
    - llm_prescreened: LLM category filter + LLM detailed matching
    - semantic_only:  Embedding similarity only (fastest, least accurate)

Usage:
    from bipartite_matcher import create_bipartite_matcher
    matcher = create_bipartite_matcher(
        falkordb_url=FALKORDB_URL,
        llm=llm,
        max_org_level=4,
        target_capability_level=2,   # L2 Sub-Business Areas
        matching_mode="hybrid",
    )
    matcher.load_graphs_from_falkordb()
    results = matcher.run()
    matcher.save_results("./results/bipartite_matches.json")
"""

import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MatcherConfig:
    """Configuration for bipartite activity-to-capability matching."""

    # --- Matching mode ---
    matching_mode: str = "hybrid"  # hybrid | llm_only | llm_prescreened | semantic_only

    # --- Org hierarchy scoping ---
    # Only activities whose PARENT org unit has level <= max_org_level are
    # matched. 99 effectively means "all activities".
    max_org_level: int = 99

    # --- Capability target level ---
    #   0 = Category, 1 = Business Area, 2 = Sub-Business Area (leaf)
    #   None = auto-detect the leaves of the capability tree.
    target_capability_level: Optional[int] = None

    # --- Semantic / hybrid settings ---
    min_semantic_score: float = 0.3
    top_k_candidates: int = 5
    use_hybrid_search: bool = True

    # --- LLM settings ---
    min_llm_score: float = 0.6
    use_llm_judge: bool = True
    llm_batch_size: int = 10

    # --- Hierarchical capability fallback ---
    enable_hierarchical_fallback: bool = True
    max_fallback_levels: int = 2

    # --- Multi-match policy (one activity → N capabilities) ---
    # Hard cap on how many capabilities may be attached to a single activity.
    max_matches_per_activity: int = 3
    # Minimum number of passing matches required before the activity counts as
    # "matched". If fewer pass, the activity is flagged unmatched and is
    # eligible for hierarchical fallback.
    min_matches_per_activity: int = 1
    # Keep secondary matches whose combined_score is within this delta of the
    # best match's combined_score (in addition to the always-kept top match).
    # Use 1.0 to effectively disable the tie-window.
    keep_within_delta: float = 0.1
    # Optional higher bar for the primary match. If the top match's
    # combined_score is below this, the activity is marked unmatched.
    # 0.0 disables this extra gate.
    min_primary_score: float = 0.0
    # If True, only keep secondary matches sharing the same L0 category
    # ancestor as the primary match. If False, allow cross-category matches.
    restrict_to_primary_category: bool = False

    # --- FalkorDB ---
    falkordb_url: Optional[str] = None
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: Optional[str] = None
    falkordb_graph_name: str = "org_hierarchy"
    sync_to_falkordb: bool = True

    # --- Output ---
    verbose: bool = True


# =============================================================================
# Embedders
# =============================================================================

class MockEmbedder:
    """Deterministic hash-based embedder for testing."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._cache: Dict[str, List[float]] = {}

    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        h = hashlib.sha256(text.encode()).digest()
        emb = [(h[i % len(h)] / 255.0) * 2 - 1 for i in range(self.dimension)]
        norm = np.sqrt(sum(x * x for x in emb))
        if norm > 0:
            emb = [x / norm for x in emb]
        self._cache[text] = emb
        return emb

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class HuggingFaceEmbedder:
    """Real embedder using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        print(f"✓ Embedder: {model_name}")

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()


# =============================================================================
# LLM Prompts
# =============================================================================

LLM_JUDGE_SYSTEM = (
    "You are an expert organizational analyst evaluating the match between "
    "an individual activity performed by a European Parliament organizational "
    "unit and a business capability. Focus on the activity itself; the owning "
    "org unit is provided only as context."
)

LLM_JUDGE_PROMPT = """Evaluate the match between this activity and the business capability.

═══════════════════════════════════════════════════════════════════════════════
ACTIVITY
═══════════════════════════════════════════════════════════════════════════════
Activity: {activity_name}
Description: {activity_text}
Weight: {activity_weight}

Owning org unit: {parent_org_name} (L{parent_org_level})
Org hierarchy path: {parent_org_path}

═══════════════════════════════════════════════════════════════════════════════
BUSINESS CAPABILITY
═══════════════════════════════════════════════════════════════════════════════
Name: {cap_name}
ID: {cap_id}
Level: {cap_level} ({cap_type})
Capability path: {cap_path}

Description:
{cap_description}

Keywords: {cap_keywords}

═══════════════════════════════════════════════════════════════════════════════
EVALUATION
═══════════════════════════════════════════════════════════════════════════════
Rate the match between the activity and the capability.

Output format:
MATCH_SCORE: [0.0 to 1.0]
MATCH_TYPE: [STRONG|MODERATE|WEAK|NO_MATCH]
JUSTIFICATION: [2-3 sentences — be specific about how the activity supports the capability]
KEY_OVERLAPS: [specific functional overlaps]
GAPS: [any significant gaps or misalignments]
"""

LLM_BATCH_RANKING_SYSTEM = (
    "You are an expert organizational analyst for the European Parliament. "
    "Your task is to identify the BEST matching business capabilities for a "
    "single organizational activity."
)

LLM_BATCH_RANKING_PROMPT = """Find the best matching capabilities for this activity.

═══════════════════════════════════════════════════════════════════════════════
ACTIVITY
═══════════════════════════════════════════════════════════════════════════════
Activity: {activity_name}
Description: {activity_text}
Owning org unit: {parent_org_name} (L{parent_org_level})

═══════════════════════════════════════════════════════════════════════════════
CANDIDATE CAPABILITIES (evaluate ALL)
═══════════════════════════════════════════════════════════════════════════════
{capabilities_list}

═══════════════════════════════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════════════════════════════
Identify the TOP 3 best matching capabilities for this single activity. Only
include those with genuine functional alignment. If fewer than 3 match, list
only those that do. If NO capabilities match, respond with "NO_MATCHES_FOUND".

Output format (repeat for each match, best first):

MATCH_1:
CAP_ID: [exact capability ID from list]
SCORE: [0.0 to 1.0]
MATCH_TYPE: [STRONG|MODERATE|WEAK]
JUSTIFICATION: [2-3 sentences]
KEY_OVERLAPS: [specific functional overlaps]

MATCH_2:
CAP_ID: [exact capability ID from list]
SCORE: [0.0 to 1.0]
MATCH_TYPE: [STRONG|MODERATE|WEAK]
JUSTIFICATION: [2-3 sentences]
KEY_OVERLAPS: [specific functional overlaps]

MATCH_3:
CAP_ID: [exact capability ID from list]
SCORE: [0.0 to 1.0]
MATCH_TYPE: [STRONG|MODERATE|WEAK]
JUSTIFICATION: [2-3 sentences]
KEY_OVERLAPS: [specific functional overlaps]
"""

LLM_CATEGORY_SCREEN_SYSTEM = (
    "You are an expert at categorizing organizational activities. Identify "
    "which high-level capability categories are most relevant for a given "
    "activity."
)

LLM_CATEGORY_SCREEN_PROMPT = """Which capability categories are most relevant for this activity?

═══════════════════════════════════════════════════════════════════════════════
ACTIVITY
═══════════════════════════════════════════════════════════════════════════════
Activity: {activity_name}
Description: {activity_text}
Owning org unit: {parent_org_name}

═══════════════════════════════════════════════════════════════════════════════
AVAILABLE CATEGORIES (L0 Capabilities)
═══════════════════════════════════════════════════════════════════════════════
{categories_list}

═══════════════════════════════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════════════════════════════
Select 1-3 categories that are MOST relevant to this activity. Only select
categories where there is genuine functional alignment.

Output format:
SELECTED_CATEGORIES: [comma-separated list of category IDs]
REASONING: [brief explanation]
"""


# =============================================================================
# Match Result
# =============================================================================

@dataclass
class MatchResult:
    """Result of matching a single activity to capabilities."""
    activity_id: str
    activity_name: str
    activity_text: str = ""
    activity_weight: int = 0
    parent_org_id: str = ""
    parent_org_name: str = ""
    parent_org_level: int = 0
    parent_org_path: str = ""
    matches: List[Dict[str, Any]] = field(default_factory=list)
    unmatched: bool = False
    fallback_level: int = 0      # 0 = matched at the configured target level

    def add_match(self, cap_id: str, cap_name: str, cap_level: int,
                  semantic_score: float, llm_score: float, match_type: str,
                  justification: str, key_overlaps: str = "", gaps: str = ""):
        self.matches.append({
            "capability_id": cap_id,
            "capability_name": cap_name,
            "capability_level": cap_level,
            "semantic_score": round(semantic_score, 4),
            "llm_score": round(llm_score, 4),
            "combined_score": round((semantic_score + llm_score) / 2, 4),
            "match_type": match_type,
            "justification": justification,
            "key_overlaps": key_overlaps,
            "gaps": gaps,
            "rank": 0,
            "is_primary": False,
        })

    def best_match(self) -> Optional[Dict]:
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m["combined_score"])

    def to_dict(self) -> Dict:
        return {
            "activity_id": self.activity_id,
            "activity_name": self.activity_name,
            "activity_text": self.activity_text,
            "activity_weight": self.activity_weight,
            "parent_org_id": self.parent_org_id,
            "parent_org_name": self.parent_org_name,
            "parent_org_level": self.parent_org_level,
            "parent_org_path": self.parent_org_path,
            "matches": self.matches,
            "unmatched": self.unmatched,
            "fallback_level": self.fallback_level,
            "best_match": self.best_match(),
        }


# =============================================================================
# Bipartite Matcher
# =============================================================================

class BipartiteCapabilityMatcher:
    """
    Maps individual :Activity leaf nodes to business capabilities.

    Both the org hierarchy (with its Activity leaves) and the capability tree
    live in the same FalkorDB graph. Match results are stored as
    (:Activity)-[:MAPS_TO_CAPABILITY]->(:Capability) edges.
    """

    def __init__(self, config: MatcherConfig, llm, embedder=None):
        self.config = config
        self.llm = llm
        self.embedder = embedder or MockEmbedder()

        # NetworkX in-memory copies
        self.org_graph = nx.DiGraph()
        self.cap_graph = nx.DiGraph()

        # Bipartite result graph (activities ↔ capabilities)
        self.bipartite_graph = nx.Graph()

        # Embedding caches
        self._activity_embeddings: Dict[str, List[float]] = {}
        self._cap_embeddings: Dict[str, List[float]] = {}

        # Results
        self.results: List[MatchResult] = []
        self.unmatched_activities: List[str] = []

        # FalkorDB connection
        self._client = None
        self._fgraph = None
        self._connected = False

    # -----------------------------------------------------------------
    # FalkorDB connection
    # -----------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to FalkorDB and load both graphs into NetworkX."""
        try:
            from falkordb import FalkorDB
        except ImportError:
            print("⚠ FalkorDB not available")
            return False
        try:
            if self.config.falkordb_url:
                self._client = FalkorDB.from_url(self.config.falkordb_url)
            else:
                self._client = FalkorDB(
                    host=self.config.falkordb_host,
                    port=self.config.falkordb_port,
                    password=self.config.falkordb_password,
                )
            self._fgraph = self._client.select_graph(self.config.falkordb_graph_name)
            self._connected = True
            if self.config.verbose:
                print(f"✓ FalkorDB connected: {self.config.falkordb_graph_name}")
            return True
        except Exception as e:
            print(f"✗ FalkorDB connection failed: {e}")
            return False

    def _query(self, cypher: str, params: dict = None) -> List[Dict[str, Any]]:
        """Execute Cypher and return list-of-dicts (header-safe)."""
        if not self._connected:
            return []
        try:
            result = self._fgraph.query(cypher, params or {})
            headers = []
            for h in result.header:
                if isinstance(h, (list, tuple)):
                    headers.append(h[-1])
                else:
                    headers.append(str(h))
            rows = []
            for row in result.result_set:
                rd = {}
                for key, val in zip(headers, row):
                    if hasattr(val, "properties"):
                        rd[key] = dict(val.properties)
                    else:
                        rd[key] = val
                rows.append(rd)
            return rows
        except Exception as e:
            if self.config.verbose:
                print(f"Query error: {e}")
            return []

    # -----------------------------------------------------------------
    # Graph loading
    # -----------------------------------------------------------------

    def load_graphs_from_falkordb(self) -> Tuple[int, int]:
        """Load org and capability sub-graphs from the shared FalkorDB graph."""
        if not self._connected:
            print("⚠ Not connected to FalkorDB")
            return (0, 0)

        # --- Org nodes (DG, OrganizationalUnit, Activity) ---
        org_result = self._fgraph.query(
            "MATCH (n) WHERE n.node_id IS NOT NULL AND "
            "(n:DG OR n:OrganizationalUnit OR n:Activity) "
            "RETURN n.node_id, n"
        )
        org_count = 0
        for row in org_result.result_set:
            nid = row[0]
            props = dict(row[1].properties)
            props.pop("node_id", None)
            self.org_graph.add_node(nid, **props)
            org_count += 1

        # Org edges
        org_edge_result = self._fgraph.query(
            "MATCH (a)-[r]->(b) "
            "WHERE (a:DG OR a:OrganizationalUnit OR a:Activity) AND "
            "      (b:DG OR b:OrganizationalUnit OR b:Activity) "
            "RETURN a.node_id, type(r), b.node_id"
        )
        for row in org_edge_result.result_set:
            if row[0] and row[2]:
                self.org_graph.add_edge(row[0], row[2], edge_type=row[1])

        # --- Capability nodes ---
        cap_result = self._fgraph.query(
            "MATCH (n:Capability) WHERE n.node_id IS NOT NULL RETURN n.node_id, n"
        )
        cap_count = 0
        for row in cap_result.result_set:
            nid = row[0]
            props = dict(row[1].properties)
            props.pop("node_id", None)
            self.cap_graph.add_node(nid, **props)
            cap_count += 1

        cap_edge_result = self._fgraph.query(
            "MATCH (a:Capability)-[r]->(b:Capability) "
            "RETURN a.node_id, type(r), b.node_id"
        )
        for row in cap_edge_result.result_set:
            if row[0] and row[2]:
                self.cap_graph.add_edge(row[0], row[2], edge_type=row[1])

        if self.config.verbose:
            print(f"  Org graph: {org_count} nodes, {self.org_graph.number_of_edges()} edges")
            print(f"  Cap graph: {cap_count} nodes, {self.cap_graph.number_of_edges()} edges")
            print(f"  Activities found: {len(self._activity_nodes())}")

        return (org_count, cap_count)

    def load_graphs_from_local(self, org_path: str, cap_path: str):
        """Load graphs from local JSON files."""
        with open(org_path, "r", encoding="utf-8") as f:
            org_data = json.load(f)
        for node in org_data.get("nodes", []):
            nid = node.pop("id")
            self.org_graph.add_node(nid, **node)
        for edge in org_data.get("edges", []):
            self.org_graph.add_edge(edge["source"], edge["target"])

        with open(cap_path, "r", encoding="utf-8") as f:
            cap_data = json.load(f)
        for node in cap_data.get("nodes", []):
            nid = node.pop("id")
            self.cap_graph.add_node(nid, **node)
        for edge in cap_data.get("edges", []):
            self.cap_graph.add_edge(edge["source"], edge["target"])

        if self.config.verbose:
            print(f"  Org graph: {self.org_graph.number_of_nodes()} nodes")
            print(f"  Cap graph: {self.cap_graph.number_of_nodes()} nodes")
            print(f"  Activities: {len(self._activity_nodes())}")

    # -----------------------------------------------------------------
    # Activity helpers
    # -----------------------------------------------------------------

    def _activity_nodes(self) -> List[str]:
        """All Activity-node IDs in the org graph."""
        return [n for n, d in self.org_graph.nodes(data=True)
                if d.get("node_type") == "activity"]

    def _parent_org_for_activity(self, activity_id: str) -> Optional[str]:
        """Walk predecessors until a non-activity org unit is found."""
        current = activity_id
        visited = set()
        while current not in visited:
            visited.add(current)
            preds = list(self.org_graph.predecessors(current))
            if not preds:
                return None
            parent = preds[0]
            pdata = self.org_graph.nodes.get(parent, {})
            if pdata.get("node_type") != "activity":
                return parent
            current = parent
        return None

    def _activity_text(self, activity_id: str) -> str:
        """Free-text representation of an activity (for embedding/LLM)."""
        d = self.org_graph.nodes.get(activity_id, {})
        parts = [d.get("name", ""), d.get("description", "")]
        if d.get("refined_description"):
            parts.append(d["refined_description"])
        return " ".join(filter(None, parts))

    def _activity_weight(self, activity_id: str) -> int:
        d = self.org_graph.nodes.get(activity_id, {})
        w = d.get("weight", 0)
        try:
            return int(w)
        except (TypeError, ValueError):
            return 0

    def _activity_parent_info(self, activity_id: str) -> Dict[str, Any]:
        parent_id = self._parent_org_for_activity(activity_id)
        if parent_id is None:
            return {"id": "", "name": "", "level": 0, "path": ""}
        pd = self.org_graph.nodes.get(parent_id, {})
        try:
            lvl = int(pd.get("level", 0) or 0)
        except (TypeError, ValueError):
            lvl = 0
        return {
            "id": parent_id,
            "name": pd.get("name", parent_id),
            "level": lvl,
            "path": self._get_org_path(parent_id),
        }

    def _activities_in_scope(self) -> List[str]:
        """Activities whose parent org unit sits at or above max_org_level."""
        result = []
        for aid in self._activity_nodes():
            info = self._activity_parent_info(aid)
            if info["level"] <= self.config.max_org_level:
                result.append(aid)
        return result

    def _get_org_path(self, org_id: str) -> str:
        """Walk org parents to build a human-readable path."""
        if org_id not in self.org_graph:
            return ""
        parts = [self.org_graph.nodes[org_id].get("name", org_id)]
        current = org_id
        visited = {current}
        while True:
            preds = list(self.org_graph.predecessors(current))
            if not preds:
                break
            parent = preds[0]
            if parent in visited:
                break
            visited.add(parent)
            pdata = self.org_graph.nodes.get(parent, {})
            pname = pdata.get("name", parent)
            if "ROOT" in pname.upper():
                break
            parts.insert(0, pname)
            current = parent
        return " > ".join(parts)

    # -----------------------------------------------------------------
    # Capability helpers
    # -----------------------------------------------------------------

    def _cap_leaves(self) -> List[str]:
        """Capability leaf nodes (no children)."""
        return [n for n in self.cap_graph.nodes()
                if self.cap_graph.out_degree(n) == 0
                and not self.cap_graph.nodes[n].get("is_master_root")]

    def _cap_nodes_at_level(self, level: int) -> List[str]:
        return [n for n in self.cap_graph.nodes()
                if self.cap_graph.nodes[n].get("level") == level]

    def _cap_max_level(self) -> int:
        levels = [self.cap_graph.nodes[n].get("level", 0)
                  for n in self.cap_graph.nodes()
                  if not self.cap_graph.nodes[n].get("is_master_root")]
        return max(levels) if levels else 0

    def _cap_target_level(self) -> int:
        cfg = self.config.target_capability_level
        if cfg is None:
            return self._cap_max_level()
        return int(cfg)

    def _cap_target_nodes(self) -> List[str]:
        """Capability nodes at the configured target level."""
        if self.config.target_capability_level is None:
            return self._cap_leaves()
        return self._cap_nodes_at_level(int(self.config.target_capability_level))

    def _get_cap_path(self, cap_id: str) -> str:
        path = [self.cap_graph.nodes[cap_id].get("name", cap_id)]
        current = cap_id
        while True:
            preds = list(self.cap_graph.predecessors(current))
            if not preds:
                break
            parent = preds[0]
            pdata = self.cap_graph.nodes.get(parent, {})
            pname = pdata.get("name", parent)
            if "ROOT" in pname.upper():
                break
            path.insert(0, pname)
            current = parent
        return " > ".join(path)

    def _cap_text(self, cap_id: str) -> str:
        data = self.cap_graph.nodes.get(cap_id, {})
        parts = [data.get("name", "")]
        if data.get("refined_description"):
            parts.append(data["refined_description"])
        elif data.get("description"):
            parts.append(data["description"])
        if data.get("capability_keywords"):
            parts.append(data["capability_keywords"])
        return " ".join(filter(None, parts))

    def _format_caps_for_llm(self, cap_ids: List[str], max_n: int = 20) -> str:
        lines = []
        for cid in cap_ids[:max_n]:
            cd = self.cap_graph.nodes.get(cid, {})
            name = cd.get("name", cid)
            desc = (cd.get("refined_description") or cd.get("description", ""))[:150]
            path = self._get_cap_path(cid)
            lines.append(f"[{cid}] {name}")
            lines.append(f"    Path: {path}")
            if desc:
                lines.append(f"    Description: {desc}")
            lines.append("")
        if len(cap_ids) > max_n:
            lines.append(f"... and {len(cap_ids) - max_n} more")
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Embedding helpers
    # -----------------------------------------------------------------

    def compute_embeddings(self, activity_ids: List[str] = None):
        """Pre-compute embeddings for activities and all capabilities."""
        if self.config.verbose:
            print("\nComputing embeddings...")

        if activity_ids is None:
            activity_ids = self._activity_nodes()
        texts = [self._activity_text(a) for a in activity_ids]
        for aid, emb in zip(activity_ids, self.embedder.embed_batch(texts)):
            self._activity_embeddings[aid] = emb

        cap_ids = list(self.cap_graph.nodes())
        cap_texts = [self._cap_text(c) for c in cap_ids]
        for cid, emb in zip(cap_ids, self.embedder.embed_batch(cap_texts)):
            self._cap_embeddings[cid] = emb

        if self.config.verbose:
            print(f"  Activity embeddings: {len(self._activity_embeddings)}")
            print(f"  Cap embeddings:      {len(self._cap_embeddings)}")

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = np.sqrt(sum(x * x for x in a))
        nb = np.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na > 0 and nb > 0 else 0.0

    def _keyword_overlap(self, activity_id: str, cap_id: str) -> float:
        act_text = self._activity_text(activity_id).lower()
        act_words = set(act_text.split())

        cap_data = self.cap_graph.nodes.get(cap_id, {})
        cap_text = (cap_data.get("capability_keywords", "") + " " +
                    cap_data.get("name", "")).lower()
        cap_words = set(cap_text.replace(",", " ").split())

        if not act_words or not cap_words:
            return 0.0
        return len(act_words & cap_words) / max(len(act_words), len(cap_words))

    # -----------------------------------------------------------------
    # Hybrid search
    # -----------------------------------------------------------------

    def hybrid_search(self, activity_id: str, candidates: List[str],
                      top_k: int = 5) -> List[Tuple[str, float]]:
        act_emb = self._activity_embeddings.get(activity_id)
        if not act_emb:
            act_emb = self.embedder.embed(self._activity_text(activity_id))
            self._activity_embeddings[activity_id] = act_emb

        scores = []
        for cid in candidates:
            cap_emb = self._cap_embeddings.get(cid)
            if not cap_emb:
                cap_emb = self.embedder.embed(self._cap_text(cid))
                self._cap_embeddings[cid] = cap_emb
            sem = self._cosine_sim(act_emb, cap_emb)
            if self.config.use_hybrid_search:
                kw = self._keyword_overlap(activity_id, cid)
                combined = 0.7 * sem + 0.3 * kw
            else:
                combined = sem
            scores.append((cid, combined))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # -----------------------------------------------------------------
    # LLM evaluation
    # -----------------------------------------------------------------

    def _parse_judge_response(self, resp: str) -> dict:
        out = {"llm_score": 0.0, "match_type": "NO_MATCH",
               "justification": "", "key_overlaps": "", "gaps": ""}
        for line in resp.split("\n"):
            line = line.strip()
            if line.startswith("MATCH_SCORE:"):
                try:
                    out["llm_score"] = min(max(float(line.split(":", 1)[1].strip()), 0), 1)
                except Exception:
                    pass
            elif line.startswith("MATCH_TYPE:"):
                out["match_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("JUSTIFICATION:"):
                out["justification"] = line.split(":", 1)[1].strip()
            elif line.startswith("KEY_OVERLAPS:"):
                out["key_overlaps"] = line.split(":", 1)[1].strip()
            elif line.startswith("GAPS:"):
                out["gaps"] = line.split(":", 1)[1].strip()
        return out

    def _parse_batch_response(self, resp: str) -> List[dict]:
        matches, cur = [], {}
        for line in resp.split("\n"):
            line = line.strip()
            if line.startswith("MATCH_") and ":" not in line[6:8]:
                if cur.get("cap_id"):
                    matches.append(cur)
                cur = {}
            elif line.startswith("CAP_ID:"):
                cur["cap_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("SCORE:"):
                try:
                    cur["score"] = float(line.split(":", 1)[1].strip())
                except Exception:
                    cur["score"] = 0.0
            elif line.startswith("MATCH_TYPE:"):
                cur["match_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("JUSTIFICATION:"):
                cur["justification"] = line.split(":", 1)[1].strip()
            elif line.startswith("KEY_OVERLAPS:"):
                cur["key_overlaps"] = line.split(":", 1)[1].strip()
        if cur.get("cap_id"):
            matches.append(cur)
        return matches

    def _parse_category_screen(self, resp: str) -> List[str]:
        for line in resp.split("\n"):
            if line.strip().startswith("SELECTED_CATEGORIES:"):
                s = line.split(":", 1)[1].strip().replace("[", "").replace("]", "")
                return [c.strip() for c in s.split(",") if c.strip()]
        return []

    def llm_evaluate(self, activity_id: str, cap_id: str) -> dict:
        """LLM judge: evaluate a single activity–capability pair."""
        parent = self._activity_parent_info(activity_id)
        ad = self.org_graph.nodes.get(activity_id, {})
        cd = self.cap_graph.nodes.get(cap_id, {})
        prompt = LLM_JUDGE_PROMPT.format(
            activity_name=ad.get("name", activity_id),
            activity_text=ad.get("description", "") or ad.get("name", ""),
            activity_weight=self._activity_weight(activity_id),
            parent_org_name=parent["name"] or "N/A",
            parent_org_level=parent["level"],
            parent_org_path=parent["path"] or "N/A",
            cap_name=cd.get("name", cap_id),
            cap_id=cap_id,
            cap_level=cd.get("level", 0),
            cap_type=cd.get("node_type", "capability"),
            cap_path=self._get_cap_path(cap_id),
            cap_description=cd.get("refined_description", cd.get("description", "N/A")),
            cap_keywords=cd.get("capability_keywords", "N/A"),
        )
        resp = self.llm.generate(prompt, LLM_JUDGE_SYSTEM)
        return self._parse_judge_response(resp)

    def llm_rank_batch(self, activity_id: str, cap_ids: List[str]) -> List[dict]:
        parent = self._activity_parent_info(activity_id)
        ad = self.org_graph.nodes.get(activity_id, {})
        prompt = LLM_BATCH_RANKING_PROMPT.format(
            activity_name=ad.get("name", activity_id),
            activity_text=ad.get("description", "") or ad.get("name", ""),
            parent_org_name=parent["name"] or "N/A",
            parent_org_level=parent["level"],
            capabilities_list=self._format_caps_for_llm(cap_ids),
        )
        resp = self.llm.generate(prompt, LLM_BATCH_RANKING_SYSTEM)
        return self._parse_batch_response(resp)

    def llm_screen_categories(self, activity_id: str) -> List[str]:
        parent = self._activity_parent_info(activity_id)
        ad = self.org_graph.nodes.get(activity_id, {})
        l0_caps = self._cap_nodes_at_level(0)
        if not l0_caps:
            return []
        cat_lines = []
        for cid in l0_caps:
            cd = self.cap_graph.nodes.get(cid, {})
            cat_lines.append(f"[{cid}] {cd.get('name', cid)}: {cd.get('description', '')[:100]}")
        prompt = LLM_CATEGORY_SCREEN_PROMPT.format(
            activity_name=ad.get("name", activity_id),
            activity_text=ad.get("description", "") or ad.get("name", ""),
            parent_org_name=parent["name"] or "N/A",
            categories_list="\n".join(cat_lines),
        )
        resp = self.llm.generate(prompt, LLM_CATEGORY_SCREEN_SYSTEM)
        return self._parse_category_screen(resp)

    # -----------------------------------------------------------------
    # Single-activity matching (dispatches on mode)
    # -----------------------------------------------------------------

    def _new_result(self, activity_id: str, fallback_level: int = 0) -> MatchResult:
        ad = self.org_graph.nodes.get(activity_id, {})
        parent = self._activity_parent_info(activity_id)
        return MatchResult(
            activity_id=activity_id,
            activity_name=ad.get("name", activity_id),
            activity_text=ad.get("description", "") or ad.get("name", ""),
            activity_weight=self._activity_weight(activity_id),
            parent_org_id=parent["id"],
            parent_org_name=parent["name"],
            parent_org_level=parent["level"],
            parent_org_path=parent["path"],
            fallback_level=fallback_level,
        )

    def _cap_l0_ancestor(self, cap_id: str) -> Optional[str]:
        """Walk cap parents until a level-0 node is found."""
        if cap_id not in self.cap_graph:
            return None
        current = cap_id
        visited = {current}
        while True:
            d = self.cap_graph.nodes.get(current, {})
            if d.get("level") == 0:
                return current
            preds = list(self.cap_graph.predecessors(current))
            if not preds or preds[0] in visited:
                return None
            current = preds[0]
            visited.add(current)

    def _finalize_matches(self, result: MatchResult) -> MatchResult:
        """Apply multi-match policy: sort, cap, tie-window, primary gate, rank."""
        if not result.matches:
            result.unmatched = True
            return result

        result.matches.sort(key=lambda m: m["combined_score"], reverse=True)
        top_score = result.matches[0]["combined_score"]

        if top_score < self.config.min_primary_score:
            result.matches = []
            result.unmatched = True
            return result

        if self.config.restrict_to_primary_category:
            primary_l0 = self._cap_l0_ancestor(result.matches[0]["capability_id"])
            if primary_l0 is not None:
                result.matches = [
                    m for m in result.matches
                    if self._cap_l0_ancestor(m["capability_id"]) == primary_l0
                ]

        delta = self.config.keep_within_delta
        kept = [result.matches[0]]
        for m in result.matches[1:]:
            if top_score - m["combined_score"] <= delta:
                kept.append(m)
        result.matches = kept[: self.config.max_matches_per_activity]

        if len(result.matches) < self.config.min_matches_per_activity:
            result.matches = []
            result.unmatched = True
            return result

        for i, m in enumerate(result.matches):
            m["rank"] = i + 1
            m["is_primary"] = (i == 0)

        return result

    def _match_activity_hybrid(self, activity_id: str, cap_candidates: List[str],
                               fallback_level: int = 0) -> MatchResult:
        result = self._new_result(activity_id, fallback_level)
        candidates = self.hybrid_search(activity_id, cap_candidates,
                                        self.config.top_k_candidates)
        candidates = [(c, s) for c, s in candidates if s >= self.config.min_semantic_score]
        if not candidates:
            result.unmatched = True
            return result

        for cap_id, sem_score in candidates:
            if self.config.use_llm_judge:
                lr = self.llm_evaluate(activity_id, cap_id)
                if lr["llm_score"] >= self.config.min_llm_score:
                    cd = self.cap_graph.nodes.get(cap_id, {})
                    result.add_match(
                        cap_id, cd.get("name", cap_id), cd.get("level", 0),
                        sem_score, lr["llm_score"], lr["match_type"],
                        lr["justification"], lr["key_overlaps"], lr["gaps"],
                    )
            else:
                cd = self.cap_graph.nodes.get(cap_id, {})
                mt = "STRONG" if sem_score > 0.7 else "MODERATE" if sem_score > 0.5 else "WEAK"
                result.add_match(
                    cap_id, cd.get("name", cap_id), cd.get("level", 0),
                    sem_score, sem_score, mt,
                    f"Matched by semantic similarity ({sem_score:.3f})",
                )
        return self._finalize_matches(result)

    def _match_activity_llm_only(self, activity_id: str, cap_candidates: List[str],
                                 fallback_level: int = 0) -> MatchResult:
        result = self._new_result(activity_id, fallback_level)
        all_matches = []
        bs = self.config.llm_batch_size
        for i in range(0, len(cap_candidates), bs):
            batch = cap_candidates[i:i + bs]
            all_matches.extend(self.llm_rank_batch(activity_id, batch))
        all_matches.sort(key=lambda x: x.get("score", 0), reverse=True)

        for m in all_matches:
            cid = m.get("cap_id", "")
            score = m.get("score", 0)
            if cid not in self.cap_graph.nodes:
                for c in cap_candidates:
                    if cid in c or c in cid:
                        cid = c
                        break
                else:
                    continue
            if score >= self.config.min_llm_score:
                cd = self.cap_graph.nodes.get(cid, {})
                result.add_match(
                    cid, cd.get("name", cid), cd.get("level", 0),
                    0.0, score, m.get("match_type", "MODERATE"),
                    m.get("justification", ""), m.get("key_overlaps", ""),
                )
        return self._finalize_matches(result)

    def _match_activity(self, activity_id: str, cap_candidates: List[str],
                        fallback_level: int = 0) -> MatchResult:
        """Dispatch to the correct matching sub-method."""
        mode = self.config.matching_mode
        if mode in ("llm_only", "llm_prescreened"):
            return self._match_activity_llm_only(activity_id, cap_candidates, fallback_level)
        # hybrid or semantic_only
        orig = self.config.use_llm_judge
        if mode == "semantic_only":
            self.config.use_llm_judge = False
        try:
            return self._match_activity_hybrid(activity_id, cap_candidates, fallback_level)
        finally:
            self.config.use_llm_judge = orig

    # -----------------------------------------------------------------
    # Hierarchical fallback on capability side
    # -----------------------------------------------------------------

    def _hierarchical_fallback(self, unmatched: List[str]) -> List[str]:
        """Try matching unmatched activities against higher capability levels."""
        if not self.config.enable_hierarchical_fallback:
            return unmatched

        base_lvl = self._cap_target_level()
        still_unmatched = list(unmatched)

        for fb in range(1, self.config.max_fallback_levels + 1):
            if not still_unmatched:
                break
            target_lvl = base_lvl - fb
            if target_lvl < 0:
                break
            higher_caps = self._cap_nodes_at_level(target_lvl)
            if not higher_caps:
                continue

            if self.config.verbose:
                print(f"\n  Fallback → capability level {target_lvl} "
                      f"({len(higher_caps)} nodes, {len(still_unmatched)} unmatched)")

            remaining = []
            for act_id in still_unmatched:
                res = self._match_activity(act_id, higher_caps, fallback_level=fb)
                # Update stored result in-place
                for r in self.results:
                    if r.activity_id == act_id:
                        r.matches = res.matches
                        r.unmatched = res.unmatched
                        r.fallback_level = fb
                        break
                if res.unmatched:
                    remaining.append(act_id)
                elif self.config.verbose:
                    best = res.best_match()
                    aname = self.org_graph.nodes[act_id].get("name", "")[:30]
                    print(f"    {act_id}: {aname} → {best['capability_name'][:30]} "
                          f"(L{target_lvl}, {best['match_type']})")
            still_unmatched = remaining

        return still_unmatched

    # -----------------------------------------------------------------
    # Main run
    # -----------------------------------------------------------------

    def run(self, target_activity_ids: List[str] = None) -> List[MatchResult]:
        """
        Run bipartite matching over activity leaf nodes.

        Args:
            target_activity_ids: Optional explicit list of Activity IDs to match.
                                 If omitted, all in-scope activities are used.
        Returns:
            List of MatchResult (one per activity).
        """
        if self.config.verbose:
            print("\n" + "═" * 70)
            print(f"BIPARTITE ACTIVITY→CAPABILITY MATCHING  "
                  f"(mode={self.config.matching_mode})")
            print("═" * 70)

        # --- Determine which activities to match ---
        if target_activity_ids:
            activities = [a for a in target_activity_ids if a in self.org_graph.nodes
                          and self.org_graph.nodes[a].get("node_type") == "activity"]
        else:
            activities = self._activities_in_scope()

        # Pre-compute embeddings for hybrid/semantic modes
        if self.config.matching_mode in ("hybrid", "semantic_only"):
            if not self._cap_embeddings or \
               any(a not in self._activity_embeddings for a in activities):
                self.compute_embeddings(activity_ids=activities)

        target_lvl = self._cap_target_level()
        cap_targets = self._cap_target_nodes()
        if self.config.verbose:
            lvl_label = f"level {target_lvl}" + (" (leaves)"
                if self.config.target_capability_level is None else "")
            print(f"  Capability targets ({lvl_label}): {len(cap_targets)}")
            print(f"  Activities to match:               {len(activities)}")

            # Breakdown by parent org level
            by_lvl: Dict[int, int] = defaultdict(int)
            for aid in activities:
                by_lvl[self._activity_parent_info(aid)["level"]] += 1
            for lvl in sorted(by_lvl):
                print(f"    Parent org L{lvl}: {by_lvl[lvl]} activities")

        self.results = []
        self.unmatched_activities = []

        # --- LLM pre-screening cache (for llm_prescreened mode) ---
        category_cache: Dict[str, List[str]] = {}

        # --- Match each activity ---
        for idx, act_id in enumerate(activities):
            ad = self.org_graph.nodes[act_id]
            aname = ad.get("name", act_id)[:40]
            if self.config.verbose:
                parent_lvl = self._activity_parent_info(act_id)["level"]
                print(f"\n[{idx+1}/{len(activities)}] L{parent_lvl} {act_id}: {aname}")

            # Determine candidate capabilities
            if self.config.matching_mode == "llm_prescreened":
                if act_id not in category_cache:
                    relevant_cats = self.llm_screen_categories(act_id)
                    if relevant_cats:
                        target_set = set(cap_targets)
                        relevant_caps = []
                        for cat_id in relevant_cats:
                            if cat_id in self.cap_graph.nodes:
                                if cat_id in target_set:
                                    relevant_caps.append(cat_id)
                                desc = nx.descendants(self.cap_graph, cat_id)
                                relevant_caps.extend(d for d in desc if d in target_set)
                        category_cache[act_id] = list(set(relevant_caps)) or cap_targets
                    else:
                        category_cache[act_id] = cap_targets
                    if self.config.verbose:
                        print(f"  Pre-screened: {len(category_cache[act_id])} capabilities")
                candidates = category_cache[act_id]
            else:
                candidates = cap_targets

            result = self._match_activity(act_id, candidates)
            self.results.append(result)

            if result.unmatched:
                self.unmatched_activities.append(act_id)
                if self.config.verbose:
                    print(f"  → UNMATCHED")
            elif self.config.verbose:
                best = result.best_match()
                print(f"  → {best['capability_name'][:35]} "
                      f"({best['combined_score']:.3f} {best['match_type']})")

        # --- Hierarchical fallback ---
        if self.unmatched_activities:
            if self.config.verbose:
                print(f"\n--- Hierarchical Fallback ({len(self.unmatched_activities)} unmatched) ---")
            self.unmatched_activities = self._hierarchical_fallback(self.unmatched_activities)

        # --- Build bipartite graph ---
        self._build_bipartite_graph()

        # --- Summary ---
        matched = sum(1 for r in self.results if not r.unmatched)
        if self.config.verbose:
            print(f"\n{'═' * 70}")
            print(f"MATCHING COMPLETE")
            print(f"  Matched:   {matched}/{len(self.results)} activities")
            print(f"  Unmatched: {len(self.unmatched_activities)}")
            print(f"{'═' * 70}")

        return self.results

    # -----------------------------------------------------------------
    # Bipartite graph (activities ↔ capabilities)
    # -----------------------------------------------------------------

    def _build_bipartite_graph(self):
        self.bipartite_graph = nx.Graph()
        for r in self.results:
            self.bipartite_graph.add_node(
                r.activity_id, bipartite=0, node_type="activity",
                name=r.activity_name, weight=r.activity_weight,
                parent_org_id=r.parent_org_id,
                parent_org_name=r.parent_org_name,
                parent_org_level=r.parent_org_level,
            )
        for r in self.results:
            for m in r.matches:
                cid = m["capability_id"]
                if cid not in self.bipartite_graph:
                    self.bipartite_graph.add_node(
                        cid, bipartite=1, node_type="capability",
                        name=m["capability_name"], level=m["capability_level"],
                    )
                self.bipartite_graph.add_edge(
                    r.activity_id, cid,
                    semantic_score=m["semantic_score"],
                    llm_score=m["llm_score"],
                    combined_score=m["combined_score"],
                    match_type=m["match_type"],
                    justification=m["justification"],
                    key_overlaps=m.get("key_overlaps", ""),
                    gaps=m.get("gaps", ""),
                )

    # -----------------------------------------------------------------
    # FalkorDB sync (edges in shared graph)
    # -----------------------------------------------------------------

    def sync_to_falkordb(self) -> int:
        """Write :MAPS_TO_CAPABILITY edges (Activity→Capability) to FalkorDB."""
        if not self._connected or not self.config.sync_to_falkordb:
            return 0
        if self.config.verbose:
            print("\nSyncing matches to FalkorDB...")

        # Clear old match edges
        try:
            self._fgraph.query("MATCH (:Activity)-[r:MAPS_TO_CAPABILITY]->() DELETE r")
        except Exception:
            pass

        count = 0
        for r in self.results:
            for m in r.matches:
                try:
                    self._fgraph.query(
                        "MATCH (a:Activity {node_id: $aid}), (c:Capability {node_id: $cid}) "
                        "MERGE (a)-[r:MAPS_TO_CAPABILITY]->(c) "
                        "SET r.semantic_score = $sem, r.llm_score = $llm, "
                        "    r.combined_score = $comb, r.match_type = $mt, "
                        "    r.justification = $just, r.fallback_level = $fb, "
                        "    r.rank = $rank, r.is_primary = $prim",
                        {
                            "aid": r.activity_id,
                            "cid": m["capability_id"],
                            "sem": m["semantic_score"],
                            "llm": m["llm_score"],
                            "comb": m["combined_score"],
                            "mt": m["match_type"],
                            "just": m["justification"],
                            "fb": r.fallback_level,
                            "rank": m.get("rank", 0),
                            "prim": m.get("is_primary", False),
                        },
                    )
                    count += 1
                except Exception as e:
                    if self.config.verbose:
                        print(f"  Sync error {r.activity_id}→{m['capability_id']}: {e}")

        if self.config.verbose:
            print(f"✓ Synced {count} MAPS_TO_CAPABILITY edges")
        return count

    # -----------------------------------------------------------------
    # Results I/O
    # -----------------------------------------------------------------

    def save_results(self, path: str) -> str:
        output = {
            "metadata": {
                "total_activities": len(self.results),
                "matched": sum(1 for r in self.results if not r.unmatched),
                "unmatched": len(self.unmatched_activities),
                "config": {
                    "matching_mode": self.config.matching_mode,
                    "max_org_level": self.config.max_org_level,
                    "target_capability_level": self.config.target_capability_level,
                    "min_semantic_score": self.config.min_semantic_score,
                    "min_llm_score": self.config.min_llm_score,
                    "top_k_candidates": self.config.top_k_candidates,
                    "use_llm_judge": self.config.use_llm_judge,
                    "hierarchical_fallback": self.config.enable_hierarchical_fallback,
                },
            },
            "matches": [r.to_dict() for r in self.results],
            "unmatched_activities": self.unmatched_activities,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        if self.config.verbose:
            print(f"✓ Results saved: {path}")
        return path

    def save_bipartite_graph(self, path: str) -> str:
        gd = {
            "metadata": {
                "activity_nodes": sum(1 for n in self.bipartite_graph.nodes()
                                      if self.bipartite_graph.nodes[n].get("bipartite") == 0),
                "cap_nodes": sum(1 for n in self.bipartite_graph.nodes()
                                 if self.bipartite_graph.nodes[n].get("bipartite") == 1),
                "edges": self.bipartite_graph.number_of_edges(),
            },
            "nodes": [{"id": n, **dict(d)} for n, d in self.bipartite_graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **dict(d)}
                      for u, v, d in self.bipartite_graph.edges(data=True)],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(gd, f, indent=2, ensure_ascii=False)
        if self.config.verbose:
            print(f"✓ Bipartite graph saved: {path}")
        return path

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def print_summary(self):
        print("\n" + "═" * 70)
        print("MATCHING SUMMARY")
        print("═" * 70)

        matched = [r for r in self.results if not r.unmatched]
        print(f"\nTotal activities processed: {len(self.results)}")
        print(f"Matched:   {len(matched)}")
        print(f"Unmatched: {len(self.unmatched_activities)}")

        # Breakdown by parent org level
        by_lvl: Dict[int, Tuple[int, int]] = {}
        for r in self.results:
            lvl = r.parent_org_level
            t, m = by_lvl.get(lvl, (0, 0))
            by_lvl[lvl] = (t + 1, m + (0 if r.unmatched else 1))
        if by_lvl:
            print("\nBy parent org level:")
            for lvl in sorted(by_lvl):
                t, m = by_lvl[lvl]
                print(f"  Level {lvl}: {m}/{t} matched")

        # Match type distribution
        type_counts: Dict[str, int] = {}
        for r in matched:
            best = r.best_match()
            if best:
                type_counts[best["match_type"]] = type_counts.get(best["match_type"], 0) + 1
        if type_counts:
            print("\nMatch types:")
            for t, c in sorted(type_counts.items()):
                print(f"  {t}: {c}")

        # Fallback stats
        fb_counts = defaultdict(int)
        for r in matched:
            fb_counts[r.fallback_level] += 1
        if any(k > 0 for k in fb_counts):
            print("\nFallback levels:")
            for lvl in sorted(fb_counts):
                label = "target" if lvl == 0 else f"fallback +{lvl}"
                print(f"  {label}: {fb_counts[lvl]}")

        # Sample matches
        print("\n" + "─" * 70)
        print("SAMPLE MATCHES (Top 5)")
        print("─" * 70)
        for r in matched[:5]:
            best = r.best_match()
            if best:
                fb = f" [fallback +{r.fallback_level}]" if r.fallback_level > 0 else ""
                print(f"\n  [{r.activity_id}] {r.activity_name} "
                      f"(org: {r.parent_org_name}, L{r.parent_org_level})")
                print(f"  → [{best['capability_id']}] {best['capability_name']}{fb}")
                print(f"    Score: {best['combined_score']:.3f} ({best['match_type']})")
                if best.get("justification"):
                    print(f"    {best['justification'][:120]}")


# =============================================================================
# Factory
# =============================================================================

def create_bipartite_matcher(
    llm,
    falkordb_url: str = None,
    falkordb_graph_name: str = "org_hierarchy",
    matching_mode: str = "hybrid",
    max_org_level: int = 99,
    target_capability_level: Optional[int] = None,
    use_mock_embedder: bool = True,
    use_llm_judge: bool = True,
    min_semantic_score: float = 0.3,
    min_llm_score: float = 0.6,
    top_k_candidates: int = 5,
    llm_batch_size: int = 10,
    enable_fallback: bool = True,
    max_fallback_levels: int = 2,
    max_matches_per_activity: int = 3,
    min_matches_per_activity: int = 1,
    keep_within_delta: float = 0.1,
    min_primary_score: float = 0.0,
    restrict_to_primary_category: bool = False,
    sync_to_falkordb: bool = True,
    verbose: bool = True,
) -> BipartiteCapabilityMatcher:
    """
    Create a BipartiteCapabilityMatcher that maps Activity leaf nodes to
    business capabilities.

    Args:
        llm: LLM instance (from llm.py)
        falkordb_url: FalkorDB connection URL
        falkordb_graph_name: Shared graph name (default: org_hierarchy)
        matching_mode: hybrid | llm_only | llm_prescreened | semantic_only
        max_org_level: Only consider activities whose parent org unit has
                       level <= this value.
        target_capability_level: Capability level to map activities to.
            0 = Category, 1 = Business Area, 2 = Sub-Business Area.
            None (default) = match against the leaves of the capability tree.
        use_mock_embedder: Use hash-based embedder (True) or HuggingFace (False)
        use_llm_judge: Use LLM for final evaluation in hybrid mode
        min_semantic_score: Minimum semantic similarity threshold
        min_llm_score: Minimum LLM score threshold
        top_k_candidates: Number of candidates for semantic pre-filter
        llm_batch_size: Capabilities per LLM batch
        enable_fallback: Enable hierarchical capability fallback
        max_fallback_levels: How many levels up to try on capability side
        sync_to_falkordb: Write match edges back to FalkorDB
        verbose: Print progress

    Returns:
        Configured BipartiteCapabilityMatcher
    """
    embedder = MockEmbedder() if use_mock_embedder else HuggingFaceEmbedder()
    config = MatcherConfig(
        matching_mode=matching_mode,
        max_org_level=max_org_level,
        target_capability_level=target_capability_level,
        min_semantic_score=min_semantic_score,
        top_k_candidates=top_k_candidates,
        use_llm_judge=use_llm_judge,
        min_llm_score=min_llm_score,
        llm_batch_size=llm_batch_size,
        enable_hierarchical_fallback=enable_fallback,
        max_fallback_levels=max_fallback_levels,
        max_matches_per_activity=max_matches_per_activity,
        min_matches_per_activity=min_matches_per_activity,
        keep_within_delta=keep_within_delta,
        min_primary_score=min_primary_score,
        restrict_to_primary_category=restrict_to_primary_category,
        falkordb_url=falkordb_url,
        falkordb_graph_name=falkordb_graph_name,
        sync_to_falkordb=sync_to_falkordb,
        verbose=verbose,
    )
    matcher = BipartiteCapabilityMatcher(config=config, llm=llm, embedder=embedder)

    if falkordb_url:
        matcher.connect()

    return matcher


if __name__ == "__main__":
    print("BipartiteCapabilityMatcher module loaded.")
    print()
    print("Subject of matching: individual :Activity leaf nodes")
    print()
    print("Matching modes:")
    print("  hybrid:          Semantic pre-filter + LLM judge (default)")
    print("  llm_only:        Pure LLM matching (most accurate)")
    print("  llm_prescreened: LLM category filter + LLM matching")
    print("  semantic_only:   Embedding similarity only (fastest)")
    print()
    print("Usage:")
    print("  matcher = create_bipartite_matcher(")
    print("      llm, falkordb_url=URL,")
    print("      max_org_level=4, target_capability_level=2)")
    print("  matcher.load_graphs_from_falkordb()")
    print("  results = matcher.run()")
