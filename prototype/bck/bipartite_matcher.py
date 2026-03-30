"""
Component 7: Bipartite Capability-Organization Matcher
Matches organizational unit leaves to business capability leaves using:
- Hybrid search (semantic + keyword)
- LLM-as-judge for final evaluation with justification
- Hierarchical fallback for unmatched nodes
- Dual storage (NetworkX bipartite graph + FalkorDB)

Architecture:
    Org Leaves (Set A) ←→ Capability Leaves (Set B)
    
    For each org leaf:
    1. Hybrid search against capability leaves
    2. LLM evaluates top candidates with justification
    3. If no match at leaf level, try higher capability levels
    4. Store match with justification in both storages
"""

import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np


@dataclass
class MatcherConfig:
    """Configuration for capability matching."""
    # Matching mode
    matching_mode: str = "hybrid"  # "hybrid", "llm_only", "semantic_only"
    
    # Semantic/hybrid settings
    min_semantic_score: float = 0.3
    top_k_candidates: int = 5
    use_hybrid_search: bool = True
    
    # LLM settings
    min_llm_score: float = 0.6
    use_llm_judge: bool = True
    llm_batch_size: int = 10  # For LLM-only mode: caps to evaluate per batch
    
    # Hierarchical fallback
    enable_hierarchical_fallback: bool = True
    max_fallback_levels: int = 2
    
    # Storage
    sync_to_falkordb: bool = True
    bipartite_graph_name: str = "org_capability_matches"
    
    # Output
    verbose: bool = True


class MockEmbedder:
    """Mock embedder using deterministic hashing for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._cache = {}
    
    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(self.dimension):
            val = (hash_bytes[i % len(hash_bytes)] / 255.0) * 2 - 1
            embedding.append(val)
        
        norm = np.sqrt(sum(x**2 for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        self._cache[text] = embedding
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class HuggingFaceEmbedder:
    """Real embedder using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            print(f"✓ Embedder: {model_name}")
        except ImportError:
            raise ImportError("Install: pip install sentence-transformers")
    
    def embed(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()


LLM_JUDGE_SYSTEM = """You are an expert organizational analyst evaluating the match between 
organizational units and business capabilities in the European Parliament context.
Assess semantic similarity, functional overlap, and organizational fit."""

LLM_JUDGE_PROMPT = """Evaluate the match between this organizational unit and business capability.

═══════════════════════════════════════════════════════════════════════════════
ORGANIZATIONAL UNIT
═══════════════════════════════════════════════════════════════════════════════
Name: {org_name}
Code: {org_code}
Level: {org_level}

Activities:
{org_activities}

Refined Description:
{org_description}

═══════════════════════════════════════════════════════════════════════════════
BUSINESS CAPABILITY
═══════════════════════════════════════════════════════════════════════════════
Name: {cap_name}
ID: {cap_id}
Level: {cap_level} ({cap_type})
Category Path: {cap_path}

Description:
{cap_description}

Keywords: {cap_keywords}

═══════════════════════════════════════════════════════════════════════════════
EVALUATION TASK
═══════════════════════════════════════════════════════════════════════════════
Rate the match and provide detailed justification.

Output format:
MATCH_SCORE: [0.0 to 1.0]
MATCH_TYPE: [STRONG|MODERATE|WEAK|NO_MATCH]
JUSTIFICATION: [2-4 sentences explaining the match rationale - be specific about which activities map to which capability aspects]
KEY_OVERLAPS: [List specific functional overlaps]
GAPS: [List any significant gaps or misalignments]
"""

# =============================================================================
# LLM-ONLY MATCHING PROMPTS
# =============================================================================

LLM_BATCH_RANKING_SYSTEM = """You are an expert organizational analyst for the European Parliament.
Your task is to identify the BEST matching business capabilities for an organizational unit.
You must analyze functional alignment, activity overlap, and organizational fit."""

LLM_BATCH_RANKING_PROMPT = """Find the best matching capabilities for this organizational unit.

═══════════════════════════════════════════════════════════════════════════════
ORGANIZATIONAL UNIT TO MATCH
═══════════════════════════════════════════════════════════════════════════════
Name: {org_name}
Code: {org_code}

Activities:
{org_activities}

Description:
{org_description}

═══════════════════════════════════════════════════════════════════════════════
CANDIDATE CAPABILITIES (evaluate ALL of these)
═══════════════════════════════════════════════════════════════════════════════
{capabilities_list}

═══════════════════════════════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════════════════════════════
Identify the TOP 3 best matching capabilities from the list above.
For each match, provide a score and justification.

IMPORTANT: Only include capabilities with genuine functional alignment.
If fewer than 3 capabilities match well, list only those that do.
If NO capabilities match, respond with "NO_MATCHES_FOUND".

Output format (repeat for each match, best first):

MATCH_1:
CAP_ID: [exact capability ID from list]
SCORE: [0.0 to 1.0]
MATCH_TYPE: [STRONG|MODERATE|WEAK]
JUSTIFICATION: [2-3 sentences explaining why this capability matches]
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

LLM_CATEGORY_SCREEN_SYSTEM = """You are an expert at categorizing organizational functions.
Identify which high-level capability categories are most relevant for an organizational unit."""

LLM_CATEGORY_SCREEN_PROMPT = """Which capability categories are most relevant for this organizational unit?

═══════════════════════════════════════════════════════════════════════════════
ORGANIZATIONAL UNIT
═══════════════════════════════════════════════════════════════════════════════
Name: {org_name}
Code: {org_code}

Activities:
{org_activities}

═══════════════════════════════════════════════════════════════════════════════
AVAILABLE CATEGORIES (L1 Capabilities)
═══════════════════════════════════════════════════════════════════════════════
{categories_list}

═══════════════════════════════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════════════════════════════
Select 1-3 categories that are MOST relevant for this organizational unit.
Only select categories where there is genuine functional alignment.

Output format:
SELECTED_CATEGORIES: [comma-separated list of category IDs]
REASONING: [brief explanation of why these categories were selected]
"""


@dataclass
class MatchResult:
    """Result of matching an org unit to capabilities."""
    org_id: str
    org_name: str
    org_level: int
    matches: List[Dict[str, Any]] = field(default_factory=list)
    unmatched: bool = False
    fallback_level: int = 0
    
    def add_match(self, cap_id: str, cap_name: str, cap_level: int, semantic_score: float,
                  llm_score: float, match_type: str, justification: str,
                  key_overlaps: str = "", gaps: str = ""):
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
            "gaps": gaps
        })
    
    def best_match(self) -> Optional[Dict]:
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m["combined_score"])
    
    def to_dict(self) -> Dict:
        return {
            "org_id": self.org_id,
            "org_name": self.org_name,
            "org_level": self.org_level,
            "matches": self.matches,
            "unmatched": self.unmatched,
            "fallback_level": self.fallback_level,
            "best_match": self.best_match()
        }


class BipartiteCapabilityMatcher:
    """
    Matches organizational unit leaves to business capability leaves.
    
    Uses bipartite graph structure:
    - Set A: Organizational unit leaves
    - Set B: Business capability leaves
    - Edges: Matches with scores and justifications
    """
    
    def __init__(self, org_graph: nx.DiGraph, cap_graph: nx.DiGraph, llm,
                 embedder=None, config: MatcherConfig = None):
        self.org_graph = org_graph
        self.cap_graph = cap_graph
        self.llm = llm
        self.embedder = embedder or MockEmbedder()
        self.config = config or MatcherConfig()
        
        self.bipartite_graph = nx.Graph()
        self._org_embeddings: Dict[str, List[float]] = {}
        self._cap_embeddings: Dict[str, List[float]] = {}
        self.results: List[MatchResult] = []
        self.unmatched_orgs: List[str] = []
    
    @classmethod
    def load_from_local(cls, org_graph_path: str, cap_graph_path: str, llm,
                        embedder=None, config: MatcherConfig = None) -> 'BipartiteCapabilityMatcher':
        """Load graphs from local JSON files."""
        print("Loading graphs from local files...")
        
        with open(org_graph_path, 'r', encoding='utf-8') as f:
            org_data = json.load(f)
        
        org_graph = nx.DiGraph()
        for node in org_data.get("nodes", []):
            node_id = node.pop("id")
            org_graph.add_node(node_id, **node)
        for edge in org_data.get("edges", []):
            org_graph.add_edge(edge["source"], edge["target"])
        print(f"  Org graph: {org_graph.number_of_nodes()} nodes, {org_graph.number_of_edges()} edges")
        
        with open(cap_graph_path, 'r', encoding='utf-8') as f:
            cap_data = json.load(f)
        
        cap_graph = nx.DiGraph()
        for node in cap_data.get("nodes", []):
            node_id = node.pop("id")
            cap_graph.add_node(node_id, **node)
        for edge in cap_data.get("edges", []):
            cap_graph.add_edge(edge["source"], edge["target"])
        print(f"  Cap graph: {cap_graph.number_of_nodes()} nodes, {cap_graph.number_of_edges()} edges")
        
        return cls(org_graph, cap_graph, llm, embedder, config)
    
    @classmethod
    def load_from_falkordb(cls, falkordb_client, org_graph_name: str, cap_graph_name: str,
                           llm, embedder=None, config: MatcherConfig = None) -> 'BipartiteCapabilityMatcher':
        """Load graphs from FalkorDB."""
        print("Loading graphs from FalkorDB...")
        
        org_db = falkordb_client.select_graph(org_graph_name)
        org_graph = nx.DiGraph()
        
        result = org_db.query("MATCH (n) RETURN n")
        for record in result.result_set:
            node = record[0]
            props = dict(node.properties)
            node_id = props.pop("node_id", str(node.id))
            org_graph.add_node(node_id, **props)
        
        result = org_db.query("MATCH (a)-[r]->(b) RETURN a.node_id, b.node_id")
        for record in result.result_set:
            org_graph.add_edge(record[0], record[1])
        print(f"  Org graph: {org_graph.number_of_nodes()} nodes")
        
        cap_db = falkordb_client.select_graph(cap_graph_name)
        cap_graph = nx.DiGraph()
        
        result = cap_db.query("MATCH (n) RETURN n")
        for record in result.result_set:
            node = record[0]
            props = dict(node.properties)
            node_id = props.pop("node_id", str(node.id))
            cap_graph.add_node(node_id, **props)
        
        result = cap_db.query("MATCH (a)-[r]->(b) RETURN a.node_id, b.node_id")
        for record in result.result_set:
            cap_graph.add_edge(record[0], record[1])
        print(f"  Cap graph: {cap_graph.number_of_nodes()} nodes")
        
        return cls(org_graph, cap_graph, llm, embedder, config)
    
    def _get_leaves(self, graph: nx.DiGraph) -> List[str]:
        """Get leaf nodes (no children)."""
        return [n for n in graph.nodes() if graph.out_degree(n) == 0]
    
    def _get_nodes_at_level(self, graph: nx.DiGraph, level: int) -> List[str]:
        """Get nodes at specific level."""
        return [n for n in graph.nodes() if graph.nodes[n].get("level", 0) == level]
    
    def _get_org_text(self, org_id: str) -> str:
        """Get text representation of org for embedding."""
        data = self.org_graph.nodes[org_id]
        parts = [data.get("name", "")]
        if data.get("refined_description"):
            parts.append(data["refined_description"])
        elif data.get("activities"):
            parts.extend(data["activities"][:3])
        return " ".join(filter(None, parts))
    
    def _get_cap_text(self, cap_id: str) -> str:
        """Get text representation of capability for embedding."""
        data = self.cap_graph.nodes[cap_id]
        parts = [data.get("name", "")]
        if data.get("refined_description"):
            parts.append(data["refined_description"])
        elif data.get("description"):
            parts.append(data["description"])
        if data.get("capability_keywords"):
            parts.append(data["capability_keywords"])
        return " ".join(filter(None, parts))
    
    def _get_capability_path(self, cap_id: str) -> str:
        """Get full path from root to capability."""
        path = [self.cap_graph.nodes[cap_id].get("name", cap_id)]
        current = cap_id
        while True:
            preds = list(self.cap_graph.predecessors(current))
            if not preds:
                break
            parent = preds[0]
            parent_name = self.cap_graph.nodes[parent].get("name", parent)
            if "ROOT" not in parent_name.upper():
                path.insert(0, parent_name)
            current = parent
        return " > ".join(path)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2:
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = np.sqrt(sum(a**2 for a in vec1))
        norm2 = np.sqrt(sum(b**2 for b in vec2))
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    def _keyword_overlap(self, org_id: str, cap_id: str) -> float:
        """Compute keyword overlap score."""
        org_data = self.org_graph.nodes[org_id]
        cap_data = self.cap_graph.nodes[cap_id]
        
        org_text = " ".join(org_data.get("activities", [])).lower()
        if org_data.get("refined_description"):
            org_text += " " + org_data["refined_description"].lower()
        org_words = set(org_text.split())
        
        cap_keywords = cap_data.get("capability_keywords", "").lower()
        cap_name = cap_data.get("name", "").lower()
        cap_words = set((cap_keywords + " " + cap_name).replace(",", " ").split())
        
        if not org_words or not cap_words:
            return 0.0
        
        overlap = len(org_words & cap_words)
        return overlap / max(len(org_words), len(cap_words))
    
    def compute_embeddings(self):
        """Pre-compute embeddings for all nodes."""
        if self.config.verbose:
            print("\nComputing embeddings...")
        
        org_leaves = self._get_leaves(self.org_graph)
        org_texts = [self._get_org_text(n) for n in org_leaves]
        org_embeddings = self.embedder.embed_batch(org_texts)
        for node_id, emb in zip(org_leaves, org_embeddings):
            self._org_embeddings[node_id] = emb
        
        for cap_id in self.cap_graph.nodes():
            cap_text = self._get_cap_text(cap_id)
            self._cap_embeddings[cap_id] = self.embedder.embed(cap_text)
        
        if self.config.verbose:
            print(f"  Org embeddings: {len(self._org_embeddings)}")
            print(f"  Cap embeddings: {len(self._cap_embeddings)}")
    
    def hybrid_search(self, org_id: str, candidate_caps: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Hybrid search combining semantic and keyword similarity."""
        org_emb = self._org_embeddings.get(org_id)
        if not org_emb:
            org_emb = self.embedder.embed(self._get_org_text(org_id))
        
        scores = []
        for cap_id in candidate_caps:
            cap_emb = self._cap_embeddings.get(cap_id)
            if not cap_emb:
                continue
            
            sem_score = self._cosine_similarity(org_emb, cap_emb)
            
            if self.config.use_hybrid_search:
                kw_score = self._keyword_overlap(org_id, cap_id)
                combined = 0.7 * sem_score + 0.3 * kw_score
            else:
                combined = sem_score
            
            scores.append((cap_id, combined))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _parse_llm_judge_response(self, response: str) -> dict:
        """Parse LLM judge response."""
        result = {"llm_score": 0.0, "match_type": "NO_MATCH", "justification": "", "key_overlaps": "", "gaps": ""}
        
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("MATCH_SCORE:"):
                try:
                    score = float(line.split(":", 1)[1].strip())
                    result["llm_score"] = min(max(score, 0.0), 1.0)
                except:
                    pass
            elif line.startswith("MATCH_TYPE:"):
                result["match_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("JUSTIFICATION:"):
                result["justification"] = line.split(":", 1)[1].strip()
            elif line.startswith("KEY_OVERLAPS:"):
                result["key_overlaps"] = line.split(":", 1)[1].strip()
            elif line.startswith("GAPS:"):
                result["gaps"] = line.split(":", 1)[1].strip()
        
        return result
    
    def llm_evaluate_match(self, org_id: str, cap_id: str) -> dict:
        """Use LLM to evaluate a potential match."""
        org_data = self.org_graph.nodes[org_id]
        cap_data = self.cap_graph.nodes[cap_id]
        
        activities = org_data.get("activities", [])
        weights = org_data.get("activity_weights", [])
        activities_text = "\n".join([f"  - ({w}%) {a}" for a, w in zip(activities, weights)]) if activities else "No activities listed"
        
        prompt = LLM_JUDGE_PROMPT.format(
            org_name=org_data.get("name", org_id),
            org_code=org_id,
            org_level=org_data.get("level", 0),
            org_activities=activities_text,
            org_description=org_data.get("refined_description", org_data.get("description", "Not available")),
            cap_name=cap_data.get("name", cap_id),
            cap_id=cap_id,
            cap_level=cap_data.get("level", 0),
            cap_type=cap_data.get("node_type", "capability"),
            cap_path=self._get_capability_path(cap_id),
            cap_description=cap_data.get("refined_description", cap_data.get("description", "Not available")),
            cap_keywords=cap_data.get("capability_keywords", "Not available")
        )
        
        response = self.llm.generate(prompt, LLM_JUDGE_SYSTEM)
        return self._parse_llm_judge_response(response)
    
    # =========================================================================
    # LLM-ONLY MATCHING METHODS
    # =========================================================================
    
    def _format_capabilities_for_llm(self, cap_ids: List[str], max_caps: int = 20) -> str:
        """Format capabilities list for LLM prompt."""
        lines = []
        for cap_id in cap_ids[:max_caps]:
            cap_data = self.cap_graph.nodes.get(cap_id, {})
            name = cap_data.get("name", cap_id)
            desc = cap_data.get("refined_description", cap_data.get("description", ""))[:150]
            path = self._get_capability_path(cap_id)
            lines.append(f"[{cap_id}] {name}")
            lines.append(f"    Path: {path}")
            if desc:
                lines.append(f"    Description: {desc}...")
            lines.append("")
        
        if len(cap_ids) > max_caps:
            lines.append(f"... and {len(cap_ids) - max_caps} more capabilities")
        
        return "\n".join(lines)
    
    def _format_org_activities(self, org_id: str) -> str:
        """Format org activities for LLM prompt."""
        org_data = self.org_graph.nodes[org_id]
        activities = org_data.get("activities", [])
        weights = org_data.get("activity_weights", [])
        
        if not activities:
            return "No activities documented"
        
        lines = []
        for act, wt in zip(activities, weights):
            lines.append(f"  - ({wt}%) {act[:200]}")
        return "\n".join(lines)
    
    def _parse_batch_ranking_response(self, response: str) -> List[dict]:
        """Parse LLM batch ranking response."""
        matches = []
        current_match = {}
        
        for line in response.split("\n"):
            line = line.strip()
            
            if line.startswith("MATCH_") and ":" not in line[6:8]:
                # New match section
                if current_match.get("cap_id"):
                    matches.append(current_match)
                current_match = {}
            elif line.startswith("CAP_ID:"):
                current_match["cap_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("SCORE:"):
                try:
                    current_match["score"] = float(line.split(":", 1)[1].strip())
                except:
                    current_match["score"] = 0.0
            elif line.startswith("MATCH_TYPE:"):
                current_match["match_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("JUSTIFICATION:"):
                current_match["justification"] = line.split(":", 1)[1].strip()
            elif line.startswith("KEY_OVERLAPS:"):
                current_match["key_overlaps"] = line.split(":", 1)[1].strip()
        
        # Don't forget last match
        if current_match.get("cap_id"):
            matches.append(current_match)
        
        return matches
    
    def _parse_category_screen_response(self, response: str) -> List[str]:
        """Parse LLM category screening response."""
        categories = []
        
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SELECTED_CATEGORIES:"):
                cats_str = line.split(":", 1)[1].strip()
                # Parse comma-separated list, handling brackets
                cats_str = cats_str.replace("[", "").replace("]", "")
                categories = [c.strip() for c in cats_str.split(",") if c.strip()]
                break
        
        return categories
    
    def llm_screen_categories(self, org_id: str) -> List[str]:
        """Use LLM to identify relevant L1 categories for an org."""
        org_data = self.org_graph.nodes[org_id]
        
        # Get L1 categories
        l1_caps = [n for n in self.cap_graph.nodes() 
                   if self.cap_graph.nodes[n].get("level", -1) == 0]
        
        if not l1_caps:
            return []
        
        # Format categories
        cat_lines = []
        for cap_id in l1_caps:
            cap_data = self.cap_graph.nodes[cap_id]
            name = cap_data.get("name", cap_id)
            desc = cap_data.get("description", "")[:100]
            cat_lines.append(f"[{cap_id}] {name}: {desc}")
        
        prompt = LLM_CATEGORY_SCREEN_PROMPT.format(
            org_name=org_data.get("name", org_id),
            org_code=org_id,
            org_activities=self._format_org_activities(org_id),
            categories_list="\n".join(cat_lines)
        )
        
        response = self.llm.generate(prompt, LLM_CATEGORY_SCREEN_SYSTEM)
        return self._parse_category_screen_response(response)
    
    def llm_rank_capabilities(self, org_id: str, cap_ids: List[str]) -> List[dict]:
        """Use LLM to rank capabilities for an org (batch evaluation)."""
        org_data = self.org_graph.nodes[org_id]
        
        prompt = LLM_BATCH_RANKING_PROMPT.format(
            org_name=org_data.get("name", org_id),
            org_code=org_id,
            org_activities=self._format_org_activities(org_id),
            org_description=org_data.get("refined_description", 
                           org_data.get("description", "Not available")),
            capabilities_list=self._format_capabilities_for_llm(cap_ids)
        )
        
        response = self.llm.generate(prompt, LLM_BATCH_RANKING_SYSTEM)
        return self._parse_batch_ranking_response(response)
    
    def match_org_llm_only(self, org_id: str, cap_candidates: List[str], 
                          fallback_level: int = 0) -> MatchResult:
        """Match org to capabilities using LLM only (no embeddings)."""
        org_data = self.org_graph.nodes[org_id]
        result = MatchResult(
            org_id=org_id,
            org_name=org_data.get("name", org_id),
            org_level=org_data.get("level", 0),
            fallback_level=fallback_level
        )
        
        # Process in batches to avoid token limits
        batch_size = self.config.llm_batch_size
        all_matches = []
        
        for i in range(0, len(cap_candidates), batch_size):
            batch = cap_candidates[i:i + batch_size]
            batch_matches = self.llm_rank_capabilities(org_id, batch)
            all_matches.extend(batch_matches)
        
        # Sort by score and take top matches
        all_matches.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        for match in all_matches:
            cap_id = match.get("cap_id", "")
            score = match.get("score", 0)
            
            # Validate cap_id exists
            if cap_id not in self.cap_graph.nodes:
                # Try to find by partial match
                for cid in cap_candidates:
                    if cap_id in cid or cid in cap_id:
                        cap_id = cid
                        break
                else:
                    continue
            
            if score >= self.config.min_llm_score:
                cap_data = self.cap_graph.nodes[cap_id]
                result.add_match(
                    cap_id=cap_id,
                    cap_name=cap_data.get("name", cap_id),
                    cap_level=cap_data.get("level", 0),
                    semantic_score=0.0,  # No semantic score in LLM-only mode
                    llm_score=score,
                    match_type=match.get("match_type", "MODERATE"),
                    justification=match.get("justification", ""),
                    key_overlaps=match.get("key_overlaps", ""),
                    gaps=""
                )
        
        if not result.matches:
            result.unmatched = True
        
        return result
    
    def run_llm_only_matching(self, target_org_ids: List[str] = None) -> List[MatchResult]:
        """Run LLM-only matching (no embedding pre-filtering)."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("LLM-ONLY CAPABILITY MATCHING")
            print("═" * 70)
            print("Mode: Full LLM evaluation (no semantic pre-filtering)")
        
        # Get org leaves
        if target_org_ids:
            org_leaves = [o for o in target_org_ids if o in self.org_graph.nodes]
        else:
            org_leaves = self._get_leaves(self.org_graph)
        
        cap_leaves = self._get_leaves(self.cap_graph)
        
        if self.config.verbose:
            print(f"\nOrg leaves to match: {len(org_leaves)}")
            print(f"Cap leaves to search: {len(cap_leaves)}")
            total_calls = len(org_leaves) * ((len(cap_leaves) // self.config.llm_batch_size) + 1)
            print(f"Estimated LLM calls: ~{total_calls}")
            print("\n--- LLM-Only Matching ---")
        
        self.results = []
        self.unmatched_orgs = []
        
        for i, org_id in enumerate(org_leaves):
            if self.config.verbose:
                org_name = self.org_graph.nodes[org_id].get("name", org_id)[:35]
                print(f"\n[{i+1}/{len(org_leaves)}] Evaluating: {org_id} - {org_name}")
            
            result = self.match_org_llm_only(org_id, cap_leaves)
            self.results.append(result)
            
            if result.unmatched:
                self.unmatched_orgs.append(org_id)
                if self.config.verbose:
                    print(f"  → NO MATCH FOUND")
            elif self.config.verbose:
                best = result.best_match()
                print(f"  → {best['capability_name'][:35]}")
                print(f"    Score: {best['llm_score']:.2f} ({best['match_type']})")
        
        # Hierarchical fallback for unmatched
        if self.config.enable_hierarchical_fallback and self.unmatched_orgs:
            if self.config.verbose:
                print(f"\n--- Hierarchical Fallback ({len(self.unmatched_orgs)} unmatched) ---")
            
            for level in range(1, self.config.max_fallback_levels + 1):
                if not self.unmatched_orgs:
                    break
                
                max_cap_level = max(self.cap_graph.nodes[n].get("level", 0) 
                                   for n in self.cap_graph.nodes())
                target_level = max_cap_level - level
                if target_level < 0:
                    break
                
                higher_caps = self._get_nodes_at_level(self.cap_graph, target_level)
                
                if self.config.verbose:
                    print(f"\nTrying L{target_level} capabilities ({len(higher_caps)} nodes)...")
                
                still_unmatched = []
                for org_id in self.unmatched_orgs:
                    result = self.match_org_llm_only(org_id, higher_caps, fallback_level=level)
                    
                    # Update existing result
                    for r in self.results:
                        if r.org_id == org_id:
                            r.matches = result.matches
                            r.unmatched = result.unmatched
                            r.fallback_level = level
                            break
                    
                    if result.unmatched:
                        still_unmatched.append(org_id)
                    elif self.config.verbose:
                        best = result.best_match()
                        org_name = self.org_graph.nodes[org_id].get("name", "")[:30]
                        print(f"  {org_id}: {org_name} → {best['capability_name'][:30]}")
                
                self.unmatched_orgs = still_unmatched
        
        self._build_bipartite_graph()
        
        matched = sum(1 for r in self.results if not r.unmatched)
        if self.config.verbose:
            print(f"\n{'═' * 70}")
            print(f"LLM-ONLY MATCHING COMPLETE")
            print(f"  Matched: {matched}/{len(self.results)}")
            print(f"  Unmatched: {len(self.unmatched_orgs)}")
            print(f"{'═' * 70}")
        
        return self.results
    
    def run_llm_prescreened_matching(self, target_org_ids: List[str] = None) -> List[MatchResult]:
        """Run LLM matching with category pre-screening (efficient LLM-only)."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("LLM PRE-SCREENED CAPABILITY MATCHING")
            print("═" * 70)
            print("Mode: LLM category screening → LLM detailed matching")
        
        # Get org leaves
        if target_org_ids:
            org_leaves = [o for o in target_org_ids if o in self.org_graph.nodes]
        else:
            org_leaves = self._get_leaves(self.org_graph)
        
        if self.config.verbose:
            print(f"\nOrg leaves to match: {len(org_leaves)}")
            print("\n--- Phase 1: Category Screening ---")
        
        self.results = []
        self.unmatched_orgs = []
        
        for i, org_id in enumerate(org_leaves):
            org_name = self.org_graph.nodes[org_id].get("name", org_id)[:35]
            
            if self.config.verbose:
                print(f"\n[{i+1}/{len(org_leaves)}] {org_id}: {org_name}")
            
            # Step 1: Screen categories
            relevant_cats = self.llm_screen_categories(org_id)
            
            if self.config.verbose:
                print(f"  Relevant categories: {relevant_cats}")
            
            # Step 2: Get capabilities under those categories
            if relevant_cats:
                relevant_caps = []
                for cat_id in relevant_cats:
                    if cat_id in self.cap_graph.nodes:
                        # Get all descendants
                        descendants = nx.descendants(self.cap_graph, cat_id)
                        # Filter to leaves
                        leaves = [d for d in descendants if self.cap_graph.out_degree(d) == 0]
                        relevant_caps.extend(leaves)
                relevant_caps = list(set(relevant_caps))  # Dedupe
            else:
                # Fallback to all leaves
                relevant_caps = self._get_leaves(self.cap_graph)
            
            if self.config.verbose:
                print(f"  Capabilities to evaluate: {len(relevant_caps)}")
            
            # Step 3: LLM match within filtered set
            result = self.match_org_llm_only(org_id, relevant_caps)
            self.results.append(result)
            
            if result.unmatched:
                self.unmatched_orgs.append(org_id)
                if self.config.verbose:
                    print(f"  → NO MATCH FOUND")
            elif self.config.verbose:
                best = result.best_match()
                print(f"  → {best['capability_name'][:35]} ({best['llm_score']:.2f})")
        
        self._build_bipartite_graph()
        
        matched = sum(1 for r in self.results if not r.unmatched)
        if self.config.verbose:
            print(f"\n{'═' * 70}")
            print(f"PRE-SCREENED MATCHING COMPLETE")
            print(f"  Matched: {matched}/{len(self.results)}")
            print(f"  Unmatched: {len(self.unmatched_orgs)}")
            print(f"{'═' * 70}")
        
        return self.results
    
    def match_org_to_capabilities(self, org_id: str, cap_candidates: List[str], fallback_level: int = 0) -> MatchResult:
        """Match a single org unit to capabilities."""
        org_data = self.org_graph.nodes[org_id]
        result = MatchResult(
            org_id=org_id,
            org_name=org_data.get("name", org_id),
            org_level=org_data.get("level", 0),
            fallback_level=fallback_level
        )
        
        candidates = self.hybrid_search(org_id, cap_candidates, self.config.top_k_candidates)
        candidates = [(c, s) for c, s in candidates if s >= self.config.min_semantic_score]
        
        if not candidates:
            result.unmatched = True
            return result
        
        for cap_id, semantic_score in candidates:
            if self.config.use_llm_judge:
                llm_result = self.llm_evaluate_match(org_id, cap_id)
                
                if llm_result["llm_score"] >= self.config.min_llm_score:
                    cap_data = self.cap_graph.nodes[cap_id]
                    result.add_match(
                        cap_id=cap_id,
                        cap_name=cap_data.get("name", cap_id),
                        cap_level=cap_data.get("level", 0),
                        semantic_score=semantic_score,
                        llm_score=llm_result["llm_score"],
                        match_type=llm_result["match_type"],
                        justification=llm_result["justification"],
                        key_overlaps=llm_result["key_overlaps"],
                        gaps=llm_result["gaps"]
                    )
            else:
                cap_data = self.cap_graph.nodes[cap_id]
                match_type = "STRONG" if semantic_score > 0.7 else "MODERATE" if semantic_score > 0.5 else "WEAK"
                result.add_match(
                    cap_id=cap_id,
                    cap_name=cap_data.get("name", cap_id),
                    cap_level=cap_data.get("level", 0),
                    semantic_score=semantic_score,
                    llm_score=semantic_score,
                    match_type=match_type,
                    justification=f"Matched by semantic similarity (score: {semantic_score:.3f})"
                )
        
        if not result.matches:
            result.unmatched = True
        
        return result
    
    def run_matching(self) -> List[MatchResult]:
        """Run bipartite matching for all org leaves."""
        if self.config.verbose:
            print("\n" + "═" * 70)
            print("BIPARTITE CAPABILITY MATCHING")
            print("═" * 70)
        
        if not self._org_embeddings:
            self.compute_embeddings()
        
        org_leaves = self._get_leaves(self.org_graph)
        cap_leaves = self._get_leaves(self.cap_graph)
        
        if self.config.verbose:
            print(f"\nOrg leaves: {len(org_leaves)}")
            print(f"Cap leaves: {len(cap_leaves)}")
            print("\n--- Phase 1: Leaf-to-Leaf Matching ---")
        
        self.results = []
        self.unmatched_orgs = []
        
        for i, org_id in enumerate(org_leaves):
            result = self.match_org_to_capabilities(org_id, cap_leaves)
            self.results.append(result)
            
            if result.unmatched:
                self.unmatched_orgs.append(org_id)
            
            if self.config.verbose:
                org_name = self.org_graph.nodes[org_id].get("name", org_id)[:30]
                if result.matches:
                    best = result.best_match()
                    print(f"[{i+1}/{len(org_leaves)}] {org_id}: {org_name}")
                    print(f"  → {best['capability_name'][:30]} ({best['combined_score']:.3f} {best['match_type']})")
                else:
                    print(f"[{i+1}/{len(org_leaves)}] {org_id}: {org_name} [UNMATCHED]")
        
        # Hierarchical fallback
        if self.config.enable_hierarchical_fallback and self.unmatched_orgs:
            if self.config.verbose:
                print(f"\n--- Phase 2: Hierarchical Fallback ({len(self.unmatched_orgs)} unmatched) ---")
            
            for level in range(1, self.config.max_fallback_levels + 1):
                if not self.unmatched_orgs:
                    break
                
                max_cap_level = max(self.cap_graph.nodes[n].get("level", 0) for n in self.cap_graph.nodes())
                target_level = max_cap_level - level
                if target_level < 0:
                    break
                
                higher_caps = self._get_nodes_at_level(self.cap_graph, target_level)
                
                if self.config.verbose:
                    print(f"\nTrying level {target_level} capabilities ({len(higher_caps)} nodes)...")
                
                still_unmatched = []
                for org_id in self.unmatched_orgs:
                    result = self.match_org_to_capabilities(org_id, higher_caps, fallback_level=level)
                    
                    for r in self.results:
                        if r.org_id == org_id:
                            r.matches = result.matches
                            r.unmatched = result.unmatched
                            r.fallback_level = level
                            break
                    
                    if result.unmatched:
                        still_unmatched.append(org_id)
                    elif self.config.verbose:
                        best = result.best_match()
                        org_name = self.org_graph.nodes[org_id].get("name", org_id)[:30]
                        print(f"  {org_id}: {org_name} → {best['capability_name'][:30]} (L{target_level})")
                
                self.unmatched_orgs = still_unmatched
        
        self._build_bipartite_graph()
        
        matched = sum(1 for r in self.results if not r.unmatched)
        if self.config.verbose:
            print(f"\n✓ Matching complete: {matched}/{len(org_leaves)} matched, {len(self.unmatched_orgs)} unmatched")
        
        return self.results
    
    def run(self, target_org_ids: List[str] = None) -> List[MatchResult]:
        """
        Unified entry point - runs matching based on config.matching_mode.
        
        Modes:
        - "hybrid": Semantic pre-filter + LLM judge (default, balanced)
        - "llm_only": Pure LLM matching (most accurate, expensive)
        - "llm_prescreened": LLM category filter + LLM matching (efficient LLM-only)
        - "semantic_only": Embedding similarity only (fastest, least accurate)
        
        Args:
            target_org_ids: Optional list of specific org IDs to match.
                           If None, matches all org leaves.
        
        Returns:
            List of MatchResult objects
        """
        mode = self.config.matching_mode
        
        if self.config.verbose:
            print(f"\n{'═' * 70}")
            print(f"MATCHING MODE: {mode.upper()}")
            print(f"{'═' * 70}")
        
        if mode == "llm_only":
            return self.run_llm_only_matching(target_org_ids)
        
        elif mode == "llm_prescreened":
            return self.run_llm_prescreened_matching(target_org_ids)
        
        elif mode == "semantic_only":
            # Disable LLM judge for semantic-only
            original_llm_judge = self.config.use_llm_judge
            self.config.use_llm_judge = False
            try:
                if target_org_ids:
                    # Filter to specific orgs
                    org_leaves = [o for o in target_org_ids if o in self.org_graph.nodes]
                    cap_leaves = self._get_leaves(self.cap_graph)
                    
                    if not self._org_embeddings:
                        self.compute_embeddings()
                    
                    self.results = []
                    self.unmatched_orgs = []
                    
                    for org_id in org_leaves:
                        result = self.match_org_to_capabilities(org_id, cap_leaves)
                        self.results.append(result)
                        if result.unmatched:
                            self.unmatched_orgs.append(org_id)
                    
                    self._build_bipartite_graph()
                    return self.results
                else:
                    return self.run_matching()
            finally:
                self.config.use_llm_judge = original_llm_judge
        
        else:  # "hybrid" (default)
            if target_org_ids:
                # Run hybrid matching on specific orgs
                org_leaves = [o for o in target_org_ids if o in self.org_graph.nodes]
                cap_leaves = self._get_leaves(self.cap_graph)
                
                if not self._org_embeddings:
                    self.compute_embeddings()
                
                self.results = []
                self.unmatched_orgs = []
                
                if self.config.verbose:
                    print(f"\nMatching {len(org_leaves)} orgs against {len(cap_leaves)} capabilities")
                
                for i, org_id in enumerate(org_leaves):
                    result = self.match_org_to_capabilities(org_id, cap_leaves)
                    self.results.append(result)
                    
                    if result.unmatched:
                        self.unmatched_orgs.append(org_id)
                    
                    if self.config.verbose:
                        org_name = self.org_graph.nodes[org_id].get("name", org_id)[:30]
                        if result.matches:
                            best = result.best_match()
                            print(f"[{i+1}/{len(org_leaves)}] {org_id}: {org_name}")
                            print(f"  → {best['capability_name'][:30]} ({best['combined_score']:.3f})")
                        else:
                            print(f"[{i+1}/{len(org_leaves)}] {org_id}: {org_name} [UNMATCHED]")
                
                self._build_bipartite_graph()
                
                matched = sum(1 for r in self.results if not r.unmatched)
                if self.config.verbose:
                    print(f"\n✓ Hybrid matching complete: {matched}/{len(org_leaves)} matched")
                
                return self.results
            else:
                return self.run_matching()
    
    def _build_bipartite_graph(self):
        """Build NetworkX bipartite graph from results."""
        self.bipartite_graph = nx.Graph()
        
        for result in self.results:
            self.bipartite_graph.add_node(
                result.org_id, bipartite=0, node_type="org",
                name=result.org_name, level=result.org_level
            )
        
        for result in self.results:
            for match in result.matches:
                cap_id = match["capability_id"]
                if cap_id not in self.bipartite_graph:
                    self.bipartite_graph.add_node(
                        cap_id, bipartite=1, node_type="capability",
                        name=match["capability_name"], level=match["capability_level"]
                    )
                
                self.bipartite_graph.add_edge(
                    result.org_id, cap_id,
                    semantic_score=match["semantic_score"],
                    llm_score=match["llm_score"],
                    combined_score=match["combined_score"],
                    match_type=match["match_type"],
                    justification=match["justification"],
                    key_overlaps=match.get("key_overlaps", ""),
                    gaps=match.get("gaps", "")
                )
    
    def save_results(self, path: str) -> str:
        """Save matching results to JSON."""
        output = {
            "metadata": {
                "total_orgs": len(self.results),
                "matched": sum(1 for r in self.results if not r.unmatched),
                "unmatched": len(self.unmatched_orgs),
                "config": {
                    "matching_mode": self.config.matching_mode,
                    "min_semantic_score": self.config.min_semantic_score,
                    "min_llm_score": self.config.min_llm_score,
                    "top_k_candidates": self.config.top_k_candidates,
                    "llm_batch_size": self.config.llm_batch_size,
                    "use_llm_judge": self.config.use_llm_judge,
                    "hierarchical_fallback": self.config.enable_hierarchical_fallback
                }
            },
            "matches": [r.to_dict() for r in self.results],
            "unmatched_orgs": self.unmatched_orgs
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved: {path}")
        return path
    
    def save_bipartite_graph(self, path: str) -> str:
        """Save bipartite graph to JSON."""
        graph_data = {
            "metadata": {
                "org_nodes": sum(1 for n in self.bipartite_graph.nodes() if self.bipartite_graph.nodes[n].get("bipartite") == 0),
                "cap_nodes": sum(1 for n in self.bipartite_graph.nodes() if self.bipartite_graph.nodes[n].get("bipartite") == 1),
                "edges": self.bipartite_graph.number_of_edges()
            },
            "nodes": [{"id": n, **dict(d)} for n, d in self.bipartite_graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, **dict(d)} for u, v, d in self.bipartite_graph.edges(data=True)]
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Bipartite graph saved: {path}")
        return path
    
    def sync_to_falkordb(self, falkordb_client) -> int:
        """Sync bipartite matches to FalkorDB."""
        if not self.config.sync_to_falkordb:
            return 0
        
        print("\nSyncing matches to FalkorDB...")
        
        try:
            graph = falkordb_client.select_graph(self.config.bipartite_graph_name)
            
            try:
                graph.query("MATCH (n) DETACH DELETE n")
            except:
                pass
            
            count = 0
            for result in self.results:
                for match in result.matches:
                    query = """
                    MERGE (o:OrgUnit {node_id: $org_id})
                    SET o.name = $org_name, o.level = $org_level
                    MERGE (c:Capability {node_id: $cap_id})
                    SET c.name = $cap_name, c.level = $cap_level
                    MERGE (o)-[r:MATCHES_CAPABILITY]->(c)
                    SET r.semantic_score = $semantic_score,
                        r.llm_score = $llm_score,
                        r.combined_score = $combined_score,
                        r.match_type = $match_type,
                        r.justification = $justification
                    """
                    graph.query(query, {
                        "org_id": result.org_id,
                        "org_name": result.org_name,
                        "org_level": result.org_level,
                        "cap_id": match["capability_id"],
                        "cap_name": match["capability_name"],
                        "cap_level": match["capability_level"],
                        "semantic_score": match["semantic_score"],
                        "llm_score": match["llm_score"],
                        "combined_score": match["combined_score"],
                        "match_type": match["match_type"],
                        "justification": match["justification"]
                    })
                    count += 1
            
            print(f"✓ Synced {count} matches to FalkorDB")
            return count
            
        except Exception as e:
            print(f"⚠ FalkorDB sync failed: {e}")
            return 0
    
    def print_summary(self):
        """Print matching summary."""
        print("\n" + "═" * 70)
        print("MATCHING SUMMARY")
        print("═" * 70)
        
        matched = [r for r in self.results if not r.unmatched]
        
        print(f"\nTotal org leaves: {len(self.results)}")
        print(f"Matched: {len(matched)}")
        print(f"Unmatched: {len(self.unmatched_orgs)}")
        
        type_counts = {}
        for r in matched:
            best = r.best_match()
            if best:
                t = best["match_type"]
                type_counts[t] = type_counts.get(t, 0) + 1
        
        if type_counts:
            print("\nMatch types:")
            for t, c in sorted(type_counts.items()):
                print(f"  {t}: {c}")
        
        print("\n" + "─" * 70)
        print("SAMPLE MATCHES (Top 3)")
        print("─" * 70)
        
        for r in matched[:3]:
            best = r.best_match()
            if best:
                print(f"\n[{r.org_id}] {r.org_name}")
                print(f"  → [{best['capability_id']}] {best['capability_name']}")
                print(f"  Score: {best['combined_score']:.3f} ({best['match_type']})")
                if best.get('justification'):
                    print(f"  Justification: {best['justification'][:100]}...")


def create_bipartite_matcher(
    org_graph: nx.DiGraph, cap_graph: nx.DiGraph, llm,
    matching_mode: str = "hybrid",
    use_mock_embedder: bool = True,
    use_llm_judge: bool = True,
    min_semantic_score: float = 0.3,
    min_llm_score: float = 0.6,
    top_k_candidates: int = 5,
    llm_batch_size: int = 10,
    enable_fallback: bool = True,
    sync_to_falkordb: bool = True,
    verbose: bool = True
) -> BipartiteCapabilityMatcher:
    """
    Create BipartiteCapabilityMatcher instance.
    
    Args:
        matching_mode: "hybrid", "llm_only", "llm_prescreened", or "semantic_only"
        use_mock_embedder: Use mock (hash-based) or real HuggingFace embeddings
        use_llm_judge: Use LLM for final evaluation in hybrid mode
        min_semantic_score: Minimum semantic similarity threshold
        min_llm_score: Minimum LLM score threshold
        top_k_candidates: Number of candidates for semantic pre-filter
        llm_batch_size: Capabilities per LLM batch in LLM-only mode
        enable_fallback: Enable hierarchical fallback for unmatched orgs
        sync_to_falkordb: Sync results to FalkorDB
        verbose: Print progress
    """
    embedder = MockEmbedder() if use_mock_embedder else HuggingFaceEmbedder()
    config = MatcherConfig(
        matching_mode=matching_mode,
        min_semantic_score=min_semantic_score,
        min_llm_score=min_llm_score,
        top_k_candidates=top_k_candidates,
        llm_batch_size=llm_batch_size,
        use_llm_judge=use_llm_judge,
        enable_hierarchical_fallback=enable_fallback,
        sync_to_falkordb=sync_to_falkordb,
        verbose=verbose
    )
    return BipartiteCapabilityMatcher(org_graph, cap_graph, llm, embedder, config)


def create_matcher_from_local(
    org_graph_path: str, cap_graph_path: str, llm,
    matching_mode: str = "hybrid",
    use_mock_embedder: bool = True,
    **kwargs
) -> BipartiteCapabilityMatcher:
    """Create matcher loading graphs from local files."""
    embedder = MockEmbedder() if use_mock_embedder else HuggingFaceEmbedder()
    config = MatcherConfig(matching_mode=matching_mode, **kwargs)
    return BipartiteCapabilityMatcher.load_from_local(org_graph_path, cap_graph_path, llm, embedder, config)


if __name__ == "__main__":
    print("BipartiteCapabilityMatcher module loaded.")
    print()
    print("Matching modes:")
    print("  - hybrid: Semantic pre-filter + LLM judge (default)")
    print("  - llm_only: Pure LLM matching (most accurate)")
    print("  - llm_prescreened: LLM category filter + LLM matching")
    print("  - semantic_only: Embedding similarity only (fastest)")
    print()
    print("Usage:")
    print("  matcher = create_bipartite_matcher(org_graph, cap_graph, llm, matching_mode='llm_only')")
    print("  results = matcher.run()")

