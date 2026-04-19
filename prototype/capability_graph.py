"""
Capability Graph Builder
Builds business capability hierarchy from the EP Business Map workbook.
- 3-level hierarchy: Category → Business Area → Sub-Business Area
- Dual labels: :Capability:Category, :Capability:BusinessArea, :Capability:SubBusinessArea
- Shares the same FalkorDB graph as the org hierarchy (org_hierarchy)
- Vector index on description_embedding for semantic matching

Usage:
    from capability_graph import create_capability_builder
    builder = create_capability_builder(falkordb_url=FALKORDB_URL)
    builder.build_from_file("data/Capability/EP_Business_Map_5.0.xlsx")
    builder.upload_to_falkordb()
"""

import json
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import networkx as nx


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_DATA_PATH = "data/Capability/EP_Business_Map_5.0.xlsx"
DEFAULT_SHEET_NAME = "Business Map"

# Column schema for EP_Business_Map_5.0.xlsx
COL_L1_NAME = "Category"
COL_L1_DESC = "Category Description"
COL_L2_NAME = "Business Area"
COL_L2_DESC = "Business Area Description"
COL_L3_NAME = "Sub-Business Area"
COL_L3_DESC = "Sub-Business Area Definition"
COL_L3_LEGACY = "Old Sub-Business Area Definition"


@dataclass
class CapabilityGraphConfig:
    """Configuration for capability graph building."""
    # Master root
    master_root_id: str = "CAPABILITY_ROOT"
    master_root_name: str = "EP Business Capability Map"

    # Local storage
    local_graph_path: str = "./graph_data/capability_graph.json"

    # FalkorDB connection (same graph as org hierarchy)
    falkordb_url: Optional[str] = None
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: Optional[str] = None
    falkordb_graph_name: str = "org_hierarchy"

    # Vector index
    vector_index_name: str = "capability_vector_idx"
    vector_dimension: int = 384
    vector_similarity: str = "cosine"

    verbose: bool = True


# =============================================================================
# Capability Graph Builder
# =============================================================================

class CapabilityGraphBuilder:
    """
    Builds business capability hierarchy from the EP Business Map workbook.
    3-level structure: Category (L0) → Business Area (L1) → Sub-Business Area (L2)

    Shares the same FalkorDB graph (org_hierarchy) as the org structure.
    Capability nodes use distinct labels (:Capability, :Category, etc.).
    """

    def __init__(self, config: CapabilityGraphConfig = None):
        self.config = config or CapabilityGraphConfig()
        self.graph = nx.DiGraph()
        self.categories: List[str] = []
        self.embeddings: Dict[str, List[float]] = {}

        # FalkorDB direct connection
        self._client = None
        self._fgraph = None
        self._connected = False

    # -----------------------------------------------------------------
    # FalkorDB Connection
    # -----------------------------------------------------------------

    def connect(self) -> bool:
        """Connect to FalkorDB capability graph."""
        try:
            from falkordb import FalkorDB
        except ImportError:
            print("⚠ FalkorDB not available (pip install falkordb)")
            return False

        try:
            if self.config.falkordb_url:
                self._client = FalkorDB.from_url(self.config.falkordb_url)
            else:
                self._client = FalkorDB(
                    host=self.config.falkordb_host,
                    port=self.config.falkordb_port,
                    password=self.config.falkordb_password
                )

            self._fgraph = self._client.select_graph(self.config.falkordb_graph_name)
            self._connected = True
            print(f"✓ FalkorDB connected: {self.config.falkordb_graph_name}")
            return True

        except Exception as e:
            print(f"✗ FalkorDB connection failed: {e}")
            return False

    def query(self, cypher: str, params: dict = None) -> List[Dict[str, Any]]:
        """Execute Cypher query and return list of dicts."""
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
                row_dict = {}
                for key, val in zip(headers, row):
                    if hasattr(val, 'properties'):
                        row_dict[key] = dict(val.properties)
                    else:
                        row_dict[key] = val
                rows.append(row_dict)
            return rows
        except Exception as e:
            if self.config.verbose:
                print(f"Query error: {e}")
            return []

    # -----------------------------------------------------------------
    # File Loading
    # -----------------------------------------------------------------

    @staticmethod
    def _generate_id(*parts) -> str:
        """Generate unique ID from parts."""
        content = ":".join(str(p) for p in parts)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    @staticmethod
    def _clean_text(text) -> str:
        if pd.isna(text):
            return ""
        return str(text).strip()

    @staticmethod
    def _read_source(path: str, sheet_name: str = DEFAULT_SHEET_NAME) -> pd.DataFrame:
        """Load capability data from XLSX or CSV based on file extension."""
        suffix = Path(path).suffix.lower()
        if suffix in (".xlsx", ".xlsm", ".xls"):
            return pd.read_excel(path, sheet_name=sheet_name)
        if suffix == ".csv":
            try:
                return pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="utf-8")
        raise ValueError(f"Unsupported capability source extension: {suffix}")

    def build_from_file(self, path: str = DEFAULT_DATA_PATH,
                        sheet_name: str = DEFAULT_SHEET_NAME) -> nx.DiGraph:
        """Build capability graph from the EP Business Map workbook (or CSV variant).

        Expected columns:
            Category | Category Description |
            Business Area | Business Area Description |
            Sub-Business Area | Sub-Business Area Definition |
            Old Sub-Business Area Definition (optional fallback)
        """
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("BUILDING CAPABILITY GRAPH")
            print("=" * 60)
            print(f"Source: {path}")

        df = self._read_source(path, sheet_name=sheet_name)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

        required = [COL_L1_NAME, COL_L2_NAME, COL_L3_NAME]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Capability source missing required columns {missing}. "
                f"Found: {df.columns.tolist()}"
            )

        has_legacy = COL_L3_LEGACY in df.columns

        if self.config.verbose:
            print(f"Rows: {len(df)}  |  Columns: {df.columns.tolist()}")

        self.graph = nx.DiGraph()
        self.categories = []

        l1_seen, l2_seen, l3_seen = {}, {}, {}

        for _, row in df.iterrows():
            # Level 0 — Category
            l1_name = self._clean_text(row.get(COL_L1_NAME))
            l1_desc = self._clean_text(row.get(COL_L1_DESC))

            if not l1_name:
                continue

            if l1_name not in l1_seen:
                l1_id = f"L1_{l1_name.upper().replace(' ', '_')[:20]}_{self._generate_id(l1_name)[:6]}"
                l1_seen[l1_name] = l1_id
                self.graph.add_node(l1_id,
                    name=l1_name, description=l1_desc,
                    level=0, node_type="category",
                    type="capability"
                )
                self.categories.append(l1_id)

            l1_id = l1_seen[l1_name]

            # Level 1 — Business Area
            l2_name = self._clean_text(row.get(COL_L2_NAME))
            l2_desc = self._clean_text(row.get(COL_L2_DESC))
            l2_key = f"{l1_name}:{l2_name}"

            if l2_name and l2_key not in l2_seen:
                l2_id = f"L2_{self._generate_id(l1_name, l2_name)}"
                l2_seen[l2_key] = l2_id
                self.graph.add_node(l2_id,
                    name=l2_name, description=l2_desc,
                    level=1, node_type="business_area",
                    type="capability",
                    parent_category=l1_name
                )
                self.graph.add_edge(l1_id, l2_id, edge_type="HAS_SUBCAPABILITY")

            l2_id = l2_seen.get(l2_key)

            # Level 2 — Sub-Business Area
            l3_name = self._clean_text(row.get(COL_L3_NAME))
            l3_desc = self._clean_text(row.get(COL_L3_DESC))
            l3_legacy = self._clean_text(row.get(COL_L3_LEGACY)) if has_legacy else ""
            l3_key = f"{l2_key}:{l3_name}"

            if l3_name and l3_key not in l3_seen:
                l3_id = f"L3_{self._generate_id(l1_name, l2_name, l3_name)}"
                l3_seen[l3_key] = l3_id

                # Use legacy definition as fallback when new definition is missing
                primary_desc = l3_desc or l3_legacy
                node_attrs = dict(
                    name=l3_name, description=primary_desc,
                    level=2, node_type="sub_business_area",
                    type="capability",
                    parent_category=l1_name,
                    parent_business_area=l2_name,
                )
                if l3_legacy:
                    node_attrs["legacy_description"] = l3_legacy

                self.graph.add_node(l3_id, **node_attrs)
                if l2_id:
                    self.graph.add_edge(l2_id, l3_id, edge_type="HAS_SUBCAPABILITY")

        if self.config.verbose:
            print(f"Categories (L0):         {len(l1_seen)}")
            print(f"Business Areas (L1):     {len(l2_seen)}")
            print(f"Sub-Business Areas (L2): {len(l3_seen)}")
            print(f"Total nodes:             {self.graph.number_of_nodes()}")
            print(f"Total edges:             {self.graph.number_of_edges()}")

        return self.graph

    # -----------------------------------------------------------------
    # FalkorDB Upload
    # -----------------------------------------------------------------

    def clear_falkordb(self):
        """Clear only Capability nodes from the shared FalkorDB graph.

        Only removes nodes with the :Capability label and the
        :CapabilityRoot label, leaving the org hierarchy intact.
        """
        if self._connected:
            try:
                self._fgraph.query("MATCH (n:CapabilityRoot) DETACH DELETE n")
                self._fgraph.query("MATCH (n:Capability) DETACH DELETE n")
                if self.config.verbose:
                    print("  Cleared existing capability data (org hierarchy preserved)")
            except Exception:
                pass

    def _create_master_root(self):
        """Create capability master root node."""
        if not self._connected:
            return
        try:
            self._fgraph.query(
                "MERGE (n:CapabilityRoot:Capability {node_id: $nid}) "
                "SET n.name = $name, n.level = -1, n.is_master_root = true, "
                "n.description = 'Master root for all business capabilities'",
                {"nid": self.config.master_root_id, "name": self.config.master_root_name}
            )
            if self.config.verbose:
                print(f"✓ Capability root: {self.config.master_root_id}")
        except Exception as e:
            print(f"✗ Root creation failed: {e}")

    def upload_to_falkordb(self, clear_existing: bool = True, create_index: bool = True) -> int:
        """Upload capability graph to FalkorDB."""
        if not self._connected:
            print("⚠ Not connected to FalkorDB")
            return 0

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("UPLOADING CAPABILITY GRAPH TO FALKORDB")
            print("=" * 60)

        if clear_existing:
            self.clear_falkordb()

        self._create_master_root()

        count = 0
        categories_uploaded = []

        # Upload nodes
        for node_id in self.graph.nodes():
            data = dict(self.graph.nodes[node_id])

            props = {"node_id": node_id}
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    props[key] = json.dumps(value, ensure_ascii=False)
                elif value is not None:
                    props[key] = str(value) if not isinstance(value, (int, float, bool)) else value

            # Determine label
            level = data.get("level", 0)
            if level == 0:
                labels = "Capability:Category"
                categories_uploaded.append(node_id)
            elif level == 1:
                labels = "Capability:BusinessArea"
            else:
                labels = "Capability:SubBusinessArea"

            props_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
            try:
                self._fgraph.query(f"CREATE (n:{labels} {{{props_str}}})", props)
                count += 1
            except Exception as e:
                print(f"  Error creating {node_id}: {e}")

        if self.config.verbose:
            print(f"  Uploaded {count} capability nodes")

        # Upload edges
        edge_count = 0
        for src, tgt in self.graph.edges():
            try:
                self._fgraph.query(
                    "MATCH (a:Capability {node_id: $src}), (b:Capability {node_id: $tgt}) "
                    "CREATE (a)-[:HAS_SUBCAPABILITY]->(b)",
                    {"src": src, "tgt": tgt}
                )
                edge_count += 1
            except Exception as e:
                print(f"  Error edge {src}->{tgt}: {e}")

        if self.config.verbose:
            print(f"  Created {edge_count} edges")

        # Connect categories to root
        for cat_id in categories_uploaded:
            try:
                self._fgraph.query(
                    "MATCH (root:CapabilityRoot {node_id: $rid}), (cat:Category {node_id: $cid}) "
                    "MERGE (root)-[:HAS_CATEGORY]->(cat)",
                    {"rid": self.config.master_root_id, "cid": cat_id}
                )
            except Exception as e:
                print(f"  Error connecting {cat_id} to root: {e}")

        if self.config.verbose:
            print(f"  Connected {len(categories_uploaded)} categories to root")

        if create_index:
            self.create_vector_index()

        # Stats
        stats = self.get_stats()
        if self.config.verbose:
            print(f"\n✓ Upload complete:")
            print(f"  Nodes:      {stats.get('nodes', 0)}")
            print(f"  Edges:      {stats.get('edges', 0)}")
            print(f"  Categories: {stats.get('categories', 0)}")

        return count

    def create_vector_index(self, index_name: str = None,
                            property_name: str = "description_embedding",
                            dimension: int = None) -> bool:
        """Create vector index on capability nodes.

        FalkorDB uses label+property as the implicit index identifier —
        the syntax does not accept a user-supplied index name.
        The ``index_name`` arg is retained for API compatibility only.
        """
        if not self._connected:
            return False

        dim = dimension or self.config.vector_dimension

        try:
            try:
                self._fgraph.query(
                    f"DROP VECTOR INDEX FOR (n:Capability) ON (n.{property_name})"
                )
            except Exception:
                pass

            self._fgraph.query(
                f"CREATE VECTOR INDEX FOR (n:Capability) ON (n.{property_name}) "
                f"OPTIONS {{dimension: {dim}, similarityFunction: '{self.config.vector_similarity}'}}"
            )
            if self.config.verbose:
                print(f"✓ Vector index: Capability.{property_name}")
            return True
        except Exception as e:
            print(f"⚠ Vector index: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get capability graph statistics from FalkorDB."""
        if not self._connected:
            return {"connected": False}
        try:
            nodes = self._fgraph.query("MATCH (n:Capability) RETURN count(n)").result_set[0][0]
            edges = self._fgraph.query(
                "MATCH (:Capability)-[r]->(:Capability) RETURN count(r)"
            ).result_set[0][0]
            cats = self._fgraph.query("MATCH (n:Category) RETURN count(n)").result_set[0][0]
            return {"connected": True, "nodes": nodes, "edges": edges, "categories": cats}
        except Exception:
            return {"connected": True, "nodes": 0, "edges": 0}

    # -----------------------------------------------------------------
    # Local Storage
    # -----------------------------------------------------------------

    def save_local(self, path: str = None) -> str:
        """Save capability graph to local JSON."""
        path = path or self.config.local_graph_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        graph_data = {
            "metadata": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "categories": self.categories,
                "master_root_id": self.config.master_root_id
            },
            "nodes": [{"id": n, **dict(d)} for n, d in self.graph.nodes(data=True)],
            "edges": [{"source": u, "target": v} for u, v in self.graph.edges()]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        if self.config.verbose:
            print(f"✓ Saved: {path}")
        return path

    def load_local(self, path: str = None) -> nx.DiGraph:
        """Load capability graph from local JSON."""
        path = path or self.config.local_graph_path

        with open(path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        self.graph = nx.DiGraph()
        self.categories = graph_data.get("metadata", {}).get("categories", [])

        for node in graph_data.get("nodes", []):
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)

        for edge in graph_data.get("edges", []):
            self.graph.add_edge(edge["source"], edge["target"])

        if self.config.verbose:
            print(f"✓ Loaded: {self.graph.number_of_nodes()} nodes")
        return self.graph

    # -----------------------------------------------------------------
    # Graph Operations
    # -----------------------------------------------------------------

    def get_children(self, node: str) -> List[str]:
        return list(self.graph.successors(node))

    def get_parent(self, node: str) -> Optional[str]:
        preds = list(self.graph.predecessors(node))
        return preds[0] if preds else None

    def get_all_capabilities(self) -> List[str]:
        """Get all capability node IDs."""
        return list(self.graph.nodes())

    def traverse_top_down(self) -> List[str]:
        """BFS from categories (root → leaves)."""
        visited, queue = [], list(self.categories)
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.append(node)
                queue.extend(self.get_children(node))
        return visited

    def traverse_bottom_up(self) -> List[str]:
        """Leaves first, grouped by level."""
        by_level = {}
        for n in self.graph.nodes():
            lvl = self.graph.nodes[n].get("level", 0)
            by_level.setdefault(lvl, []).append(n)
        result = []
        for lvl in sorted(by_level.keys(), reverse=True):
            result.extend(sorted(by_level[lvl]))
        return result

    def print_tree(self):
        """Print ASCII tree of capabilities."""
        def _print(node, prefix="", last=True):
            data = self.graph.nodes.get(node, {})
            name = data.get("name", node)[:45]
            level = data.get("level", 0)
            connector = "└── " if last else "├── "
            print(f"{prefix}{connector}[L{level}] {name}")
            children = sorted(self.get_children(node))
            for i, child in enumerate(children):
                _print(child, prefix + ("    " if last else "│   "), i == len(children) - 1)

        print(f"[ROOT] {self.config.master_root_name}")
        for i, cat in enumerate(sorted(self.categories)):
            _print(cat, "  ", i == len(self.categories) - 1)


# =============================================================================
# Factory
# =============================================================================

def create_capability_builder(
    data_path: str = None,
    sheet_name: str = DEFAULT_SHEET_NAME,
    falkordb_url: str = None,
    falkordb_graph_name: str = "org_hierarchy",
    local_graph_path: str = "./graph_data/capability_graph.json",
    vector_dimension: int = 384,
    verbose: bool = True
) -> CapabilityGraphBuilder:
    """
    Create CapabilityGraphBuilder.

    Args:
        data_path: Path to capability XLSX/CSV (optional, call build_from_file later).
                   Defaults to data/Capability/EP_Business_Map_5.0.xlsx when triggered.
        sheet_name: Worksheet name when data_path is an XLSX (default: "Business Map").
        falkordb_url: FalkorDB connection URL.
        falkordb_graph_name: FalkorDB graph name (defaults to org_hierarchy — shared graph).
        local_graph_path: Path for local JSON backup.
        vector_dimension: Embedding dimension (384 for MiniLM).
        verbose: Print progress.

    Returns:
        Configured CapabilityGraphBuilder
    """
    config = CapabilityGraphConfig(
        falkordb_url=falkordb_url,
        falkordb_graph_name=falkordb_graph_name,
        local_graph_path=local_graph_path,
        vector_dimension=vector_dimension,
        verbose=verbose
    )
    builder = CapabilityGraphBuilder(config=config)

    if falkordb_url:
        builder.connect()

    if data_path:
        builder.build_from_file(data_path, sheet_name=sheet_name)

    return builder
