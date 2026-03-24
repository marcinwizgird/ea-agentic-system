"""
Component 6: Capability Graph Builder
Builds business capability hierarchy from CSV.
- 3-level hierarchy: Category → Business Area → Sub-Business Area
- Vector index on descriptions for semantic matching
- Stores locally and uploads to FalkorDB
"""

import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import networkx as nx

# FalkorDB
try:
    from falkordb import FalkorDB as FalkorDBClient
    FALKORDB_AVAILABLE = True
except ImportError:
    FALKORDB_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CapabilityGraphConfig:
    """Configuration for capability graph building."""
    # Master root settings
    master_root_id: str = "CAPABILITY_ROOT"
    master_root_name: str = "Business Capability Map"
    
    # Local storage
    local_graph_path: str = "./graph_data/capability_graph.json"
    
    # FalkorDB connection
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: Optional[str] = None
    falkordb_graph_name: str = "capability_map"
    falkordb_url: Optional[str] = None
    
    # Vector index settings
    vector_index_name: str = "capability_vector_idx"
    vector_dimension: int = 384
    vector_similarity: str = "cosine"


# =============================================================================
# FalkorDB Store for Capabilities
# =============================================================================

class CapabilityFalkorDBStore:
    """FalkorDB store for capability graph."""
    
    def __init__(self, config: CapabilityGraphConfig):
        self.config = config
        self.client = None
        self.graph = None
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to FalkorDB."""
        if not FALKORDB_AVAILABLE:
            print("⚠ FalkorDB not available (pip install falkordb)")
            return False
        
        try:
            if self.config.falkordb_url:
                self.client = FalkorDBClient.from_url(self.config.falkordb_url)
            else:
                self.client = FalkorDBClient(
                    host=self.config.falkordb_host,
                    port=self.config.falkordb_port,
                    password=self.config.falkordb_password
                )
            
            self.graph = self.client.select_graph(self.config.falkordb_graph_name)
            self._connected = True
            print(f"✓ FalkorDB connected: {self.config.falkordb_graph_name}")
            return True
            
        except Exception as e:
            print(f"✗ FalkorDB connection failed: {e}")
            return False
    
    def clear(self):
        """Clear all nodes and edges."""
        if self._connected:
            try:
                self.graph.query("MATCH (n) DETACH DELETE n")
                print("  Cleared existing capability data")
            except:
                pass
    
    def create_vector_index(self) -> bool:
        """Create vector index on description_embedding."""
        if not self._connected:
            return False
        
        try:
            try:
                self.graph.query(f"DROP INDEX {self.config.vector_index_name}")
            except:
                pass
            
            query = f"""
            CREATE VECTOR INDEX {self.config.vector_index_name}
            FOR (n:Capability)
            ON (n.description_embedding)
            OPTIONS {{
                dimension: {self.config.vector_dimension},
                similarityFunction: '{self.config.vector_similarity}'
            }}
            """
            self.graph.query(query)
            print(f"✓ Vector index created: {self.config.vector_index_name}")
            return True
            
        except Exception as e:
            print(f"⚠ Vector index creation: {e}")
            return False
    
    def create_master_root(self) -> bool:
        """Create the capability master root node."""
        if not self._connected:
            return False
        
        try:
            query = """
            MERGE (n:CapabilityRoot:Capability {node_id: $node_id})
            SET n.name = $name,
                n.level = -1,
                n.is_master_root = true,
                n.description = 'Master root for all business capabilities'
            RETURN n
            """
            self.graph.query(query, {
                "node_id": self.config.master_root_id,
                "name": self.config.master_root_name
            })
            print(f"✓ Capability root created: {self.config.master_root_id}")
            return True
        except Exception as e:
            print(f"✗ Capability root creation failed: {e}")
            return False
    
    def upload_graph(self, nx_graph: nx.DiGraph, embeddings: Dict[str, List[float]] = None) -> int:
        """Upload capability graph to FalkorDB."""
        if not self._connected:
            print("⚠ Not connected to FalkorDB")
            return 0
        
        embeddings = embeddings or {}
        count = 0
        categories = []
        
        for node_id in nx_graph.nodes():
            data = dict(nx_graph.nodes[node_id])
            
            props = {"node_id": node_id}
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    props[key] = json.dumps(value, ensure_ascii=False)
                elif value is not None:
                    props[key] = str(value) if not isinstance(value, (int, float, bool)) else value
            
            if node_id in embeddings:
                props["description_embedding"] = embeddings[node_id]
            
            level = data.get("level", 0)
            if level == 0:
                labels = "Capability:Category"
                categories.append(node_id)
            elif level == 1:
                labels = "Capability:BusinessArea"
            else:
                labels = "Capability:SubBusinessArea"
            
            props_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
            query = f"CREATE (n:{labels} {{{props_str}}}) RETURN n"
            
            try:
                self.graph.query(query, props)
                count += 1
            except Exception as e:
                print(f"  Error creating capability {node_id}: {e}")
        
        for from_id, to_id in nx_graph.edges():
            query = """
            MATCH (a:Capability {node_id: $from_id}), (b:Capability {node_id: $to_id})
            CREATE (a)-[:HAS_SUBCAPABILITY]->(b)
            """
            try:
                self.graph.query(query, {"from_id": from_id, "to_id": to_id})
            except Exception as e:
                print(f"  Error creating edge {from_id}->{to_id}: {e}")
        
        for cat_id in categories:
            query = """
            MATCH (root:CapabilityRoot {node_id: $root_id}), (cat:Category {node_id: $cat_id})
            MERGE (root)-[:HAS_CATEGORY]->(cat)
            """
            try:
                self.graph.query(query, {
                    "root_id": self.config.master_root_id,
                    "cat_id": cat_id
                })
            except Exception as e:
                print(f"  Error connecting {cat_id} to root: {e}")
        
        print(f"  Connected {len(categories)} categories to capability root")
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self._connected:
            return {"connected": False}
        
        try:
            nodes = self.graph.query("MATCH (n:Capability) RETURN count(n)").result_set[0][0]
            edges = self.graph.query("MATCH (:Capability)-[r]->(:Capability) RETURN count(r)").result_set[0][0]
            categories = self.graph.query("MATCH (n:Category) RETURN count(n)").result_set[0][0]
            
            return {
                "connected": True,
                "nodes": nodes,
                "edges": edges,
                "categories": categories
            }
        except:
            return {"connected": True, "nodes": 0, "edges": 0}


# =============================================================================
# Mock FalkorDB Store
# =============================================================================

class MockCapabilityFalkorDBStore:
    """Mock FalkorDB for testing."""
    
    def __init__(self, config: CapabilityGraphConfig = None):
        self.config = config or CapabilityGraphConfig()
        self.nodes = {}
        self.edges = []
        self.vector_index_created = False
    
    def connect(self) -> bool:
        print("✓ FalkorDB (Capability): Mock mode")
        return True
    
    def clear(self):
        self.nodes = {}
        self.edges = []
    
    def create_vector_index(self) -> bool:
        self.vector_index_created = True
        print(f"✓ Vector index created (mock): description_embedding")
        return True
    
    def create_master_root(self) -> bool:
        self.nodes[self.config.master_root_id] = {
            "name": self.config.master_root_name,
            "level": -1,
            "is_master_root": True
        }
        print(f"✓ Capability root created (mock): {self.config.master_root_id}")
        return True
    
    def upload_graph(self, nx_graph: nx.DiGraph, embeddings: Dict[str, List[float]] = None) -> int:
        categories = []
        for node_id in nx_graph.nodes():
            data = dict(nx_graph.nodes[node_id])
            self.nodes[node_id] = data
            if data.get("level", 0) == 0:
                categories.append(node_id)
        
        for from_id, to_id in nx_graph.edges():
            self.edges.append((from_id, to_id))
        
        for cat_id in categories:
            self.edges.append((self.config.master_root_id, cat_id))
        
        print(f"  Connected {len(categories)} categories to capability root")
        return len(nx_graph.nodes())
    
    def get_stats(self) -> Dict[str, Any]:
        categories = sum(1 for d in self.nodes.values() if d.get("level", 0) == 0)
        return {
            "connected": True,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "categories": categories,
            "has_vector_index": self.vector_index_created,
            "mock": True
        }


# =============================================================================
# Capability Graph Builder
# =============================================================================

class CapabilityGraphBuilder:
    """
    Builds business capability hierarchy from CSV.
    3-level structure: Category → Business Area → Sub-Business Area
    """
    
    def __init__(self, config: CapabilityGraphConfig = None, use_mock_falkordb: bool = True):
        self.config = config or CapabilityGraphConfig()
        self.graph = nx.DiGraph()
        self.categories = []
        self.embeddings: Dict[str, List[float]] = {}
        
        if use_mock_falkordb or not FALKORDB_AVAILABLE:
            self.falkordb = MockCapabilityFalkorDBStore(self.config)
        else:
            self.falkordb = CapabilityFalkorDBStore(self.config)
        
        self.falkordb.connect()
    
    def _generate_id(self, *parts) -> str:
        """Generate unique ID from parts."""
        content = ":".join(str(p) for p in parts)
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _clean_text(self, text: str) -> str:
        """Clean text for processing."""
        if pd.isna(text):
            return ""
        return str(text).strip()
    
    def _detect_csv_format(self, df: pd.DataFrame) -> dict:
        """Detect CSV column format and return column mapping."""
        columns = df.columns.tolist()
        
        # Format 1: Original format (Category, Business Area, Sub-Business Area)
        if "Category" in columns and "Business Area" in columns:
            return {
                "format": "original",
                "l1_name": "Category",
                "l1_desc": "Category Description",
                "l2_name": "Business Area",
                "l2_desc": "Business Area Description",
                "l3_name": "Sub-Business Area",
                "l3_desc": "Sub-Business Area Description"
            }
        
        # Format 2: DG ITEC format (Capability L1, L2, L3)
        if "Capability L1" in columns:
            # Determine L2 and L3 description column names
            # Could be "Description/Definition" / "Description/Definition.1" OR
            # "L2 Description/Definition" / "L3 Description/Definition"
            if "L2 Description/Definition" in columns:
                l2_desc_col = "L2 Description/Definition"
                l3_desc_col = "L3 Description/Definition"
            elif "Description/Definition.1" in columns:
                l2_desc_col = "Description/Definition"
                l3_desc_col = "Description/Definition.1"
            else:
                l2_desc_col = "Description/Definition"
                l3_desc_col = "Description/Definition"
            
            return {
                "format": "itec",
                "l1_name": "Capability L1",
                "l1_desc": "L1 Description/Definition",
                "l2_name": "Capability L2",
                "l2_desc": l2_desc_col,
                "l3_name": "Capability L3",
                "l3_desc": l3_desc_col
            }
        
        # Format 3: Generic L1/L2/L3
        if "L1" in columns or "Level 1" in columns:
            l1_col = "L1" if "L1" in columns else "Level 1"
            l2_col = "L2" if "L2" in columns else "Level 2"
            l3_col = "L3" if "L3" in columns else "Level 3"
            return {
                "format": "generic",
                "l1_name": l1_col,
                "l1_desc": f"{l1_col} Description",
                "l2_name": l2_col,
                "l2_desc": f"{l2_col} Description",
                "l3_name": l3_col,
                "l3_desc": f"{l3_col} Description"
            }
        
        # Default: try to infer from first columns
        print(f"⚠ Unknown CSV format. Columns: {columns[:6]}")
        return None
    
    def build_from_csv(self, csv_path: str) -> nx.DiGraph:
        """Build capability graph from CSV file. Auto-detects format."""
        print("\n" + "=" * 50)
        print("BUILDING CAPABILITY GRAPH")
        print("=" * 50)
        
        # Try utf-8-sig first to handle BOM, fallback to utf-8
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
        except:
            df = pd.read_csv(csv_path, encoding='utf-8')
        
        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        
        print(f"Rows in CSV: {len(df)}")
        print(f"Columns: {df.columns.tolist()[:6]}...")
        
        # Detect format
        col_map = self._detect_csv_format(df)
        if not col_map:
            print("⚠ Could not detect CSV format")
            return self.graph
        
        print(f"Detected format: {col_map['format']}")
        
        self.graph = nx.DiGraph()
        self.categories = []
        
        l1_seen = {}
        l2_seen = {}
        l3_seen = {}
        
        for _, row in df.iterrows():
            # Level 1 (Category equivalent)
            l1_name = self._clean_text(row.get(col_map["l1_name"], ""))
            l1_desc = self._clean_text(row.get(col_map["l1_desc"], ""))
            
            if l1_name and l1_name not in l1_seen:
                l1_id = f"L1_{l1_name.upper().replace(' ', '_')[:20]}_{self._generate_id(l1_name)[:6]}"
                l1_seen[l1_name] = l1_id
                
                self.graph.add_node(l1_id,
                    name=l1_name,
                    description=l1_desc,
                    level=0,
                    node_type="category"
                )
                self.categories.append(l1_id)
            
            l1_id = l1_seen.get(l1_name)
            
            # Level 2 (Business Area equivalent)
            l2_name = self._clean_text(row.get(col_map["l2_name"], ""))
            l2_desc = self._clean_text(row.get(col_map["l2_desc"], ""))
            l2_key = f"{l1_name}:{l2_name}"
            
            if l2_name and l2_key not in l2_seen:
                l2_id = f"L2_{self._generate_id(l1_name, l2_name)}"
                l2_seen[l2_key] = l2_id
                
                self.graph.add_node(l2_id,
                    name=l2_name,
                    description=l2_desc,
                    level=1,
                    node_type="business_area",
                    parent_category=l1_name
                )
                
                if l1_id:
                    self.graph.add_edge(l1_id, l2_id)
            
            l2_id = l2_seen.get(l2_key)
            
            # Level 3 (Sub-Business Area equivalent)
            l3_name = self._clean_text(row.get(col_map["l3_name"], ""))
            l3_desc = self._clean_text(row.get(col_map["l3_desc"], ""))
            l3_key = f"{l2_key}:{l3_name}"
            
            if l3_name and l3_key not in l3_seen:
                l3_id = f"L3_{self._generate_id(l1_name, l2_name, l3_name)}"
                l3_seen[l3_key] = l3_id
                
                self.graph.add_node(l3_id,
                    name=l3_name,
                    description=l3_desc,
                    level=2,
                    node_type="sub_business_area",
                    parent_category=l1_name,
                    parent_business_area=l2_name
                )
                
                if l2_id:
                    self.graph.add_edge(l2_id, l3_id)
        
        print(f"Categories (L1): {len(l1_seen)}")
        print(f"Business Areas (L2): {len(l2_seen)}")
        print(f"Sub-Business Areas (L3): {len(l3_seen)}")
        print(f"Total nodes: {self.graph.number_of_nodes()}")
        print(f"Total edges: {self.graph.number_of_edges()}")
        
        return self.graph
    
    def save_local(self, path: str = None) -> str:
        """Save graph to local JSON file."""
        path = path or self.config.local_graph_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        graph_data = {
            "metadata": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "categories": self.categories,
                "master_root_id": self.config.master_root_id
            },
            "nodes": [
                {"id": n, **{k: v for k, v in d.items()}}
                for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v}
                for u, v in self.graph.edges()
            ]
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Capability graph saved: {path}")
        return path
    
    def load_local(self, path: str = None) -> nx.DiGraph:
        """Load graph from local JSON file."""
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
        
        print(f"✓ Capability graph loaded: {self.graph.number_of_nodes()} nodes")
        return self.graph
    
    def upload_to_falkordb(self, clear_existing: bool = True, create_index: bool = True) -> int:
        """Upload graph to FalkorDB."""
        print("\n" + "=" * 50)
        print("UPLOADING CAPABILITY GRAPH TO FALKORDB")
        print("=" * 50)
        
        if clear_existing:
            self.falkordb.clear()
        
        self.falkordb.create_master_root()
        count = self.falkordb.upload_graph(self.graph, self.embeddings)
        
        if create_index:
            self.falkordb.create_vector_index()
        
        stats = self.falkordb.get_stats()
        print(f"\n✓ Upload complete:")
        print(f"  Nodes: {stats.get('nodes', 0)}")
        print(f"  Edges: {stats.get('edges', 0)}")
        print(f"  Categories: {stats.get('categories', 0)}")
        
        return count
    
    # =========================================================================
    # Graph Operations
    # =========================================================================
    
    def get_children(self, node: str) -> List[str]:
        return list(self.graph.successors(node))
    
    def get_parent(self, node: str) -> Optional[str]:
        preds = list(self.graph.predecessors(node))
        return preds[0] if preds else None
    
    def get_all_descendants(self, node: str) -> List[str]:
        """Get all descendants (recursive)."""
        descendants = []
        queue = self.get_children(node)
        while queue:
            child = queue.pop(0)
            if child not in descendants:
                descendants.append(child)
                queue.extend(self.get_children(child))
        return descendants
    
    def get_all_ancestors(self, node: str) -> List[str]:
        """Get all ancestors (recursive)."""
        ancestors = []
        current = self.get_parent(node)
        while current:
            ancestors.append(current)
            current = self.get_parent(current)
        return ancestors
    
    def traverse_top_down(self) -> List[str]:
        """BFS from categories."""
        visited, queue = [], list(self.categories)
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.append(node)
                queue.extend(self.get_children(node))
        return visited
    
    def traverse_bottom_up(self) -> List[str]:
        """Leaves first."""
        by_level = {}
        for n in self.graph.nodes():
            lvl = self.graph.nodes[n].get("level", 0)
            by_level.setdefault(lvl, []).append(n)
        result = []
        for lvl in sorted(by_level.keys(), reverse=True):
            result.extend(sorted(by_level[lvl]))
        return result
    
    def print_tree(self, include_master_root: bool = True):
        """Print ASCII tree."""
        def _print(node, prefix="", last=True):
            data = self.graph.nodes.get(node, {})
            name = data.get("name", node)[:40]
            level = data.get("level", 0)
            print(f"{prefix}{'└── ' if last else '├── '}[L{level}] {name}")
            children = sorted(self.get_children(node))
            for i, child in enumerate(children):
                _print(child, prefix + ("    " if last else "│   "), i == len(children) - 1)
        
        if include_master_root:
            print(f"[{self.config.master_root_id}] {self.config.master_root_name}")
            for i, cat in enumerate(sorted(self.categories)):
                data = self.graph.nodes.get(cat, {})
                name = data.get("name", cat)
                prefix = "    "
                connector = "└── " if i == len(self.categories) - 1 else "├── "
                print(f"{prefix}{connector}[L0] {name}")
                children = sorted(self.get_children(cat))
                for j, child in enumerate(children):
                    child_prefix = prefix + ("    " if i == len(self.categories) - 1 else "│   ")
                    _print(child, child_prefix, j == len(children) - 1)
        else:
            for i, cat in enumerate(sorted(self.categories)):
                _print(cat, "", i == len(self.categories) - 1)
    
    def export(self) -> Dict:
        """Export graph to dictionary."""
        return {
            "master_root": {
                "id": self.config.master_root_id,
                "name": self.config.master_root_name
            },
            "nodes": [{"id": n, **dict(d)} for n, d in self.graph.nodes(data=True)],
            "edges": [{"source": u, "target": v} for u, v in self.graph.edges()]
        }


# =============================================================================
# Factory
# =============================================================================

def create_capability_graph_builder(
    local_graph_path: str = "./graph_data/capability_graph.json",
    master_root_id: str = "CAPABILITY_ROOT",
    master_root_name: str = "Business Capability Map",
    falkordb_url: str = None,
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    falkordb_graph_name: str = "capability_map",
    vector_dimension: int = 384,
    use_mock_falkordb: bool = True
) -> CapabilityGraphBuilder:
    """Create CapabilityGraphBuilder instance."""
    config = CapabilityGraphConfig(
        local_graph_path=local_graph_path,
        master_root_id=master_root_id,
        master_root_name=master_root_name,
        falkordb_url=falkordb_url,
        falkordb_host=falkordb_host,
        falkordb_port=falkordb_port,
        falkordb_graph_name=falkordb_graph_name,
        vector_dimension=vector_dimension
    )
    return CapabilityGraphBuilder(config=config, use_mock_falkordb=use_mock_falkordb)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CAPABILITY GRAPH BUILDER TEST")
    print("=" * 60)
    
    builder = create_capability_graph_builder(
        master_root_name="EP Business Capabilities",
        use_mock_falkordb=True
    )
    
    csv_path = "/mnt/user-data/uploads/business_map_table__1_.csv"
    builder.build_from_csv(csv_path)
    
    print("\nCapability Hierarchy:")
    builder.print_tree(include_master_root=True)
    
    builder.save_local()
    builder.upload_to_falkordb()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
