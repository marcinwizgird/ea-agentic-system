"""
Component 4: Graph Builder (Updated)
Builds organizational hierarchy from ChromaDB collection.
- Master root node for all organizations
- Vector index on activities property
- Stores graph locally (NetworkX JSON) and uploads to FalkorDB.
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

import networkx as nx

from vector_db import VectorDB, create_vector_db

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
class GraphConfig:
    """Configuration for graph building."""
    # Master root settings
    master_root_id: str = "MASTER_ROOT"
    master_root_name: str = "Organization Master Root"

    # Local storage
    local_graph_path: str = "./graph_data/organization_graph.json"

    # FalkorDB connection
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: Optional[str] = None
    falkordb_graph_name: str = "org_hierarchy"
    falkordb_url: Optional[str] = None  # Cloud URL (overrides host/port)

    # Vector index settings
    vector_index_name: str = "activities_vector_idx"
    vector_dimension: int = 384  # Default for all-MiniLM-L6-v2
    vector_similarity: str = "cosine"  # cosine, euclidean, or ip (inner product)


# =============================================================================
# FalkorDB Store
# =============================================================================

class FalkorDBStore:
    """FalkorDB graph database store with vector index support."""

    def __init__(self, config: GraphConfig):
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
                print("  Cleared existing data")
            except:
                pass

    def create_vector_index(self) -> bool:
        """
        Create vector index on activities_embedding property.
        This enables vector similarity search on organizational activities.
        """
        if not self._connected:
            return False

        try:
            # Drop existing index if exists
            try:
                self.graph.query(f"DROP INDEX {self.config.vector_index_name}")
            except:
                pass  # Index doesn't exist

            # Create vector index on OrganizationalUnit nodes
            # FalkorDB uses CREATE VECTOR INDEX syntax
            query = f"""
            CREATE VECTOR INDEX {self.config.vector_index_name}
            FOR (n:OrganizationalUnit)
            ON (n.activities_embedding)
            OPTIONS {{
                dimension: {self.config.vector_dimension},
                similarityFunction: '{self.config.vector_similarity}'
            }}
            """

            self.graph.query(query)
            print(f"✓ Vector index created: {self.config.vector_index_name}")
            print(f"  Dimension: {self.config.vector_dimension}, Similarity: {self.config.vector_similarity}")
            return True

        except Exception as e:
            print(f"⚠ Vector index creation: {e}")
            # Try alternative syntax for older FalkorDB versions
            try:
                alt_query = f"""
                CALL db.idx.vector.createNodeIndex(
                    'OrganizationalUnit',
                    'activities_embedding',
                    {self.config.vector_dimension},
                    '{self.config.vector_similarity}'
                )
                """
                self.graph.query(alt_query)
                print(f"✓ Vector index created (alt method): activities_embedding")
                return True
            except Exception as e2:
                print(f"⚠ Vector index not created: {e2}")
                return False

    def create_master_root(self) -> bool:
        """Create the master root node."""
        if not self._connected:
            return False

        try:
            query = """
            MERGE (n:MasterRoot:OrganizationalUnit {node_id: $node_id})
            SET n.name = $name,
                n.level = -1,
                n.is_master_root = true,
                n.description = 'Master root node for all organizational hierarchies'
            RETURN n
            """
            self.graph.query(query, {
                "node_id": self.config.master_root_id,
                "name": self.config.master_root_name
            })
            print(f"✓ Master root created: {self.config.master_root_id}")
            return True
        except Exception as e:
            print(f"✗ Master root creation failed: {e}")
            return False

    def upload_graph(self, nx_graph: nx.DiGraph, embeddings: Dict[str, List[float]] = None) -> int:
        """
        Upload NetworkX graph to FalkorDB.

        Args:
            nx_graph: NetworkX graph to upload
            embeddings: Optional dict of node_id -> embedding vector for activities

        Returns:
            Number of nodes uploaded
        """
        if not self._connected:
            print("⚠ Not connected to FalkorDB")
            return 0

        embeddings = embeddings or {}
        count = 0
        org_roots = []  # Track organization root nodes

        # Create nodes
        for node_id in nx_graph.nodes():
            data = dict(nx_graph.nodes[node_id])

            # Serialize lists/dicts to JSON strings
            props = {"node_id": node_id}
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    props[key] = json.dumps(value, ensure_ascii=False)
                elif value is not None:
                    props[key] = value

            # Add embedding if available
            if node_id in embeddings:
                props["activities_embedding"] = embeddings[node_id]

            # Determine labels
            level = data.get("level", 0)
            if level == 0:
                labels = "OrganizationalUnit:OrgRoot"
                org_roots.append(node_id)
            else:
                labels = "OrganizationalUnit"

            # Build Cypher query
            props_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
            query = f"CREATE (n:{labels} {{{props_str}}}) RETURN n"

            try:
                self.graph.query(query, props)
                count += 1
            except Exception as e:
                print(f"  Error creating node {node_id}: {e}")

        # Create edges between nodes
        for from_id, to_id in nx_graph.edges():
            query = """
            MATCH (a {node_id: $from_id}), (b {node_id: $to_id})
            CREATE (a)-[:HAS_CHILD]->(b)
            """
            try:
                self.graph.query(query, {"from_id": from_id, "to_id": to_id})
            except Exception as e:
                print(f"  Error creating edge {from_id}->{to_id}: {e}")

        # Connect organization roots to master root
        for org_root in org_roots:
            query = """
            MATCH (master:MasterRoot {node_id: $master_id}), (org:OrgRoot {node_id: $org_id})
            MERGE (master)-[:HAS_ORGANIZATION]->(org)
            """
            try:
                self.graph.query(query, {
                    "master_id": self.config.master_root_id,
                    "org_id": org_root
                })
            except Exception as e:
                print(f"  Error connecting {org_root} to master: {e}")

        print(f"  Connected {len(org_roots)} organization(s) to master root")
        return count

    def vector_search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """
        Search for similar nodes using vector similarity.

        Args:
            query_embedding: Query vector
            k: Number of results

        Returns:
            List of matching nodes with scores
        """
        if not self._connected:
            return []

        try:
            query = f"""
            CALL db.idx.vector.queryNodes(
                'OrganizationalUnit',
                'activities_embedding',
                $k,
                vecf32($embedding)
            ) YIELD node, score
            RETURN node.node_id AS id, node.name AS name, score
            ORDER BY score DESC
            """
            result = self.graph.query(query, {
                "k": k,
                "embedding": query_embedding
            })

            return [
                {"id": row[0], "name": row[1], "score": row[2]}
                for row in result.result_set
            ]
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if not self._connected:
            return {"connected": False}

        try:
            nodes = self.graph.query("MATCH (n) RETURN count(n)").result_set[0][0]
            edges = self.graph.query("MATCH ()-[r]->() RETURN count(r)").result_set[0][0]
            orgs = self.graph.query("MATCH (n:OrgRoot) RETURN count(n)").result_set[0][0]

            # Check for vector index
            has_vector_idx = False
            try:
                self.graph.query("CALL db.indexes()")
                has_vector_idx = True
            except:
                pass

            return {
                "connected": True,
                "nodes": nodes,
                "edges": edges,
                "organizations": orgs,
                "has_vector_index": has_vector_idx
            }
        except:
            return {"connected": True, "nodes": 0, "edges": 0}


# =============================================================================
# Mock FalkorDB Store
# =============================================================================

class MockFalkorDBStore:
    """Mock FalkorDB for testing without connection."""

    def __init__(self, config: GraphConfig = None):
        self.config = config or GraphConfig()
        self.nodes = {}
        self.edges = []
        self.master_root_created = False
        self.vector_index_created = False

    def connect(self) -> bool:
        print("✓ FalkorDB: Mock mode (in-memory)")
        return True

    def clear(self):
        self.nodes = {}
        self.edges = []
        self.master_root_created = False

    def create_vector_index(self) -> bool:
        self.vector_index_created = True
        print(f"✓ Vector index created (mock): activities_embedding")
        return True

    def create_master_root(self) -> bool:
        self.nodes[self.config.master_root_id] = {
            "name": self.config.master_root_name,
            "level": -1,
            "is_master_root": True
        }
        self.master_root_created = True
        print(f"✓ Master root created (mock): {self.config.master_root_id}")
        return True

    def upload_graph(self, nx_graph: nx.DiGraph, embeddings: Dict[str, List[float]] = None) -> int:
        org_roots = []
        for node_id in nx_graph.nodes():
            data = dict(nx_graph.nodes[node_id])
            self.nodes[node_id] = data
            if data.get("level", 0) == 0:
                org_roots.append(node_id)

        for from_id, to_id in nx_graph.edges():
            self.edges.append((from_id, to_id))

        # Connect to master root
        for org_root in org_roots:
            self.edges.append((self.config.master_root_id, org_root))

        print(f"  Connected {len(org_roots)} organization(s) to master root")
        return len(nx_graph.nodes())

    def vector_search(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        # Simple mock - return first k nodes
        results = []
        for node_id, data in list(self.nodes.items())[:k]:
            results.append({"id": node_id, "name": data.get("name", ""), "score": 0.9})
        return results

    def get_stats(self) -> Dict[str, Any]:
        orgs = sum(1 for d in self.nodes.values() if d.get("level", 0) == 0)
        return {
            "connected": True,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "organizations": orgs,
            "has_vector_index": self.vector_index_created,
            "mock": True
        }


# =============================================================================
# Graph Builder
# =============================================================================

class GraphBuilder:
    """
    Builds organizational hierarchy graph from ChromaDB collection.
    - Creates master root for all organizations
    - Sets up vector index on activities
    - Stores locally and uploads to FalkorDB
    """

    def __init__(self, config: GraphConfig = None, use_mock_falkordb: bool = True):
        self.config = config or GraphConfig()
        self.graph = nx.DiGraph()
        self.roots = []  # Organization roots (level 0)
        self.embeddings: Dict[str, List[float]] = {}  # Node embeddings

        # Initialize FalkorDB store
        if use_mock_falkordb or not FALKORDB_AVAILABLE:
            self.falkordb = MockFalkorDBStore(self.config)
        else:
            self.falkordb = FalkorDBStore(self.config)

        self.falkordb.connect()

    def _infer_hierarchy(self, code: str) -> Tuple[int, Optional[str]]:
        """Infer level and parent from entity code."""
        code = str(code).strip()
        if re.match(r'^\d+$', code): return 0, None
        if re.match(r'^\d+-\d+$', code): return 1, code.split('-')[0]
        if re.match(r'^\d+[A-Z]$', code): return 1, re.match(r'^(\d+)', code).group(1)
        if re.match(r'^\d+[A-Z]\d+$', code):
            m = re.match(r'^(\d+[A-Z])', code)
            return 2, m.group(1) if m else None
        return 0, None

    def build_from_chromadb(self, vector_db: VectorDB, compute_embeddings: bool = True) -> nx.DiGraph:
        """
        Build hierarchy graph from ChromaDB collection.

        Args:
            vector_db: VectorDB instance with ingested documents
            compute_embeddings: Whether to compute embeddings for activities

        Returns:
            NetworkX DiGraph
        """
        print("\n" + "=" * 50)
        print("BUILDING GRAPH FROM CHROMADB")
        print("=" * 50)

        # Get all documents from ChromaDB
        if vector_db.store is None:
            all_docs = vector_db._mock_docs
            print(f"Documents in collection: {len(all_docs)} (mock)")
        else:
            try:
                collection = vector_db.store._collection
                result = collection.get(include=["documents", "metadatas", "embeddings"])
                all_docs = []
                doc_embeddings = result.get("embeddings", [])

                for i, doc_text in enumerate(result.get("documents", [])):
                    metadata = result.get("metadatas", [{}])[i] if result.get("metadatas") else {}
                    from vector_db import Document
                    doc = Document(page_content=doc_text, metadata=metadata)
                    # Store embedding if available
                    if doc_embeddings and i < len(doc_embeddings):
                        doc.metadata["_embedding"] = doc_embeddings[i]
                    all_docs.append(doc)
                print(f"Documents in collection: {len(all_docs)}")
            except Exception as e:
                print(f"Error reading ChromaDB: {e}")
                all_docs = []

        # Extract organizational units
        entities: Dict[str, Dict] = {}

        for doc in all_docs:
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}

            if metadata.get("doc_type") not in ["organizational_unit", None]:
                continue

            code = metadata.get("entity_code", "")
            if not code:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                code_match = re.search(r'Code:\s*(\S+)', content)
                if code_match:
                    code = code_match.group(1)

            if not code:
                continue

            if code not in entities:
                level, parent = self._infer_hierarchy(code)
                entities[code] = {
                    "name": metadata.get("entity_name", ""),
                    "level": level,
                    "parent": parent,
                    "activities": [],
                    "weights": [],
                    "files": set(),
                    "refined_description": "",
                    "embeddings": []  # Store embeddings for this entity
                }

            # Extract name
            if not entities[code]["name"]:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                name_match = re.search(r'Organizational Unit:\s*(.+?)(?:\n|$)', content)
                if name_match:
                    entities[code]["name"] = name_match.group(1).strip()

            # Extract activities
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            activity_matches = re.findall(r'-\s*\((\d+)%\)\s*(.+?)(?:\n|$)', content)
            for weight, activity in activity_matches:
                if activity not in entities[code]["activities"]:
                    entities[code]["activities"].append(activity.strip())
                    entities[code]["weights"].append(int(weight))

            # Store embedding
            if "_embedding" in metadata:
                entities[code]["embeddings"].append(metadata["_embedding"])

            if metadata.get("file"):
                entities[code]["files"].add(metadata["file"])

        print(f"Unique entities found: {len(entities)}")

        # Build NetworkX graph
        self.graph = nx.DiGraph()
        self.roots = []
        self.embeddings = {}

        for code, data in entities.items():
            # Compute average embedding for activities
            if data["embeddings"]:
                avg_embedding = [
                    sum(e[i] for e in data["embeddings"]) / len(data["embeddings"])
                    for i in range(len(data["embeddings"][0]))
                ]
                self.embeddings[code] = avg_embedding

            node_data = {
                "name": data["name"],
                "level": data["level"],
                "parent_code": data["parent"],
                "activities": data["activities"],
                "activity_weights": data["weights"],
                "source_files": list(data["files"]),
                "refined_description": data["refined_description"]
            }

            self.graph.add_node(code, **node_data)

            if data["level"] == 0:
                self.roots.append(code)

        # Add edges
        for code, data in entities.items():
            parent = data["parent"]
            if parent and parent in self.graph:
                self.graph.add_edge(parent, code)

        print(f"Graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        print(f"Organization roots: {len(self.roots)}")
        print(f"Embeddings computed: {len(self.embeddings)}")

        return self.graph

    def save_local(self, path: str = None) -> str:
        """Save graph to local JSON file."""
        path = path or self.config.local_graph_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        graph_data = {
            "metadata": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "roots": self.roots,
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

        print(f"✓ Graph saved locally: {path}")
        return path

    def load_local(self, path: str = None) -> nx.DiGraph:
        """Load graph from local JSON file."""
        path = path or self.config.local_graph_path

        with open(path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        self.graph = nx.DiGraph()
        self.roots = graph_data.get("metadata", {}).get("roots", [])

        for node in graph_data.get("nodes", []):
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)

        for edge in graph_data.get("edges", []):
            self.graph.add_edge(edge["source"], edge["target"])

        print(f"✓ Graph loaded: {self.graph.number_of_nodes()} nodes")
        return self.graph

    def upload_to_falkordb(self, clear_existing: bool = True, create_index: bool = True) -> int:
        """
        Upload graph to FalkorDB with master root and vector index.

        Args:
            clear_existing: Clear existing data before upload
            create_index: Create vector index on activities

        Returns:
            Number of nodes uploaded
        """
        print("\n" + "=" * 50)
        print("UPLOADING TO FALKORDB")
        print("=" * 50)

        if clear_existing:
            self.falkordb.clear()

        # Create master root
        self.falkordb.create_master_root()

        # Upload graph with embeddings
        count = self.falkordb.upload_graph(self.graph, self.embeddings)

        # Create vector index
        if create_index:
            self.falkordb.create_vector_index()

        stats = self.falkordb.get_stats()
        print(f"\n✓ Upload complete:")
        print(f"  Nodes: {stats.get('nodes', 0)}")
        print(f"  Edges: {stats.get('edges', 0)}")
        print(f"  Organizations: {stats.get('organizations', 0)}")
        print(f"  Vector index: {stats.get('has_vector_index', False)}")

        return count

    def search_similar_activities(self, query_embedding: List[float], k: int = 5) -> List[Dict]:
        """Search for nodes with similar activities using vector index."""
        return self.falkordb.vector_search(query_embedding, k)

    # =========================================================================
    # Graph Operations
    # =========================================================================

    def get_children(self, node: str) -> List[str]:
        return list(self.graph.successors(node))

    def get_parent(self, node: str) -> Optional[str]:
        preds = list(self.graph.predecessors(node))
        return preds[0] if preds else None

    def traverse_top_down(self) -> List[str]:
        """BFS from roots."""
        visited, queue = [], list(self.roots)
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

    def update_node(self, node: str, **kwargs):
        if node in self.graph:
            self.graph.nodes[node].update(kwargs)

    def print_tree(self, include_master_root: bool = True):
        """Print ASCII tree."""
        def _print(node, prefix="", last=True):
            data = self.graph.nodes.get(node, {})
            name = data.get("name", node)[:35]
            print(f"{prefix}{'└── ' if last else '├── '}[{node}] {name}")
            children = sorted(self.get_children(node))
            for i, child in enumerate(children):
                _print(child, prefix + ("    " if last else "│   "), i == len(children) - 1)

        if include_master_root:
            print(f"[{self.config.master_root_id}] {self.config.master_root_name}")
            for i, root in enumerate(sorted(self.roots)):
                prefix = "    "
                connector = "└── " if i == len(self.roots) - 1 else "├── "
                data = self.graph.nodes.get(root, {})
                name = data.get("name", root)[:35]
                print(f"{prefix}{connector}[{root}] {name}")
                children = sorted(self.get_children(root))
                for j, child in enumerate(children):
                    child_prefix = prefix + ("    " if i == len(self.roots) - 1 else "│   ")
                    _print(child, child_prefix, j == len(children) - 1)
        else:
            for i, root in enumerate(sorted(self.roots)):
                _print(root, "", i == len(self.roots) - 1)

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

def create_graph_builder(
    local_graph_path: str = "./graph_data/organization_graph.json",
    master_root_id: str = "MASTER_ROOT",
    master_root_name: str = "Organization Master Root",
    falkordb_url: str = None,
    falkordb_host: str = "localhost",
    falkordb_port: int = 6379,
    vector_dimension: int = 384,
    use_mock_falkordb: bool = True
) -> GraphBuilder:
    """
    Create GraphBuilder instance.

    Args:
        local_graph_path: Path for local graph storage
        master_root_id: ID for master root node
        master_root_name: Name for master root node
        falkordb_url: FalkorDB cloud URL (optional)
        falkordb_host: FalkorDB host
        falkordb_port: FalkorDB port
        vector_dimension: Dimension for vector embeddings
        use_mock_falkordb: Use mock instead of real FalkorDB
    """
    config = GraphConfig(
        local_graph_path=local_graph_path,
        master_root_id=master_root_id,
        master_root_name=master_root_name,
        falkordb_url=falkordb_url,
        falkordb_host=falkordb_host,
        falkordb_port=falkordb_port,
        vector_dimension=vector_dimension
    )
    return GraphBuilder(config=config, use_mock_falkordb=use_mock_falkordb)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GRAPH BUILDER TEST (with Master Root & Vector Index)")
    print("=" * 60)

    from vector_db import create_vector_db
    from ingestion import create_ingestion_plugin

    # Ingest
    db = create_vector_db()
    plugin = create_ingestion_plugin(vector_db=db)
    plugin.ingest(".")

    # Build graph
    builder = create_graph_builder(
        master_root_id="ORG_MASTER",
        master_root_name="European Parliament Organizations",
        use_mock_falkordb=True
    )
    builder.build_from_chromadb(db)

    print("\nHierarchy (with Master Root):")
    builder.print_tree(include_master_root=True)

    # Save & upload
    builder.save_local()
    builder.upload_to_falkordb()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
