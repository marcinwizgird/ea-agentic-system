"""
Organizational Graph Pipeline

This module provides the main pipeline for:
1. Loading organizational data from various sources (Excel, etc.)
2. Vectorizing documents in vector store (ChromaDB, etc.)
3. Building organizational graph from vectorized documents
4. Uploading graph to graph store (NetworkX local, FalkorDB cloud)

The pipeline follows the architecture:
    DataSource → VectorStore → GraphBuilder → GraphStore
    
Key principle: Graph is constructed FROM vectorized documents in the vector index,
enabling semantic search and similarity-based operations on the graph.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from storage_base import (
    StorageConfig, VectorStoreBase, GraphStoreBase, 
    DataSourceBase, PipelineBase, OrgUnit, Activity
)
from storage_impl import (
    create_vector_store, create_graph_store, create_data_source,
    NetworkXGraphStore
)


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the organizational graph pipeline."""
    
    # Data source settings
    data_source_type: str = "excel"
    data_source_path: str = None  # File or directory path
    activities_as_nodes: bool = True
    activity_id_prefix: str = "ACT"
    
    # Vector store settings
    vector_store_type: str = "mock"  # "mock" or "chroma"
    vector_collection_name: str = "org_documents"
    vector_persist_dir: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Graph store settings
    graph_store_type: str = "networkx"  # "networkx" or "falkordb"
    graph_local_path: str = "./graph_data/organization_graph.json"
    graph_name: str = "org_hierarchy"
    master_root_id: str = "ORG_MASTER"
    master_root_name: str = "European Parliament Organizations"
    
    # FalkorDB settings (if using)
    falkordb_url: Optional[str] = None
    
    # Pipeline flags
    clear_existing: bool = True
    create_vector_index: bool = True
    save_local: bool = True
    verbose: bool = True


# =============================================================================
# Graph Builder (from Vector Store)
# =============================================================================

class OrgGraphBuilder:
    """
    Builds organizational hierarchy graph from vectorized documents.
    
    This builder:
    1. Retrieves documents from vector store
    2. Builds NetworkX graph with proper hierarchy
    3. Computes embeddings for graph nodes
    4. Uploads to graph store with vector indices
    """
    
    def __init__(self, config: PipelineConfig, 
                 vector_store: VectorStoreBase,
                 graph_store: GraphStoreBase):
        self.config = config
        self.vector_store = vector_store
        self.graph_store = graph_store
        
        self.graph = nx.DiGraph()
        self.embeddings: Dict[str, List[float]] = {}
        self.org_roots: List[str] = []
        self.activity_nodes: List[str] = []
    
    def build_from_vector_store(self) -> nx.DiGraph:
        """
        Build graph from documents in vector store.
        
        The vector store contains vectorized documents for each org unit.
        We retrieve all documents and build the hierarchy graph.
        """
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("BUILDING GRAPH FROM VECTOR STORE")
            print("=" * 60)
        
        # Get all documents from vector store
        documents = self.vector_store.get_all_documents()
        
        if self.config.verbose:
            print(f"Documents in vector store: {len(documents)}")
        
        # Clear existing graph
        self.graph = nx.DiGraph()
        self.org_roots = []
        self.activity_nodes = []
        
        # Build nodes from documents
        org_nodes = {}
        activity_nodes = {}
        
        for doc in documents:
            metadata = doc.get('metadata', {})
            content = doc.get('content', '')
            doc_id = doc.get('id', '')
            
            node_type = metadata.get('node_type', 'organization')
            
            if node_type == 'activity':
                # Activity node
                act_id = metadata.get('activity_id', doc_id)
                activity_nodes[act_id] = {
                    'name': metadata.get('name', ''),
                    'description': content,
                    'weight': metadata.get('weight', 0),
                    'parent_org': metadata.get('parent_org', ''),
                    'parent_org_name': metadata.get('parent_org_name', ''),
                    'level': metadata.get('level', 0),
                    'entity_type': 'activity',
                    'node_type': 'activity'
                }
            else:
                # Organizational node
                code = metadata.get('entity_code', doc_id)
                org_nodes[code] = {
                    'name': metadata.get('entity_name', code),
                    'level': metadata.get('level', 0),
                    'parent_code': metadata.get('parent_code', ''),
                    'entity_type': metadata.get('entity_type', 'organization'),
                    'node_type': 'organization',
                    'content': content
                }
        
        if self.config.verbose:
            print(f"Org nodes: {len(org_nodes)}")
            print(f"Activity nodes: {len(activity_nodes)}")
        
        # Add org nodes to graph
        for code, data in org_nodes.items():
            self.graph.add_node(code, **data)
            
            # Get embedding from vector store
            embedding = self.vector_store.get_embedding(data.get('content', code))
            self.embeddings[code] = embedding
            
            # Track roots
            if data.get('level', 0) == 0:
                self.org_roots.append(code)
        
        # Add activity nodes to graph
        for act_id, data in activity_nodes.items():
            self.graph.add_node(act_id, **data)
            self.activity_nodes.append(act_id)
            
            # Get embedding
            embedding = self.vector_store.get_embedding(data.get('description', act_id))
            self.embeddings[act_id] = embedding
        
        # Build org hierarchy edges
        for code, data in org_nodes.items():
            parent = data.get('parent_code')
            if parent and parent in self.graph.nodes:
                self.graph.add_edge(parent, code, edge_type='HAS_CHILD')
        
        # Build activity edges
        for act_id, data in activity_nodes.items():
            parent_org = data.get('parent_org')
            if parent_org and parent_org in self.graph.nodes:
                self.graph.add_edge(parent_org, act_id, 
                                   edge_type='HAS_ACTIVITY',
                                   weight=data.get('weight', 0))
        
        if self.config.verbose:
            print(f"\nGraph built:")
            print(f"  Nodes: {self.graph.number_of_nodes()}")
            print(f"  Edges: {self.graph.number_of_edges()}")
            print(f"  Org roots: {len(self.org_roots)}")
            print(f"  Activities: {len(self.activity_nodes)}")
        
        return self.graph
    
    def upload_to_graph_store(self) -> int:
        """Upload built graph to graph store."""
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("UPLOADING TO GRAPH STORE")
            print("=" * 60)
        
        # Clear existing if configured
        if self.config.clear_existing:
            self.graph_store.clear()
        
        # Add master root
        self.graph_store.add_node(
            self.config.master_root_id,
            name=self.config.master_root_name,
            level=-1,
            node_type='master_root',
            is_master_root=True
        )
        
        # Upload graph with embeddings
        count = self.graph_store.upload_graph(self.graph, self.embeddings)
        
        # Connect org roots to master root
        for root_id in self.org_roots:
            self.graph_store.add_edge(
                self.config.master_root_id, root_id,
                edge_type='HAS_ORGANIZATION'
            )
        
        # Create vector index if configured
        if self.config.create_vector_index:
            self.graph_store.create_vector_index(
                index_name="org_embedding_idx",
                node_label="OrganizationalUnit",
                property_name="embedding",
                dimension=self.config.embedding_dimension
            )
        
        if self.config.verbose:
            stats = self.graph_store.get_stats()
            print(f"\n✓ Upload complete:")
            print(f"  Nodes: {stats.get('nodes', count)}")
            print(f"  Edges: {stats.get('edges', 0)}")
        
        return count
    
    def save_local(self, path: str = None) -> str:
        """Save graph to local file."""
        path = path or self.config.graph_local_path
        
        if isinstance(self.graph_store, NetworkXGraphStore):
            self.graph_store.graph = self.graph
            return self.graph_store.save(path)
        else:
            # Fallback: save NetworkX graph directly
            graph_data = {
                "metadata": {
                    "nodes": self.graph.number_of_nodes(),
                    "edges": self.graph.number_of_edges(),
                    "org_roots": self.org_roots,
                    "master_root_id": self.config.master_root_id
                },
                "nodes": [{"id": n, **d} for n, d in self.graph.nodes(data=True)],
                "edges": [{"source": u, "target": v, **d} for u, v, d in self.graph.edges(data=True)]
            }
            
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"✓ Graph saved: {path}")
            return path


# =============================================================================
# Main Pipeline
# =============================================================================

class OrgGraphPipeline(PipelineBase):
    """
    Main pipeline for organizational graph construction.
    
    Workflow:
    1. Load data from source (Excel files)
    2. Vectorize documents in vector store
    3. Build graph from vectorized documents
    4. Upload to graph store
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize storage config
        self.storage_config = StorageConfig(
            vector_collection_name=self.config.vector_collection_name,
            vector_persist_dir=self.config.vector_persist_dir,
            embedding_model=self.config.embedding_model,
            embedding_dimension=self.config.embedding_dimension,
            graph_local_path=self.config.graph_local_path,
            graph_name=self.config.graph_name,
            master_root_id=self.config.master_root_id,
            master_root_name=self.config.master_root_name,
            falkordb_url=self.config.falkordb_url,
            activities_as_nodes=self.config.activities_as_nodes,
            activity_id_prefix=self.config.activity_id_prefix
        )
        
        # Initialize stores
        self.vector_store: VectorStoreBase = None
        self.graph_store: GraphStoreBase = None
        self.data_source: DataSourceBase = None
        self.graph_builder: OrgGraphBuilder = None
        
        # Statistics
        self._stats = {
            'status': 'initialized',
            'org_units': 0,
            'activities': 0,
            'documents': 0,
            'graph_nodes': 0,
            'graph_edges': 0
        }
    
    def _init_stores(self):
        """Initialize storage components."""
        # Vector store
        self.vector_store = create_vector_store(
            self.config.vector_store_type,
            self.storage_config
        )
        
        # Graph store
        self.graph_store = create_graph_store(
            self.config.graph_store_type,
            self.storage_config
        )
        self.graph_store.connect()
        
        # Data source
        if self.config.data_source_path:
            path = Path(self.config.data_source_path)
            if path.is_file():
                self.data_source = create_data_source(
                    self.config.data_source_type,
                    file_path=str(path),
                    activities_as_nodes=self.config.activities_as_nodes,
                    activity_id_prefix=self.config.activity_id_prefix
                )
            else:
                self.data_source = create_data_source(
                    self.config.data_source_type,
                    directory=str(path),
                    activities_as_nodes=self.config.activities_as_nodes,
                    activity_id_prefix=self.config.activity_id_prefix
                )
        
        # Graph builder
        self.graph_builder = OrgGraphBuilder(
            self.config,
            self.vector_store,
            self.graph_store
        )
    
    def _vectorize_data(self, org_units: List[OrgUnit], activities: List[Activity]) -> int:
        """Vectorize organizational data in vector store."""
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("VECTORIZING DATA")
            print("=" * 60)
        
        if self.config.clear_existing:
            self.vector_store.clear()
        
        documents = []
        
        # Create documents for org units
        for org in org_units:
            # Build content string
            activities_text = "\n".join([
                f"- ({w}%) {a}" 
                for a, w in zip(org.activities, org.activity_weights)
            ]) if org.activities else ""
            
            content = f"""Organizational Unit: {org.name}
Code: {org.code}
Level: {org.level}
Type: {org.entity_type}
Parent: {org.parent_code or 'Root'}

Activities:
{activities_text}"""
            
            documents.append({
                'content': content,
                'metadata': {
                    'entity_code': org.code,
                    'entity_name': org.name,
                    'level': org.level,
                    'parent_code': org.parent_code or '',
                    'entity_type': org.entity_type,
                    'node_type': 'organization',
                    'activity_count': len(org.activities)
                }
            })
        
        # Create documents for activities (if as nodes)
        for activity in activities:
            content = activity.description
            
            documents.append({
                'content': content,
                'metadata': {
                    'activity_id': activity.id,
                    'name': activity.description[:50] + '...' if len(activity.description) > 50 else activity.description,
                    'weight': activity.weight,
                    'parent_org': activity.parent_org_code,
                    'parent_org_name': activity.parent_org_name,
                    'node_type': 'activity',
                    'level': -1  # Activities don't have org level
                }
            })
        
        # Add to vector store
        ids = [doc['metadata'].get('entity_code') or doc['metadata'].get('activity_id') 
               for doc in documents]
        self.vector_store.add_documents(documents, ids=ids)
        
        if self.config.verbose:
            print(f"Vectorized {len(documents)} documents")
            print(f"  Org units: {len(org_units)}")
            print(f"  Activities: {len(activities)}")
        
        self._stats['documents'] = len(documents)
        return len(documents)
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Workflow:
        1. Initialize stores
        2. Load data from source
        3. Vectorize in vector store
        4. Build graph from vector store
        5. Upload to graph store
        6. Save locally (if configured)
        
        Returns:
            Pipeline results and statistics
        """
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("ORGANIZATIONAL GRAPH PIPELINE")
            print("=" * 70)
            print(f"Data source: {self.config.data_source_path}")
            print(f"Vector store: {self.config.vector_store_type}")
            print(f"Graph store: {self.config.graph_store_type}")
            print(f"Activities as nodes: {self.config.activities_as_nodes}")
        
        self._stats['status'] = 'running'
        
        # Step 1: Initialize
        self._init_stores()
        
        # Step 2: Load data
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("STEP 1: LOADING DATA")
            print("=" * 60)
        
        org_units, activities = self.data_source.load()
        self._stats['org_units'] = len(org_units)
        self._stats['activities'] = len(activities)
        
        # Step 3: Vectorize
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("STEP 2: VECTORIZING")
            print("=" * 60)
        
        self._vectorize_data(org_units, activities)
        
        # Step 4: Build graph from vector store
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("STEP 3: BUILDING GRAPH FROM VECTOR STORE")
            print("=" * 60)
        
        graph = self.graph_builder.build_from_vector_store()
        self._stats['graph_nodes'] = graph.number_of_nodes()
        self._stats['graph_edges'] = graph.number_of_edges()
        
        # Step 5: Upload to graph store
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("STEP 4: UPLOADING TO GRAPH STORE")
            print("=" * 60)
        
        self.graph_builder.upload_to_graph_store()
        
        # Step 6: Save locally
        if self.config.save_local:
            if self.config.verbose:
                print("\n" + "=" * 60)
                print("STEP 5: SAVING LOCAL")
                print("=" * 60)
            self.graph_builder.save_local()
        
        self._stats['status'] = 'completed'
        
        if self.config.verbose:
            print("\n" + "=" * 70)
            print("PIPELINE COMPLETE")
            print("=" * 70)
            for key, value in self._stats.items():
                print(f"  {key}: {value}")
        
        return self._stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return self._stats.copy()
    
    def get_graph(self) -> nx.DiGraph:
        """Get the built graph."""
        if self.graph_builder:
            return self.graph_builder.graph
        return nx.DiGraph()
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar nodes using vector store."""
        if self.vector_store:
            return self.vector_store.search(query, k)
        return []
    
    def print_tree(self, show_activities: bool = False):
        """Print the organizational tree."""
        if not self.graph_builder or not self.graph_builder.graph:
            print("No graph available")
            return
        
        graph = self.graph_builder.graph
        roots = self.graph_builder.org_roots
        
        def _print_node(node_id: str, prefix: str = "", is_last: bool = True):
            data = graph.nodes.get(node_id, {})
            name = data.get('name', node_id)[:45]
            node_type = data.get('node_type', 'organization')
            
            connector = "└── " if is_last else "├── "
            
            if node_type == 'activity':
                weight = data.get('weight', 0)
                print(f"{prefix}{connector}📋 ({weight}%) {name}")
            else:
                children = list(graph.successors(node_id))
                org_children = [c for c in children if graph.nodes[c].get('node_type') != 'activity']
                act_children = [c for c in children if graph.nodes[c].get('node_type') == 'activity']
                
                level = data.get('level', 0)
                print(f"{prefix}{connector}[{node_id}] {name} (L{level})")
                
                new_prefix = prefix + ("    " if is_last else "│   ")
                
                # Print org children
                for i, child in enumerate(sorted(org_children)):
                    _print_node(child, new_prefix, i == len(org_children) - 1 and not (show_activities and act_children))
                
                # Print activity children if enabled
                if show_activities:
                    for i, child in enumerate(sorted(act_children)):
                        _print_node(child, new_prefix, i == len(act_children) - 1)
        
        print("\n" + "=" * 60)
        print("ORGANIZATIONAL TREE")
        print("=" * 60)
        print(f"[{self.config.master_root_id}] {self.config.master_root_name}")
        
        for i, root in enumerate(sorted(roots)):
            _print_node(root, "    ", i == len(roots) - 1)


# =============================================================================
# Factory Functions
# =============================================================================

def create_pipeline(
    data_source_path: str,
    activities_as_nodes: bool = True,
    vector_store_type: str = "mock",
    graph_store_type: str = "networkx",
    falkordb_url: str = None,
    verbose: bool = True,
    **kwargs
) -> OrgGraphPipeline:
    """
    Create and configure an organizational graph pipeline.
    
    Args:
        data_source_path: Path to Excel file or directory
        activities_as_nodes: Create activity leaf nodes
        vector_store_type: "mock" or "chroma"
        graph_store_type: "networkx" or "falkordb"
        falkordb_url: FalkorDB connection URL (if using)
        verbose: Print progress
        **kwargs: Additional config options
    
    Returns:
        Configured OrgGraphPipeline instance
    """
    config = PipelineConfig(
        data_source_path=data_source_path,
        activities_as_nodes=activities_as_nodes,
        vector_store_type=vector_store_type,
        graph_store_type=graph_store_type,
        falkordb_url=falkordb_url,
        verbose=verbose,
        **kwargs
    )
    return OrgGraphPipeline(config)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("ORGANIZATIONAL GRAPH PIPELINE")
    print("=" * 70)
    
    # Default test path
    data_path = "./data/samples"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    # Create and run pipeline
    pipeline = create_pipeline(
        data_source_path=data_path,
        activities_as_nodes=True,
        vector_store_type="mock",
        graph_store_type="networkx",
        verbose=True
    )
    
    results = pipeline.run()
    
    print("\n")
    pipeline.print_tree(show_activities=False)
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
