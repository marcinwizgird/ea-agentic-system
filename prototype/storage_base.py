"""
Storage Abstractions for Organization Graph System

This module defines abstract interfaces for:
- VectorStore: Embedding storage and similarity search
- GraphStore: Graph database storage (NetworkX local, FalkorDB cloud)
- DataSource: Loading organizational data from various sources

These abstractions enable swapping implementations without changing business logic.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StorageConfig:
    """Unified configuration for storage systems."""
    
    # Vector store settings
    vector_collection_name: str = "org_documents"
    vector_persist_dir: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Graph store settings
    graph_local_path: str = "./graph_data/organization_graph.json"
    graph_name: str = "org_hierarchy"
    master_root_id: str = "ORG_MASTER"
    master_root_name: str = "European Parliament Organizations"
    
    # FalkorDB settings
    falkordb_url: Optional[str] = None
    falkordb_host: str = "localhost"
    falkordb_port: int = 6379
    falkordb_password: Optional[str] = None
    
    # Org structure settings
    activities_as_nodes: bool = True
    activity_id_prefix: str = "ACT"


# =============================================================================
# Vector Store Abstraction
# =============================================================================

class VectorStoreBase(ABC):
    """Abstract base class for vector storage."""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]], ids: List[str] = None) -> List[str]:
        """
        Add documents to vector store.
        
        Args:
            documents: List of dicts with 'content' and 'metadata' keys
            ids: Optional document IDs
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def add_embeddings(self, embeddings: List[List[float]], 
                       metadatas: List[Dict[str, Any]], 
                       ids: List[str]) -> List[str]:
        """
        Add pre-computed embeddings to store.
        
        Args:
            embeddings: List of embedding vectors
            metadatas: List of metadata dicts
            ids: Document IDs
            
        Returns:
            List of document IDs
        """
        pass
    
    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents by text query.
        
        Returns:
            List of dicts with 'content', 'metadata', 'score' keys
        """
        pass
    
    @abstractmethod
    def search_by_vector(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search by embedding vector."""
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        pass
    
    @abstractmethod
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all documents from store."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get document count."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents."""
        pass


# =============================================================================
# Graph Store Abstraction
# =============================================================================

class GraphStoreBase(ABC):
    """Abstract base class for graph storage."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to graph store. Returns True if successful."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all nodes and edges."""
        pass
    
    @abstractmethod
    def add_node(self, node_id: str, **properties) -> bool:
        """Add a node with properties."""
        pass
    
    @abstractmethod
    def add_edge(self, source_id: str, target_id: str, edge_type: str = "HAS_CHILD", **properties) -> bool:
        """Add an edge between nodes."""
        pass
    
    @abstractmethod
    def upload_graph(self, graph: nx.DiGraph, embeddings: Dict[str, List[float]] = None) -> int:
        """
        Upload NetworkX graph to store.
        
        Args:
            graph: NetworkX DiGraph
            embeddings: Optional dict of node_id -> embedding vector
            
        Returns:
            Number of nodes uploaded
        """
        pass
    
    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node by ID."""
        pass
    
    @abstractmethod
    def get_children(self, node_id: str) -> List[str]:
        """Get child node IDs."""
        pass
    
    @abstractmethod
    def get_parent(self, node_id: str) -> Optional[str]:
        """Get parent node ID."""
        pass
    
    @abstractmethod
    def vector_search(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search nodes by embedding similarity."""
        pass
    
    @abstractmethod
    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute Cypher query."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        pass
    
    @abstractmethod
    def create_vector_index(self, index_name: str, node_label: str, 
                           property_name: str, dimension: int) -> bool:
        """Create vector index on node property."""
        pass


# =============================================================================
# Data Source Abstraction
# =============================================================================

@dataclass
class OrgUnit:
    """Represents an organizational unit."""
    code: str
    name: str
    level: int
    parent_code: Optional[str]
    entity_type: str
    activities: List[str] = field(default_factory=list)
    activity_weights: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Activity:
    """Represents an activity within an org unit."""
    id: str
    description: str
    weight: int
    parent_org_code: str
    parent_org_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataSourceBase(ABC):
    """Abstract base class for organizational data sources."""
    
    @abstractmethod
    def load(self) -> Tuple[List[OrgUnit], List[Activity]]:
        """
        Load organizational data.
        
        Returns:
            Tuple of (org_units, activities)
        """
        pass
    
    @abstractmethod
    def get_hierarchy(self) -> Dict[str, List[str]]:
        """
        Get parent -> children mapping.
        
        Returns:
            Dict mapping parent code to list of child codes
        """
        pass


# =============================================================================
# Pipeline Interface
# =============================================================================

class PipelineBase(ABC):
    """Abstract base class for data processing pipelines."""
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the pipeline.
        
        Returns:
            Dict with pipeline results and statistics
        """
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        pass


# =============================================================================
# Factory Protocol
# =============================================================================

class StorageFactory:
    """Factory for creating storage instances."""
    
    _vector_store_registry: Dict[str, type] = {}
    _graph_store_registry: Dict[str, type] = {}
    _data_source_registry: Dict[str, type] = {}
    
    @classmethod
    def register_vector_store(cls, name: str, store_class: type):
        """Register a vector store implementation."""
        cls._vector_store_registry[name] = store_class
    
    @classmethod
    def register_graph_store(cls, name: str, store_class: type):
        """Register a graph store implementation."""
        cls._graph_store_registry[name] = store_class
    
    @classmethod
    def register_data_source(cls, name: str, source_class: type):
        """Register a data source implementation."""
        cls._data_source_registry[name] = source_class
    
    @classmethod
    def create_vector_store(cls, name: str, config: StorageConfig, **kwargs) -> VectorStoreBase:
        """Create vector store instance."""
        if name not in cls._vector_store_registry:
            raise ValueError(f"Unknown vector store: {name}. Available: {list(cls._vector_store_registry.keys())}")
        return cls._vector_store_registry[name](config, **kwargs)
    
    @classmethod
    def create_graph_store(cls, name: str, config: StorageConfig, **kwargs) -> GraphStoreBase:
        """Create graph store instance."""
        if name not in cls._graph_store_registry:
            raise ValueError(f"Unknown graph store: {name}. Available: {list(cls._graph_store_registry.keys())}")
        return cls._graph_store_registry[name](config, **kwargs)
    
    @classmethod
    def create_data_source(cls, name: str, **kwargs) -> DataSourceBase:
        """Create data source instance."""
        if name not in cls._data_source_registry:
            raise ValueError(f"Unknown data source: {name}. Available: {list(cls._data_source_registry.keys())}")
        return cls._data_source_registry[name](**kwargs)
