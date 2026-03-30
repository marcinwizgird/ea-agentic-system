"""
Concrete Storage Implementations

Implements:
- ChromaVectorStore: ChromaDB with HuggingFace embeddings
- MockVectorStore: In-memory mock for testing
- NetworkXGraphStore: Local NetworkX graph with JSON persistence
- FalkorDBGraphStore: FalkorDB cloud graph database
- ExcelDataSource: Load org data from Excel files
"""

import json
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import networkx as nx
import numpy as np

from storage_base import (
    VectorStoreBase, GraphStoreBase, DataSourceBase,
    StorageConfig, StorageFactory, OrgUnit, Activity
)


# =============================================================================
# Vector Store Implementations
# =============================================================================

class MockVectorStore(VectorStoreBase):
    """In-memory mock vector store for testing."""
    
    def __init__(self, config: StorageConfig, **kwargs):
        self.config = config
        self._documents: List[Dict[str, Any]] = []
        self._embeddings: List[List[float]] = []
        self._ids: List[str] = []
        print("✓ VectorStore: Mock (in-memory)")
    
    def _mock_embedding(self, text: str) -> List[float]:
        """Generate deterministic mock embedding from text hash."""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = []
        for i in range(self.config.embedding_dimension):
            val = (hash_bytes[i % len(hash_bytes)] / 255.0) * 2 - 1
            embedding.append(val)
        # Normalize
        norm = np.sqrt(sum(x**2 for x in embedding))
        return [x / norm for x in embedding] if norm > 0 else embedding
    
    def add_documents(self, documents: List[Dict[str, Any]], ids: List[str] = None) -> List[str]:
        ids = ids or [f"doc_{len(self._ids) + i}" for i in range(len(documents))]
        for i, doc in enumerate(documents):
            content = doc.get('content', '')
            self._documents.append(doc)
            self._embeddings.append(self._mock_embedding(content))
            self._ids.append(ids[i])
        return ids
    
    def add_embeddings(self, embeddings: List[List[float]], 
                       metadatas: List[Dict[str, Any]], 
                       ids: List[str]) -> List[str]:
        for i, emb in enumerate(embeddings):
            self._documents.append({'content': '', 'metadata': metadatas[i]})
            self._embeddings.append(emb)
            self._ids.append(ids[i])
        return ids
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self._mock_embedding(query)
        return self.search_by_vector(query_emb, k)
    
    def search_by_vector(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if not self._embeddings:
            return []
        
        # Compute cosine similarities
        scores = []
        for i, emb in enumerate(self._embeddings):
            dot = sum(a * b for a, b in zip(embedding, emb))
            scores.append((i, dot))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:k]:
            results.append({
                'content': self._documents[idx].get('content', ''),
                'metadata': self._documents[idx].get('metadata', {}),
                'score': score,
                'id': self._ids[idx]
            })
        return results
    
    def get_embedding(self, text: str) -> List[float]:
        return self._mock_embedding(text)
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        return [
            {'id': self._ids[i], **self._documents[i]}
            for i in range(len(self._documents))
        ]
    
    def count(self) -> int:
        return len(self._documents)
    
    def clear(self) -> None:
        self._documents = []
        self._embeddings = []
        self._ids = []


class ChromaVectorStore(VectorStoreBase):
    """ChromaDB vector store with HuggingFace embeddings."""
    
    def __init__(self, config: StorageConfig, **kwargs):
        self.config = config
        self._store = None
        self._embeddings_model = None
        
        try:
            from langchain_chroma import Chroma
            from langchain_huggingface import HuggingFaceEmbeddings
            
            self._embeddings_model = HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            Path(config.vector_persist_dir).mkdir(parents=True, exist_ok=True)
            
            self._store = Chroma(
                collection_name=config.vector_collection_name,
                embedding_function=self._embeddings_model,
                persist_directory=config.vector_persist_dir
            )
            print(f"✓ VectorStore: ChromaDB ({config.vector_persist_dir})")
            
        except ImportError as e:
            raise ImportError(f"ChromaDB dependencies not available: {e}")
    
    def add_documents(self, documents: List[Dict[str, Any]], ids: List[str] = None) -> List[str]:
        from langchain_core.documents import Document
        
        docs = [
            Document(page_content=d.get('content', ''), metadata=d.get('metadata', {}))
            for d in documents
        ]
        return self._store.add_documents(docs, ids=ids)
    
    def add_embeddings(self, embeddings: List[List[float]], 
                       metadatas: List[Dict[str, Any]], 
                       ids: List[str]) -> List[str]:
        # ChromaDB requires texts, so we use empty strings with pre-computed embeddings
        texts = [""] * len(embeddings)
        self._store._collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
            documents=texts
        )
        return ids
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        results = self._store.similarity_search_with_score(query, k=k)
        return [
            {'content': doc.page_content, 'metadata': doc.metadata, 'score': score}
            for doc, score in results
        ]
    
    def search_by_vector(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        results = self._store.similarity_search_by_vector_with_relevance_scores(embedding, k=k)
        return [
            {'content': doc.page_content, 'metadata': doc.metadata, 'score': score}
            for doc, score in results
        ]
    
    def get_embedding(self, text: str) -> List[float]:
        return self._embeddings_model.embed_query(text)
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        collection = self._store._collection
        data = collection.get(include=['documents', 'metadatas'])
        return [
            {'id': data['ids'][i], 'content': data['documents'][i], 'metadata': data['metadatas'][i]}
            for i in range(len(data['ids']))
        ]
    
    def count(self) -> int:
        try:
            return self._store._collection.count()
        except:
            return 0
    
    def clear(self) -> None:
        try:
            collection = self._store._collection
            ids = collection.get()["ids"]
            if ids:
                collection.delete(ids=ids)
        except:
            pass


# =============================================================================
# Graph Store Implementations
# =============================================================================

class NetworkXGraphStore(GraphStoreBase):
    """Local NetworkX graph with JSON persistence."""
    
    def __init__(self, config: StorageConfig, **kwargs):
        self.config = config
        self.graph = nx.DiGraph()
        self._embeddings: Dict[str, List[float]] = {}
        print(f"✓ GraphStore: NetworkX (local)")
    
    def connect(self) -> bool:
        return True  # Always connected for local store
    
    def clear(self) -> None:
        self.graph = nx.DiGraph()
        self._embeddings = {}
    
    def add_node(self, node_id: str, **properties) -> bool:
        self.graph.add_node(node_id, **properties)
        return True
    
    def add_edge(self, source_id: str, target_id: str, edge_type: str = "HAS_CHILD", **properties) -> bool:
        self.graph.add_edge(source_id, target_id, edge_type=edge_type, **properties)
        return True
    
    def upload_graph(self, graph: nx.DiGraph, embeddings: Dict[str, List[float]] = None) -> int:
        self.graph = graph.copy()
        self._embeddings = embeddings or {}
        return self.graph.number_of_nodes()
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id in self.graph.nodes:
            return dict(self.graph.nodes[node_id])
        return None
    
    def get_children(self, node_id: str) -> List[str]:
        return list(self.graph.successors(node_id))
    
    def get_parent(self, node_id: str) -> Optional[str]:
        preds = list(self.graph.predecessors(node_id))
        return preds[0] if preds else None
    
    def vector_search(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if not self._embeddings:
            return []
        
        scores = []
        for node_id, emb in self._embeddings.items():
            dot = sum(a * b for a, b in zip(embedding, emb))
            scores.append((node_id, dot))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for node_id, score in scores[:k]:
            node_data = self.get_node(node_id) or {}
            results.append({
                'node_id': node_id,
                'score': score,
                **node_data
            })
        return results
    
    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # NetworkX doesn't support Cypher - return empty
        raise NotImplementedError("NetworkX does not support Cypher queries")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'embeddings': len(self._embeddings),
            'backend': 'networkx'
        }
    
    def create_vector_index(self, index_name: str, node_label: str, 
                           property_name: str, dimension: int) -> bool:
        # No-op for NetworkX - embeddings stored in memory
        return True
    
    def save(self, path: str = None) -> str:
        """Save graph to JSON file."""
        path = path or self.config.graph_local_path
        
        graph_data = {
            "metadata": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "master_root_id": self.config.master_root_id
            },
            "nodes": [
                {"id": n, **{k: v for k, v in d.items() if not isinstance(v, (list, dict)) or k in ['activities', 'activity_weights']}}
                for n, d in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self.graph.edges(data=True)
            ]
        }
        
        # Handle complex types
        for node in graph_data["nodes"]:
            for key, value in list(node.items()):
                if isinstance(value, (list, dict)):
                    node[key] = value  # Keep as-is for JSON
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Graph saved: {path}")
        return path
    
    def load(self, path: str = None) -> nx.DiGraph:
        """Load graph from JSON file."""
        path = path or self.config.graph_local_path
        
        with open(path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        self.graph = nx.DiGraph()
        
        for node in graph_data.get("nodes", []):
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
        
        for edge in graph_data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            self.graph.add_edge(source, target, **edge)
        
        print(f"✓ Graph loaded: {self.graph.number_of_nodes()} nodes")
        return self.graph


class FalkorDBGraphStore(GraphStoreBase):
    """FalkorDB cloud graph database."""
    
    def __init__(self, config: StorageConfig, **kwargs):
        self.config = config
        self._client = None
        self._graph = None
        self._connected = False
        
        try:
            from falkordb import FalkorDB
            self._falkordb_cls = FalkorDB
        except ImportError:
            self._falkordb_cls = None
            print("⚠ FalkorDB not available")
    
    def connect(self) -> bool:
        if not self._falkordb_cls:
            return False
        
        try:
            if self.config.falkordb_url:
                self._client = self._falkordb_cls.from_url(self.config.falkordb_url)
            else:
                self._client = self._falkordb_cls(
                    host=self.config.falkordb_host,
                    port=self.config.falkordb_port,
                    password=self.config.falkordb_password
                )
            
            self._graph = self._client.select_graph(self.config.graph_name)
            self._connected = True
            print(f"✓ GraphStore: FalkorDB ({self.config.graph_name})")
            return True
            
        except Exception as e:
            print(f"✗ FalkorDB connection failed: {e}")
            return False
    
    def clear(self) -> None:
        if self._connected:
            try:
                self._graph.query("MATCH (n) DETACH DELETE n")
            except:
                pass
    
    def add_node(self, node_id: str, **properties) -> bool:
        if not self._connected:
            return False
        
        # Serialize complex types
        props = {"node_id": node_id}
        for key, value in properties.items():
            if isinstance(value, (list, dict)):
                props[key] = json.dumps(value, ensure_ascii=False)
            elif value is not None:
                props[key] = value
        
        # Determine label
        node_type = properties.get('node_type', 'organization')
        if node_type == 'activity':
            label = 'Activity'
        elif properties.get('level', 0) == 0:
            label = 'DG'
        else:
            label = 'OrganizationalUnit'
        
        props_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
        query = f"CREATE (n:{label} {{{props_str}}})"
        
        try:
            self._graph.query(query, props)
            return True
        except Exception as e:
            print(f"Error adding node {node_id}: {e}")
            return False
    
    def add_edge(self, source_id: str, target_id: str, edge_type: str = "HAS_CHILD", **properties) -> bool:
        if not self._connected:
            return False
        
        query = f"""
        MATCH (a {{node_id: $source}}), (b {{node_id: $target}})
        CREATE (a)-[r:{edge_type}]->(b)
        """
        
        try:
            self._graph.query(query, {"source": source_id, "target": target_id})
            return True
        except Exception as e:
            print(f"Error adding edge {source_id}->{target_id}: {e}")
            return False
    
    def upload_graph(self, graph: nx.DiGraph, embeddings: Dict[str, List[float]] = None) -> int:
        if not self._connected:
            return 0
        
        embeddings = embeddings or {}
        count = 0
        
        # Create nodes
        for node_id in graph.nodes():
            props = dict(graph.nodes[node_id])
            if node_id in embeddings:
                props['embedding'] = embeddings[node_id]
            if self.add_node(node_id, **props):
                count += 1
        
        # Create edges
        for source, target, data in graph.edges(data=True):
            edge_type = data.get('edge_type', 'HAS_CHILD')
            self.add_edge(source, target, edge_type)
        
        print(f"  Uploaded {count} nodes to FalkorDB")
        return count
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if not self._connected:
            return None
        
        try:
            result = self._graph.query(
                "MATCH (n {node_id: $id}) RETURN n",
                {"id": node_id}
            )
            if result.result_set:
                return dict(result.result_set[0][0].properties)
        except:
            pass
        return None
    
    def get_children(self, node_id: str) -> List[str]:
        if not self._connected:
            return []
        
        try:
            result = self._graph.query(
                "MATCH ({node_id: $id})-[:HAS_CHILD|HAS_ACTIVITY]->(child) RETURN child.node_id",
                {"id": node_id}
            )
            return [row[0] for row in result.result_set]
        except:
            return []
    
    def get_parent(self, node_id: str) -> Optional[str]:
        if not self._connected:
            return None
        
        try:
            result = self._graph.query(
                "MATCH (parent)-[:HAS_CHILD|HAS_ACTIVITY]->({node_id: $id}) RETURN parent.node_id",
                {"id": node_id}
            )
            if result.result_set:
                return result.result_set[0][0]
        except:
            pass
        return None
    
    def vector_search(self, embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        if not self._connected:
            return []
        
        try:
            # FalkorDB vector search syntax
            query = f"""
            CALL db.idx.vector.queryNodes(
                'OrganizationalUnit', 
                'embedding', 
                {k}, 
                vecf32($embedding)
            ) YIELD node, score
            RETURN node, score
            ORDER BY score DESC
            """
            result = self._graph.query(query, {"embedding": embedding})
            return [
                {'node_id': row[0].properties.get('node_id'), 'score': row[1], **row[0].properties}
                for row in result.result_set
            ]
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
    
    def query(self, cypher: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if not self._connected:
            return []
        
        try:
            result = self._graph.query(cypher, params or {})
            return [dict(zip(result.header, row)) for row in result.result_set]
        except Exception as e:
            print(f"Query error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        if not self._connected:
            return {'connected': False}
        
        try:
            nodes = self._graph.query("MATCH (n) RETURN count(n) as count").result_set[0][0]
            edges = self._graph.query("MATCH ()-[r]->() RETURN count(r) as count").result_set[0][0]
            return {
                'connected': True,
                'nodes': nodes,
                'edges': edges,
                'backend': 'falkordb'
            }
        except:
            return {'connected': True, 'backend': 'falkordb'}
    
    def create_vector_index(self, index_name: str, node_label: str, 
                           property_name: str, dimension: int) -> bool:
        if not self._connected:
            return False
        
        try:
            query = f"""
            CREATE VECTOR INDEX {index_name}
            FOR (n:{node_label})
            ON (n.{property_name})
            OPTIONS {{dimension: {dimension}, similarityFunction: 'cosine'}}
            """
            self._graph.query(query)
            print(f"✓ Vector index created: {index_name}")
            return True
        except Exception as e:
            print(f"⚠ Vector index creation: {e}")
            return False


# =============================================================================
# Data Source Implementations
# =============================================================================

class ExcelDataSource(DataSourceBase):
    """Load organizational data from Excel files."""
    
    # EP hierarchy parsing patterns
    PATTERNS = [
        (r'^(\d{2})$', 0, lambda m: None, 'dg'),
        (r'^(\d{2})-(\d{2})$', 1, lambda m: m.group(1), 'direct_unit'),
        (r'^(\d{2})([A-Z])$', 1, lambda m: m.group(1), 'directorate'),
        (r'^(\d{2})([A-Z])(\d{2})$', 2, lambda m: m.group(1) + m.group(2), 'unit'),
        (r'^(\d{2})([A-Z])(\d{2})(\d{2})$', 3, lambda m: m.group(1) + m.group(2) + m.group(3), 'sub_unit'),
    ]
    
    def __init__(self, file_path: str = None, directory: str = None,
                 col_code: str = None, col_name: str = None,
                 col_activity: str = None, col_percentage: str = None,
                 activities_as_nodes: bool = True,
                 activity_id_prefix: str = "ACT",
                 **kwargs):
        self.file_path = file_path
        self.directory = directory
        self.col_code = col_code
        self.col_name = col_name
        self.col_activity = col_activity
        self.col_percentage = col_percentage
        self.activities_as_nodes = activities_as_nodes
        self.activity_id_prefix = activity_id_prefix
        
        self._org_units: List[OrgUnit] = []
        self._activities: List[Activity] = []
        self._hierarchy: Dict[str, List[str]] = {}
    
    def _parse_code(self, code: str) -> Tuple[int, Optional[str], str]:
        """Parse org code to get level, parent, and type."""
        code = str(code).strip()
        
        for pattern, level, parent_fn, entity_type in self.PATTERNS:
            match = re.match(pattern, code)
            if match:
                parent = parent_fn(match)
                return level, parent, entity_type
        
        # Fallback for unknown patterns
        return 0, None, 'unknown'
    
    def _detect_column(self, columns: List[str], keywords: List[str]) -> Optional[str]:
        """Auto-detect column by keywords."""
        for col in columns:
            col_lower = col.lower()
            for kw in keywords:
                if kw in col_lower:
                    return col
        return None
    
    def _load_file(self, file_path: Path) -> Tuple[List[OrgUnit], List[Activity]]:
        """Load single Excel file."""
        org_units = []
        activities = []
        
        df = pd.read_excel(file_path)
        columns = df.columns.tolist()
        
        # Auto-detect columns
        col_code = self.col_code or self._detect_column(columns, ['code', 'entity code'])
        col_name = self.col_name or self._detect_column(columns, ['name', 'entity name'])
        col_activity = self.col_activity or self._detect_column(columns, ['activity', 'activities', 'mission'])
        col_percentage = self.col_percentage or self._detect_column(columns, ['%', 'percent', 'weight'])
        
        if not col_code:
            col_code = columns[0]
        
        # Group by org code
        org_data: Dict[str, Dict] = {}
        
        for _, row in df.iterrows():
            code = str(row.get(col_code, "")).strip()
            if not code or code == "nan":
                continue
            
            if code not in org_data:
                level, parent, entity_type = self._parse_code(code)
                org_data[code] = {
                    'name': str(row.get(col_name, code)) if col_name else code,
                    'level': level,
                    'parent': parent,
                    'entity_type': entity_type,
                    'activities': [],
                    'weights': []
                }
            
            if col_activity:
                activity = str(row.get(col_activity, "")).strip()
                weight = 0
                if col_percentage and pd.notna(row.get(col_percentage)):
                    try:
                        weight = int(float(row.get(col_percentage)))
                    except:
                        pass
                
                if activity and activity != "nan":
                    org_data[code]['activities'].append(activity)
                    org_data[code]['weights'].append(weight)
        
        # Create OrgUnit objects
        for code, data in org_data.items():
            org_unit = OrgUnit(
                code=code,
                name=data['name'],
                level=data['level'],
                parent_code=data['parent'],
                entity_type=data['entity_type'],
                activities=data['activities'] if not self.activities_as_nodes else [],
                activity_weights=data['weights'] if not self.activities_as_nodes else [],
                metadata={'file': file_path.name}
            )
            org_units.append(org_unit)
            
            # Build hierarchy
            if data['parent']:
                if data['parent'] not in self._hierarchy:
                    self._hierarchy[data['parent']] = []
                self._hierarchy[data['parent']].append(code)
            
            # Create Activity objects if enabled
            if self.activities_as_nodes:
                for idx, (act_text, weight) in enumerate(zip(data['activities'], data['weights'])):
                    act_id = f"{code}_{self.activity_id_prefix}_{idx+1:03d}"
                    activity = Activity(
                        id=act_id,
                        description=act_text,
                        weight=weight,
                        parent_org_code=code,
                        parent_org_name=data['name'],
                        metadata={'index': idx}
                    )
                    activities.append(activity)
        
        return org_units, activities
    
    def load(self) -> Tuple[List[OrgUnit], List[Activity]]:
        """Load all organizational data."""
        self._org_units = []
        self._activities = []
        self._hierarchy = {}
        
        files = []
        if self.file_path:
            files.append(Path(self.file_path))
        elif self.directory:
            dir_path = Path(self.directory)
            files.extend(dir_path.glob("*.xlsx"))
            files.extend(dir_path.glob("*.xls"))
        
        for file_path in files:
            print(f"Loading: {file_path.name}")
            orgs, acts = self._load_file(file_path)
            self._org_units.extend(orgs)
            self._activities.extend(acts)
            print(f"  → {len(orgs)} orgs, {len(acts)} activities")
        
        print(f"\nTotal: {len(self._org_units)} orgs, {len(self._activities)} activities")
        return self._org_units, self._activities
    
    def get_hierarchy(self) -> Dict[str, List[str]]:
        return self._hierarchy


# =============================================================================
# Register Implementations
# =============================================================================

StorageFactory.register_vector_store("mock", MockVectorStore)
StorageFactory.register_vector_store("chroma", ChromaVectorStore)
StorageFactory.register_graph_store("networkx", NetworkXGraphStore)
StorageFactory.register_graph_store("falkordb", FalkorDBGraphStore)
StorageFactory.register_data_source("excel", ExcelDataSource)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_vector_store(store_type: str = "mock", config: StorageConfig = None, **kwargs) -> VectorStoreBase:
    """Create vector store instance."""
    config = config or StorageConfig()
    return StorageFactory.create_vector_store(store_type, config, **kwargs)


def create_graph_store(store_type: str = "networkx", config: StorageConfig = None, **kwargs) -> GraphStoreBase:
    """Create graph store instance."""
    config = config or StorageConfig()
    return StorageFactory.create_graph_store(store_type, config, **kwargs)


def create_data_source(source_type: str = "excel", **kwargs) -> DataSourceBase:
    """Create data source instance."""
    return StorageFactory.create_data_source(source_type, **kwargs)
