# Organizational AI Prototype

Modular prototype for organizational hierarchy processing, capability mapping, and bipartite matching.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE                                        │
│                           (pipeline.py)                                      │
│                                                                              │
│   DataSource ──▶ VectorStore ──▶ GraphBuilder ──▶ GraphStore                │
│    (Excel)      (ChromaDB)      (from vectors)   (NX/FalkorDB)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Storage Abstractions (storage_base.py, storage_impl.py)

| Abstraction | Implementations | Purpose |
|-------------|-----------------|---------|
| `VectorStoreBase` | MockVectorStore, ChromaVectorStore | Embedding storage & similarity search |
| `GraphStoreBase` | NetworkXGraphStore, FalkorDBGraphStore | Graph database operations |
| `DataSourceBase` | ExcelDataSource | Load organizational data |

### Key Principle
**Graph is constructed FROM vectorized documents in the vector index**, enabling:
- Semantic search on graph nodes
- Similarity-based operations
- Unified embedding space for matching

## Quick Start

```python
from pipeline import create_pipeline

# Create and run pipeline
pipeline = create_pipeline(
    data_source_path="./data/activities.xlsx",
    activities_as_nodes=True,
    vector_store_type="mock",     # or "chroma"
    graph_store_type="networkx",  # or "falkordb"
)

results = pipeline.run()
pipeline.print_tree()

# Semantic search
results = pipeline.search_similar("budget management", k=5)
```

## Detailed Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION                                │
│  Excel Files → ExcelDataSource → VectorStore (ChromaDB)                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────────┐               ┌───────────────────────────┐
│   ORGANIZATIONAL GRAPH    │               │   CAPABILITY GRAPH        │
│   org_graph_builder.py    │               │   capability_graph.py     │
│                           │               │                           │
│   [ORG_MASTER]            │               │   [CAP_ROOT]              │
│     └── DG CASP           │               │     ├── STEERING          │
│           ├── Dir A       │               │     ├── CORE              │
│           │   └── Units   │               │     └── SUPPORTING        │
│           │       └── 📋  │ ◄─ Activities │                           │
│           └── Dir B       │    as nodes   │                           │
└───────────────────────────┘               └───────────────────────────┘
            │                                               │
            ▼                                               ▼
┌───────────────────────────┐               ┌───────────────────────────┐
│   REFINEMENT AGENT        │               │   CAPABILITY REFINEMENT   │
│   refinement_agent.py     │               │   capability_refinement.py│
│                           │               │                           │
│   • Augmented descriptions│               │   • Enhanced descriptions │
│   • Activity summaries    │               │   • Capability keywords   │
│   • Bidirectional         │               │   • Top-down refinement   │
└───────────────────────────┘               └───────────────────────────┘
            │                                               │
            └───────────────────┬───────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      BIPARTITE MATCHING                                 │
│                      bipartite_matcher.py                               │
│                                                                         │
│   Matching Modes:                                                       │
│   • hybrid       - Semantic pre-filter + LLM judge (default)           │
│   • llm_only     - Pure LLM matching (most accurate)                   │
│   • llm_prescreened - LLM category filter + LLM matching               │
│   • semantic_only - Embedding similarity only (fastest)                │
│                                                                         │
│   Org Leaves ←──────── Hybrid Search + LLM Judge ────────→ Cap Leaves  │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DUAL STORAGE                                    │
│            NetworkX (local JSON) + FalkorDB (graph database)           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Components

### Core Pipeline
| Module | Description |
|--------|-------------|
| `pipeline.py` | **Main orchestrator** - DataSource → VectorStore → GraphBuilder → GraphStore |
| `storage_base.py` | Abstract interfaces for VectorStore, GraphStore, DataSource |
| `storage_impl.py` | Concrete implementations (Mock, Chroma, NetworkX, FalkorDB) |

### Organizational Graph
| Module | Description |
|--------|-------------|
| `org_graph_builder.py` | EP-specific org hierarchy parser with activities as nodes |
| `refinement_agent.py` | Augmented org description refinement |

### Capability Graph
| Module | Description |
|--------|-------------|
| `capability_graph.py` | Capability hierarchy from CSV → FalkorDB |
| `capability_refinement.py` | Capability description refinement |

### Matching
| Module | Description |
|--------|-------------|
| `bipartite_matcher.py` | Bipartite matching with multiple modes (hybrid, llm_only, semantic) |

### Legacy (still functional)
| Module | Description |
|--------|-------------|
| `vector_db.py` | ChromaDB with HuggingFace embeddings |
| `ingestion.py` | Excel → ChromaDB ingestion |
| `graph_builder.py` | Org hierarchy: ChromaDB → NetworkX → FalkorDB |
| `llm.py` | Claude integration |

## EP Organizational Hierarchy Rules

The system handles European Parliament-specific organizational codes:

| Pattern | Level | Parent | Example |
|---------|-------|--------|---------|
| `XX` | 0 (DG) | None | `22` |
| `XX-NN` | 1 (Direct Unit) | `XX` | `22-10` → parent `22` |
| `XXA` | 1 (Directorate) | `XX` | `22A` → parent `22` |
| `XXANN` | 2 (Unit) | `XXA` | `22A10` → parent `22A` |
| `XXANNNN` | 3 (Sub-unit) | `XXANN` | `22B0040` → parent `22B00` |

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `test_pipeline.ipynb` | **Main test notebook** - full pipeline with all options |
| `test_org_hierarchy.ipynb` | Org hierarchy building with comparison |
| `test_capability_tree.ipynb` | Capability tree building |
| `test_capability_mapping.ipynb` | Org-to-capability matching |

## Usage Examples

### Using the Pipeline (Recommended)

```python
from pipeline import create_pipeline

# Create and run pipeline
pipeline = create_pipeline(
    data_source_path="./data/activities.xlsx",
    activities_as_nodes=True,
    vector_store_type="mock",     # or "chroma"
    graph_store_type="networkx",  # or "falkordb"
)

results = pipeline.run()
pipeline.print_tree()

# Semantic search
results = pipeline.search_similar("budget management", k=5)
```

### Using Individual Components

```python
# 1. Create storage instances
from storage_impl import create_vector_store, create_graph_store, create_data_source

vector_store = create_vector_store("chroma")
graph_store = create_graph_store("networkx")
data_source = create_data_source("excel", file_path="./data/activities.xlsx")

# 2. Load and vectorize data
org_units, activities = data_source.load()
# ... vectorize documents ...

# 3. Build graph from vector store
# ... build graph ...
org_builder.build_from_chromadb(db)
org_builder.upload_to_falkordb()

# 2. Build capability graph
from capability_graph import create_capability_graph_builder

cap_builder = create_capability_graph_builder(falkordb_url="redis://...")
cap_builder.build_from_csv("business_capabilities.csv")
cap_builder.upload_to_falkordb()

# 3. Refine descriptions (stored in both NetworkX and FalkorDB)
from refinement_agent import create_refinement_agent
from llm import create_llm

llm = create_llm()
org_agent = create_refinement_agent(org_builder, llm, strategy="bidirectional")
org_agent.run()

# 4. Run bipartite matching
from bipartite_matcher import create_bipartite_matcher

matcher = create_bipartite_matcher(
    org_builder.graph, cap_builder.graph, llm,
    use_llm_judge=True
)
results = matcher.run_matching()
matcher.save_results("matches.json")
```

## Refinement Agent

Augmented description refinement preserving ALL original activity details:

```python
from refinement_agent import create_refinement_agent

agent = create_refinement_agent(
    graph_builder=org_builder,
    llm=llm,
    strategy="bidirectional",  # "top_down", "bottom_up", "bidirectional"
    sync_to_falkordb=True,     # Immediate sync to FalkorDB
    verbose=True
)

result = agent.run()
# {strategy, total_nodes, refined_descriptions, activity_summaries, synced_to_falkordb}

agent.export_refinements("refinements.json")
```

### Output Fields
- `refined_description`: 4-6 sentence augmented description with ALL activity details
- `refined_activity_summary`: Bullet-point summary grouping activities by theme

## Bipartite Matching

Matches org leaves to capability leaves:

```
Algorithm:
1. Get leaves from both graphs
2. For each org leaf:
   a. Hybrid search (semantic + keyword) against cap leaves
   b. Select top-K candidates
   c. LLM evaluates each candidate (score + justification)
3. If no match at leaf level:
   a. Try higher capability levels (hierarchical fallback)
4. Build bipartite graph with match edges + justifications
5. Store in NetworkX and FalkorDB
```

### Loading Graphs (No Reprocessing)

```python
# From local JSON files
matcher = create_matcher_from_local(
    org_graph_path="org_graph.json",
    cap_graph_path="cap_graph.json",
    llm=llm
)

# From FalkorDB
from falkordb import FalkorDB
client = FalkorDB.from_url("redis://...")
matcher = BipartiteCapabilityMatcher.load_from_falkordb(
    client, "org_hierarchy", "capability_map", llm
)
```

### Match Output with Justification

```json
{
  "org_id": "24A10",
  "org_name": "Budget Committee Secretariat",
  "matches": [
    {
      "capability_id": "SBA_abc123",
      "capability_name": "Budgetary Planning",
      "semantic_score": 0.78,
      "llm_score": 0.85,
      "combined_score": 0.815,
      "match_type": "STRONG",
      "justification": "Strong functional alignment: budget preparation activities (40%) directly map to budgetary planning capability...",
      "key_overlaps": "budget preparation, financial planning, committee support",
      "gaps": "None significant"
    }
  ],
  "fallback_level": 0
}
```

## Dual Storage

All components support immediate dual storage:

| Storage | Purpose |
|---------|---------|
| **NetworkX + JSON** | Local persistence, fast iteration |
| **FalkorDB** | Production graph database |

## FalkorDB Graph Structure

### Organization Graph (`org_hierarchy`)
- Labels: `MasterRoot`, `OrgRoot`, `OrganizationalUnit`
- Properties: `node_id`, `name`, `level`, `activities`, `refined_description`, `refined_activity_summary`

### Capability Graph (`capability_map`)
- Labels: `CapabilityRoot`, `Category`, `BusinessArea`, `SubBusinessArea`
- Properties: `node_id`, `name`, `description`, `refined_description`, `capability_keywords`

### Bipartite Matches (`org_capability_matches`)
- Relationship: `(OrgUnit)-[MATCHES_CAPABILITY]->(Capability)`
- Properties: `semantic_score`, `llm_score`, `combined_score`, `match_type`, `justification`

## Notebook

Use `notebook_with_refinement.ipynb` for interactive testing:
- Components 1-4: Original pipeline (VectorDB, Ingestion, Graph, LLM)
- Component 5: Org refinement
- Component 6: Capability graph + refinement
- Component 7: Bipartite matching
- Independent test cells for each module

## Requirements

```
pandas>=2.0
openpyxl>=3.1.0
networkx>=3.0
numpy>=1.24.0
langchain-core>=0.2.0
langchain-chroma>=0.1.0
chromadb>=0.4.0
langchain-huggingface>=0.0.3
sentence-transformers>=2.2.0
langchain-anthropic>=0.1.0
falkordb>=1.0.0
redis>=4.0.0
```
