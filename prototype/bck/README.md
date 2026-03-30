# Organizational AI Prototype

Simplified, modular prototype for organizational hierarchy processing.

## Components

| File | Description |
|------|-------------|
| `vector_db.py` | ChromaDB with HuggingFace embeddings |
| `ingestion.py` | Excel file ingestion → LangChain Documents |
| `graph_builder.py` | NetworkX hierarchy graph |
| `llm.py` | Claude Opus 4.6 integration |
| `agent.py` | Description refinement agent |
| `main.py` | Pipeline orchestration |
| `notebook.ipynb` | Interactive testing notebook |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python main.py --data ./data --output ./output --mock

# With Claude Opus 4.6
export ANTHROPIC_API_KEY="your-key"
python main.py --data ./data --output ./output
```

## Component Usage

Each component can be used independently:

```python
# 1. Vector DB
from vector_db import create_vector_db
db = create_vector_db()
db.add_texts(["doc1", "doc2"])
results = db.search("query")

# 2. Ingestion
from ingestion import create_ingestion_plugin
plugin = create_ingestion_plugin(vector_db=db)
plugin.ingest("./excel_files")

# 3. Graph Builder
from graph_builder import create_graph_builder
builder = create_graph_builder()
builder.load_excel("file.xlsx")
builder.print_tree()

# 4. LLM
from llm import create_llm
llm = create_llm()  # Uses ANTHROPIC_API_KEY
response = llm.generate("prompt")

# 5. Agent
from agent import create_agent
agent = create_agent(builder, llm)
agent.run(strategy="top_down")
```

## Notebook

Toggle jobs in the notebook:

```python
RUN_INGESTION = True   # Excel → ChromaDB
RUN_GRAPH = True       # Build hierarchy
RUN_AGENT = True       # LLM refinement
RUN_EXPORT = True      # Save results
USE_MOCK_LLM = True    # False for real Claude
```

## Requirements

- Python 3.9+
- pandas, openpyxl, networkx
- langchain-core, langchain-chroma, chromadb
- langchain-huggingface, sentence-transformers
- langchain-anthropic (for Claude Opus 4.6)
