"""
Main Orchestration Module
Ties all components together with configurable pipeline.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

from vector_db import VectorDB, create_vector_db
from ingestion import ExcelIngestionPlugin, create_ingestion_plugin
from graph_builder import GraphBuilder, create_graph_builder
from llm import LLM, create_llm
from agent import RefinementAgent, create_agent


# =============================================================================
# Pipeline Configuration
# =============================================================================

class PipelineConfig:
    """Configuration for the pipeline."""
    
    def __init__(self):
        # Components to run
        self.run_ingestion = True
        self.run_graph = True
        self.run_agent = True
        self.run_export = True
        
        # Paths
        self.data_directory = "./data"
        self.excel_file = None  # Auto-detect if None
        self.output_directory = "./output"
        self.chroma_directory = "./chroma_db"
        
        # LLM
        self.use_mock_llm = True
        self.anthropic_api_key = None
        
        # Agent
        self.refinement_strategy = "top_down"  # or "bottom_up"
        
        # Verbose
        self.verbose = True


# =============================================================================
# Pipeline
# =============================================================================

class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Components (initialized on demand)
        self.vector_db: Optional[VectorDB] = None
        self.ingestion: Optional[ExcelIngestionPlugin] = None
        self.graph: Optional[GraphBuilder] = None
        self.llm: Optional[LLM] = None
        self.agent: Optional[RefinementAgent] = None
        
        # Results
        self.results = {}
    
    def _log(self, msg: str):
        if self.config.verbose:
            print(msg)
    
    def _find_excel(self) -> Optional[str]:
        """Find first Excel file in data directory."""
        data_dir = Path(self.config.data_directory)
        if not data_dir.exists():
            return None
        for pattern in ["*.xlsx", "*.xls"]:
            files = list(data_dir.glob(pattern))
            if files:
                return str(files[0])
        return None
    
    # =========================================================================
    # Step 1: Ingestion
    # =========================================================================
    
    def step_ingestion(self) -> Dict[str, Any]:
        """Ingest Excel files into ChromaDB."""
        self._log("\n" + "=" * 60)
        self._log("STEP 1: INGESTION")
        self._log("=" * 60)
        
        # Create vector DB
        self.vector_db = create_vector_db(
            collection_name="org_documents",
            persist_directory=self.config.chroma_directory
        )
        
        # Create ingestion plugin
        self.ingestion = create_ingestion_plugin(vector_db=self.vector_db)
        
        # Run ingestion
        stats = self.ingestion.ingest(self.config.data_directory)
        self.results["ingestion"] = stats
        
        return stats
    
    # =========================================================================
    # Step 2: Graph Building
    # =========================================================================
    
    def step_graph(self) -> Dict[str, Any]:
        """Build organizational graph."""
        self._log("\n" + "=" * 60)
        self._log("STEP 2: GRAPH BUILDING")
        self._log("=" * 60)
        
        # Find Excel file
        excel_file = self.config.excel_file or self._find_excel()
        if not excel_file:
            raise FileNotFoundError(f"No Excel file in {self.config.data_directory}")
        
        # Build graph
        self.graph = create_graph_builder()
        self.graph.load_excel(excel_file)
        
        self._log("\nHierarchy:")
        self.graph.print_tree()
        
        stats = {
            "nodes": self.graph.graph.number_of_nodes(),
            "edges": self.graph.graph.number_of_edges(),
            "roots": len(self.graph.roots)
        }
        self.results["graph"] = stats
        
        return stats
    
    # =========================================================================
    # Step 3: Agent Refinement
    # =========================================================================
    
    def step_agent(self) -> Dict[str, Any]:
        """Run refinement agent."""
        self._log("\n" + "=" * 60)
        self._log("STEP 3: AGENT REFINEMENT")
        self._log("=" * 60)
        
        if self.graph is None:
            raise ValueError("Graph not built. Run step_graph() first.")
        
        # Create LLM
        self.llm = create_llm(
            api_key=self.config.anthropic_api_key,
            use_mock=self.config.use_mock_llm
        )
        
        # Create agent
        self.agent = create_agent(self.graph, self.llm)
        
        # Run refinement
        result = self.agent.run(
            strategy=self.config.refinement_strategy,
            verbose=self.config.verbose
        )
        self.results["agent"] = result
        
        return result
    
    # =========================================================================
    # Step 4: Export
    # =========================================================================
    
    def step_export(self) -> Dict[str, str]:
        """Export results to files."""
        self._log("\n" + "=" * 60)
        self._log("STEP 4: EXPORT")
        self._log("=" * 60)
        
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        files = {}
        
        # Export graph
        if self.graph:
            graph_file = output_dir / "graph.json"
            with open(graph_file, 'w') as f:
                json.dump(self.graph.export(), f, indent=2, ensure_ascii=False)
            files["graph"] = str(graph_file)
            self._log(f"  ✓ {graph_file}")
        
        # Export refinements
        if self.agent and self.agent.results:
            refine_file = output_dir / "refinements.json"
            with open(refine_file, 'w') as f:
                json.dump(self.agent.results, f, indent=2, ensure_ascii=False)
            files["refinements"] = str(refine_file)
            self._log(f"  ✓ {refine_file}")
        
        # Export summary
        summary_file = output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        files["summary"] = str(summary_file)
        self._log(f"  ✓ {summary_file}")
        
        self.results["export"] = files
        return files
    
    # =========================================================================
    # Run Pipeline
    # =========================================================================
    
    def run(self) -> Dict[str, Any]:
        """Run the full pipeline."""
        self._log("\n" + "=" * 60)
        self._log("ORGANIZATIONAL AI PIPELINE")
        self._log("=" * 60)
        
        if self.config.run_ingestion:
            self.step_ingestion()
        
        if self.config.run_graph:
            self.step_graph()
        
        if self.config.run_agent:
            self.step_agent()
        
        if self.config.run_export:
            self.step_export()
        
        self._log("\n" + "=" * 60)
        self._log("PIPELINE COMPLETE")
        self._log("=" * 60)
        
        return self.results
    
    # =========================================================================
    # Search
    # =========================================================================
    
    def search(self, query: str, k: int = 5):
        """Search vectorized documents."""
        if self.vector_db is None:
            self.vector_db = create_vector_db(
                persist_directory=self.config.chroma_directory
            )
        return self.vector_db.search(query, k=k)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Organizational AI Pipeline")
    parser.add_argument("--data", "-d", default="./data", help="Data directory")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--strategy", "-s", default="top_down", choices=["top_down", "bottom_up"])
    parser.add_argument("--api-key", help="Anthropic API key")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-agent", action="store_true")
    
    args = parser.parse_args()
    
    config = PipelineConfig()
    config.data_directory = args.data
    config.output_directory = args.output
    config.refinement_strategy = args.strategy
    config.anthropic_api_key = args.api_key
    config.use_mock_llm = args.mock or not args.api_key
    config.run_ingestion = not args.skip_ingest
    config.run_agent = not args.skip_agent
    
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
