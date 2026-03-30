"""
Component 2: Excel Ingestion
Reads Excel files from local directory and vectorizes in ChromaDB.
"""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}

from vector_db import VectorDB, create_vector_db


# =============================================================================
# Ingestion Plugin
# =============================================================================

class ExcelIngestionPlugin:
    """Ingests Excel files from local directory into ChromaDB."""
    
    def __init__(self, vector_db: VectorDB = None):
        self.vector_db = vector_db
        self.stats = {"files": 0, "documents": 0}
    
    def _ensure_db(self):
        if self.vector_db is None:
            self.vector_db = create_vector_db()
    
    def _doc_id(self, file: str, sheet: str, code: str) -> str:
        return hashlib.md5(f"{file}:{sheet}:{code}".encode()).hexdigest()[:12]
    
    def _infer_level(self, code: str) -> int:
        """Infer org level from code pattern."""
        import re
        code = str(code).strip()
        if re.match(r'^\d+$', code): return 0
        if re.match(r'^\d+-\d+$', code): return 1
        if re.match(r'^\d+[A-Z]$', code): return 1
        if re.match(r'^\d+[A-Z]\d+$', code): return 2
        return 0
    
    def _infer_parent(self, code: str) -> Optional[str]:
        """Infer parent code from code pattern."""
        import re
        code = str(code).strip()
        if re.match(r'^\d+$', code): return None
        if re.match(r'^\d+-\d+$', code): return code.split('-')[0]
        if re.match(r'^\d+[A-Z]$', code): return re.match(r'^(\d+)', code).group(1)
        if re.match(r'^\d+[A-Z]\d+$', code):
            m = re.match(r'^(\d+[A-Z])', code)
            return m.group(1) if m else None
        return None
    
    def _process_file(self, file_path: Path) -> List[Document]:
        """Process single Excel file into documents."""
        documents = []
        
        try:
            excel = pd.ExcelFile(file_path)
            
            for sheet in excel.sheet_names:
                df = pd.read_excel(excel, sheet_name=sheet)
                if df.empty:
                    continue
                
                # Find columns (flexible matching)
                cols = list(df.columns)
                code_col = next((c for c in cols if 'code' in c.lower()), cols[0] if cols else None)
                name_col = next((c for c in cols if 'name' in c.lower() or 'entity' in c.lower()), cols[1] if len(cols) > 1 else None)
                activity_col = next((c for c in cols if 'activ' in c.lower() or 'mission' in c.lower()), cols[2] if len(cols) > 2 else None)
                weight_col = next((c for c in cols if c == '%' or 'weight' in c.lower()), None)
                
                if not code_col:
                    continue
                
                # Group by entity code
                entities: Dict[str, Dict] = {}
                
                for _, row in df.iterrows():
                    code = str(row.get(code_col, "")).strip()
                    if not code or code == "nan":
                        continue
                    
                    if code not in entities:
                        entities[code] = {
                            "name": str(row.get(name_col, "")) if name_col else code,
                            "activities": [],
                            "weights": []
                        }
                    
                    if activity_col:
                        activity = str(row.get(activity_col, "")).strip()
                        weight = int(row.get(weight_col, 0)) if weight_col and pd.notna(row.get(weight_col)) else 0
                        if activity and activity != "nan":
                            entities[code]["activities"].append(activity)
                            entities[code]["weights"].append(weight)
                
                # Create documents
                for code, data in entities.items():
                    level = self._infer_level(code)
                    parent = self._infer_parent(code)
                    
                    activities_text = "\n".join([
                        f"- ({w}%) {a}" for a, w in zip(data["activities"], data["weights"])
                    ]) or "No activities listed"
                    
                    content = f"""Organizational Unit: {data['name']}
Code: {code}
Level: {level}
Parent: {parent or 'Root'}

Activities:
{activities_text}"""
                    
                    doc = Document(
                        page_content=content,
                        metadata={
                            "entity_code": code,
                            "entity_name": data["name"],
                            "level": level,
                            "parent_code": parent or "",
                            "file": file_path.name,
                            "sheet": sheet,
                            "activity_count": len(data["activities"])
                        }
                    )
                    documents.append(doc)
        
        except Exception as e:
            print(f"  Error: {e}")
        
        return documents
    
    def ingest(self, directory: str, patterns: List[str] = None) -> Dict[str, Any]:
        """
        Ingest all Excel files from directory.
        
        Args:
            directory: Path to directory
            patterns: File patterns (default: ["*.xlsx", "*.xls"])
        """
        self._ensure_db()
        patterns = patterns or ["*.xlsx", "*.xls"]
        
        print("\n" + "=" * 50)
        print("EXCEL INGESTION")
        print("=" * 50)
        print(f"Directory: {directory}")
        
        # Find files
        dir_path = Path(directory)
        files = []
        for pattern in patterns:
            files.extend(dir_path.glob(pattern))
        files = sorted(set(files))
        
        print(f"Files found: {len(files)}\n")
        
        # Process each file
        all_docs = []
        for file_path in files:
            print(f"Processing: {file_path.name}")
            docs = self._process_file(file_path)
            all_docs.extend(docs)
            print(f"  → {len(docs)} documents")
            self.stats["files"] += 1
        
        # Add to vector DB
        if all_docs:
            ids = [self._doc_id(d.metadata["file"], d.metadata["sheet"], d.metadata["entity_code"]) for d in all_docs]
            self.vector_db.add_documents(all_docs, ids=ids)
            self.stats["documents"] = len(all_docs)
        
        print("\n" + "=" * 50)
        print(f"Total: {self.stats['files']} files, {self.stats['documents']} documents")
        print(f"ChromaDB count: {self.vector_db.count()}")
        print("=" * 50)
        
        return self.stats
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search ingested documents."""
        self._ensure_db()
        return self.vector_db.search(query, k=k)


# =============================================================================
# Factory
# =============================================================================

def create_ingestion_plugin(vector_db: VectorDB = None) -> ExcelIngestionPlugin:
    """Create ingestion plugin."""
    return ExcelIngestionPlugin(vector_db=vector_db)


if __name__ == "__main__":
    plugin = create_ingestion_plugin()
    plugin.ingest("/mnt/user-data/uploads")
    
    print("\nSearch test:")
    results = plugin.search("budget", k=3)
    for doc in results:
        print(f"  [{doc.metadata['entity_code']}] {doc.page_content[:50]}...")
