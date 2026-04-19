"""
Organizational Graph Builder for European Parliament Structure

Handles EP-specific organizational hierarchy patterns:
- "22"       → Top entity (DG level)
- "22-10"    → Direct leaf unit attached to top entity
- "22A"      → Directorate (sub-department of DG)
- "22A10"    → Unit (sub-department of Directorate)
- "22A1020"  → Sub-unit (deeper nesting with 2-digit increments)
- "22B0040"  → Sub-unit under implicit parent "22B00"

Activities and percentages are aggregated per organizational unit.
"""

import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

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
class OrgGraphConfig:
    """Configuration for organizational graph building."""
    # Master root settings
    master_root_id: str = "ORG_MASTER"
    master_root_name: str = "European Parliament Organizations"
    
    # Activity nodes settings
    activities_as_nodes: bool = True  # NEW: Create activity leaf nodes
    activity_id_prefix: str = "ACT"   # Activity node ID prefix
    
    # Local storage
    local_graph_path: str = "../graph_data/organization_graph.json"
    
    # FalkorDB connection
    falkordb_url: Optional[str] = None
    falkordb_graph_name: str = "org_hierarchy"
    
    # Column mappings (can be customized per Excel file)
    col_entity_code: str = "Entity Code"
    col_entity_name: str = "Entity Name"
    col_activity: str = "Key Activities (Missions principales)"
    col_percentage: str = "%"


# =============================================================================
# Hierarchy Parser
# =============================================================================

class EPHierarchyParser:
    """
    Parser for European Parliament organizational codes.
    
    Patterns:
    - "22"       → Level 0 (DG/Top Entity)
    - "22-10"    → Level 1 (Direct unit under DG, leaf)
    - "22A"      → Level 1 (Directorate)
    - "22A10"    → Level 2 (Unit under Directorate)
    - "22A1020"  → Level 3 (Sub-unit, 2-digit increments)
    - "22B0040"  → Level 3 (Sub-unit under implicit 22B00)
    """
    
    # Regex patterns for hierarchy detection
    PATTERNS = [
        # Pattern, Level, Parent extraction function, Type
        (r'^(\d{2})$', 0, lambda m: None, 'dg'),                    # "22" - DG
        (r'^(\d{2})-(\d{2})$', 1, lambda m: m.group(1), 'direct_unit'),  # "22-10" - Direct unit
        (r'^(\d{2})([A-Z])$', 1, lambda m: m.group(1), 'directorate'),   # "22A" - Directorate
        (r'^(\d{2})([A-Z])(\d{2})$', 2, lambda m: m.group(1) + m.group(2), 'unit'),  # "22A10" - Unit
        (r'^(\d{2})([A-Z])(\d{2})(\d{2})$', 3, lambda m: m.group(1) + m.group(2) + m.group(3), 'sub_unit'),  # "22A1020"
        (r'^(\d{2})([A-Z])(\d{2})(\d{2})(\d{2})$', 4, lambda m: m.group(1) + m.group(2) + m.group(3) + m.group(4), 'sub_sub_unit'),  # "22A102030"
    ]
    
    @classmethod
    def parse(cls, code: str) -> Dict[str, Any]:
        """
        Parse an organizational code and return hierarchy info.
        
        Returns:
            {
                'code': original code,
                'level': hierarchy level (0=DG, 1=Directorate, 2=Unit, etc.),
                'parent': parent code or None,
                'type': entity type string,
                'dg_code': the DG code (first 2 digits)
            }
        """
        code = str(code).strip()
        
        for pattern, level, parent_fn, entity_type in cls.PATTERNS:
            match = re.match(pattern, code)
            if match:
                parent = parent_fn(match)
                dg_code = code[:2]
                return {
                    'code': code,
                    'level': level,
                    'parent': parent,
                    'type': entity_type,
                    'dg_code': dg_code
                }
        
        # Unknown pattern - try to infer
        return cls._infer_hierarchy(code)
    
    @classmethod
    def _infer_hierarchy(cls, code: str) -> Dict[str, Any]:
        """Infer hierarchy for non-standard codes."""
        code = str(code).strip()
        
        # Extract DG code (first 2 digits)
        dg_match = re.match(r'^(\d{2})', code)
        dg_code = dg_match.group(1) if dg_match else code[:2]
        
        # Try to find parent by removing last segment
        if '-' in code:
            # "XX-YY" format
            parent = code.split('-')[0]
            level = 1
            entity_type = 'direct_unit'
        elif len(code) > 3 and code[2].isalpha():
            # Has letter after DG code
            # Find where numbers start after letter
            letter_pos = 2
            num_start = 3
            
            if len(code) == 3:
                # "22A" - Directorate
                parent = code[:2]
                level = 1
                entity_type = 'directorate'
            elif len(code) == 5:
                # "22A10" - Unit
                parent = code[:3]
                level = 2
                entity_type = 'unit'
            elif len(code) == 7:
                # "22A1020" or "22B0040"
                parent = code[:5]
                level = 3
                entity_type = 'sub_unit'
            else:
                # Longer codes - take parent as all but last 2 digits
                parent = code[:-2]
                level = (len(code) - 3) // 2 + 1
                entity_type = 'sub_unit'
        else:
            # Unknown
            parent = None
            level = 0
            entity_type = 'unknown'
        
        return {
            'code': code,
            'level': level,
            'parent': parent,
            'type': entity_type,
            'dg_code': dg_code
        }
    
    @classmethod
    def get_all_ancestors(cls, code: str) -> List[str]:
        """Get all ancestor codes from root to immediate parent."""
        ancestors = []
        current = code
        
        while True:
            info = cls.parse(current)
            if info['parent'] is None:
                break
            ancestors.insert(0, info['parent'])
            current = info['parent']
        
        return ancestors


# =============================================================================
# Mock FalkorDB Store
# =============================================================================

class MockOrgFalkorDBStore:
    """Mock FalkorDB for testing."""
    
    def __init__(self, config: OrgGraphConfig = None):
        self.config = config or OrgGraphConfig()
        self.nodes = {}
        self.edges = []
    
    def connect(self) -> bool:
        print("✓ FalkorDB (Org): Mock mode")
        return True
    
    def clear(self):
        self.nodes = {}
        self.edges = []
    
    def create_master_root(self) -> bool:
        self.nodes[self.config.master_root_id] = {
            "name": self.config.master_root_name,
            "level": -1,
            "is_master_root": True
        }
        return True
    
    def upload_graph(self, nx_graph: nx.DiGraph, dg_roots: List[str]) -> int:
        for node_id in nx_graph.nodes():
            self.nodes[node_id] = dict(nx_graph.nodes[node_id])
        
        for from_id, to_id in nx_graph.edges():
            self.edges.append((from_id, to_id))
        
        # Connect DG roots to master
        for dg_id in dg_roots:
            self.edges.append((self.config.master_root_id, dg_id))
        
        return len(nx_graph.nodes())
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "connected": True,
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "mock": True
        }


# =============================================================================
# Real FalkorDB Store
# =============================================================================

class OrgFalkorDBStore:
    """Real FalkorDB store for organizational graph."""
    
    def __init__(self, config: OrgGraphConfig):
        self.config = config
        self.client = None
        self.graph = None
        self._connected = False
    
    def connect(self) -> bool:
        if not FALKORDB_AVAILABLE:
            print("⚠ FalkorDB not available")
            return False
        
        try:
            if self.config.falkordb_url:
                self.client = FalkorDBClient.from_url(self.config.falkordb_url)
            else:
                return False
            
            self.graph = self.client.select_graph(self.config.falkordb_graph_name)
            self._connected = True
            print(f"✓ FalkorDB connected: {self.config.falkordb_graph_name}")
            return True
        except Exception as e:
            print(f"✗ FalkorDB connection failed: {e}")
            return False
    
    def clear(self):
        if self._connected:
            try:
                self.graph.query("MATCH (n) DETACH DELETE n")
            except:
                pass
    
    def create_master_root(self) -> bool:
        if not self._connected:
            return False
        
        try:
            query = """
            MERGE (root:MasterRoot {node_id: $node_id})
            SET root.name = $name, root.level = -1, root.is_master_root = true
            """
            self.graph.query(query, {
                "node_id": self.config.master_root_id,
                "name": self.config.master_root_name
            })
            return True
        except Exception as e:
            print(f"Error creating master root: {e}")
            return False
    
    def upload_graph(self, nx_graph: nx.DiGraph, dg_roots: List[str]) -> int:
        if not self._connected:
            return 0
        
        count = 0
        
        # Create nodes
        for node_id in nx_graph.nodes():
            data = nx_graph.nodes[node_id]
            try:
                # Determine label based on level
                level = data.get('level', 0)
                if level == 0:
                    label = "DG"
                elif level == 1:
                    label = "Directorate"
                elif level == 2:
                    label = "Unit"
                else:
                    label = "SubUnit"
                
                query = f"""
                MERGE (n:{label} {{node_id: $node_id}})
                SET n.name = $name,
                    n.level = $level,
                    n.entity_type = $entity_type,
                    n.activities = $activities,
                    n.activity_weights = $weights
                """
                self.graph.query(query, {
                    "node_id": node_id,
                    "name": data.get("name", ""),
                    "level": level,
                    "entity_type": data.get("entity_type", ""),
                    "activities": data.get("activities", []),
                    "weights": data.get("activity_weights", [])
                })
                count += 1
            except Exception as e:
                print(f"Error creating node {node_id}: {e}")
        
        # Create edges
        for from_id, to_id in nx_graph.edges():
            try:
                query = """
                MATCH (a {node_id: $from_id}), (b {node_id: $to_id})
                MERGE (a)-[:HAS_CHILD]->(b)
                """
                self.graph.query(query, {"from_id": from_id, "to_id": to_id})
            except Exception as e:
                print(f"Error creating edge {from_id}->{to_id}: {e}")
        
        # Connect DG roots to master
        for dg_id in dg_roots:
            try:
                query = """
                MATCH (root:MasterRoot {node_id: $root_id}), (dg {node_id: $dg_id})
                MERGE (root)-[:HAS_ORGANIZATION]->(dg)
                """
                self.graph.query(query, {
                    "root_id": self.config.master_root_id,
                    "dg_id": dg_id
                })
            except Exception as e:
                print(f"Error connecting DG {dg_id} to root: {e}")
        
        print(f"  Uploaded {count} nodes to FalkorDB")
        return count


# =============================================================================
# Organizational Graph Builder
# =============================================================================

class OrgGraphBuilder:
    """
    Builds organizational hierarchy graph from Excel files.
    
    Handles EP-specific organizational codes and aggregates
    activities with their percentages per organizational unit.
    """
    
    def __init__(self, config: OrgGraphConfig = None, use_mock_falkordb: bool = True):
        self.config = config or OrgGraphConfig()
        self.graph = nx.DiGraph()
        self.dg_roots = []  # Top-level DG codes
        self.parser = EPHierarchyParser()
        
        # FalkorDB store
        if use_mock_falkordb or not FALKORDB_AVAILABLE or not self.config.falkordb_url:
            self.falkordb = MockOrgFalkorDBStore(self.config)
        else:
            self.falkordb = OrgFalkorDBStore(self.config)
        
        self.falkordb.connect()
    
    def build_from_excel(self, excel_path: str, 
                         col_code: str = None,
                         col_name: str = None,
                         col_activity: str = None,
                         col_percentage: str = None) -> nx.DiGraph:
        """
        Build organizational graph from Excel file.
        
        Args:
            excel_path: Path to Excel file
            col_code: Column name for entity code (auto-detect if None)
            col_name: Column name for entity name (auto-detect if None)
            col_activity: Column name for activity description (auto-detect if None)
            col_percentage: Column name for percentage (auto-detect if None)
        
        Returns:
            NetworkX DiGraph with organizational hierarchy
        """
        print("\n" + "=" * 60)
        print("BUILDING ORGANIZATIONAL GRAPH")
        print("=" * 60)
        print(f"Activities as nodes: {self.config.activities_as_nodes}")
        
        # Load Excel
        df = pd.read_excel(excel_path)
        print(f"Loaded: {excel_path}")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        
        # Auto-detect columns if not specified
        columns = df.columns.tolist()
        
        col_code = col_code or self._detect_column(columns, ['entity code', 'code', 'org code', 'unit code'])
        col_name = col_name or self._detect_column(columns, ['entity name', 'name', 'org name', 'unit name'])
        col_activity = col_activity or self._detect_column(columns, ['activity', 'activities', 'key activities', 'mission'])
        col_percentage = col_percentage or self._detect_column(columns, ['%', 'percent', 'percentage', 'weight'])
        
        print(f"\nColumn mapping:")
        print(f"  Code: {col_code}")
        print(f"  Name: {col_name}")
        print(f"  Activity: {col_activity}")
        print(f"  Percentage: {col_percentage}")
        
        # Aggregate activities per org unit
        org_data = self._aggregate_activities(df, col_code, col_name, col_activity, col_percentage)
        
        print(f"\nUnique organizations: {len(org_data)}")
        
        # Build graph
        self.graph = nx.DiGraph()
        self.dg_roots = []
        self.activity_nodes = []  # Track activity nodes
        
        # First pass: create all organizational nodes
        for code, data in org_data.items():
            hierarchy = self.parser.parse(code)
            
            self.graph.add_node(code,
                name=data['name'],
                level=hierarchy['level'],
                entity_type=hierarchy['type'],
                node_type='organization',
                dg_code=hierarchy['dg_code'],
                activities=data['activities'] if not self.config.activities_as_nodes else [],
                activity_weights=data['weights'] if not self.config.activities_as_nodes else []
            )
            
            # Track DG roots
            if hierarchy['level'] == 0:
                self.dg_roots.append(code)
        
        # Second pass: create org hierarchy edges (parent -> child)
        for code in org_data.keys():
            hierarchy = self.parser.parse(code)
            parent = hierarchy['parent']
            
            if parent:
                # Check if parent exists, if not create implicit parent
                if parent not in self.graph.nodes:
                    # Create implicit parent node
                    parent_hierarchy = self.parser.parse(parent)
                    self.graph.add_node(parent,
                        name=f"[Implicit] {parent}",
                        level=parent_hierarchy['level'],
                        entity_type=parent_hierarchy['type'],
                        node_type='organization',
                        dg_code=parent_hierarchy['dg_code'],
                        activities=[],
                        activity_weights=[],
                        is_implicit=True
                    )
                    print(f"  Created implicit parent: {parent}")
                
                self.graph.add_edge(parent, code, edge_type='HAS_CHILD')
        
        # Third pass: create activity nodes (if enabled)
        if self.config.activities_as_nodes:
            activity_count = 0
            for code, data in org_data.items():
                org_level = self.graph.nodes[code].get('level', 0)
                
                for idx, (activity, weight) in enumerate(zip(data['activities'], data['weights'])):
                    # Generate activity node ID
                    act_id = f"{code}_{self.config.activity_id_prefix}_{idx+1:03d}"
                    
                    # Create short name from activity text
                    act_name = self._create_activity_name(activity)
                    
                    self.graph.add_node(act_id,
                        name=act_name,
                        description=activity,
                        weight=weight,
                        level=org_level + 1,  # One level deeper than org
                        entity_type='activity',
                        node_type='activity',
                        parent_org=code,
                        parent_org_name=data['name'],
                        dg_code=self.parser.parse(code)['dg_code']
                    )
                    
                    # Create edge from org to activity with weight
                    self.graph.add_edge(code, act_id, 
                        edge_type='HAS_ACTIVITY',
                        weight=weight
                    )
                    
                    self.activity_nodes.append(act_id)
                    activity_count += 1
            
            print(f"  Activity nodes created: {activity_count}")
        
        # Statistics
        org_nodes = sum(1 for n in self.graph.nodes() 
                       if self.graph.nodes[n].get('node_type') == 'organization')
        act_nodes = sum(1 for n in self.graph.nodes() 
                       if self.graph.nodes[n].get('node_type') == 'activity')
        
        print(f"\nGraph built:")
        print(f"  Total nodes: {self.graph.number_of_nodes()}")
        print(f"    - Organizational: {org_nodes}")
        print(f"    - Activities: {act_nodes}")
        print(f"  Total edges: {self.graph.number_of_edges()}")
        print(f"  DG roots: {len(self.dg_roots)} - {self.dg_roots}")
        
        return self.graph
    
    def _create_activity_name(self, activity_text: str, max_words: int = 6) -> str:
        """Create short name from activity description."""
        # Take first N words
        words = activity_text.split()[:max_words]
        name = ' '.join(words)
        if len(activity_text.split()) > max_words:
            name += '...'
        return name
    
    def _detect_column(self, columns: List[str], keywords: List[str]) -> Optional[str]:
        """Detect column by keywords."""
        for col in columns:
            col_lower = col.lower()
            for kw in keywords:
                if kw in col_lower:
                    return col
        # Fallback to index
        return columns[0] if columns else None
    
    def _aggregate_activities(self, df: pd.DataFrame, 
                              col_code: str, col_name: str,
                              col_activity: str, col_percentage: str) -> Dict[str, Dict]:
        """Aggregate activities per organizational unit."""
        org_data = {}
        
        for _, row in df.iterrows():
            code = str(row[col_code]).strip()
            name = str(row[col_name]).strip() if pd.notna(row[col_name]) else ""
            activity = str(row[col_activity]).strip() if pd.notna(row[col_activity]) else ""
            
            # Handle percentage
            pct = row[col_percentage]
            if pd.isna(pct):
                pct = 0
            else:
                try:
                    pct = int(float(pct))
                except:
                    pct = 0
            
            if code not in org_data:
                org_data[code] = {
                    'name': name,
                    'activities': [],
                    'weights': []
                }
            
            if activity:
                org_data[code]['activities'].append(activity)
                org_data[code]['weights'].append(pct)
        
        return org_data
    
    def save_local(self, path: str = None) -> str:
        """Save graph to local JSON file."""
        path = path or self.config.local_graph_path
        
        graph_data = {
            "metadata": {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "dg_roots": self.dg_roots
            },
            "nodes": [],
            "edges": []
        }
        
        for node_id in self.graph.nodes():
            data = dict(self.graph.nodes[node_id])
            data["id"] = node_id
            graph_data["nodes"].append(data)
        
        for from_id, to_id in self.graph.edges():
            graph_data["edges"].append({
                "source": from_id,
                "target": to_id
            })
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Org graph saved: {path}")
        return path
    
    def load_local(self, path: str = None) -> nx.DiGraph:
        """Load graph from local JSON file."""
        path = path or self.config.local_graph_path
        
        with open(path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        self.graph = nx.DiGraph()
        self.dg_roots = graph_data.get("metadata", {}).get("dg_roots", [])
        
        for node in graph_data.get("nodes", []):
            node_id = node.pop("id")
            self.graph.add_node(node_id, **node)
        
        for edge in graph_data.get("edges", []):
            self.graph.add_edge(edge["source"], edge["target"])
        
        print(f"✓ Loaded org graph: {self.graph.number_of_nodes()} nodes")
        return self.graph
    
    def upload_to_falkordb(self) -> int:
        """Upload graph to FalkorDB."""
        print("\nUploading to FalkorDB...")
        
        self.falkordb.clear()
        self.falkordb.create_master_root()
        count = self.falkordb.upload_graph(self.graph, self.dg_roots)
        
        print(f"✓ Uploaded {count} nodes to FalkorDB")
        return count
    
    def print_tree(self, root_id: str = None, indent: int = 0, max_depth: int = 10, 
                   show_activities: bool = True):
        """Print organizational tree structure."""
        if root_id is None:
            # Print from all DG roots
            print("\n" + "=" * 60)
            print("ORGANIZATIONAL TREE")
            print("=" * 60)
            for dg_id in sorted(self.dg_roots):
                self.print_tree(dg_id, 0, max_depth, show_activities)
            return
        
        if indent > max_depth * 4:
            return
        
        data = self.graph.nodes.get(root_id, {})
        name = data.get('name', root_id)[:50]
        level = data.get('level', 0)
        node_type = data.get('node_type', 'organization')
        
        prefix = "│   " * (indent // 4)
        if indent > 0:
            prefix = prefix[:-4] + "├── "
        
        if node_type == 'activity':
            weight = data.get('weight', 0)
            print(f"{prefix}📋 [{root_id}] ({weight}%) {name}")
        else:
            # Count child activities vs org children
            children = list(self.graph.successors(root_id))
            act_children = [c for c in children if self.graph.nodes[c].get('node_type') == 'activity']
            org_children = [c for c in children if self.graph.nodes[c].get('node_type') != 'activity']
            
            if act_children:
                print(f"{prefix}[{root_id}] {name} (L{level}, {len(act_children)} activities)")
            else:
                n_activities = len(data.get('activities', []))
                print(f"{prefix}[{root_id}] {name} (L{level}, {n_activities} activities)")
            
            # Print org children first
            for child_id in sorted(org_children):
                self.print_tree(child_id, indent + 4, max_depth, show_activities)
            
            # Then print activity children (if enabled)
            if show_activities and act_children:
                for child_id in sorted(act_children):
                    self.print_tree(child_id, indent + 4, max_depth, show_activities)
            
            return  # Already handled children
        
        children = list(self.graph.successors(root_id))
        for child_id in sorted(children):
            self.print_tree(child_id, indent + 4, max_depth, show_activities)
    
    def get_leaves(self, include_activities: bool = True) -> List[str]:
        """
        Get all leaf nodes (no children).
        
        Args:
            include_activities: If True, return activity nodes as leaves.
                              If False, return org nodes that have no org children.
        """
        if include_activities:
            return [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
        else:
            # Return org nodes with no org children (may have activity children)
            leaves = []
            for n in self.graph.nodes():
                if self.graph.nodes[n].get('node_type') == 'organization':
                    children = list(self.graph.successors(n))
                    org_children = [c for c in children 
                                   if self.graph.nodes[c].get('node_type') == 'organization']
                    if not org_children:
                        leaves.append(n)
            return leaves
    
    def get_activity_nodes(self) -> List[str]:
        """Get all activity nodes."""
        return [n for n in self.graph.nodes() 
                if self.graph.nodes[n].get('node_type') == 'activity']
    
    def get_org_nodes(self) -> List[str]:
        """Get all organizational nodes (non-activity)."""
        return [n for n in self.graph.nodes() 
                if self.graph.nodes[n].get('node_type') == 'organization']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        levels = {}
        entity_types = {}
        node_types = {'organization': 0, 'activity': 0}
        total_weight = 0
        
        for node_id in self.graph.nodes():
            data = self.graph.nodes[node_id]
            level = data.get('level', 0)
            entity_type = data.get('entity_type', 'unknown')
            node_type = data.get('node_type', 'organization')
            
            levels[level] = levels.get(level, 0) + 1
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            node_types[node_type] = node_types.get(node_type, 0) + 1
            
            if node_type == 'activity':
                total_weight += data.get('weight', 0)
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "dg_roots": len(self.dg_roots),
            "org_nodes": node_types.get('organization', 0),
            "activity_nodes": node_types.get('activity', 0),
            "org_leaves": len(self.get_leaves(include_activities=False)),
            "total_activity_weight": total_weight,
            "by_level": levels,
            "by_entity_type": entity_types
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_org_graph_builder(
    local_graph_path: str = "./graph_data/organization_graph.json",
    master_root_id: str = "ORG_MASTER",
    master_root_name: str = "European Parliament Organizations",
    activities_as_nodes: bool = True,
    activity_id_prefix: str = "ACT",
    falkordb_url: str = None,
    falkordb_graph_name: str = "org_hierarchy",
    use_mock_falkordb: bool = True
) -> OrgGraphBuilder:
    """
    Create OrgGraphBuilder instance.
    
    Args:
        local_graph_path: Path for saving/loading local graph JSON
        master_root_id: ID for master root node
        master_root_name: Name for master root node
        activities_as_nodes: If True, create activity leaf nodes under org units
        activity_id_prefix: Prefix for activity node IDs (e.g., "ACT")
        falkordb_url: FalkorDB connection URL
        falkordb_graph_name: FalkorDB graph name
        use_mock_falkordb: Use mock FalkorDB (for testing)
    """
    config = OrgGraphConfig(
        master_root_id=master_root_id,
        master_root_name=master_root_name,
        activities_as_nodes=activities_as_nodes,
        activity_id_prefix=activity_id_prefix,
        local_graph_path=local_graph_path,
        falkordb_url=falkordb_url,
        falkordb_graph_name=falkordb_graph_name
    )
    return OrgGraphBuilder(config, use_mock_falkordb)


if __name__ == "__main__":
    print("OrgGraphBuilder module loaded.")
    print()
    print("Usage:")
    print("  builder = create_org_graph_builder()")
    print("  builder.build_from_excel('activities.xlsx')")
    print("  builder.save_local()")
    print("  builder.print_tree()")
