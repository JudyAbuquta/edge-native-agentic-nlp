import json
import os
from typing import List, Dict, Tuple, Optional

import networkx as nx

# Default path to the graph data JSON
DEFAULT_GRAPH_PATH = os.path.join("data", "raw", "graph_data.json")


class TrafficKnowledgeGraph:
    """
    Wrapper around a NetworkX graph for easy use by agents.

    Nodes are stored with IDs (INT_1, RD_1, HOSP_1, etc.) but can be
    accessed using their human-readable 'label' values
    (e.g., 'Intersection 4', 'Central Hospital').
    """

    def __init__(self, graph_json_path: str = DEFAULT_GRAPH_PATH) -> None:
        self.graph_json_path = graph_json_path
        self.G: nx.Graph = nx.Graph()
        self.label_to_id: Dict[str, str] = {}
        self.id_to_label: Dict[str, str] = {}
        self._load_graph()

    def _load_graph(self) -> None:
        """Load nodes and edges from JSON and build the NetworkX graph."""
        with open(self.graph_json_path, "r") as f:
            data = json.load(f)

        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Add nodes
        for node in nodes:
            node_id = node["id"]
            label = node.get("label", node_id)
            self.G.add_node(node_id, **node)

            # Build label ↔ id maps
            self.label_to_id[label] = node_id
            self.id_to_label[node_id] = label

        # Add edges
        for edge in edges:
            src = edge["from"]
            dst = edge["to"]
            rel = edge.get("relation", "CONNECTED_TO")
            self.G.add_edge(src, dst, relation=rel)

    # ----------------- Core utilities -----------------

    def get_neighbors(self, node_label: str) -> List[str]:
        """
        Return a list of neighbor node labels for a given node label.

        Example:
            get_neighbors('Intersection 4')
        """
        node_id = self.label_to_id.get(node_label)
        if node_id is None:
            return []

        neighbor_ids = list(self.G.neighbors(node_id))
        neighbor_labels = [self.id_to_label[n] for n in neighbor_ids]
        return neighbor_labels

    def find_path(self, source_label: str, target_label: str) -> List[str]:
        """
        Find a shortest path between two labeled nodes.
        Returns a list of labels or [] if no path exists.

        Example:
            find_path('Intersection 4', 'Central Hospital')
        """
        src_id = self.label_to_id.get(source_label)
        tgt_id = self.label_to_id.get(target_label)

        if src_id is None or tgt_id is None:
            return []

        try:
            path_ids = nx.shortest_path(self.G, src_id, tgt_id)
            return [self.id_to_label[n] for n in path_ids]
        except nx.NetworkXNoPath:
            return []

    def get_node_type(self, node_label: str) -> Optional[str]:
        """
        Return the 'type' of a node (intersection, road, hospital, etc.)
        or None if not found.
        """
        node_id = self.label_to_id.get(node_label)
        if node_id is None:
            return None

        data = self.G.nodes[node_id]
        return data.get("type")

    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """
        Return all node labels that have the given type.

        Example:
            get_nodes_by_type('hospital')
        """
        labels = []
        for node_id, data in self.G.nodes(data=True):
            if data.get("type") == node_type:
                labels.append(self.id_to_label.get(node_id, node_id))
        return labels

    def find_nearest_node_of_type(
        self,
        source_label: str,
        target_type: str
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Find the nearest node of a given type from a source node.

        Returns (target_label, path_labels) or None.

        Example:
            find_nearest_node_of_type('Intersection 4', 'hospital')
        """
        src_id = self.label_to_id.get(source_label)
        if src_id is None:
            return None

        # Candidates
        target_ids = [
            node_id
            for node_id, data in self.G.nodes(data=True)
            if data.get("type") == target_type
        ]

        if not target_ids:
            return None

        best_target = None
        best_path_ids = None

        for tgt_id in target_ids:
            try:
                path_ids = nx.shortest_path(self.G, src_id, tgt_id)
                if best_path_ids is None or len(path_ids) < len(best_path_ids):
                    best_path_ids = path_ids
                    best_target = tgt_id
            except nx.NetworkXNoPath:
                continue

        if best_target is None or best_path_ids is None:
            return None

        target_label = self.id_to_label[best_target]
        path_labels = [self.id_to_label[n] for n in best_path_ids]
        return target_label, path_labels


# ----------------- Simple module-level helpers -----------------


def load_graph(path: str = DEFAULT_GRAPH_PATH) -> TrafficKnowledgeGraph:
    """Convenience function to load the knowledge graph."""
    return TrafficKnowledgeGraph(graph_json_path=path)


def get_neighbors(node_label: str) -> List[str]:
    """Quick helper that loads the graph and returns neighbors."""
    kg = load_graph()
    return kg.get_neighbors(node_label)


def find_path(source_label: str, target_label: str) -> List[str]:
    """Quick helper that loads the graph and returns a path."""
    kg = load_graph()
    return kg.find_path(source_label, target_label)
