"""
Region Graph Construction for RAGAF-Diffusion

This module constructs a graph from extracted sketch regions where:
- Nodes represent sketch regions
- Edges represent spatial relationships (adjacency, overlap, proximity)

The graph is used in RAGAF attention to model region-text interactions.

Author: RAGAF-Diffusion Research Team
"""

import numpy as np
import torch
import networkx as nx
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cdist

from data.region_extraction import SketchRegion


@dataclass
class RegionGraph:
    """
    Graph representation of sketch regions.
    
    Attributes:
        num_nodes: Number of nodes (regions)
        node_features: Node feature matrix (N, D)
        edge_index: Edge connectivity (2, E)
        edge_weights: Edge weights (E,)
        region_masks: List of region masks for each node
        adjacency_matrix: Adjacency matrix (N, N)
    """
    num_nodes: int
    node_features: torch.Tensor  # (N, D)
    edge_index: torch.Tensor  # (2, E)
    edge_weights: torch.Tensor  # (E,)
    region_masks: List[np.ndarray]
    adjacency_matrix: Optional[torch.Tensor] = None


class RegionGraphBuilder:
    """
    Builds a graph from sketch regions for RAGAF attention.
    
    Graph construction strategies:
    1. Spatial adjacency: Connect regions that touch or overlap
    2. K-nearest neighbors: Connect K closest regions by centroid distance
    3. Radius-based: Connect regions within a distance threshold
    """
    
    def __init__(
        self,
        graph_type: str = "hybrid",  # "adjacency", "knn", "radius", "hybrid"
        knn_k: int = 5,
        radius_threshold: float = 50.0,
        feature_type: str = "spatial",  # "spatial", "geometric", "combined"
        normalize_features: bool = True,
        add_self_loops: bool = True,
        image_size: Tuple[int, int] = (512, 512)
    ):
        """
        Initialize graph builder.
        
        Args:
            graph_type: Type of graph construction
            knn_k: Number of nearest neighbors (for knn mode)
            radius_threshold: Distance threshold (for radius mode)
            feature_type: Type of node features to compute
            normalize_features: Whether to normalize node features
            add_self_loops: Whether to add self-loops to graph
            image_size: Image size (H, W) for normalization
        """
        self.graph_type = graph_type
        self.knn_k = knn_k
        self.radius_threshold = radius_threshold
        self.feature_type = feature_type
        self.normalize_features = normalize_features
        self.add_self_loops = add_self_loops
        self.image_size = image_size
        
    def build_graph(self, regions: List[SketchRegion]) -> RegionGraph:
        """
        Build graph from list of regions.
        
        Args:
            regions: List of SketchRegion objects
        
        Returns:
            RegionGraph object
        """
        if len(regions) == 0:
            # Return empty graph
            return self._create_empty_graph()
        
        num_nodes = len(regions)
        
        # Compute node features
        node_features = self._compute_node_features(regions)
        
        # Build edges based on graph type
        if self.graph_type == "adjacency":
            edge_index, edge_weights = self._build_adjacency_edges(regions)
        elif self.graph_type == "knn":
            edge_index, edge_weights = self._build_knn_edges(regions)
        elif self.graph_type == "radius":
            edge_index, edge_weights = self._build_radius_edges(regions)
        elif self.graph_type == "hybrid":
            edge_index, edge_weights = self._build_hybrid_edges(regions)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        
        # Add self-loops if requested
        if self.add_self_loops:
            edge_index, edge_weights = self._add_self_loops(
                edge_index, edge_weights, num_nodes
            )
        
        # Compute adjacency matrix
        adjacency_matrix = self._compute_adjacency_matrix(
            edge_index, edge_weights, num_nodes
        )
        
        # Collect region masks
        region_masks = [r.mask for r in regions]
        
        return RegionGraph(
            num_nodes=num_nodes,
            node_features=node_features,
            edge_index=edge_index,
            edge_weights=edge_weights,
            region_masks=region_masks,
            adjacency_matrix=adjacency_matrix
        )
    
    def _compute_node_features(self, regions: List[SketchRegion]) -> torch.Tensor:
        """
        Compute node features for each region.
        
        Features can include:
        - Spatial: normalized centroid (cx, cy), normalized bbox (x, y, w, h)
        - Geometric: area, aspect ratio, compactness
        - Combined: concatenation of above
        
        Args:
            regions: List of regions
        
        Returns:
            Node feature matrix (N, D)
        """
        features_list = []
        
        H, W = self.image_size
        
        for region in regions:
            features = []
            
            # Spatial features
            if self.feature_type in ["spatial", "combined"]:
                cx, cy = region.centroid
                x, y, w, h = region.bbox
                
                # Normalize to [0, 1]
                spatial_feats = [
                    cx / W, cy / H,  # Normalized centroid
                    x / W, y / H,    # Normalized bbox top-left
                    w / W, h / H     # Normalized bbox size
                ]
                features.extend(spatial_feats)
            
            # Geometric features
            if self.feature_type in ["geometric", "combined"]:
                area = region.area
                _, _, w, h = region.bbox
                
                # Normalize area
                normalized_area = area / (H * W)
                
                # Aspect ratio
                aspect_ratio = w / max(h, 1e-6)
                
                # Compactness (circularity measure)
                perimeter = len(region.contour) if region.contour is not None else 0
                compactness = (4 * np.pi * area) / max(perimeter ** 2, 1e-6)
                compactness = min(compactness, 1.0)  # Clamp to [0, 1]
                
                geometric_feats = [
                    normalized_area,
                    aspect_ratio,
                    compactness
                ]
                features.extend(geometric_feats)
            
            features_list.append(features)
        
        # Convert to tensor
        node_features = torch.tensor(features_list, dtype=torch.float32)
        
        # Normalize if requested
        if self.normalize_features and node_features.shape[0] > 1:
            mean = node_features.mean(dim=0, keepdim=True)
            std = node_features.std(dim=0, keepdim=True) + 1e-6
            node_features = (node_features - mean) / std
        
        return node_features
    
    def _build_adjacency_edges(
        self, regions: List[SketchRegion]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges based on spatial adjacency (overlap or touching).
        
        Args:
            regions: List of regions
        
        Returns:
            edge_index (2, E), edge_weights (E,)
        """
        edges = []
        weights = []
        
        num_regions = len(regions)
        
        for i in range(num_regions):
            for j in range(i + 1, num_regions):
                # Check if regions overlap or are adjacent
                mask_i = regions[i].mask
                mask_j = regions[j].mask
                
                # Compute intersection
                intersection = np.logical_and(mask_i, mask_j)
                overlap_area = np.sum(intersection)
                
                # Check if bboxes are close (within threshold)
                bbox_i = regions[i].bbox
                bbox_j = regions[j].bbox
                
                bbox_distance = self._bbox_distance(bbox_i, bbox_j)
                
                # Connect if overlap or close proximity
                if overlap_area > 0 or bbox_distance < 10:
                    # Undirected edge (add both directions)
                    edges.append([i, j])
                    edges.append([j, i])
                    
                    # Weight based on overlap or proximity
                    if overlap_area > 0:
                        weight = 1.0
                    else:
                        weight = 1.0 / (1.0 + bbox_distance)
                    
                    weights.append(weight)
                    weights.append(weight)
        
        if len(edges) == 0:
            # No edges found, create empty edge_index
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weights = torch.zeros(0, dtype=torch.float32)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).T
            edge_weights = torch.tensor(weights, dtype=torch.float32)
        
        return edge_index, edge_weights
    
    def _build_knn_edges(
        self, regions: List[SketchRegion]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges based on k-nearest neighbors by centroid distance.
        
        Args:
            regions: List of regions
        
        Returns:
            edge_index (2, E), edge_weights (E,)
        """
        num_regions = len(regions)
        
        if num_regions <= 1:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32)
        
        # Extract centroids
        centroids = np.array([r.centroid for r in regions])  # (N, 2)
        
        # Compute pairwise distances
        distances = cdist(centroids, centroids)  # (N, N)
        
        edges = []
        weights = []
        
        k = min(self.knn_k, num_regions - 1)
        
        for i in range(num_regions):
            # Get k nearest neighbors (excluding self)
            nearest_indices = np.argsort(distances[i])[1:k+1]
            
            for j in nearest_indices:
                edges.append([i, j])
                # Weight inversely proportional to distance
                dist = distances[i, j]
                weight = 1.0 / (1.0 + dist)
                weights.append(weight)
        
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_weights = torch.tensor(weights, dtype=torch.float32)
        
        return edge_index, edge_weights
    
    def _build_radius_edges(
        self, regions: List[SketchRegion]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges based on distance threshold.
        
        Args:
            regions: List of regions
        
        Returns:
            edge_index (2, E), edge_weights (E,)
        """
        num_regions = len(regions)
        
        if num_regions <= 1:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32)
        
        centroids = np.array([r.centroid for r in regions])
        distances = cdist(centroids, centroids)
        
        edges = []
        weights = []
        
        for i in range(num_regions):
            for j in range(num_regions):
                if i != j and distances[i, j] < self.radius_threshold:
                    edges.append([i, j])
                    weight = 1.0 / (1.0 + distances[i, j])
                    weights.append(weight)
        
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weights = torch.zeros(0, dtype=torch.float32)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).T
            edge_weights = torch.tensor(weights, dtype=torch.float32)
        
        return edge_index, edge_weights
    
    def _build_hybrid_edges(
        self, regions: List[SketchRegion]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges using hybrid approach (adjacency + knn).
        
        Args:
            regions: List of regions
        
        Returns:
            edge_index (2, E), edge_weights (E,)
        """
        # Get edges from both methods
        adj_edges, adj_weights = self._build_adjacency_edges(regions)
        knn_edges, knn_weights = self._build_knn_edges(regions)
        
        # Combine edges (remove duplicates)
        if adj_edges.shape[1] > 0 and knn_edges.shape[1] > 0:
            all_edges = torch.cat([adj_edges, knn_edges], dim=1)
            all_weights = torch.cat([adj_weights, knn_weights], dim=0)
            
            # Remove duplicate edges
            edge_set = {}
            for i in range(all_edges.shape[1]):
                edge = (all_edges[0, i].item(), all_edges[1, i].item())
                weight = all_weights[i].item()
                
                if edge not in edge_set:
                    edge_set[edge] = weight
                else:
                    # Keep max weight
                    edge_set[edge] = max(edge_set[edge], weight)
            
            edges = list(edge_set.keys())
            weights = list(edge_set.values())
            
            edge_index = torch.tensor(edges, dtype=torch.long).T
            edge_weights = torch.tensor(weights, dtype=torch.float32)
        elif adj_edges.shape[1] > 0:
            edge_index, edge_weights = adj_edges, adj_weights
        elif knn_edges.shape[1] > 0:
            edge_index, edge_weights = knn_edges, knn_weights
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weights = torch.zeros(0, dtype=torch.float32)
        
        return edge_index, edge_weights
    
    def _add_self_loops(
        self, 
        edge_index: torch.Tensor, 
        edge_weights: torch.Tensor, 
        num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add self-loops to graph."""
        self_loop_edges = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
        self_loop_weights = torch.ones(num_nodes, dtype=torch.float32)
        
        edge_index = torch.cat([edge_index, self_loop_edges], dim=1)
        edge_weights = torch.cat([edge_weights, self_loop_weights], dim=0)
        
        return edge_index, edge_weights
    
    def _compute_adjacency_matrix(
        self, 
        edge_index: torch.Tensor, 
        edge_weights: torch.Tensor, 
        num_nodes: int
    ) -> torch.Tensor:
        """Compute dense adjacency matrix."""
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        
        if edge_index.shape[1] > 0:
            adj[edge_index[0], edge_index[1]] = edge_weights
        
        return adj
    
    def _bbox_distance(
        self, 
        bbox1: Tuple[int, int, int, int], 
        bbox2: Tuple[int, int, int, int]
    ) -> float:
        """
        Compute minimum distance between two bounding boxes.
        
        Args:
            bbox1: (x, y, w, h)
            bbox2: (x, y, w, h)
        
        Returns:
            Distance (0 if overlapping)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Get bbox corners
        left1, right1 = x1, x1 + w1
        top1, bottom1 = y1, y1 + h1
        
        left2, right2 = x2, x2 + w2
        top2, bottom2 = y2, y2 + h2
        
        # Check for overlap
        if right1 < left2:
            dx = left2 - right1
        elif right2 < left1:
            dx = left1 - right2
        else:
            dx = 0
        
        if bottom1 < top2:
            dy = top2 - bottom1
        elif bottom2 < top1:
            dy = top1 - bottom2
        else:
            dy = 0
        
        return np.sqrt(dx**2 + dy**2)
    
    def _create_empty_graph(self) -> RegionGraph:
        """Create empty graph for edge case."""
        return RegionGraph(
            num_nodes=0,
            node_features=torch.zeros((0, 6), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_weights=torch.zeros(0, dtype=torch.float32),
            region_masks=[],
            adjacency_matrix=torch.zeros((0, 0), dtype=torch.float32)
        )


if __name__ == "__main__":
    # Example usage
    print("Region Graph Construction Module for RAGAF-Diffusion")
    print("=" * 60)
    
    from data.region_extraction import RegionExtractor
    
    # Create synthetic sketch
    sketch = np.ones((256, 256), dtype=np.uint8) * 255
    cv2.rectangle(sketch, (50, 50), (100, 100), 0, -1)
    cv2.circle(sketch, (180, 80), 30, 0, -1)
    cv2.ellipse(sketch, (150, 180), (40, 20), 0, 0, 360, 0, -1)
    
    # Extract regions
    extractor = RegionExtractor(min_region_area=200)
    regions = extractor.extract_regions(sketch)
    
    # Build graph
    graph_builder = RegionGraphBuilder(
        graph_type="hybrid",
        image_size=(256, 256)
    )
    graph = graph_builder.build_graph(regions)
    
    print(f"Graph constructed:")
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.edge_index.shape[1]}")
    print(f"  Node features shape: {graph.node_features.shape}")
    print(f"  Adjacency matrix shape: {graph.adjacency_matrix.shape}")
    print(f"\nEdge index:\n{graph.edge_index}")
    print(f"\nEdge weights:\n{graph.edge_weights}")
