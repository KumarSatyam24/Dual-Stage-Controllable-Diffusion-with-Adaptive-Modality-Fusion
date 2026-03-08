"""
Region Extraction from Sketches for RAGAF-Diffusion

This module extracts meaningful regions from sketch images using:
- Connected component analysis
- Contour detection
- Region merging and filtering

Each region becomes a node in the region graph used for RAGAF attention.

Author: RAGAF-Diffusion Research Team
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import torch
from dataclasses import dataclass
from scipy import ndimage


@dataclass
class SketchRegion:
    """
    Represents a single region extracted from a sketch.
    
    Attributes:
        region_id: Unique identifier for this region
        mask: Binary mask for this region (H, W)
        bbox: Bounding box (x, y, w, h)
        centroid: Region centroid (cx, cy)
        area: Region area in pixels
        contour: Region contour points
    """
    region_id: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[float, float]  # (cx, cy)
    area: int
    contour: Optional[np.ndarray] = None


class RegionExtractor:
    """
    Extract regions from sketch images for graph construction.
    
    Regions are extracted using connected components and filtered by size.
    Small regions are merged or removed to avoid noise.
    """
    
    def __init__(
        self,
        min_region_area: int = 100,
        max_num_regions: int = 50,
        merge_nearby_regions: bool = True,
        merge_distance_threshold: float = 20.0,
        use_morphology: bool = True,
        morphology_kernel_size: int = 3
    ):
        """
        Initialize region extractor.
        
        Args:
            min_region_area: Minimum area (pixels) for a valid region
            max_num_regions: Maximum number of regions to extract
            merge_nearby_regions: Whether to merge spatially close regions
            merge_distance_threshold: Distance threshold for region merging
            use_morphology: Apply morphological operations to clean sketch
            morphology_kernel_size: Kernel size for morphological operations
        """
        self.min_region_area = min_region_area
        self.max_num_regions = max_num_regions
        self.merge_nearby_regions = merge_nearby_regions
        self.merge_distance_threshold = merge_distance_threshold
        self.use_morphology = use_morphology
        self.morphology_kernel_size = morphology_kernel_size
        
    def extract_regions(self, sketch: np.ndarray) -> List[SketchRegion]:
        """
        Extract regions from sketch image.
        
        Args:
            sketch: Sketch image (H, W) in range [0, 255]
                   Assumes white background, black strokes
        
        Returns:
            List of SketchRegion objects
        """
        # Preprocess sketch
        binary_sketch = self._preprocess_sketch(sketch)
        
        # Find connected components
        regions = self._find_connected_components(binary_sketch)
        
        # Filter by area
        regions = self._filter_by_area(regions)
        
        # Optionally merge nearby regions
        if self.merge_nearby_regions:
            regions = self._merge_nearby_regions(regions)
        
        # Limit number of regions
        if len(regions) > self.max_num_regions:
            # Sort by area (descending) and keep top-k
            regions = sorted(regions, key=lambda r: r.area, reverse=True)
            regions = regions[:self.max_num_regions]
            # Re-assign region IDs
            for i, region in enumerate(regions):
                region.region_id = i
        
        return regions
    
    def _preprocess_sketch(self, sketch: np.ndarray) -> np.ndarray:
        """
        Preprocess sketch for region extraction.
        
        Args:
            sketch: Sketch image (H, W)
        
        Returns:
            Binary sketch (H, W) with dtype bool
        """
        # Binarize (assume black strokes on white background)
        # Invert so strokes are 1, background is 0
        binary = sketch < 128
        
        if self.use_morphology:
            # Apply closing to connect nearby strokes
            kernel = np.ones(
                (self.morphology_kernel_size, self.morphology_kernel_size), 
                np.uint8
            )
            binary = cv2.morphologyEx(
                binary.astype(np.uint8), 
                cv2.MORPH_CLOSE, 
                kernel
            ).astype(bool)
        
        return binary
    
    def _find_connected_components(self, binary_sketch: np.ndarray) -> List[SketchRegion]:
        """
        Find connected components in binary sketch.
        
        Args:
            binary_sketch: Binary sketch (H, W)
        
        Returns:
            List of SketchRegion objects
        """
        # Label connected components
        labeled, num_labels = ndimage.label(binary_sketch)
        
        regions = []
        
        # Extract each component
        for region_id in range(1, num_labels + 1):  # Skip background (0)
            # Create mask for this region
            mask = (labeled == region_id)
            
            # Compute properties
            area = np.sum(mask)
            
            # Skip very small regions
            if area < 10:  # Absolute minimum
                continue
            
            # Compute bounding box
            rows, cols = np.where(mask)
            if len(rows) == 0:
                continue
            
            x_min, x_max = cols.min(), cols.max()
            y_min, y_max = rows.min(), rows.max()
            bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
            
            # Compute centroid
            centroid = (cols.mean(), rows.mean())
            
            # Find contour
            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            contour = contours[0] if len(contours) > 0 else None
            
            # Create region object
            region = SketchRegion(
                region_id=region_id - 1,  # 0-indexed
                mask=mask,
                bbox=bbox,
                centroid=centroid,
                area=area,
                contour=contour
            )
            
            regions.append(region)
        
        return regions
    
    def _filter_by_area(self, regions: List[SketchRegion]) -> List[SketchRegion]:
        """
        Filter regions by minimum area threshold.
        
        Args:
            regions: List of regions
        
        Returns:
            Filtered regions
        """
        filtered = [r for r in regions if r.area >= self.min_region_area]
        
        # Re-assign region IDs
        for i, region in enumerate(filtered):
            region.region_id = i
        
        return filtered
    
    def _merge_nearby_regions(self, regions: List[SketchRegion]) -> List[SketchRegion]:
        """
        Merge regions that are spatially close.
        
        Args:
            regions: List of regions
        
        Returns:
            Merged regions
        """
        if len(regions) <= 1:
            return regions
        
        # Compute pairwise centroid distances
        centroids = np.array([r.centroid for r in regions])
        
        # Simple greedy merging based on distance
        merged = []
        merged_flags = [False] * len(regions)
        
        for i, region_i in enumerate(regions):
            if merged_flags[i]:
                continue
            
            # Start a new merged region
            current_mask = region_i.mask.copy()
            merged_ids = [i]
            merged_flags[i] = True
            
            # Find nearby regions to merge
            for j, region_j in enumerate(regions):
                if i == j or merged_flags[j]:
                    continue
                
                dist = np.linalg.norm(
                    np.array(region_i.centroid) - np.array(region_j.centroid)
                )
                
                if dist < self.merge_distance_threshold:
                    current_mask = np.logical_or(current_mask, region_j.mask)
                    merged_ids.append(j)
                    merged_flags[j] = True
            
            # Compute properties of merged region
            area = np.sum(current_mask)
            rows, cols = np.where(current_mask)
            
            if len(rows) == 0:
                continue
            
            x_min, x_max = cols.min(), cols.max()
            y_min, y_max = rows.min(), rows.max()
            bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
            centroid = (cols.mean(), rows.mean())
            
            # Find contour
            mask_uint8 = current_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_uint8, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            contour = contours[0] if len(contours) > 0 else None
            
            merged_region = SketchRegion(
                region_id=len(merged),
                mask=current_mask,
                bbox=bbox,
                centroid=centroid,
                area=area,
                contour=contour
            )
            
            merged.append(merged_region)
        
        return merged
    
    def visualize_regions(
        self, 
        sketch: np.ndarray, 
        regions: List[SketchRegion],
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Visualize extracted regions on sketch.
        
        Args:
            sketch: Original sketch (H, W)
            regions: List of regions
            show_labels: Whether to show region ID labels
        
        Returns:
            Visualization image (H, W, 3)
        """
        # Create RGB visualization
        vis = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        
        # Generate random colors for each region
        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(regions), 3), dtype=np.uint8)
        
        # Draw each region
        for i, region in enumerate(regions):
            # Create colored mask
            color_mask = np.zeros_like(vis)
            color_mask[region.mask] = colors[i]
            
            # Blend with original
            vis = cv2.addWeighted(vis, 0.7, color_mask, 0.3, 0)
            
            # Draw bounding box
            x, y, w, h = region.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), colors[i].tolist(), 2)
            
            # Draw centroid
            cx, cy = int(region.centroid[0]), int(region.centroid[1])
            cv2.circle(vis, (cx, cy), 3, (255, 0, 0), -1)
            
            # Draw label
            if show_labels:
                label = f"R{region.region_id}"
                cv2.putText(
                    vis, label, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
        
        return vis


def batch_extract_regions(
    sketches: torch.Tensor, 
    extractor: RegionExtractor
) -> List[List[SketchRegion]]:
    """
    Extract regions from a batch of sketches.
    
    Args:
        sketches: Batch of sketches (B, 1, H, W) in range [0, 1]
        extractor: RegionExtractor instance
    
    Returns:
        List of region lists, one per sketch
    """
    batch_regions = []
    
    # Convert to numpy
    sketches_np = (sketches.squeeze(1).cpu().numpy() * 255).astype(np.uint8)
    
    for sketch in sketches_np:
        regions = extractor.extract_regions(sketch)
        batch_regions.append(regions)
    
    return batch_regions


if __name__ == "__main__":
    # Example usage
    print("Region Extraction Module for RAGAF-Diffusion")
    print("=" * 60)
    
    # Create a synthetic sketch with multiple regions
    sketch = np.ones((256, 256), dtype=np.uint8) * 255  # White background
    
    # Draw some black regions (strokes)
    cv2.rectangle(sketch, (50, 50), (100, 100), 0, -1)
    cv2.circle(sketch, (180, 80), 30, 0, -1)
    cv2.ellipse(sketch, (150, 180), (40, 20), 0, 0, 360, 0, -1)
    
    # Extract regions
    extractor = RegionExtractor(min_region_area=200, max_num_regions=20)
    regions = extractor.extract_regions(sketch)
    
    print(f"Extracted {len(regions)} regions:")
    for region in regions:
        print(f"  Region {region.region_id}: area={region.area}, "
              f"centroid={region.centroid}, bbox={region.bbox}")
    
    # Visualize
    vis = extractor.visualize_regions(sketch, regions)
    print(f"\nVisualization shape: {vis.shape}")
