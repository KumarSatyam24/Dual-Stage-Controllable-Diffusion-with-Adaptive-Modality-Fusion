"""
Sketchy Dataset Loader for RAGAF-Diffusion

The Sketchy dataset contains sketch-photo pairs for various object categories.
Used primarily for structure-preserving single-object experiments.

Dataset structure expected:
sketchy/
├── sketch/
│   ├── tx_000000000000/
│   │   ├── airplane/
│   │   │   ├── n000001.png
│   │   │   └── ...
│   └── ...
├── photo/
│   ├── tx_000000000000/
│   │   ├── airplane/
│   │   │   ├── n000001.jpg
│   │   │   └── ...
│   └── ...

Author: RAGAF-Diffusion Research Team
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data.sketch_extraction import SketchExtractor
from data.region_extraction import RegionExtractor
from data.region_graph import RegionGraphBuilder, RegionGraph


class SketchyDataset(Dataset):
    """
    Sketchy dataset loader for RAGAF-Diffusion training.
    
    Returns:
    - sketch: Sketch image (1, H, W)
    - photo: Corresponding photo (3, H, W)
    - text_prompt: Generated text prompt (e.g., "A photo of a {category}")
    - region_graph: Graph structure of sketch regions
    - category: Object category label
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",  # "train", "val", "test"
        categories: Optional[List[str]] = None,
        image_size: int = 512,
        sketch_extractor: Optional[SketchExtractor] = None,
        region_extractor: Optional[RegionExtractor] = None,
        graph_builder: Optional[RegionGraphBuilder] = None,
        prompt_template: str = "A photo of a {category}",
        augment: bool = True,
        preload_graphs: bool = False
    ):
        """
        Initialize Sketchy dataset.
        
        Args:
            root_dir: Path to Sketchy dataset root (should contain sketch/ and photo/)
            split: Dataset split
            categories: List of categories to include (None = all)
            image_size: Target image size
            sketch_extractor: SketchExtractor instance (not needed for Sketchy, sketches provided)
            region_extractor: RegionExtractor instance
            graph_builder: RegionGraphBuilder instance
            prompt_template: Template for text prompt generation
            augment: Whether to apply data augmentation
            preload_graphs: Whether to precompute all graphs (memory intensive)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.prompt_template = prompt_template
        self.augment = augment
        self.preload_graphs = preload_graphs
        
        # Initialize extractors
        self.sketch_extractor = sketch_extractor  # Not used for Sketchy (sketches provided)
        self.region_extractor = region_extractor or RegionExtractor(
            min_region_area=100,
            max_num_regions=50
        )
        self.graph_builder = graph_builder or RegionGraphBuilder(
            graph_type="hybrid",
            image_size=(image_size, image_size)
        )
        
        # Define transforms
        self.photo_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.sketch_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        if self.augment and split == "train":
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
            ])
        else:
            self.augment_transform = None
        
        # Load dataset pairs
        self.data_pairs = self._load_data_pairs(categories)
        
        print(f"Loaded {len(self.data_pairs)} sketch-photo pairs from Sketchy dataset ({split} split)")
        
        # Optionally preload graphs
        self.preloaded_graphs = {}
        if self.preload_graphs:
            print("Preloading region graphs...")
            self._preload_all_graphs()
    
    def _load_data_pairs(self, categories: Optional[List[str]]) -> List[Dict]:
        """
        Load sketch-photo pairs from Sketchy dataset.
        
        Args:
            categories: List of categories to include
        
        Returns:
            List of data pair dictionaries
        """
        sketch_dir = self.root_dir / "sketch" / "tx_000000000000"
        photo_dir = self.root_dir / "photo" / "tx_000000000000"
        
        if not sketch_dir.exists() or not photo_dir.exists():
            raise FileNotFoundError(
                f"Sketchy dataset not found at {self.root_dir}. "
                f"Expected structure: {self.root_dir}/sketch/ and {self.root_dir}/photo/"
            )
        
        # Get all categories
        all_categories = [d.name for d in sketch_dir.iterdir() if d.is_dir()]
        
        if categories is not None:
            # Filter to specified categories
            all_categories = [c for c in all_categories if c in categories]
        
        data_pairs = []
        
        for category in all_categories:
            cat_sketch_dir = sketch_dir / category
            cat_photo_dir = photo_dir / category
            
            if not cat_sketch_dir.exists() or not cat_photo_dir.exists():
                continue
            
            # Get all sketch files
            sketch_files = sorted(cat_sketch_dir.glob("*.png"))
            
            for sketch_path in sketch_files:
                # Find corresponding photo
                # Sketchy naming: n02691156_10151-1.png (sketch) -> n02691156_10151.jpg (photo)
                # Multiple sketches (-1, -2, -3, etc.) can correspond to one photo
                sketch_stem = sketch_path.stem
                
                # Remove sketch variant suffix (e.g., "-1", "-2", "-3")
                if '-' in sketch_stem:
                    photo_base = sketch_stem.rsplit('-', 1)[0]
                else:
                    photo_base = sketch_stem
                
                photo_name = photo_base + ".jpg"
                photo_path = cat_photo_dir / photo_name
                
                if photo_path.exists():
                    data_pairs.append({
                        "sketch_path": str(sketch_path),
                        "photo_path": str(photo_path),
                        "category": category,
                        "file_id": sketch_path.stem
                    })
        
        # Split dataset (simple split based on file_id hash)
        # For proper splits, you should use official Sketchy splits
        data_pairs = self._apply_split(data_pairs)
        
        return data_pairs
    
    def _apply_split(self, data_pairs: List[Dict]) -> List[Dict]:
        """
        Apply train/val/test split.
        Simple hash-based split for demonstration.
        For real experiments, use official Sketchy splits.
        
        Args:
            data_pairs: All data pairs
        
        Returns:
            Split data pairs
        """
        # Hash-based deterministic split
        train_ratio, val_ratio = 0.7, 0.15
        
        split_pairs = []
        for pair in data_pairs:
            # Use file_id hash for deterministic split
            hash_val = hash(pair["file_id"]) % 100
            
            if self.split == "train" and hash_val < train_ratio * 100:
                split_pairs.append(pair)
            elif self.split == "val" and train_ratio * 100 <= hash_val < (train_ratio + val_ratio) * 100:
                split_pairs.append(pair)
            elif self.split == "test" and hash_val >= (train_ratio + val_ratio) * 100:
                split_pairs.append(pair)
        
        return split_pairs
    
    def _preload_all_graphs(self):
        """Preload all region graphs (memory intensive)."""
        for idx in range(len(self.data_pairs)):
            sketch_path = self.data_pairs[idx]["sketch_path"]
            sketch_pil = Image.open(sketch_path).convert("L")
            sketch_np = np.array(sketch_pil.resize((self.image_size, self.image_size)))
            
            regions = self.region_extractor.extract_regions(sketch_np)
            graph = self.graph_builder.build_graph(regions)
            
            self.preloaded_graphs[idx] = graph
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single data sample.
        
        Returns:
            Dictionary containing:
            - sketch: Tensor (1, H, W)
            - photo: Tensor (3, H, W)
            - text_prompt: String
            - region_graph: RegionGraph object
            - category: String
        """
        pair = self.data_pairs[idx]
        
        # Load sketch and photo
        sketch_pil = Image.open(pair["sketch_path"]).convert("L")
        photo_pil = Image.open(pair["photo_path"]).convert("RGB")
        
        # Apply augmentation (same random seed for sketch and photo)
        if self.augment_transform is not None:
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            photo_pil = self.augment_transform(photo_pil)
            
            torch.manual_seed(seed)
            np.random.seed(seed % (2**32))
            sketch_pil = self.augment_transform(sketch_pil)
        
        # Convert to tensors
        sketch = self.sketch_transform(sketch_pil)
        photo = self.photo_transform(photo_pil)
        
        # Generate text prompt
        category = pair["category"].replace("_", " ")
        text_prompt = self.prompt_template.format(category=category)
        
        # Extract region graph
        if self.preload_graphs:
            region_graph = self.preloaded_graphs[idx]
        else:
            sketch_np = (sketch.squeeze(0).numpy() * 255).astype(np.uint8)
            regions = self.region_extractor.extract_regions(sketch_np)
            region_graph = self.graph_builder.build_graph(regions)
        
        return {
            "sketch": sketch,
            "photo": photo,
            "text_prompt": text_prompt,
            "region_graph": region_graph,
            "category": pair["category"],
            "file_id": pair["file_id"]
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function to handle region graphs.
    
    Args:
        batch: List of samples
    
    Returns:
        Batched dictionary
    """
    sketches = torch.stack([item["sketch"] for item in batch])
    photos = torch.stack([item["photo"] for item in batch])
    text_prompts = [item["text_prompt"] for item in batch]
    region_graphs = [item["region_graph"] for item in batch]
    categories = [item["category"] for item in batch]
    file_ids = [item["file_id"] for item in batch]
    
    return {
        "sketch": sketches,
        "photo": photos,
        "text_prompt": text_prompts,
        "region_graph": region_graphs,
        "category": categories,
        "file_id": file_ids
    }


if __name__ == "__main__":
    # Example usage
    print("Sketchy Dataset Loader for RAGAF-Diffusion")
    print("=" * 60)
    
    # NOTE: Update this path to your Sketchy dataset location
    SKETCHY_ROOT = os.getenv("SKETCHY_ROOT", "/path/to/sketchy/dataset")
    
    if not os.path.exists(SKETCHY_ROOT):
        print(f"WARNING: Sketchy dataset not found at {SKETCHY_ROOT}")
        print("Please set SKETCHY_ROOT environment variable or update the path")
        print("\nExpected structure:")
        print("  sketchy/")
        print("  ├── sketch/tx_000000000000/")
        print("  └── photo/tx_000000000000/")
    else:
        # Create dataset
        dataset = SketchyDataset(
            root_dir=SKETCHY_ROOT,
            split="train",
            categories=None,  # Use all categories
            image_size=512,
            augment=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Get a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample:")
            print(f"  Sketch shape: {sample['sketch'].shape}")
            print(f"  Photo shape: {sample['photo'].shape}")
            print(f"  Text prompt: {sample['text_prompt']}")
            print(f"  Category: {sample['category']}")
            print(f"  Region graph nodes: {sample['region_graph'].num_nodes}")
