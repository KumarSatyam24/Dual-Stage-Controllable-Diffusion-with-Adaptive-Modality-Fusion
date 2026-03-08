"""
MS COCO Dataset Loader for RAGAF-Diffusion

MS COCO dataset with automatically generated sketches from images.
Used for multi-object and complex scene evaluation.

Dataset structure expected:
coco/
├── train2017/
├── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json

Author: RAGAF-Diffusion Research Team
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO

from data.sketch_extraction import SketchExtractor
from data.region_extraction import RegionExtractor
from data.region_graph import RegionGraphBuilder, RegionGraph


class COCODataset(Dataset):
    """
    MS COCO dataset loader with automatic sketch generation for RAGAF-Diffusion.
    
    Returns:
    - sketch: Auto-generated sketch from image (1, H, W)
    - photo: Original COCO image (3, H, W)
    - text_prompt: COCO caption
    - region_graph: Graph structure of sketch regions
    - image_id: COCO image ID
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",  # "train", "val"
        image_size: int = 512,
        sketch_method: str = "canny",  # "canny", "xdog", "hed"
        sketch_extractor: Optional[SketchExtractor] = None,
        region_extractor: Optional[RegionExtractor] = None,
        graph_builder: Optional[RegionGraphBuilder] = None,
        use_all_captions: bool = False,  # Use all 5 captions or just first one
        augment: bool = True,
        max_samples: Optional[int] = None,  # Limit dataset size for quick experiments
        cache_sketches: bool = False  # Cache generated sketches to disk
    ):
        """
        Initialize COCO dataset.
        
        Args:
            root_dir: Path to COCO dataset root
            split: Dataset split ("train" or "val")
            image_size: Target image size
            sketch_method: Sketch extraction method
            sketch_extractor: SketchExtractor instance
            region_extractor: RegionExtractor instance
            graph_builder: RegionGraphBuilder instance
            use_all_captions: Use all captions (5 per image) or just one
            augment: Whether to apply data augmentation
            max_samples: Limit number of samples (for debugging)
            cache_sketches: Cache generated sketches to disk
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.image_size = image_size
        self.sketch_method = sketch_method
        self.use_all_captions = use_all_captions
        self.augment = augment
        self.cache_sketches = cache_sketches
        
        # Set up paths
        if split == "train":
            self.image_dir = self.root_dir / "train2017"
            ann_file = self.root_dir / "annotations" / "captions_train2017.json"
        elif split == "val":
            self.image_dir = self.root_dir / "val2017"
            ann_file = self.root_dir / "annotations" / "captions_val2017.json"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        if not self.image_dir.exists() or not ann_file.exists():
            raise FileNotFoundError(
                f"COCO dataset not found at {self.root_dir}. "
                f"Expected {self.image_dir} and {ann_file}"
            )
        
        # Load COCO annotations
        print(f"Loading COCO {split} annotations...")
        self.coco = COCO(str(ann_file))
        
        # Initialize extractors
        self.sketch_extractor = sketch_extractor or SketchExtractor(
            method=sketch_method,
            invert=True  # White background, black edges
        )
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
        
        # Prepare data pairs
        self.data_pairs = self._prepare_data_pairs(max_samples)
        
        # Setup sketch cache
        if self.cache_sketches:
            self.sketch_cache_dir = self.root_dir / f"sketch_cache_{sketch_method}_{split}"
            self.sketch_cache_dir.mkdir(exist_ok=True)
        else:
            self.sketch_cache_dir = None
        
        print(f"Loaded {len(self.data_pairs)} image-caption pairs from COCO {split}")
    
    def _prepare_data_pairs(self, max_samples: Optional[int]) -> List[Dict]:
        """
        Prepare image-caption pairs.
        
        Args:
            max_samples: Maximum number of samples
        
        Returns:
            List of data pairs
        """
        data_pairs = []
        
        # Get all image IDs
        img_ids = self.coco.getImgIds()
        
        if max_samples is not None:
            img_ids = img_ids[:max_samples]
        
        for img_id in img_ids:
            # Get image info
            img_info = self.coco.loadImgs(img_id)[0]
            
            # Get captions
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            captions = [ann["caption"] for ann in anns]
            
            if len(captions) == 0:
                continue
            
            # If use_all_captions, create one sample per caption
            # Otherwise, just use the first caption
            if self.use_all_captions:
                for caption in captions:
                    data_pairs.append({
                        "image_id": img_id,
                        "image_filename": img_info["file_name"],
                        "caption": caption
                    })
            else:
                data_pairs.append({
                    "image_id": img_id,
                    "image_filename": img_info["file_name"],
                    "caption": captions[0]
                })
        
        return data_pairs
    
    def _get_cached_sketch_path(self, image_id: int) -> Path:
        """Get path to cached sketch file."""
        return self.sketch_cache_dir / f"{image_id:012d}.png"
    
    def _load_or_generate_sketch(self, image_pil: Image.Image, image_id: int) -> Image.Image:
        """
        Load cached sketch or generate new one.
        
        Args:
            image_pil: PIL Image
            image_id: COCO image ID
        
        Returns:
            Sketch as PIL Image (grayscale)
        """
        if self.cache_sketches:
            cache_path = self._get_cached_sketch_path(image_id)
            
            if cache_path.exists():
                # Load cached sketch
                return Image.open(cache_path).convert("L")
        
        # Generate sketch
        sketch_np = self.sketch_extractor.extract(image_pil)
        sketch_pil = Image.fromarray(sketch_np, mode="L")
        
        # Cache if enabled
        if self.cache_sketches:
            cache_path = self._get_cached_sketch_path(image_id)
            sketch_pil.save(cache_path)
        
        return sketch_pil
    
    def __len__(self) -> int:
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single data sample.
        
        Returns:
            Dictionary containing:
            - sketch: Tensor (1, H, W)
            - photo: Tensor (3, H, W)
            - text_prompt: String (caption)
            - region_graph: RegionGraph object
            - image_id: COCO image ID
        """
        pair = self.data_pairs[idx]
        
        # Load image
        image_path = self.image_dir / pair["image_filename"]
        photo_pil = Image.open(image_path).convert("RGB")
        
        # Generate or load sketch
        sketch_pil = self._load_or_generate_sketch(photo_pil, pair["image_id"])
        
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
        
        # Extract region graph
        sketch_np = (sketch.squeeze(0).numpy() * 255).astype(np.uint8)
        regions = self.region_extractor.extract_regions(sketch_np)
        region_graph = self.graph_builder.build_graph(regions)
        
        return {
            "sketch": sketch,
            "photo": photo,
            "text_prompt": pair["caption"],
            "region_graph": region_graph,
            "image_id": pair["image_id"]
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
    image_ids = [item["image_id"] for item in batch]
    
    return {
        "sketch": sketches,
        "photo": photos,
        "text_prompt": text_prompts,
        "region_graph": region_graphs,
        "image_id": image_ids
    }


if __name__ == "__main__":
    # Example usage
    print("COCO Dataset Loader for RAGAF-Diffusion")
    print("=" * 60)
    
    # NOTE: Update this path to your COCO dataset location
    COCO_ROOT = os.getenv("COCO_ROOT", "/path/to/coco/dataset")
    
    if not os.path.exists(COCO_ROOT):
        print(f"WARNING: COCO dataset not found at {COCO_ROOT}")
        print("Please set COCO_ROOT environment variable or update the path")
        print("\nExpected structure:")
        print("  coco/")
        print("  ├── train2017/")
        print("  ├── val2017/")
        print("  └── annotations/")
    else:
        # Create dataset with limited samples for testing
        dataset = COCODataset(
            root_dir=COCO_ROOT,
            split="train",
            image_size=512,
            sketch_method="canny",
            augment=True,
            max_samples=100,  # Limit to 100 images for quick test
            cache_sketches=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Get a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample:")
            print(f"  Sketch shape: {sample['sketch'].shape}")
            print(f"  Photo shape: {sample['photo'].shape}")
            print(f"  Text prompt: {sample['text_prompt']}")
            print(f"  Image ID: {sample['image_id']}")
            print(f"  Region graph nodes: {sample['region_graph'].num_nodes}")
