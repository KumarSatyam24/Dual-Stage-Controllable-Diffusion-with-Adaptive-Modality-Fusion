"""
Sketch Extraction Utilities for RAGAF-Diffusion

This module provides utilities for extracting sketch representations from images.
Used primarily for MS COCO dataset to generate sketch-like edge maps.

Author: RAGAF-Diffusion Research Team
"""

import cv2
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import torch
from torchvision import transforms


class SketchExtractor:
    """
    Extract sketch-like edge maps from natural images using edge detection algorithms.
    
    Methods:
    - Canny edge detection (default)
    - HED (Holistically-Nested Edge Detection) - optional, requires pretrained model
    - XDoG (eXtended Difference of Gaussians)
    """
    
    def __init__(
        self, 
        method: str = "canny",
        canny_low_threshold: int = 50,
        canny_high_threshold: int = 150,
        gaussian_kernel: int = 5,
        invert: bool = True
    ):
        """
        Initialize sketch extractor.
        
        Args:
            method: Extraction method ('canny', 'xdog', 'hed')
            canny_low_threshold: Lower threshold for Canny edge detection
            canny_high_threshold: Upper threshold for Canny edge detection
            gaussian_kernel: Gaussian blur kernel size for preprocessing
            invert: If True, invert sketch (white background, black edges)
        """
        self.method = method
        self.canny_low = canny_low_threshold
        self.canny_high = canny_high_threshold
        self.gaussian_kernel = gaussian_kernel
        self.invert = invert
        
    def extract(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract sketch from input image.
        
        Args:
            image: PIL Image or numpy array (H, W, 3)
            
        Returns:
            Sketch as numpy array (H, W) in range [0, 255]
        """
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Apply Gaussian blur to reduce noise
        if self.gaussian_kernel > 0:
            gray = cv2.GaussianBlur(gray, (self.gaussian_kernel, self.gaussian_kernel), 0)
        
        # Extract edges based on method
        if self.method == "canny":
            sketch = self._canny_extraction(gray)
        elif self.method == "xdog":
            sketch = self._xdog_extraction(gray)
        elif self.method == "hed":
            sketch = self._hed_extraction(image)
        else:
            raise ValueError(f"Unknown sketch extraction method: {self.method}")
        
        # Invert if needed (white background, black lines)
        if self.invert:
            sketch = 255 - sketch
            
        return sketch
    
    def _canny_extraction(self, gray: np.ndarray) -> np.ndarray:
        """
        Canny edge detection.
        
        Args:
            gray: Grayscale image (H, W)
            
        Returns:
            Edge map (H, W) in range [0, 255]
        """
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        return edges
    
    def _xdog_extraction(
        self, 
        gray: np.ndarray,
        sigma: float = 0.5,
        k: float = 200.0,
        gamma: float = 0.98,
        epsilon: float = 0.1,
        phi: float = 10.0
    ) -> np.ndarray:
        """
        XDoG (eXtended Difference of Gaussians) extraction.
        Produces sketch-like stylized edges.
        
        Args:
            gray: Grayscale image (H, W)
            sigma: Gaussian kernel standard deviation
            k: DoG multiplier
            gamma: Threshold parameter
            epsilon: Soft threshold
            phi: Sharpness parameter
            
        Returns:
            XDoG sketch (H, W) in range [0, 255]
        """
        # Normalize to [0, 1]
        img = gray.astype(np.float32) / 255.0
        
        # Compute two Gaussians with different sigma
        g1 = cv2.GaussianBlur(img, (0, 0), sigma)
        g2 = cv2.GaussianBlur(img, (0, 0), sigma * 1.6)
        
        # Difference of Gaussians
        dog = g1 - gamma * g2
        
        # XDoG threshold
        dog = np.where(dog < epsilon, 1.0, 1.0 + np.tanh(phi * (dog - epsilon)))
        
        # Convert back to [0, 255]
        sketch = (dog * 255).astype(np.uint8)
        
        return sketch
    
    def _hed_extraction(self, image: np.ndarray) -> np.ndarray:
        """
        HED (Holistically-Nested Edge Detection) extraction.
        Note: Requires pretrained HED model. Falls back to Canny if not available.
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            Edge map (H, W) in range [0, 255]
        """
        # TODO: Implement HED model inference
        # For now, fallback to Canny
        print("Warning: HED extraction not yet implemented. Falling back to Canny.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        return self._canny_extraction(gray)
    
    def batch_extract(self, images: Union[list, torch.Tensor]) -> torch.Tensor:
        """
        Extract sketches from a batch of images.
        
        Args:
            images: List of PIL Images or torch.Tensor (B, C, H, W)
            
        Returns:
            Batch of sketches as torch.Tensor (B, 1, H, W) in range [0, 1]
        """
        sketches = []
        
        if isinstance(images, torch.Tensor):
            # Convert tensor to numpy
            images = images.permute(0, 2, 3, 1).cpu().numpy()
            images = (images * 255).astype(np.uint8)
        
        for img in images:
            sketch = self.extract(img)
            sketches.append(sketch)
        
        # Stack and convert to tensor
        sketches = np.stack(sketches, axis=0)  # (B, H, W)
        sketches = torch.from_numpy(sketches).unsqueeze(1).float() / 255.0  # (B, 1, H, W)
        
        return sketches


class SketchAugmentation:
    """
    Augmentation techniques for sketch data to improve robustness.
    """
    
    @staticmethod
    def add_noise(sketch: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        Add random noise to sketch to simulate hand-drawn variations.
        
        Args:
            sketch: Sketch image (H, W) in range [0, 255]
            noise_level: Noise intensity [0, 1]
            
        Returns:
            Noisy sketch (H, W)
        """
        noise = np.random.randn(*sketch.shape) * noise_level * 255
        noisy = sketch.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def random_stroke_width(sketch: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Randomly dilate or erode sketch to vary stroke width.
        
        Args:
            sketch: Sketch image (H, W) in range [0, 255]
            kernel_size: Morphological operation kernel size
            
        Returns:
            Modified sketch (H, W)
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if np.random.rand() > 0.5:
            # Dilate (thicker strokes)
            return cv2.dilate(sketch, kernel, iterations=1)
        else:
            # Erode (thinner strokes)
            return cv2.erode(sketch, kernel, iterations=1)


def save_sketch(sketch: np.ndarray, path: str):
    """
    Save sketch to file.
    
    Args:
        sketch: Sketch array (H, W)
        path: Output file path
    """
    Image.fromarray(sketch).save(path)


if __name__ == "__main__":
    # Example usage
    print("Sketch Extraction Module for RAGAF-Diffusion")
    print("=" * 60)
    
    # Create extractor
    extractor = SketchExtractor(method="canny", invert=True)
    
    # Test with a dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    sketch = extractor.extract(dummy_image)
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Sketch shape: {sketch.shape}")
    print(f"Sketch range: [{sketch.min()}, {sketch.max()}]")
    print("\nExtraction methods available: 'canny', 'xdog', 'hed'")
