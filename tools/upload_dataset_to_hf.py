#!/usr/bin/env python3
"""
Upload Sketchy Dataset to Hugging Face Hub
"""

import os
import argparse
import tarfile
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Upload Sketchy dataset to Hugging Face Hub")
    parser.add_argument("--dataset-dir", type=str, 
                        default="/workspace/sketchy",
                        help="Directory containing the dataset")
    parser.add_argument("--repo-name", type=str, 
                        default="sketchy-dataset",
                        help="Repository name on Hugging Face")
    parser.add_argument("--username", type=str, default=None,
                        help="Hugging Face username (optional, will auto-detect)")
    parser.add_argument("--private", action="store_true",
                        help="Make the dataset repository private")
    parser.add_argument("--path-in-repo", type=str, default="dataset",
                        help="Path in the repository to upload to (default: dataset)")
    parser.add_argument("--compress", action="store_true", default=True,
                        help="Compress dataset before uploading (default: True)")
    parser.add_argument("--no-compress", dest="compress", action="store_false",
                        help="Upload without compression")
    args = parser.parse_args()
    
    print("🤗 Hugging Face Dataset Upload")
    print("=" * 60)
    
    # Initialize API
    api = HfApi()
    
    # Get username
    try:
        user_info = api.whoami()
        username = args.username or user_info['name']
        print(f"✅ Logged in as: {username}")
    except Exception as e:
        print(f"❌ Not authenticated! Please run: huggingface-cli login")
        print(f"   Error: {e}")
        return
    
    # Setup
    dataset_dir = Path(args.dataset_dir)
    repo_id = f"{username}/{args.repo_name}"
    
    print(f"📁 Dataset directory: {dataset_dir}")
    print(f"📦 Repository: {repo_id}")
    print(f"🔒 Private: {args.private}")
    print()
    
    # Check if dataset exists
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("\nSearching for dataset in common locations...")
        
        # Search common locations
        search_paths = [
            "/workspace/sketchy",
            "/root/data/sketchy",
            "/root/sketchy",
            "/root/datasets/sketchy",
            Path.home() / "data" / "sketchy",
        ]
        
        for path in search_paths:
            if Path(path).exists():
                print(f"✅ Found dataset at: {path}")
                dataset_dir = Path(path)
                break
        else:
            print("❌ Could not find dataset in any common location")
            return
    
    # Calculate dataset size
    print("📊 Calculating dataset size...")
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            file_path = Path(root) / file
            total_size += file_path.stat().st_size
            file_count += 1
    
    size_gb = total_size / (1024**3)
    print(f"   Files: {file_count}")
    print(f"   Total size: {size_gb:.2f} GB")
    print()
    
    # Compression info
    if args.compress:
        print("💾 Compression: ENABLED")
        print("   Format: tar.gz")
        print(f"   Estimated compressed size: ~{size_gb * 0.3:.2f} GB (assuming 70% compression)")
    else:
        print("💾 Compression: DISABLED")
    print()
    
    # Confirm upload
    print("⚠️  WARNING: This will upload the dataset to Hugging Face.")
    if args.compress:
        print(f"   Original size: {size_gb:.2f} GB")
        print(f"   Estimated compressed: ~{size_gb * 0.3:.2f} GB")
        print(f"   Files: Will be in a single tar.gz archive")
    else:
        print(f"   Size: {size_gb:.2f} GB")
        print(f"   Files: {file_count}")
    print()
    
    response = input("Continue? (yes/NO): ")
    if response.lower() != "yes":
        print("❌ Cancelled")
        return
    
    print()
    
    # Create repository
    try:
        print("📁 Creating repository...")
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            exist_ok=True,
            private=args.private
        )
        print("✅ Repository ready!")
    except Exception as e:
        print(f"⚠️  Repository creation issue: {e}")
        print("   Continuing with upload...")
    
    print()
    
    # Create README for dataset
    readme_content = f"""---
license: mit
task_categories:
- image-to-image
tags:
- sketch
- image-generation
- computer-vision
- sketch-to-image
size_categories:
- 10K<n<100K
---

# Sketchy Dataset

This is the Sketchy database used for sketch-to-image generation research.

## Dataset Details

- **Uploaded:** {datetime.now().strftime("%Y-%m-%d")}
- **Files:** {file_count}
- **Total Size:** {size_gb:.2f} GB
- **Format:** Images (sketches and photos)

## Dataset Structure

```
sketchy/
├── train/
│   ├── sketches/
│   └── photos/
├── val/
│   ├── sketches/
│   └── photos/
└── test/
    ├── sketches/
    └── photos/
```

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}")

# Access examples
sketch = dataset['train']['sketch'][0]
photo = dataset['train']['photo'][0]
```

## Citation

If you use this dataset, please cite the original Sketchy database paper:

```bibtex
@inproceedings{{sangkloy2016sketchy,
  title={{The sketchy database: learning to retrieve badly drawn bunnies}},
  author={{Sangkloy, Patsorn and Burnell, Nathan and Ham, Cusuh and Hays, James}},
  booktitle={{ACM Transactions on Graphics (TOG)}},
  year={{2016}}
}}
```

## License

Please refer to the original Sketchy database license terms.
"""
    
    # Save README
    readme_path = "/tmp/dataset_README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    print("📝 Uploading README...")
    try:
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("✅ README uploaded!")
    except Exception as e:
        print(f"⚠️  README upload failed: {e}")
    
    print()
    
    # Compress if requested
    archive_path = None
    if args.compress:
        print("�️  Compressing dataset...")
        archive_name = f"sketchy_dataset_{datetime.now().strftime('%Y%m%d')}.tar.gz"
        archive_path = Path("/tmp") / archive_name
        
        try:
            with tarfile.open(archive_path, "w:gz") as tar:
                # Add progress tracking
                print(f"   Creating archive: {archive_path}")
                tar.add(dataset_dir, arcname=dataset_dir.name)
            
            # Get compressed size
            compressed_size_gb = archive_path.stat().st_size / (1024**3)
            compression_ratio = (1 - compressed_size_gb / size_gb) * 100
            
            print(f"   ✅ Compression complete!")
            print(f"   Original size: {size_gb:.2f} GB")
            print(f"   Compressed size: {compressed_size_gb:.2f} GB")
            print(f"   Compression ratio: {compression_ratio:.1f}%")
            print()
            
        except Exception as e:
            print(f"   ❌ Compression failed: {e}")
            print("   Falling back to uncompressed upload...")
            args.compress = False
            archive_path = None
    
    # Upload dataset
    print("📤 Uploading dataset...")
    print(f"   Repository: {repo_id}")
    
    if args.compress and archive_path:
        print(f"   File: {archive_path.name}")
        print(f"   Size: {compressed_size_gb:.2f} GB")
        print("   This may take a while...")
        print()
        
        try:
            api.upload_file(
                path_or_fileobj=str(archive_path),
                path_in_repo=f"{args.path_in_repo}/{archive_path.name}",
                repo_id=repo_id,
                repo_type="dataset",
            )
            
            print()
            print("=" * 60)
            print("🎉 Dataset Upload Complete!")
            print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")
            print()
            print("📦 To extract the dataset:")
            print(f"   tar -xzf {archive_path.name}")
            print("=" * 60)
            
            # Clean up temporary archive
            print()
            print("🧹 Cleaning up temporary files...")
            archive_path.unlink()
            print("   ✅ Cleanup complete!")
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            print("\nTip: For very large files:")
            print("  1. Check your internet connection")
            print("  2. Try uploading without compression (--no-compress)")
            print("  3. Use git-lfs: huggingface-cli lfs-enable-largefiles")
            
            # Clean up on failure
            if archive_path and archive_path.exists():
                archive_path.unlink()
    
    else:
        print(f"   Uploading to: {args.path_in_repo}/ in repository")
        print("   This may take a while for large datasets...")
        print()
        
        try:
            # Upload folder directly (without compression)
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(dataset_dir)
                    
                    print(f"   Uploading: {rel_path}")
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=f"{args.path_in_repo}/{rel_path}",
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
            
            print()
            print("=" * 60)
            print("🎉 Dataset Upload Complete!")
            print(f"🔗 View at: https://huggingface.co/datasets/{repo_id}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")

if __name__ == "__main__":
    main()
