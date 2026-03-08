"""
Script to check the format of your downloaded Sketchy dataset.
This will help identify the actual structure and guide you on any needed reorganization.
"""

import os
from pathlib import Path
import sys

def explore_directory(path, max_depth=4, current_depth=0, prefix=""):
    """Recursively explore directory structure."""
    path = Path(path)
    
    if not path.exists():
        print(f"❌ Path does not exist: {path}")
        return
    
    if current_depth >= max_depth:
        return
    
    try:
        items = sorted(path.iterdir())
        
        # Separate directories and files
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        
        # Show first few directories
        if dirs:
            print(f"{prefix}📁 Directories ({len(dirs)} total):")
            for d in dirs[:5]:  # Show first 5
                print(f"{prefix}  - {d.name}/")
            if len(dirs) > 5:
                print(f"{prefix}  ... and {len(dirs) - 5} more")
        
        # Show file statistics
        if files:
            # Group by extension
            extensions = {}
            for f in files:
                ext = f.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
            
            print(f"{prefix}📄 Files ({len(files)} total):")
            for ext, count in sorted(extensions.items()):
                example_files = [f.name for f in files if f.suffix.lower() == ext][:3]
                print(f"{prefix}  - {ext if ext else 'no extension'}: {count} files")
                if example_files:
                    print(f"{prefix}    Examples: {', '.join(example_files[:2])}")
        
        # Recurse into first directory to show structure
        if dirs and current_depth < 2:
            print(f"{prefix}└─ Exploring '{dirs[0].name}/':")
            explore_directory(dirs[0], max_depth, current_depth + 1, prefix + "   ")
            
    except PermissionError:
        print(f"{prefix}❌ Permission denied")
    except Exception as e:
        print(f"{prefix}❌ Error: {e}")


def check_sketchy_dataset(root_path):
    """Check Sketchy dataset structure."""
    print("="*70)
    print("🔍 SKETCHY DATASET STRUCTURE CHECKER")
    print("="*70)
    
    root = Path(root_path)
    
    if not root.exists():
        print(f"\n❌ Directory not found: {root}")
        print("\nPlease provide the correct path to your Sketchy dataset.")
        return False
    
    print(f"\n📂 Checking: {root}")
    print(f"   Absolute path: {root.absolute()}\n")
    
    # Check top-level structure
    print("="*70)
    print("TOP-LEVEL STRUCTURE:")
    print("="*70)
    explore_directory(root, max_depth=1)
    
    print("\n" + "="*70)
    print("EXPECTED STRUCTURE:")
    print("="*70)
    print("""
sketchy/
├── sketch/
│   └── tx_000000000000/
│       ├── airplane/
│       │   ├── n000001.png
│       │   └── ...
│       └── ... (125 categories)
└── photo/
    └── tx_000000000000/
        ├── airplane/
        │   ├── n000001.jpg
        │   └── ...
        └── ... (125 categories)
    """)
    
    # Check for common variations
    print("\n" + "="*70)
    print("CHECKING COMMON LOCATIONS:")
    print("="*70)
    
    possible_sketch_paths = [
        root / "sketch" / "tx_000000000000",
        root / "sketch",
        root / "sketches",
        root / "Sketch",
        root / "256x256" / "sketch" / "tx_000000000000",
    ]
    
    possible_photo_paths = [
        root / "photo" / "tx_000000000000",
        root / "photo",
        root / "photos",
        root / "Photo",
        root / "256x256" / "photo" / "tx_000000000000",
    ]
    
    sketch_found = None
    photo_found = None
    
    for path in possible_sketch_paths:
        if path.exists():
            print(f"✅ Found sketches at: {path.relative_to(root)}")
            sketch_found = path
            
            # Check if it has category subdirectories
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if subdirs:
                print(f"   Contains {len(subdirs)} subdirectories")
                print(f"   Examples: {', '.join([d.name for d in subdirs[:5]])}")
                
                # Check first category
                first_cat = subdirs[0]
                sketch_files = list(first_cat.glob("*.png"))
                print(f"   '{first_cat.name}/' has {len(sketch_files)} .png files")
            break
    
    if not sketch_found:
        print("❌ Sketch directory not found in common locations")
    
    print()
    
    for path in possible_photo_paths:
        if path.exists():
            print(f"✅ Found photos at: {path.relative_to(root)}")
            photo_found = path
            
            # Check if it has category subdirectories
            subdirs = [d for d in path.iterdir() if d.is_dir()]
            if subdirs:
                print(f"   Contains {len(subdirs)} subdirectories")
                print(f"   Examples: {', '.join([d.name for d in subdirs[:5]])}")
                
                # Check first category
                first_cat = subdirs[0]
                photo_files = list(first_cat.glob("*.jpg"))
                print(f"   '{first_cat.name}/' has {len(photo_files)} .jpg files")
            break
    
    if not photo_found:
        print("❌ Photo directory not found in common locations")
    
    # Detailed analysis if found
    if sketch_found and photo_found:
        print("\n" + "="*70)
        print("✅ DATASET VALIDATION:")
        print("="*70)
        
        sketch_categories = [d.name for d in sketch_found.iterdir() if d.is_dir()]
        photo_categories = [d.name for d in photo_found.iterdir() if d.is_dir()]
        
        print(f"Sketch categories: {len(sketch_categories)}")
        print(f"Photo categories: {len(photo_categories)}")
        
        common_cats = set(sketch_categories) & set(photo_categories)
        print(f"Common categories: {len(common_cats)}")
        
        if len(common_cats) > 0:
            print(f"\n✅ Dataset appears valid!")
            print(f"\nCategories: {', '.join(sorted(common_cats)[:10])}")
            if len(common_cats) > 10:
                print(f"           ... and {len(common_cats) - 10} more")
            
            # Count total pairs
            total_pairs = 0
            for cat in list(common_cats)[:5]:  # Check first 5 categories
                sketch_files = list((sketch_found / cat).glob("*.png"))
                photo_files = list((photo_found / cat).glob("*.jpg"))
                pairs = len(set([f.stem for f in sketch_files]) & set([f.stem for f in photo_files]))
                total_pairs += pairs
            
            print(f"\nEstimated sketch-photo pairs (first 5 categories): {total_pairs}")
            
            # Show what SKETCHY_ROOT should be
            print("\n" + "="*70)
            print("ENVIRONMENT VARIABLE SETUP:")
            print("="*70)
            print(f"\nYour SKETCHY_ROOT should be set to:")
            print(f"  {root.absolute()}")
            print(f"\nRun this command:")
            print(f"  export SKETCHY_ROOT={root.absolute()}")
            print(f"  echo 'export SKETCHY_ROOT={root.absolute()}' >> ~/.zshrc")
            
            return True
        else:
            print("\n❌ No common categories found between sketch and photo directories")
            return False
    
    return False


if __name__ == "__main__":
    # Try to get path from environment variable or command line
    import os
    
    if len(sys.argv) > 1:
        sketchy_path = sys.argv[1]
    else:
        sketchy_path = os.getenv("SKETCHY_ROOT")
        
        if not sketchy_path:
            print("="*70)
            print("Please provide the path to your Sketchy dataset:")
            print("="*70)
            print("\nUsage:")
            print("  python check_sketchy_format.py /path/to/sketchy")
            print("\nOr set the environment variable:")
            print("  export SKETCHY_ROOT=/path/to/sketchy")
            print("  python check_sketchy_format.py")
            print("\n" + "="*70)
            
            # Try common locations
            common_paths = [
                Path.home() / "datasets" / "sketchy",
                Path.home() / "Downloads" / "sketchy",
                Path.home() / "Downloads" / "sketchy_database",
                Path("/tmp/sketchy"),
                Path.cwd() / "sketchy",
            ]
            
            print("\nTrying common locations...")
            for path in common_paths:
                if path.exists():
                    print(f"\n✅ Found potential dataset at: {path}")
                    sketchy_path = str(path)
                    break
            
            if not sketchy_path:
                print("\n❌ No Sketchy dataset found in common locations.")
                print("\nPlease specify the path manually.")
                sys.exit(1)
    
    check_sketchy_dataset(sketchy_path)
