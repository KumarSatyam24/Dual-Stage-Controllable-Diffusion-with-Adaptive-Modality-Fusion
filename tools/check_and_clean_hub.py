"""Check and optionally clean HuggingFace Hub repository."""
from huggingface_hub import HfApi, list_repo_files
import os

api = HfApi(token=os.environ.get('HF_TOKEN'))
repo_id = "DrRORAL/ragaf-diffusion-checkpoints"

print("Checking HuggingFace Hub repository...")
print(f"Repo: {repo_id}\n")

try:
    files = list_repo_files(repo_id=repo_id, token=os.environ.get('HF_TOKEN'))
    checkpoint_files = [f for f in files if f.endswith('.pt')]
    
    print(f"Found {len(checkpoint_files)} checkpoint files:")
    for f in checkpoint_files:
        print(f"  - {f}")
    
    if checkpoint_files:
        print("\n" + "="*60)
        print("⚠️  WARNING: These are from the BROKEN model!")
        print("="*60)
        print("\nThe old checkpoints have NO WORKING sketch conditioning.")
        print("They trained for 10 epochs ignoring the sketch input entirely.")
        print("\nRecommendation: DELETE them to avoid confusion with new model.\n")
        
        delete = input("Delete all old checkpoints? (yes/no): ").strip().lower()
        
        if delete == 'yes':
            print("\nDeleting old checkpoints...")
            for f in checkpoint_files:
                print(f"  Deleting {f}...")
                api.delete_file(
                    path_in_repo=f,
                    repo_id=repo_id,
                    token=os.environ.get('HF_TOKEN')
                )
            print("\n✅ All old checkpoints deleted!")
            print("New checkpoints from fixed model will upload automatically during training.")
        else:
            print("\nSkipping deletion. You can delete manually later with:")
            print(f"  huggingface-cli delete {repo_id} <filename>")
    else:
        print("\n✅ No old checkpoints found - repository is clean!")
        
except Exception as e:
    print(f"Error: {e}")
    if "404" in str(e) or "not found" in str(e).lower():
        print("\n✅ Repository doesn't exist yet - will be created on first upload")
        print("This is normal for a fresh start!")
