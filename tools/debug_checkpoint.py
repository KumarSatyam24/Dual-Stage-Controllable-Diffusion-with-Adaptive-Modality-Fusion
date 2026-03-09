#!/usr/bin/env python3
"""
Debug checkpoint to understand training state
"""
import torch
import sys

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "/root/checkpoints/stage1/epoch_10.pt"

print(f"Loading checkpoint: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location='cpu')

print("\n" + "="*70)
print("CHECKPOINT CONTENTS")
print("="*70)

print("\nAvailable keys:")
for key in ckpt.keys():
    if isinstance(ckpt[key], torch.Tensor):
        print(f"  {key}: {ckpt[key].shape}")
    else:
        print(f"  {key}: {type(ckpt[key]).__name__}")

print("\n" + "="*70)
print("TRAINING METADATA")
print("="*70)

if 'epoch' in ckpt:
    print(f"Epoch: {ckpt['epoch']}")
if 'global_step' in ckpt:
    print(f"Global Step: {ckpt['global_step']}")
if 'train_loss' in ckpt:
    print(f"Training Loss: {ckpt['train_loss']}")
if 'val_loss' in ckpt:
    print(f"Validation Loss: {ckpt['val_loss']}")
if 'learning_rate' in ckpt:
    print(f"Learning Rate: {ckpt['learning_rate']}")

print("\n" + "="*70)
print("OPTIMIZER STATE")
print("="*70)

if 'optimizer_state_dict' in ckpt:
    opt_state = ckpt['optimizer_state_dict']
    if 'param_groups' in opt_state:
        for i, pg in enumerate(opt_state['param_groups']):
            print(f"\nParameter Group {i}:")
            print(f"  Learning Rate: {pg.get('lr', 'N/A')}")
            print(f"  Weight Decay: {pg.get('weight_decay', 'N/A')}")
            print(f"  Betas: {pg.get('betas', 'N/A')}")

print("\n" + "="*70)
print("MODEL STATE")
print("="*70)

if 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
    print(f"\nTotal parameters in state dict: {len(state_dict)}")
    
    # Sample some keys
    keys = list(state_dict.keys())
    print(f"\nFirst 10 parameter names:")
    for key in keys[:10]:
        print(f"  {key}: {state_dict[key].shape}")
    
    # Check if sketch encoder exists
    sketch_keys = [k for k in keys if 'sketch' in k.lower()]
    print(f"\nSketch-related parameters: {len(sketch_keys)}")
    for key in sketch_keys[:5]:
        print(f"  {key}: {state_dict[key].shape}")

print("\n" + "="*70)
print("LOSS HISTORY (if available)")
print("="*70)

if 'loss_history' in ckpt:
    losses = ckpt['loss_history']
    print(f"Number of recorded losses: {len(losses)}")
    if len(losses) > 0:
        print(f"First loss: {losses[0]:.6f}")
        print(f"Last loss: {losses[-1]:.6f}")
        print(f"Min loss: {min(losses):.6f}")
        print(f"Max loss: {max(losses):.6f}")

print("\n" + "="*70)
