# Testing Epoch 2 with Different Sketches and Prompts

## Quick Test Instructions

You now have flexible tools to test the epoch 2 model with any sketch!

### Method 1: Quick Single Test

```bash
python3 quick_test.py <sketch_path> "<prompt>"
```

**Examples:**
```bash
# Test with airplane sketch - different prompts
python3 quick_test.py input_sketch.png "a red fighter jet in the sky"
python3 quick_test.py input_sketch.png "a commercial airliner at sunset"
python3 quick_test.py input_sketch.png "a vintage propeller plane"
python3 quick_test.py input_sketch.png "a military bomber aircraft"

# If you have other sketches
python3 quick_test.py my_car_sketch.png "a sports car on a highway"
python3 quick_test.py my_cat_sketch.png "a fluffy cat sitting on a sofa"
```

### Method 2: Test Multiple Variations

```bash
python3 test_epoch2_custom.py --test-all
```

This will test the airplane sketch with 3 different prompts automatically.

### Method 3: Custom Test with Output Name

```bash
python3 test_epoch2_custom.py --sketch <path> --prompt "<text>" --output <name>
```

**Example:**
```bash
python3 test_epoch2_custom.py \
  --sketch input_sketch.png \
  --prompt "a blue passenger airplane" \
  --output my_blue_plane.png
```

---

## How to Create New Sketches

### Option 1: Download from Sketchy Dataset
The training uses the Sketchy dataset. You can use any sketch from there:
```bash
# Sketches are in: /network_volume/datasets/sketchy/rendered_256x256/
ls /network_volume/datasets/sketchy/rendered_256x256/airplane/
```

### Option 2: Draw Your Own
1. Use any drawing tool (Paint, GIMP, etc.)
2. Draw simple black lines on white background
3. Save as PNG
4. Upload to the workspace
5. Test with: `python3 quick_test.py your_sketch.png "your prompt"`

### Option 3: Use Simple Shapes
```python
# Create a simple test sketch programmatically
from PIL import Image, ImageDraw

img = Image.new('RGB', (256, 256), 'white')
draw = ImageDraw.Draw(img)

# Draw a simple house
draw.rectangle([50, 150, 200, 230], outline='black', width=3)  # House body
draw.polygon([50, 150, 125, 80, 200, 150], outline='black', width=3)  # Roof
draw.rectangle([100, 180, 140, 230], outline='black', width=3)  # Door

img.save('house_sketch.png')

# Test it
# python3 quick_test.py house_sketch.png "a beautiful cottage in the countryside"
```

---

## What to Test

### Different Prompts on Same Sketch:
- ✈️ "a military fighter jet"
- ✈️ "a commercial passenger airplane"
- ✈️ "a vintage biplane from the 1920s"
- ✈️ "a futuristic spacecraft"
- ✈️ "a paper airplane"

### Different Styles:
- "a photorealistic airplane"
- "an airplane in watercolor painting style"
- "a cartoon airplane"
- "an airplane sketch drawing"

### Different Contexts:
- "an airplane flying over mountains"
- "an airplane at sunset"
- "an airplane in stormy weather"
- "an airplane on a runway"

---

## Expected Results at Epoch 2

✅ **Working:** Sketch structure should be followed
✅ **Working:** Basic shape and orientation correct
⚠️ **Limited:** Fine details may not be perfect yet (only 2 epochs)
⚠️ **Limited:** Some prompts may work better than others

**This will improve significantly by epoch 10!**

---

## Testing Progress

As training continues, test each checkpoint:

```bash
# Epoch 2 (current)
python3 quick_test.py input_sketch.png "test prompt" epoch2_result.png

# Epoch 4 (when available - ~2 hours from now)
# Download epoch_4.pt and test
python3 quick_test.py input_sketch.png "test prompt" epoch4_result.png

# Epoch 6, 8, 10...
# Same process
```

Compare outputs to see quality improvement!

---

## Troubleshooting

### Sketch not working?
- Make sure it's PNG format
- Check it's 256x256 or will be resized
- Black lines on white background work best

### Out of memory?
- The model uses ~15 GB VRAM
- Close other GPU processes
- Check with: `nvidia-smi`

### Different results each time?
- Normal! Diffusion models are stochastic
- Each run gives slightly different output
- Use same random seed for reproducibility (can add to script)

---

## Next Steps

1. **Test the airplane with different prompts** to see variation
2. **Try different sketches** if you have them
3. **Wait for epoch 4** and compare quality improvement
4. **Document interesting results** for your paper/report

**The key finding:** Sketch conditioning is working! The architecture fix was successful! 🎉
