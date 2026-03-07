# Stage 1 Optimization - Guidance Scale 2.5

## 🎯 Problem Identified

After analyzing the output images from the first test (guidance_scale=7.5), you noticed:

### ❌ Issues with Original Settings (guidance_scale=7.5):

1. **Sketch structure changed**
   - Output didn't preserve the exact sketch shape
   - Pose/orientation modified

2. **Extra objects added**
   - Single object sketch → multiple objects in output
   - Example: eyewear sketch → boy wearing eyewear (added person)

3. **Too detailed for Stage 1**
   - Stage 1 should focus on sketch structure ONLY
   - Complex prompts caused unwanted elaboration

---

## ✅ Solution: Optimized Settings

### New Configuration:

```python
guidance_scale = 2.5  # Changed from 7.5
prompt = f"a {category}"  # Minimal, just category name
```

### Why This Works:

#### **Lower Guidance Scale (2.5 vs 7.5)**

**Guidance Scale** controls how much the model follows the text prompt vs the unconditional prediction:

```
guidance_scale = 1.0   → Ignore text completely (only sketch)
guidance_scale = 2.5   → Light text guidance (✅ OPTIMAL for sketch fidelity)
guidance_scale = 7.5   → Strong text guidance (❌ can distort sketch)
guidance_scale = 15.0  → Very strong text (❌ destroys sketch structure)
```

**At 2.5:**
- ✅ Sketch structure preserved better
- ✅ Less likely to add extra objects
- ✅ Cleaner, simpler outputs
- ✅ Better balance: sketch guidance > text guidance

**At 7.5 (old):**
- ❌ Text prompt too influential
- ❌ Model adds details not in sketch
- ❌ Sketch structure can be distorted
- ❌ Extra objects appear

#### **Minimal Prompts**

**Old prompts:** `"a photorealistic {category}"`
- Word "photorealistic" caused model to elaborate
- Added unnecessary details

**New prompts:** `"a {category}"`
- Just the category name
- No style modifiers
- Lets sketch be the primary guide

---

## 📊 Expected Improvements

### Sketch Fidelity:
- **Before:** Structure changed, extra details added
- **After:** Structure preserved exactly as in sketch

### Object Count:
- **Before:** Single sketch → multiple objects
- **After:** Single sketch → single object

### Simplicity:
- **Before:** Complex, over-detailed
- **After:** Clean, focused on sketch structure

---

## 🧪 How to Test

### Run the optimized regeneration:

```bash
python3 regenerate_optimized.py
```

This will:
1. ✅ Use guidance_scale=2.5
2. ✅ Use minimal prompts (just category name)
3. ✅ Generate all 125 categories
4. ✅ Create new comparison grids: `optimized_grid_XX.png`

### Compare results:

**Old (guidance=7.5):**
```
/workspace/comparison_grid_01.png
/workspace/comparison_grid_02.png
...
```

**New (guidance=2.5):**
```
/workspace/optimized_grid_01.png
/workspace/optimized_grid_02.png
...
```

Look for:
- ✅ Better sketch structure preservation
- ✅ Fewer extra objects
- ✅ Cleaner outputs

---

## 🔬 Technical Details

### Classifier-Free Guidance (CFG)

The formula used during inference:

```python
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

Where:
- `noise_pred_uncond` = prediction without text (sketch only)
- `noise_pred_text` = prediction with text + sketch
- `guidance_scale` = how much to amplify text influence

**Lower scale = sketch dominates**
**Higher scale = text dominates**

For Stage 1, we want **sketch to dominate**, so use lower scale!

---

## 📈 Guidance Scale Comparison

| Scale | Sketch Fidelity | Text Influence | Use Case |
|-------|----------------|----------------|----------|
| 1.0 | Perfect | None | Sketch-only generation |
| 2.5 | Excellent ✅ | Light | **Stage 1 (optimal)** |
| 5.0 | Good | Moderate | Balanced |
| 7.5 | Fair | Strong | Text-focused (old setting) |
| 10.0+ | Poor | Very Strong | Text-only (ignores sketch) |

---

## 🎯 Stage 1 vs Stage 2 Goals

### Stage 1 Goal:
- ✅ Follow sketch structure EXACTLY
- ✅ Generate basic shape/form
- ✅ Single object matching sketch
- ❌ Don't add extra details or objects

**Optimal settings:**
- Guidance scale: **2.5**
- Prompt: **Minimal** (category name only)

### Stage 2 Goal (future):
- ✅ Add fine details
- ✅ Improve textures
- ✅ Enhance quality
- ✅ Region-adaptive refinement

**Different settings for Stage 2:**
- May use higher guidance (3.0-5.0)
- Can use detailed prompts
- Focuses on refinement, not structure

---

## 🚀 Next Steps

1. **Run optimized regeneration:**
   ```bash
   python3 regenerate_optimized.py
   ```

2. **Compare old vs new grids:**
   - Check sketch structure preservation
   - Count objects (should match sketch)
   - Assess overall quality

3. **If satisfied with results:**
   - ✅ Use guidance_scale=2.5 for all Stage 1 inference
   - ✅ Proceed to Stage 2 training
   - ✅ Document findings

4. **If still issues:**
   - Try guidance_scale=2.0 (even more sketch-focused)
   - Try guidance_scale=3.0 (slightly more text influence)
   - Adjust prompts further if needed

---

## 💡 Key Takeaway

**For Stage 1 sketch-guided diffusion:**
- Lower guidance scale = better sketch fidelity ✅
- Minimal prompts = less interference ✅
- Let the sketch be the primary guide ✅

**Don't retrain - just adjust inference parameters!** 🎯

This is the beauty of guidance scale: it's a **hyperparameter you can tune at test time** without any retraining! 🎉

---

## 📝 Command Summary

```bash
# Regenerate with optimal settings
python3 regenerate_optimized.py

# Output will be in:
test_outputs_optimized/           # Individual images
test_outputs_optimized/optimized_grid_XX.png  # Comparison grids
/workspace/optimized_grid_XX.png  # Sample grids copied here

# Compare with old results:
ls /workspace/comparison_grid_*.png  # Old (guidance=7.5)
ls /workspace/optimized_grid_*.png   # New (guidance=2.5)
```

---

## ✅ Expected Outcome

With guidance_scale=2.5 and minimal prompts:

- ✅ **Sketch structure preserved exactly**
- ✅ **Single object matching sketch (no extras)**
- ✅ **Clean, focused outputs**
- ✅ **Better alignment with Stage 1 goals**

Perfect for a sketch-to-image generation model where **the sketch is the primary guide**! 🎨✨
