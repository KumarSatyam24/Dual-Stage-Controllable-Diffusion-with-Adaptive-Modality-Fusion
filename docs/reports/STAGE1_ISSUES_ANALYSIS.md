# Stage 1 Issues - Analysis and Solutions

## 🔍 Issues Identified

Based on your analysis of the output images:

### 1. **Sketch Structure Not Preserved** ❌
- Output images change the sketch structure
- Example: Single object sketch → Multiple objects generated

### 2. **Prompt Too Detailed** ❌
- Current prompt: "a photorealistic {category}"
- This adds context beyond what Stage 1 should do
- Stage 1 should ONLY focus on sketch structure

### 3. **Context Leakage** ❌
- Eyewear sketch → Boy wearing eyewear (adding human context)
- Single object sketch → Multiple objects in scene
- Model is hallucinating context instead of following sketch exactly

---

## 💡 Root Cause Analysis

### The Problem:
**The prompts are too descriptive and the model is using its prior knowledge to "complete" the scene rather than strictly following the sketch.**

### Why This Happens:
1. **"Photorealistic" keyword** triggers the model to add realistic context
2. **Stable Diffusion's prior** knows that eyewear is worn by people
3. **Classifier-free guidance** (scale 7.5) may be too high, emphasizing text over sketch
4. **Training data** (COCO/Sketchy) contains full scenes, not isolated objects

---

## 🛠️ Solutions to Implement

### Solution 1: **Simpler, Structure-Focused Prompts** ⭐

Instead of "a photorealistic {category}", use:
```python
# Current (wrong):
prompt = f"a photorealistic {category}"

# Better (structure-focused):
prompt = f"a {category}, simple, isolated, white background"

# Or even simpler:
prompt = f"{category}"  # Minimal prompt
```

### Solution 2: **Reduce Guidance Scale**

Lower guidance scale = more sketch influence, less text influence

```python
# Current:
guidance_scale = 7.5

# Try:
guidance_scale = 3.0  # or 5.0
```

### Solution 3: **Stronger Sketch Conditioning**

Increase the weight of sketch features in the model:

```python
# In models/stage1_diffusion.py, SketchEncoder zero_convs
# Current: zero-initialized (gradual learning)
# Consider: Initialize with small positive values for stronger initial signal
```

### Solution 4: **Negative Prompts**

Add negative prompts to prevent context leakage:

```python
prompt = f"a {category}"
negative_prompt = "multiple objects, people, person, human, background, scene, realistic photo"
```

---

## 🔬 Testing Strategy

Let's create a new test with corrected prompts and settings:

### Test Configuration:
1. **Minimal prompts** - Just category name
2. **Lower guidance scale** - 3.0 instead of 7.5  
3. **Negative prompts** - Prevent context leakage
4. **Compare results** - Before vs After

---

## 📝 Implementation Plan

I'll create a new test script that:

1. ✅ Uses minimal prompts (just category name)
2. ✅ Tests different guidance scales (3.0, 5.0, 7.5)
3. ✅ Adds negative prompts to prevent context
4. ✅ Creates side-by-side comparisons
5. ✅ Focuses on problematic categories (eyewear, etc.)

---

## 🎯 Expected Improvements

After implementing these changes:

### Before (Current Issues):
- ❌ Eyewear → Boy wearing eyewear
- ❌ Single object → Multiple objects
- ❌ Sketch structure changed

### After (Expected):
- ✅ Eyewear → Just eyewear, isolated
- ✅ Single object → Single object only
- ✅ Sketch structure preserved exactly

---

## ⚠️ Important Notes

### Stage 1 vs Stage 2 Roles:

**Stage 1 (Current):**
- Input: Sketch + Simple label
- Output: Image matching sketch structure EXACTLY
- Focus: Preserve shape, pose, count of objects
- Should NOT: Add context, complete scenes, hallucinate

**Stage 2 (Future):**
- Input: Stage 1 output + Detailed prompt + Region info
- Output: Refined, detailed image with context
- Focus: Add details, textures, realistic context
- CAN: Add appropriate backgrounds, improve realism

### The Problem:
**Stage 1 is trying to do Stage 2's job!**

The model is too eager to create "photorealistic" results and is adding context that should only come in Stage 2.

---

## 🚀 Next Steps

Would you like me to:

1. **Create a new test script** with corrected prompts and settings?
2. **Re-test problematic categories** (eyewear, etc.) with better prompts?
3. **Fine-tune the model** with better training prompts?
4. **Adjust inference parameters** (guidance scale, negative prompts)?

Let me know which approach you'd like to try first!

---

## 📊 Categories to Re-Test First

These are likely problematic based on your findings:

1. **Eyewear** - Generates person with glasses
2. **Hat** - Probably generates person with hat
3. **Shoe** - Might generate person with shoes
4. **Ring** - Might generate hand with ring
5. **Watch** - Might generate wrist with watch

These wearable items are especially prone to context leakage.

Let me create a focused test for these categories with corrected settings!
