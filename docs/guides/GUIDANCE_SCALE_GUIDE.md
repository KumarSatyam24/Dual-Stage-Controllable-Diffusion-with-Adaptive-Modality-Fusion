# Guidance Scale - No Retraining Needed!

## ❓ Do I Need to Retrain to Change Guidance Scale?

### **NO! ✅**

Guidance scale is an **inference-time parameter** - you can change it anytime without retraining!

---

## 🎯 What is Guidance Scale?

Guidance scale controls how strongly the model follows your text prompt vs the sketch:

```python
noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

### **Low Guidance Scale (1.0 - 3.0):**
- ✅ More creative/diverse
- ✅ Better sketch fidelity (follows structure more)
- ⚠️ Less attention to text prompt
- ⚠️ May ignore some prompt details

### **Medium Guidance Scale (5.0 - 7.5):** ⭐ **Recommended**
- ✅ Balanced between sketch and text
- ✅ Good for most cases
- ✅ Follows sketch structure reasonably well
- ✅ Respects text prompt

### **High Guidance Scale (10.0 - 15.0):**
- ✅ Very strong text adherence
- ⚠️ May distort sketch structure
- ⚠️ Can add extra objects/details
- ⚠️ May oversaturate colors

---

## 🔧 How to Change Guidance Scale

### **Option 1: When Creating Pipeline**

```python
pipeline = Stage1DiffusionPipeline(
    model,
    num_inference_steps=30,
    guidance_scale=5.0,  # ← Change this value!
    device=device
)
```

### **Option 2: Test Different Values**

```python
# Test different guidance scales
for scale in [1.0, 3.0, 5.0, 7.5, 10.0]:
    pipeline = Stage1DiffusionPipeline(
        model,
        guidance_scale=scale,
        device=device
    )
    
    output = pipeline.generate(
        text_prompt="an airplane",
        sketch=sketch_tensor,
        height=256,
        width=256,
    )
    
    output.save(f"output_scale_{scale}.png")
```

---

## 🎨 Recommended Values for Your Issues

Based on your observations:

### **Problem 1: Sketch Structure Changed**
**Solution:** Lower guidance scale
```python
guidance_scale=3.0  # or even 2.0
```

### **Problem 2: Multiple Objects from Single Sketch**
**Solution:** Lower guidance scale + simpler prompts
```python
guidance_scale=3.0
prompt = "an airplane"  # Not "airplanes flying in formation"
```

### **Problem 3: Extra Details (boy wearing eyewear instead of just eyewear)**
**Solution:** Much lower guidance scale + minimal prompts
```python
guidance_scale=2.0  # or even 1.5
prompt = "eyewear"  # Not "a person wearing glasses"
```

---

## 🧪 Quick Test Script

Let me create a script to test different guidance scales:

```python
# Test various guidance scales
guidance_scales = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0]

for scale in guidance_scales:
    pipeline = Stage1DiffusionPipeline(
        model,
        num_inference_steps=30,
        guidance_scale=scale,
        device=device
    )
    
    output = pipeline.generate(
        text_prompt="eyewear",  # Simple prompt
        sketch=eyewear_sketch,
        height=256,
        width=256,
    )
    
    output.save(f"eyewear_scale_{scale}.png")
```

Compare outputs to find the best scale!

---

## 📊 Expected Results

| Guidance Scale | Sketch Fidelity | Text Following | Use Case |
|---------------|-----------------|----------------|----------|
| 1.0 | ⭐⭐⭐⭐⭐ | ⭐ | Strict sketch following |
| 2.0 | ⭐⭐⭐⭐ | ⭐⭐ | Strong sketch, minimal text |
| 3.0 | ⭐⭐⭐ | ⭐⭐⭐ | **Recommended for Stage 1** |
| 5.0 | ⭐⭐ | ⭐⭐⭐⭐ | Balanced |
| 7.5 | ⭐ | ⭐⭐⭐⭐⭐ | Strong text (default) |
| 10.0 | ⭐ | ⭐⭐⭐⭐⭐ | Very strong text |

---

## 🎯 Recommendation for Stage 1

Based on your description that Stage 1 should:
- Focus on sketch structure
- Output single objects matching sketch count
- Not add extra details

### **Use Low Guidance Scale:**

```python
pipeline = Stage1DiffusionPipeline(
    model,
    num_inference_steps=30,
    guidance_scale=2.5,  # ← Lower than default 7.5
    device=device
)
```

### **Use Minimal Prompts:**

```python
# Good prompts for Stage 1
"an airplane"
"eyewear"
"a chair"
"a dog"

# Bad prompts (too detailed)
"a commercial airplane flying in the sky"
"a person wearing stylish sunglasses"
"a modern office chair with wheels"
"a golden retriever sitting in grass"
```

---

## 🔄 No Retraining Needed!

**Key Point:** Guidance scale is applied during inference, not training.

### **Training:** (Already done ✅)
- Learned to follow sketches
- Learned to follow text
- Learned to balance both

### **Inference:** (You control this!)
- Choose how much to follow sketch vs text
- Change guidance_scale anytime
- Test different values instantly

---

## 🚀 Action Items

1. **Test lower guidance scales:**
   - Try 2.0, 2.5, 3.0
   - Compare with current 7.5

2. **Simplify prompts:**
   - Use category name only
   - Remove descriptions/details

3. **Regenerate problematic categories:**
   - Eyewear with scale 2.0
   - Other multi-object issues

4. **Find optimal scale:**
   - Test on 5-10 categories
   - Pick best balance

5. **Regenerate all 125 categories:**
   - Use optimal scale (probably 2.5-3.0)
   - Use minimal prompts
   - Compare with previous results

---

## 💡 Why This Happens

**High guidance scale (7.5):**
- Model tries VERY hard to match text prompt
- "eyewear" → thinks "people wear eyewear"
- Adds context/scene beyond sketch
- Can distort sketch to fit text better

**Low guidance scale (2.5):**
- Model respects sketch structure more
- "eyewear" → just generates eyewear object
- Stays closer to sketch boundaries
- Text provides style hints only

---

## ✅ Summary

**Question:** Do I need to retrain to change guidance scale?

**Answer:** **NO!** Just regenerate with different scale.

**Action:** Test guidance scales 2.0-3.0 with minimal prompts.

**Time:** Minutes to regenerate, not hours to retrain! 🚀
